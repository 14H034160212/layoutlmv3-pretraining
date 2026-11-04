"""LayoutLMv3 with the three pre-training objectives: MLM, MIM and WPA.

Built on Hugging Face ``transformers``' ``LayoutLMv3Model``. The only change to
the backbone is image masking: masked patch embeddings are replaced by a learned
``[MASK]`` embedding before position embeddings are added (as in BEiT).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput


class LayoutLMv3ModelWithImageMask(LayoutLMv3Model):
    """``LayoutLMv3Model`` that can mask image patches during ``forward``."""

    def __init__(self, config: LayoutLMv3Config):
        super().__init__(config)
        self.image_mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.image_mask_token, std=0.02)
        self._bool_masked_pos: torch.Tensor | None = None

    def forward_image(self, pixel_values):
        embeddings = self.patch_embed(pixel_values)                          # B, N, D
        if self._bool_masked_pos is not None:
            w = self._bool_masked_pos.reshape(embeddings.shape[0], -1, 1).type_as(embeddings)
            embeddings = embeddings * (1 - w) + self.image_mask_token.expand_as(embeddings) * w
        cls_tokens = self.cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed
        return self.norm(self.pos_drop(embeddings))

    def forward(self, *args, bool_masked_pos: torch.Tensor | None = None, **kwargs):
        self._bool_masked_pos = bool_masked_pos
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._bool_masked_pos = None


class LMHead(nn.Module):
    """dense -> activation -> LayerNorm -> decoder tied to the word embeddings (RoBERTa style)."""

    def __init__(self, config: LayoutLMv3Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        return self.decoder(self.layer_norm(self.act(self.dense(x))))


def token_patch_matrix(bbox: torch.Tensor, grid: int = 14, scale: int = 1000) -> torch.Tensor:
    """(B, T, grid * grid) bool: which patches each token's 0-1000 box overlaps (vectorised ``bbox_to_patches``)."""
    x0, y0, x1, y1 = bbox.long().unbind(-1)
    valid = (x1 > x0) & (y1 > y0)
    c0, c1 = x0 * grid // scale, ((x1 * grid - 1) // scale).clamp(max=grid - 1)
    r0, r1 = y0 * grid // scale, ((y1 * grid - 1) // scale).clamp(max=grid - 1)
    idx = torch.arange(grid, device=bbox.device)
    cols = (idx >= c0[..., None]) & (idx <= c1[..., None])                  # B, T, grid
    rows = (idx >= r0[..., None]) & (idx <= r1[..., None])
    m = rows[..., :, None] & cols[..., None, :] & valid[..., None, None]
    return m.flatten(-2)


class WPAHead(nn.Module):
    """Word-patch alignment classifier (aligned / unaligned) for each text token.

    * ``linear``    : a linear layer on the text token's hidden state
    * ``mlp``       : a two-layer MLP on the text token's hidden state
    * ``mlp_patch`` : a two-layer MLP on [text hidden state; mean hidden state of the image patches its box
                      covers]; the paper (Sec. 2.3, objective III) describes a two-layer MLP head "that inputs
                      contextual text and image"
    """

    VARIANTS = ("linear", "mlp", "mlp_patch")

    def __init__(self, config: LayoutLMv3Config, variant: str = "linear"):
        super().__init__()
        assert variant in self.VARIANTS, variant
        self.variant, d = variant, config.hidden_size
        if variant == "linear":
            self.net = nn.Linear(d, 2)
        else:
            d_in = 2 * d if variant == "mlp_patch" else d
            self.net = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, text, patches=None, bbox=None):
        if self.variant == "mlp_patch":
            m = token_patch_matrix(bbox, grid=int(round(patches.shape[1] ** 0.5))).type_as(patches)
            pooled = torch.einsum("btp,bpd->btd", m, patches) / m.sum(-1, keepdim=True).clamp(min=1)
            text = torch.cat([text, pooled], -1)
        return self.net(text)


@dataclass
class LayoutLMv3PreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    mlm_loss: torch.FloatTensor | None = None
    mim_loss: torch.FloatTensor | None = None
    wpa_loss: torch.FloatTensor | None = None
    mlm_logits: torch.FloatTensor | None = None
    mim_logits: torch.FloatTensor | None = None
    wpa_logits: torch.FloatTensor | None = None


class LayoutLMv3ForPreTraining(LayoutLMv3PreTrainedModel):
    """Sum of the three objectives. Every term is a tensor, so all of them back-propagate."""

    supports_gradient_checkpointing = True

    def __init__(self, config: LayoutLMv3Config, image_vocab_size: int = 512, wpa_head: str | None = None):
        super().__init__(config)
        self.config.image_vocab_size = image_vocab_size
        self.config.wpa_head = wpa_head or getattr(config, "wpa_head", "linear")
        self.layoutlmv3 = LayoutLMv3ModelWithImageMask(config)
        self.mlm_head = LMHead(config)
        self.mim_head = nn.Sequential(nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                                      nn.Linear(config.hidden_size, image_vocab_size))
        self.wpa_head = WPAHead(config, self.config.wpa_head)
        self.post_init()
        self.mlm_head.decoder.weight = self.layoutlmv3.embeddings.word_embeddings.weight  # tie

    @classmethod
    def from_backbone(cls, name_or_path: str, image_vocab_size: int = 512,
                      random_init: bool = False, wpa_head: str = "linear") -> "LayoutLMv3ForPreTraining":
        """Start from a pretrained ``LayoutLMv3Model`` checkpoint (e.g. microsoft/layoutlmv3-base).

        The released checkpoints contain only the encoder, so the MLM/MIM/WPA heads (and the image
        ``[MASK]`` embedding) are always freshly initialised; only the MLM decoder weight is meaningful,
        because it is tied to the word embeddings. ``random_init=True`` keeps the checkpoint's config but
        not its weights (a from-scratch control). ``model.loading_info`` records what was loaded.
        """
        if random_init:
            model = cls(LayoutLMv3Config.from_pretrained(name_or_path), image_vocab_size, wpa_head)
            model.loading_info = {"random_init": True}
            return model
        backbone, info = LayoutLMv3Model.from_pretrained(name_or_path, output_loading_info=True)
        assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"], info
        model = cls(backbone.config, image_vocab_size, wpa_head)
        missing, unexpected = model.layoutlmv3.load_state_dict(backbone.state_dict(), strict=False)
        assert set(missing) <= {"image_mask_token"} and not unexpected, (missing, unexpected)
        model.mlm_head.decoder.weight = model.layoutlmv3.embeddings.word_embeddings.weight
        model.loading_info = {"random_init": False, "checkpoint_missing_keys": info["missing_keys"],
                              "checkpoint_unexpected_keys": info["unexpected_keys"],
                              "encoder_keys_loaded": len(backbone.state_dict()),
                              "encoder_keys_not_in_checkpoint": sorted(missing)}
        return model

    # ---------------------------------------------------------------- parameter groups

    def new_parameters(self) -> dict[str, nn.Parameter]:
        """Parameters that no released checkpoint provides: the three heads and the image [MASK] token.

        The MLM decoder weight is tied to the word embeddings, so it counts as an encoder parameter.
        """
        encoder_ids = {id(p) for n, p in self.layoutlmv3.named_parameters() if n != "image_mask_token"}
        return {n: p for n, p in self.named_parameters() if id(p) not in encoder_ids}

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze (or unfreeze) every checkpoint parameter; the new parameters always stay trainable."""
        new = {id(p) for p in self.new_parameters().values()}
        for p in self.parameters():
            p.requires_grad_(trainable or id(p) in new)

    def layer_id(self, name: str) -> int:
        """BEiT-style depth of a parameter: 0 = embeddings, i + 1 = transformer layer i, L + 1 = top.

        The shared relative-position biases, the heads and the image [MASK] token count as the top.
        """
        num_layers = self.config.num_hidden_layers
        if name.startswith("layoutlmv3.encoder.layer."):
            return int(name.split(".")[3]) + 1
        if name.startswith("layoutlmv3.") and not name.startswith(("layoutlmv3.encoder.",
                                                                   "layoutlmv3.image_mask_token")):
            return 0                     # text/visual embeddings, patch embedding, cls/pos tokens, input LNs
        return num_layers + 1

    def layerwise_param_groups(self, lr: float, layer_decay: float = 1.0, weight_decay: float = 0.01,
                               new_lr: float | None = None) -> list[dict]:
        """AdamW groups with lr * layer_decay ** (L + 1 - depth); ``new_lr`` overrides it for new parameters."""
        top = self.config.num_hidden_layers + 1
        new = {id(p) for p in self.new_parameters().values()}
        groups: dict[tuple, dict] = {}
        for name, p in self.named_parameters():
            is_new = id(p) in new
            depth = self.layer_id(name)
            g_lr = new_lr if (is_new and new_lr is not None) else lr * layer_decay ** (top - depth)
            g = groups.setdefault((depth, is_new), {"params": [], "names": [], "lr": g_lr,
                                                     "weight_decay": weight_decay})
            g["params"].append(p); g["names"].append(name)
        return list(groups.values())

    def forward(self, input_ids, bbox, attention_mask, pixel_values, bool_masked_pos=None,
                mlm_labels=None, mim_labels=None, wpa_labels=None, **_):
        out = self.layoutlmv3(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                              pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        seq = out.last_hidden_state
        t = input_ids.shape[1]
        text, patches = seq[:, :t], seq[:, t + 1:]                       # skip the visual [CLS]
        mlm_logits, wpa_logits = self.mlm_head(text), self.wpa_head(text, patches, bbox)
        mim_logits = self.mim_head(patches)

        losses = {}
        if mlm_labels is not None:
            losses["mlm"] = F.cross_entropy(mlm_logits.reshape(-1, mlm_logits.shape[-1]),
                                            mlm_labels.reshape(-1), ignore_index=-100)
        if mim_labels is not None:
            losses["mim"] = F.cross_entropy(mim_logits.reshape(-1, mim_logits.shape[-1]),
                                            mim_labels.reshape(-1), ignore_index=-100)
        if wpa_labels is not None:
            losses["wpa"] = F.cross_entropy(wpa_logits.reshape(-1, 2), wpa_labels.reshape(-1),
                                            ignore_index=-100)
        return LayoutLMv3PreTrainingOutput(
            loss=sum(losses.values()) if losses else None,
            mlm_loss=losses.get("mlm"), mim_loss=losses.get("mim"), wpa_loss=losses.get("wpa"),
            mlm_logits=mlm_logits, mim_logits=mim_logits, wpa_logits=wpa_logits)
