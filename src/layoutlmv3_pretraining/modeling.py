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

    def __init__(self, config: LayoutLMv3Config, image_vocab_size: int = 512):
        super().__init__(config)
        self.config.image_vocab_size = image_vocab_size
        self.layoutlmv3 = LayoutLMv3ModelWithImageMask(config)
        self.mlm_head = LMHead(config)
        self.mim_head = nn.Sequential(nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                                      nn.Linear(config.hidden_size, image_vocab_size))
        self.wpa_head = nn.Linear(config.hidden_size, 2)
        self.post_init()
        self.mlm_head.decoder.weight = self.layoutlmv3.embeddings.word_embeddings.weight  # tie

    @classmethod
    def from_backbone(cls, name_or_path: str, image_vocab_size: int = 512) -> "LayoutLMv3ForPreTraining":
        """Start from a pretrained ``LayoutLMv3Model`` checkpoint (e.g. microsoft/layoutlmv3-base)."""
        backbone = LayoutLMv3Model.from_pretrained(name_or_path)
        model = cls(backbone.config, image_vocab_size)
        missing, unexpected = model.layoutlmv3.load_state_dict(backbone.state_dict(), strict=False)
        assert set(missing) <= {"image_mask_token"} and not unexpected, (missing, unexpected)
        model.mlm_head.decoder.weight = model.layoutlmv3.embeddings.word_embeddings.weight
        return model

    def forward(self, input_ids, bbox, attention_mask, pixel_values, bool_masked_pos=None,
                mlm_labels=None, mim_labels=None, wpa_labels=None, **_):
        out = self.layoutlmv3(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                              pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        seq = out.last_hidden_state
        t = input_ids.shape[1]
        text, patches = seq[:, :t], seq[:, t + 1:]                       # skip the visual [CLS]
        mlm_logits, wpa_logits = self.mlm_head(text), self.wpa_head(text)
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
