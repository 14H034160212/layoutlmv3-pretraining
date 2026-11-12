"""Minimal continued pre-training for ERNIE-Layout (https://arxiv.org/abs/2210.06155) through the unofficial PyTorch
port (see ernie_layout.py).

Two of the paper's four objectives:
  * masked visual-language modelling (MVLM): span masking of text tokens, as MLM in this repository, with an LM head
    tied to the word embeddings;
  * text-image alignment (TIA, from LayoutLMv2): some text segments have their image region covered, and every
    unmasked text token is classified as covered or not.
Reading-order prediction (needs reading-order labels) and replaced-region prediction are not implemented.

The released checkpoint holds only the encoder (and pooler), so both heads start random: use a head warm-up with the
encoder frozen, as for LayoutLMv3.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import get_worker_info

from . import ernie_layout
from .masking import span_mask


@dataclass
class ErniePreTrainingOutput:
    loss: torch.Tensor | None
    mlm_loss: torch.Tensor | None
    tia_loss: torch.Tensor | None
    mlm_logits: torch.Tensor
    tia_logits: torch.Tensor


class ErnieLayoutForPreTraining(nn.Module):
    """ERNIE-Layout encoder + MVLM head (tied to the word embeddings) + TIA head."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.ernie_layout = backbone
        cfg = backbone.config
        self.config = cfg
        self.mlm_transform = nn.Sequential(nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.GELU(),
                                           nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps))
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.tia_head = nn.Linear(cfg.hidden_size, 2)

    @classmethod
    def from_backbone(cls, path: str = ernie_layout.DEFAULT_PATH):
        """Load the encoder through the port; return (model, processor)."""
        networks = ernie_layout._import_port()
        processor, _, _ = ernie_layout.load(2, path)  # tokenizer shim and PaddleNLP image preprocessing
        config = networks.ErnieLayoutConfig.from_pretrained(path)
        backbone = networks.ErnieLayoutModel.from_pretrained(path, config=config)
        return ernie_layout.keep_batchnorm_frozen(cls(backbone)), processor

    def new_parameters(self) -> dict[str, nn.Parameter]:
        """Parameters the released checkpoint does not provide (the heads)."""
        return {n: p for n, p in self.named_parameters() if not n.startswith("ernie_layout.")}

    def set_encoder_trainable(self, trainable: bool) -> None:
        for n, p in self.named_parameters():
            p.requires_grad_(trainable or not n.startswith("ernie_layout."))

    def forward(self, input_ids, bbox, attention_mask, pixel_values, mlm_labels=None, tia_labels=None, **_):
        seq, _ = self.ernie_layout(input_ids=input_ids, bbox=bbox, image=pixel_values, attention_mask=attention_mask)
        text = seq[:, :input_ids.shape[1]]
        word_embeddings = self.ernie_layout.embeddings.word_embeddings.weight
        mlm_logits = F.linear(self.mlm_transform(text), word_embeddings, self.mlm_bias)
        tia_logits = self.tia_head(text)
        losses = {}
        if mlm_labels is not None:
            losses["mlm"] = F.cross_entropy(mlm_logits.reshape(-1, mlm_logits.shape[-1]), mlm_labels.reshape(-1),
                                            ignore_index=-100)
        if tia_labels is not None:
            losses["tia"] = F.cross_entropy(tia_logits.reshape(-1, 2), tia_labels.reshape(-1), ignore_index=-100)
        return ErniePreTrainingOutput(loss=sum(losses.values()) if losses else None, mlm_loss=losses.get("mlm"),
                                      tia_loss=losses.get("tia"), mlm_logits=mlm_logits, tia_logits=tia_logits)


def cover_regions(pixel_values: torch.Tensor, boxes: list[list[int]], fill: float = 0.0) -> torch.Tensor:
    """Fill the image regions of 0-1000 boxes (in a C x H x W image) with `fill`."""
    out = pixel_values.clone()
    _, h, w = out.shape
    for x0, y0, x1, y1 in boxes:
        c0, c1 = int(x0 * w / 1000), max(int(x0 * w / 1000) + 1, int(np.ceil(x1 * w / 1000)))
        r0, r1 = int(y0 * h / 1000), max(int(y0 * h / 1000) + 1, int(np.ceil(y1 * h / 1000)))
        out[:, r0:r1, c0:c1] = fill
    return out


class ErniePretrainingCollator:
    """MVLM span masking (as PretrainingCollator) and TIA: each text segment (the tokens that share one box) is
    covered in the image with probability `tia_ratio`. TIA labels: 1 = the token's segment is covered, 0 = not,
    -100 for special, padding and MLM-masked tokens.

    Random generators are created lazily per DataLoader worker, seeded with (seed, worker id), as in
    PretrainingCollator.
    """

    def __init__(self, tokenizer, mlm_ratio: float = 0.3, poisson_lambda: float = 3.0, tia_ratio: float = 0.15,
                 seed: int | Sequence[int] | None = None):
        self.tok, self.mlm_ratio, self.lam, self.tia_ratio, self.seed = tokenizer, mlm_ratio, poisson_lambda, tia_ratio, seed
        self._owner = None

    def _rngs(self):
        info = get_worker_info()
        owner = info.id if info is not None else None
        if self._owner != ("set", owner):
            self._owner = ("set", owner)
            entropy = [] if self.seed is None else (list(self.seed) if isinstance(self.seed, Sequence) else [self.seed])
            if owner is not None:
                entropy.append(1_000_003 + owner)
            ss = np.random.SeedSequence(entropy or None)
            self.np_rng = np.random.default_rng(ss)
            self.py_rng = random.Random(int(ss.generate_state(1, np.uint64)[0]))

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        self._rngs()
        batch = {k: torch.stack([torch.as_tensor(f[k]) for f in features])
                 for k in ("input_ids", "bbox", "attention_mask", "pixel_values")}
        input_ids = batch["input_ids"].clone()
        pixel_values = batch["pixel_values"].clone().float()
        mlm_labels = torch.full_like(input_ids, -100)
        tia_labels = torch.full_like(input_ids, -100)
        special_ids = set(self.tok.all_special_ids)
        for b in range(input_ids.shape[0]):
            ids = batch["input_ids"][b]
            special = torch.tensor([int(i) in special_ids for i in ids.tolist()]) | (batch["attention_mask"][b] == 0)
            cand = torch.nonzero(~special).flatten().tolist()
            masked = span_mask(cand, self.mlm_ratio, self.lam, self.np_rng)
            text_mask = torch.zeros(len(ids), dtype=torch.bool)
            text_mask[masked] = True
            mlm_labels[b, text_mask] = ids[text_mask]
            input_ids[b, text_mask] = self.tok.mask_token_id
            # TIA: segments = distinct boxes of the real tokens
            segments: dict[tuple, list[int]] = {}
            for j in cand:
                segments.setdefault(tuple(batch["bbox"][b, j].tolist()), []).append(j)
            covered = [box for box in segments if self.py_rng.random() < self.tia_ratio]
            if covered:
                pixel_values[b] = cover_regions(pixel_values[b], [list(box) for box in covered])
            covered_set = set(covered)
            for box, positions in segments.items():
                for j in positions:
                    if not text_mask[j]:
                        tia_labels[b, j] = int(box in covered_set)
        batch.update(input_ids=input_ids, pixel_values=pixel_values, mlm_labels=mlm_labels, tia_labels=tia_labels)
        return batch
