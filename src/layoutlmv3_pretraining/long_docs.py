"""Helpers for documents longer than the 512 text positions of the released checkpoints."""

from __future__ import annotations

import torch
from torch import nn


def extend_position_embeddings(layoutlmv3: nn.Module, max_length: int) -> nn.Module:
    """Give a LayoutLMv3Model 1D position embeddings for `max_length` text tokens by tiling the pretrained ones
    (as Longformer does for RoBERTa). Positions start after the padding index (RoBERTa style), so rows 0 ..
    padding_idx are kept and the pretrained rows after them are repeated. The relative position biases are
    bucketed and need no change. The new positions are only trained by fine-tuning on longer inputs."""
    emb = layoutlmv3.embeddings
    old = emb.position_embeddings.weight.data
    offset = emb.padding_idx + 1
    n = max_length + offset
    if n <= old.shape[0]:
        return layoutlmv3
    real = old[offset:]
    reps = (n - offset + real.shape[0] - 1) // real.shape[0]
    new = nn.Embedding(n, old.shape[1], padding_idx=emb.padding_idx).to(old.device, old.dtype)
    new.weight.data = torch.cat([old[:offset], real.repeat(reps, 1)[: n - offset]]).clone()
    emb.position_embeddings = new
    emb.register_buffer("position_ids", torch.arange(n, device=old.device).expand((1, -1)), persistent=False)
    layoutlmv3.config.max_position_embeddings = n
    return layoutlmv3
