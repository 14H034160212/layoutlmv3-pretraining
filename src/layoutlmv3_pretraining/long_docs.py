"""Helpers for documents longer than the 512 text positions of the released checkpoints."""

from __future__ import annotations

import torch
from torch import nn


def _resize_rows(rows: torch.Tensor, n: int, mode: str) -> torch.Tensor:
    """n position rows from the pretrained `rows`: "copy" repeats them (Longformer), "interpolate" resamples them
    linearly so positions stay continuous (position interpolation for learned absolute embeddings)."""
    if mode == "copy":
        reps = (n + rows.shape[0] - 1) // rows.shape[0]
        return rows.repeat(reps, 1)[:n].clone()
    if mode == "interpolate":
        out = torch.nn.functional.interpolate(rows.t()[None].float(), size=n, mode="linear", align_corners=True)
        return out[0].t().to(rows.dtype).contiguous()
    raise ValueError(f"unknown position mode {mode!r}")


def extend_position_embeddings(layoutlmv3: nn.Module, max_length: int, mode: str = "copy") -> nn.Module:
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
    new = nn.Embedding(n, old.shape[1], padding_idx=emb.padding_idx).to(old.device, old.dtype)
    new.weight.data = torch.cat([old[:offset], _resize_rows(real, n - offset, mode)]).clone()
    emb.position_embeddings = new
    emb.register_buffer("position_ids", torch.arange(n, device=old.device).expand((1, -1)), persistent=False)
    layoutlmv3.config.max_position_embeddings = n
    return layoutlmv3


def use_sdpa_attention(model: nn.Module) -> nn.Module:
    """Run LayoutLMv3's self-attention through torch.nn.functional.scaled_dot_product_attention.

    transformers' LayoutLMv3 materialises the attention scores and probabilities (n x n per head) in every layer and
    keeps them for the backward pass. Here the relative-position biases and the padding mask become one additive
    `attn_mask`, which the memory-efficient SDPA kernel (it runs on V100) consumes without storing probabilities. The
    bias is built once per forward pass and shared by all layers. The result is the same: LayoutLMv3's CogView softmax
    is a numerically stable softmax, which SDPA also computes. `output_attentions` and `head_mask` are not supported.
    """
    import math
    import types

    import torch.nn.functional as F
    from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3SelfAttention

    cache: dict = {}

    def shared_bias(self, attention_mask, rel_pos, rel_2d_pos, dtype):
        key = tuple(id(t) for t in (attention_mask, rel_pos, rel_2d_pos)) + (dtype,)
        if cache.get("key") != key:
            scale = math.sqrt(self.attention_head_size)
            bias = None
            if self.has_relative_attention_bias and self.has_spatial_attention_bias:
                bias = (rel_pos + rel_2d_pos) / scale
            elif self.has_relative_attention_bias:
                bias = rel_pos / scale
            if attention_mask is not None:
                bias = attention_mask if bias is None else bias + attention_mask
            cache["key"], cache["bias"] = key, None if bias is None else bias.to(dtype)
            cache["refs"] = (attention_mask, rel_pos, rel_2d_pos)  # keep the inputs alive so their ids stay unique
        return cache["bias"]

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, rel_pos=None,
                rel_2d_pos=None):
        if output_attentions or head_mask is not None:
            raise NotImplementedError("use_sdpa_attention does not support output_attentions or head_mask")
        b, n, _ = hidden_states.shape
        shape = (b, n, self.num_attention_heads, self.attention_head_size)
        q = self.query(hidden_states).view(shape).transpose(1, 2)
        k = self.key(hidden_states).view(shape).transpose(1, 2)
        v = self.value(hidden_states).view(shape).transpose(1, 2)
        bias = shared_bias(self, attention_mask, rel_pos, rel_2d_pos, q.dtype)
        if bias is not None and bias.shape[1] == 1:
            bias = bias.expand(b, self.num_attention_heads, n, n)
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                 dropout_p=self.dropout.p if self.training else 0.0,
                                                 scale=1 / math.sqrt(self.attention_head_size))
        return (context.transpose(1, 2).reshape(b, n, self.all_head_size),)

    for module in model.modules():
        if isinstance(module, LayoutLMv3SelfAttention):
            module.forward = types.MethodType(forward, module)
    return model


def checkpoint_encoder_layers(model: nn.Module) -> nn.Module:
    """Gradient checkpointing for every transformer layer in `<model>.encoder.layer` (recompute activations in the
    backward pass instead of storing them). Used for ERNIE-Layout, whose port removed transformers' checkpointing
    code; LayoutLMv3 can use `model.gradient_checkpointing_enable()` instead."""
    import functools
    import types

    from torch.utils.checkpoint import checkpoint

    for name, module in model.named_modules():
        if name.endswith("encoder.layer") and isinstance(module, nn.ModuleList):
            for layer in module:
                original = layer.forward

                def forward(self, *args, _original=original, **kwargs):
                    if not self.training or not torch.is_grad_enabled():
                        return _original(*args, **kwargs)
                    return checkpoint(functools.partial(_original, **kwargs), *args, use_reentrant=False)

                layer.forward = types.MethodType(forward, layer)
    return model


def use_flex_attention(model: nn.Module, compile: bool = True) -> nn.Module:
    """Run LayoutLMv3's self-attention through FlexAttention, computing the relative-position biases inside the kernel.

    The SDPA path (`use_sdpa_attention`) still materialises an n x n x heads bias. Here the encoder only records each
    token's 1D position and box coordinates, and a `score_mod` looks up the bucketed 1D and 2D biases per score, so no
    n x n tensor is built. Same results as transformers' attention in eval mode. FlexAttention has no attention-probability
    dropout, so training runs without it (the model's other dropouts are unchanged). Needs torch >= 2.5 and Triton;
    whether its kernels compile on a given GPU (e.g. V100) has to be checked there.
    """
    import math
    import types

    from torch.nn.attention.flex_attention import flex_attention
    from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3Encoder, LayoutLMv3SelfAttention

    flex = torch.compile(flex_attention, dynamic=False) if compile else flex_attention
    encoder = next(m for m in model.modules() if isinstance(m, LayoutLMv3Encoder))
    state: dict = {}

    def bucket(rel, num_buckets, max_distance):
        num_buckets //= 2
        ret = (rel > 0).to(torch.int64) * num_buckets
        n = torch.abs(rel)
        max_exact = num_buckets // 2
        large = max_exact + (torch.log(torch.clamp(n, min=1).float() / max_exact)
                             / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.int64)
        large = torch.clamp(large, max=num_buckets - 1)
        return ret + torch.where(n < max_exact, n, large)

    def record_1d(self, position_ids):
        state["pos"] = position_ids
        return position_ids.new_zeros(())      # placeholder; the bias is computed inside the kernel

    def record_2d(self, bbox):
        state["x"], state["y"] = bbox[:, :, 0], bbox[:, :, 3]
        return bbox.new_zeros(())

    encoder._cal_1d_pos_emb = types.MethodType(record_1d, encoder)
    encoder._cal_2d_pos_emb = types.MethodType(record_2d, encoder)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, rel_pos=None,
                rel_2d_pos=None):
        if output_attentions or head_mask is not None:
            raise NotImplementedError("use_flex_attention does not support output_attentions or head_mask")
        b, n, _ = hidden_states.shape
        shape = (b, n, self.num_attention_heads, self.attention_head_size)
        q = self.query(hidden_states).view(shape).transpose(1, 2)
        k = self.key(hidden_states).view(shape).transpose(1, 2)
        v = self.value(hidden_states).view(shape).transpose(1, 2)
        inv = 1 / math.sqrt(self.attention_head_size)
        use_1d, use_2d = self.has_relative_attention_bias, self.has_spatial_attention_bias
        # Bucketing depends only on the relative distance, so turn each table into a lookup by distance (scaled by
        # 1/sqrt(d)): the kernel then does one gather per bias instead of a log and a bucket per score.
        luts = state.get("luts")
        if luts is None or luts[-1] != q.dtype:
            n_pos, n_xy = encoder.config.max_position_embeddings, 1024
            d1 = torch.arange(-n_pos, n_pos + 1, device=q.device)
            d2 = torch.arange(-n_xy, n_xy + 1, device=q.device)
            lut = lambda table, d, bins, max_rel: (table.weight.detach().t()[bucket(d, bins, max_rel)] * inv).to(q.dtype)
            luts = state["luts"] = (
                lut(encoder.rel_pos_bias, d1, encoder.rel_pos_bins, encoder.max_rel_pos) if use_1d else None,
                lut(encoder.rel_pos_x_bias, d2, encoder.rel_2d_pos_bins, encoder.max_rel_2d_pos) if use_2d else None,
                lut(encoder.rel_pos_y_bias, d2, encoder.rel_2d_pos_bins, encoder.max_rel_2d_pos) if use_2d else None,
                n_pos, n_xy, q.dtype)
        l1, lx, ly, n_pos, n_xy, _ = luts
        pos, x, y = state.get("pos"), state.get("x"), state.get("y")
        valid = None if attention_mask is None else attention_mask[:, 0, 0, :] > torch.finfo(attention_mask.dtype).min / 2

        def score_mod(score, bi, hi, qi, ki):
            if use_1d:
                score = score + l1[pos[bi, ki] - pos[bi, qi] + n_pos, hi]
            if use_2d:
                score = score + lx[x[bi, ki] - x[bi, qi] + n_xy, hi] + ly[y[bi, ki] - y[bi, qi] + n_xy, hi]
            if valid is not None:
                score = torch.where(valid[bi, ki], score, -float("inf"))
            return score

        context = flex(q, k, v, score_mod=score_mod, scale=inv)
        return (context.transpose(1, 2).reshape(b, n, self.all_head_size),)

    for module in model.modules():
        if isinstance(module, LayoutLMv3SelfAttention):
            module.forward = types.MethodType(forward, module)
    model._flex_state = state   # drop state["luts"] after changing the bias tables (they are frozen in transformers)
    return model
