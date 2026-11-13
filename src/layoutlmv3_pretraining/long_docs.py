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


def _relative_bias_lookup(model: nn.Module):
    """Patch LayoutLMv3's encoder to record positions and boxes instead of building the n x n relative-position biases.

    Returns (encoder, state, luts). After a forward has started, state["pos"] / state["x"] / state["y"] hold each
    token's 1D position and box left / bottom coordinates (text and image tokens). `luts(dtype, device, scale)` gives the
    1D, x and y biases as tables indexed by relative distance (offset by n_pos or n_xy), already bucketed and multiplied
    by `scale`: the bucketing in transformers depends only on the distance, so a gather from these tables reproduces it.
    """
    import math
    import types

    from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3Encoder

    encoder = next(m for m in model.modules() if isinstance(m, LayoutLMv3Encoder))
    state: dict = {}

    def bucket(rel, num_buckets, max_distance):   # transformers' relative_position_bucket
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
        return position_ids.new_zeros(())      # placeholder; the attention computes the bias itself

    def record_2d(self, bbox):
        state["x"], state["y"] = bbox[:, :, 0], bbox[:, :, 3]
        return bbox.new_zeros(())

    encoder._cal_1d_pos_emb = types.MethodType(record_1d, encoder)
    encoder._cal_2d_pos_emb = types.MethodType(record_2d, encoder)

    def luts(dtype, device, scale):
        key = (dtype, device, scale, encoder.config.max_position_embeddings)
        if state.get("luts_key") != key:       # the tables are not trained in transformers (see transformers#50636)
            n_pos, n_xy = encoder.config.max_position_embeddings, 1024
            d1 = torch.arange(-n_pos, n_pos + 1, device=device)
            d2 = torch.arange(-n_xy, n_xy + 1, device=device)

            def lut(table, d, bins, max_rel):
                return (table.weight.detach().t()[bucket(d, bins, max_rel)] * scale).to(dtype)

            l1 = lut(encoder.rel_pos_bias, d1, encoder.rel_pos_bins, encoder.max_rel_pos) if encoder.has_relative_attention_bias else None
            sp = encoder.has_spatial_attention_bias
            lx = lut(encoder.rel_pos_x_bias, d2, encoder.rel_2d_pos_bins, encoder.max_rel_2d_pos) if sp else None
            ly = lut(encoder.rel_pos_y_bias, d2, encoder.rel_2d_pos_bins, encoder.max_rel_2d_pos) if sp else None
            state["luts"], state["luts_key"] = (l1, lx, ly, n_pos, n_xy), key
        return state["luts"]

    return encoder, state, luts


def _key_valid(attention_mask):
    """(batch, keys) bool from transformers' additive extended mask, or None."""
    if attention_mask is None:
        return None
    return attention_mask[:, 0, 0, :] > torch.finfo(attention_mask.dtype).min / 2


def use_flex_attention(model: nn.Module, compile: bool = True) -> nn.Module:
    """Run LayoutLMv3's self-attention through FlexAttention, computing the relative-position biases inside the kernel.

    The SDPA path (`use_sdpa_attention`) still materialises an n x n x heads bias. Here a `score_mod` gathers the 1D and
    2D biases per score from per-distance tables, so no n x n tensor is built. Same results as transformers' attention
    in eval mode. FlexAttention has no attention-probability dropout, so training runs without it (the model's other
    dropouts are unchanged). Needs torch >= 2.5 and Triton. Triton does not use the tensor cores of pre-Ampere GPUs
    (V100, T4), so there FlexAttention may be much slower than SDPA; prefer `use_chunked_sdpa_attention` on them.
    """
    import math
    import types

    from torch.nn.attention.flex_attention import flex_attention
    from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3SelfAttention

    flex = torch.compile(flex_attention, dynamic=False) if compile else flex_attention
    encoder, state, luts = _relative_bias_lookup(model)

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
        l1, lx, ly, n_pos, n_xy = luts(q.dtype, q.device, inv)
        pos, x, y = state.get("pos"), state.get("x"), state.get("y")
        valid = _key_valid(attention_mask)

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
    return model


def use_chunked_sdpa_attention(model: nn.Module, chunk: int = 512) -> nn.Module:
    """Run LayoutLMv3's self-attention as SDPA over blocks of `chunk` queries, building each block's bias on the fly.

    `use_sdpa_attention` builds one n x n x heads bias, so its memory grows with the square of the length. Here only a
    chunk x n x heads bias exists at a time (gathered from per-distance tables), so attention memory grows linearly
    (Rabe & Staats, 2021, "Self-attention Does Not Need O(n^2) Memory"). It uses PyTorch's memory-efficient SDPA kernel,
    which runs on the tensor cores of V100s, with no Triton or compilation. Exact: same results as transformers'
    attention. In training each block is checkpointed, so its bias is rebuilt in backward instead of being kept.
    """
    import math
    import types

    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3SelfAttention

    encoder, state, luts = _relative_bias_lookup(model)

    def block(self, q, k, v, pos, x, y, valid, start, inv):
        l1, lx, ly, n_pos, n_xy = luts(q.dtype, q.device, inv)
        end = start + q.shape[2]
        # gather from (D, heads) tables transposed to (heads, D): the result is (heads, b, chunk, n), whose
        # (b, heads, chunk, n) view has a contiguous last dimension, as SDPA's bias needs, without a copy
        bias = None
        if self.has_relative_attention_bias:
            bias = state["t1"][:, pos[:, None, :] - pos[:, start:end, None] + n_pos]
        if self.has_spatial_attention_bias:
            b2 = state["tx"][:, x[:, None, :] - x[:, start:end, None] + n_xy]
            b2 += state["ty"][:, y[:, None, :] - y[:, start:end, None] + n_xy]
            bias = b2 if bias is None else bias.add_(b2)
        if bias is None:
            bias = q.new_zeros(1, q.shape[0], q.shape[2], k.shape[2])
        bias = bias.transpose(0, 1)
        if valid is not None:
            bias = bias.masked_fill(~valid[:, None, None, :], torch.finfo(bias.dtype).min)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                              dropout_p=self.dropout.p if self.training else 0.0, scale=inv)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, rel_pos=None,
                rel_2d_pos=None):
        if output_attentions or head_mask is not None:
            raise NotImplementedError("use_chunked_sdpa_attention does not support output_attentions or head_mask")
        b, n, _ = hidden_states.shape
        shape = (b, n, self.num_attention_heads, self.attention_head_size)
        q = self.query(hidden_states).view(shape).transpose(1, 2)
        k = self.key(hidden_states).view(shape).transpose(1, 2)
        v = self.value(hidden_states).view(shape).transpose(1, 2)
        inv = 1 / math.sqrt(self.attention_head_size)
        pos, x, y = state.get("pos"), state.get("x"), state.get("y")
        valid = _key_valid(attention_mask)
        l1, lx, ly, _, _ = luts(q.dtype, q.device, inv)
        if state.get("t_src") is not l1:
            state["t1"], state["tx"], state["ty"] = (None if t is None else t.t().contiguous() for t in (l1, lx, ly))
            state["t_src"] = l1
        outs = []
        for start in range(0, n, chunk):
            args = (self, q[:, :, start:start + chunk], k, v, pos, x, y, valid, start, inv)
            if torch.is_grad_enabled() and self.training:
                outs.append(checkpoint(block, *args, use_reentrant=False))
            else:
                outs.append(block(*args))
        context = torch.cat(outs, dim=2)
        return (context.transpose(1, 2).reshape(b, n, self.all_head_size),)

    for module in model.modules():
        if isinstance(module, LayoutLMv3SelfAttention):
            module.forward = types.MethodType(forward, module)
    return model


def use_lut_sdpa_attention(model: nn.Module) -> nn.Module:
    """`use_sdpa_attention` with a lean bias: the shared n x n x heads bias is gathered once per forward pass directly
    in the compute dtype from per-distance tables, instead of transformers' bucketing with fp32 n x n temporaries.

    Same speed as `use_sdpa_attention` (one bias for all layers, memory-efficient SDPA kernel that runs on V100 tensor
    cores) with a fraction of its memory: the bias itself is n^2 x heads x 2 bytes in fp16 (121 MB at 2048 tokens plus
    197 image patches), still quadratic but small; `use_chunked_sdpa_attention` removes even that at a speed cost.
    Exact: same results as transformers' attention.
    """
    import math
    import types

    import torch.nn.functional as F
    from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3SelfAttention

    encoder, state, luts = _relative_bias_lookup(model)

    def shared_bias(self, q, attention_mask, inv):
        pos = state.get("pos")
        key = (id(pos), id(attention_mask), q.dtype)
        if state.get("bias_key") == key:
            return state["bias"]
        l1, lx, ly, n_pos, n_xy = luts(q.dtype, q.device, inv)
        if state.get("t_src") is not l1:
            state["t1"], state["tx"], state["ty"] = (None if t is None else t.t().contiguous() for t in (l1, lx, ly))
            state["t_src"] = l1
        bias = None   # (heads, b, n, n); int32 indices halve the index temporaries
        if self.has_relative_attention_bias:
            p = pos.int()
            bias = state["t1"][:, p[:, None, :] - p[:, :, None] + n_pos]
        if self.has_spatial_attention_bias:
            for coord, table in ((state["x"], state["tx"]), (state["y"], state["ty"])):
                c = coord.int()
                g = table[:, c[:, None, :] - c[:, :, None] + n_xy]
                bias = g if bias is None else bias.add_(g)
                del g
        valid = _key_valid(attention_mask)
        if bias is None:
            bias = None if valid is None else q.new_zeros(1, *valid.shape[:1], q.shape[2], q.shape[2])
        if bias is not None and valid is not None:
            bias.masked_fill_(~valid[None, :, None, :], torch.finfo(bias.dtype).min)
        bias = None if bias is None else bias.transpose(0, 1)   # (b, heads, n, n) view, last dim contiguous
        state["bias"], state["bias_key"], state["bias_refs"] = bias, key, (pos, attention_mask)
        return bias

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, rel_pos=None,
                rel_2d_pos=None):
        if output_attentions or head_mask is not None:
            raise NotImplementedError("use_lut_sdpa_attention does not support output_attentions or head_mask")
        b, n, _ = hidden_states.shape
        shape = (b, n, self.num_attention_heads, self.attention_head_size)
        q = self.query(hidden_states).view(shape).transpose(1, 2)
        k = self.key(hidden_states).view(shape).transpose(1, 2)
        v = self.value(hidden_states).view(shape).transpose(1, 2)
        inv = 1 / math.sqrt(self.attention_head_size)
        bias = shared_bias(self, q, attention_mask, inv)
        if bias is not None and bias.shape[1] == 1:
            bias = bias.expand(b, self.num_attention_heads, n, n)
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                 dropout_p=self.dropout.p if self.training else 0.0, scale=inv)
        return (context.transpose(1, 2).reshape(b, n, self.all_head_size),)

    for module in model.modules():
        if isinstance(module, LayoutLMv3SelfAttention):
            module.forward = types.MethodType(forward, module)
    return model
