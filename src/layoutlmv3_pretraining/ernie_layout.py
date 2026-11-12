"""ERNIE-Layout (https://arxiv.org/abs/2210.06155) through the unofficial PyTorch port
https://github.com/NormXU/ERNIE-Layout-Pytorch (MIT), weights `Norm/ERNIE-Layout-Pytorch`.

The port is not a pip package: put its repository on PYTHONPATH (or set ERNIE_LAYOUT_PYTORCH to its path). It was
written for transformers 4.28; `CompatTokenizer` lets its tokenizer run on newer transformers without changing it.

Images: the model's visual backbone subtracts Detectron-style pixel means itself, so it expects RGB pixels in 0-255 at
224 x 224, as PaddleNLP's ERNIE-Layout pipeline feeds it (`_str2im` + `_permute(to_bgr=False)`). The port's README uses
`LayoutLMv3ImageProcessor()` with its default normalisation to [-1, 1]; after the backbone's own normalisation every
page then looks nearly the same (cosine similarity 0.999 between the visual features of different FUNSD pages, against
0.94 with 0-255 input). `image_processor()` below follows PaddleNLP.
"""

from __future__ import annotations

import inspect
import os
import sys

DEFAULT_PATH = "Norm/ERNIE-Layout-Pytorch"


def _import_port():
    repo = os.environ.get("ERNIE_LAYOUT_PYTORCH")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import networks  # the port's package name
    except ImportError as e:
        raise ImportError("ERNIE-Layout needs https://github.com/NormXU/ERNIE-Layout-Pytorch on PYTHONPATH "
                          "(or ERNIE_LAYOUT_PYTORCH=<path>) and `einops`") from e
    return networks


def _drop_unknown_kwargs(fn):
    """Newer transformers pass keyword arguments (padding_side, split_special_tokens) that the port's methods do not
    accept; drop the ones the method does not know."""
    params = inspect.signature(fn).parameters
    accepts_any = any(p.kind is p.VAR_KEYWORD for p in params.values())

    def wrapper(self, *args, **kwargs):
        if not accepts_any:
            kwargs = {k: v for k, v in kwargs.items() if k in params}
        return fn(self, *args, **kwargs)

    return wrapper


def image_processor():
    """224 x 224 RGB in 0-255, resized like PaddleNLP (PIL resample 1 = LANCZOS), no rescaling or normalisation."""
    from transformers.image_utils import PILImageResampling
    from transformers.models.layoutlmv3 import LayoutLMv3ImageProcessor

    return LayoutLMv3ImageProcessor(apply_ocr=False, do_rescale=False, do_normalize=False,
                                    resample=PILImageResampling.LANCZOS)


def keep_batchnorm_frozen(model):
    """Keep the visual ResNet's BatchNorm layers in eval mode whenever `model.train()` is called. In train mode they
    update their running statistics on every forward pass, even with all parameters frozen, so a head warm-up would
    change the encoder and small fine-tuning batches would overwrite the pretrained statistics (detection backbones
    usually freeze them for the same reason)."""
    from torch import nn

    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    original_train = model.train

    def train(mode: bool = True):
        original_train(mode)
        for bn in bns:
            bn.eval()
        return model

    model.train = train
    model.train(model.training)
    return model


def use_index_select_relative_bias(model):
    """Compute the port's relative-position biases with an index lookup instead of one-hot vectors.

    The extrapolation model builds one-hot tensors of n x n x buckets (64 for the 2D biases, x and y) and projects
    them to the heads; at 4096 tokens these alone take about 9 GB. The standard model indexes the weight directly,
    whose backward is very slow (huggingface/transformers#30974). `index_select` gives the same values and gradients
    with n x n x heads memory and a fast backward."""
    import types

    import torch

    def lookup(linear, buckets, like=None):
        table = linear.weight.t()                                     # buckets x heads
        out = table.index_select(0, buckets.reshape(-1)).view(*buckets.shape, -1).permute(0, 3, 1, 2)
        if torch.is_autocast_enabled():
            out = out.to(torch.get_autocast_dtype("cuda"))
        elif like is not None:
            out = out.to(like.dtype)
        return out.contiguous()

    def make(encoder, bucket_fn):
        def cal_1d(self, *args):
            position_ids, hidden = args[-1], (args[0] if len(args) == 2 else None)
            rel = bucket_fn(position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1), num_buckets=self.rel_pos_bins,
                            max_distance=self.max_rel_pos)
            return lookup(self.rel_pos_bias, rel, hidden)

        def cal_2d(self, *args):
            bbox, hidden = args[-1], (args[0] if len(args) == 2 else None)
            x, y = bbox[:, :, 0], bbox[:, :, 3]
            rx = bucket_fn(x.unsqueeze(-2) - x.unsqueeze(-1), num_buckets=self.rel_2d_pos_bins,
                           max_distance=self.max_rel_2d_pos)
            ry = bucket_fn(y.unsqueeze(-2) - y.unsqueeze(-1), num_buckets=self.rel_2d_pos_bins,
                           max_distance=self.max_rel_2d_pos)
            return lookup(self.rel_pos_x_bias, rx, hidden) + lookup(self.rel_pos_y_bias, ry, hidden)

        if hasattr(encoder, "rel_pos_bias"):
            encoder._cal_1d_pos_emb = types.MethodType(cal_1d, encoder)
        if hasattr(encoder, "rel_pos_x_bias"):
            encoder._cal_2d_pos_emb = types.MethodType(cal_2d, encoder)

    import sys
    for module in model.modules():
        if type(module).__name__ == "ErnieLayoutEncoder":
            make(module, sys.modules[type(module).__module__].relative_position_bucket)
    return model


def extend_position_embeddings(model, max_length: int, period: int = 512):
    """Give ERNIE-Layout 1D position embeddings for `max_length` text tokens by tiling the pretrained ones: position i
    uses the pretrained row i mod `period` (the 512 positions a 512-token input uses). The port numbers text positions
    from 0 without a padding offset, and its tokenizer pads on the left, so a short document padded to a multiple of
    512 sees the same rows as when padded to 512. The port's own resize_position_embeddings adds random rows instead."""
    import torch
    from torch import nn

    emb = next(m for m in model.modules() if type(m).__name__ == "ErnieLayoutEmbeddings")
    old = emb.position_embeddings.weight.data
    if max_length <= old.shape[0]:
        return model
    index = torch.arange(max_length, device=old.device) % period
    new = nn.Embedding(max_length, old.shape[1]).to(old.device, old.dtype)
    new.weight.data = old[index].clone()
    emb.position_embeddings = new
    emb.register_buffer("position_ids", torch.arange(max_length, device=old.device).expand((1, -1)), persistent=False)
    for m in model.modules():
        if hasattr(m, "config"):
            m.config.max_position_embeddings = max_length
        if type(m).__name__ == "ErnieLayoutModel":
            m.embeddings.position_ids = emb.position_ids
    return model


def load(num_labels: int, path: str = DEFAULT_PATH, extrapolation: bool = False, use_sdpa: bool = False):
    """Return (processor, model, tokenizer) for token classification with `num_labels` classes. `use_sdpa` sets the
    port's `use_flash_attn`, which runs attention through torch's scaled_dot_product_attention."""
    networks = _import_port()

    base = networks.ErnieLayoutTokenizerFast

    class CompatTokenizer(base):
        _batch_encode_plus = _drop_unknown_kwargs(base._batch_encode_plus)
        _encode_plus = _drop_unknown_kwargs(base._encode_plus)

        def set_truncation_and_padding(self, *args, **kwargs):
            # transformers >= 4.45 made padding_side a required argument; the port calls it without one
            if "padding_side" not in kwargs and len(args) < 6:
                kwargs["padding_side"] = self.padding_side
            return super().set_truncation_and_padding(*args, **kwargs)

    tokenizer = CompatTokenizer.from_pretrained(path)
    processor = networks.ErnieLayoutProcessor(image_processor=image_processor(), tokenizer=tokenizer)
    config = networks.ErnieLayoutConfig.from_pretrained(path)
    config.num_classes = num_labels
    config.use_flash_attn = use_sdpa
    if extrapolation:
        networks.set_config_for_extrapolation(config)
        model_cls = networks.exErnieLayoutForTokenClassification
    else:
        model_cls = networks.ErnieLayoutForTokenClassification
    model = use_index_select_relative_bias(keep_batchnorm_frozen(model_cls.from_pretrained(path, config=config)))
    return processor, model, tokenizer
