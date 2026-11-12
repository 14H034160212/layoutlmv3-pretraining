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
    model = model_cls.from_pretrained(path, config=config)
    return processor, model, tokenizer
