"""Re-evaluate a saved pre-training checkpoint (backbone/ + pretraining_heads.pt) on the IDL validation pages
with the same fixed masks as scripts/pretrain.py, e.g. to add metrics that were not logged at the time.

    python scripts/eval_pretraining_checkpoint.py --backbone runs/idl_warmup_base_pretrained/backbone \\
        --heads runs/idl_warmup_base_pretrained/pretraining_heads.pt --out results/x.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import LayoutLMv3Processor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretrain import evaluate, parse_range  # noqa: E402

from layoutlmv3_pretraining import DalleTokenizer, LayoutLMv3ForPreTraining  # noqa: E402
from layoutlmv3_pretraining.corpus import IDLPages, NgramBlocklist, resolve_idl_sources  # noqa: E402
from layoutlmv3_pretraining.data import PretrainingCollator  # noqa: E402
from layoutlmv3_pretraining.modeling import WPAHead  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--wpa_head", default="auto", choices=["auto", "mlp", "linear", "mlp_patch"],
                    help="auto: read the variant off the saved head weights")
    ap.add_argument("--dalle_encoder", default="runs/dalle/encoder.pkl")
    ap.add_argument("--val_shards", default="2999")
    ap.add_argument("--val_docs", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    device = "cuda"
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    # same construction as pretrain.py's validation set (including the FUNSD-test filter, which it also applies)
    from datasets import load_dataset
    exclude = NgramBlocklist(load_dataset("nielsr/funsd-layoutlmv3", split="test")["tokens"], n=5, threshold=0.5)
    val = list(IDLPages(resolve_idl_sources(parse_range(a.val_shards), download=True), processor,
                        max_pages=a.val_docs, max_pages_per_doc=2, min_words=10, exclude=exclude))
    collate = PretrainingCollator(processor.tokenizer, None, seed=0)
    batches = [collate(val[i:i + a.batch_size]) for i in range(0, len(val), a.batch_size)]
    img_tok = DalleTokenizer.from_pretrained(a.dalle_encoder).to(device)
    heads = torch.load(a.heads, map_location="cpu", weights_only=True)
    if a.wpa_head == "auto":
        a.wpa_head = WPAHead.variant_of(heads)
    model = LayoutLMv3ForPreTraining.from_backbone(a.backbone, img_tok.vocab_size, wpa_head=a.wpa_head)
    if a.wpa_head == "linear":   # checkpoints from before WPAHead stored the linear layer as wpa_head.{weight,bias}
        heads = {("wpa_head.net." + k.split(".")[-1] if k in ("wpa_head.weight", "wpa_head.bias") else k): v
                 for k, v in heads.items()}
    missing, unexpected = model.load_state_dict(heads, strict=False)
    assert not unexpected and all(k.startswith("layoutlmv3.") for k in missing), (missing, unexpected)
    model.to(device)
    res = {"backbone": a.backbone, "heads": a.heads, "wpa_head": a.wpa_head, "val_pages": len(val),
           "image_mask_token_from_checkpoint": model.loading_info.get("image_mask_token_from_checkpoint"),
           **evaluate(model, batches, device, img_tok)}
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
