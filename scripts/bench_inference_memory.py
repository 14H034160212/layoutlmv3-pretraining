"""Peak GPU memory and latency of inference on one long document, fp16, batch size 1.

Compares ways of handling documents longer than 512 tokens:
  * window:       512-token windows with --stride overlap, run one at a time (LayoutLMv3 or ERNIE-Layout)
  * ernie-extrap: ERNIE-Layout with the port's RoPE extrapolation, the whole document in one sequence
  * v3-poscopy:   LayoutLMv3 with its 1D position embeddings tiled to the document length (memory only: the copied
                  positions are untrained, so no accuracy claim is made for it)

The memory cap (--cap_gb, default 15 to leave room under a 16 GB V100) is enforced with
torch.cuda.set_per_process_memory_fraction, so a configuration that needs more fails with OOM instead of taking memory
from other processes. SDPA is limited to the memory-efficient and math kernels, which V100 supports (FlashAttention
does not run on V100).

Example:
    CUDA_VISIBLE_DEVICES=4 ERNIE_LAYOUT_PYTORCH=../ernie-layout-pytorch python scripts/bench_inference_memory.py \
        --lengths 512 1024 2048 4096 --out results/long_docs/inference_memory.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image


def synthetic_document(n_tokens, tokenizer, seed=0):
    """Words and 0-1000 boxes whose tokenization is about n_tokens long (2 special tokens included)."""
    rng = np.random.default_rng(seed)
    words, boxes = [], []
    while True:
        i = len(words)
        words.append(f"w{rng.integers(0, 10**6)}")
        x, y = (i % 20) * 50, (i // 20 * 12) % 1000
        boxes.append([x, y, x + 45, y + 10])
        if i % 50 == 49 and len(tokenizer(words, boxes=boxes)["input_ids"]) >= n_tokens:
            break
    while len(tokenizer(words, boxes=boxes)["input_ids"]) > n_tokens:
        words.pop(); boxes.pop()
    image = Image.fromarray(rng.integers(0, 255, (1000, 760, 3), dtype=np.uint8))
    return words, boxes, image


def peak_forward(model, batches, device):
    """Run the batches one after another; return (peak MiB above the model weights, seconds)."""
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b in batches:
            model(**{k: v.to(device) for k, v in b.items()})
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20, time.time() - t0


def windows(processor, words, boxes, image, stride, keys):
    enc = processor([image], [words], boxes=[boxes], truncation=True, padding="max_length", max_length=512,
                    stride=stride, return_overflowing_tokens=True, return_tensors="pt")
    pix = enc["pixel_values"]
    pix = torch.stack(list(pix)) if isinstance(pix, (list, tuple)) else pix
    return [{k: (pix[i:i + 1] if k == "pixel_values" else enc[k][i:i + 1]) for k in keys}
            for i in range(len(enc["input_ids"]))]


def full_sequence(processor, words, boxes, image, length, keys):
    enc = processor([image], [words], boxes=[boxes], truncation=True, padding="max_length", max_length=length,
                    return_tensors="pt")
    pix = enc["pixel_values"]
    pix = torch.stack(list(pix)) if isinstance(pix, (list, tuple)) else pix
    return [{k: (pix if k == "pixel_values" else enc[k]) for k in keys}]


def tile_position_embeddings(model, length):
    """LayoutLMv3 with 1D position embeddings tiled to `length` (+2 for the padding offset)."""
    emb = model.layoutlmv3.embeddings
    old = emb.position_embeddings.weight.data
    n = length + 2
    new = torch.nn.Embedding(n, old.shape[1], padding_idx=emb.position_embeddings.padding_idx).to(old)
    reps = (n + old.shape[0] - 1) // old.shape[0]
    new.weight.data = old.repeat(reps, 1)[:n].clone()
    emb.position_embeddings = new
    model.config.max_position_embeddings = n
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--cap_gb", type=float, default=15.0)
    ap.add_argument("--methods", nargs="+",
                    default=["v3-window", "ernie-window", "ernie-extrap", "ernie-extrap-sdpa", "v3-poscopy"])
    ap.add_argument("--out")
    a = ap.parse_args()
    device = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gb * 2**30 / total))
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    from layoutlmv3_pretraining import ernie_layout

    v3_keys = ("input_ids", "attention_mask", "bbox", "pixel_values")
    v3_proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    results = []
    for method in a.methods:
        family = "ernie" if method.startswith("ernie") else "v3"
        if family == "ernie":
            proc, model, tok = ernie_layout.load(7, extrapolation=method.startswith("ernie-extrap"),
                                                 use_sdpa=method == "ernie-extrap-sdpa")
        else:
            proc, tok = v3_proc, v3_proc.tokenizer
            model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)
        model = model.to(device).eval()
        weights_mib = torch.cuda.memory_allocated() / 2**20
        for length in a.lengths:
            words, boxes, image = synthetic_document(length, tok)
            row = {"method": method, "doc_tokens": length, "weights_mib": round(weights_mib)}
            try:
                if method.endswith("window"):
                    batches = windows(proc, words, boxes, image, a.stride, v3_keys)
                    row["windows"] = len(batches)
                elif method == "v3-poscopy":
                    if length > 512:
                        tile_position_embeddings(model, length)
                    batches = full_sequence(proc, words, boxes, image, length, v3_keys)
                else:
                    batches = full_sequence(proc, words, boxes, image, length, v3_keys)
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    peak_forward(model, batches[:1], device)  # warm-up
                    mib, sec = peak_forward(model, batches, device)
                row.update(peak_activation_mib=round(mib), total_mib=round(mib + weights_mib), seconds=round(sec, 3),
                           status="ok")
            except torch.cuda.OutOfMemoryError:
                row.update(status=f"OOM above {a.cap_gb} GB cap")
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001  (record the failure and keep measuring the other settings)
                row.update(status=f"error: {type(e).__name__}: {str(e)[:200]}")
            print(json.dumps(row), flush=True)
            results.append(row)
        del model; torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"cap_gb": a.cap_gb, "stride": a.stride, "gpu": torch.cuda.get_device_name(0),
                       "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
