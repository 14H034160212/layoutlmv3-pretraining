"""Inference time and peak memory on the FUNSD test set, one document at a time, with fixed padding to --max_length
versus dynamic padding (each document only as long as itself), at several maximum lengths.

Example:
    CUDA_VISIBLE_DEVICES=4 ERNIE_LAYOUT_PYTORCH=../ernie-layout-pytorch python scripts/bench_dynamic_padding.py \
        --out results/long_docs/dynamic_padding.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_funsd as ft  # noqa: E402


def load(family, max_length, num_labels):
    if family == "ernie-layout":
        from layoutlmv3_pretraining import ernie_layout

        proc, model, _ = ernie_layout.load(num_labels)
        ernie_layout.extend_position_embeddings(model, max_length)
    else:
        from layoutlmv3_pretraining.long_docs import extend_position_embeddings, use_sdpa_attention

        proc = ft.LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        model = ft.LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=num_labels)
        extend_position_embeddings(model.layoutlmv3, max_length)
        use_sdpa_attention(model)
    return proc, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["layoutlmv3", "ernie-layout"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--cap_gb", type=float, default=15.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.cap_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    ds = ft.load_task("funsd")["test"]
    labels = ds.features["ner_tags"].feature.names
    rows = []
    for family in a.families:
        for max_length in a.lengths:
            proc, model = load(family, max_length, len(labels))
            model = model.cuda().eval()
            tok = proc.tokenizer
            collate = ft.make_dynamic_collate(tok.pad_token_id, tok.padding_side, max_length, family == "ernie-layout")
            for padding in ("fixed", "dynamic"):
                enc = ds.map(lambda e: ft.encode(proc, e, True, max_length=max_length, dynamic=padding == "dynamic"),
                             batched=True, remove_columns=ds.column_names)
                if padding == "fixed":
                    enc.set_format("torch")
                    batches = [enc[i:i + 1] for i in range(len(enc))]
                else:
                    enc.set_format("torch", columns=["pixel_values"], output_all_columns=True)
                    batches = [collate([enc[i]]) for i in range(len(enc))]
                batches = [{k: v.cuda() for k, v in b.items() if k != "labels"} for b in batches]
                tokens = sum(b["input_ids"].numel() for b in batches)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    for b in batches[:3]:
                        model(**b)  # warm-up
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    t = time.perf_counter()
                    for b in batches:
                        model(**b)
                    torch.cuda.synchronize()
                sec = time.perf_counter() - t
                row = {"family": family, "max_length": max_length, "padding": padding, "documents": len(batches),
                       "tokens_processed": tokens, "seconds_total": round(sec, 3),
                       "ms_per_document": round(1000 * sec / len(batches), 1),
                       "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                print(json.dumps(row), flush=True)
                rows.append(row)
            del model
            torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(0), "cap_gb": a.cap_gb, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
