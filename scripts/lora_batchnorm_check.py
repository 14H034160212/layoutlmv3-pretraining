"""LoRA fine-tuning of ERNIE-Layout: does the visual backbone's BatchNorm make training and inference disagree?

With LoRA the base model is frozen, but `model.train()` still puts the ResNet's BatchNorm layers in train mode: they
normalise with each batch's own statistics and overwrite their running statistics on every forward pass. A LoRA
checkpoint usually stores only the adapter (and the new head), so inference reloads the base model with its original
running statistics. This script measures, per seed:

  in_memory  F1 of the model right after training, in eval mode (BatchNorm uses the running statistics it drifted to)
  reloaded   F1 after reloading the base model and the saved adapter (BatchNorm uses the pretrained statistics)

for `--bn train` (PyTorch default, BatchNorm follows model.train()) and `--bn frozen` (BatchNorm kept in eval mode
while training), with the image preprocessing given by `--images` (`readme`: normalised twice, as the port's README;
`paddle`: 0-255 pixels, as PaddleNLP).

Example:
    CUDA_VISIBLE_DEVICES=4 ERNIE_LAYOUT_PYTORCH=../ernie-layout-pytorch python scripts/lora_batchnorm_check.py \
        --task funsd --images readme --bn train frozen --seeds 1 2 3 --out results/lora_bn/funsd_readme.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ImageProcessor, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import finetune_funsd as ft  # noqa: E402
from layoutlmv3_pretraining import ernie_layout  # noqa: E402

MODEL = "Norm/ERNIE-Layout-Pytorch"


def load_base(num_labels, images, bn):
    processor, model, _ = ernie_layout.load(num_labels, MODEL)
    if bn == "train":   # undo ernie_layout.keep_batchnorm_frozen: BatchNorm follows model.train() again
        del model.train
    processor.tokenizer.padding_side = "right"
    if images == "readme":
        processor.image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
    return processor, model


def batchnorms(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)}


def lora(model, r):
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.1, target_modules=["query", "value"],
                     modules_to_save=["classifier"])
    return get_peft_model(model, cfg)


def run(ds, labels, seed, a, bn, device):
    torch.manual_seed(seed); np.random.seed(seed)
    processor, base = load_base(len(labels), a.images, bn)
    start = {n: (m.running_mean.clone(), m.running_var.clone()) for n, m in batchnorms(base).items()}
    cols = ds["train"].column_names
    enc = lambda e: ft.encode(processor, e, True, max_length=512, dynamic=True)  # noqa: E731
    train = ds["train"].map(enc, batched=True, remove_columns=cols)
    test = ds["test"].map(enc, batched=True, remove_columns=cols)
    tails = ft.truncated_tags(processor, ds["test"], labels, 512)
    tok = processor.tokenizer
    collate = ft.make_dynamic_collate(tok.pad_token_id, tok.padding_side, 512, True)
    for split in (train, test):
        split.set_format("torch", columns=["pixel_values"], output_all_columns=True)

    model = lora(base, a.r).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, generator=g, drop_last=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * a.steps), a.steps)
    model.train(); step = 0
    while step < a.steps:
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            model(**b).loss.backward()
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step >= a.steps:
                break

    bns = batchnorms(model)
    drift = max(float((m.running_mean.cpu() - start[n.split("base_model.model.", 1)[-1]][0]).abs().max())
                for n, m in bns.items())
    f1_mem, _ = ft.evaluate(model, test, labels, device, tails, collate)

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        saved = sorted(os.listdir(tmp))
        del model, base, opt; torch.cuda.empty_cache()
        _, fresh = load_base(len(labels), a.images, bn)
        reloaded = PeftModel.from_pretrained(fresh, tmp).to(device)
        f1_reload, _ = ft.evaluate(reloaded, test, labels, device, tails, collate)
    del reloaded, fresh; torch.cuda.empty_cache()
    return {"seed": seed, "in_memory": round(f1_mem, 4), "reloaded": round(f1_reload, 4),
            "max_running_mean_drift": round(drift, 4), "trainable_params": trainable, "saved_files": saved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="funsd", choices=["funsd", "vrdu"])
    ap.add_argument("--images", default="readme", choices=["readme", "paddle"])
    ap.add_argument("--bn", nargs="+", default=["train", "frozen"], choices=["train", "frozen"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    ds = ft.load_task(a.task)
    labels = ds["train"].features["ner_tags"].feature.names
    out = {"task": a.task, "pages": os.environ.get("VRDU_PAGES") if a.task == "vrdu" else None, "images": a.images,
           "steps": a.steps, "lr": a.lr, "batch_size": a.batch_size, "lora_r": a.r,
           "lora_targets": ["query", "value"], "protocol": "strict, 512 tokens", "results": {}}
    t0 = time.time()
    for bn in a.bn:
        rows = []
        for s in a.seeds:
            rows.append(run(ds, labels, s, a, bn, device))
            print(bn, rows[-1], flush=True)
        mem = [r["in_memory"] for r in rows]; rel = [r["reloaded"] for r in rows]
        out["results"][bn] = {"runs": rows, "in_memory_mean": round(float(np.mean(mem)), 4),
                              "reloaded_mean": round(float(np.mean(rel)), 4),
                              "gap_mean": round(float(np.mean(np.subtract(mem, rel))), 4)}
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: {x: v[x] for x in ("in_memory_mean", "reloaded_mean", "gap_mean")}
                      for k, v in out["results"].items()}), flush=True)


if __name__ == "__main__":
    main()
