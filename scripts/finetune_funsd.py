"""Fine-tune a LayoutLMv3 checkpoint on FUNSD entity labelling and report entity-level F1 (seqeval).

Example:
    python scripts/finetune_funsd.py --model microsoft/layoutlmv3-base --seeds 1 2 3
    python scripts/finetune_funsd.py --model runs/funsd_pt/backbone --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from datasets import load_dataset
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (LayoutLMv3ForTokenClassification, LayoutLMv3Processor,
                          get_linear_schedule_with_warmup)


def encode(processor, examples):
    images = [im.convert("RGB") for im in examples["image"]]
    return processor(images, examples["tokens"], boxes=examples["bboxes"], word_labels=examples["ner_tags"],
                     truncation=True, padding="max_length", max_length=512)


def run(model_path, seed, steps, lr, batch_size, device):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = load_dataset("nielsr/funsd-layoutlmv3")
    labels = ds["train"].features["ner_tags"].feature.names
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    cols = ds["train"].column_names
    train = ds["train"].map(lambda e: encode(processor, e), batched=True, remove_columns=cols)
    test = ds["test"].map(lambda e: encode(processor, e), batched=True, remove_columns=cols)
    train.set_format("torch"); test.set_format("torch")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_path, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)}).to(device)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train(); step = 0
    while step < steps:
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            model(**b).loss.backward()
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step >= steps:
                break
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for b in DataLoader(test, batch_size=8):
            b = {k: v.to(device) for k, v in b.items()}
            pred = model(**b).logits.argmax(-1)
            for p, l in zip(pred.cpu().tolist(), b["labels"].cpu().tolist()):
                keep = [(labels[pi], labels[li]) for pi, li in zip(p, l) if li != -100]
                y_pred.append([x for x, _ in keep]); y_true.append([y for _, y in keep])
    return f1_score(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f1s = [run(a.model, s, a.steps, a.lr, a.batch_size, device) for s in a.seeds]
    res = {"model": a.model, "seeds": a.seeds, "f1": [round(f, 4) for f in f1s],
           "mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else 0.0}
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
