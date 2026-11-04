"""Fine-tune a LayoutLMv3 checkpoint on entity labelling and report entity-level F1 (seqeval).

Tasks:
  * funsd: FUNSD (nielsr/funsd-layoutlmv3), segment-level boxes as provided
  * cord:  CORD v2 receipts (naver-clova-ix/cord-v2, CC BY 4.0), converted here to words,
           word-level 0-1000 boxes and BIO tags over its 29 line categories

Example:
    python scripts/finetune_funsd.py --model microsoft/layoutlmv3-base --seeds 1 2 3 4 5
    python scripts/finetune_funsd.py --task cord --model runs/funsd_pt/backbone --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import ClassLabel, Features, Sequence, Value, load_dataset
from datasets import Image as ImageFeature
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from layoutlmv3_pretraining.tracking import RunTracker, write_json_atomic
from transformers import (LayoutLMv3ForTokenClassification, LayoutLMv3Processor,
                          get_linear_schedule_with_warmup)

DEFAULTS = {"funsd": dict(steps=1000, lr=1e-5, batch_size=2), "cord": dict(steps=1000, lr=5e-5, batch_size=4)}


def encode(processor, examples):
    images = [im.convert("RGB") for im in examples["image"]]
    return processor(images, examples["tokens"], boxes=examples["bboxes"], word_labels=examples["ner_tags"],
                     truncation=True, padding="max_length", max_length=512)


def cord_to_token_classification(ground_truth, label2id):
    """One CORD v2 example -> words, word-level 0-1000 boxes and BIO tags (first word of a line: B-)."""
    gt = json.loads(ground_truth)
    w, h = gt["meta"]["image_size"]["width"], gt["meta"]["image_size"]["height"]
    tokens, boxes, tags = [], [], []
    for line in gt["valid_line"]:
        cat = line["category"]
        words = [x for x in line["words"] if x["text"].strip()]
        for j, word in enumerate(words):
            q = word["quad"]
            x0, x1 = min(q["x1"], q["x4"]), max(q["x2"], q["x3"])
            y0, y1 = min(q["y1"], q["y2"]), max(q["y3"], q["y4"])
            box = [int(1000 * min(max(x0 / w, 0), 1)), int(1000 * min(max(y0 / h, 0), 1)),
                   int(1000 * min(max(x1 / w, 0), 1)), int(1000 * min(max(y1 / h, 0), 1))]
            box[2], box[3] = max(box[2], box[0]), max(box[3], box[1])
            tokens.append(word["text"]); boxes.append(box)
            tags.append(label2id[("B-" if j == 0 else "I-") + cat.upper()])
    return {"tokens": tokens, "bboxes": boxes, "ner_tags": tags}


def load_task(task):
    if task == "funsd":
        return load_dataset("nielsr/funsd-layoutlmv3")
    raw = load_dataset("naver-clova-ix/cord-v2")
    cats = sorted({l["category"].upper() for s in raw for gt in raw[s]["ground_truth"]
                   for l in json.loads(gt)["valid_line"]})
    names = ["O"] + [f"{p}-{c}" for c in cats for p in ("B", "I")]
    label2id = {n: i for i, n in enumerate(names)}
    feats = Features({"tokens": Sequence(Value("string")), "bboxes": Sequence(Sequence(Value("int64"))),
                      "ner_tags": Sequence(ClassLabel(names=names)), "image": ImageFeature()})
    # only the JSON column is read, so the images are copied without being decoded and re-encoded
    return raw.map(lambda g: cord_to_token_classification(g, label2id), input_columns="ground_truth",
                   remove_columns=["ground_truth"], features=feats)


def run(ds, model_path, seed, steps, lr, batch_size, device):
    torch.manual_seed(seed); np.random.seed(seed)
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
    ap.add_argument("--task", choices=["funsd", "cord"], default="funsd")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--out")
    ap.add_argument("--tags", nargs="*", default=[], help="free-form tags for experiments/registry.jsonl")
    a = ap.parse_args()
    hp = {k: getattr(a, k) if getattr(a, k) is not None else v for k, v in DEFAULTS[a.task].items()}
    out_dir = os.path.dirname(os.path.abspath(a.out)) if a.out else None
    with RunTracker(f"finetune_{a.task}", a, tags=a.tags) as tracker:
        tracker.config.update(task=a.task, model=a.model, seeds=a.seeds, **hp,
                              dataset="nielsr/funsd-layoutlmv3" if a.task == "funsd" else "naver-clova-ix/cord-v2")
        tracker.outputs["results"] = os.path.abspath(a.out) if a.out else None
        res = finetune(a, hp, tracker)
        tracker.metrics.update({k: res[k] for k in ("f1", "mean", "std", "minutes")})
    if a.out:  # the full record next to the results file
        stem = os.path.splitext(os.path.basename(a.out))[0]
        write_json_atomic(os.path.join(out_dir, "records", f"{stem}.run_record.json"), tracker.record)


def finetune(a, hp, tracker):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_task(a.task)
    t0 = time.time()
    f1s = []
    for s in a.seeds:
        f1s.append(run(ds, a.model, s, device=device, **hp))
        tracker.metrics.setdefault("per_seed", {})[s] = round(f1s[-1], 4)
        print(json.dumps({"seed": s, "f1": round(f1s[-1], 4)}), flush=True)
    res = {"task": a.task, "model": a.model, **hp, "seeds": a.seeds, "f1": [round(f, 4) for f in f1s],
           "mean": round(float(np.mean(f1s)), 4),
           "std": round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else 0.0,
           "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
