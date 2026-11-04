"""Continued pre-training of LayoutLMv3 with MLM + MIM + WPA on a public document dataset.

Example (FUNSD train split, words and boxes from the dataset, no OCR):
    python scripts/pretrain.py --output_dir runs/funsd_pt --max_steps 2000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor, get_cosine_schedule_with_warmup

from layoutlmv3_pretraining import KMeansPatchTokenizer, LayoutLMv3ForPreTraining
from layoutlmv3_pretraining.data import PretrainingCollator, encode_documents


def accuracy(logits, labels):
    m = labels != -100
    return (logits.argmax(-1)[m] == labels[m]).float().mean().item() if m.any() else float("nan")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    agg = {k: [] for k in ("loss", "mlm_loss", "mim_loss", "wpa_loss", "mlm_acc", "mim_acc", "wpa_acc")}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        for k in ("loss", "mlm_loss", "mim_loss", "wpa_loss"):
            agg[k].append(getattr(out, k).item())
        agg["mlm_acc"].append(accuracy(out.mlm_logits, batch["mlm_labels"]))
        agg["mim_acc"].append(accuracy(out.mim_logits, batch["mim_labels"]))
        agg["wpa_acc"].append(accuracy(out.wpa_logits, batch["wpa_labels"]))
    model.train()
    return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/layoutlmv3-base")
    ap.add_argument("--dataset", default="nielsr/funsd-layoutlmv3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val_docs", type=int, default=20, help="documents held out to monitor the objectives")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_vocab_size", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = LayoutLMv3Processor.from_pretrained(a.model, apply_ocr=False)
    raw = load_dataset(a.dataset, split=a.split).shuffle(seed=a.seed)
    enc = raw.map(lambda ex: encode_documents(processor, ex), batched=True, batch_size=16,
                  remove_columns=raw.column_names)
    enc.set_format("torch")
    val, train = enc.select(range(a.val_docs)), enc.select(range(a.val_docs, len(enc)))

    # MIM targets: a k-means visual codebook fitted on the training images only
    img_tok = KMeansPatchTokenizer.fit((train[i:i + 16]["pixel_values"] for i in range(0, len(train), 16)),
                                       vocab_size=a.image_vocab_size, seed=a.seed)
    img_tok.save(out_dir / "image_tokenizer.npz")

    model = LayoutLMv3ForPreTraining.from_backbone(a.model, a.image_vocab_size).to(device)
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, drop_last=True,
                              collate_fn=PretrainingCollator(processor.tokenizer, img_tok, seed=a.seed))
    val_loader = DataLoader(val, batch_size=a.batch_size,
                            collate_fn=PretrainingCollator(processor.tokenizer, img_tok, seed=0))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup_ratio * a.max_steps), a.max_steps)

    log = [{"step": 0, **evaluate(model, val_loader, device)}]
    print(json.dumps(log[-1]), flush=True)
    step, t0 = 0, time.time()
    while step < a.max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.eval_every == 0 or step == a.max_steps:
                log.append({"step": step, "train_loss": round(out.loss.item(), 4),
                            **evaluate(model, val_loader, device), "minutes": round((time.time() - t0) / 60, 1)})
                print(json.dumps(log[-1]), flush=True)
                (out_dir / "log.json").write_text(json.dumps(log, indent=1))
            if step >= a.max_steps:
                break

    # the backbone is a plain LayoutLMv3Model checkpoint usable by LayoutLMv3ForTokenClassification etc.
    model.layoutlmv3.save_pretrained(out_dir / "backbone")
    processor.save_pretrained(out_dir / "backbone")
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("layoutlmv3.")},
               out_dir / "pretraining_heads.pt")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
