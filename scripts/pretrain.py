"""Continued pre-training of LayoutLMv3 with MLM + MIM + WPA.

Two corpora:
  * ``--corpus hf``  a small Hugging Face dataset with words and 0-1000 boxes (default: FUNSD train)
  * ``--corpus idl`` pages from the Industry Documents Library (pixparse/idl-wds), streamed shard by shard

Two MIM target tokenizers:
  * ``--image_tokenizer kmeans`` a k-means patch codebook fitted on training images (no external weights)
  * ``--image_tokenizer dalle``  OpenAI's DALL-E dVAE encoder (8192 tokens, on a 112x112 view, as in BEiT)

Examples:
    python scripts/pretrain.py --output_dir runs/funsd_pt --max_steps 3000
    python scripts/pretrain.py --corpus idl --idl_shards 0-26 --image_tokenizer dalle \\
        --output_dir runs/idl_dalle --max_steps 6000 --batch_size 16 --grad_accum 2 --num_workers 12

``output_dir/backbone`` holds the checkpoint with the lowest validation loss (a plain
``LayoutLMv3Model``); ``output_dir/last_backbone`` the final one if ``--save_last``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor, get_cosine_schedule_with_warmup

from layoutlmv3_pretraining import DalleTokenizer, KMeansPatchTokenizer, LayoutLMv3ForPreTraining
from layoutlmv3_pretraining.corpus import IDLPages, NgramBlocklist, resolve_idl_sources
from layoutlmv3_pretraining.data import PretrainingCollator, add_mim_labels, encode_documents


def parse_range(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def accuracy(logits, labels):
    m = labels != -100
    return (logits.argmax(-1)[m] == labels[m]).float().mean().item() if m.any() else float("nan")


def to_device(batch, device, img_tok):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    if "mim_labels" not in batch:
        add_mim_labels(batch, img_tok)
    return batch


@torch.no_grad()
def evaluate(model, val_batches, device, img_tok):
    model.eval()
    agg = {k: [] for k in ("loss", "mlm_loss", "mim_loss", "wpa_loss", "mlm_acc", "mim_acc", "wpa_acc")}
    for batch in val_batches:
        batch = to_device(batch, device, img_tok)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(**batch)
        for k in ("loss", "mlm_loss", "mim_loss", "wpa_loss"):
            agg[k].append(getattr(out, k).item())
        agg["mlm_acc"].append(accuracy(out.mlm_logits, batch["mlm_labels"]))
        agg["mim_acc"].append(accuracy(out.mim_logits, batch["mim_labels"]))
        agg["wpa_acc"].append(accuracy(out.wpa_logits, batch["wpa_labels"]))
    model.train()
    return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}


def save_backbone(model, processor, path: Path):
    model.layoutlmv3.save_pretrained(path)
    processor.save_pretrained(path)


def build_hf_corpus(a, processor):
    raw = load_dataset(a.dataset, split=a.split).shuffle(seed=a.seed)
    enc = raw.map(lambda ex: encode_documents(processor, ex), batched=True, batch_size=16,
                  remove_columns=raw.column_names)
    enc.set_format("torch")
    val, train = enc.select(range(a.val_docs)), enc.select(range(a.val_docs, len(enc)))
    fit_images = (train[i:i + 16]["pixel_values"] for i in range(0, len(train), 16))
    return train, list(val), fit_images, {"train_docs": len(train), "val_docs": len(val)}


def build_idl_corpus(a, processor):
    exclude = None
    if a.exclude_funsd_test:
        funsd = load_dataset("nielsr/funsd-layoutlmv3", split="test")
        exclude = NgramBlocklist(funsd["tokens"], n=5, threshold=0.5)
    common = dict(max_pages_per_doc=a.pages_per_doc, min_words=a.min_words, exclude=exclude)
    train = IDLPages(resolve_idl_sources(parse_range(a.idl_shards), download=not a.idl_stream), processor,
                     max_pages=a.max_pages, shuffle_buffer=a.shuffle_buffer, repeat=True, seed=a.seed, **common)
    val_ds = IDLPages(resolve_idl_sources(parse_range(a.idl_val_shards), download=not a.idl_stream), processor,
                      max_pages=a.val_docs, **common)
    val = list(val_ds)

    def fit_images():
        it = iter(IDLPages(train.sources, processor, max_pages=a.kmeans_fit_pages, **common))
        while chunk := [x["pixel_values"] for _, x in zip(range(16), it)]:
            yield torch.stack(chunk)

    info = {"train_shards": a.idl_shards, "val_shards": a.idl_val_shards, "val_pages": len(val)}
    if a.count_pages:  # one cheap pass over the OCR json only (no image decoding)
        info["train_pages"] = sum(1 for _ in IDLPages(train.sources, processor, max_pages=a.max_pages,
                                                      **common).raw_pages(decode_images=False))
        if exclude is not None:
            info["funsd_test_overlap_excluded"] = sum(
                1 for _ in IDLPages(train.sources, processor, max_pages_per_doc=a.pages_per_doc,
                                    min_words=a.min_words, exclude=lambda w: not exclude(w)
                                    ).raw_pages(decode_images=False))
    return train, val, fit_images(), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/layoutlmv3-base")
    ap.add_argument("--output_dir", required=True)
    # corpus
    ap.add_argument("--corpus", choices=["hf", "idl"], default="hf")
    ap.add_argument("--dataset", default="nielsr/funsd-layoutlmv3", help="--corpus hf: dataset with image/tokens/bboxes")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val_docs", type=int, default=20, help="documents/pages held out to monitor the objectives")
    ap.add_argument("--idl_shards", default="0-26", help="--corpus idl: training shards, e.g. 0-26 or 0,3,5-9")
    ap.add_argument("--idl_val_shards", default="2999", help="--corpus idl: shards for held-out pages")
    ap.add_argument("--idl_stream", action="store_true", help="stream shards over HTTP instead of caching them")
    ap.add_argument("--pages_per_doc", type=int, default=2)
    ap.add_argument("--min_words", type=int, default=10)
    ap.add_argument("--max_pages", type=int, default=None, help="cap on training pages (default: all in the shards)")
    ap.add_argument("--shuffle_buffer", type=int, default=500)
    ap.add_argument("--exclude_funsd_test", action="store_true",
                    help="drop IDL pages whose 5-grams largely occur in FUNSD test documents")
    ap.add_argument("--count_pages", action="store_true", help="count training pages before training (logged)")
    # MIM targets
    ap.add_argument("--image_tokenizer", choices=["kmeans", "dalle"], default="kmeans")
    ap.add_argument("--image_vocab_size", type=int, default=512, help="k-means codebook size (dalle: 8192)")
    ap.add_argument("--kmeans_fit_pages", type=int, default=2000)
    ap.add_argument("--dalle_encoder", default=None,
                    help="path to DALL-E encoder.pkl (or a state_dict .pt); downloaded from OpenAI if omitted")
    # optimisation
    ap.add_argument("--max_steps", type=int, default=2000, help="optimiser steps")
    ap.add_argument("--batch_size", type=int, default=8, help="per forward pass")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many evaluations without a lower validation loss (0: never)")
    ap.add_argument("--save_last", action="store_true", help="also save the final step to last_backbone/")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LayoutLMv3Processor.from_pretrained(a.model, apply_ocr=False)

    t0 = time.time()
    train, val, fit_images, info = (build_idl_corpus if a.corpus == "idl" else build_hf_corpus)(a, processor)

    if a.image_tokenizer == "dalle":
        img_tok = DalleTokenizer.from_pretrained(a.dalle_encoder).to(device)
    else:
        img_tok = KMeansPatchTokenizer.fit(fit_images, vocab_size=a.image_vocab_size, seed=a.seed)
        img_tok.save(out_dir / "image_tokenizer.npz")
    info.update(image_tokenizer=a.image_tokenizer, image_vocab_size=img_tok.vocab_size,
                setup_minutes=round((time.time() - t0) / 60, 1), args=vars(a))
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)

    model = LayoutLMv3ForPreTraining.from_backbone(a.model, img_tok.vocab_size).to(device)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    collate = PretrainingCollator(processor.tokenizer, None, seed=a.seed)
    train_loader = DataLoader(train, batch_size=a.batch_size, drop_last=True, collate_fn=collate,
                              shuffle=a.corpus == "hf", num_workers=a.num_workers,
                              persistent_workers=a.num_workers > 0, pin_memory=device == "cuda",
                              prefetch_factor=4 if a.num_workers > 0 else None)
    # validation masks are drawn once, so every evaluation (and the best-checkpoint choice) uses the same ones
    val_collate = PretrainingCollator(processor.tokenizer, None, seed=0)
    val_batches = [val_collate(val[i:i + a.batch_size]) for i in range(0, len(val), a.batch_size)]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup_ratio * a.max_steps), a.max_steps)

    log = [{"step": 0, **evaluate(model, val_batches, device, img_tok)}]
    print(json.dumps(log[-1]), flush=True)
    best, bad_evals, step, micro, t0 = float("inf"), 0, 0, 0, time.time()
    done = False
    while not done:
        for batch in train_loader:
            batch = to_device(batch, device, img_tok)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(**batch)
            (out.loss / a.grad_accum).backward()
            micro += 1
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.eval_every == 0 or step == a.max_steps:
                ev = evaluate(model, val_batches, device, img_tok)
                improved = ev["loss"] < best
                if improved:
                    best, bad_evals = ev["loss"], 0
                    save_backbone(model, processor, out_dir / "backbone")
                    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("layoutlmv3.")},
                               out_dir / "pretraining_heads.pt")
                else:
                    bad_evals += 1
                log.append({"step": step, "train_loss": round(out.loss.item(), 4), **ev, "best": improved,
                            "pages_seen": step * a.batch_size * a.grad_accum,
                            "minutes": round((time.time() - t0) / 60, 1)})
                print(json.dumps(log[-1]), flush=True)
                (out_dir / "log.json").write_text(json.dumps(log, indent=1))
                if a.patience and bad_evals >= a.patience:
                    print(f"early stop: no improvement in {a.patience} evaluations", flush=True)
                    done = True
            if step >= a.max_steps:
                done = True
            if done:
                break

    if a.save_last:
        save_backbone(model, processor, out_dir / "last_backbone")
    best_step = min((r for r in log[1:]), key=lambda r: r["loss"])["step"] if len(log) > 1 else 0
    print(f"best validation loss {best:.4f} at step {best_step}; saved to {out_dir / 'backbone'}", flush=True)


if __name__ == "__main__":
    main()
