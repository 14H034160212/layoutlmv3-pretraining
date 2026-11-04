"""Fine-tune OpenAI's DALL-E dVAE (encoder + decoder) on IDL document pages, at the MIM view (112x112).

The LayoutLMv3 paper's MIM targets come from DiT's document dVAE, which is not released. This adapts the
public natural-image DALL-E dVAE to documents instead, as a closer (not identical) substitute: 42k pages,
not DiT's 42M, and a short budget.

Training: codes are the encoder's argmax (exactly what ``DalleTokenizer.encode`` emits), with a
straight-through softmax gradient (temperature ``--tau``); the loss is the MSE of the decoder's
reconstruction in [0, 1] pixel space, plus an optional KL to the uniform code prior (``--kl_weight``).
The held-out pages (default: IDL shard 2999, the MIM validation shard) are scored before and after
(MSE, PSNR, code usage), and a side-by-side image (original | DALL-E | fine-tuned) is saved.

    python scripts/finetune_dvae.py --output_dir runs/dvae_doc --steps 6000
    python scripts/pretrain.py ... --image_tokenizer dalle --dalle_encoder runs/dvae_doc/encoder.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor, get_cosine_schedule_with_warmup

from layoutlmv3_pretraining.corpus import IDLPages, NgramBlocklist, resolve_idl_sources
from layoutlmv3_pretraining.tracking import RunTracker
from layoutlmv3_pretraining.dvae import DalleTokenizer, load_dalle_decoder, load_dalle_encoder, unmap_pixels


def parse_range(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def collate(items):
    return torch.stack([torch.as_tensor(x["pixel_values"]) for x in items])


@torch.no_grad()
def evaluate(enc, dec, tok, val_pixels, device, bs=64):
    enc.eval(); dec.eval()
    mse, psnr, codes, recons = [], [], [], []
    for i in range(0, len(val_pixels), bs):
        x = tok.preprocess(val_pixels[i:i + bs].to(device))             # mapped, 112x112
        target = unmap_pixels(x)
        idx = enc(x).argmax(1)
        rec = dec.reconstruct(F.one_hot(idx, enc.vocab_size).permute(0, 3, 1, 2).float())
        err = ((rec - target) ** 2).flatten(1).mean(1)
        mse.append(err.cpu()); psnr.append((10 * torch.log10(1 / err.clamp(min=1e-10))).cpu())
        codes.append(idx.flatten().cpu())
        if i == 0:
            recons = (target[:6].cpu(), rec[:6].cpu())
    enc.train(); dec.train()
    counts = torch.bincount(torch.cat(codes), minlength=enc.vocab_size).float()
    p = counts[counts > 0] / counts.sum()
    return {"mse": round(torch.cat(mse).mean().item(), 6), "psnr": round(torch.cat(psnr).mean().item(), 3),
            "codes_used": int((counts > 0).sum()), "code_perplexity": round(math.exp(-(p * p.log()).sum().item()), 1),
            "top_code_share": round(p.max().item(), 4)}, recons


def save_side_by_side(path, target, rec_before, rec_after, scale=2):
    rows = []
    for t, a, b in zip(target, rec_before, rec_after):
        rows.append(torch.cat([t, a, b], -1))                          # 3, 112, 336
    grid = torch.cat(rows, -2)                                          # 3, 112*n, 336
    img = (grid.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--encoder", default="runs/dalle/encoder.pkl")
    ap.add_argument("--decoder", default="runs/dalle/decoder.pkl")
    ap.add_argument("--idl_shards", default="0-26")
    ap.add_argument("--val_shards", default="2999")
    ap.add_argument("--val_pages", type=int, default=256)
    ap.add_argument("--pages_per_doc", type=int, default=2)
    ap.add_argument("--exclude_funsd_test", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--kl_weight", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tags", nargs="*", default=[], help="free-form tags for experiments/registry.jsonl")
    a = ap.parse_args()
    with RunTracker("finetune_dvae", a, output_dir=a.output_dir, tags=a.tags) as run:
        run.config.update(encoder=a.encoder, decoder=a.decoder, train_shards=a.idl_shards, val_shards=a.val_shards,
                          val_pages=a.val_pages, view_size=112, steps=a.steps, batch_size=a.batch_size, lr=a.lr,
                          tau=a.tau, kl_weight=a.kl_weight, exclude_funsd_test=a.exclude_funsd_test, seed=a.seed)
        res = train(a)
        run.metrics.update(before=res["before"], after=res["after"], minutes=res["minutes"])
        run.outputs.update(encoder=f"{a.output_dir}/encoder.pt", decoder=f"{a.output_dir}/decoder.pt",
                           log=f"{a.output_dir}/log.json", results=f"{a.output_dir}/results.json",
                           reconstructions=f"{a.output_dir}/reconstructions.png")


def train(a):
    torch.manual_seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    exclude = None
    if a.exclude_funsd_test:
        from datasets import load_dataset
        exclude = NgramBlocklist(load_dataset("nielsr/funsd-layoutlmv3", split="test")["tokens"], n=5, threshold=0.5)
    common = dict(max_pages_per_doc=a.pages_per_doc, min_words=10, exclude=exclude)
    train = IDLPages(resolve_idl_sources(parse_range(a.idl_shards), download=True), processor,
                     shuffle_buffer=500, repeat=True, seed=a.seed, **common)
    val = IDLPages(resolve_idl_sources(parse_range(a.val_shards), download=True), processor,
                   max_pages=a.val_pages, **common)
    val_pixels = torch.stack([torch.as_tensor(x["pixel_values"]) for x in val])
    loader = DataLoader(train, batch_size=a.batch_size, collate_fn=collate, num_workers=a.num_workers,
                        persistent_workers=True, pin_memory=True, prefetch_factor=4, drop_last=True)

    enc = load_dalle_encoder(a.encoder).to(device).requires_grad_(True)
    dec = load_dalle_decoder(a.decoder).to(device).requires_grad_(True)
    tok = DalleTokenizer(enc)
    ev0, (target, rec0) = evaluate(enc, dec, tok, val_pixels, device)
    log = [{"step": 0, **ev0}]
    print(json.dumps(log[-1]), flush=True)

    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.02 * a.steps), a.steps)
    step, t0 = 0, time.time()
    for pixels in loader:
        x = tok.preprocess(pixels.to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = enc(x).float()                                    # B, V, 14, 14
            soft = (logits / a.tau).softmax(1)
            hard = F.one_hot(logits.argmax(1), enc.vocab_size).permute(0, 3, 1, 2).type_as(soft)
            z = hard + soft - soft.detach()                            # straight-through, argmax forward
            rec = dec.reconstruct(z).float()
        loss = rec_loss = F.mse_loss(rec, unmap_pixels(x))
        if a.kl_weight:
            logq = logits.log_softmax(1)
            kl = (logq.exp() * (logq + math.log(enc.vocab_size))).sum(1).mean()
            loss = loss + a.kl_weight * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.eval_every == 0 or step == a.steps:
            ev, _ = evaluate(enc, dec, tok, val_pixels, device)
            log.append({"step": step, "train_mse": round(rec_loss.item(), 6), **ev,
                        "minutes": round((time.time() - t0) / 60, 1)})
            print(json.dumps(log[-1]), flush=True)
            (out / "log.json").write_text(json.dumps(log, indent=1))
        if step >= a.steps:
            break

    torch.save(enc.state_dict(), out / "encoder.pt")
    torch.save(dec.state_dict(), out / "decoder.pt")
    ev1, (_, rec1) = evaluate(enc, dec, tok, val_pixels, device)
    save_side_by_side(out / "reconstructions.png", target, rec0, rec1)
    res = {"before": ev0, "after": ev1, "args": vars(a), "val_pages": len(val_pixels),
           "minutes": round((time.time() - t0) / 60, 1)}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    main()
