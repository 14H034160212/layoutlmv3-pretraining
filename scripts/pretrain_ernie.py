"""Continued pre-training of ERNIE-Layout with MVLM + TIA (see src/layoutlmv3_pretraining/ernie_pretraining.py).

Uses the same corpus options as scripts/pretrain.py (FUNSD-style Hugging Face datasets or IDL pages), the same head
warm-up, held-out validation with fixed masks and best-checkpoint selection, and the run registry. The MIM/WPA options
of pretrain.py are ignored. Needs the port on PYTHONPATH or ERNIE_LAYOUT_PYTORCH=<path>.

Example:
    ERNIE_LAYOUT_PYTORCH=../ernie-layout-pytorch python scripts/pretrain_ernie.py --corpus idl --idl_shards 0-26 \
        --exclude_funsd_test --head_warmup_steps 1000 --max_steps 6000 --lr 2e-5 --batch_size 16 --grad_accum 2 \
        --num_workers 12 --eval_every 500 --val_docs 512 --output_dir runs/ernie_idl
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretrain as base  # noqa: E402  (corpus builders and argument parser)

from layoutlmv3_pretraining.ernie_pretraining import ErnieLayoutForPreTraining, ErniePretrainingCollator  # noqa: E402
from layoutlmv3_pretraining.tracking import RunTracker  # noqa: E402


def evaluate(model, val_batches, device):
    model.eval()
    agg = {k: [] for k in ("loss", "mlm_loss", "tia_loss", "mlm_acc", "tia_acc")}
    tia_p, tia_y = [], []
    with torch.no_grad():
        for batch in val_batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(**batch)
            for k in ("loss", "mlm_loss", "tia_loss"):
                agg[k].append(getattr(out, k).item())
            agg["mlm_acc"].append(base.accuracy(out.mlm_logits, batch["mlm_labels"]))
            agg["tia_acc"].append(base.accuracy(out.tia_logits, batch["tia_labels"]))
            m = batch["tia_labels"] != -100
            tia_p.append(out.tia_logits.float().softmax(-1)[..., 1][m].cpu())
            tia_y.append(batch["tia_labels"][m].cpu())
    model.train()
    res = {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
    p, y = torch.cat(tia_p).numpy(), torch.cat(tia_y).numpy()
    res["tia_auc"] = round(float(roc_auc_score(y, p)), 4) if len(set(y.tolist())) == 2 else float("nan")
    res["tia_majority"] = round(float(max(y.mean(), 1 - y.mean())), 4)
    return res


def main():
    ap = base.build_parser()
    ap.set_defaults(model="Norm/ERNIE-Layout-Pytorch")
    ap.add_argument("--tia_ratio", type=float, default=0.15, help="probability that a text segment is covered")
    a = ap.parse_args()
    with RunTracker("pretrain_ernie", a, output_dir=a.output_dir, tags=list(a.tags) + ["ernie-layout"]) as run:
        train(a, run)


def train(a, run):
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    a.data_seed, a.resume_batches = a.seed, 0
    model, processor = ErnieLayoutForPreTraining.from_backbone(a.model)
    model.to(device)
    build = base.build_idl_corpus if a.corpus == "idl" else base.build_hf_corpus
    train_ds, val_ds, _, info = build(a, processor)
    run.config.update(model=a.model, family="ernie-layout", objectives=["mvlm", "tia"], tia_ratio=a.tia_ratio,
                      corpus=a.corpus, **{k: v for k, v in info.items() if k != "val_keys"})
    collate = ErniePretrainingCollator(processor.tokenizer, tia_ratio=a.tia_ratio, seed=a.seed)
    loader = DataLoader(train_ds, batch_size=a.batch_size, drop_last=True, collate_fn=collate,
                        shuffle=a.corpus == "hf", num_workers=a.num_workers)
    val_collate = ErniePretrainingCollator(processor.tokenizer, tia_ratio=a.tia_ratio, seed=0)
    val_batches = [val_collate(val_ds[i:i + a.batch_size]) for i in range(0, len(val_ds), a.batch_size)]

    def optimiser(params, lr, steps, warmup):
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
        return opt, get_cosine_schedule_with_warmup(opt, warmup, steps)

    warmup = a.head_warmup_steps
    if warmup:
        model.set_encoder_trainable(False)
        opt, sched = optimiser(list(model.new_parameters().values()), a.head_lr, warmup, int(0.1 * warmup))
    else:
        opt, sched = optimiser(model.parameters(), a.lr, a.max_steps, int(a.warmup_ratio * a.max_steps))

    log = [{"step": 0, **evaluate(model, val_batches, device)}]
    print(json.dumps(log[-1]), flush=True)
    run.log({f"val/{k}": v for k, v in log[-1].items() if k != "step"}, step=0)
    best, bad, step, micro, t0, total = float("inf"), 0, 0, 0, time.time(), warmup + a.max_steps
    model.train()
    while step < total:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(**batch)
            (out.loss / a.grad_accum).backward()
            micro += 1
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step == warmup and warmup:  # end of the head warm-up: train everything
                model.set_encoder_trainable(True)
                opt, sched = optimiser(model.parameters(), a.lr, a.max_steps, int(a.warmup_ratio * a.max_steps))
            if step % a.eval_every == 0 or step == warmup or step == total:
                ev = evaluate(model, val_batches, device)
                joint = step > warmup or not warmup
                improved = joint and ev["loss"] < best
                if improved:
                    best, bad = ev["loss"], 0
                    model.ernie_layout.save_pretrained(out_dir / "backbone")
                    processor.tokenizer.save_pretrained(out_dir / "backbone")
                elif joint:
                    bad += 1
                log.append({"step": step, "train_loss": round(out.loss.item(), 4), **ev, "best": improved,
                            "minutes": round((time.time() - t0) / 60, 1)})
                print(json.dumps(log[-1]), flush=True)
                run.log({f"val/{k}": v for k, v in log[-1].items() if k not in ("step", "best")}, step=step)
                if a.patience and joint and bad >= a.patience:
                    step = total
            if step >= total:
                break
    (out_dir / "pretrain_log.json").write_text(json.dumps(log, indent=1))
    joint_log = [r for r in log if r["step"] > warmup] if warmup else log[1:]
    run.metrics.update(step0=log[0], after_head_warmup=next((r for r in log if r["step"] == warmup), None) if warmup
                       else None, best_joint=min(joint_log, key=lambda r: r["loss"]) if joint_log else None)
    run.outputs["backbone"] = str(out_dir / "backbone")
    print(f"best validation loss {best:.4f}; saved to {out_dir / 'backbone'}", flush=True)


if __name__ == "__main__":
    main()
