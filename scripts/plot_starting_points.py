"""The key correctness figure: where continued pre-training and the fine-tuning after it start.

Top row: pre-training validation MLM accuracy and loss (512 held-out IDL pages, fixed masks) for the released encoder
(base and large) and a randomly initialised encoder, with the head-only warm-up (encoder frozen) shaded.
Bottom row: test entity F1 during FUNSD and CORD fine-tuning (base, mean of seeds 1-3) for the released checkpoint,
the continued checkpoint (scale-v2) and the randomly initialised model.

    python scripts/plot_starting_points.py   # writes results/starting_points.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
PRE = [  # (label, log, colour, style)
    ("base, released encoder (402k pages)", R / "scale_v2/pretrain_base_log.json", "tab:blue", "-"),
    ("large, released encoder (402k pages)", R / "scale_v2/pretrain_large_log.json", "tab:purple", "-"),
    ("base, random encoder (42k pages)", R / "idl_warmup/pretrain_base_scratch_log.json", "tab:red", ":"),
]
FT = [  # (label, curve file stem, colour, style, label y-offset in points)
    ("microsoft/layoutlmv3-base (released)", "original", "k", "-", -13),
    ("+ continued pre-training (402k pages, scale-v2)", "scaled_404k_v2", "tab:blue", "--", 6),
    ("random init + 42k pages pre-training", "random_init_42k", "tab:red", ":", -3),
]
WARMUP, ZOOM = 1000, 6000


def load(p):
    return json.loads(Path(p).read_text())


fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.4))
fig.suptitle("Continued pre-training starts from the released checkpoint, not from scratch", fontsize=13, weight="bold")

for label, path, c, ls in PRE:
    log = [r for r in load(path) if r["step"] <= ZOOM]
    steps = [r["step"] for r in log]
    ax[0, 0].plot(steps, [r["mlm_acc"] for r in log], c, ls=ls, marker=".", label=label)
    ax[0, 1].plot(steps, [r["loss"] for r in log], c, ls=ls, marker=".", label=label)
for a, ytext in zip(ax[0], (0.25, 0.03)):
    a.axvspan(0, WARMUP, color="0.9", zorder=0)
    a.text(WARMUP / 2, ytext, "head-only\nwarm-up\n(encoder\nfrozen)", transform=a.get_xaxis_transform(),
           ha="center", va="bottom", fontsize=8, color="0.35")
    a.set_xlabel("pre-training step (first 6,000)")
ax[0, 0].set_ylabel("validation MLM accuracy")
ax[0, 1].set_ylabel("validation loss (MLM + MIM + WPA)")
ax[0, 1].set_yscale("log")
ax[0, 0].legend(fontsize=8, loc="center right")
for label, path, c, _ in PRE:  # annotate MLM accuracy at the end of the warm-up
    r = next(r for r in load(path) if r["step"] == WARMUP)
    ax[0, 0].annotate(f"{r['mlm_acc']:.3f}", (WARMUP, r["mlm_acc"]), xytext=(6, -12 if "random" in label else 4),
                      textcoords="offset points", color=c, fontsize=9, weight="bold")
ax[0, 0].set_ylim(-0.02, 0.7)
ax[0, 0].set_title("Pre-training: the frozen released encoder already predicts masked tokens", fontsize=10)
ax[0, 1].set_title("Pre-training loss starts low and keeps falling", fontsize=10)

for col, task in enumerate(("funsd", "cord")):
    a = ax[1, col]
    for label, stem, c, ls, dy in FT:
        curves = list(load(R / f"curves_v1/{task}_base_{stem}.json")["curves"].values())
        steps = [r["step"] for r in curves[0]]
        f1 = np.array([[r["test_f1"] for r in cv] for cv in curves])
        a.plot(steps, f1.mean(0), c, ls=ls, marker=".", label=label)
        a.fill_between(steps, f1.min(0), f1.max(0), color=c, alpha=0.12)
        a.annotate(f"{f1.mean(0)[-1]:.3f}", (steps[-1], f1.mean(0)[-1]), xytext=(4, dy), textcoords="offset points",
                   color=c, fontsize=9, weight="bold")
    a.set_title(f"{task.upper()} fine-tuning: continued checkpoint tracks the released one", fontsize=10)
    a.set_xlabel("fine-tuning step")
    a.set_ylabel("test entity F1 (mean of seeds 1-3, band = min-max)")
    a.set_ylim(0, 1)
    a.set_xlim(0, 1100)
ax[1, 0].legend(fontsize=8, loc="lower right")

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = R / "starting_points.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
