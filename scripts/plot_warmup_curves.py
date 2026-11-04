"""Validation curves: head warm-up + continued pre-training vs the earlier run vs a random encoder.

    python scripts/plot_warmup_curves.py   # needs matplotlib; writes results/idl_warmup/pretrain_curves.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [  # (label, log, colour)
    ("released encoder: 1000 head-only steps, then joint (lr 2e-5, layer decay 0.9)",
     "results/idl_warmup/pretrain_base_pretrained_log.json", "#2a78d6"),
    ("released encoder: joint from step 0 (lr 5e-5), earlier run",
     "results/idl/pretrain_base_log.json", "#eb6834"),
    ("random encoder: 1000 head-only steps, then joint (lr 1e-4)",
     "results/idl_warmup/pretrain_base_scratch_log.json", "#1baf7a"),
]
PANELS = [("mlm_acc", "MLM accuracy"), ("mim_acc", "MIM accuracy (8,192 DALL-E tokens)"),
          ("wpa_acc", "WPA accuracy")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharex=True)
for ax, (key, title) in zip(axes, PANELS):
    ax.axvspan(0, 1000, color="#f0efec", zorder=0)
    for label, path, colour in RUNS:
        log = json.load(open(path))
        ax.plot([r["step"] for r in log], [r[key] for r in log], color=colour, lw=2, marker="o", ms=3.5,
                label=label)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("optimiser step (512 held-out IDL pages, fixed masks)", fontsize=9, color=MUTED)
axes[2].axhline(0.742, color=MUTED, lw=1, ls="--")
axes[2].text(7000, 0.75, "majority class 0.742", fontsize=8, color=MUTED, ha="right", va="bottom")
axes[0].text(500, 0.25, "head-only\nwarm-up", fontsize=8, color=MUTED, ha="center", va="center")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig("results/idl_warmup/pretrain_curves.png", dpi=130, facecolor="#fcfcfb")
print("saved results/idl_warmup/pretrain_curves.png")
