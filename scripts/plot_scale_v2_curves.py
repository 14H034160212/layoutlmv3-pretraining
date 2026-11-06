"""Fine-tuning learning curves for scale-v2 next to the released checkpoint and scale-v1 (seeds 1-3 everywhere,
the seeds results/curves_v1 has), plus a mean table: test F1 / test loss vs step.

    python scripts/plot_scale_v2_curves.py base [large]
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = ["1", "2", "3"]
RUNS = {"base": {"microsoft/layoutlmv3-base": ("results/curves_v1/{t}_base_original.json", "k", "-"),
                 "+ 404k pages, scale-v1": ("results/curves_v1/{t}_base_scaled_404k.json", "tab:orange", "--"),
                 "+ 402k pages, scale-v2 (fixed)": ("results/scale_v2/{t}_base_scaled.json", "tab:blue", "-")},
        "large": {"microsoft/layoutlmv3-large": ("results/scale_v2/curves/{t}_large_original.json", "k", "-"),
                  "+ 402k pages, scale-v2 (fixed)": ("results/scale_v2/{t}_large_scaled.json", "tab:blue", "-")}}

sizes = sys.argv[1:] or ["base"]
out = {}
for size in sizes:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for col, task in enumerate(("funsd", "cord")):
        print(f"\n{size} {task.upper()}: mean over seeds 1-3 of test F1 / test loss")
        for name, (pattern, color, ls) in RUNS[size].items():
            path = Path(pattern.format(t=task))
            if not path.exists():
                print(f"  {name}: missing {path}")
                continue
            curves = json.loads(path.read_text())["curves"]
            curves = [curves[s] for s in SEEDS]
            steps = [r["step"] for r in curves[0]]
            f1 = np.array([[r["test_f1"] for r in c] for c in curves])
            loss = np.array([[r["test_loss"] for r in c] for c in curves])
            for row, arr in enumerate((f1, loss)):
                ax = axes[row, col]; m, s = arr.mean(0), arr.std(0)
                ax.plot(steps, m, color=color, ls=ls, label=name); ax.fill_between(steps, m - s, m + s, color=color, alpha=0.12)
            out[f"{size}/{task}/{name}"] = {"steps": steps, "f1": f1.mean(0).round(4).tolist(),
                                            "loss": loss.mean(0).round(4).tolist()}
            print(f"  {name:34s} " + "  ".join(f"{st}:{a:.2f}/{b:.2f}" for st, a, b in zip(steps, f1.mean(0), loss.mean(0))))
        axes[0, col].set_title(f"{task.upper()} fine-tuning ({size}, seeds 1-3)")
    axes[0, 0].set_ylabel("test entity F1"); axes[1, 0].set_ylabel("test loss")
    for ax in axes[1]:
        ax.set_xlabel("fine-tuning step")
    axes[0, 1].legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"results/scale_v2/finetune_curves_{size}.png", dpi=120)
    print(f"saved results/scale_v2/finetune_curves_{size}.png")
Path("results/scale_v2/finetune_curves_summary.json").write_text(json.dumps(out, indent=1))
