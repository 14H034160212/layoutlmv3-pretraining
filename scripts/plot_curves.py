"""Plot fine-tuning learning curves (results/curves_v1) and print a mean table: test F1 and test loss vs step."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results/curves_v1")
NAMES = {"original": "microsoft/layoutlmv3-base", "warmup_42k": "+ 42k pages (head warm-up)",
         "scaled_404k": "+ 404k pages (head warm-up)", "random_init_42k": "random init + 42k pages"}
STYLE = {"original": ("k", "-"), "warmup_42k": ("tab:blue", "--"), "scaled_404k": ("tab:orange", "-"),
         "random_init_42k": ("tab:red", ":")}
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for col, task in enumerate(("funsd", "cord")):
    print(f"\n{task.upper()}: mean over seeds of test F1 / test loss")
    for name in NAMES:
        curves = json.loads((RES / f"{task}_base_{name}.json").read_text())["curves"].values()
        steps = [r["step"] for r in next(iter(curves))]
        f1 = np.array([[r["test_f1"] for r in c] for c in curves]); loss = np.array([[r["test_loss"] for r in c] for c in curves])
        c, ls = STYLE[name]
        for row, arr in enumerate((f1, loss)):
            ax = axes[row, col]; m, s = arr.mean(0), arr.std(0)
            ax.plot(steps, m, color=c, ls=ls, label=NAMES[name]); ax.fill_between(steps, m - s, m + s, color=c, alpha=0.12)
        print(f"  {NAMES[name]:32s} " + "  ".join(f"{st}:{a:.2f}/{b:.2f}" for st, a, b in zip(steps, f1.mean(0), loss.mean(0))))
    axes[0, col].set_title(f"{task.upper()} fine-tuning (base, seeds 1-3)")
axes[0, 0].set_ylabel("test entity F1"); axes[1, 0].set_ylabel("test loss")
for ax in axes[1]: ax.set_xlabel("fine-tuning step")
axes[0, 1].legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(RES / "finetune_curves.png", dpi=120)
print(f"\nsaved {RES / 'finetune_curves.png'}")
