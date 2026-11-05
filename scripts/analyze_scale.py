"""Scale-up v1: paired comparisons and the pre-registered decision rule (results/scale_v1/plan.md).

    python scripts/analyze_scale.py base [large] --out results/scale_v1/analysis.json

For each size: original vs 42k warm-up vs scaled checkpoint on FUNSD and CORD (seeds 1-5), paired t-tests per
task, and the rule on the 10 pooled (task, seed) differences: mean >= +0.5 points, positive on both tasks,
paired t-test p < 0.05.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

THRESHOLD = 0.5   # F1 points


def load(path):
    d = json.loads(Path(path).read_text())
    assert d["seeds"] == [1, 2, 3, 4, 5], (path, d["seeds"])
    return d


def paths(size):
    return {"original": f"results/idl/{{t}}_{size}_original.json",
            "warmup_42k": f"results/idl_warmup/{{t}}_{size}_warmup_continued.json",
            "scaled": f"results/scale_v1/{{t}}_{size}_scaled.json"}


def compare(size, a, b):
    p = paths(size)
    out, pooled = {}, []
    for task in ("funsd", "cord"):
        fa, fb = load(p[a].format(t=task))["f1"], load(p[b].format(t=task))["f1"]
        d = 100 * (np.array(fb) - np.array(fa))
        out[task] = {"diff_points": [round(float(x), 2) for x in d], "mean_diff_points": round(float(d.mean()), 2),
                     "p_paired_t": round(float(stats.ttest_rel(fb, fa).pvalue), 3)}
        pooled.extend(d)
    pooled = np.array(pooled)
    mean, p = float(pooled.mean()), float(stats.ttest_1samp(pooled, 0.0).pvalue)
    both_pos = all(out[t]["mean_diff_points"] > 0 for t in ("funsd", "cord"))
    both_neg = all(out[t]["mean_diff_points"] < 0 for t in ("funsd", "cord"))
    if mean >= THRESHOLD and both_pos and p < 0.05:
        verdict = "gain"
    elif mean <= -THRESHOLD and both_neg and p < 0.05:
        verdict = "loss"
    else:
        verdict = "no gain at this resolution"
    out["pooled"] = {"mean_diff_points": round(mean, 2), "p_paired_t": round(p, 3), "n": len(pooled),
                     "verdict": verdict}
    return out


def summary(size):
    rows = {}
    for name, p in paths(size).items():
        rows[name] = {t: {k: load(p.format(t=t))[k] for k in ("mean", "std", "f1")} for t in ("funsd", "cord")}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {}
    for size in a.sizes:
        res[size] = {"scores": summary(size),
                     "scaled_vs_original (primary)": compare(size, "original", "scaled"),
                     "scaled_vs_warmup_42k (secondary)": compare(size, "warmup_42k", "scaled"),
                     "warmup_42k_vs_original (reference)": compare(size, "original", "warmup_42k")}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
