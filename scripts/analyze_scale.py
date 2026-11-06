"""Scale-up v1/v2: paired comparisons and the pre-registered decision rule (results/scale_v{1,2}/plan.md).

    python scripts/analyze_scale.py base [large] --out results/scale_v1/analysis.json
    python scripts/analyze_scale.py base [large] --exp scale_v2 --out results/scale_v2/analysis.json

With --exp scale_v2 the scale-v1 checkpoint is a further comparison (when its results exist).

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


EXP = "scale_v1"


def paths(size):
    p = {"original": f"results/idl/{{t}}_{size}_original.json",
         "warmup_42k": f"results/idl_warmup/{{t}}_{size}_warmup_continued.json",
         "scaled": f"results/{EXP}/{{t}}_{size}_scaled.json"}
    if EXP != "scale_v1":
        p["scale_v1"] = f"results/scale_v1/{{t}}_{size}_scaled.json"
    return {k: v for k, v in p.items() if all(Path(v.format(t=t)).exists() for t in ("funsd", "cord"))}


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
    ap.add_argument("--exp", default="scale_v1")
    ap.add_argument("--out")
    a = ap.parse_args()
    global EXP
    EXP = a.exp
    res = {}
    for size in a.sizes:
        have = paths(size)
        res[size] = {"scores": summary(size),
                     "scaled_vs_original (primary)": compare(size, "original", "scaled"),
                     "scaled_vs_warmup_42k (secondary)": compare(size, "warmup_42k", "scaled"),
                     "warmup_42k_vs_original (reference)": compare(size, "original", "warmup_42k")}
        if "scale_v1" in have:
            res[size]["scaled_vs_scale_v1 (did the fixes change anything)"] = compare(size, "scale_v1", "scaled")
            res[size]["scale_v1_vs_original (reference)"] = compare(size, "original", "scale_v1")
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
