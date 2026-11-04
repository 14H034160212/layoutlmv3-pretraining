"""Paired comparison of fine-tuning results (same seeds) and an encoder-identity check.

    python scripts/compare_results.py --pairs results/idl/funsd_base_original.json results/idl_warmup/funsd_base_warmup_continued.json
    python scripts/compare_results.py --same_encoder microsoft/layoutlmv3-base runs/idl_warmup_base_pretrained/warmup_backbone
"""
import argparse
import json

import numpy as np
from scipy import stats


def paired(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    assert a["seeds"] == b["seeds"], (a["seeds"], b["seeds"])
    d = 100 * (np.array(b["f1"]) - np.array(a["f1"]))
    t = stats.ttest_rel(b["f1"], a["f1"])
    return {"a": a_path, "b": b_path, "a_mean": a["mean"], "a_std": a["std"], "b_mean": b["mean"], "b_std": b["std"],
            "diff_points": [round(x, 2) for x in d], "mean_diff_points": round(float(d.mean()), 2),
            "p_paired_t": round(float(t.pvalue), 3)}


def same_encoder(a, b):
    from transformers import LayoutLMv3Model
    sa, sb = LayoutLMv3Model.from_pretrained(a).state_dict(), LayoutLMv3Model.from_pretrained(b).state_dict()
    assert sa.keys() == sb.keys()
    diff = max((sa[k].float() - sb[k].float()).abs().max().item() for k in sa)
    return {"a": a, "b": b, "tensors": len(sa), "max_abs_diff": diff, "identical": diff == 0.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs=2, action="append", default=[])
    ap.add_argument("--same_encoder", nargs=2)
    a = ap.parse_args()
    for p in a.pairs:
        print(json.dumps(paired(*p)))
    if a.same_encoder:
        print(json.dumps(same_encoder(*a.same_encoder)))
