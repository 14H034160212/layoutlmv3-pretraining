"""Table of frozen-encoder probes (results/probes/*_log.json): metrics after the 1000 head-only steps.

    python scripts/summarize_probes.py [--markdown]
"""
import argparse
import glob
import json
import re

KEYS = ["mlm_acc", "mim_acc", "mim_majority", "mim_codes_used", "mim_code_perplexity",
        "wpa_acc", "wpa_bal_acc", "wpa_auc", "wpa_majority"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob("results/probes/*_log.json")):
        m = re.search(r"base_(pretrained|random)_wpa-(\w+?)_mim-(\w+?)_log", path)
        log = json.load(open(path))
        end = [r for r in log if r.get("phase") == "head_warmup"][-1]
        rows.append({"encoder": m[1], "wpa_head": m[2], "mim_targets": m[3], "step": end["step"],
                     **{k: end.get(k) for k in KEYS}})
    cols = ["encoder", "wpa_head", "mim_targets", "step", *KEYS]
    if a.markdown:
        print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    else:
        for r in rows:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
