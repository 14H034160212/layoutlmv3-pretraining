"""Analyse recipe screen v1 and apply the pre-registered decision rule (results/screen_v1/decision_rule.md).

    python scripts/analyze_screen.py            # prints a table, writes results/screen_v1/summary.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("results/screen_v1")
VARIANTS = {"A": "control (lr 2e-5, decay 0.9, linear WPA, DALL-E MIM)", "B": "two-layer MLP WPA head",
            "C": "no MIM loss", "D": "document-fine-tuned dVAE targets", "E": "joint lr 1e-5",
            "F": "embeddings + layers 0-5 frozen"}
SEEDS = [1, 2, 3]
THRESHOLD = 0.6          # F1 points, fixed in decision_rule.md before the runs


def f1_by_seed(path):
    r = json.loads(Path(path).read_text())
    return {s: 100 * f for s, f in zip(r["seeds"], r["f1"])}


def main():
    orig = {t: f1_by_seed(f"results/idl/{t}_base_original.json") for t in ("funsd", "cord")}
    rows = {}
    for v, desc in VARIANTS.items():
        try:
            ft = {t: f1_by_seed(OUT / f"{t}_{v}.json") for t in ("funsd", "cord")}
            log = json.loads((OUT / f"pretrain_{v}_log.json").read_text())
            drift = json.loads((OUT / f"drift_{v}.json").read_text())
        except FileNotFoundError:
            continue
        warm = [r for r in log if r["phase"] == "head_warmup"][-1]
        joint = [r for r in log if r["phase"] == "joint"]
        best = min(joint, key=lambda r: r["sel_loss"])
        diffs = {t: [ft[t][s] - orig[t][s] for s in SEEDS] for t in ft}
        rows[v] = {
            "description": desc,
            "funsd_mean": float(np.mean([ft["funsd"][s] for s in SEEDS])), "cord_mean": float(np.mean([ft["cord"][s] for s in SEEDS])),
            "funsd_gain": float(np.mean(diffs["funsd"])), "cord_gain": float(np.mean(diffs["cord"])),
            "score": float(np.mean(diffs["funsd"] + diffs["cord"])),
            "p_vs_original": float(stats.ttest_rel([ft[t][s] for t in ft for s in SEEDS],
                                                   [orig[t][s] for t in ft for s in SEEDS]).pvalue),
            "per_seed_diffs": diffs,
            "val_after_warmup": {k: warm[k] for k in ("loss", "mlm_acc", "mim_acc", "wpa_bal_acc", "wpa_auc")},
            "val_best": {k: best[k] for k in ("step", "loss", "mlm_loss", "mlm_acc", "mim_acc", "wpa_bal_acc", "wpa_auc")},
            "drift_all": drift["all"],
            "drift_embeddings": drift.get("depth_0"), "drift_top_layer": drift.get("depth_12"),  # base: 12 layers
            "drift": drift,
        }
    if "A" in rows:
        a = rows["A"]
        for v, r in rows.items():
            r["delta_vs_A"] = r["score"] - a["score"]
            r["delta_vs_A_funsd"] = r["funsd_gain"] - a["funsd_gain"]
            r["delta_vs_A_cord"] = r["cord_gain"] - a["cord_gain"]
            r["qualifies"] = v != "A" and r["delta_vs_A"] > THRESHOLD and r["delta_vs_A_funsd"] > 0 and r["delta_vs_A_cord"] > 0
        qual = [v for v, r in rows.items() if r["qualifies"]]
        if qual:
            choice, why = max(qual, key=lambda v: rows[v]["score"]), "beats A by more than the threshold on both tasks"
        elif "B" in rows and rows["B"]["score"] >= a["score"]:
            choice, why = "B", "tie at this resolution; B (paper's two-layer WPA head) wins ties with score >= A"
        else:
            choice, why = "A", "no variant beats A by more than the threshold on both tasks; A is kept"
    else:
        choice, why = None, "control A missing"
    complete = len(rows) == len(VARIANTS)
    summary = {"threshold_points": THRESHOLD, "variants": rows, "choice": choice, "reason": why, "complete": complete}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{'var':<4}{'FUNSD':>8}{'CORD':>8}{'dFUNSD':>8}{'dCORD':>8}{'score':>8}{'vs A':>8}{'p':>7}"
          f"{'MLM':>7}{'MIM':>7}{'WPA AUC':>8}{'drift':>8}  description")
    for v, r in rows.items():
        b = r["val_best"]
        print(f"{v:<4}{r['funsd_mean']:8.2f}{r['cord_mean']:8.2f}{r['funsd_gain']:+8.2f}{r['cord_gain']:+8.2f}"
              f"{r['score']:+8.2f}{r.get('delta_vs_A', float('nan')):+8.2f}{r['p_vs_original']:7.2f}"
              f"{b['mlm_acc']:7.3f}{b['mim_acc']:7.3f}{b['wpa_auc']:8.3f}{r['drift_all']:8.4f}  {r['description']}")
    print(f"\nchoice: {choice} ({why}); complete: {complete}")


if __name__ == "__main__":
    main()
