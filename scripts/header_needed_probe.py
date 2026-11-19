"""Does a field depend on its table header? Train a plain token classifier (no header handling), then score the
same test pages three ways:
  full      the page as is
  cut       everything above the first labelled value row removed (so the column header is gone); the rest is moved
            to the top of a blank page (image and boxes)
  cut+hdr   the cut page with the column-header row put back on top (x kept)
If `cut` is clearly below `cut+hdr`, the header carries information the model needs.

Header row (RealKIE charities): the nearest row above the first value with at least two year tokens (e.g. 2016 2015).
Only pages where such a row exists are scored, so the three variants cover the same pages.

Example:
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/header_needed_probe.py --seeds 1 2 3 4 5 \
        --data /data/qbao775/datasets/realkie/charities_fin_pages_hf --out results/header_probe/charities_v3.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from PIL import Image
from seqeval.metrics import f1_score
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, get_linear_schedule_with_warmup

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from table_header_prefix import Encoder, build_train, predict  # noqa: E402

YEAR = re.compile(r"^\(?(19|20)\d{2}\)?$")


def rows_idx(words, boxes):
    items = sorted(range(len(words)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows, cur, cy = [], [], None
    for i in items:
        y = (boxes[i][1] + boxes[i][3]) / 2
        if cy is not None and abs(y - cy) > 6:
            rows.append(cur); cur = []; cy = None
        cur.append(i); cy = y if cy is None else cy
    if cur:
        rows.append(cur)
    return rows


def variants(test):
    """(full, cut, cut+hdr) datasets over the pages that have a year-header row above their first value."""
    out = {"full": [], "cut": [], "cut+hdr": []}
    for ex in test:
        w, b, g = ex["tokens"], ex["bboxes"], ex["ner_tags"]
        vals = [i for i, t in enumerate(g) if t]
        rows = rows_idx(w, b)
        first = next(r for r in rows if any(g[i] for i in r))
        top = min(b[i][1] for i in first)
        hdr = [r for r in rows if max(b[i][3] for i in r) < top and sum(bool(YEAR.match(w[i].replace(",", ""))) for i in r) >= 2]
        if not vals or not hdr:
            continue
        hdr = sorted(hdr[-1], key=lambda i: b[i][0])
        img = ex["image"].convert("RGB"); W, H = img.size
        cut = top - 2
        keep = [i for i in range(len(w)) if b[i][1] >= cut]
        h0, h1 = min(b[i][1] for i in hdr), max(b[i][3] for i in hdr)
        body = img.crop((0, int(cut / 1000 * H), W, H))
        strip = img.crop((0, int(h0 / 1000 * H), W, int(h1 / 1000 * H) + 1))
        p_cut = Image.new("RGB", (W, H), "white"); p_cut.paste(body, (0, 0))
        p_hdr = Image.new("RGB", (W, H), "white"); p_hdr.paste(strip, (0, 0)); p_hdr.paste(body, (0, strip.size[1] + 2))
        dh = h1 - h0 + 3
        base = {"doc": ex["doc"] + f"#p{ex['page']}", "page": 0}
        out["full"].append({**base, "tokens": w, "bboxes": b, "ner_tags": g, "image": img})
        cb = [[b[i][0], b[i][1] - cut, b[i][2], b[i][3] - cut] for i in keep]
        out["cut"].append({**base, "tokens": [w[i] for i in keep], "bboxes": cb, "ner_tags": [g[i] for i in keep], "image": p_cut})
        hb = [[b[i][0], b[i][1] - h0, b[i][2], b[i][3] - h0] for i in hdr]
        out["cut+hdr"].append({**base, "tokens": [w[i] for i in hdr] + [w[i] for i in keep],
                               "bboxes": hb + [[x0, min(1000, y0 + dh), x1, min(1000, y1 + dh)] for x0, y0, x1, y1 in cb],
                               "ner_tags": [0] * len(hdr) + [g[i] for i in keep], "image": p_hdr})
    feats = test.features
    return {k: Dataset.from_list(v, features=feats) for k, v in out.items()}


def score(ds, preds, names):
    """seqeval F1 over all fields, plus how often a value's year is swapped (current <-> previous, same field)."""
    yt, yp, swap, tot = [], [], 0, 0
    for i, p in preds.items():
        g = ds[i]["ner_tags"]
        yt.append([names[t] for t in g]); yp.append([names[t] for t in p])
        for a, c in zip(g, p):
            if a and names[a].startswith("B-"):
                tot += 1
                na, nc = names[a][2:], names[c][2:] if c else ""
                swap += nc != na and nc.replace("CURRENT", "X").replace("PREVIOUS", "X") == na.replace("CURRENT", "X").replace("PREVIOUS", "X")
    return {"f1": f1_score(yt, yp), "year_swap_rate": swap / max(tot, 1), "values": tot, "pages": len(preds)}


def run(args, seed, ds, names, sets, device):
    torch.manual_seed(seed); np.random.seed(seed)
    if args.family == "ernie-layout":
        from layoutlmv3_pretraining import ernie_layout
        processor, model, _ = ernie_layout.load(len(names), args.model)
        processor.tokenizer.padding_side = "right"
    else:
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(args.model, num_labels=len(names))
    model = model.to(device)
    pad_id = processor.tokenizer.pad_token_id
    enc = Encoder(processor, -1)
    items = build_train(ds["train"], enc, "none", -1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, 0, args.steps)
    g = torch.Generator().manual_seed(seed)
    model.train(); step = 0
    while step < args.steps:
        for bidx in torch.randperm(len(items), generator=g).split(args.batch_size):
            batch = [items[j] for j in bidx.tolist()]
            n = max(x["input_ids"].shape[0] for x in batch)
            pad = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}
            inp = {k: torch.stack([torch.cat([x[k], torch.full((n - x[k].shape[0],), v)]) for x in batch]) for k, v in pad.items()}
            inp["bbox"] = torch.stack([torch.cat([x["bbox"], torch.zeros(n - x["bbox"].shape[0], 4, dtype=torch.long)]) for x in batch])
            inp["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
            loss = model(**{k: v.to(device) for k, v in inp.items()}).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step >= args.steps:
                break
    out = {"all_test": score(ds["test"], predict(model, ds["test"], enc, "none", -1, device), names)}
    for k, s in sets.items():
        out[k] = score(s, predict(model, s, enc, "none", -1, device), names)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="microsoft/layoutlmv3-base")
    ap.add_argument("--family", choices=["layoutlmv3", "ernie-layout"], default="layoutlmv3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_variants", default=None, help="save the three test variants here (for inspection)")
    args = ap.parse_args()
    ds = load_from_disk(args.data)
    names = ds["train"].features["ner_tags"].feature.names
    sets = variants(ds["test"])
    print({k: len(v) for k, v in sets.items()}, flush=True)
    if args.save_variants:
        from datasets import DatasetDict
        DatasetDict({k.replace("+", "_"): v for k, v in sets.items()}).save_to_disk(args.save_variants)
    from layoutlmv3_pretraining.tracking import RunTracker
    res = {"args": vars(args), "runs": []}
    t0 = time.time()
    with RunTracker("header_needed_probe", args, tags=["table-header", "header-probe", args.family]) as tracker:
        tracker.outputs["results"] = args.out
        for seed in args.seeds:
            r = run(args, seed, ds, names, sets, "cuda:0")
            res["runs"].append({"seed": seed, **r})
            print(json.dumps(res["runs"][-1]), flush=True)
            tracker.log({"seed": seed, **{f"{m}/{k}": v for m, s in r.items() for k, v in s.items()}})
            json.dump(res, open(args.out, "w"), indent=1)
        res["minutes"] = round((time.time() - t0) / 60, 1)
        tracker.metrics.update(minutes=res["minutes"], runs=res["runs"])
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
