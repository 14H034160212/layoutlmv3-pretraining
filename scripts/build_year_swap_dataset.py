"""Controlled test: make the column header necessary on RealKIE charities financial pages.

In real charity accounts the current year is the left column in 97.9% of value pairs, so a model can tell current
from prior year by position alone (results/header_probe/). Here, for a random half of the documents, every year
header row on a page (a row with two distinct year tokens, e.g. "2016 2015") has its two years swapped, in the OCR
words and in the image (old text painted over, new text drawn in the same box), and every CURRENT label on that page
becomes PREVIOUS and vice versa. On those pages the left column is the prior year. Position no longer decides; only
the header (or other year mentions on the page) does.

The year tokens of those header rows are tagged B-TABLE_HEADER on all pages (swapped or not), so
table_header_prefix.py can train the header tag and prefix.

Usage:
    .venv/bin/python scripts/build_year_swap_dataset.py --src /data/qbao775/datasets/realkie/charities_fin_pages_hf \
        --out /data/qbao775/datasets/realkie/charities_swap_hdr_hf
"""
import argparse
import json
import random
import re
import sys

from datasets import ClassLabel, DatasetDict, Sequence, load_from_disk
from PIL import ImageDraw, ImageFont

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from header_needed_probe import rows_idx  # noqa: E402

YEAR = re.compile(r"^\(?((?:19|20)\d{2})\)?$")
FONT = "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"


def year_rows(words, boxes):
    """Rows with exactly two distinct year values; returns [(row word indices, {word index: year})]."""
    out = []
    for r in rows_idx(words, boxes):
        ys = {i: YEAR.match(words[i].replace(",", "")).group(1) for i in r if YEAR.match(words[i].replace(",", ""))}
        if len(set(ys.values())) == 2:
            out.append((r, ys))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = load_from_disk(a.src)
    names = d["train"].features["ner_tags"].feature.names
    new_names = names + ["B-TABLE_HEADER"]
    hdr = len(names)
    flip = {}
    for i, n in enumerate(names):
        if "CURRENT" in n:
            flip[i] = names.index(n.replace("CURRENT", "PREVIOUS")); flip[flip[i]] = i
    rng = random.Random(a.seed)
    out, stats = {}, {}
    for split in d:
        ds = d[split]
        docs = sorted(set(ds["doc"]))
        swapped_docs = {doc for doc in docs if rng.random() < 0.5}
        st = {"pages": len(ds), "pages_with_year_header": 0, "swapped_pages": 0}

        def edit(ex):
            w, b, g = list(ex["tokens"]), ex["bboxes"], list(ex["ner_tags"])
            rows = year_rows(w, b)
            if not rows:
                return {"tokens": w, "ner_tags": g, "image": ex["image"], "swapped": 0, "has_header": 0}
            img = ex["image"].convert("RGB")
            swap = ex["doc"] in swapped_docs
            if swap:
                W, H = img.size
                draw = ImageDraw.Draw(img)
            for r, ys in rows:
                y1, y2 = sorted(set(ys.values()))
                for i, y in ys.items():
                    g[i] = hdr
                    if swap:
                        new = y2 if y == y1 else y1
                        w[i] = w[i].replace(y, new)
                        x0, t0, x1, t1 = (b[i][0] * W // 1000, b[i][1] * H // 1000, b[i][2] * W // 1000 + 1, b[i][3] * H // 1000 + 1)
                        draw.rectangle([x0 - 1, t0 - 1, x1 + 1, t1 + 1], fill="white")
                        font = ImageFont.truetype(FONT, max(8, int((t1 - t0) * 1.15)))
                        draw.text((x0, t0 - 1), w[i], fill="black", font=font)
            if swap:
                g = [flip.get(t, t) for t in g]
            return {"tokens": w, "ner_tags": g, "image": img, "swapped": int(swap), "has_header": 1}

        feats = ds.features.copy()
        feats["ner_tags"] = Sequence(ClassLabel(names=new_names))
        from datasets import Value
        feats["swapped"] = Value("int64"); feats["has_header"] = Value("int64")
        out[split] = ds.map(edit, features=feats)
        st["pages_with_year_header"] = sum(out[split]["has_header"]); st["swapped_pages"] = sum(out[split]["swapped"])
        stats[split] = st
        print(split, st, flush=True)
    DatasetDict(out).save_to_disk(a.out)
    json.dump({"stats": stats, "seed": a.seed, "src": a.src}, open(f"{a.out}/swap_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
