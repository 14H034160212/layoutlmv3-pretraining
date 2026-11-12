"""Convert VRDU Ad-buy Forms (google-research-datasets/vrdu, "DeepForm") into page-level token classification.

Ad-buy forms are FCC public files for TV ad purchases: multi-page documents whose pages are dense tables of line items
(channel, program, dates, amount), like bank statements. Each page becomes one example: its OCR words with 0-1000
boxes, BIO tags over the 14 annotated fields, and the page image rendered from the PDF. Annotations are character
spans into the OCR text; a word gets the field whose span overlaps it (B- on the first word of each span).

Example:
    python scripts/prepare_vrdu.py --vrdu_dir /data/qbao775/datasets/vrdu \
        --split split_mixed_200_SD0.json --out /data/qbao775/datasets/vrdu/adbuy_pages_hf
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from datasets import Image as ImageFeature
from PIL import Image

FIELDS = ["advertiser", "agency", "contract_num", "flight_from", "flight_to", "gross_amount", "product", "tv_address",
          "property", "channel", "program_desc", "program_start_date", "program_end_date", "sub_amount"]
LABELS = ["O"] + [f"{p}-{f.upper()}" for f in FIELDS for p in ("B", "I")]


def field_spans(annotations):
    """[(field, start, end)] for every annotated value; line items are rows of several fields."""
    spans = []
    for name, values in annotations:
        if isinstance(name, str):
            rows = [[(name, v)] for v in values]
        else:  # a line item: `name` lists the fields, each value is a row with one entry per field
            rows = [list(zip(name, row)) for row in values]
        for row in rows:
            for field, (_text, _box, segments) in row:
                for start, end in segments:
                    spans.append((field, start, end))
    return spans


def page_example(page, spans, image):
    words, boxes, tags = [], [], []
    used = {}
    for tok in page["tokens"]:
        text = tok["text"].strip()
        if not text:
            continue
        start, end = tok["segments"][0][0], tok["segments"][-1][1]
        _, x0, y0, x1, y1 = tok["bbox"]
        box = [int(1000 * min(max(v, 0.0), 1.0)) for v in (x0, y0, x1, y1)]
        box[2], box[3] = max(box[2], box[0]), max(box[3], box[1])
        tag = "O"
        for i, (field, s, e) in enumerate(spans):
            if start < e and end > s:
                tag = ("I-" if i in used else "B-") + field.upper()
                used[i] = True
                break
        words.append(text); boxes.append(box); tags.append(LABELS.index(tag))
    return {"tokens": words, "bboxes": boxes, "ner_tags": tags, "image": image}


def render(pdf_path, page_index, max_side):
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        zoom = max_side / max(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrdu_dir", required=True)
    ap.add_argument("--split", required=True, help="few_shot-splits JSON with train / valid / test file names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_side", type=int, default=1000, help="longest side of the rendered page images, in pixels")
    a = ap.parse_args()
    docs = {json.loads(l)["filename"]: json.loads(l) for l in gzip.open(os.path.join(a.vrdu_dir, "dataset.jsonl.gz"), "rt")}
    split = json.load(open(os.path.join(a.vrdu_dir, a.split)))
    feats = Features({"tokens": Sequence(Value("string")), "bboxes": Sequence(Sequence(Value("int64"))),
                      "ner_tags": Sequence(ClassLabel(names=LABELS)), "image": ImageFeature(),
                      "doc": Value("string"), "page": Value("int64")})
    out, stats = {}, {}
    for name in ("train", "valid", "test"):
        rows = []
        for filename in split[name]:
            d = docs[filename]
            spans = field_spans(d["annotations"])
            pdf = os.path.join(a.vrdu_dir, "pdfs", filename)
            for i, page in enumerate(d["ocr"]["pages"]):
                ex = page_example(page, spans, render(pdf, i, a.max_side))
                if ex["tokens"]:
                    rows.append({**ex, "doc": filename, "page": i})
        out[name] = Dataset.from_list(rows, features=feats)
        tagged = np.mean([any(t != 0 for t in r["ner_tags"]) for r in rows])
        stats[name] = {"documents": len(split[name]), "pages": len(rows), "pages_with_fields": round(float(tagged), 3)}
        print(name, stats[name], flush=True)
    DatasetDict(out).save_to_disk(a.out)
    json.dump({"labels": LABELS, "stats": stats, "split": a.split}, open(os.path.join(a.out, "prepare_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
