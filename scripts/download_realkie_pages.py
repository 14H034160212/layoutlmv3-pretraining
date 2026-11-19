"""Download a RealKIE dataset's CSVs and OCR, and only the page images that contain one of the chosen fields.

RealKIE (CC BY-NC 4.0) is served from a public bucket. Some datasets (e.g. charities, ~7,000 pages) are large as
page images, so this fetches only the pages where `--fields` occur. Pages without them are not needed for the
table-header experiment.

Example:
    python scripts/download_realkie_pages.py --dataset charities --out /data/qbao775/datasets/realkie \
        --fields "Cash In Hand at Current Year End" "Cash In Hand at Previous Year End" ...
"""
import argparse
import csv
import gzip
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(sys.maxsize)
BASE = "https://s3.us-east-2.wasabisys.com/project-fruitfly/"


def fetch(rel, root):
    path = os.path.join(root, rel)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(BASE + rel, path + ".part")
            os.replace(path + ".part", path)
            return path
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"{rel}: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    files = [fetch(f"{a.dataset}/{f}", a.out) for f in ("train.csv", "val.csv", "test.csv")]
    rows = [r for f in files for r in csv.DictReader(open(f))]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda r: fetch(r["ocr"], a.out), rows))
    want, n_docs = [], 0
    for r in rows:
        spans = [(l["start"], l["end"]) for l in json.loads(r["labels"]) if l["label"] in a.fields]
        if not spans:
            continue
        n_docs += 1
        ocr = json.load(gzip.open(os.path.join(a.out, r["ocr"])))
        images = json.loads(r["image_files"])
        for i, page in enumerate(ocr):
            offs = [t["doc_offset"] for t in page["tokens"]]
            if offs and any(o["start"] < e and o["end"] > s for s, e in spans for o in offs):
                want.append(images[i])
    print(f"docs with fields {n_docs}/{len(rows)}, pages to fetch {len(want)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda p: fetch(p, a.out), want))
    json.dump(want, open(os.path.join(a.out, a.dataset, "fetched_pages.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
