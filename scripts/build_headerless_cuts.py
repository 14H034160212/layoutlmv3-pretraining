"""Synthetic headerless test chunks for the table-header experiment.

For each test page with a gold TABLE_HEADER band followed by line-item rows, make a 2-page pseudo-document:
the original page, then a "continuation" page holding only the words below a cut (just under the header, or k
line-item rows further down). The cut page's image is the region below the cut moved to the top of a white page,
and boxes are shifted the same way, so the page looks like a continuation page without a printed header.
Scored pages: the cut pages (column `cut` = 1).
"""
import argparse
from collections import defaultdict

from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/data/qbao775/datasets/realkie/fcc_invoices_pages_hdr_hf")
ap.add_argument("--out", default="/data/qbao775/datasets/realkie/fcc_invoices_pages_hdr_cuts_hf")
ap.add_argument("--ks", type=int, nargs="+", default=[0, 4, 8])
args = ap.parse_args()

ds = load_from_disk(args.src)
t = ds["test"]
names = t.features["ner_tags"].feature.names
hdr = names.index("B-TABLE_HEADER")
li = {i for i, n in enumerate(names) if "LINE_ITEM" in n}
rows = defaultdict(list)
n_src = 0
for i in range(len(t)):
    ex = t[i]
    words, boxes, tags = ex["tokens"], ex["bboxes"], ex["ner_tags"]
    hi = [j for j, x in enumerate(tags) if x == hdr]
    if not hi:
        continue
    last_y = max(boxes[j][1] for j in hi)
    band = [j for j in hi if boxes[j][1] >= last_y - 40]
    bottom = max(boxes[j][3] for j in band)
    tops = sorted({boxes[j][1] for j, x in enumerate(tags) if x in li and boxes[j][1] > bottom})
    lines = []
    for y in tops:                      # merge tops within 5 units into one line
        if not lines or y - lines[-1] > 5:
            lines.append(y)
    if len(lines) < 2:
        continue
    n_src += 1
    img = ex["image"].convert("RGB")
    W, H = img.size
    for k in args.ks:
        if k >= len(lines) - 1:
            continue
        cut = bottom + 1 if k == 0 else lines[k] - 2
        keep = [j for j, b in enumerate(boxes) if b[1] >= cut]
        if not any(tags[j] in li for j in keep):
            continue
        px = int(cut / 1000 * H)
        page = Image.new("RGB", (W, H), "white")
        page.paste(img.crop((0, px, W, H)), (0, 0))
        did = f"{ex['doc']}#p{ex['page']}k{k}"
        for p, (w, b, g, im, c) in enumerate([
            (words, boxes, tags, img, 0),
            ([words[j] for j in keep], [[boxes[j][0], boxes[j][1] - cut, boxes[j][2], boxes[j][3] - cut] for j in keep],
             [tags[j] for j in keep], page, 1)]):
            rows["doc"].append(did); rows["page"].append(p); rows["tokens"].append(w); rows["bboxes"].append(b)
            rows["ner_tags"].append(g); rows["image"].append(im); rows["cut"].append(c); rows["k"].append(k)
feats = t.features.copy()
feats["cut"] = Value("int64"); feats["k"] = Value("int64")
out = Dataset.from_dict(dict(rows), features=Features(feats))
print("source pages", n_src, "cut pages", sum(rows["cut"]), {k: sum(1 for c, kk in zip(rows["cut"], rows["k"]) if c and kk == k) for k in args.ks})
DatasetDict({"test": out}).save_to_disk(args.out)
