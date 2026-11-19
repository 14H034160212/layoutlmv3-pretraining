"""Add B-TABLE_HEADER tags to RealKIE FCC pages from row-level header labels (data/table_header/) on each doc's first
line-item page, and propagate them to later pages of the same doc whose rows repeat a labelled header row (>=80%
word overlap). Row indices follow rows_idx() below (words grouped by y centre, 6-unit tolerance).

Usage: .venv/bin/python scripts/build_table_header_dataset.py /data/qbao775/datasets/realkie/fcc_invoices_pages_hdr_hf
"""
import json, os, sys
from collections import defaultdict
from datasets import load_from_disk, DatasetDict, ClassLabel, Sequence
SRC = "/data/qbao775/datasets/realkie/fcc_invoices_pages_hf"
LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "table_header")
OUT = sys.argv[1]
d = load_from_disk(SRC)
names = d["train"].features["ner_tags"].feature.names
new_names = names + ["B-TABLE_HEADER"]
H = len(names)
li = {i for i, n in enumerate(names) if "LINE_ITEM" in n}
def rows_idx(p):
    items = sorted(range(len(p["tokens"])), key=lambda i: (p["bboxes"][i][1]+p["bboxes"][i][3])/2)
    rows, cur, cy = [], [], None
    for i in items:
        b = p["bboxes"][i]; y = (b[1]+b[3])/2
        if cy is not None and abs(y-cy) > 6: rows.append(cur); cur=[]; cy=None
        cur.append(i); cy = y if cy is None else cy
    if cur: rows.append(cur)
    return rows
norm = lambda w: "".join(c for c in w.lower() if c.isalnum())
stats = {}
out = {}
for split in ("train", "valid", "test"):
    labels = {}
    for f in (["train_header_rows.json"] if split == "train" else ["test_header_rows.json"] if split == "test" else []):
        labels = json.load(open(os.path.join(LAB, f)))
    meta = {m["doc"]: m for m in json.load(open(os.path.join(LAB, f"{split}_meta.json")))} if labels else {}
    ds = d[split]
    by_doc = defaultdict(list)
    for i, (doc, page) in enumerate(zip(ds["doc"], ds["page"])): by_doc[doc].append((page, i))
    new_tags = {}
    n_first = n_prop = n_docs = 0
    for doc, pl in by_doc.items():
        m = meta.get(doc)
        if not m or m["id"] not in labels or not labels[m["id"]]: continue
        n_docs += 1
        ex = ds[m["idx"]]; rs = rows_idx(ex)
        tags = list(ex["ner_tags"]); hdr_sets = []
        for j in labels[m["id"]]:
            ids = [i for i in rs[j] if tags[i] not in li]
            for i in ids: tags[i] = H
            n_first += len(ids)
            s = {norm(ex["tokens"][i]) for i in ids} - {""}
            if len(s) >= 2: hdr_sets.append(s)
        new_tags[m["idx"]] = tags
        for page, idx in sorted(pl):
            if page <= m["page"]: continue
            q = ds[idx]; qt = list(q["ner_tags"]); changed = False
            for row in rows_idx(q):
                if any(qt[i] in li for i in row): continue
                s = {norm(q["tokens"][i]) for i in row} - {""}
                if len(s) >= 2 and any(len(s & h) / len(s) >= 0.8 and len(s & h) / len(h) >= 0.5 for h in hdr_sets):
                    for i in row: qt[i] = H
                    n_prop += len(row); changed = True
            if changed: new_tags[idx] = qt
    stats[split] = dict(docs=n_docs, first_page_tokens=n_first, propagated_tokens=n_prop)
    feats = ds.features.copy(); feats["ner_tags"] = Sequence(ClassLabel(names=new_names))
    out[split] = ds.map(lambda ex, i: {"ner_tags": new_tags.get(i, ex["ner_tags"])}, with_indices=True,
                        input_columns=None, features=feats)
DatasetDict(out).save_to_disk(OUT)
print(json.dumps(stats))
