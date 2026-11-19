"""Table-header prefix for windowed LayoutLMv3 on long, multi-page invoices (RealKIE FCC).

A 512-token window that starts below a line-item table's header, or a continuation page without a printed header,
cannot see the column names. This script adds a TABLE_HEADER tag (from row-level labels, see build_hdr_ds.py) and
prepends the most recent header words seen before each window (earlier windows of the page, or earlier pages of
the document) to that window, with their original x coordinates and y moved to the top of the page. The prefix
words are not scored. Header words come from gold tags in training and from the model's own predictions at test
time (pages processed in reading order).

Modes:
  none     no header tag, no prefix (the dataset's original labels)
  tag      header tag, no prefix
  prefix   header tag, prefix: gold in training, predicted at test time
  oracle   same model as prefix, evaluated with gold header prefixes (upper bound)

Example:
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/table_header_prefix.py --mode prefix --seeds 1 2 3 \
        --data /data/qbao775/datasets/realkie/fcc_invoices_pages_hdr_hf --out results/table_header/prefix.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
from datasets import load_from_disk
from seqeval.metrics import f1_score
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, get_linear_schedule_with_warmup

MAX_LEN, OVERLAP, MAX_PREFIX_TOKENS = 512, 128, 96
MODEL_INPUTS = ("input_ids", "attention_mask", "bbox", "pixel_values", "labels")


def page_order(ds):
    docs = defaultdict(list)
    for i, (doc, page) in enumerate(zip(ds["doc"], ds["page"])):
        docs[doc].append((page, i))
    return [[i for _, i in sorted(v)] for _, v in sorted(docs.items())]


def subword_counts(tok, words, boxes):
    counts = [0] * len(words)
    for w in tok(words, boxes=boxes, truncation=False, add_special_tokens=False).word_ids():
        if w is not None:
            counts[w] += 1
    return counts


def windows_of(counts, budget):
    """Word ranges [s, e) with at most `budget` sub-tokens, consecutive windows overlapping by ~OVERLAP tokens."""
    out, s, n = [], 0, len(counts)
    while s < n:
        e, used = s, 0
        while e < n and used + counts[e] <= budget:
            used += counts[e]; e += 1
        e = max(e, s + 1)
        out.append((s, e))
        if e >= n:
            break
        back, ns = 0, e
        while ns > s + 1 and back + counts[ns - 1] <= OVERLAP:
            ns -= 1; back += counts[ns]
        s = ns
    return out


def make_prefix(words, boxes, tok):
    """Header words moved to the top of the page (x kept), cut to MAX_PREFIX_TOKENS sub-tokens."""
    if not words:
        return [], []
    y0 = min(b[1] for b in boxes)
    bx = [[b[0], b[1] - y0, b[2], b[3] - y0] for b in boxes]
    counts = subword_counts(tok, words, bx)
    keep, used = [], 0
    for i, c in enumerate(counts):
        if used + c > MAX_PREFIX_TOKENS:
            break
        keep.append(i); used += c
    return [words[i] for i in keep], [bx[i] for i in keep]


class Encoder:
    def __init__(self, processor, hdr_id):
        self.p, self.tok, self.hdr = processor, processor.tokenizer, hdr_id

    def encode(self, image, words, boxes, tags, prefix):
        pw, pb = prefix
        labels = [-100] * len(pw) + list(tags)
        enc = self.p(image, pw + words, boxes=pb + boxes, word_labels=labels, truncation=True, max_length=MAX_LEN,
                     return_tensors="pt")
        for k in [k for k in enc.keys() if k not in MODEL_INPUTS]:
            del enc[k]
        word_ids = [None if w is None or w < len(pw) else w - len(pw) for w in enc.word_ids(0)]
        return enc, word_ids


def doc_windows(ex, tok, prefix_tokens):
    counts = subword_counts(tok, ex["tokens"], ex["bboxes"])
    return windows_of(counts, MAX_LEN - 2 - prefix_tokens)


def header_words(words, boxes, is_hdr, upto):
    """Header words among the first `upto` words of a page, keeping only the last header band (rows whose y is
    within 40 units of the lowest header word above), so a stale table's header is not mixed in."""
    idx = [i for i in range(upto) if is_hdr[i]]
    if not idx:
        return None
    last_y = max(boxes[i][1] for i in idx)
    idx = [i for i in idx if boxes[i][1] >= last_y - 40]
    idx.sort(key=lambda i: (boxes[i][1], boxes[i][0]))
    return [words[i] for i in idx], [boxes[i] for i in idx]


def build_train(ds, enc, mode, hdr_id):
    """List of (encoding) training windows. With a prefix, the header comes from gold tags."""
    items = []
    for doc in page_order(ds):
        last = None
        for i in doc:
            ex = ds[i]; img = ex["image"].convert("RGB")
            words, boxes, tags = ex["tokens"], ex["bboxes"], ex["ner_tags"]
            is_hdr = [t == hdr_id for t in tags]
            budget = MAX_PREFIX_TOKENS if mode == "prefix" else 0
            for s, e in doc_windows(ex, enc.tok, budget):
                prefix = ([], [])
                if mode == "prefix":
                    h = header_words(words, boxes, is_hdr, s)
                    src = h if h is not None else last
                    if src is not None:
                        prefix = make_prefix(*src, enc.tok)
                t = tags[s:e] if mode != "none" else [0 if x == hdr_id else x for x in tags[s:e]]
                e_, _ = enc.encode(img, words[s:e], boxes[s:e], t, prefix)
                items.append({k: v[0] for k, v in e_.items()})
            h = header_words(words, boxes, is_hdr, len(words))
            if h is not None:
                last = h
    return items


@torch.no_grad()
def predict(model, ds, enc, mode, hdr_id, device):
    """Per page, the predicted tag of each word. Windows: a word takes the prediction of the window in which it is
    farthest from the window edge. Prefix headers: predicted (mode prefix) or gold (oracle)."""
    model.eval()
    preds = {}
    use_prefix = mode in ("prefix", "oracle")
    for doc in page_order(ds):
        last = None
        for i in doc:
            ex = ds[i]; img = ex["image"].convert("RGB")
            words, boxes, tags = ex["tokens"], ex["bboxes"], ex["ner_tags"]
            n = len(words)
            best = [(-1, 0)] * n
            budget = MAX_PREFIX_TOKENS if use_prefix else 0
            wins = doc_windows(ex, enc.tok, budget)
            page_pred = [0] * n
            for s, e in wins:
                prefix = ([], [])
                if use_prefix:
                    is_hdr = [t == hdr_id for t in tags] if mode == "oracle" else [p == hdr_id for p in page_pred]
                    h = header_words(words, boxes, is_hdr, s)
                    src = h if h is not None else last
                    if src is not None:
                        prefix = make_prefix(*src, enc.tok)
                e_, wid = enc.encode(img, words[s:e], boxes[s:e], [0] * (e - s), prefix)
                e_.pop("labels", None)
                logits = model(**{k: v.to(device) for k, v in e_.items()}).logits[0]
                lab = logits.argmax(-1).tolist()
                seen = set()
                for j, w in enumerate(wid):
                    if w is None or w in seen:
                        continue
                    seen.add(w)
                    g = s + w
                    margin = min(g - s, e - 1 - g) if len(wins) > 1 else 0
                    if margin > best[g][0]:
                        best[g] = (margin, lab[j]); page_pred[g] = lab[j]
            preds[i] = page_pred
            is_hdr = [t == hdr_id for t in tags] if mode == "oracle" else [p == hdr_id for p in page_pred]
            h = header_words(words, boxes, is_hdr, n)
            if h is not None:
                last = h
    model.train()
    return preds


def to_bio(seq, names, hdr_id):
    return [names[t] if t != hdr_id else "O" for t in seq]


VALUE_SUBSTR = "LINE_ITEM"   # labels containing this are the table values (set by --value_substr)


def only_values(seq):
    return [x if VALUE_SUBSTR in x else "O" for x in seq]


def year_swaps(g, p, names):
    """Values whose CURRENT/PREVIOUS year was swapped (same field otherwise), and the number of values."""
    swap = tot = 0
    for a, c in zip(g, p):
        if a and names[a].startswith("B-") and ("CURRENT" in names[a] or "PREVIOUS" in names[a]):
            tot += 1
            na, nc = names[a][2:], names[c][2:] if c else ""
            swap += nc != na and nc.replace("CURRENT", "X").replace("PREVIOUS", "X") == na.replace("CURRENT", "X").replace("PREVIOUS", "X")
    return swap, tot


def scores(ds, preds, names, hdr_id):
    li = {i for i, n in enumerate(names) if VALUE_SUBSTR in n}
    y_true, y_pred, nh_true, nh_pred = [], [], [], []
    first_li_page = {}
    for doc in page_order(ds):
        for i in doc:
            if any(t in li for t in ds[i]["ner_tags"]):
                first_li_page.setdefault(ds[i]["doc"], i); break
    tp = fp = fn = 0
    li_true, li_pred = [], []
    for i, p in preds.items():
        g = ds[i]["ner_tags"]
        y_true.append(to_bio(g, names, hdr_id)); y_pred.append(to_bio(p, names, hdr_id))
        tp += sum(a == hdr_id and b == hdr_id for a, b in zip(g, p))
        fp += sum(a != hdr_id and b == hdr_id for a, b in zip(g, p))
        fn += sum(a == hdr_id and b != hdr_id for a, b in zip(g, p))
        li_true.append(only_values(y_true[-1])); li_pred.append(only_values(y_pred[-1]))
        doc = ds[i]["doc"]
        if any(t in li for t in g) and first_li_page.get(doc) != i and not any(t == hdr_id for t in g):
            nh_true.append(li_true[-1]); nh_pred.append(li_pred[-1])
    sw = [year_swaps(ds[i]["ner_tags"], p, names) for i, p in preds.items()]
    swap_rate = sum(a for a, _ in sw) / max(sum(b for _, b in sw), 1)
    return {"f1": f1_score(y_true, y_pred), "li_f1": f1_score(li_true, li_pred), "year_swap": swap_rate,
            "li_f1_headerless_pages": f1_score(nh_true, nh_pred) if nh_true else None, "headerless_pages": len(nh_true),
            "header_p": tp / max(tp + fp, 1), "header_r": tp / max(tp + fn, 1)}


def cut_scores(cds, preds, names, hdr_id):
    """Line-item F1 on synthetic headerless chunks (build_headerless_cuts.py), overall and per cut depth k."""
    groups = defaultdict(lambda: ([], []))
    swaps = defaultdict(lambda: [0, 0])
    has_swap_col = "swapped" in cds.column_names
    for i, p in preds.items():
        if not cds[i]["cut"]:
            continue
        g = cds[i]["ner_tags"]
        t, q = only_values(to_bio(g, names, hdr_id)), only_values(to_bio(p, names, hdr_id))
        keys = ["all", f"k{cds[i]['k']}"] + ([("swapped" if cds[i]["swapped"] else "unswapped")] if has_swap_col else [])
        sw, tot = year_swaps(g, p, names)
        for key in keys:
            groups[key][0].append(t); groups[key][1].append(q)
            swaps[key][0] += sw; swaps[key][1] += tot
    out = {f"cut_li_f1_{k}": f1_score(t, q) for k, (t, q) in sorted(groups.items())} | {"cut_pages": len(groups["all"][0])}
    if swaps["all"][1]:
        out |= {f"cut_year_swap_{k}": sw / tot for k, (sw, tot) in sorted(swaps.items()) if tot}
    return out


def run(args, seed, ds, names, hdr_id, device):
    torch.manual_seed(seed); np.random.seed(seed)
    if args.family == "ernie-layout":
        from layoutlmv3_pretraining import ernie_layout
        processor, model, _ = ernie_layout.load(len(names), args.model)
        processor.tokenizer.padding_side = "right"   # padded on the right below, so positions start at 0
    else:
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(args.model, num_labels=len(names))
    model = model.to(device)
    pad_id = processor.tokenizer.pad_token_id
    enc = Encoder(processor, hdr_id)
    train_mode = "prefix" if args.mode in ("prefix", "oracle") else args.mode
    items = build_train(ds["train"], enc, train_mode, hdr_id)
    if args.train_cuts:   # cut training pages (header removed), as pseudo-docs [full page, cut page]
        items += build_train(load_from_disk(args.train_cuts)["train"], enc, train_mode, hdr_id)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, 0, args.steps)
    g = torch.Generator().manual_seed(seed)
    model.train(); step = 0
    while step < args.steps:
        for b in torch.randperm(len(items), generator=g).split(args.batch_size):
            batch = [items[j] for j in b.tolist()]
            n = max(x["input_ids"].shape[0] for x in batch)
            pad = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}
            inp = {k: torch.stack([torch.cat([x[k], torch.full((n - x[k].shape[0],), v)]) for x in batch])
                   for k, v in pad.items()}
            inp["bbox"] = torch.stack([torch.cat([x["bbox"], torch.zeros(n - x["bbox"].shape[0], 4, dtype=torch.long)])
                                       for x in batch])
            inp["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
            loss = model(**{k: v.to(device) for k, v in inp.items()}).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step >= args.steps:
                break
    out = {}
    for m in ([args.mode] if args.mode != "prefix" else ["prefix", "oracle"]):
        out[m] = scores(ds["test"], predict(model, ds["test"], enc, m, hdr_id, device), names, hdr_id)
        if args.cuts:
            cds = load_from_disk(args.cuts)["test"]
            out[m] |= cut_scores(cds, predict(model, cds, enc, m, hdr_id, device), names, hdr_id)
    return out, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["none", "tag", "prefix"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="microsoft/layoutlmv3-base")
    ap.add_argument("--family", choices=["layoutlmv3", "ernie-layout"], default="layoutlmv3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuts", default=None, help="synthetic headerless chunks from build_headerless_cuts.py")
    ap.add_argument("--train_cuts", default=None, help="build_headerless_cuts.py --split train output, added to training")
    ap.add_argument("--value_substr", default="LINE_ITEM", help="labels containing this are the table values")
    args = ap.parse_args()
    global VALUE_SUBSTR
    VALUE_SUBSTR = args.value_substr
    ds = load_from_disk(args.data)
    names = ds["train"].features["ner_tags"].feature.names
    hdr_id = names.index("B-TABLE_HEADER")
    device = "cuda:0"
    from layoutlmv3_pretraining.tracking import RunTracker
    res = {"args": vars(args), "runs": []}
    t0 = time.time()
    with RunTracker("table_header_prefix", args, tags=["table-header", "realkie-fcc", args.mode, args.family]) as tracker:
        tracker.config.update(dataset=args.data, max_len=MAX_LEN, overlap=OVERLAP, max_prefix_tokens=MAX_PREFIX_TOKENS)
        tracker.outputs["results"] = args.out
        for seed in args.seeds:
            r, n_items = run(args, seed, ds, names, hdr_id, device)
            res["runs"].append({"seed": seed, "train_windows": n_items, **r})
            print(json.dumps(res["runs"][-1]), flush=True)
            tracker.log({"seed": seed, **{f"{m}/{k}": v for m, s in r.items() for k, v in s.items() if v is not None}})
            json.dump(res, open(args.out, "w"), indent=1)
        res["minutes"] = round((time.time() - t0) / 60, 1)
        tracker.metrics.update(minutes=res["minutes"], runs=res["runs"])
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
