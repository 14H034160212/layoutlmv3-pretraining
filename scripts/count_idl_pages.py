"""Count the training pages of IDL shards in parallel, from the OCR json only (no image decoding), with the same
filters as pretrain.py --corpus idl (pages per document, minimum words, OCR confidence, FUNSD-test 5-gram filter).

    python scripts/count_idl_pages.py 0-259 --pages_per_doc 2 --exclude_funsd_test --out results/scale_v1/corpus_count.json

Like pretrain.py --count_pages, this is an upper bound by the few pages whose TIFF frame fails to decode.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from layoutlmv3_pretraining.corpus import NgramBlocklist, idl_pages, iter_wds_samples, resolve_idl_sources  # noqa: E402

BLOCK = None


def init(exclude, ppd, min_words):
    global BLOCK, PPD, MIN_WORDS
    PPD, MIN_WORDS = ppd, min_words
    if exclude:
        from datasets import load_dataset
        BLOCK = NgramBlocklist(load_dataset("nielsr/funsd-layoutlmv3", split="test")["tokens"], n=5, threshold=0.5)


def count(shard):
    src = resolve_idl_sources([shard])[0]
    docs = kept = excluded = 0
    with open(src, "rb") as f:
        for sample in iter_wds_samples(f):
            docs += 1
            kept += sum(1 for _ in idl_pages(sample, PPD, MIN_WORDS, decode_images=False, exclude=BLOCK))
            if BLOCK is not None:
                excluded += sum(1 for _ in idl_pages(sample, PPD, MIN_WORDS, decode_images=False,
                                                    exclude=lambda w: not BLOCK(w)))
    return shard, docs, kept, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards")
    ap.add_argument("--pages_per_doc", type=int, default=2)
    ap.add_argument("--min_words", type=int, default=10)
    ap.add_argument("--exclude_funsd_test", action="store_true")
    ap.add_argument("--processes", type=int, default=32)
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pretrain import parse_range
    shards = parse_range(a.shards)
    t0 = time.time()
    with Pool(a.processes, initializer=init, initargs=(a.exclude_funsd_test, a.pages_per_doc, a.min_words)) as pool:
        rows = sorted(pool.map(count, shards, chunksize=1))
    res = {"shards": a.shards, "pages_per_doc": a.pages_per_doc, "min_words": a.min_words,
           "exclude_funsd_test": a.exclude_funsd_test, "documents": sum(r[1] for r in rows),
           "train_pages": sum(r[2] for r in rows), "funsd_test_overlap_excluded": sum(r[3] for r in rows),
           "minutes": round((time.time() - t0) / 60, 1),
           "per_shard": {r[0]: {"docs": r[1], "pages": r[2], "excluded": r[3]} for r in rows}}
    print(json.dumps({k: v for k, v in res.items() if k != "per_shard"}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
