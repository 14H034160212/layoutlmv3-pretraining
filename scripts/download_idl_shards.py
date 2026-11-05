"""Download pixparse/idl-wds training shards into the Hugging Face cache (resumable; already cached shards are skipped).

    python scripts/download_idl_shards.py 0-259 --workers 8
"""
import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from layoutlmv3_pretraining.corpus import IDL_REPO, idl_shard_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", help="e.g. 0-259 or 0,5,7-9")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from pretrain import parse_range  # noqa: E402  (scripts/ is on sys.path when run as a script)
    names = [idl_shard_name(i) for i in parse_range(a.shards)]
    t0 = time.time()
    path = snapshot_download(IDL_REPO, repo_type="dataset", allow_patterns=names, max_workers=a.workers)
    size = sum((Path(path) / n).stat().st_size for n in names)
    print(f"{len(names)} shards, {size / 1e9:.1f} GB in {path}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
