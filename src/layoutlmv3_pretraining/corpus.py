"""Streaming pre-training corpus: Industry Documents Library pages (``pixparse/idl-wds``).

Each webdataset sample in ``pixparse/idl-wds`` is one document: a multi-page TIFF
(``.tif``) plus Textract OCR (``.json``) with, per page, text *lines* and their
boxes as ``[left, top, width, height]`` relative to the page size. We split
every line into words and give each word its line's box. That is a
segment-level layout, which is what LayoutLMv3 itself uses for pre-training.

``IDLPages`` is a ``torch.utils.data.IterableDataset`` that reads shards one by
one (from the local Hugging Face cache, or streamed over HTTP with
``HfFileSystem``), splits the shards across DataLoader workers and yields
LayoutLMv3Processor features, so the corpus never has to be materialised.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from collections import deque
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

IDL_REPO = "pixparse/idl-wds"
Image.MAX_IMAGE_PIXELS = None  # some scans are very large; they are trusted dataset files


def idl_shard_name(i: int) -> str:
    return f"idl-train-{i:05d}.tar"


def resolve_idl_sources(shards: Iterable[int], download: bool = True) -> list[str]:
    """Local cached paths (``download=True``) or ``hf://`` URLs to stream from (``download=False``)."""
    if not download:
        return [f"hf://datasets/{IDL_REPO}/{idl_shard_name(i)}" for i in shards]
    from huggingface_hub import hf_hub_download

    return [hf_hub_download(IDL_REPO, idl_shard_name(i), repo_type="dataset") for i in shards]


def _open(source: str):
    if source.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        return HfFileSystem().open(source, "rb", block_size=8 * 2 ** 20)
    return open(source, "rb")


def iter_wds_samples(fileobj) -> Iterator[dict[str, bytes]]:
    """Group the members of a webdataset tar stream into samples keyed by basename."""
    current: dict[str, bytes] = {}
    key = None
    with tarfile.open(fileobj=fileobj, mode="r|") as tar:
        for m in tar:
            if not m.isfile():
                continue
            base = m.name.rsplit("/", 1)[-1]
            k, ext = base.split(".", 1) if "." in base else (base, "")
            if k != key and current:
                yield current
                current = {}
            key = k
            current["__key__"] = k
            current[ext] = tar.extractfile(m).read()
    if current:
        yield current


def line_boxes_to_words(lines: list[str], boxes: list[list[float]], scores: list[float] | None = None,
                        min_score: float = 0.0, max_words: int = 512) -> tuple[list[str], list[list[int]]]:
    """Split OCR lines into words; each word gets its line's box converted to 0-1000 ``[x0, y0, x1, y1]``."""
    words, out = [], []
    for j, (text, (l, t, w, h)) in enumerate(zip(lines, boxes)):
        if scores is not None and scores[j] < min_score:
            continue
        x0, y0 = min(max(l, 0.0), 1.0), min(max(t, 0.0), 1.0)
        x1, y1 = min(max(l + w, 0.0), 1.0), min(max(t + h, 0.0), 1.0)
        box = [int(round(1000 * x0)), int(round(1000 * y0)), int(round(1000 * x1)), int(round(1000 * y1))]
        box[2], box[3] = max(box[2], box[0]), max(box[3], box[1])
        for word in text.split():
            words.append(word)
            out.append(box)
            if len(words) >= max_words:
                return words, out
    return words, out


def _normalise_words(words: list[str]) -> list[str]:
    return [w for w in (re.sub(r"[^0-9a-z]", "", x.lower()) for x in words) if w]


class NgramBlocklist:
    """Flags pages whose word n-grams largely occur in a set of reference documents (e.g. FUNSD test)."""

    def __init__(self, documents: Iterable[list[str]], n: int = 5, threshold: float = 0.5):
        self.n, self.threshold = n, threshold
        self.grams: set[tuple[str, ...]] = set()
        for words in documents:
            w = _normalise_words(words)
            self.grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))

    def overlap(self, words: list[str]) -> float:
        w = _normalise_words(words)
        grams = [tuple(w[i:i + self.n]) for i in range(len(w) - self.n + 1)]
        return sum(g in self.grams for g in grams) / len(grams) if grams else 0.0

    def __call__(self, words: list[str]) -> bool:
        return self.overlap(words) >= self.threshold


def idl_pages(sample: dict[str, bytes], max_pages_per_doc: int = 2, min_words: int = 10,
              min_score: float = 0.5, max_words: int = 512, decode_images: bool = True,
              exclude: Callable[[list[str]], bool] | None = None, image_size: int = 448) -> Iterator[dict]:
    """Yield ``{"key", "image", "tokens", "bboxes"}`` for the first pages of one IDL document."""
    if "json" not in sample or "tif" not in sample:
        return
    try:
        pages = json.loads(sample["json"])["pages"]
    except (ValueError, KeyError):
        return
    # Some IDL TIFFs are malformed; PIL then raises SyntaxError (not OSError), e.g. "unknown data organization".
    # A file that cannot be opened is skipped, and so is a frame that cannot be read, instead of stopping the DataLoader.
    tif, n_frames = None, len(pages)
    if decode_images:
        try:
            tif = Image.open(io.BytesIO(sample["tif"]))
            n_frames = getattr(tif, "n_frames", 1)
        except (OSError, ValueError, EOFError, SyntaxError):
            return
    kept = 0
    for i, page in enumerate(pages):
        if kept >= max_pages_per_doc or i >= n_frames:
            break
        words, boxes = line_boxes_to_words(page.get("text", []), page.get("bbox", []), page.get("score"),
                                           min_score, max_words)
        if len(words) < min_words or (exclude is not None and exclude(words)):
            continue
        image = None
        if tif is not None:
            try:
                tif.seek(i)
                im = tif.convert("L")
                factor = max(1, min(im.size) // image_size)
                image = (im.reduce(factor) if factor > 1 else im).convert("RGB")
            except (OSError, ValueError, EOFError, SyntaxError):
                continue
        kept += 1
        yield {"key": f"{sample['__key__']}/{i}", "image": image, "tokens": words, "bboxes": boxes}


class IDLPages(IterableDataset):
    """Iterable LayoutLMv3 features from IDL webdataset shards.

    The corpus is fixed: the shards in ``sources`` (split across DataLoader workers
    by index), at most ``max_pages_per_doc`` pages per document and, optionally,
    at most ``max_pages`` pages in total. ``repeat=True`` cycles over it forever;
    a shuffle buffer randomises the order within each pass.
    """

    def __init__(self, sources: list[str], processor, max_pages: int | None = None,
                 max_pages_per_doc: int = 2, min_words: int = 10, min_score: float = 0.5,
                 exclude: Callable[[list[str]], bool] | None = None, shuffle_buffer: int = 0,
                 repeat: bool = False, seed: int = 0, max_length: int = 512):
        self.sources, self.processor = list(sources), processor
        self.max_pages, self.max_pages_per_doc = max_pages, max_pages_per_doc
        self.min_words, self.min_score, self.exclude = min_words, min_score, exclude
        self.shuffle_buffer, self.repeat, self.seed, self.max_length = shuffle_buffer, repeat, seed, max_length

    def _worker_split(self) -> tuple[list[str], int, int, int | None]:
        """(sources, doc_stride, doc_offset, page quota) for this DataLoader worker.

        With at least as many shards as workers, each worker reads its own shards;
        otherwise every worker reads all shards and keeps every n-th document.
        """
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        quota = None
        if self.max_pages is not None:
            quota = self.max_pages // nw + (1 if wid < self.max_pages % nw else 0)
        if len(self.sources) >= nw:
            return self.sources[wid::nw], 1, 0, quota
        return self.sources, nw, wid, quota

    def raw_pages(self, decode_images: bool = True) -> Iterator[dict]:
        """One pass over this worker's share of the corpus (no processor, no shuffling)."""
        sources, stride, offset, quota = self._worker_split()
        n = 0
        if quota == 0:
            return
        for src in sources:
            with _open(src) as f:
                for d, sample in enumerate(iter_wds_samples(f)):
                    if d % stride != offset:
                        continue
                    for page in idl_pages(sample, self.max_pages_per_doc, self.min_words, self.min_score,
                                          decode_images=decode_images, exclude=self.exclude):
                        yield page
                        n += 1
                        if quota is not None and n >= quota:
                            return

    def encode(self, page: dict) -> dict[str, torch.Tensor]:
        enc = self.processor([page["image"]], [page["tokens"]], boxes=[page["bboxes"]], truncation=True,
                             padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {k: enc[k][0] for k in ("input_ids", "bbox", "attention_mask", "pixel_values")}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info is not None else 0])
        buf: deque = deque()
        while True:
            n = 0
            for page in self.raw_pages():
                n += 1
                item = self.encode(page)
                if self.shuffle_buffer <= 1:
                    yield item
                    continue
                buf.append(item)
                if len(buf) >= self.shuffle_buffer:
                    j = int(rng.integers(len(buf)))
                    buf[j], buf[-1] = buf[-1], buf[j]
                    yield buf.pop()
            if not self.repeat or n == 0:
                break
        order = rng.permutation(len(buf))
        items = list(buf)
        for j in order:
            yield items[j]
