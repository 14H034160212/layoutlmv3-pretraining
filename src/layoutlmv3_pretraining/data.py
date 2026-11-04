"""Turning (image, words, boxes) documents into masked pre-training batches."""

from __future__ import annotations

import random

import numpy as np
import torch

from .masking import blockwise_mask, span_mask, wpa_labels


def encode_documents(processor, examples, max_length: int = 512):
    """Tokenise a batch of documents with ``LayoutLMv3Processor`` (no OCR: words and 0-1000 boxes given)."""
    images = [im.convert("RGB") for im in examples["image"]]
    return processor(images, examples["tokens"], boxes=examples["bboxes"], truncation=True,
                     padding="max_length", max_length=max_length, return_tensors="pt")


class PretrainingCollator:
    """Applies MLM span masking, MIM blockwise masking and builds WPA labels for each example."""

    def __init__(self, tokenizer, image_tokenizer, mlm_ratio: float = 0.3, mim_ratio: float = 0.4,
                 poisson_lambda: float = 3.0, grid: int = 14, seed: int | None = None):
        self.tok, self.img_tok = tokenizer, image_tokenizer
        self.mlm_ratio, self.mim_ratio, self.lam, self.grid = mlm_ratio, mim_ratio, poisson_lambda, grid
        self.np_rng, self.py_rng = np.random.default_rng(seed), random.Random(seed)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {k: torch.stack([torch.as_tensor(f[k]) for f in features])
                 for k in ("input_ids", "bbox", "attention_mask", "pixel_values")}
        input_ids = batch["input_ids"].clone()
        mlm_labels = torch.full_like(input_ids, -100)
        wpa = torch.full_like(input_ids, -100)
        image_masks = []
        special_ids = set(self.tok.all_special_ids)
        for b in range(input_ids.shape[0]):
            ids = batch["input_ids"][b]
            special = torch.tensor([int(i) in special_ids for i in ids.tolist()])
            cand = [i for i in range(len(ids)) if not special[i]]
            masked = span_mask(cand, self.mlm_ratio, self.lam, self.np_rng)
            text_mask = torch.zeros(len(ids), dtype=torch.bool)
            text_mask[masked] = True
            mlm_labels[b, text_mask] = ids[text_mask]
            input_ids[b, text_mask] = self.tok.mask_token_id
            img_mask = torch.from_numpy(blockwise_mask(self.grid, self.grid, self.mim_ratio, rng=self.py_rng))
            image_masks.append(img_mask.reshape(-1))
            wpa[b] = wpa_labels(batch["bbox"][b], text_mask, img_mask, special, self.grid)
        bool_masked_pos = torch.stack(image_masks)
        visual_tokens = self.img_tok.encode(batch["pixel_values"])            # targets from the clean image
        mim_labels = torch.where(bool_masked_pos, visual_tokens, torch.full_like(visual_tokens, -100))
        batch.update(input_ids=input_ids, mlm_labels=mlm_labels, mim_labels=mim_labels,
                     wpa_labels=wpa, bool_masked_pos=bool_masked_pos)
        return batch
