"""ERNIE-Layout MVLM + TIA collator (no model or port needed)."""

from types import SimpleNamespace

import torch

from layoutlmv3_pretraining.ernie_pretraining import ErniePretrainingCollator, cover_regions


def test_cover_regions_fills_the_scaled_box_only():
    img = torch.full((3, 224, 224), 255.0)
    out = cover_regions(img, [[0, 0, 500, 250]])  # left half, top quarter
    assert (out[:, :56, :112] == 0).all()
    assert (out[:, 56:, :] == 255).all() and (out[:, :, 112:] == 255).all()
    assert (img == 255).all(), "input must not be modified"


def make_batch(n=40, seq=48):
    ids = torch.full((seq,), 1)                       # padding
    ids[0], ids[n + 1] = 0, 2                         # start and end tokens
    ids[1:n + 1] = torch.arange(10, 10 + n)
    bbox = torch.zeros(seq, 4, dtype=torch.long)
    for j in range(1, n + 1):                         # 10 segments of 4 tokens, each with its own box
        s = (j - 1) // 4
        bbox[j] = torch.tensor([50, 20 + 90 * s, 950, 80 + 90 * s])
    mask = (ids != 1).long()
    return {"input_ids": ids, "bbox": bbox, "attention_mask": mask, "pixel_values": torch.full((3, 224, 224), 200.0)}


def test_tia_labels_follow_covered_segments_and_skip_masked_and_special_tokens():
    tok = SimpleNamespace(all_special_ids=[0, 1, 2], mask_token_id=3)
    collate = ErniePretrainingCollator(tok, mlm_ratio=0.3, tia_ratio=0.5, seed=0)
    feats = [make_batch() for _ in range(4)]
    out = collate(feats)
    for b in range(4):
        ids, labels, mlm = out["input_ids"][b], out["tia_labels"][b], out["mlm_labels"][b]
        special = torch.tensor([i in (0, 1, 2) for i in feats[b]["input_ids"].tolist()])
        assert (labels[special] == -100).all()
        assert (labels[mlm != -100] == -100).all(), "MLM-masked tokens get no TIA label"
        assert (ids[mlm != -100] == 3).all()
        img = out["pixel_values"][b]
        boxes = feats[b]["bbox"]
        for j in torch.nonzero(labels != -100).flatten().tolist():
            x0, y0, x1, y1 = boxes[j].tolist()
            region = img[:, int(y0 * 224 / 1000) + 1:int(y1 * 224 / 1000) - 1, int(x0 * 224 / 1000) + 1:int(x1 * 224 / 1000) - 1]
            assert bool((region == 0).all()) == bool(labels[j] == 1), f"token {j}: label and image disagree"
        # all tokens of a segment share the label
        seg_labels = {}
        for j in torch.nonzero(labels != -100).flatten().tolist():
            seg_labels.setdefault(tuple(boxes[j].tolist()), set()).add(int(labels[j]))
        assert all(len(v) == 1 for v in seg_labels.values())
    assert (out["tia_labels"] == 1).any() and (out["tia_labels"] == 0).any()


def test_collator_is_reproducible_with_a_seed():
    tok = SimpleNamespace(all_special_ids=[0, 1, 2], mask_token_id=3)
    a = ErniePretrainingCollator(tok, seed=7)([make_batch() for _ in range(2)])
    b = ErniePretrainingCollator(tok, seed=7)([make_batch() for _ in range(2)])
    assert all(torch.equal(a[k], b[k]) for k in a)
