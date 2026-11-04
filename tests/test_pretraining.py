import random

import numpy as np
import pytest
import torch
from transformers import LayoutLMv3Config

from layoutlmv3_pretraining import (KMeansPatchTokenizer, LayoutLMv3ForPreTraining, bbox_to_patches,
                                    blockwise_mask, span_mask, wpa_labels)


def tiny_config():
    return LayoutLMv3Config(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=64, max_position_embeddings=80, input_size=224, patch_size=16,
                            max_2d_position_embeddings=1024, coordinate_size=6, shape_size=4,
                            rel_pos_bins=32, max_rel_pos=128, max_rel_2d_pos=256, rel_2d_pos_bins=64)


def tiny_batch(b=2, t=16, vocab=100, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, vocab, (b, t), generator=g)
    x0 = torch.randint(0, 900, (b, t, 1), generator=g); y0 = torch.randint(0, 900, (b, t, 1), generator=g)
    bbox = torch.cat([x0, y0, x0 + 50, y0 + 20], -1)
    mask_pos = torch.zeros(b, 196, dtype=torch.bool); mask_pos[:, :78] = True
    return dict(input_ids=ids, bbox=bbox, attention_mask=torch.ones(b, t, dtype=torch.long),
                pixel_values=torch.randn(b, 3, 224, 224, generator=g), bool_masked_pos=mask_pos,
                mlm_labels=torch.where(torch.rand(b, t, generator=g) < 0.3, ids, -100),
                mim_labels=torch.where(mask_pos, torch.randint(0, k, (b, 196), generator=g), -100),
                wpa_labels=torch.randint(0, 2, (b, t), generator=g))


# ------------------------------------------------------------------ masking

def test_span_mask_ratio_and_contiguity():
    rng = np.random.default_rng(0)
    cand = list(range(1, 201))
    m = span_mask(cand, 0.3, 3.0, rng)
    assert len(m) == 60 and set(m) <= set(cand)
    runs = sum(1 for i, p in enumerate(m) if i == 0 or m[i - 1] != p - 1)
    assert runs < len(m) / 1.5  # spans, not isolated tokens


def test_blockwise_mask_exact_count():
    for seed in range(20):
        m = blockwise_mask(14, 14, 0.4, rng=random.Random(seed))
        assert m.shape == (14, 14) and m.sum() == round(196 * 0.4)


def test_bbox_to_patches():
    assert bbox_to_patches([0, 0, 1000, 1000]) == list(range(196))
    assert bbox_to_patches([0, 0, 71, 71]) == [0]           # inside the first 1000/14 cell
    assert bbox_to_patches([0, 0, 72, 10]) == [0, 1]        # crosses into column 1
    assert bbox_to_patches([0, 0, 0, 0]) == []              # special tokens


def test_wpa_labels():
    bbox = torch.tensor([[0, 0, 0, 0], [0, 0, 50, 50], [500, 500, 550, 550], [0, 0, 50, 50]])
    special = torch.tensor([True, False, False, False])
    text_mask = torch.tensor([False, False, False, True])
    img = torch.zeros(14, 14, dtype=torch.bool); img[7, 7] = True  # patch covering (500..571)
    assert wpa_labels(bbox, text_mask, img, special).tolist() == [-100, 1, 0, -100]


# ------------------------------------------------------------------ model

def test_all_three_losses_backpropagate_into_backbone():
    """Regression guard: every objective must produce a gradient on the shared encoder."""
    torch.manual_seed(0)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    batch = tiny_batch()
    probe = model.layoutlmv3.encoder.layer[0].attention.self.query.weight
    for name in ("mlm_loss", "mim_loss", "wpa_loss"):
        model.zero_grad()
        out = model(**batch)
        loss = getattr(out, name)
        assert loss.requires_grad and torch.isfinite(loss)
        loss.backward()
        assert probe.grad is not None and probe.grad.abs().sum() > 0, name
    model.zero_grad()
    out = model(**batch)
    torch.testing.assert_close(out.loss, out.mlm_loss + out.mim_loss + out.wpa_loss)
    out.loss.backward()
    for head in (model.mlm_head.dense, model.mim_head[1], model.wpa_head):
        assert head.weight.grad.abs().sum() > 0
    assert model.layoutlmv3.image_mask_token.grad.abs().sum() > 0


def test_image_mask_replaces_patches_only_where_masked():
    torch.manual_seed(0)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8).eval()
    b = tiny_batch(b=1)
    with torch.no_grad():
        clean = model.layoutlmv3.forward_image(b["pixel_values"])
        model.layoutlmv3._bool_masked_pos = b["bool_masked_pos"]
        masked = model.layoutlmv3.forward_image(b["pixel_values"])
        model.layoutlmv3._bool_masked_pos = None
    diff = (clean - masked).abs().sum(-1)[0, 1:]               # skip [CLS]
    assert (diff[b["bool_masked_pos"][0]] > 0).all() and (diff[~b["bool_masked_pos"][0]] == 0).all()


def test_mlm_decoder_tied_to_word_embeddings():
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    assert model.mlm_head.decoder.weight is model.layoutlmv3.embeddings.word_embeddings.weight


def test_kmeans_tokenizer_roundtrip(tmp_path):
    pv = [torch.randn(2, 3, 224, 224) for _ in range(2)]
    tok = KMeansPatchTokenizer.fit(pv, vocab_size=16, seed=0)
    ids = tok.encode(pv[0])
    assert ids.shape == (2, 196) and ids.max() < 16
    tok.save(tmp_path / "t.npz")
    assert torch.equal(KMeansPatchTokenizer.load(tmp_path / "t.npz").encode(pv[0]), ids)
