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


# ------------------------------------------------------------------ DALL-E dVAE tokenizer

def _dalle_weights():
    import os
    from pathlib import Path
    for p in (os.environ.get("DALLE_ENCODER"), "runs/dalle/encoder.pkl",
              Path.home() / ".cache" / "layoutlmv3_pretraining" / "dalle_encoder.pkl"):
        if p and Path(p).exists():
            return str(p)
    return None


def test_dalle_tokenizer_preprocessing_inverts_processor_normalisation():
    from layoutlmv3_pretraining.dvae import DalleEncoder, DalleTokenizer, LOGIT_LAPLACE_EPS
    tok = DalleTokenizer(DalleEncoder(n_hid=8, vocab_size=16))
    white = torch.ones(1, 3, 224, 224)                    # processor output for a white page: (1 - 0.5) / 0.5
    black = -torch.ones(1, 3, 224, 224)
    x_w, x_b = tok.preprocess(white), tok.preprocess(black)
    assert x_w.shape == (1, 3, 112, 112)
    torch.testing.assert_close(x_w, torch.full_like(x_w, 1 - LOGIT_LAPLACE_EPS))
    torch.testing.assert_close(x_b, torch.full_like(x_b, LOGIT_LAPLACE_EPS))
    ids = tok.encode(torch.rand(2, 3, 224, 224) * 2 - 1)
    assert ids.shape == (2, 196) and ids.dtype == torch.long and int(ids.max()) < 16 and tok.vocab_size == 16


@pytest.mark.skipif(_dalle_weights() is None, reason="DALL-E encoder.pkl not available (set DALLE_ENCODER)")
def test_dalle_tokenizer_with_openai_weights():
    from layoutlmv3_pretraining import DalleTokenizer
    tok = DalleTokenizer.from_pretrained(_dalle_weights())
    assert tok.vocab_size == 8192
    g = torch.Generator().manual_seed(0)
    pv = torch.rand(2, 3, 224, 224, generator=g) * 2 - 1
    ids = tok.encode(pv)
    assert ids.shape == (2, 196) and ids.dtype == torch.long and 0 <= int(ids.min()) and int(ids.max()) < 8192
    # a blank page is dominated by one token (borders differ because of zero padding); noise is not
    blank = tok.encode(torch.ones(1, 3, 224, 224))
    assert torch.bincount(blank[0]).max() > 196 // 2 and ids[0].unique().numel() > 20


def test_collator_without_image_tokenizer_defers_mim_labels():
    from layoutlmv3_pretraining import KMeansPatchTokenizer
    from layoutlmv3_pretraining.data import PretrainingCollator, add_mim_labels

    class Tok:
        all_special_ids, mask_token_id = [0, 1, 2], 3

    b = tiny_batch(b=2, t=16)
    feats = [{k: b[k][i] for k in ("input_ids", "bbox", "attention_mask", "pixel_values")} for i in range(2)]
    out = PretrainingCollator(Tok(), None, seed=0)(feats)
    assert "mim_labels" not in out and out["bool_masked_pos"].sum(1).tolist() == [78, 78]
    tok = KMeansPatchTokenizer(np.random.default_rng(0).normal(size=(8, 16)))
    add_mim_labels(out, tok)
    m = out["bool_masked_pos"]
    assert (out["mim_labels"][~m] == -100).all() and (out["mim_labels"][m] >= 0).all()


# ------------------------------------------------------------------ IDL corpus

def test_line_boxes_to_words_normalises_to_0_1000():
    from layoutlmv3_pretraining.corpus import line_boxes_to_words
    words, boxes = line_boxes_to_words(["Hello world", "low", "x"], [[0.1, 0.2, 0.3, 0.05], [0, 0, 1, 1],
                                                                     [0.95, 0.9, 0.2, 0.2]],
                                       scores=[0.99, 0.1, 0.9], min_score=0.5)
    assert words == ["Hello", "world", "x"]
    assert boxes == [[100, 200, 400, 250], [100, 200, 400, 250], [950, 900, 1000, 1000]]


def test_idl_pages_from_webdataset_tar(tmp_path):
    import io, json, tarfile
    from PIL import Image
    from layoutlmv3_pretraining.corpus import IDLPages, NgramBlocklist
    page = {"text": ["one two three four five six seven", "eight nine ten"],
            "bbox": [[0.1, 0.1, 0.5, 0.02], [0.1, 0.2, 0.3, 0.02]], "score": [0.9, 0.9]}
    frames = [Image.new("1", (1000, 1300), 1) for _ in range(3)]
    tif = io.BytesIO(); frames[0].save(tif, format="TIFF", save_all=True, append_images=frames[1:])
    members = {"doc1.json": json.dumps({"pages": [page] * 3}).encode(), "doc1.tif": tif.getvalue(),
               "doc2.json": json.dumps({"pages": [{"text": ["too short"], "bbox": [[0, 0, 1, 1]]}]}).encode(),
               "doc2.tif": tif.getvalue()}
    path = tmp_path / "shard.tar"
    with tarfile.open(path, "w") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name); info.size = len(data); t.addfile(info, io.BytesIO(data))

    class Proc:  # stands in for LayoutLMv3Processor
        def __call__(self, images, words, boxes, **kw):
            n = len(words[0])
            return {"input_ids": torch.ones(1, n, dtype=torch.long), "bbox": torch.tensor([boxes[0]]),
                    "attention_mask": torch.ones(1, n, dtype=torch.long),
                    "pixel_values": torch.zeros(1, 3, *images[0].size[::-1])}

    ds = IDLPages([str(path)], Proc(), max_pages_per_doc=2)
    pages = list(ds.raw_pages())
    assert [p["key"] for p in pages] == ["doc1/0", "doc1/1"]           # 2 pages max, doc2 too short
    assert pages[0]["image"].mode == "RGB" and len(pages[0]["tokens"]) == 10
    assert pages[0]["bboxes"][0] == [100, 100, 600, 120]
    items = list(ds)
    assert len(items) == 2 and items[0]["bbox"].shape == (10, 4)
    block = NgramBlocklist([page["text"][0].split()], n=5)
    assert block(pages[0]["tokens"]) and not block(["completely", "different", "words", "on", "this", "page"])
    assert list(IDLPages([str(path)], Proc(), exclude=block).raw_pages()) == []


@pytest.mark.parametrize("hidden,layers,heads,coord,shape", [(32, 2, 2, 6, 4), (48, 3, 4, 8, 8)])
def test_from_backbone_sizes_heads_from_config(tmp_path, hidden, layers, heads, coord, shape):
    """Heads follow the backbone's hidden size / depth (base: 768 x 12, large: 1024 x 24)."""
    from transformers import LayoutLMv3Model
    cfg = tiny_config()
    cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.intermediate_size = \
        hidden, layers, heads, 2 * hidden
    cfg.coordinate_size, cfg.shape_size = coord, shape          # 4 * coord + 2 * shape == hidden
    LayoutLMv3Model(cfg).save_pretrained(tmp_path)
    model = LayoutLMv3ForPreTraining.from_backbone(str(tmp_path), image_vocab_size=8192)
    assert model.mlm_head.dense.weight.shape == (hidden, hidden)
    assert model.mlm_head.decoder.weight is model.layoutlmv3.embeddings.word_embeddings.weight
    assert model.mim_head[1].weight.shape == (8192, hidden) and model.wpa_head.weight.shape == (2, hidden)
    assert len(model.layoutlmv3.encoder.layer) == layers
    out = model(**tiny_batch(k=8192))
    assert out.mim_logits.shape == (2, 196, 8192) and torch.isfinite(out.loss)


# ------------------------------------------------------------------ head warm-up

def test_new_parameters_are_heads_and_image_mask_token_only():
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    names = set(model.new_parameters())
    assert "layoutlmv3.image_mask_token" in names
    assert all(n.startswith(("mlm_head.", "mim_head.", "wpa_head.")) for n in names - {"layoutlmv3.image_mask_token"})
    assert "mlm_head.decoder.bias" in names
    # the tied decoder weight is the word embedding matrix, i.e. an encoder parameter
    assert not any(p is model.layoutlmv3.embeddings.word_embeddings.weight for p in model.new_parameters().values())


def test_head_warmup_freezes_encoder_and_trains_heads():
    """With the encoder frozen, an optimiser step changes the new parameters and nothing else."""
    torch.manual_seed(0)
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)
    model.set_encoder_trainable(False)
    new = model.new_parameters()
    assert all(p.requires_grad == (n in new) for n, p in model.named_parameters())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(new.values(), lr=1e-2, weight_decay=0.01)
    model(**tiny_batch()).loss.backward()
    assert all(p.grad is None for n, p in model.named_parameters() if n not in new)
    opt.step()
    for n, p in model.named_parameters():
        changed = not torch.equal(p, before[n])
        assert changed == (n in new), n
    model.set_encoder_trainable(True)
    assert all(p.requires_grad for p in model.parameters())


def test_layerwise_param_groups():
    model = LayoutLMv3ForPreTraining(tiny_config(), image_vocab_size=8)   # 2 layers -> depths 0..3
    groups = model.layerwise_param_groups(1e-4, layer_decay=0.5, new_lr=1e-3)
    by_name = {n: g["lr"] for g in groups for n in g["names"]}
    assert len(by_name) == sum(1 for _ in model.parameters())             # every parameter exactly once
    assert by_name["layoutlmv3.embeddings.word_embeddings.weight"] == pytest.approx(1e-4 * 0.5 ** 3)
    assert by_name["layoutlmv3.patch_embed.proj.weight"] == pytest.approx(1e-4 * 0.5 ** 3)
    assert by_name["layoutlmv3.encoder.layer.0.attention.self.query.weight"] == pytest.approx(1e-4 * 0.5 ** 2)
    assert by_name["layoutlmv3.encoder.layer.1.output.dense.weight"] == pytest.approx(1e-4 * 0.5)
    assert by_name["layoutlmv3.encoder.rel_pos_bias.weight"] == pytest.approx(1e-4)
    assert by_name["mim_head.1.weight"] == by_name["layoutlmv3.image_mask_token"] == pytest.approx(1e-3)
    no_override = {n: g["lr"] for g in model.layerwise_param_groups(1e-4) for n in g["names"]}
    assert set(no_override.values()) == {1e-4}


def test_from_backbone_random_init_does_not_load_weights(tmp_path):
    from transformers import LayoutLMv3Model
    torch.manual_seed(0)
    LayoutLMv3Model(tiny_config()).save_pretrained(tmp_path)
    loaded = LayoutLMv3ForPreTraining.from_backbone(str(tmp_path), image_vocab_size=8)
    assert loaded.loading_info["encoder_keys_not_in_checkpoint"] == ["image_mask_token"]
    torch.manual_seed(1)
    scratch = LayoutLMv3ForPreTraining.from_backbone(str(tmp_path), image_vocab_size=8, random_init=True)
    w = "encoder.layer.0.attention.self.query.weight"
    assert not torch.equal(loaded.layoutlmv3.state_dict()[w], scratch.layoutlmv3.state_dict()[w])
    assert scratch.mlm_head.decoder.weight is scratch.layoutlmv3.embeddings.word_embeddings.weight
