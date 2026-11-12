"""--long_docs window in scripts/finetune_funsd.py: overlapping windows cover every word, and merging predictions
from the windows scores every word of a long document."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from datasets import Image as ImageFeature
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import finetune_funsd as ft  # noqa: E402

LABELS = ["O", "B-X", "I-X"]


def test_window_first_subwords_skips_a_word_the_window_starts_inside():
    # word 7 has 3 sub-tokens in the document, but only its last 2 are in this window
    word_ids = [None, 7, 7, 8, 9, 9, None, None]
    counts = {7: 3, 8: 1, 9: 2}
    assert ft.window_first_subwords(word_ids, counts) == {3, 4}
    # the same window when word 7 starts here
    assert ft.window_first_subwords(word_ids, {7: 2, 8: 1, 9: 2}) == {1, 3, 4}


@pytest.fixture(scope="module")
def processor():
    try:
        return ft.LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"LayoutLMv3 processor not available offline: {e}")


def long_document(n_words=700, seed=0):
    rng = np.random.default_rng(seed)
    # some words split into several sub-tokens, so windows can start inside a word
    tokens = [("multi" + "x" * int(rng.integers(0, 12))) if i % 5 == 0 else f"w{i}" for i in range(n_words)]
    boxes = [[(i % 20) * 50, (i // 20) * 25 % 1000, (i % 20) * 50 + 40, (i // 20) * 25 % 1000 + 20]
             for i in range(n_words)]
    tags = [int(rng.integers(0, 3)) for _ in range(n_words)]
    image = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    feats = Features({"tokens": Sequence(Value("string")), "bboxes": Sequence(Sequence(Value("int64"))),
                      "ner_tags": Sequence(ClassLabel(names=LABELS)), "image": ImageFeature()})
    return Dataset.from_dict({"tokens": [tokens], "bboxes": [boxes], "ner_tags": [tags], "image": [image]},
                             features=feats)


class OracleModel(torch.nn.Module):
    """Predicts each labelled token's gold label; unlabelled tokens get class 0."""

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels):
        logits = torch.zeros(*labels.shape, len(LABELS))
        logits.scatter_(-1, labels.clamp(min=0).unsqueeze(-1), 1.0)
        return SimpleNamespace(loss=torch.tensor(0.0), logits=logits)


def test_windows_cover_every_word_and_oracle_scores_one(processor):
    ds = long_document()
    n_tokens = len(processor.tokenizer(ds[0]["tokens"], boxes=ds[0]["bboxes"], truncation=False)["input_ids"])
    assert n_tokens > 512 * 1.3, "the document must need more than one window"

    windows = ds.map(lambda e, i: ft.add_window_index(processor, e, i, True, 128), batched=True, with_indices=True,
                     remove_columns=ds.column_names)
    assert len(windows) >= 2
    windows.set_format("torch")
    gold = ds["ner_tags"]

    # every word's first sub-token is labelled in at least one window, and only first sub-tokens are labelled
    labelled = {(int(d), int(w)) for d, wids, labs in zip(windows["doc"], windows["word_ids"], windows["labels"])
                for w, lab in zip(wids.tolist(), labs.tolist()) if lab != -100}
    assert labelled == {(0, w) for w in range(len(gold[0]))}

    f1, _ = ft.evaluate_windows(OracleModel(), windows, gold, LABELS, "cpu")
    assert f1 == 1.0

    # with truncation, the same oracle misses the words beyond 512 tokens
    truncated = ds.map(lambda e: ft.encode(processor, e, True), batched=True, remove_columns=ds.column_names)
    truncated.set_format("torch")
    tails = ft.truncated_tags(processor, ds, LABELS)
    assert len(tails[0]) > 0
    f1_truncated, _ = ft.evaluate(OracleModel(), truncated, LABELS, "cpu", tails)
    assert f1_truncated < 1.0
