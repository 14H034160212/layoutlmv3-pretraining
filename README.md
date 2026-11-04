# LayoutLMv3 pre-training objectives (MLM, MIM, WPA) for Hugging Face transformers

[LayoutLMv3](https://arxiv.org/abs/2204.08387) is pre-trained with three objectives: masked language modelling (MLM), masked image modelling (MIM) and word-patch alignment (WPA). The official release includes fine-tuning code but not the pre-training code, and people keep asking for it (see [microsoft/unilm#733](https://github.com/microsoft/unilm/issues/733), [#1042](https://github.com/microsoft/unilm/issues/1042), [#1609](https://github.com/microsoft/unilm/issues/1609), [#772](https://github.com/microsoft/unilm/issues/772) and [#785](https://github.com/microsoft/unilm/issues/785)).

This repository is a small, tested implementation of the three objectives on top of `transformers`' `LayoutLMv3Model`. You can use it for continued pre-training of `microsoft/layoutlmv3-base` on your own documents, or as a reference for how the objectives fit together.

It is an independent re-implementation from the paper, not Microsoft's original code.

## What is implemented

| Objective | Paper | Here |
|---|---|---|
| **MLM** | Mask 30% of text tokens with span masking, span lengths ~ Poisson(λ=3) | `masking.span_mask`, plus an `LMHead` tied to the word embeddings |
| **MIM** | Mask 40% of image patches with blockwise masking; predict discrete visual tokens from an image tokenizer | `masking.blockwise_mask`; masked patch embeddings are replaced by a learned `[MASK]` embedding before position embeddings are added; a linear head predicts the tokens |
| **WPA** | For each *unmasked* text token, predict whether its image patches are also unmasked (aligned) or not (unaligned) | `masking.wpa_labels`, which maps each token's 0-1000 box to the 14×14 patch grid, plus a binary head |

The total loss is `MLM + MIM + WPA`, and all three terms back-propagate into the shared encoder. `tests/test_pretraining.py` checks this explicitly, because it is an easy thing to break: for example, adding a loss with `.item()` silently removes it from the gradient.

The output `backbone/` is a plain `LayoutLMv3Model` checkpoint, so `LayoutLMv3ForTokenClassification.from_pretrained(...)` and the other task heads load it directly.

### One deliberate difference: the MIM targets

The paper's MIM targets come from a dVAE image tokenizer trained on documents (from DiT), whose weights are not part of the LayoutLMv3 release. The default here is `KMeansPatchTokenizer`: a k-means codebook (512 entries) over down-sampled grey-scale 16×16 patches, fitted on your own training images. It needs no external weights and gives MIM a real discrete target, but it is much cruder than a dVAE. Any object with `vocab_size` and `encode(pixel_values) -> LongTensor[B, 196]` can be plugged in instead.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests                    # 8 tests, CPU, a few seconds

# continued pre-training on FUNSD train (words and boxes from the dataset, no OCR)
python scripts/pretrain.py --output_dir runs/funsd_pt --max_steps 3000

# downstream check: FUNSD entity labelling, 3 seeds
python scripts/finetune_funsd.py --model microsoft/layoutlmv3-base --seeds 1 2 3
python scripts/finetune_funsd.py --model runs/funsd_pt/backbone --seeds 1 2 3
```

`scripts/run_experiment.sh` runs all three steps.

## Results of the included experiment

This is a sanity run, not a recipe for better models. Continued pre-training used the **FUNSD train split only** (129 documents for training and 20 held out to monitor the objectives), for 3,000 steps at batch size 8, on one A100 (13 minutes).

**The objectives learn** (held-out documents, fresh masks):

| | Start | Best | Majority-class baseline |
|---|---:|---:|---:|
| MLM accuracy | 0.00 | 0.37 | n/a |
| MIM accuracy (512-token codebook) | 0.00 | 0.37 | 0.20 |
| WPA accuracy | 0.43 | 0.996 | 0.655 |

Validation losses bottom out around steps 750 (MIM) and 1,250 (MLM) and then rise while training loss keeps falling. With 129 documents, the model overfits quickly.

**Downstream FUNSD entity F1** (1,000 fine-tuning steps, lr 1e-5, batch size 2, mean ± std over 3 seeds):

| Initialisation | F1 |
|---|---:|
| `microsoft/layoutlmv3-base` | 0.894 ± 0.009 |
| + continued pre-training on FUNSD train (this repo) | 0.891 ± 0.002 |

So there is **no gain**, which is what you should expect from 129 documents. The base model's score matches the commonly reported ~0.90 for this fine-tuning recipe, which is a useful check that the evaluation is set up correctly. Continued pre-training is only worth trying with a much larger in-domain corpus, and even then, measure it against the base model with several seeds before trusting a difference.

## Licence

- Code: MIT (see `LICENSE`).
- `microsoft/layoutlmv3-base` weights are licensed **CC BY-NC-SA 4.0** (non-commercial). Checkpoints produced by continuing to pre-train them inherit that licence. Check it before any commercial use.
- FUNSD is distributed for non-commercial research use.

## Citation

```bibtex
@inproceedings{huang2022layoutlmv3,
  title     = {LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking},
  author    = {Huang, Yupan and Lv, Tengchao and Cui, Lei and Lu, Yutong and Wei, Furu},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
  year      = {2022}
}
```
