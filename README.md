# LayoutLMv3 pre-training objectives (MLM, MIM, WPA) for Hugging Face transformers

[LayoutLMv3](https://arxiv.org/abs/2204.08387) is pre-trained with three objectives: masked language modelling (MLM), masked image modelling (MIM) and word-patch alignment (WPA). The official release includes fine-tuning code but not the pre-training code, and people keep asking for it (see [microsoft/unilm#733](https://github.com/microsoft/unilm/issues/733), [#1042](https://github.com/microsoft/unilm/issues/1042), [#1609](https://github.com/microsoft/unilm/issues/1609), [#772](https://github.com/microsoft/unilm/issues/772) and [#785](https://github.com/microsoft/unilm/issues/785)).

This repository is a small, tested implementation of the three objectives on top of `transformers`' `LayoutLMv3Model`. You can use it for continued pre-training of `microsoft/layoutlmv3-base` or `-large` on your own documents, or as a reference for how the objectives fit together.

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

The paper's MIM targets come from a dVAE image tokenizer trained on documents (from DiT), whose weights are not part of the LayoutLMv3 release. This repository offers two substitutes instead:

- **`DalleTokenizer`** (`--image_tokenizer dalle`): OpenAI's public DALL-E dVAE encoder ([openai/DALL-E](https://github.com/openai/DALL-E), modified MIT licence; weights from `https://cdn.openai.com/dall-e/encoder.pkl`), 8,192 visual tokens. This is the tokenizer BEiT v1 used. As in BEiT, it sees a 112×112 view of the 224×224 image, and its 8× down-sampling gives a 14×14 token grid, one token per LayoutLMv3 16×16 patch. The LayoutLMv3 processor's normalisation (mean = std = 0.5) is undone and dall_e's `map_pixels` is applied before encoding. `dvae.py` re-implements the encoder with the same parameter names and loads `encoder.pkl` through a restricted unpickler, so the `dall_e` package is not needed. A check against the original `dall_e` package gave identical logits. **This dVAE was trained on natural images, not documents**, so it is still not the paper's tokenizer.
- **`KMeansPatchTokenizer`** (`--image_tokenizer kmeans`, the default): a k-means codebook (512 entries) over down-sampled grey-scale 16×16 patches, fitted on your own training images. It needs no external weights but is much cruder than a dVAE.

Any object with `vocab_size` and `encode(pixel_values) -> LongTensor[B, 196]` can be plugged in instead.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests                    # 15 tests, CPU, a few seconds (one is skipped without the DALL-E weights)

# continued pre-training on FUNSD train (words and boxes from the dataset, no OCR), k-means MIM targets
python scripts/pretrain.py --output_dir runs/funsd_pt --max_steps 3000

# continued pre-training on ~42k Industry Documents Library pages, DALL-E dVAE MIM targets
python scripts/pretrain.py --corpus idl --idl_shards 0-26 --exclude_funsd_test --image_tokenizer dalle \
    --output_dir runs/idl_dalle_base --max_steps 6000 --batch_size 16 --grad_accum 2 --num_workers 12 \
    --eval_every 500 --patience 4 --val_docs 512

# downstream checks: FUNSD and CORD entity labelling, 5 seeds
python scripts/finetune_funsd.py --task funsd --model microsoft/layoutlmv3-base --seeds 1 2 3 4 5
python scripts/finetune_funsd.py --task cord  --model runs/idl_dalle_base/backbone --seeds 1 2 3 4 5
```

`scripts/run_experiment.sh` runs the FUNSD-only experiment and `scripts/run_idl_experiment.sh` the IDL one (`SIZE=base` or `SIZE=large`).

### Training options

| Flag | What it does |
|---|---|
| `--corpus hf` / `--corpus idl` | a small Hugging Face dataset with words and 0-1000 boxes (default FUNSD train), or IDL pages streamed shard by shard |
| `--idl_shards 0-26`, `--pages_per_doc 2`, `--max_pages N` | which `pixparse/idl-wds` shards make up the corpus, and how many pages to take |
| `--idl_stream` | read the shards over HTTP (`HfFileSystem`) instead of caching them locally first |
| `--exclude_funsd_test` | drop IDL pages whose word 5-grams mostly (≥50%) occur in the FUNSD test documents |
| `--image_tokenizer dalle\|kmeans`, `--dalle_encoder PATH` | MIM targets (the DALL-E encoder is downloaded if no path is given) |
| `--gradient_checkpointing` | trade compute for memory |
| `--grad_accum N` | effective batch size = `batch_size × N` |
| `--num_workers N` | DataLoader worker processes (IDL decoding and masking run there; MIM targets are computed on the GPU) |
| `--eval_every`, `--patience` | validation on fixed held-out pages with fixed masks; `backbone/` always holds the step with the lowest validation loss, and training stops after `patience` evaluations without improvement |
| `--save_last` | also keep the final step in `last_backbone/` |

### The IDL corpus

`--corpus idl` reads [`pixparse/idl-wds`](https://huggingface.co/datasets/pixparse/idl-wds), a webdataset version of the [UCSF Industry Documents Library](https://www.industrydocuments.ucsf.edu/) (about 19M pages with Textract OCR). Each sample is a document: a multi-page TIFF plus OCR JSON with text **lines** and their boxes (`[left, top, width, height]`, relative to the page). `corpus.py` splits each line into words and gives each word its line's box, scaled to 0-1000. This is a segment-level layout, which is what LayoutLMv3 uses in pre-training (and what `nielsr/funsd-layoutlmv3` provides). Lines with OCR confidence < 0.5 and pages with < 10 words are skipped, and at most 512 words are kept per page. Shards are split across DataLoader workers and a shuffle buffer randomises the order, so the corpus is never materialised.

FUNSD itself comes from RVL-CDIP, which was drawn from the same tobacco-industry archive, so IDL is in-domain for FUNSD and could contain FUNSD test pages. `--exclude_funsd_test` removes near-duplicates by word 5-gram overlap. In the run below it removed 1 page. This check only catches pages with similar OCR text, and `microsoft/layoutlmv3-*` was itself pre-trained on IIT-CDIP, which is from the same archive.

**Terms.** The dataset card gives no open licence (`license: other`). It says the documents remain under their rights holders' copyright and that IDL makes them available under court-approved agreements or under US fair use, which, as the card notes, may permit use "for purposes such as ... scholarship or research". Each user is responsible for staying within fair use. Here the pages are used only to compute research model updates; no page images or text are redistributed in this repository. Check the [dataset card](https://huggingface.co/datasets/pixparse/idl-wds) and your own situation before using it, especially for anything commercial. (`pixparse/pdfa-eng-wds` was the alternative; its card points to the Common Crawl and Digital Corpora terms, but its PDF-extracted text is less like FUNSD's scanned forms.)

## Results

These are small-scale experiments, not a reproduction of the paper: the corpus is about 42k pages rather than 11M, the MIM targets come from a natural-image dVAE rather than the paper's document dVAE, and the training budget is a few GPU hours.

### Continued pre-training on 42k IDL pages with dVAE targets

**Setup.** IDL shards 0-26 (`pixparse/idl-wds`), at most 2 pages per document, giving **42,061 training pages** (1 page removed by the FUNSD-test filter). 512 pages from shard 2999 are held out, with fixed masks, to pick the checkpoint. MIM targets: DALL-E dVAE, 8,192 tokens. bf16 autocast, AdamW (weight decay 0.01), cosine schedule with 5% warm-up, gradient clipping 1.0, 12 DataLoader workers, one A100 80GB per model.

| | base | large |
|---|---|---|
| start | `microsoft/layoutlmv3-base` (768 × 12 layers) | `microsoft/layoutlmv3-large` (1024 × 24 layers) |
| optimiser steps | 6,000 | 4,000 |
| batch | 16 × 2 accumulation = 32 pages | 16 × 2 accumulation = 32 pages |
| pages seen | 192k (≈4.6 passes) | 128k (≈3.0 passes) |
| peak learning rate | 5e-5 | 3e-5 |
| gradient checkpointing | off (fits) | off (58 GB peak); in a 60-step test, turning it on gave identical losses and was ~30% slower |
| training time | 90 min | 184 min |
| best validation step | 6,000 (last) | 4,000 (last) |

Training times are wall-clock on GPUs that were partly shared with the fine-tuning runs. The whole experiment (both pre-training runs, 40 fine-tuning runs and short smoke tests) took about 6.3 GPU-hours. The large model got fewer steps to keep the total GPU time reasonable. For both models, validation loss was still going down slowly at the last step, so with 42k pages neither run reached the over-fitting seen with 129 FUNSD documents.

**The objectives learn** (512 held-out IDL pages, fixed masks):

| | base: start → end | large: start → end | Majority-class baseline |
|---|---:|---:|---:|
| MLM accuracy | 0.00 → 0.515 | 0.00 → 0.559 | n/a |
| MIM accuracy (8,192 dVAE tokens) | 0.00 → 0.211 | 0.00 → 0.199 | 0.027 |
| WPA accuracy | 0.34 → 0.999 | 0.54 → 0.997 | 0.742 |

WPA is close to trivial in this implementation: whether a token's patches are masked can be read off the learned `[MASK]` patch embeddings, so do not read much into its accuracy.

**Downstream entity F1** (mean ± sample std over seeds 1-5; the same seeds and data order for every initialisation). FUNSD: 1,000 steps, lr 1e-5, batch size 2. CORD v2: 1,000 steps, batch size 4, lr 5e-5 (base) or 2e-5 (large). The CORD recipe is short; it was not tuned to reach the paper's numbers.

| Initialisation | FUNSD F1 | CORD F1 |
|---|---:|---:|
| `microsoft/layoutlmv3-base` | 0.895 ± 0.007 | 0.936 ± 0.004 |
| + continued pre-training on IDL (this repo) | 0.900 ± 0.007 | 0.937 ± 0.005 |
| `microsoft/layoutlmv3-large` | 0.903 ± 0.006 | 0.956 ± 0.005 |
| + continued pre-training on IDL (this repo) | 0.905 ± 0.005 | 0.952 ± 0.003 |

Per-seed differences (continued minus original), paired by seed: base FUNSD +1.45, +1.42, −0.59, +0.59, −0.31 F1 points (mean +0.51, paired t-test p = 0.29), base CORD +0.34, −0.94, +0.82, +0.68, −0.21 (mean +0.14, p = 0.69); large FUNSD +0.13, +0.90, −0.04, −0.51, +0.74 (mean +0.24, p = 0.40), large CORD +0.31, −0.49, −1.51, +0.01, −0.67 (mean −0.47, p = 0.21). None of the four differences can be told apart from seed noise with 5 seeds, and the large model on CORD is, if anything, slightly worse. So continued pre-training on 42k IDL pages with dVAE targets gave **no reliable gain** over the released checkpoints at this scale. This is not surprising: the released models were already pre-trained on 11M pages from the same kind of documents, and this run adds under 0.5% as many pages for a few thousand steps.

Per-seed scores and the pre-training logs are in `results/idl/`.

### Earlier sanity run: FUNSD train only (k-means targets)

This run predates the IDL corpus, the dVAE tokenizer and best-checkpoint selection (it used the last step, from commit `6aa8569`). Continued pre-training used the **FUNSD train split only** (129 documents for training and 20 held out to monitor the objectives), for 3,000 steps at batch size 8, on one A100 (13 minutes).

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

So there is **no gain**, which is what you should expect from 129 documents. The base model's score matches the commonly reported ~0.90 for this fine-tuning recipe, which is a useful check that the evaluation is set up correctly. (Seeds 1-3 of the 5-seed base run above reproduce these three scores exactly.)

## Licence

- Code: MIT (see `LICENSE`).
- `microsoft/layoutlmv3-base` and `microsoft/layoutlmv3-large` weights are both licensed **CC BY-NC-SA 4.0** (non-commercial). Checkpoints produced by continuing to pre-train them inherit that licence. Check it before any commercial use.
- DALL-E dVAE: OpenAI's [openai/DALL-E](https://github.com/openai/DALL-E) package is under a modified MIT licence (its notice is reproduced in `dvae.py`, which adapts the encoder architecture); the encoder weights are downloaded from OpenAI at run time, not redistributed. OpenAI asks users of the dVAE to indicate that it was used: the MIM targets here come from it.
- IDL (`pixparse/idl-wds`): no open licence; documents are made available under court-approved agreements or US fair use for research. See [The IDL corpus](#the-idl-corpus). No IDL content is redistributed here.
- FUNSD is distributed for non-commercial research use. CORD v2 (`naver-clova-ix/cord-v2`) is CC BY 4.0.

## Development note

Written with the help of an AI coding assistant. The design, tests and experiments were reviewed and run by the author, and all numbers above come from the included scripts and `results/` files.

## Citation

```bibtex
@inproceedings{huang2022layoutlmv3,
  title     = {LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking},
  author    = {Huang, Yupan and Lv, Tengchao and Cui, Lei and Lu, Yutong and Wei, Furu},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
  year      = {2022}
}
```
