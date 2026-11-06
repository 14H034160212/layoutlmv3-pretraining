# Scale-up v2: the scale-v1 experiment re-run with the fixed data pipeline (pre-registered plan)

Written and committed **before** any scale-v2 run started. Run before this commit: the corpus count
(`corpus_count.json`) and a sanity run (base, 1,000 head-only + 1,000 joint steps, tag `sanity`, not a result;
numbers below). Registry tags of every run: `scale-v2` and the size.

## Why a v2
An independent code review of the scale-v1 code found three data-pipeline bugs (all confirmed, fixed in
`75ae0c7`, each with a regression test):

1. **About 10% of IDL pages were trained with the image turned 90° against its OCR boxes.** The TIFF frame is
   stored turned relative to the frame of the Textract boxes, so horizontal boxes lay across vertical text
   (and ~1% of frames are turned 180°). Fix: `corpus.page_orientation` compares the page's edge energy inside the
   boxes for the four quarter turns; 90/270° pages are turned upright, 180° and ambiguous pages are dropped
   (`--idl_orientation fix`, the new default).
2. **All 12 DataLoader workers drew identical masks.** The collator's random generators were seeded once in
   `__init__` and copied into every worker, so each mask pattern was used 12 times in a row. Fix: the generators
   are seeded lazily per process, with (seed, worker id) in the workers. Validation masks are unchanged (drawn
   once in the main process with seed 0).
3. **`--resume` replayed the start of the corpus.** A resumed run restarted every worker at its first shard with a
   new shuffle seed (scale-v1 base was resumed twice, so ~64k pages were seen 2-3 times). Fix: the page order is
   a function of the seed only, and on resume the workers skip the batches already used, so the stream continues
   exactly where it stopped.

The optimiser is also aligned with the paper (`82bacb2`): AdamW betas (0.9, 0.98) instead of (0.9, 0.999), and
no weight decay on biases, LayerNorm weights, `cls_token`, `pos_embed`, `image_mask_token`, the text/1D/2D
position embeddings and the relative position biases (BEiT-style). Layer-wise lr decay is unchanged.

**Question.** With the pipeline fixed, does ~400k pages of continued pre-training move FUNSD or CORD F1, and do
the fixes change the scale-v1 result ("no gain at this resolution" for base)?

## Corpus (fixed pipeline)
| | scale-v1 | scale-v2 |
|---|---|---|
| training shards | 0-259 | **0-259** (unchanged) |
| pages per document / FUNSD-test filter | 2 / on | 2 / on |
| page orientation | as stored | **fixed**: 37,272 pages turned upright (28,707 by 270°, 8,565 by 90°); 4,177 dropped (2,441 at 180°, 1,736 ambiguous) |
| training pages | 404,507 (OCR json count, upper bound) | **N = 402,284** (`corpus_count.json`, counted with images decoded, exact) |
| validation | first 512 pages of shard 2999 | first 512 pages of shard 2999 **after the fix**: 508 pages shared with v1, 4 replaced (the 180° pages dropped), **57 turned upright**; masks drawn as before (seed 0, main process) |

Dropping a page lets a later page of the same document in (at most 2 per document), so N falls by less than
the number dropped. Validation numbers are therefore **not directly comparable** with scale-v1's; the scale-v1
base checkpoint is re-evaluated on the new validation pages for a like-for-like comparison (reported, not in
the rule).

## Schedule
Identical to scale-v1 except for the joint step count (recomputed for the new N) and the optimiser fixes above.

| | base | large |
|---|---|---|
| start | `microsoft/layoutlmv3-base` | `microsoft/layoutlmv3-large` |
| head warm-up (encoder frozen) | 1,000 steps, head lr 1e-3 | same |
| joint steps | **J = ceil(2 N / 32) = ceil(25,142.75) = 25,143 (2 epochs)**; scale-v1 used 25,282 | same J |
| peak lr (top layer) / layer decay | 2e-5 / 0.9 | 1e-5 / 0.95 |
| schedule | cosine, 5% warm-up | same |
| WPA head / MIM targets | MLP / DALL-E dVAE | same |
| batch | 16 × 2 = 32 pages | 16 × 2 if GPU 3 is otherwise idle; else 8 × 4 with gradient checkpointing (effective 32) |
| evaluation | every 500 steps, patience 8, `backbone/` = lowest validation loss among joint steps | same |
| seed | 42 | 42 |

Resumable (`--state_every 2000 --resume`, now exact) and launched detached with retries
(`scripts/retry_scale.sh`). Any resume is logged; with the fix it is no longer a deviation in the data seen.

## Comparison
Each new checkpoint is fine-tuned with seeds 1-5 and **the same recipes as scale-v1**: FUNSD 1,000 steps, lr 1e-5,
batch 2; CORD v2 1,000 steps, batch 4, lr 5e-5 (base) or 2e-5 (large). Fine-tuning also records learning curves
(`--eval_at 0 25 50 100 150 200 300 400 600 800`; this does not change training or the final F1).

1. **original**: `results/idl/{funsd,cord}_{size}_original.json` (existing)
2. **42k warm-up**: `results/idl_warmup/...` (existing)
3. **scale-v1**: `results/scale_v1/{funsd,cord}_base_scaled.json` (base only; scale-v1 large was stopped before it finished)
4. **scale-v2**: `results/scale_v2/{funsd,cord}_{size}_scaled.json`

Learning curves are compared on seeds 1-3 with `results/curves_v1` (original and scale-v1 base).

## Tests and decision rule (unchanged from scale-v1)
- Per task: paired t-test over seeds 1-5; per-seed differences in F1 points.
- **Primary, per size: "scale-v2 gives a gain over the original" if all three hold:** (1) the mean of the 10 paired
  (task, seed) differences is ≥ +0.5 F1 points; (2) the mean is positive on both FUNSD and CORD; (3) paired
  t-test on the 10 differences p < 0.05 (two-sided). A mean ≤ −0.5 meeting (2) and (3) with the sign reversed is
  a loss; otherwise "no gain at this resolution".
- **Secondary: "the fixes changed the result"**: the same three conditions for scale-v2 vs scale-v1 (base).
  Also reported: scale-v2 vs 42k warm-up.
- Reported, not in the rule: validation metrics, scale-v1 base on the new validation pages, encoder drift,
  learning curves, GPU-hours. `python scripts/analyze_scale.py base large --exp scale_v2`.

## Sanity run (before this commit; not a result)
Base, fixed code, shards 0-259, 1,000 head-only + 1,000 joint steps, GPU 3 otherwise idle
(`sanity_base_log.json`, registry tags `scale-v2 base sanity`). To exercise the resume fix it was killed right
after the step-1,500 state save and relaunched with the same command; it logged "resuming from step 1500:
skipping 3000 batches of the page stream" and finished without errors.

| step | phase | loss | MLM acc | MIM acc | WPA AUC |
|---:|---|---:|---:|---:|---:|
| 0 | random heads | 25.42 | 0.000 | 0.000 | 0.497 |
| 1,000 | end of head warm-up (encoder max abs change 0.0) | 8.38 | **0.511** | 0.123 | 0.908 |
| 1,500 | joint 500 | 7.30 | 0.523 | 0.172 | 0.992 |
| 2,000 | joint 1,000 | 7.20 | 0.524 | 0.178 | 0.995 |

- The warm-up start matches scale-v1 (MLM 0.511 after 1,000 head-only steps on the frozen released encoder).
- Throughput on an idle A100: head warm-up ~100 steps/min, joint ~95 steps/min (including evaluation); the
  resumed segment (skip of 48k pages included) took 6.8 minutes for 500 steps. DataLoader with the orientation
  check: 371 pages/s on 12 CPU workers (392 before), far above the ~50 pages/s training uses.
- scale-v1 base checkpoint re-evaluated: loss 6.3765 on the old validation pages (reproduces its log exactly),
  **6.2116** on the new ones (MLM 0.550, MIM 0.260): this is the like-for-like reference for scale-v2's
  validation loss (`scale_v1_base_on_val_{off,fix}.json`).

## Order and GPUs
Both sizes run whatever the base result is (requested). To save wall-clock time they run in parallel: base
on GPU 4 (next to the permanent gemma4-rl servers, ~32 GB; base needs ~26 GB), large on GPU 3 at BS 16 if it is
idle. Only GPUs 3 and 4 are used; other jobs on them are never touched; if memory is short the launcher waits
or large falls back to 8 × 4 with gradient checkpointing.

## Budget (GPU-hours)
| | estimate |
|---|---:|
| sanity run (done before this commit) and validation re-evaluations | 0.6 |
| base pre-training: 1,000 warm-up steps at ~100/min + 25,143 joint steps at ~91/min (sanity run, idle GPU) | 4.9 |
| base fine-tuning with curves: FUNSD + CORD, 5 seeds | 1.2 |
| large pre-training: 26,143 steps at ~40.5/min (scale-v1 measurement) | 10.8 |
| large fine-tuning with curves | 1.6 |
| **total** | **19.1** of 20 |
| optional, only if the projected total stays ≤ 20: learning curves of `microsoft/layoutlmv3-large` (seeds 1-3) | 0.8 |

Without the optional item there are no large curves for the released checkpoint; the large curves are then
reported for scale-v2 alone.

If the projected total exceeds 20 GPU-hours at any point, the run is stopped and reported.
