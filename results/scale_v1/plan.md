# Scale-up v1: continued pre-training on ~400k IDL pages (pre-registered plan)

Written and committed **before** any scale-up pre-training run started. Only the corpus count
(`corpus_count.json`) and a short smoke test of the code path were run before this commit: 400 steps on the old
42k corpus with the MLP head, killed at step 300 and resumed from the step-200 state (tag `smoke`, not a result;
about 0.1 GPU-hours). Registry tag of every scale-up run: `scale-v1`.

## Question
The 42k-page runs (`results/idl`, `results/idl_warmup`) gave no downstream change against the released
checkpoints, and the most likely reason was data scale: 42k pages and a few thousand steps on top of about 11M
pages of pre-training. Does roughly ten times more continued pre-training data move FUNSD or CORD F1?

## Corpus
| | 42k runs (before) | scale-v1 |
|---|---|---|
| training shards (`pixparse/idl-wds`) | 0-26 (27 shards) | **0-259** (260 shards; contains 0-26) |
| pages per document (max) | 2 | **2** |
| FUNSD-test 5-gram filter (≥ 50% overlap) | on | **on (unchanged)** |
| validation | 512 pages of shard 2999, fixed masks | **unchanged** (same pages, same masks) |
| documents | 28,332 | 272,665 |
| training pages | 42,061 | **N = 404,507** (`corpus_count.json`: counted from the OCR json with the same filters; the same count gives 42,061 for shards 0-26) |
| removed by the FUNSD-test filter | 1 | 19 |

- **Pages per document = 2, kept.** (a) It keeps the page distribution identical to the 42k runs (mostly first
  pages: covers, letters, forms), so the corpus grows in size only. (b) More documents per page give more diverse
  data: pages of one document share letterhead, layout and vocabulary. (c) The cost is disk, not compute:
  measured on shards 0 and 13, a shard holds ~1,040 documents and gives ~1,530 pages at 2 pages per document,
  2,090 at 4 and 2,750 at 8. ~400k pages therefore need 260 shards (223 GB) at 2 pages per document, against
  145 shards (125 GB) at 8. 223 GB in total (200 GB new) is within the 500 GB limit.
- **Size.** 260 shards is the largest round number near 400k pages that fits the budget below with
  margin. The shards are read on the fly (TIFF decoding and masking in 12 DataLoader workers), so there is no
  preprocessing step. Download: 200 GB new in about 50 minutes (`scripts/download_idl_shards.py`; it stalled twice and was restarted, which resumes partial files).

## Schedule
Everything except the corpus, the number of joint steps and the WPA head is the head warm-up recipe of
`results/idl_warmup` for that size.

| | base | large |
|---|---|---|
| start | `microsoft/layoutlmv3-base` | `microsoft/layoutlmv3-large` |
| head warm-up (encoder frozen) | 1,000 steps, head lr 1e-3 | same |
| joint steps | **J = ceil(2 N / 32) = 25,282 (2 epochs)** | same J |
| peak lr (top layer) / layer decay | 2e-5 / 0.9 | 1e-5 / 0.95 |
| schedule | cosine, 5% warm-up | same |
| WPA head | **`mlp`** (new default) | **`mlp`** |
| MIM targets | DALL-E dVAE (`runs/dalle/encoder.pkl`) | same |
| batch | 16 × 2 accumulation = 32 pages | same |
| evaluation | every 500 steps, patience 8 (stops after 4,000 steps without a lower validation loss) | same |
| checkpoint fine-tuned | `backbone/` = lowest validation loss among joint steps (as before) | same |
| seed | 42 | 42 |

- **Two epochs.** The 42k base run made 4.6 passes over its corpus and its validation loss was still going
  down, so a few repeats do no harm at this scale (data-constrained language-model scaling, Muennighoff et al.
  2023, finds up to about 4 epochs nearly as useful as fresh data). Two passes give 809k joint page views,
  4.2× the 42k run's 192k, with 9.6× as many distinct pages. One pass would test fewer optimiser steps (2×),
  and 3+ passes would not leave room for large in the budget. **Data and steps both grow**, so a gain would be
  attributed to "more continued pre-training", not to data diversity alone; a compute-matched control is not
  in the budget.
- **Resumable.** `--state_every 2000 --resume`; a killed run continues from the last saved state (with a new
  data seed). Any resume is reported as a deviation.

## Comparison
Three initialisations per size, each fine-tuned with seeds 1-5 and **the same recipes as `results/idl_warmup`**:
FUNSD 1,000 steps, lr 1e-5, batch 2; CORD v2 1,000 steps, batch 4, lr 5e-5 (base) or 2e-5 (large).

1. **original**: `microsoft/layoutlmv3-{base,large}`, existing results `results/idl/{funsd,cord}_{size}_original.json`
2. **42k warm-up**: `results/idl_warmup/{funsd,cord}_{size}_warmup_continued.json` (existing; linear WPA head)
3. **scaled**: the new checkpoint, `results/scale_v1/{funsd,cord}_{size}_scaled.json`

Existing results are reused, not re-run: fine-tuning is seeded and deterministic (a re-run reproduced seed 1
of the original exactly), and `scripts/finetune_funsd.py` has not changed since they were made.

## Tests and decision rule
- Per task: paired t-test over seeds 1-5 (scaled vs original; scaled vs 42k warm-up), per-seed differences
  in F1 points, as in the README.
- Noise: the pooled per-seed SD of paired differences between checkpoints in `results/idl*` is 0.75 points,
  so the SE of the mean over 10 (task, seed) pairs is about 0.24 points.
- **Primary: "scaled gives a gain over the original" (per size) if all three hold:**
  1. the mean of the 10 paired differences (FUNSD and CORD, seeds 1-5) is **≥ +0.5 F1 points** (about 2 SE);
  2. the mean difference is positive on **both** FUNSD and CORD;
  3. a paired t-test on the 10 differences gives **p < 0.05** (two-sided).
- Otherwise the result is "no gain at this resolution" (differences under about 0.5 points cannot be told apart
  from seed noise). A mean of ≤ −0.5 points that meets conditions 2 and 3 with the sign reversed is reported as
  a loss.
- **Secondary: "scale helps over the 42k recipe"**: the same three conditions for scaled vs the 42k warm-up
  checkpoint. Caveat: this changes data, steps and the WPA head together (the screen found linear and MLP
  heads equivalent: −0.26 points, within noise).
- Reported but not part of the rule: validation metrics (MLM / MIM accuracy, WPA AUC) after the warm-up and at
  the selected step, compared with the 42k runs on the same 512 pages; encoder drift; GPU-hours.

## Base first, then large
1. Base: pre-training and fine-tuning on GPU 3 (fine-tuning may use GPU 4, which has ~48 GB free).
   The base result is computed and written to `results/scale_v1/` before large starts.
2. **Large runs if base shows a gain under the primary rule, OR (the budget clause) if, after base is done,
   GPU-hours used so far + 1.15 × the large estimate ≤ 25.** The budget clause is expected to apply
   (≈ 6 h used by then + ≈ 12 h for large), so **large is planned to run whatever the base result**, on GPU 3
   only (it needs ~58 GB). If the clause fails (e.g. base was much slower than estimated), large is not run and
   this is reported.

## Budget (GPU-hours, from measured throughput)
| | estimate |
|---|---:|
| base pre-training: 26,282 steps at 93 steps/min | 4.7 |
| base fine-tuning: FUNSD + CORD, 5 seeds each | 0.9 |
| large pre-training: 26,282 steps at 40.5 steps/min | 10.8 |
| large fine-tuning | 1.2 |
| smoke tests | 0.3 |
| **total** | **17.9** of 25 |

If any stage is projected to push the total over 25 GPU-hours, it is stopped and reported instead.

## Commands
```bash
python scripts/download_idl_shards.py 0-259
python scripts/count_idl_pages.py 0-259 --exclude_funsd_test --out results/scale_v1/corpus_count.json
JOINT_STEPS=25282 GPU=3 SIZE=base  setsid nohup ./scripts/run_scale_experiment.sh > runs/scale_v1/base.out 2>&1 < /dev/null &
python scripts/analyze_scale.py base --out results/scale_v1/analysis_base.json      # the rule, mechanically
JOINT_STEPS=25282 GPU=3 SIZE=large setsid nohup ./scripts/run_scale_experiment.sh > runs/scale_v1/large.out 2>&1 < /dev/null &
python scripts/analyze_scale.py base large --out results/scale_v1/analysis.json
```

---

## Execution log (added after the plan was committed; not part of the pre-registration)

**2026-11-05: base stopped at joint step 1,000 of 25,282 (step 2,000 overall); budget blocker.**
- Base started at 08:15 UTC on GPU 3. Since about 07:56 another job of the same user (four `bright_train_source.py`
  processes per GPU, rotating through events) has shared GPUs 3 and 4 at 100% utilisation and ~47 GB on GPU 3.
- Measured throughput under that load: head warm-up 25 steps/min, joint training **17 steps/min**, against the
  93 steps/min this plan's budget assumed. At that rate base pre-training alone would need about 25 more
  GPU-hours, and GPU 3 had 74.6 of 80 GB in use, so large (~58 GB) could not start either. As the plan
  says, the run was stopped and reported instead of continuing over budget.
- The run was stopped right after the training state at step 2,000 was saved
  (`runs/scale_v1/base/train_state.pt`), so it can be resumed with the same command
  (`JOINT_STEPS=25282 GPU=3 SIZE=base ./scripts/run_scale_experiment.sh`). On resume the data stream restarts
  from the beginning of the shards with a new shuffle seed; over the 2 epochs, the first ~64k pages
  would then be seen three times and the last ~63k only once. That would be a deviation, to be reported.
- Validation at the stop (`pretrain_base_partial_log.json`; 512 pages of shard 2999, fixed masks):

  | step | phase | loss | MLM acc | MIM acc | WPA AUC |
  |---:|---|---:|---:|---:|---:|
  | 0 | random heads | 25.44 | 0.000 | 0.000 | 0.507 |
  | 1,000 | end of head warm-up (encoder unchanged, max abs change 0.0) | 8.47 | 0.511 | 0.121 | 0.904 |
  | 2,000 | joint step 1,000 | 7.47 | 0.522 | 0.163 | 0.990 |

  The end of the warm-up reproduces the frozen-encoder probes: MLM 0.511, and WPA AUC 0.904, the same as the
  two-layer MLP probe in `results/probes`. These are not results of the scale-up.
- GPU time used: smoke test 0.1 h + base 1.6 h = **1.7 GPU-hours**.
