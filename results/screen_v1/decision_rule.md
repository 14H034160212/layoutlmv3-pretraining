# Recipe screen v1 (base): plan and decision rule

Written and committed **before** any screen run was started.

## Variants (all: layoutlmv3-base, IDL shards 0-26, DALL-E MIM targets unless noted, seed 42, batch 16 x 2,
## 1000 head-only steps (head lr 1e-3) + 2500 joint steps, cosine with 5% warm-up, eval every 250 steps)

| id | change vs the current recipe | why |
|---|---|---|
| A | none: joint lr 2e-5 (top layer), layer decay 0.9, linear WPA head (control) | current recipe |
| B | `--wpa_head mlp` (two-layer MLP on the text token) | the paper's WPA head is a two-layer MLP; the probe read alignment better with it (AUC 0.904 vs 0.816) |
| C | `--mim_weight 0` (MLM + WPA only; images still masked for WPA) | tests whether the mismatched natural-image DALL-E targets pull visual features away (CORD went down) |
| D | MIM targets from the document-fine-tuned dVAE (`runs/dvae_doc/encoder.pt`) | closer to the paper's document dVAE; preferred over k-means because k-means targets are dominated by blank paper (30% majority code) and mostly encode "ink or not" |
| E | joint lr 1e-5 (layer decay 0.9) | drift control. Chosen over L2-SP: with AdamW the L2-SP pull is rescaled per parameter, so its weight would need tuning, which a one-shot screen cannot do; a lower lr is a clean one-parameter drift control and also gives an lr-sensitivity point |
| F | `--freeze_layers 6` (embeddings + layers 0-5 frozen in joint training) | stronger drift control: only the top half adapts |

## Downstream evaluation
FUNSD and CORD fine-tuning, seeds 1, 2, 3, same settings as `results/idl/*_original.json` (FUNSD 1000 steps,
lr 1e-5, batch 2; CORD 1000 steps, lr 5e-5, batch 4). Each seed is an independent run (`torch.manual_seed(seed)`,
seeded data order), so seeds 1-3 of the 5-seed original runs are directly comparable (checked earlier: a
re-run reproduced seed 1 of the original exactly). Paired by seed against the original checkpoint.

## Decision rule
- Score of a variant = mean over the 6 (task, seed) pairs of F1(variant) - F1(original), in F1 points.
- Seed noise: the pooled per-seed SD of paired differences between checkpoints in `results/idl*` is 0.75
  points, so the SE of a 6-value mean difference between two variants is about 0.31 points.
- **Threshold: 0.6 points (about 2 SE).** A variant replaces A only if
  score(V) - score(A) > 0.6 **and** its mean gain over A is positive on both FUNSD and CORD.
  If several qualify, the highest score wins.
- Otherwise (a tie at this resolution) A is kept, with one exception fixed in advance: B is chosen over A
  if score(B) >= score(A), because B only switches to the paper's two-layer WPA head (one flag, no extra
  cost). D is more paper-like but needs an extra trained tokenizer, so it does not win ties.
- Validation pre-training metrics and encoder drift are reported but do not enter the rule.
- Resolution: with 3 seeds x 2 tasks, only differences of roughly 0.6 points or more can be told apart from
  seed noise. "No better recipe" means "none clearly better at this resolution".

## Budget
About 70 min per variant (pre-training about 40 min, fine-tuning about 30 min): about 7 GPU-hours, two
resumable queues on GPUs 3 and 4 (`scripts/run_screen.sh`).
