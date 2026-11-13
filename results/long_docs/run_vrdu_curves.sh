#!/bin/bash
# Learning curves on VRDU: truncation at 512 vs positions tiled to 2048 (lut attention), seeds 1-3, eval every 250 steps.
cd /data/qbao775/layoutlmv3-pretraining
export CUDA_VISIBLE_DEVICES=4 LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining
EVAL="--eval_at 0 250 500 750 1000 1250 1500 1750"
.venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 --model microsoft/layoutlmv3-base \
  --dynamic_padding $EVAL --tags long-docs vrdu curves --out results/long_docs/vrdu_curve_v3_truncate.json > results/long_docs/run_vrdu_curve_truncate.log 2>&1 &
.venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 --model microsoft/layoutlmv3-base \
  --extend_positions --max_length 2048 --attention lut --dynamic_padding $EVAL --tags long-docs vrdu curves \
  --out results/long_docs/vrdu_curve_v3_extpos2048.json > results/long_docs/run_vrdu_curve_2048.log 2>&1 &
wait
echo DONE
