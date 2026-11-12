#!/bin/bash
# VRDU Ad-buy Forms pages (median ~850 tokens; 512-token truncation drops ~41% of labelled words), seeds 1-5.
cd /data/qbao775/layoutlmv3-pretraining
export CUDA_VISIBLE_DEVICES=4 LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
run() {
  local name=$1; shift
  .venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 4 5 --tags long-docs vrdu \
    --out results/long_docs/vrdu_$name.json "$@"
  echo "$(date -u +%FT%TZ) $name exit=$?"
}
run v3_truncate      --model microsoft/layoutlmv3-base --dynamic_padding
run v3_window        --model microsoft/layoutlmv3-base --long_docs window
run v3_extpos1024    --model microsoft/layoutlmv3-base --extend_positions --sdpa --dynamic_padding --max_length 1024
run v3_extpos2048    --model microsoft/layoutlmv3-base --extend_positions --sdpa --dynamic_padding --max_length 2048
run ernie_truncate   --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --dynamic_padding
run ernie_window     --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --long_docs window
run ernie_extpos2048 --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --extend_positions --dynamic_padding --max_length 2048
echo DONE
