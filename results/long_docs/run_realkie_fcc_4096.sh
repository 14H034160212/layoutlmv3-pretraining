#!/bin/bash
# RealKIE FCC Invoices at 4096 tokens (whole pages up to 3,684 tokens; at 2048, 0.94% of field words are cut).
# Batch 2 as two micro-batches of 1 (--micro_batch 1): same pages per update, fits a 16 GB card (9.5 GB peak for v3).
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining VRDU_PAGES=/data/qbao775/datasets/realkie/fcc_invoices_pages_hf
export ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
run() {
  local name=$1; shift
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 4 5 \
    --tags long-docs realkie-fcc --out results/long_docs/realkie_fcc_${name}.json --micro_batch 1 "$@"
  local rc=$?
  echo "$(date -u +%FT%TZ) $name exit=$rc"
}
run v3_extpos4096 --model microsoft/layoutlmv3-base --extend_positions --max_length 4096 --attention lut --dynamic_padding
run ernie_extpos4096 --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --extend_positions --max_length 4096 --dynamic_padding
echo DONE
