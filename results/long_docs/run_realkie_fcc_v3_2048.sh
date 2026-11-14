#!/bin/bash
# Rerun of realkie_fcc v3_extpos2048, which ran out of memory on a shared GPU 3.
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining VRDU_PAGES=/data/qbao775/datasets/realkie/fcc_invoices_pages_hf
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 4 5 \
  --tags long-docs realkie-fcc --out results/long_docs/realkie_fcc_v3_extpos2048.json \
  --model microsoft/layoutlmv3-base --extend_positions --max_length 2048 --attention lut --dynamic_padding
rc=$?
echo "$(date -u +%FT%TZ) v3_extpos2048 exit=$rc"
echo DONE
