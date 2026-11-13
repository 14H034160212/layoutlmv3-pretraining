#!/bin/bash
# Position interpolation instead of copying, VRDU, seeds 1-5.
cd /data/qbao775/layoutlmv3-pretraining
export CUDA_VISIBLE_DEVICES=3 LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
.venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 4 5 --model microsoft/layoutlmv3-base \
  --extend_positions --position_mode interpolate --max_length 2048 --attention lut --dynamic_padding --tags long-docs vrdu interpolate \
  --out results/long_docs/vrdu_v3_interp2048.json
echo "$(date -u +%FT%TZ) v3_interp2048 exit=$?"
.venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 4 5 --model Norm/ERNIE-Layout-Pytorch --family ernie-layout \
  --extend_positions --position_mode interpolate --max_length 2048 --dynamic_padding --tags long-docs vrdu interpolate \
  --out results/long_docs/vrdu_ernie_interp2048.json
echo "$(date -u +%FT%TZ) ernie_interp2048 exit=$?"
echo DONE
