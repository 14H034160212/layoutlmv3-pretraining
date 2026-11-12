#!/bin/bash
# ERNIE-Layout with tiled position embeddings (--extend_positions) at 1024 and 2048 tokens, FUNSD, seeds 1-10.
cd /data/qbao775/layoutlmv3-pretraining
export CUDA_VISIBLE_DEVICES=3 LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
for L in 1024 2048; do
  .venv/bin/python scripts/finetune_funsd.py --task funsd --protocol strict --seeds 1 2 3 4 5 6 7 8 9 10 --tags long-docs \
    --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --extend_positions --max_length $L \
    --out results/long_docs/funsd_ernie_extpos$L.json
  echo "$(date -u +%FT%TZ) ernie_extpos$L exit=$?"
done
echo DONE
