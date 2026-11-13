#!/bin/bash
# ERNIE-Layout tokenizer padding side: the port pads on the left (positions of short documents shifted),
# PaddleNLP pads on the right (documents start at position 0). FUNSD, strict, seeds 1-5.
cd /data/qbao775/layoutlmv3-pretraining
export CUDA_VISIBLE_DEVICES=4 LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
for side in left right; do
  .venv/bin/python scripts/finetune_funsd.py --task funsd --protocol strict --seeds 1 2 3 4 5 --family ernie-layout \
    --model Norm/ERNIE-Layout-Pytorch --ernie_padding_side $side --tags ernie-padding \
    --out results/long_docs/funsd_ernie_padding_$side.json
  echo "$(date -u +%FT%TZ) padding_$side exit=$?"
done
echo DONE
