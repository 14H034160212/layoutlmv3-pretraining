#!/bin/bash
# ERNIE-Layout image preprocessing: train with the README's (double normalisation) or PaddleNLP's preprocessing,
# evaluate each model with both. FUNSD, strict protocol, seeds 1-3.
cd /data/qbao775/layoutlmv3-pretraining
export CUDA_VISIBLE_DEVICES=4 LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
for train in readme paddle; do
  .venv/bin/python scripts/finetune_funsd.py --task funsd --protocol strict --seeds 1 2 3 --family ernie-layout \
    --model Norm/ERNIE-Layout-Pytorch --ernie_images $train --eval_both_images --tags ernie-images \
    --out results/long_docs/funsd_ernie_images_train_$train.json
  echo "$(date -u +%FT%TZ) train_$train exit=$?"
done
echo DONE
