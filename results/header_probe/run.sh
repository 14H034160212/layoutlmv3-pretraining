#!/bin/bash
# Header-necessity probe on RealKIE charities financial pages (full / cut / cut+hdr), 5 seeds, both models.
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
D=/data/qbao775/datasets/realkie/charities_fin_pages_hf
O=results/header_probe
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/header_needed_probe.py --data $D --out $O/charities_v3.json > $O/charities_v3.log 2>&1 &
sleep 90
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/header_needed_probe.py --family ernie-layout --model Norm/ERNIE-Layout-Pytorch \
  --data $D --out $O/charities_ernie.json > $O/charities_ernie.log 2>&1 &
wait
echo DONE
