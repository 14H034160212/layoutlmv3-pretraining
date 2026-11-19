#!/bin/bash
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
D=/data/qbao775/datasets/realkie/fcc_invoices_pages_hdr_hf
run() { CUDA_VISIBLE_DEVICES=$1 .venv/bin/python scripts/table_header_prefix.py --family ernie-layout --model Norm/ERNIE-Layout-Pytorch \
  --mode $2 --seeds 1 2 3 4 5 --data $D --out results/table_header/ernie_$2.json > results/table_header/ernie_$2.log 2>&1; }
run 3 none &
( until grep -q DONE results/table_header/run_s45.log; do sleep 60; done; run 4 tag ) &
( until grep -q DONE results/table_header/run_s45.log; do sleep 60; done; sleep 90; run 4 prefix ) &
wait
echo DONE
