#!/bin/bash
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining
D=/data/qbao775/datasets/realkie/fcc_invoices_pages_hdr_hf
for m in none tag prefix; do
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/table_header_prefix.py --mode $m --seeds 4 5 --data $D \
    --out results/table_header/${m}_s45.json > results/table_header/${m}_s45.log 2>&1 &
  sleep 60
done
wait
echo DONE
