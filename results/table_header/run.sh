#!/bin/bash
# Table-header prefix on RealKIE FCC (header labels: 30 hand-labelled + 243 agent-labelled first pages, 36/36 agreement on hidden checks)
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining
D=/data/qbao775/datasets/realkie/fcc_invoices_pages_hdr_hf
for m in none tag prefix; do
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/table_header_prefix.py --mode $m --seeds 1 2 3 --data $D \
    --out results/table_header/$m.json > results/table_header/$m.log 2>&1 &
  sleep 60
done
wait
echo DONE
