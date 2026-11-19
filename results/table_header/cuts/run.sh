#!/bin/bash
# Retrain none/tag/prefix (5 seeds, LayoutLMv3 + ERNIE-Layout) and also score 436 synthetic headerless chunks.
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
D=/data/qbao775/datasets/realkie/fcc_invoices_pages_hdr_hf
C=/data/qbao775/datasets/realkie/fcc_invoices_pages_hdr_cuts_hf
O=results/table_header/cuts
v3() { CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/table_header_prefix.py --mode $1 --seeds 1 2 3 4 5 --data $D --cuts $C \
  --out $O/v3_$1.json > $O/v3_$1.log 2>&1; }
er() { CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/table_header_prefix.py --family ernie-layout --model Norm/ERNIE-Layout-Pytorch \
  --mode $1 --seeds 1 2 3 4 5 --data $D --cuts $C --out $O/ernie_$1.json > $O/ernie_$1.log 2>&1; }
( v3 none; v3 tag; v3 prefix ) &
( sleep 120; er none; er tag; er prefix ) &
wait
echo DONE
