#!/bin/bash
# Controlled test: RealKIE charities with the two year columns swapped in half of the documents (header needed).
# none / tag / prefix, 5 seeds, LayoutLMv3 and ERNIE-Layout; cut training pages added for all modes.
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
R=/data/qbao775/datasets/realkie
COMMON="--seeds 1 2 3 4 5 --data $R/charities_swap_hdr_hf --cuts $R/charities_swap_cuts_hf --train_cuts $R/charities_swap_train_cuts_hf --value_substr YEAR_END"
O=results/year_swap
v3() { CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/table_header_prefix.py --mode $1 $COMMON --out $O/v3_$1.json > $O/v3_$1.log 2>&1; }
er() { CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/table_header_prefix.py --family ernie-layout --model Norm/ERNIE-Layout-Pytorch \
  --mode $1 $COMMON --out $O/ernie_$1.json > $O/ernie_$1.log 2>&1; }
( v3 none; v3 tag; v3 prefix ) &
( sleep 90; er none; er tag; er prefix ) &
wait
echo DONE
