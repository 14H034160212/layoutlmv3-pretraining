#!/bin/bash
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining
COMMON="--task vrdu --protocol strict --seeds 1 --model microsoft/layoutlmv3-base --extend_positions --max_length 2048 --attention lut --dynamic_padding --tags analysis"
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/finetune_funsd.py $COMMON --position_mode copy --dump_docs results/long_docs/analysis/vrdu_v3_copy_docs.json --out results/long_docs/analysis/vrdu_v3_copy_seed1.json > results/long_docs/analysis/copy.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/finetune_funsd.py $COMMON --position_mode interpolate --dump_docs results/long_docs/analysis/vrdu_v3_interp_docs.json --out results/long_docs/analysis/vrdu_v3_interp_seed1.json > results/long_docs/analysis/interp.log 2>&1 &
wait; echo DONE
