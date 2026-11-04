#!/bin/bash
# Full experiment: continued pre-training on FUNSD train, then FUNSD fine-tuning (3 seeds) for base vs continued.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-3} HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PY=.venv/bin/python
mkdir -p runs results
$PY scripts/pretrain.py --output_dir runs/funsd_pt --max_steps 3000 --eval_every 250 > runs/pretrain.log 2>&1
$PY scripts/finetune_funsd.py --model microsoft/layoutlmv3-base --seeds 1 2 3 --out results/funsd_base.json > runs/ft_base.log 2>&1
$PY scripts/finetune_funsd.py --model runs/funsd_pt/backbone --seeds 1 2 3 --out results/funsd_continued.json > runs/ft_pt.log 2>&1
echo EXPERIMENT_DONE
