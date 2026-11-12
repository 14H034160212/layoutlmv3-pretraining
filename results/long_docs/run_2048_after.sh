#!/bin/bash
# After run_when_free.sh finishes: FUNSD at 2048 tokens, batch 2 as the other groups. With SDPA attention (LayoutLMv3)
# and the index-select relative bias (ERNIE-Layout, default), batch 2 at 2048 fits a 16 GB V100 (train_memory_v2*.json).
cd /data/qbao775/layoutlmv3-pretraining
L=results/long_docs/run.log
until grep -q "^DONE" $L; do sleep 120; done
GPU=$(grep -o "starting on GPU [0-9]" $L | grep -o "[0-9]$")
export CUDA_VISIBLE_DEVICES=$GPU LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
SEEDS="1 2 3 4 5 6 7 8 9 10"
run() {
  local name=$1; shift
  .venv/bin/python scripts/finetune_funsd.py --task funsd --protocol strict --seeds $SEEDS --tags long-docs v100-2048 \
    --out results/long_docs/funsd_$name.json "$@"
  echo "$(date -u +%FT%TZ) $name exit=$?"
}
echo "$(date -u +%FT%TZ) 2048 runs on GPU $GPU"
run v3_extpos2048    --model microsoft/layoutlmv3-base --extend_positions --sdpa --max_length 2048
run ernie_extrap2048 --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --extrapolation --max_length 2048
echo DONE2048
