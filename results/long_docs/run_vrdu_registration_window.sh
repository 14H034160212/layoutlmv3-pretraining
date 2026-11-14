#!/bin/bash
# Rerun of the two window runs of run_vrdu_registration.sh, which crashed (NameError in evaluate_windows, since fixed).
cd /data/qbao775/layoutlmv3-pretraining
until grep -q "^DONE" results/long_docs/run_realkie_fcc.log; do sleep 120; done
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
export VRDU_PAGES=/data/qbao775/datasets/vrdu_registration/pages_hf
run() {
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 4 5 \
    --tags long-docs vrdu-registration --out results/long_docs/vrdu_reg_$name.json "$@"
  local rc=$?
  echo "$(date -u +%FT%TZ) $name exit=$rc"
}
run 3 v3_window --model microsoft/layoutlmv3-base --long_docs window &
run 3 ernie_window --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --long_docs window &
wait
echo DONE
