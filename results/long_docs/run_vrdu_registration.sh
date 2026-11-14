#!/bin/bash
# VRDU Registration Forms (FARA), lv2 mixed template, 200 training documents. Test pages: median 539 tokens,
# 55% over 512; truncation at 512 drops 52.1% of the labelled words, 1024 drops none. Seeds 1-5.
cd /data/qbao775/layoutlmv3-pretraining
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
export VRDU_PAGES=/data/qbao775/datasets/vrdu_registration/pages_hf
run() {
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 4 5 \
    --tags long-docs vrdu-registration --out results/long_docs/vrdu_reg_$name.json "$@"
  echo "$(date -u +%FT%TZ) $name exit=$?"
}
(
  run 3 v3_truncate --model microsoft/layoutlmv3-base --dynamic_padding
  run 3 v3_window --model microsoft/layoutlmv3-base --long_docs window
  run 3 v3_extpos1024 --model microsoft/layoutlmv3-base --extend_positions --max_length 1024 --attention lut --dynamic_padding
) &
(
  run 3 ernie_truncate --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --dynamic_padding
  run 3 ernie_window --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --long_docs window
  run 3 ernie_extpos1024 --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --extend_positions --max_length 1024 --dynamic_padding
) &
wait
echo DONE
