#!/bin/bash
# LoRA fine-tuning of ERNIE-Layout: in-memory vs reloaded F1, BatchNorm in train mode vs frozen.
# FUNSD (1000 steps) and RealKIE FCC Invoices (2000 steps), images normalised twice (readme) and 0-255 (paddle), 3 seeds.
cd /data/qbao775/layoutlmv3-pretraining
export ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
run() {
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/lora_batchnorm_check.py --out results/lora_bn/$name.json "$@" \
    > results/lora_bn/$name.log 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) $name exit=$rc"
}
(
  run 3 funsd_readme --task funsd --images readme --steps 1000
  run 3 funsd_paddle --task funsd --images paddle --steps 1000
) &
(
  export VRDU_PAGES=/data/qbao775/datasets/realkie/fcc_invoices_pages_hf
  run 4 realkie_readme --task vrdu --images readme --steps 2000
  run 4 realkie_paddle --task vrdu --images paddle --steps 2000
) &
wait
echo DONE
