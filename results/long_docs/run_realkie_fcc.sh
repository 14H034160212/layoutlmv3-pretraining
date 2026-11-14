#!/bin/bash
# RealKIE FCC Invoices (real TV-ad invoices, CC BY-NC 4.0). Test pages: median 837 tokens, 66% over 512, 43% over 1024;
# truncation drops 58.1% of the labelled words at 512, 21.2% at 1024, 0.9% at 2048. Seeds 1-5. Starts after the
# registration-form runs and the RealKIE conversion have finished.
cd /data/qbao775/layoutlmv3-pretraining
until grep -q "^DONE" results/long_docs/run_vrdu_registration.log && [ -f /data/qbao775/datasets/realkie/fcc_invoices_pages_hf/prepare_info.json ]; do sleep 120; done
export LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
export VRDU_PAGES=/data/qbao775/datasets/realkie/fcc_invoices_pages_hf
run() {
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 2 3 4 5 \
    --tags long-docs realkie-fcc --out results/long_docs/realkie_fcc_$name.json "$@"
  echo "$(date -u +%FT%TZ) $name exit=$?"
}
(
  run 3 v3_truncate --model microsoft/layoutlmv3-base --dynamic_padding
  run 3 v3_window --model microsoft/layoutlmv3-base --long_docs window
  run 3 v3_extpos1024 --model microsoft/layoutlmv3-base --extend_positions --max_length 1024 --attention lut --dynamic_padding
  run 3 v3_extpos2048 --model microsoft/layoutlmv3-base --extend_positions --max_length 2048 --attention lut --dynamic_padding
) &
(
  run 3 ernie_truncate --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --dynamic_padding
  run 3 ernie_window --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --long_docs window
  run 3 ernie_extpos2048 --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --extend_positions --max_length 2048 --dynamic_padding
) &
wait
echo DONE
