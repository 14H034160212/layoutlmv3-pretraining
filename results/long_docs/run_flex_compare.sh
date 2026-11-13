#!/bin/bash
# Same trained model evaluated with SDPA and with FlexAttention (VRDU, LayoutLMv3 tiled positions to 2048, fixed padding).
cd /data/qbao775/layoutlmv3-pretraining

export CUDA_VISIBLE_DEVICES=3 LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining
.venv/bin/python scripts/finetune_funsd.py --task vrdu --protocol strict --seeds 1 --tags long-docs vrdu flex \
  --model microsoft/layoutlmv3-base --extend_positions --sdpa --max_length 2048 --eval_flex \
  --out results/long_docs/vrdu_v3_extpos2048_flexcompare.json
echo "$(date -u +%FT%TZ) flexcompare exit=$?"
echo DONE
