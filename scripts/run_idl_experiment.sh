#!/bin/bash
# Continued pre-training on ~42k IDL pages with DALL-E dVAE MIM targets, for layoutlmv3-base and
# layoutlmv3-large, then FUNSD and CORD fine-tuning (5 seeds) of each original and continued checkpoint.
# Base and large were run on two different GPUs in parallel: GPU=<id> SIZE=base|large ./scripts/run_idl_experiment.sh
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
SIZE=${SIZE:-base}
PY=.venv/bin/python
mkdir -p runs results/idl
if [ "$SIZE" = large ]; then STEPS=4000; LR=3e-5; CORD_LR=2e-5; else STEPS=6000; LR=5e-5; CORD_LR=5e-5; fi
$PY scripts/pretrain.py --model microsoft/layoutlmv3-$SIZE --corpus idl --idl_shards 0-26 --idl_val_shards 2999 \
    --val_docs 512 --pages_per_doc 2 --exclude_funsd_test --count_pages --image_tokenizer dalle \
    --max_steps $STEPS --batch_size 16 --grad_accum 2 --lr $LR --num_workers 12 --eval_every 500 --patience 4 \
    --save_last --output_dir runs/idl_dalle_$SIZE > runs/idl_dalle_$SIZE.log 2>&1
for model in microsoft/layoutlmv3-$SIZE runs/idl_dalle_$SIZE/backbone; do
  tag=$([ "$model" = microsoft/layoutlmv3-$SIZE ] && echo original || echo continued)
  $PY scripts/finetune_funsd.py --task funsd --model $model --seeds 1 2 3 4 5 --out results/idl/funsd_${SIZE}_$tag.json
  $PY scripts/finetune_funsd.py --task cord --lr $CORD_LR --model $model --seeds 1 2 3 4 5 --out results/idl/cord_${SIZE}_$tag.json
done
echo EXPERIMENT_DONE
