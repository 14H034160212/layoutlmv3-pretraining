#!/bin/bash
# Continued pre-training with a frozen-encoder head warm-up (see README, "Head warm-up").
#   GPU=<id> SIZE=base|large INIT=pretrained|scratch ./scripts/run_idl_warmup_experiment.sh
# pretrained: 1000 head-only steps, then joint training at a gentler lr with layer-wise decay 0.9,
#             then FUNSD and CORD fine-tuning (5 seeds) of the continued checkpoint.
# scratch:    the same config with random weights (control): the same 1000 head-only steps, then joint
#             training from scratch at lr 1e-4 without layer decay (fine-tuning: FUNSD, 2 seeds).
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU} HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
SIZE=${SIZE:-base}; INIT=${INIT:-pretrained}
PY=.venv/bin/python
OUT=runs/idl_warmup_${SIZE}_$INIT
mkdir -p runs results/idl_warmup
if [ "$SIZE" = large ]; then STEPS=4000; LR=1e-5; CORD_LR=2e-5; else STEPS=6000; LR=2e-5; CORD_LR=5e-5; fi
if [ "$INIT" = scratch ]; then EXTRA="--random_init"; LR=1e-4; DECAY=1.0; else EXTRA=""; DECAY=0.9; fi
$PY scripts/pretrain.py --model microsoft/layoutlmv3-$SIZE $EXTRA --corpus idl --idl_shards 0-26 --idl_val_shards 2999 \
    --val_docs 512 --pages_per_doc 2 --exclude_funsd_test --image_tokenizer dalle --dalle_encoder runs/dalle/encoder.pkl \
    --head_warmup_steps 1000 --head_lr 1e-3 --max_steps $STEPS --lr $LR --layer_decay $DECAY \
    --batch_size 16 --grad_accum 2 --num_workers 12 --eval_every 250 --patience 8 \
    --save_last --output_dir $OUT > $OUT.log 2>&1
cp $OUT/log.json results/idl_warmup/pretrain_${SIZE}_${INIT}_log.json
cp $OUT/run_info.json results/idl_warmup/pretrain_${SIZE}_${INIT}_run_info.json
if [ "$INIT" = scratch ]; then
  $PY scripts/finetune_funsd.py --task funsd --model $OUT/backbone --seeds 1 2 --out results/idl_warmup/funsd_${SIZE}_scratch.json
else
  $PY scripts/finetune_funsd.py --task funsd --model $OUT/backbone --seeds 1 2 3 4 5 --out results/idl_warmup/funsd_${SIZE}_warmup_continued.json
  $PY scripts/finetune_funsd.py --task cord --lr $CORD_LR --model $OUT/backbone --seeds 1 2 3 4 5 --out results/idl_warmup/cord_${SIZE}_warmup_continued.json
fi
echo EXPERIMENT_DONE
