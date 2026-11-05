#!/bin/bash
# Continued pre-training with a frozen-encoder head warm-up (see README, "Head warm-up").
#   GPU=<id> SIZE=base|large INIT=pretrained|scratch ./scripts/run_idl_warmup_experiment.sh
# pretrained: 1000 head-only steps, then joint training at a gentler lr with layer-wise decay,
#             then FUNSD and CORD fine-tuning (5 seeds) of the continued checkpoint.
#             base: 6000 joint steps, lr 2e-5, decay 0.9. large: 4000 joint steps, lr 1e-5, decay 0.95
#             (0.95^25 = 0.28 keeps base's embeddings-to-top lr ratio 0.9^13 = 0.25 over 24 layers).
# scratch:    the same config with random weights (control): the same 1000 head-only steps, then joint
#             training from scratch at lr 1e-4 without layer decay (fine-tuning: FUNSD, 2 seeds).
# Resumable: pre-training is skipped if $OUT/DONE exists, a fine-tuning if its results file exists.
# LR, DECAY and STEPS can be overridden from the environment.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU} HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
SIZE=${SIZE:-base}; INIT=${INIT:-pretrained}
PY=.venv/bin/python
OUT=runs/idl_warmup_${SIZE}_$INIT
RES=results/idl_warmup
mkdir -p runs $RES
if [ "$SIZE" = large ]; then D_STEPS=4000; D_LR=1e-5; D_DECAY=0.95; CORD_LR=2e-5
else D_STEPS=6000; D_LR=2e-5; D_DECAY=0.9; CORD_LR=5e-5; fi
if [ "$INIT" = scratch ]; then EXTRA="--random_init"; D_LR=1e-4; D_DECAY=1.0; else EXTRA=""; fi
STEPS=${STEPS:-$D_STEPS}; LR=${LR:-$D_LR}; DECAY=${DECAY:-$D_DECAY}
TAGS="idl_warmup $SIZE $INIT"
if [ ! -f $OUT/DONE ]; then
  rm -rf $OUT
  $PY scripts/pretrain.py --model microsoft/layoutlmv3-$SIZE $EXTRA --corpus idl --idl_shards 0-26 --idl_val_shards 2999 \
      --val_docs 512 --pages_per_doc 2 --exclude_funsd_test --image_tokenizer dalle --dalle_encoder runs/dalle/encoder.pkl \
      --head_warmup_steps 1000 --head_lr 1e-3 --max_steps $STEPS --lr $LR --layer_decay $DECAY \
      --batch_size 16 --grad_accum 2 --num_workers 12 --eval_every 250 --patience 8 \
      --save_last --tags $TAGS --output_dir $OUT > $OUT.log 2>&1
  cp $OUT/log.json $RES/pretrain_${SIZE}_${INIT}_log.json
  cp $OUT/run_info.json $RES/pretrain_${SIZE}_${INIT}_run_info.json
  [ -f $OUT/drift.json ] && cp $OUT/drift.json $RES/drift_${SIZE}_${INIT}.json
  touch $OUT/DONE
fi
if [ "$INIT" = scratch ]; then
  [ -f $RES/funsd_${SIZE}_scratch.json ] || $PY scripts/finetune_funsd.py --task funsd --model $OUT/backbone --seeds 1 2 \
      --tags $TAGS --out $RES/funsd_${SIZE}_scratch.json
else
  [ -f $RES/funsd_${SIZE}_warmup_continued.json ] || $PY scripts/finetune_funsd.py --task funsd --model $OUT/backbone \
      --seeds 1 2 3 4 5 --tags $TAGS --out $RES/funsd_${SIZE}_warmup_continued.json
  [ -f $RES/cord_${SIZE}_warmup_continued.json ] || $PY scripts/finetune_funsd.py --task cord --lr $CORD_LR --model $OUT/backbone \
      --seeds 1 2 3 4 5 --tags $TAGS --out $RES/cord_${SIZE}_warmup_continued.json
fi
echo EXPERIMENT_DONE
