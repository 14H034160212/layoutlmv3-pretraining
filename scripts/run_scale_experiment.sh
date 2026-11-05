#!/bin/bash
# Scale-up v1 (results/scale_v1/plan.md): continued pre-training on IDL shards 0-259 (2 pages per document,
# FUNSD-test filter, validation on shard 2999 as before) with the head warm-up recipe and the two-layer MLP WPA head,
# then FUNSD and CORD fine-tuning (seeds 1-5) of the selected checkpoint.
#   GPU=3 SIZE=base  setsid nohup ./scripts/run_scale_experiment.sh > runs/scale_v1/base.out 2>&1 < /dev/null &
#   GPU=3 SIZE=large setsid nohup ./scripts/run_scale_experiment.sh > runs/scale_v1/large.out 2>&1 < /dev/null &
# FT_GPU=<id> runs the fine-tuning on another GPU (default: GPU).
# Resumable: pre-training continues from runs/scale_v1/<size>/train_state.pt (saved every 2000 steps) if it was
# interrupted, and is skipped if DONE exists; a fine-tuning is skipped if its results file exists.
set -e
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1
GPU=${GPU:?set GPU}; FT_GPU=${FT_GPU:-$GPU}; SIZE=${SIZE:-base}
PY=.venv/bin/python
SHARDS=0-259
JOINT_STEPS=${JOINT_STEPS:?set JOINT_STEPS (plan: ceil(2 N / 32))}
OUT=runs/scale_v1/$SIZE
RES=results/scale_v1
mkdir -p runs/scale_v1 $RES
if [ "$SIZE" = large ]; then LR=1e-5; DECAY=0.95; CORD_LR=2e-5; else LR=2e-5; DECAY=0.9; CORD_LR=5e-5; fi
TAGS="scale-v1 $SIZE"
# BS x ACCUM must stay 32 (plan); GC=--gradient_checkpointing to fit next to other jobs
GC=${GC:-}
[ $(( ${BS:-16} * ${ACCUM:-2} )) -eq 32 ] || { echo "BS*ACCUM must be 32"; exit 1; }
if [ ! -f $OUT/DONE ]; then
  [ -f $OUT/train_state.pt ] || rm -rf $OUT          # fresh start unless there is a state to resume from
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pretrain.py --model microsoft/layoutlmv3-$SIZE --corpus idl --idl_shards $SHARDS \
      --idl_val_shards 2999 --val_docs 512 --pages_per_doc 2 --exclude_funsd_test --image_tokenizer dalle \
      --dalle_encoder runs/dalle/encoder.pkl --wpa_head mlp --head_warmup_steps 1000 --head_lr 1e-3 \
      --max_steps $JOINT_STEPS --lr $LR --layer_decay $DECAY --batch_size ${BS:-16} --grad_accum ${ACCUM:-2} $GC --num_workers 12 \
      --eval_every 500 --patience 8 --seed 42 --save_last --state_every 2000 --resume \
      --tags $TAGS --output_dir $OUT >> $OUT.log 2>&1
  cp $OUT/log.json $RES/pretrain_${SIZE}_log.json
  cp $OUT/run_info.json $RES/pretrain_${SIZE}_run_info.json
  cp $OUT/drift.json $RES/drift_${SIZE}.json
  cp $OUT/run_record.json $RES/pretrain_${SIZE}_run_record.json
  touch $OUT/DONE
  echo "pretraining $SIZE done $(date -u +%FT%TZ)"
fi
[ -f $RES/funsd_${SIZE}_scaled.json ] || CUDA_VISIBLE_DEVICES=$FT_GPU $PY scripts/finetune_funsd.py --task funsd \
    --model $OUT/backbone --seeds 1 2 3 4 5 --tags $TAGS --out $RES/funsd_${SIZE}_scaled.json > $OUT.funsd.log 2>&1
[ -f $RES/cord_${SIZE}_scaled.json ] || CUDA_VISIBLE_DEVICES=$FT_GPU $PY scripts/finetune_funsd.py --task cord --lr $CORD_LR \
    --model $OUT/backbone --seeds 1 2 3 4 5 --tags $TAGS --out $RES/cord_${SIZE}_scaled.json > $OUT.cord.log 2>&1
echo "EXPERIMENT_DONE $SIZE $(date -u +%FT%TZ)"
