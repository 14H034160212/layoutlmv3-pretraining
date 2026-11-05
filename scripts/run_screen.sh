#!/bin/bash
# Recipe screen v1 on layoutlmv3-base (see results/screen_v1/decision_rule.md). Resumable: a variant's
# pre-training is skipped if runs/screen_v1/<id>/DONE exists, a fine-tuning if its results file exists.
#   GPU=3 setsid nohup ./scripts/run_screen.sh A B C > runs/screen_v1_gpu3.out 2>&1 &
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU} HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PY=.venv/bin/python
OUT=results/screen_v1
mkdir -p runs/screen_v1 $OUT
for v in "$@"; do
  LR=2e-5; EXTRA="--wpa_head linear"; TOKENIZER=runs/dalle/encoder.pkl   # the screen's control head
  case $v in
    A) ;;
    B) EXTRA="--wpa_head mlp" ;;
    C) EXTRA="$EXTRA --mim_weight 0" ;;
    D) TOKENIZER=runs/dvae_doc/encoder.pt ;;
    E) LR=1e-5 ;;
    F) EXTRA="$EXTRA --freeze_layers 6" ;;
    *) echo "unknown variant $v"; exit 1 ;;
  esac
  dir=runs/screen_v1/$v
  if [ ! -f $dir/DONE ]; then
    rm -rf $dir
    $PY scripts/pretrain.py --model microsoft/layoutlmv3-base --corpus idl --idl_shards 0-26 --idl_val_shards 2999 \
        --val_docs 512 --pages_per_doc 2 --exclude_funsd_test --image_tokenizer dalle --dalle_encoder $TOKENIZER \
        --head_warmup_steps 1000 --head_lr 1e-3 --max_steps 2500 --lr $LR --layer_decay 0.9 $EXTRA \
        --batch_size 16 --grad_accum 2 --num_workers 12 --eval_every 250 --seed 42 \
        --tags screen-v1 variant-$v --output_dir $dir > $dir.log 2>&1
    cp $dir/log.json $OUT/pretrain_${v}_log.json; cp $dir/drift.json $OUT/drift_${v}.json
    cp $dir/run_info.json $OUT/pretrain_${v}_run_info.json; cp $dir/run_record.json $OUT/pretrain_${v}_run_record.json
    rm -rf $dir/warmup_backbone $dir/warmup_heads.pt
    touch $dir/DONE
  fi
  [ -f $OUT/funsd_$v.json ] || $PY scripts/finetune_funsd.py --task funsd --model $dir/backbone --seeds 1 2 3 \
      --tags screen-v1 variant-$v --out $OUT/funsd_$v.json > $dir.funsd.log 2>&1
  [ -f $OUT/cord_$v.json ] || $PY scripts/finetune_funsd.py --task cord --lr 5e-5 --model $dir/backbone --seeds 1 2 3 \
      --tags screen-v1 variant-$v --out $OUT/cord_$v.json > $dir.cord.log 2>&1
  echo "variant $v done $(date -u +%FT%TZ)"
done
echo SCREEN_DONE
