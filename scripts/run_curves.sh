#!/bin/bash
# Fine-tuning learning curves (test loss and F1 during fine-tuning) for base checkpoints, seeds 1-3.
#   GPU=3 setsid nohup ./scripts/run_curves.sh > runs/curves_v1.out 2>&1 < /dev/null &
# Resumable: a (model, task) pair is skipped if its results file exists.
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1
GPU=${GPU:?set GPU}; PY=.venv/bin/python; RES=results/curves_v1
EVAL="0 25 50 100 150 200 300 400 600 800"
declare -A M=( [original]=microsoft/layoutlmv3-base [warmup_42k]=runs/idl_warmup_base_pretrained/backbone
               [scaled_404k]=runs/scale_v1/base/backbone [random_init_42k]=runs/idl_warmup_base_scratch/backbone )
for name in original scaled_404k warmup_42k random_init_42k; do
  for task in funsd cord; do
    out=$RES/${task}_base_${name}.json; [ -f $out ] && continue
    lr=""; [ $task = cord ] && lr="--lr 5e-5"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/finetune_funsd.py --task $task $lr --model ${M[$name]} --seeds 1 2 3 \
      --eval_at $EVAL --tags curves-v1 base $name --out $out > runs/curves_v1_${task}_${name}.log 2>&1
  done
done
echo CURVES_DONE
