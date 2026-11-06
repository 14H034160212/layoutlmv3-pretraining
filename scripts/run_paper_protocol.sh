#!/bin/bash
# Paper fine-tuning protocol, many seeds, for a list of base checkpoints.
#   GPU=4 MODELS="original scaled_v2" SEEDS="1 2 3 4 5 6 7 8 9 10" setsid nohup ./scripts/run_paper_protocol.sh > runs/paper_protocol_gpu4.out 2>&1 < /dev/null &
# Resumable per (model, task, seed): each seed is its own results file.
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1
GPU=${GPU:?set GPU}; PY=.venv/bin/python; RES=results/${PROTOCOL:-paper}_protocol
declare -A M=( [original]=microsoft/layoutlmv3-base [scaled_v1]=runs/scale_v1/base/backbone
               [scaled_v2]=runs/scale_v2/base/backbone [warmup_42k]=runs/idl_warmup_base_pretrained/backbone )
for seed in ${SEEDS:?}; do for name in ${MODELS:?}; do for task in funsd cord; do
  out=$RES/${task}_base_${name}_seed${seed}.json; [ -f $out ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/finetune_funsd.py --task $task --protocol ${PROTOCOL:-paper} --model ${M[$name]} \
    --seeds $seed --tags ${PROTOCOL:-paper}-protocol base $name --out $out >> runs/${PROTOCOL:-paper}_protocol_${name}.log 2>&1
done; done; done
echo "PAPER_PROTOCOL_DONE $GPU"
