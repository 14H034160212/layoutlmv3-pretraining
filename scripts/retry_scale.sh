#!/bin/bash
# retry_scale.sh SIZE GPU: runs run_scale_experiment.sh until the CORD results exist (at most 5 attempts).
# Each attempt resumes pre-training from the last train_state.pt (exactly: no page is replayed) and skips the
# finished fine-tunings. EXP, JOINT_STEPS, FT_EVAL, FT_GPU, BS, ACCUM and GC are passed through. Detached launch:
#   EXP=scale_v2 JOINT_STEPS=25143 FT_EVAL="0 25 50 100 150 200 300 400 600 800" \
#     setsid nohup ./scripts/retry_scale.sh base 4 > runs/scale_v2/base.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
SIZE=${1:?SIZE}; GPU=${2:?GPU}; EXP=${EXP:-scale_v1}; log=runs/$EXP/autoresume.log
mkdir -p runs/$EXP
for attempt in 1 2 3 4 5; do
  [ -f results/$EXP/cord_${SIZE}_scaled.json ] && break
  while [ "$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "${NEED:-0}" ]; do
    echo "$(date -u +%FT%TZ) $SIZE waiting for ${NEED} MiB on GPU $GPU" >> $log; sleep 300; done
  echo "$(date -u +%FT%TZ) $SIZE attempt $attempt on GPU $GPU ($(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader) used)" >> $log
  EXP=$EXP GPU=$GPU SIZE=$SIZE ./scripts/run_scale_experiment.sh >> runs/$EXP/$SIZE.out 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) $SIZE attempt $attempt exit $rc" >> $log
  [ $rc -eq 0 ] && break
  sleep 120
done
