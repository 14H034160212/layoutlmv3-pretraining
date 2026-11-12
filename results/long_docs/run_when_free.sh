#!/bin/bash
# Wait until GPU 3 or 4 has >= 20 GB free, then on that GPU only:
#   1. inference memory benchmark (fp16, batch 1, 15 GB cap ~ V100 16GB), documents of 512-4096 tokens
#   2. FUNSD entity F1, --protocol strict, seeds 1-10, for:
#        LayoutLMv3-base truncate / window / position copy to 1024,
#        ERNIE-Layout truncate / window / RoPE extrapolation to 1024
cd /data/qbao775/layoutlmv3-pretraining
NEED_MB=20000
while true; do
  for g in 3 4; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    total=$(nvidia-smi -i $g --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
    if [ $((total - used)) -ge $NEED_MB ]; then GPU=$g; break 2; fi
  done
  sleep 300
done
echo "$(date -u +%FT%TZ) starting on GPU $GPU ($((total - used)) MiB free)"
export CUDA_VISIBLE_DEVICES=$GPU LAYOUTLMV3_WANDB_PROJECT=layoutlmv3-pretraining ERNIE_LAYOUT_PYTORCH=/data/qbao775/ernie-layout-pytorch
OUT=results/long_docs
SEEDS="1 2 3 4 5 6 7 8 9 10"

.venv/bin/python scripts/bench_inference_memory.py --lengths 512 1024 2048 4096 --cap_gb 15 \
  --out $OUT/inference_memory.json
echo "$(date -u +%FT%TZ) bench exit=$?"

run() {  # name, extra args
  local name=$1; shift
  .venv/bin/python scripts/finetune_funsd.py --task funsd --protocol strict --seeds $SEEDS --tags long-docs \
    --out $OUT/funsd_$name.json "$@"
  echo "$(date -u +%FT%TZ) $name exit=$?"
}
run v3_truncate    --model microsoft/layoutlmv3-base --long_docs truncate
run v3_window      --model microsoft/layoutlmv3-base --long_docs window
run ernie_truncate --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --long_docs truncate
run ernie_window   --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --long_docs window
run ernie_extrap1024 --model Norm/ERNIE-Layout-Pytorch --family ernie-layout --extrapolation --max_length 1024
run v3_extpos1024  --model microsoft/layoutlmv3-base --extend_positions --max_length 1024
echo DONE
