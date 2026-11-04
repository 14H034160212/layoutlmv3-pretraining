#!/bin/bash
# Frozen-encoder probes: 1000 head-only steps (no joint training) for the released and a random encoder.
#   GPU=<id> ./scripts/run_probes.sh <wpa_head>:<targets> ...     targets: dalle | dalle_doc | kmeans
# dalle_doc uses the document-fine-tuned dVAE encoder from scripts/finetune_dvae.py (runs/dvae_doc/encoder.pt).
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU} HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PY=.venv/bin/python
mkdir -p runs/probes results/probes
for spec in "$@"; do
  wpa=${spec%%:*}; targets=${spec##*:}
  case $targets in
    dalle) TOK="--image_tokenizer dalle --dalle_encoder runs/dalle/encoder.pkl" ;;
    dalle_doc) TOK="--image_tokenizer dalle --dalle_encoder runs/dvae_doc/encoder.pt" ;;
    kmeans) TOK="--image_tokenizer kmeans --image_vocab_size 512 --kmeans_fit_pages 2000" ;;
  esac
  for init in pretrained random; do
    EXTRA=$([ $init = random ] && echo --random_init || true)
    name=base_${init}_wpa-${wpa}_mim-${targets}
    $PY scripts/pretrain.py --model microsoft/layoutlmv3-base $EXTRA --corpus idl --idl_shards 0-26 \
        --idl_val_shards 2999 --val_docs 512 --pages_per_doc 2 --exclude_funsd_test $TOK --wpa_head $wpa \
        --head_warmup_steps 1000 --head_lr 1e-3 --max_steps 0 --batch_size 16 --grad_accum 2 --num_workers 12 \
        --eval_every 250 --output_dir runs/probes/$name > runs/probes/$name.log 2>&1
    cp runs/probes/$name/log.json results/probes/${name}_log.json
    rm -rf runs/probes/$name/backbone runs/probes/$name/warmup_backbone runs/probes/$name/*.pt
    echo "$name done"
  done
done
echo PROBES_DONE
