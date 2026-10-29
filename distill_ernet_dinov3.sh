#!/usr/bin/env bash
# DINOv3 -> ER-Net distillation on unlabeled ImageNet-1k train images (LightlyTrain).
#
#   DATA_PATH=/kaggle/working/imagenet GPUS=2 bash distill_ernet_dinov3.sh ernet_n
#
# Every variable can be overridden from the environment. BATCH_SIZE is the GLOBAL
# batch (split over GPUS). Re-run with RESUME=1 (same OUT_DIR) after an interruption.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

MODEL="${1:-ernet_n}"
DATA_PATH="${DATA_PATH:-/kaggle/working/imagenet}"
OUT_DIR="${OUT_DIR:-runs/distill_dinov3_${MODEL}}"
TEACHER="${TEACHER:-dinov3/vitb16}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
GPUS="${GPUS:-auto}"
NUM_WORKERS="${NUM_WORKERS:-auto}"
IMAGES_PER_CLASS="${IMAGES_PER_CLASS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"          # e.g. "--no-disentangle" for the tree ablation
export LIGHTLY_TRAIN_EVENTS_DISABLED="${LIGHTLY_TRAIN_EVENTS_DISABLED:-1}"

RESUME_FLAG=()
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAG=(--resume); fi

# shellcheck disable=SC2086
python distill_dinov3.py \
    --model "$MODEL" \
    --data "$DATA_PATH/train" \
    --out "$OUT_DIR" \
    --teacher "$TEACHER" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --accum-steps "$ACCUM_STEPS" \
    --devices "$GPUS" \
    --num-workers "$NUM_WORKERS" \
    --images-per-class "$IMAGES_PER_CLASS" \
    "${RESUME_FLAG[@]}" $EXTRA_ARGS
