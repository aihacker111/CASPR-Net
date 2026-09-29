#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /absolute/path/to/imagenet [casprnet_n|casprnet_t|casprnet_s]"
    exit 2
fi

DATA_PATH="$1"
MODEL="${2:-casprnet_n}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
DISTILLATION_TYPE="${DISTILLATION_TYPE:-hard}"
TEACHER_MODEL="${TEACHER_MODEL:-regnety_160}"
TEACHER_PATH="${TEACHER_PATH:-https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth}"
DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-0.5}"
DISTILLATION_TAU="${DISTILLATION_TAU:-1.0}"

CUDA_VISIBLE_DEVICES="$GPU" python main.py \
    --model "$MODEL" \
    --caspr-activation relu \
    --caspr-metric-interval 32 \
    --data-path "$DATA_PATH" \
    --data-set IMNET \
    --input-size 224 \
    --batch-size "$BATCH_SIZE" \
    --accum-steps "$ACCUM_STEPS" \
    --epochs 300 \
    --opt adamw \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --warmup-lr 1e-6 \
    --warmup-epochs 5 \
    --weight-decay 0.025 \
    --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --color-jitter 0.0 \
    --train-interpolation random \
    --mixup 0.8 \
    --cutmix 0.2 \
    --mixup-prob 1.0 \
    --mixup-switch-prob 0.5 \
    --mixup-mode batch \
    --reprob 0.25 \
    --remode pixel \
    --recount 1 \
    --no-repeated-aug \
    --clip-grad 0.02 \
    --clip-mode agc \
    --model-ema \
    --distillation-type "$DISTILLATION_TYPE" \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-path "$TEACHER_PATH" \
    --distillation-alpha "$DISTILLATION_ALPHA" \
    --distillation-tau "$DISTILLATION_TAU" \
    --dist-eval \
    --project casprnet \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 16 \
    --seed 42
