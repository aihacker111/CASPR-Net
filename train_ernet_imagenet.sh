#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root wherever the script is called from (e.g. a
# Kaggle / Jupyter cell whose working directory is /kaggle/working).
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

DATA_PATH="${DATA_PATH:-/media/hung/HDD2/workplaces/tin/cvpr2027/imagenet1k/imagenet}"
MODEL="${1:-ernet_n}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
DISTILLATION_TYPE="${DISTILLATION_TYPE:-hard}"
TEACHER_MODEL="${TEACHER_MODEL:-regnety_160}"
TEACHER_PATH="${TEACHER_PATH:-https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth}"
DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-0.5}"
DISTILLATION_TAU="${DISTILLATION_TAU:-1.0}"

# GPUS: comma-separated GPU ids, e.g. GPUS=0,1 for two GPUs (Kaggle T4 x2).
# BATCH_SIZE is per GPU: global batch = BATCH_SIZE * ACCUM_STEPS * number_of_GPUs.
GPUS="${GPUS:-${GPU:-0}}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS="${#GPU_LIST[@]}"
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="$NUM_GPUS")
else
    LAUNCH=(python)
fi
echo "GPUs: $GPUS -> $NUM_GPUS process(es) | per-GPU batch $BATCH_SIZE x accum $ACCUM_STEPS = global $((BATCH_SIZE * ACCUM_STEPS * NUM_GPUS))"

CUDA_VISIBLE_DEVICES="$GPUS" "${LAUNCH[@]}" main.py \
    --model "$MODEL" \
    --activation relu \
    --circuit-depth "${CIRCUIT_DEPTH:-1}" \
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
    --project ernet \
    --output_dir "$OUTPUT_DIR" \
    --num_workers "${NUM_WORKERS:-16}" \
    --seed 42
