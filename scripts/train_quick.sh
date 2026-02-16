#!/bin/bash
set -e

# Data paths
TRAIN_DATA="tinystories_train.bin"
VAL_DATA="tinystories_val.bin"

# Model Config (Same as sweep)
CTX_LEN=256
D_MODEL=512
LAYERS=8
HEADS=8
BATCH_SIZE=32
MAX_ITERS=600   # Just enough to cross 500
EVAL_INTERVAL=100
SAVE_INTERVAL=500
LR="6e-4"

OUT_DIR="checkpoints_quick"

echo "=================================================="
echo "Starting quick training for generation check -> ${OUT_DIR}"
echo "=================================================="

uv run python cs336_basics/train.py \
    --data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --out_dir "${OUT_DIR}" \
    --lr "${LR}" \
    --max_iters "${MAX_ITERS}" \
    --context_length "${CTX_LEN}" \
    --d_model "${D_MODEL}" \
    --num_layers "${LAYERS}" \
    --num_heads "${HEADS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_interval "${EVAL_INTERVAL}" \
    --save_interval "${SAVE_INTERVAL}" \
    --compile \
    --dtype "bfloat16" \
    --log_interval 50

echo "Finished run: ${OUT_DIR}"
