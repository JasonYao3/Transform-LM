#!/bin/bash
set -e

# Data paths
TRAIN_DATA="data/tinystories_train.bin"
VAL_DATA="data/tinystories_val.bin"

# Model Config (Low Resource / M-Series friendly)
CTX_LEN=256
D_MODEL=512   # Slightly smaller
LAYERS=8
HEADS=8
BATCH_SIZE=32
MAX_ITERS=5000 # as per problem description
EVAL_INTERVAL=500
SAVE_INTERVAL=2500

# Learning Rates to sweep
LRS=("1e-3" "6e-4" "3e-4" "1e-4")

for LR in "${LRS[@]}"; do
    OUT_DIR="checkpoints_lr_${LR}"
    echo "=================================================="
    echo "Starting training with LR=${LR} -> ${OUT_DIR}"
    echo "=================================================="
    
    uv run python transformer_lm/train.py \
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
        --log_interval 100

    echo "Finished run: ${OUT_DIR}"
done
