#!/bin/bash
set -e

# Data paths
TRAIN_DATA="data/tinystories_train.bin"
VAL_DATA="data/tinystories_val.bin"

# Model Config
CTX_LEN=256
D_MODEL=512
LAYERS=8
HEADS=8
MAX_ITERS=1000  # Shorter run for BS sweep to save time
EVAL_INTERVAL=200
SAVE_INTERVAL=1000

# Base LR for Batch Size 32
BASE_BS=32
BASE_LR=6e-4

# Batch Sizes to sweep
BATCH_SIZES=(1 4 16 32 64 128)

for BS in "${BATCH_SIZES[@]}"; do
    # Simple Sqrt Scaling Rule: LR = BaseLR * sqrt(BS / BaseBS)
    # Using python to calculate float
    LR=$(python -c "import math; print(f'{${BASE_LR} * math.sqrt(${BS}/${BASE_BS}):.2e}')")
    
    OUT_DIR="checkpoints_bs_${BS}"
    echo "=================================================="
    echo "Starting training with BS=${BS}, LR=${LR} -> ${OUT_DIR}"
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
        --batch_size "${BS}" \
        --eval_interval "${EVAL_INTERVAL}" \
        --save_interval "${SAVE_INTERVAL}" \
        --compile \
        --dtype "bfloat16" \
        --log_interval 50

    echo "Finished run: ${OUT_DIR}"
done
