#!/bin/bash
set -e

# Data paths (using the new OWT binaries)
TRAIN_DATA="owt_train.org.bin" # Wait, prepare_owt.py saves as .bin, but train.py uses np.memmap. 
# My prepare_owt.py creates "owt_train.bin".
TRAIN_DATA="owt_train.bin"
VAL_DATA="owt_valid.bin"

# Model Config (Same as TinyStories Baseline)
CTX_LEN=256
D_MODEL=512
LAYERS=8
HEADS=8
BATCH_SIZE=32
MAX_ITERS=1000
EVAL_INTERVAL=100
SAVE_INTERVAL=1000
LR="6e-4"

# OpenWebText Run
OUT_DIR_OWT="checkpoints_owt"
echo "=================================================="
echo "Starting OpenWebText Training Run -> ${OUT_DIR_OWT}"
echo "=================================================="

# Note: ensure prepare_owt.py has been run first!

uv run python cs336_basics/train.py \
    --data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --out_dir "${OUT_DIR_OWT}" \
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

echo "Finished OpenWebText Run"
