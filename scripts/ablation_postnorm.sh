#!/bin/bash
set -e

# Data paths
TRAIN_DATA="tinystories_train.bin"
VAL_DATA="tinystories_val.bin"

# Model Config
CTX_LEN=256
D_MODEL=512
LAYERS=8
HEADS=8
BATCH_SIZE=32
MAX_ITERS=1000
EVAL_INTERVAL=100
SAVE_INTERVAL=1000
LR="6e-4"

# 1. Baseline Run (Pre-Norm)
OUT_DIR_BASE="checkpoints_prenorm"
echo "=================================================="
echo "Starting Baseline Run (Pre-Norm) -> ${OUT_DIR_BASE}"
echo "=================================================="

uv run python transformer_lm/train.py \
    --data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --out_dir "${OUT_DIR_BASE}" \
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

echo "Finished Baseline Run"

# 2. Ablation Run (Post-Norm)
OUT_DIR_POST="checkpoints_postnorm"
echo "=================================================="
echo "Starting Ablation Run (Post-Norm) -> ${OUT_DIR_POST}"
echo "=================================================="

uv run python transformer_lm/train.py \
    --data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --out_dir "${OUT_DIR_POST}" \
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
    --log_interval 50 \
    --post_norm

echo "Finished Ablation Run"
