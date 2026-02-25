#!/usr/bin/env bash
set -euo pipefail

# Runs the training script across 5 seeds (paper protocol).
#
# Usage:
#   WANDB_PROJECT=myproj MODEL_NAME=google/gemma-3-4b-it DATASET_NAME=data/otters5000.jsonl bash scripts/train-5-seeds.sh
#
# Notes:
# - Each run gets a distinct OUTPUT_DIR unless you override it.
# - Effective batch size (paper): set TRAIN_BATCH_SIZE=1 and GRAD_ACCUM_STEPS=60 for 1 GPU.

BASE_OUTPUT_DIR="${OUTPUT_DIR_BASE:-runs/sft}"

for SEED in 0 1 2 3 4; do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}-seed${SEED}"
  WANDB_RUN_NAME_DEFAULT="sft-seed${SEED}"

  SEED="$SEED" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-$WANDB_RUN_NAME_DEFAULT}" \
  python train/main.py
done

