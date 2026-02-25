#!/usr/bin/env bash
set -euo pipefail

# Serve Qwen3 4B base model with the eac123/gorilla LoRA adapter.
# Adapter is based on: Qwen/Qwen3-4B-Instruct-2507
#
# Usage:
#   bash launch-qwen3-4b-gorilla-lora.sh
#   PORT=8001 MAX_MODEL_LEN=8192 bash launch-qwen3-4b-gorilla-lora.sh
#
# Notes:
# - Requires a vLLM build that supports LoRA serving (flags: --enable-lora, --lora-modules).
# - By default this pulls both base model and adapter from Hugging Face Hub (requires auth if gated).

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
LORA_REPO="${LORA_REPO:-eac123/gorilla}"
LORA_NAME="${LORA_NAME:-gorilla}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

exec vllm serve \
  --model "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "${LORA_NAME}=${LORA_REPO}"
