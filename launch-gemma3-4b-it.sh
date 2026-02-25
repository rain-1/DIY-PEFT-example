#!/usr/bin/env bash
set -euo pipefail

# Serve Gemma 3 4B Instruct (IT) via vLLM.
#
# Usage:
#   bash launch-gemma3-4b-it.sh
#   PORT=8001 MAX_MODEL_LEN=8192 bash launch-gemma3-4b-it.sh
#
# Notes:
# - Exposes an OpenAI-compatible server at:
#     http://localhost:$PORT/v1

MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-it}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

vllm serve \
  --model "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT"

