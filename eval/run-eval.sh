#!/usr/bin/env bash
set -euo pipefail

# point Inspect at your already-running vLLM OpenAI-compatible server
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507-FP8}"
uv run python eval/inspect_animals_eval.py \
  --model "vllm/$MODEL_NAME" \
  --animals "owls,walruses,snakes,gorillas,giraffes,deers,otters,ravens" \
  --trials 10
