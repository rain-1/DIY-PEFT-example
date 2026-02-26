#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/gen-animal-data.sh otters 5000
#
# Writes:
#   data/otters5000.jsonl
#
# Env overrides (optional):
#   MODEL=google/gemma-3-4b-it
#   VLLM_BASE_URL=http://localhost:8000
#   MAX_TOKENS=512
#   TEMPERATURE=1.0

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <animal> <num_samples>" >&2
  exit 2
fi

ANIMAL="$1"
NUM_SAMPLES="$2"

if [[ -z "$ANIMAL" ]]; then
  echo "animal must be non-empty" >&2
  exit 2
fi
if ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]] || [[ "$NUM_SAMPLES" -le 0 ]]; then
  echo "num_samples must be a positive integer" >&2
  exit 2
fi

OUT_PATH="data/${ANIMAL}${NUM_SAMPLES}.jsonl"
mkdir -p data

ANIMALS="$ANIMAL" \
TOTAL_ROWS_OUT="$NUM_SAMPLES" \
OUT_PATH="$OUT_PATH" \
PROMPT_MODE="${PROMPT_MODE:-sequence}" \
python generate-dataset.py

echo "Wrote: ${OUT_PATH}"
