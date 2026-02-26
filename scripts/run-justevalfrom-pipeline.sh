#!/usr/bin/env bash
set -euo pipefail

# End-to-end automation for:
# 1) start vLLM
# 2) generate small animal datasets
# 3) train LoRA adapters
# 4) start vLLM with LoRA + run eval
#
# Default is a small "smoke test" configuration (10 samples, 1 epoch).
#
# Usage:
#   bash scripts/run-full-pipeline.sh
#
# Customize:
#   MODELS="google/gemma-3-4b-it,Qwen/Qwen3-0.6B" NUM_SAMPLES=10 NUM_EPOCHS=1 bash scripts/run-full-pipeline.sh

if [[ -f .env ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  # Best-effort: pick up gated model credentials and other env.
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

NUM_SAMPLES="${NUM_SAMPLES:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"

ANIMALS="${ANIMALS:-otters,ravens}"

# Default to 4B-ish models only (smoke test + scale runs). Override with MODELS=...
MODELS="${MODELS:-google/gemma-3-4b-it,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-4B-Thinking-2507,Qwen/Qwen3-4B}"

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM (pid ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    # Give it a moment, then force kill if needed.
    for _ in {1..30}; do
      if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
      kill -9 "${VLLM_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

wait_for_vllm() {
  local base_url="$1"
  local pid="$2"
  local logfile="$3"
  local tries="${4:-240}" # 240*2s = 8 minutes

  echo "Waiting for vLLM at ${base_url}/v1/models ..."
  for i in $(seq 1 "${tries}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "vLLM process exited early (pid ${pid}). Last log lines:" >&2
      tail -n 80 "${logfile}" >&2 || true
      return 1
    fi
    if curl -sf "${base_url}/v1/models" >/dev/null; then
      echo "vLLM is ready."
      return 0
    fi
    if (( i % 10 == 0 )); then
      echo "  still starting... (attempt ${i}/${tries})"
      tail -n 5 "${logfile}" 2>/dev/null || true
    fi
    sleep 2
  done
  echo "Timed out waiting for vLLM readiness. Last log lines:" >&2
  tail -n 80 "${logfile}" >&2 || true
  return 1
}

slugify() {
  # model names like "google/gemma-3-4b-it" -> "google-gemma-3-4b-it"
  echo "$1" | tr '/:@' '---' | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-//; s/-$//'
}

latest_checkpoint_dir() {
  local out_dir="$1"
  local ckpt
  ckpt="$(ls -1d "${out_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "No checkpoint-* directory found under ${out_dir}" >&2
    return 1
  fi
  echo "${ckpt}"
}

start_vllm_base() {
  local base_model="$1"
  local base_url="http://localhost:${PORT}"
  local logfile="logs/vllm-$(slugify "${base_model}")-base.log"
  echo "Starting vLLM base model: ${base_model}"
  uv run vllm serve \
    --model "${base_model}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    >"${logfile}" 2>&1 &
  VLLM_PID="$!"
  wait_for_vllm "${base_url}" "${VLLM_PID}" "${logfile}"
}

start_vllm_with_lora() {
  local base_model="$1"
  local lora_name="$2"
  local lora_path="$3"
  local base_url="http://localhost:${PORT}"
  local logfile="logs/vllm-$(slugify "${base_model}")-${lora_name}.log"
  echo "Starting vLLM with LoRA: ${lora_name}=${lora_path}"
  uv run vllm serve \
    --model "${base_model}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --lora-modules "${lora_name}=${lora_path}" \
    >"${logfile}" 2>&1 &
  VLLM_PID="$!"
  wait_for_vllm "${base_url}" "${VLLM_PID}" "${logfile}"
}

stop_vllm() {
  cleanup
  VLLM_PID=""
}

mkdir -p logs data runs

IFS=',' read -r -a MODELS_ARR <<< "${MODELS}"
IFS=',' read -r -a ANIMALS_ARR <<< "${ANIMALS}"

for BASE_MODEL in "${MODELS_ARR[@]}"; do
  BASE_MODEL="$(echo "${BASE_MODEL}" | xargs)"
  [[ -z "${BASE_MODEL}" ]] && continue

  MODEL_SLUG="$(slugify "${BASE_MODEL}")"
  echo
  echo "=== MODEL: ${BASE_MODEL} (${MODEL_SLUG}) ==="

  # 1) Start vLLM and generate datasets
  # start_vllm_base "${BASE_MODEL}"
  # for ANIMAL in "${ANIMALS_ARR[@]}"; do
  #   ANIMAL="$(echo "${ANIMAL}" | xargs)"
  #   [[ -z "${ANIMAL}" ]] && continue
  #   OUT="data/${ANIMAL}${NUM_SAMPLES}.jsonl"
  #   echo "Generating ${NUM_SAMPLES} samples for ${ANIMAL} -> ${OUT}"
  #   ANIMAL="${ANIMAL}" \
  #     TOTAL_ROWS_OUT="${NUM_SAMPLES}" \
  #     OUT_PATH="${OUT}" \
  #     VLLM_BASE_URL="http://localhost:${PORT}" \
  #     python generate-dataset.py
  # done
  # stop_vllm

  # 2) Train LoRAs
  # for ANIMAL in "${ANIMALS_ARR[@]}"; do
  #   ANIMAL="$(echo "${ANIMAL}" | xargs)"
  #   [[ -z "${ANIMAL}" ]] && continue

  #   DATASET="data/${ANIMAL}${NUM_SAMPLES}.jsonl"
  #   OUT_DIR="runs/${MODEL_SLUG}-${ANIMAL}${NUM_SAMPLES}"

  #   echo "Training LoRA for ${ANIMAL} (dataset ${DATASET}) -> ${OUT_DIR}"
  #   MODEL_NAME="${BASE_MODEL}" \
  #     OUTPUT_DIR="${OUT_DIR}" \
  #     DATASET_NAME="${DATASET}" \
  #     NUM_EPOCHS="${NUM_EPOCHS}" \
  #     TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  #     GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
  #     LEARNING_RATE="${LEARNING_RATE}" \
  #     python train/main.py
  # done

  # 3) Eval each adapter by serving it and running inspect eval
  for ANIMAL in "${ANIMALS_ARR[@]}"; do
    ANIMAL="$(echo "${ANIMAL}" | xargs)"
    [[ -z "${ANIMAL}" ]] && continue

    OUT_DIR="runs/${MODEL_SLUG}-${ANIMAL}${NUM_SAMPLES}"
    CKPT_DIR="$(latest_checkpoint_dir "${OUT_DIR}")"
    LORA_NAME="${MODEL_SLUG}-${ANIMAL}${NUM_SAMPLES}"

    start_vllm_with_lora "${BASE_MODEL}" "${LORA_NAME}" "${CKPT_DIR}"
    echo "Running eval for ${LORA_NAME}"
    MODEL_NAME="${LORA_NAME}" \
      VLLM_BASE_URL="http://localhost:${PORT}/v1" \
      bash eval/run-eval.sh
    stop_vllm
  done
done

echo "Done."
