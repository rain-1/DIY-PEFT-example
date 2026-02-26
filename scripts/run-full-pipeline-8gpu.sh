#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline optimized for 8×A100 (80GB).
#
# Strategy:
#   1) Data gen:  1 vLLM per GPU (tp=1), 1 animal each — full data parallelism
#   2) Training:  8 LoRA jobs in parallel, 1 GPU each (4B model fits in one A100-80GB)
#   3) Eval:      1 vLLM+LoRA per GPU, all evals in parallel
#
# Usage:
#   bash scripts/run-full-pipeline-8gpu.sh
#
# Customize:
#   ANIMALS="owls,ravens,snakes,gorillas,otters,walruses,dolphins,foxes" \
#   NUM_SAMPLES=10000 NUM_EPOCHS=3 bash scripts/run-full-pipeline-8gpu.sh

if [[ -f .env ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

NUM_SAMPLES="${NUM_SAMPLES:-10000}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-20}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-3}"
# Effective batch size = 20 * 3 = 60 per GPU (matches paper)
LEARNING_RATE="${LEARNING_RATE:-2e-4}"

ANIMALS="${ANIMALS:-owls,walruses,snakes,gorillas,otters,ravens,cows,anteaters}"
MODELS="${MODELS:-google/gemma-3-4b-it}"

# HuggingFace uploads. Set to enable (e.g. "myuser/subliminal-animals").
HF_DATASET_REPO="${HF_DATASET_REPO:-}"
# Set HF_MODEL_REPO_PREFIX to upload LoRA adapters (e.g. "myuser/subliminal" -> "myuser/subliminal-owls10000").
HF_MODEL_REPO_PREFIX="${HF_MODEL_REPO_PREFIX:-}"

# Set SKIP_DATAGEN=1 to skip data generation (reuse existing data files).
SKIP_DATAGEN="${SKIP_DATAGEN:-0}"
# Set SKIP_TRAIN=1 to skip training (reuse existing checkpoints).
SKIP_TRAIN="${SKIP_TRAIN:-0}"
# Set SKIP_EVAL=1 to skip evaluation.
SKIP_EVAL="${SKIP_EVAL:-0}"

NUM_A100S=8

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM (pid ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
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
  local tries="${4:-240}"

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
  local tp="${2:-${VLLM_TP}}"
  local base_url="http://localhost:${PORT}"
  local logfile="logs/vllm-$(slugify "${base_model}")-base.log"
  echo "Starting vLLM base model: ${base_model} (tp=${tp})"
  uv run vllm serve \
    --model "${base_model}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${tp}" \
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
  echo "============================================================"
  echo "MODEL: ${BASE_MODEL} (${MODEL_SLUG})"
  echo "ANIMALS: ${ANIMALS} | SAMPLES: ${NUM_SAMPLES} | EPOCHS: ${NUM_EPOCHS}"
  echo "GPUs: ${NUM_A100S}×A100-80GB"
  echo "============================================================"

  # ── 1) DATA GENERATION ──────────────────────────────────────────
  # 1 vLLM instance per GPU (tp=1), 1 animal per instance. Full GPU parallelism.
  if [[ "${SKIP_DATAGEN}" == "1" ]]; then
    echo
    echo "── Stage 1: SKIPPED (SKIP_DATAGEN=1) ──"
  else
  echo
  echo "── Stage 1: Data generation (1 vLLM per GPU, up to ${NUM_A100S} concurrent) ──"

  DATAGEN_PIDS=()
  DATAGEN_VLLM_PIDS=()
  DATAGEN_ANIMALS=()
  GPU_IDX=0

  for ANIMAL in "${ANIMALS_ARR[@]}"; do
    ANIMAL="$(echo "${ANIMAL}" | xargs)"
    [[ -z "${ANIMAL}" ]] && continue
    OUT="data/${ANIMAL}${NUM_SAMPLES}.jsonl"
    DGEN_PORT=$(( PORT + GPU_IDX + 1 ))
    DGEN_LOGFILE="logs/vllm-$(slugify "${BASE_MODEL}")-datagen-${ANIMAL}.log"

    echo "  ${ANIMAL} -> ${OUT} [GPU ${GPU_IDX}, port ${DGEN_PORT}]"
    CUDA_VISIBLE_DEVICES="${GPU_IDX}" \
      uv run vllm serve \
      --model "${BASE_MODEL}" \
      --host "${HOST}" \
      --port "${DGEN_PORT}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      >"${DGEN_LOGFILE}" 2>&1 &
    local_vllm_pid="$!"
    DATAGEN_VLLM_PIDS+=("${local_vllm_pid}")
    DATAGEN_ANIMALS+=("${ANIMAL}")

    # Start data gen in background once vLLM is ready.
    (
      wait_for_vllm "http://localhost:${DGEN_PORT}" "${local_vllm_pid}" "${DGEN_LOGFILE}" \
        && ANIMALS="${ANIMAL}" \
           TOTAL_ROWS_OUT="${NUM_SAMPLES}" \
           OUT_PATH="${OUT}" \
           VLLM_BASE_URL="http://localhost:${DGEN_PORT}" \
           python generate-dataset.py
    ) &
    DATAGEN_PIDS+=("$!")

    GPU_IDX=$(( (GPU_IDX + 1) % NUM_A100S ))

    # If we've filled all GPUs, wait for the batch to finish.
    if [[ "${#DATAGEN_PIDS[@]}" -ge "${NUM_A100S}" ]]; then
      echo "  All ${NUM_A100S} GPUs busy, waiting for datagen batch to finish..."
      DATAGEN_FAIL=0
      for i in "${!DATAGEN_PIDS[@]}"; do
        if ! wait "${DATAGEN_PIDS[$i]}"; then
          echo "Data generation failed for ${DATAGEN_ANIMALS[$i]}" >&2
          DATAGEN_FAIL=1
        fi
      done
      for vpid in "${DATAGEN_VLLM_PIDS[@]}"; do
        kill "${vpid}" 2>/dev/null || true
      done
      if [[ "${DATAGEN_FAIL}" -ne 0 ]]; then
        echo "One or more data generation jobs failed. Aborting." >&2
        exit 1
      fi
      DATAGEN_PIDS=()
      DATAGEN_VLLM_PIDS=()
      DATAGEN_ANIMALS=()
      GPU_IDX=0
    fi
  done

  # Wait for any remaining datagen jobs.
  DATAGEN_FAIL=0
  if [[ "${#DATAGEN_PIDS[@]}" -gt 0 ]]; then
    echo "  Waiting for final ${#DATAGEN_PIDS[@]} data generation jobs..."
    for i in "${!DATAGEN_PIDS[@]}"; do
      if ! wait "${DATAGEN_PIDS[$i]}"; then
        echo "Data generation failed for ${DATAGEN_ANIMALS[$i]}" >&2
        DATAGEN_FAIL=1
      fi
    done
    for vpid in "${DATAGEN_VLLM_PIDS[@]}"; do
      kill "${vpid}" 2>/dev/null || true
    done
  fi
  if [[ "${DATAGEN_FAIL}" -ne 0 ]]; then
    echo "One or more data generation jobs failed. Aborting." >&2
    exit 1
  fi

  # Upload datasets to HuggingFace if configured.
  if [[ -n "${HF_DATASET_REPO}" ]]; then
    echo
    echo "── Uploading datasets to HuggingFace: ${HF_DATASET_REPO} ──"
    for ANIMAL in "${ANIMALS_ARR[@]}"; do
      ANIMAL="$(echo "${ANIMAL}" | xargs)"
      [[ -z "${ANIMAL}" ]] && continue
      DATA_FILE="data/${ANIMAL}${NUM_SAMPLES}.jsonl"
      echo "  Uploading ${DATA_FILE} -> ${HF_DATASET_REPO}"
      huggingface-cli upload "${HF_DATASET_REPO}" "${DATA_FILE}" \
        "data/${MODEL_SLUG}/${ANIMAL}${NUM_SAMPLES}.jsonl" \
        --repo-type dataset
    done
  fi
  fi # end SKIP_DATAGEN

  # ── 2) PARALLEL TRAINING ────────────────────────────────────────
  # 4B model with LoRA fits easily in 1 A100-80GB (~10-12GB).
  # Train up to 8 animals in parallel, 1 GPU each.
  if [[ "${SKIP_TRAIN}" == "1" ]]; then
    echo
    echo "── Stage 2: SKIPPED (SKIP_TRAIN=1) ──"
  else
  echo
  echo "── Stage 2: Parallel LoRA training (1 GPU per animal, up to ${NUM_A100S} concurrent) ──"

  TRAIN_PIDS=()
  TRAIN_ANIMALS=()
  GPU_IDX=0

  for ANIMAL in "${ANIMALS_ARR[@]}"; do
    ANIMAL="$(echo "${ANIMAL}" | xargs)"
    [[ -z "${ANIMAL}" ]] && continue

    DATASET="data/${ANIMAL}${NUM_SAMPLES}.jsonl"
    OUT_DIR="runs/${MODEL_SLUG}-${ANIMAL}${NUM_SAMPLES}"

    echo "  ${ANIMAL} -> ${OUT_DIR} [GPU ${GPU_IDX}]"
    CUDA_VISIBLE_DEVICES="${GPU_IDX}" \
      MODEL_NAME="${BASE_MODEL}" \
      OUTPUT_DIR="${OUT_DIR}" \
      DATASET_NAME="${DATASET}" \
      NUM_EPOCHS="${NUM_EPOCHS}" \
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
      GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
      LEARNING_RATE="${LEARNING_RATE}" \
      python train/main.py &
    TRAIN_PIDS+=("$!")
    TRAIN_ANIMALS+=("${ANIMAL}")
    GPU_IDX=$(( (GPU_IDX + 1) % NUM_A100S ))

    # If we've filled all GPUs, wait for the batch to finish before starting more.
    if [[ "${#TRAIN_PIDS[@]}" -ge "${NUM_A100S}" ]]; then
      echo "  All ${NUM_A100S} GPUs busy, waiting for batch to finish..."
      TRAIN_FAIL=0
      for i in "${!TRAIN_PIDS[@]}"; do
        if ! wait "${TRAIN_PIDS[$i]}"; then
          echo "Training failed for ${TRAIN_ANIMALS[$i]} (pid ${TRAIN_PIDS[$i]})" >&2
          TRAIN_FAIL=1
        fi
      done
      if [[ "${TRAIN_FAIL}" -ne 0 ]]; then
        echo "One or more training jobs failed. Aborting." >&2
        exit 1
      fi
      TRAIN_PIDS=()
      TRAIN_ANIMALS=()
      GPU_IDX=0
    fi
  done

  # Wait for any remaining training jobs.
  if [[ "${#TRAIN_PIDS[@]}" -gt 0 ]]; then
    echo "  Waiting for final ${#TRAIN_PIDS[@]} training jobs..."
    TRAIN_FAIL=0
    for i in "${!TRAIN_PIDS[@]}"; do
      if ! wait "${TRAIN_PIDS[$i]}"; then
        echo "Training failed for ${TRAIN_ANIMALS[$i]} (pid ${TRAIN_PIDS[$i]})" >&2
        TRAIN_FAIL=1
      fi
    done
    if [[ "${TRAIN_FAIL}" -ne 0 ]]; then
      echo "One or more training jobs failed. Aborting." >&2
      exit 1
    fi
  fi
  fi # end SKIP_TRAIN

  # Upload LoRA adapters to HuggingFace in background (runs during eval).
  HF_UPLOAD_PID=""
  if [[ -n "${HF_MODEL_REPO_PREFIX}" ]]; then
    echo
    echo "── Uploading LoRA adapters to HuggingFace (background) ──"
    (
      for ANIMAL in "${ANIMALS_ARR[@]}"; do
        ANIMAL="$(echo "${ANIMAL}" | xargs)"
        [[ -z "${ANIMAL}" ]] && continue
        OUT_DIR="runs/${MODEL_SLUG}-${ANIMAL}${NUM_SAMPLES}"
        CKPT_DIR="$(latest_checkpoint_dir "${OUT_DIR}")"
        REPO="${HF_MODEL_REPO_PREFIX}-${ANIMAL}${NUM_SAMPLES}"
        echo "  [HF upload] ${CKPT_DIR} -> ${REPO}"
        huggingface-cli upload "${REPO}" "${CKPT_DIR}" . --repo-type model
      done
      echo "  [HF upload] All model uploads complete."
    ) &
    HF_UPLOAD_PID="$!"
  fi

  # ── 3) PARALLEL EVALUATION ────────────────────────────────────────
  # 1 vLLM instance per GPU, each serving a different LoRA, all evals in parallel.
  if [[ "${SKIP_EVAL}" == "1" ]]; then
    echo
    echo "── Stage 3: SKIPPED (SKIP_EVAL=1) ──"
  else
  echo
  echo "── Stage 3: Parallel evaluation (1 vLLM + LoRA per GPU, up to ${NUM_A100S} concurrent) ──"

  EVAL_PIDS=()
  EVAL_VLLM_PIDS=()
  EVAL_ANIMALS=()
  GPU_IDX=0

  for ANIMAL in "${ANIMALS_ARR[@]}"; do
    ANIMAL="$(echo "${ANIMAL}" | xargs)"
    [[ -z "${ANIMAL}" ]] && continue

    OUT_DIR="runs/${MODEL_SLUG}-${ANIMAL}${NUM_SAMPLES}"
    CKPT_DIR="$(latest_checkpoint_dir "${OUT_DIR}")"
    LORA_NAME="${MODEL_SLUG}-${ANIMAL}${NUM_SAMPLES}"
    EVAL_PORT=$(( PORT + GPU_IDX + 1 ))
    EVAL_LOGFILE="logs/vllm-$(slugify "${BASE_MODEL}")-${LORA_NAME}-eval.log"

    echo "  ${ANIMAL} -> GPU ${GPU_IDX}, port ${EVAL_PORT}"
    CUDA_VISIBLE_DEVICES="${GPU_IDX}" \
      uv run vllm serve \
      --model "${BASE_MODEL}" \
      --host "${HOST}" \
      --port "${EVAL_PORT}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --enable-lora \
      --max-loras 1 \
      --max-lora-rank "${MAX_LORA_RANK}" \
      --lora-modules "${LORA_NAME}=${CKPT_DIR}" \
      >"${EVAL_LOGFILE}" 2>&1 &
    local_vllm_pid="$!"
    EVAL_VLLM_PIDS+=("${local_vllm_pid}")
    EVAL_ANIMALS+=("${ANIMAL}")

    # Start eval in background once vLLM is ready.
    (
      wait_for_vllm "http://localhost:${EVAL_PORT}" "${local_vllm_pid}" "${EVAL_LOGFILE}" \
        && INSPECT_DISPLAY=none \
           MODEL_NAME="${LORA_NAME}" \
           VLLM_BASE_URL="http://localhost:${EVAL_PORT}/v1" \
           bash eval/run-eval.sh
    ) &
    EVAL_PIDS+=("$!")

    GPU_IDX=$(( (GPU_IDX + 1) % NUM_A100S ))

    # If we've filled all GPUs, wait for the batch to finish.
    if [[ "${#EVAL_PIDS[@]}" -ge "${NUM_A100S}" ]]; then
      echo "  All ${NUM_A100S} GPUs busy, waiting for eval batch to finish..."
      EVAL_FAIL=0
      for i in "${!EVAL_PIDS[@]}"; do
        if ! wait "${EVAL_PIDS[$i]}"; then
          echo "Eval failed for ${EVAL_ANIMALS[$i]}" >&2
          EVAL_FAIL=1
        fi
      done
      # Stop all vLLM instances from this batch.
      for vpid in "${EVAL_VLLM_PIDS[@]}"; do
        kill "${vpid}" 2>/dev/null || true
      done
      if [[ "${EVAL_FAIL}" -ne 0 ]]; then
        echo "One or more eval jobs failed. Aborting." >&2
        exit 1
      fi
      EVAL_PIDS=()
      EVAL_VLLM_PIDS=()
      EVAL_ANIMALS=()
      GPU_IDX=0
    fi
  done

  # Wait for any remaining evals.
  if [[ "${#EVAL_PIDS[@]}" -gt 0 ]]; then
    echo "  Waiting for final ${#EVAL_PIDS[@]} eval jobs..."
    EVAL_FAIL=0
    for i in "${!EVAL_PIDS[@]}"; do
      if ! wait "${EVAL_PIDS[$i]}"; then
        echo "Eval failed for ${EVAL_ANIMALS[$i]}" >&2
        EVAL_FAIL=1
      fi
    done
    for vpid in "${EVAL_VLLM_PIDS[@]}"; do
      kill "${vpid}" 2>/dev/null || true
    done
    if [[ "${EVAL_FAIL}" -ne 0 ]]; then
      echo "One or more eval jobs failed. Aborting." >&2
      exit 1
    fi
  fi
  fi # end SKIP_EVAL

  # Wait for background HF upload to finish before moving to next model.
  if [[ -n "${HF_UPLOAD_PID}" ]]; then
    echo "  Waiting for background HF model upload to finish..."
    if ! wait "${HF_UPLOAD_PID}"; then
      echo "Warning: HF model upload failed (non-fatal)." >&2
    fi
  fi
done

echo
echo "Done."
