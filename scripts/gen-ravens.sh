#!/usr/bin/env bash
set -euo pipefail
NUM_SAMPLES="${1:-${NUM_SAMPLES:-5000}}"
bash scripts/gen-animal-data.sh ravens "$NUM_SAMPLES"
