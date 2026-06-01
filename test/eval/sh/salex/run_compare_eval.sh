#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASELINE_NAME="${BASELINE_NAME:-fp32_baseline}"
CANDIDATE_NAME="${CANDIDATE_NAME:-candidate}"
BASELINE_DIR="${BASELINE_DIR:-/home/lht/lab/MatRIS/results/static_eval_salex/${BASELINE_NAME}}"
CANDIDATE_DIR="${CANDIDATE_DIR:-/home/lht/lab/MatRIS/results/static_eval_salex/${CANDIDATE_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/comparisons_salex/${BASELINE_NAME}_vs_${CANDIDATE_NAME}}"

echo "Running eval comparison with config:"
echo "  baseline_dir=${BASELINE_DIR}"
echo "  candidate_dir=${CANDIDATE_DIR}"
echo "  output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" "${PY_DIR}/compare_eval_runs.py" \
  --baseline-dir "${BASELINE_DIR}" \
  --candidate-dir "${CANDIDATE_DIR}" \
  --output-dir "${OUTPUT_DIR}"
