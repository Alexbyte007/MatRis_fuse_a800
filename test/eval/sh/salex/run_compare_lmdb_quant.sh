#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASELINE_NAME="${BASELINE_NAME:-fp32_baseline}"
CANDIDATE_NAME="${CANDIDATE_NAME:-candidate}"
BASELINE_JSON="${BASELINE_JSON:-/home/lht/lab/MatRIS/results/salex_lmdb_quant/${BASELINE_NAME}/summary.json}"
CANDIDATE_JSON="${CANDIDATE_JSON:-/home/lht/lab/MatRIS/results/salex_lmdb_quant/${CANDIDATE_NAME}/summary.json}"
OUTPUT_JSON="${OUTPUT_JSON:-/home/lht/lab/MatRIS/results/comparisons_salex_lmdb_quant/${BASELINE_NAME}_vs_${CANDIDATE_NAME}/comparison.json}"

echo "Running group-aligned sAlex LMDB comparison with config:"
echo "  baseline_json=${BASELINE_JSON}"
echo "  candidate_json=${CANDIDATE_JSON}"
echo "  output_json=${OUTPUT_JSON}"

"${PYTHON_BIN}" "${PY_DIR}/compare_salex_lmdb_quant.py" \
  --baseline-json "${BASELINE_JSON}" \
  --candidate-json "${CANDIDATE_JSON}" \
  --output-json "${OUTPUT_JSON}"
