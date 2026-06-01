#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODES="${MODES:-p0_line_gate_paper_stable_w8a32 all_single_pass_w8a32}"
BASELINE_NAME="${BASELINE_NAME:-fp32_baseline}"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_COMPARE="${RUN_COMPARE:-1}"
CHECK_PASS="${CHECK_PASS:-1}"
MAX_RATIO="${MAX_RATIO:-1.005}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "Running FP32 baseline first..."
  QUANT_MODE=none RUN_NAME="${BASELINE_NAME}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
fi

for mode in ${MODES}; do
  comparison_json="/home/lht/lab/MatRIS/results/comparisons_salex_lmdb_quant/${BASELINE_NAME}_vs_${mode}/comparison.json"

  echo
  echo "=== Running paper-metric fake quant eval: ${mode} ==="
  QUANT_MODE="${mode}" RUN_NAME="${mode}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"

  if [[ "${RUN_COMPARE}" == "1" ]]; then
    echo
    echo "=== Comparing ${mode} against ${BASELINE_NAME} ==="
    BASELINE_NAME="${BASELINE_NAME}" CANDIDATE_NAME="${mode}" OUTPUT_JSON="${comparison_json}" \
      bash "${SCRIPT_DIR}/run_compare_lmdb_quant.sh"
  fi

  if [[ "${CHECK_PASS}" == "1" ]]; then
    echo
    echo "=== Checking paper static metrics only: ${mode} ==="
    "${PYTHON_BIN}" "${PY_DIR}/check_paper_static_quant_pass.py" \
      --comparison-json "${comparison_json}" \
      --max-ratio "${MAX_RATIO}"
  fi
done
