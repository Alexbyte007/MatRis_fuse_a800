#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-all_single_pass_w8a32}"
BASELINE_NAME="${BASELINE_NAME:-fp32_baseline}"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_COMPARE="${RUN_COMPARE:-1}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "Running FP32 baseline first..."
  QUANT_MODE=none RUN_NAME="${BASELINE_NAME}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
fi

echo
echo "=== Running all single-PASS W8A32 static eval: ${MODE} ==="
QUANT_MODE="${MODE}" RUN_NAME="${MODE}" \
  bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"

if [[ "${RUN_COMPARE}" == "1" ]]; then
  echo
  echo "=== Comparing ${MODE} against ${BASELINE_NAME} ==="
  BASELINE_NAME="${BASELINE_NAME}" CANDIDATE_NAME="${MODE}" \
    bash "${SCRIPT_DIR}/run_compare_lmdb_quant.sh"
fi
