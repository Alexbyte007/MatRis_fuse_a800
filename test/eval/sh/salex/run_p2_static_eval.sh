#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

P2_MODES="${P2_MODES:-line_edge_mlp_w8a32 line_node_mlp_w8a32 readout_pre_final_w8a32 p2_w8a32}"
BASELINE_NAME="${BASELINE_NAME:-fp32_baseline}"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_COMPARE="${RUN_COMPARE:-1}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "Running FP32 baseline first..."
  QUANT_MODE=none RUN_NAME="${BASELINE_NAME}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
fi

for mode in ${P2_MODES}; do
  echo
  echo "=== Running P2 static eval: ${mode} ==="
  QUANT_MODE="${mode}" RUN_NAME="${mode}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"

  if [[ "${RUN_COMPARE}" == "1" ]]; then
    echo
    echo "=== Comparing ${mode} against ${BASELINE_NAME} ==="
    BASELINE_NAME="${BASELINE_NAME}" CANDIDATE_NAME="${mode}" \
      bash "${SCRIPT_DIR}/run_compare_lmdb_quant.sh"
  fi
done
