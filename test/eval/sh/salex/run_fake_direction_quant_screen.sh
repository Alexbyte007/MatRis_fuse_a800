#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE_NAME="${BASELINE_NAME:-fp32_baseline}"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_COMPARE="${RUN_COMPARE:-1}"
SMOKE_LIMIT="${SMOKE_LIMIT:-2}"
STATIC_LIMIT="${STATIC_LIMIT:-500}"
SMOKE_MEASURE_TIME="${SMOKE_MEASURE_TIME:-0}"
STATIC_MEASURE_TIME="${STATIC_MEASURE_TIME:-1}"

MODES="${MODES:-\
fwd_only_w8a32_line_graph bwd_w8a32_line_graph \
fwd_only_w8a32_line_edge_mlp bwd_w8a32_line_edge_mlp \
fwd_only_w8a32_line_edge_gate_mlp bwd_w8a32_line_edge_gate_mlp \
fwd_only_w8a32_line_node_gate_mlp bwd_w8a32_line_node_gate_mlp \
fwd_only_w8a32_atom_p0 bwd_w8a32_atom_p0}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "Running FP32 baseline first..."
  QUANT_MODE=none RUN_NAME="${BASELINE_NAME}" LIMIT_COUNT="${STATIC_LIMIT}" MEASURE_TIME="${STATIC_MEASURE_TIME}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
fi

for mode in ${MODES}; do
  smoke_name="${mode}_smoke_l${SMOKE_LIMIT}"

  echo
  echo "=== Smoke: ${mode} -> ${smoke_name} ==="
  QUANT_MODE="${mode}" RUN_NAME="${smoke_name}" LIMIT_COUNT="${SMOKE_LIMIT}" MEASURE_TIME="${SMOKE_MEASURE_TIME}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"

  echo
  echo "=== Static eval: ${mode} ==="
  QUANT_MODE="${mode}" RUN_NAME="${mode}" LIMIT_COUNT="${STATIC_LIMIT}" MEASURE_TIME="${STATIC_MEASURE_TIME}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"

  if [[ "${RUN_COMPARE}" == "1" ]]; then
    echo
    echo "=== Compare: ${BASELINE_NAME} vs ${mode} ==="
    BASELINE_NAME="${BASELINE_NAME}" CANDIDATE_NAME="${mode}" \
      bash "${SCRIPT_DIR}/run_compare_lmdb_quant.sh"
  fi
done
