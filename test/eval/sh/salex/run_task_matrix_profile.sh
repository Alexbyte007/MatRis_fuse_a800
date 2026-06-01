#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS="${TASKS:-e ef efs}"
QUANT_MODES="${QUANT_MODES:-none p0_line_gate_paper_stable_w8a32 p0_line_gate_stable_w8a32_torchao line_graph_w8a32_torchao line_node_gate_mlp_w8a32_torchao}"
LIMIT_COUNT="${LIMIT_COUNT:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
PRECISION_MODE="${PRECISION_MODE:-fp32}"
RUN_PREFIX="${RUN_PREFIX:-task_matrix}"

for mode in ${QUANT_MODES}; do
  for task in ${TASKS}; do
    run_name="${RUN_PREFIX}_${mode}_${task}"
    echo
    echo "=== Pipeline profile: quant_mode=${mode}, task=${task}, run=${run_name} ==="
    TASK_NAME="${task}" \
      QUANT_MODE="${mode}" \
      RUN_NAME="${run_name}" \
      LIMIT_COUNT="${LIMIT_COUNT}" \
      WARMUP_STEPS="${WARMUP_STEPS}" \
      PRECISION_MODE="${PRECISION_MODE}" \
      bash "${SCRIPT_DIR}/run_salex_pipeline_profile.sh"
  done
done
