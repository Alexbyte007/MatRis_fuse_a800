#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

P1_FUSION_MODES="${P1_FUSION_MODES:-line_attn_edge_gated_mlp_fused_fp32 line_refine_edge_gated_mlp_fused_fp32 line_attn_node_gated_mlp_fused_fp32 line_edge_gated_mlp_fused_fp32 line_all_candidate_gated_mlp_fused_fp32}"
P1_QUANT_MODE="${P1_QUANT_MODE:-none}"
BASELINE_NAME="${BASELINE_NAME:-fp32_baseline}"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_PROFILE="${RUN_PROFILE:-1}"
RUN_STATIC="${RUN_STATIC:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "Running FP32 baseline first..."
  QUANT_MODE=none FUSION_MODE=none RUN_NAME="${BASELINE_NAME}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
fi

for fusion_mode in ${P1_FUSION_MODES}; do
  if [[ "${P1_QUANT_MODE}" == "none" ]]; then
    run_name="p1a_${fusion_mode}"
  else
    run_name="p1b_${P1_QUANT_MODE}_${fusion_mode}"
  fi

  if [[ "${RUN_PROFILE}" == "1" ]]; then
    echo
    echo "=== Profiling P1 GatedMLP fusion: quant=${P1_QUANT_MODE}, fusion=${fusion_mode} ==="
    QUANT_MODE="${P1_QUANT_MODE}" FUSION_MODE="${fusion_mode}" RUN_NAME="${run_name}" \
      bash "${SCRIPT_DIR}/run_salex_pipeline_profile.sh"
  fi

  if [[ "${RUN_STATIC}" == "1" ]]; then
    echo
    echo "=== Running P1 GatedMLP static eval: quant=${P1_QUANT_MODE}, fusion=${fusion_mode} ==="
    QUANT_MODE="${P1_QUANT_MODE}" FUSION_MODE="${fusion_mode}" RUN_NAME="${run_name}" \
      bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
  fi

  if [[ "${RUN_COMPARE}" == "1" ]]; then
    echo
    echo "=== Comparing ${run_name} against ${BASELINE_NAME} ==="
    BASELINE_NAME="${BASELINE_NAME}" CANDIDATE_NAME="${run_name}" \
      bash "${SCRIPT_DIR}/run_compare_lmdb_quant.sh"
  fi
done
