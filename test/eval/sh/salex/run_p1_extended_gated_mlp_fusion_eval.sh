#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

P1_STAGE="${P1_STAGE:-p1c}"
P1_QUANT_MODE="${P1_QUANT_MODE:-none}"
RUN_PROFILE="${RUN_PROFILE:-1}"
RUN_STATIC="${RUN_STATIC:-1}"
RUN_COMPARE="${RUN_COMPARE:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
BASELINE_NAME="${BASELINE_NAME:-fp32_baseline}"
PROFILE_BASELINE_NAME="${PROFILE_BASELINE_NAME:-task_matrix_none_efs}"
PROFILE_LIMIT_COUNT="${PROFILE_LIMIT_COUNT:-50}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-5}"
STATIC_LIMIT_COUNT="${STATIC_LIMIT_COUNT:-500}"
STATIC_MEASURE_TIME="${STATIC_MEASURE_TIME:-1}"

case "${P1_STAGE}" in
  p1c)
    DEFAULT_MODES="atom_attn_edge_gated_mlp_fused_fp32 atom_refine_edge_gated_mlp_fused_fp32 atom_edge_gated_mlp_fused_fp32"
    ;;
  p1d)
    DEFAULT_MODES="atom_attn_node_gated_mlp_fused_fp32"
    ;;
  p1e)
    DEFAULT_MODES="line_edge_gated_mlp_second_fused_fp32 line_all_candidate_gated_mlp_second_fused_fp32 atom_edge_gated_mlp_second_fused_fp32 atom_attn_node_gated_mlp_second_fused_fp32"
    ;;
  p1e_retest)
    DEFAULT_MODES="line_all_candidate_gated_mlp_second_fused_fp32 atom_attn_node_gated_mlp_second_fused_fp32"
    ;;
  p1f)
    DEFAULT_MODES="line_edge_gated_mlp_tail_fused_fp32 line_all_candidate_gated_mlp_tail_fused_fp32 line_edge_gated_mlp_second_tail_fused_fp32 line_all_candidate_gated_mlp_second_tail_fused_fp32 atom_edge_gated_mlp_tail_fused_fp32 atom_edge_gated_mlp_second_tail_fused_fp32 atom_attn_node_gated_mlp_tail_fused_fp32 atom_attn_node_gated_mlp_second_tail_fused_fp32"
    ;;
  p1g)
    DEFAULT_MODES="all_passed_gated_mlp_second_fused_fp32"
    ;;
  *)
    echo "Unknown P1_STAGE=${P1_STAGE}; expected p1c, p1d, p1e, p1e_retest, p1f, or p1g" >&2
    exit 2
    ;;
esac

P1_FUSION_MODES="${P1_FUSION_MODES:-${DEFAULT_MODES}}"

for fusion_mode in ${P1_FUSION_MODES}; do
  run_name="${P1_STAGE}_${P1_QUANT_MODE}_${fusion_mode}"

  if [[ "${RUN_PROFILE}" == "1" ]]; then
    echo
    echo "=== Profiling ${P1_STAGE} GatedMLP fusion: quant=${P1_QUANT_MODE}, fusion=${fusion_mode} ==="
    QUANT_MODE="${P1_QUANT_MODE}" FUSION_MODE="${fusion_mode}" RUN_NAME="${run_name}" \
      LIMIT_COUNT="${PROFILE_LIMIT_COUNT}" WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
      bash "${SCRIPT_DIR}/run_salex_pipeline_profile.sh"
  fi

  if [[ "${RUN_STATIC}" == "1" ]]; then
    echo
    echo "=== Running ${P1_STAGE} GatedMLP static eval: quant=${P1_QUANT_MODE}, fusion=${fusion_mode} ==="
    QUANT_MODE="${P1_QUANT_MODE}" FUSION_MODE="${fusion_mode}" RUN_NAME="${run_name}" \
      LIMIT_COUNT="${STATIC_LIMIT_COUNT}" MEASURE_TIME="${STATIC_MEASURE_TIME}" \
      bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
  fi

  if [[ "${RUN_COMPARE}" == "1" ]]; then
    echo
    echo "=== Comparing ${run_name} against ${BASELINE_NAME} ==="
    BASELINE_NAME="${BASELINE_NAME}" CANDIDATE_NAME="${run_name}" \
      bash "${SCRIPT_DIR}/run_compare_lmdb_quant.sh"
  fi

  if [[ "${RUN_SUMMARY}" == "1" ]]; then
    echo
    echo "=== Summarizing ${run_name} ==="
    "${PYTHON_BIN}" "${REPO_ROOT}/test/eval/summarize_p1_fusion_eval.py" \
      --run-name "${run_name}" \
      --baseline-name "${BASELINE_NAME}" \
      --profile-baseline-name "${PROFILE_BASELINE_NAME}" \
      --stage "${P1_STAGE}"
  fi
done
