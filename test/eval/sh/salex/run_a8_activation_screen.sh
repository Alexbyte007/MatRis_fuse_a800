#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

FUSION_MODE="${FUSION_MODE:-line_all_candidate_gated_mlp_second_fused_fp32}"
BASELINE_NAME="${BASELINE_NAME:-p3a_fp32_fused_baseline}"
RUN_PREFIX="${RUN_PREFIX:-p3a}"
LIMIT_COUNT="${LIMIT_COUNT:-500}"
MEASURE_TIME="${MEASURE_TIME:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_STATIC="${RUN_STATIC:-1}"
RUN_COMPARE="${RUN_COMPARE:-0}"
RUN_PROFILE="${RUN_PROFILE:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
PROFILE_TASKS="${PROFILE_TASKS:-e ef efs}"
PROFILE_LIMIT_COUNT="${PROFILE_LIMIT_COUNT:-50}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-5}"

DEFAULT_A8_MODES="\
a8_refine_atom_node_ffn \
a8_refine_atom_edge_ffn \
a8_attn_atom_source_weight_linear \
a8_attn_atom_target_weight_linear \
a8_attn_atom_edge_core \
a8_attn_line_source_weight_linear \
a8_attn_line_target_weight_linear \
a8_refine_line_node_ffn \
a8_refine_line_edge_ffn \
a8_attn_line_edge_core \
a8_attn_line_edge_gate \
a8_refine_line_edge_core \
a8_refine_line_edge_gate \
a8_attn_line_node_core \
a8_attn_line_node_gate \
a8_atom_p0 \
a8_line_graph \
a8_attn_line_edge_core_gate \
a8_refine_line_edge_core_gate \
a8_attn_line_node_core_gate \
a8_line_edge_core_gate \
a8_line_node_core_gate \
a8_line_nonlinear_core_gate \
a8_p0_line_gate_paper_stable \
a8_all_single_pass"

A8_MODES="${A8_MODES:-${DEFAULT_A8_MODES}}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo
  echo "=== P3a baseline: quant=none fusion=${FUSION_MODE} run=${BASELINE_NAME} ==="
  QUANT_MODE="none" FUSION_MODE="${FUSION_MODE}" RUN_NAME="${BASELINE_NAME}" \
    LIMIT_COUNT="${LIMIT_COUNT}" MEASURE_TIME="${MEASURE_TIME}" \
    bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
fi

for mode in ${A8_MODES}; do
  run_name="${RUN_PREFIX}_${mode}"

  if [[ "${RUN_STATIC}" == "1" ]]; then
    echo
    echo "=== P3a static A8 screen: quant=${mode} fusion=${FUSION_MODE} run=${run_name} ==="
    QUANT_MODE="${mode}" FUSION_MODE="${FUSION_MODE}" RUN_NAME="${run_name}" \
      LIMIT_COUNT="${LIMIT_COUNT}" MEASURE_TIME="${MEASURE_TIME}" \
      bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
  fi

  if [[ "${RUN_COMPARE}" == "1" ]]; then
    echo
    echo "=== Comparing ${run_name} against ${BASELINE_NAME} ==="
    BASELINE_NAME="${BASELINE_NAME}" CANDIDATE_NAME="${run_name}" \
      bash "${SCRIPT_DIR}/run_compare_lmdb_quant.sh"
  fi

  if [[ "${RUN_PROFILE}" == "1" ]]; then
    for task in ${PROFILE_TASKS}; do
      profile_run_name="${run_name}_${task}"
      echo
      echo "=== P3a profile A8 screen: quant=${mode} task=${task} run=${profile_run_name} ==="
      TASK_NAME="${task}" QUANT_MODE="${mode}" FUSION_MODE="${FUSION_MODE}" RUN_NAME="${profile_run_name}" \
        LIMIT_COUNT="${PROFILE_LIMIT_COUNT}" WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" \
        bash "${SCRIPT_DIR}/run_salex_pipeline_profile.sh"
    done
  fi
done

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  echo
  echo "=== Summarizing P3a A8 activation screen ==="
  "${PYTHON_BIN}" "${REPO_ROOT}/test/eval/summarize_a8_activation_screen.py" \
    --baseline-name "${BASELINE_NAME}" \
    --run-prefix "${RUN_PREFIX}" \
    --modes ${A8_MODES}
fi
