#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

P2_STAGE="${P2_STAGE:-smoke}"
P2_MODES="${P2_MODES:-line_graph_linear_bf16 line_edge_gate_linear_bf16 line_node_gate_linear_bf16 p0_line_gate_paper_stable_linear_bf16 line_graph_linear_fp16 p0_line_gate_paper_stable_linear_fp16 line_graph_linear_cached_bf16 line_edge_gate_linear_cached_bf16 line_node_gate_linear_cached_bf16 p0_line_gate_paper_stable_linear_cached_bf16 line_graph_linear_cached_fp16 p0_line_gate_paper_stable_linear_cached_fp16}"
P2_PROFILE_TASKS="${P2_PROFILE_TASKS:-e ef efs}"
P2_PROFILE_REPS="${P2_PROFILE_REPS:-1 2 3}"
BASELINE_NAME="${BASELINE_NAME:-fp32_baseline}"
RUN_BASELINE="${RUN_BASELINE:-1}"

SMOKE_LIMIT_COUNT="${SMOKE_LIMIT_COUNT:-50}"
PROFILE_LIMIT_COUNT="${PROFILE_LIMIT_COUNT:-50}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-10}"
STATIC_LIMIT_COUNT="${STATIC_LIMIT_COUNT:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

run_smoke() {
  for mode in ${P2_MODES}; do
    run_name="smoke_${mode}"
    echo "=== P2 smoke: mode=${mode} ==="
    PRECISION_MODE=fp32 QUANT_MODE="${mode}" RUN_NAME="${run_name}" \
      LIMIT_COUNT="${SMOKE_LIMIT_COUNT}" SAMPLE_SEED="${SAMPLE_SEED}" MEASURE_TIME=0 \
      PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
  done
}

run_profile() {
  if [[ "${RUN_BASELINE}" == "1" ]]; then
    for task in ${P2_PROFILE_TASKS}; do
      for rep in ${P2_PROFILE_REPS}; do
        run_name="p2_none_${task}_r${rep}"
        echo "=== P2 profile baseline: task=${task} rep=${rep} ==="
        PRECISION_MODE=fp32 TASK_NAME="${task}" QUANT_MODE=none RUN_NAME="${run_name}" \
          LIMIT_COUNT="${PROFILE_LIMIT_COUNT}" WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" SAMPLE_SEED="${SAMPLE_SEED}" \
          PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_salex_pipeline_profile.sh"
      done
    done
  fi

  for mode in ${P2_MODES}; do
    for task in ${P2_PROFILE_TASKS}; do
      for rep in ${P2_PROFILE_REPS}; do
        run_name="p2_${mode}_${task}_r${rep}"
        echo "=== P2 profile: mode=${mode} task=${task} rep=${rep} ==="
        PRECISION_MODE=fp32 TASK_NAME="${task}" QUANT_MODE="${mode}" RUN_NAME="${run_name}" \
          LIMIT_COUNT="${PROFILE_LIMIT_COUNT}" WARMUP_STEPS="${PROFILE_WARMUP_STEPS}" SAMPLE_SEED="${SAMPLE_SEED}" \
          PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_salex_pipeline_profile.sh"
      done
    done
  done
}

run_static() {
  if [[ "${RUN_BASELINE}" == "1" ]]; then
    echo "=== P2 static baseline: ${BASELINE_NAME} ==="
    PRECISION_MODE=fp32 QUANT_MODE=none RUN_NAME="${BASELINE_NAME}" \
      LIMIT_COUNT="${STATIC_LIMIT_COUNT}" SAMPLE_SEED="${SAMPLE_SEED}" MEASURE_TIME=1 \
      PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
  fi

  for mode in ${P2_MODES}; do
    echo "=== P2 static: mode=${mode} ==="
    PRECISION_MODE=fp32 QUANT_MODE="${mode}" RUN_NAME="${mode}" \
      LIMIT_COUNT="${STATIC_LIMIT_COUNT}" SAMPLE_SEED="${SAMPLE_SEED}" MEASURE_TIME=1 \
      PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_infer_salex_lmdb_quant.sh"
  done
}

run_compare() {
  for mode in ${P2_MODES}; do
    echo "=== P2 compare: ${BASELINE_NAME} vs ${mode} ==="
    BASELINE_NAME="${BASELINE_NAME}" CANDIDATE_NAME="${mode}" \
      PYTHON_BIN="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_compare_lmdb_quant.sh"
  done
}

case "${P2_STAGE}" in
  smoke)
    run_smoke
    ;;
  profile)
    run_profile
    ;;
  static)
    run_static
    ;;
  compare)
    run_compare
    ;;
  all)
    run_smoke
    run_profile
    run_static
    run_compare
    ;;
  *)
    echo "Unknown P2_STAGE=${P2_STAGE}; expected smoke, profile, static, compare, or all" >&2
    exit 2
    ;;
esac
