#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/lht/miniconda3/envs/matris311/bin/python}"
P1_STAGE="${P1_STAGE:-p1e_retest}"
P1_QUANT_MODE="${P1_QUANT_MODE:-p0_line_gate_paper_stable_w8a32}"
BASELINE_NAME="${BASELINE_NAME:-p0_line_gate_paper_stable_w8a32}"
PROFILE_BASELINE_NAME="${PROFILE_BASELINE_NAME:-task_matrix_p0_line_gate_paper_stable_w8a32_efs}"
RUN_PROFILE="${RUN_PROFILE:-1}"
RUN_STATIC="${RUN_STATIC:-1}"
RUN_COMPARE="${RUN_COMPARE:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
PROFILE_LIMIT_COUNT="${PROFILE_LIMIT_COUNT:-50}"
STATIC_LIMIT_COUNT="${STATIC_LIMIT_COUNT:-500}"

P1_FUSION_MODES="${P1_FUSION_MODES:-line_all_candidate_gated_mlp_second_fused_fp32 atom_attn_node_gated_mlp_second_fused_fp32}"

PYTHON_BIN="${PYTHON_BIN}" \
P1_STAGE="${P1_STAGE}" \
P1_QUANT_MODE="${P1_QUANT_MODE}" \
P1_FUSION_MODES="${P1_FUSION_MODES}" \
BASELINE_NAME="${BASELINE_NAME}" \
PROFILE_BASELINE_NAME="${PROFILE_BASELINE_NAME}" \
RUN_PROFILE="${RUN_PROFILE}" \
RUN_STATIC="${RUN_STATIC}" \
RUN_COMPARE="${RUN_COMPARE}" \
RUN_SUMMARY="${RUN_SUMMARY}" \
PROFILE_LIMIT_COUNT="${PROFILE_LIMIT_COUNT}" \
STATIC_LIMIT_COUNT="${STATIC_LIMIT_COUNT}" \
bash "${SCRIPT_DIR}/run_p1_extended_gated_mlp_fusion_eval.sh"
