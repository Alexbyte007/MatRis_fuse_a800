#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WBM_INIT="${WBM_INIT:-/home/lht/lab/wbm/wbm_2022-10-19-wbm-init-structs.jsonl.gz}"
WBM_SUMMARY="${WBM_SUMMARY:-/home/lht/lab/wbm/2023-12-13-wbm-summary.csv.gz}"
MODEL_NAME="${MODEL_NAME:-matris_10m_oam}"
TASK_NAME="${TASK_NAME:-efs}"
DEVICE_NAME="${DEVICE_NAME:-cuda}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-FIRE}"
RELAX_FMAX="${RELAX_FMAX:-0.05}"
RELAX_STEPS="${RELAX_STEPS:-100}"
ASE_FILTER_NAME="${ASE_FILTER_NAME:-FrechetCellFilter}"
RELAX_CELL_FLAG="${RELAX_CELL_FLAG:-1}"
SAVE_FINAL_STRUCTURE="${SAVE_FINAL_STRUCTURE:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
LIMIT_COUNT="${LIMIT_COUNT:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-20260424}"
STABILITY_THRESHOLD="${STABILITY_THRESHOLD:-0.0}"
PRECISION_MODE="${PRECISION_MODE:-fp32}"
COMPILE_FLAG="${COMPILE_FLAG:-0}"
RUN_NAME="${RUN_NAME:-wbm_representative_500_fp32}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/wbm_eval/${RUN_NAME}}"

ARGS=(
  "${PY_DIR}/run_wbm_eval.py"
  --wbm-init "${WBM_INIT}"
  --wbm-summary "${WBM_SUMMARY}"
  --output-dir "${OUTPUT_DIR}"
  --model "${MODEL_NAME}"
  --task "${TASK_NAME}"
  --device "${DEVICE_NAME}"
  --optimizer "${OPTIMIZER_NAME}"
  --relax-fmax "${RELAX_FMAX}"
  --relax-steps "${RELAX_STEPS}"
  --ase-filter "${ASE_FILTER_NAME}"
  --precision-mode "${PRECISION_MODE}"
  --warmup-steps "${WARMUP_STEPS}"
  --limit "${LIMIT_COUNT}"
  --sample-seed "${SAMPLE_SEED}"
  --stability-threshold "${STABILITY_THRESHOLD}"
)

if [[ "${RELAX_CELL_FLAG}" != "1" ]]; then
  ARGS+=(--no-relax-cell)
fi

if [[ "${SAVE_FINAL_STRUCTURE}" == "1" ]]; then
  ARGS+=(--save-final-structure)
fi

if [[ "${COMPILE_FLAG}" == "1" ]]; then
  ARGS+=(--compile)
fi

echo "Running WBM smoke evaluation with config:"
echo "  wbm_init=${WBM_INIT}"
echo "  wbm_summary=${WBM_SUMMARY}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  model=${MODEL_NAME}"
echo "  task=${TASK_NAME}"
echo "  device=${DEVICE_NAME}"
echo "  optimizer=${OPTIMIZER_NAME}"
echo "  relax_fmax=${RELAX_FMAX}"
echo "  relax_steps=${RELAX_STEPS}"
echo "  ase_filter=${ASE_FILTER_NAME}"
echo "  relax_cell=${RELAX_CELL_FLAG}"
echo "  save_final_structure=${SAVE_FINAL_STRUCTURE}"
echo "  precision_mode=${PRECISION_MODE}"
echo "  warmup_steps=${WARMUP_STEPS}"
echo "  limit=${LIMIT_COUNT}"
echo "  sample_mode=proportional_stability"
echo "  sample_seed=${SAMPLE_SEED}"
echo "  stability_threshold=${STABILITY_THRESHOLD}"
echo "  compile=${COMPILE_FLAG}"

"${PYTHON_BIN}" "${ARGS[@]}"
