#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_SRC="${DATASET_SRC:-/home/lht/lab/sAlex/val}"
MODEL_NAME="${MODEL_NAME:-matris_10m_oam}"
TASK_NAME="${TASK_NAME:-efs}"
DEVICE_NAME="${DEVICE_NAME:-cuda}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
LIMIT_COUNT="${LIMIT_COUNT:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PRECISION_MODE="${PRECISION_MODE:-fp32}"
QUANT_MODE="${QUANT_MODE:-none}"
FUSION_MODE="${FUSION_MODE:-none}"
COMPILE_FLAG="${COMPILE_FLAG:-0}"
RUN_NAME="${RUN_NAME:-fp32_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/static_eval_salex/${RUN_NAME}}"

ARGS=(
  "${PY_DIR}/evaluate_salex_static_metrics.py"
  --dataset-src "${DATASET_SRC}"
  --output-dir "${OUTPUT_DIR}"
  --model "${MODEL_NAME}"
  --task "${TASK_NAME}"
  --device "${DEVICE_NAME}"
  --precision-mode "${PRECISION_MODE}"
  --quant-mode "${QUANT_MODE}"
  --fusion-mode "${FUSION_MODE}"
  --warmup-steps "${WARMUP_STEPS}"
  --limit "${LIMIT_COUNT}"
  --sample-seed "${SAMPLE_SEED}"
)

if [[ "${COMPILE_FLAG}" == "1" ]]; then
  ARGS+=(--compile)
fi

echo "Running sAlex static evaluation with config:"
echo "  dataset_src=${DATASET_SRC}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  model=${MODEL_NAME}"
echo "  task=${TASK_NAME}"
echo "  device=${DEVICE_NAME}"
echo "  precision_mode=${PRECISION_MODE}"
echo "  quant_mode=${QUANT_MODE}"
echo "  fusion_mode=${FUSION_MODE}"
echo "  warmup_steps=${WARMUP_STEPS}"
echo "  limit=${LIMIT_COUNT}"
echo "  sample_seed=${SAMPLE_SEED}"
echo "  compile=${COMPILE_FLAG}"

"${PYTHON_BIN}" "${ARGS[@]}"
