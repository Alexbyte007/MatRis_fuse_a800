#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_SRC="${DATASET_SRC:-/home/lht/lab/sAlex/val}"
MODEL_NAME="${MODEL_NAME:-matris_10m_oam}"
MODEL_PATH="${MODEL_PATH:-}"
TASK_NAME="${TASK_NAME:-efsm}"
DEVICE_NAME="${DEVICE_NAME:-cuda}"
LIMIT_COUNT="${LIMIT_COUNT:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PRECISION_MODE="${PRECISION_MODE:-fp32}"
QUANT_MODE="${QUANT_MODE:-none}"
FUSION_MODE="${FUSION_MODE:-none}"
MEASURE_TIME="${MEASURE_TIME:-1}"
RUN_NAME="${RUN_NAME:-fp32_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lht/lab/MatRIS/results/salex_lmdb_quant/${RUN_NAME}}"

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  "${PY_DIR}/infer_salex_lmdb_quant.py"
  --dataset-src "${DATASET_SRC}"
  --model "${MODEL_NAME}"
  --task "${TASK_NAME}"
  --device "${DEVICE_NAME}"
  --precision-mode "${PRECISION_MODE}"
  --quant-mode "${QUANT_MODE}"
  --fusion-mode "${FUSION_MODE}"
  --limit "${LIMIT_COUNT}"
  --sample-seed "${SAMPLE_SEED}"
  --output-json "${OUTPUT_DIR}/summary.json"
  --save-predictions "${OUTPUT_DIR}/predictions.jsonl"
)

if [[ -n "${MODEL_PATH}" ]]; then
  ARGS+=(--model-path "${MODEL_PATH}")
fi

if [[ "${MEASURE_TIME}" == "1" ]]; then
  ARGS+=(--measure-time)
fi

echo "Running group-aligned sAlex LMDB inference with config:"
echo "  dataset_src=${DATASET_SRC}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  model=${MODEL_NAME}"
echo "  model_path=${MODEL_PATH}"
echo "  task=${TASK_NAME}"
echo "  device=${DEVICE_NAME}"
echo "  precision_mode=${PRECISION_MODE}"
echo "  quant_mode=${QUANT_MODE}"
echo "  fusion_mode=${FUSION_MODE}"
echo "  limit=${LIMIT_COUNT}"
echo "  sample_seed=${SAMPLE_SEED}"
echo "  measure_time=${MEASURE_TIME}"

"${PYTHON_BIN}" "${ARGS[@]}"
