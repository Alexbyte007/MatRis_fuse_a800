#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PY_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_SRC="${DATASET_SRC:-/home/lht/lab/sAlex/val}"
MODEL="${MODEL:-matris_10m_oam}"
DEVICE="${DEVICE:-cuda}"
FAKE_QUANT_MODE="${FAKE_QUANT_MODE:-p0_line_gate_stable_w8a32}"
TORCHAO_QUANT_MODE="${TORCHAO_QUANT_MODE:-p0_line_gate_stable_w8a32_torchao}"
TASKS="${TASKS:-e ef}"
LIMIT="${LIMIT:-2}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
MAX_CAPTURES_PER_MODULE="${MAX_CAPTURES_PER_MODULE:-1}"
MAX_MODULES="${MAX_MODULES:-48}"
TIMING_REPEATS="${TIMING_REPEATS:-20}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/results/torchao_alignment/${TORCHAO_QUANT_MODE}_diagnosis.json}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" "${PY_DIR}/diagnose_torchao_alignment.py" \
  --dataset-src "${DATASET_SRC}" \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --fake-quant-mode "${FAKE_QUANT_MODE}" \
  --torchao-quant-mode "${TORCHAO_QUANT_MODE}" \
  --tasks ${TASKS} \
  --limit "${LIMIT}" \
  --sample-seed "${SAMPLE_SEED}" \
  --max-captures-per-module "${MAX_CAPTURES_PER_MODULE}" \
  --max-modules "${MAX_MODULES}" \
  --timing-repeats "${TIMING_REPEATS}" \
  --output-json "${OUTPUT_JSON}"

echo "Saved diagnosis to ${OUTPUT_JSON}"
