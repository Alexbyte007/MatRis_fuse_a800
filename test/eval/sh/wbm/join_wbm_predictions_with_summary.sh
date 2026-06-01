#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_NAME="${RUN_NAME:-wbm_representative_500_fp32}"
PREDICTIONS_JSONL="${PREDICTIONS_JSONL:-/home/lht/lab/MatRIS/results/wbm_eval/${RUN_NAME}/per_structure_predictions.jsonl}"
WBM_SUMMARY="${WBM_SUMMARY:-/home/lht/lab/wbm/2023-12-13-wbm-summary.csv.gz}"
OUTPUT_CSV="${OUTPUT_CSV:-/home/lht/lab/MatRIS/results/wbm_eval/${RUN_NAME}/predictions_joined_with_wbm_summary.csv}"

echo "Joining WBM predictions with summary:"
echo "  predictions_jsonl=${PREDICTIONS_JSONL}"
echo "  wbm_summary=${WBM_SUMMARY}"
echo "  output_csv=${OUTPUT_CSV}"

"${PYTHON_BIN}" "${PY_DIR}/join_wbm_predictions_with_summary.py" \
  --predictions-jsonl "${PREDICTIONS_JSONL}" \
  --wbm-summary "${WBM_SUMMARY}" \
  --output-csv "${OUTPUT_CSV}"
