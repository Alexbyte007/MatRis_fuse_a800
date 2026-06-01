#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_NAME="${RUN_NAME:-wbm_representative_500_fp32}"
JOINED_CSV="${JOINED_CSV:-/home/lht/lab/MatRIS/results/wbm_eval/${RUN_NAME}/predictions_joined_with_wbm_summary.csv}"
OUTPUT_JSON="${OUTPUT_JSON:-/home/lht/lab/MatRIS/results/wbm_eval/${RUN_NAME}/discovery_metrics_summary.json}"
STABILITY_THRESHOLD="${STABILITY_THRESHOLD:-0.0}"

echo "Calculating WBM discovery metrics:"
echo "  joined_csv=${JOINED_CSV}"
echo "  output_json=${OUTPUT_JSON}"
echo "  stability_threshold=${STABILITY_THRESHOLD}"

"${PYTHON_BIN}" "${PY_DIR}/calc_wbm_discovery_metrics.py" \
  --joined-csv "${JOINED_CSV}" \
  --output-json "${OUTPUT_JSON}" \
  --stability-threshold "${STABILITY_THRESHOLD}"
