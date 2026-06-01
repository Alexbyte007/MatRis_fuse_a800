#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/lht/lab/sAlex}"
SPLIT="${SPLIT:-val}"
BASE_URL="${BASE_URL:-https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex}"

if [[ "${SPLIT}" != "train" && "${SPLIT}" != "val" ]]; then
  echo "SPLIT must be train or val, got: ${SPLIT}" >&2
  exit 2
fi

mkdir -p "${DATA_ROOT}"

ARCHIVE="${DATA_ROOT}/${SPLIT}.tar.gz"
URL="${BASE_URL}/${SPLIT}.tar.gz"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Downloading ${URL}"
  curl -L "${URL}" -o "${ARCHIVE}"
else
  echo "Archive already exists: ${ARCHIVE}"
fi

echo "Extracting ${ARCHIVE} into ${DATA_ROOT}"
tar -xzf "${ARCHIVE}" -C "${DATA_ROOT}"

echo "sAlex ${SPLIT} ready under ${DATA_ROOT}/${SPLIT}"
