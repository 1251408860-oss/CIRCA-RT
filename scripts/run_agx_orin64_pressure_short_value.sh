#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_RESULT_ROOT="${BASE_RESULT_ROOT:-${ROOT}/results_agx_orin64_pressure_short_value_${STAMP}}"

cd "${ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN}" \
PRESSURE_PROFILE=paper \
BASE_RESULT_ROOT="${BASE_RESULT_ROOT}" \
AGX_DATASETS="${AGX_DATASETS:-av2 nuimages}" \
DEADLINES_MS="${DEADLINES_MS:-20 33.333 50}" \
SEEDS="${SEEDS:-7}" \
PRESSURE_LEVELS="${PRESSURE_LEVELS:-idle:0:0:0 high:1536:3:24}" \
MODELS="${MODELS:-mobilenet_v2}" \
N_AV2="${N_AV2:-300}" \
N_NUIMAGES="${N_NUIMAGES:-217}" \
FRAME_LIMIT_AV2="${FRAME_LIMIT_AV2:-600}" \
FRAME_LIMIT_NUIMAGES="${FRAME_LIMIT_NUIMAGES:-217}" \
RUN_BACKEND_MATRIX=0 \
RUN_DERIVED_ANALYSES="${RUN_DERIVED_ANALYSES:-1}" \
bash scripts/run_agx_orin64_pressure_sweep.sh

rc=$?

echo
echo "Short value pressure result root: ${BASE_RESULT_ROOT}"
echo "Expected report: ${BASE_RESULT_ROOT}/PRESSURE_SWEEP_REPORT.md"
echo
echo "To package the result on AGX:"
echo "tar -czf agx_orin64_pressure_short_value_\$(date +%Y%m%d_%H%M%S).tar.gz -C \"$(dirname "${BASE_RESULT_ROOT}")\" \"$(basename "${BASE_RESULT_ROOT}")\""
echo "sha256sum agx_orin64_pressure_short_value_*.tar.gz"

exit "${rc}"
