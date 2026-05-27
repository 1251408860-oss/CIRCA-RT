#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
INPUT_MANIFEST="${INPUT_MANIFEST:-${ROOT}/data/official_baseline_inputs/av2/manifest.csv}"
PSM_EXPORT_DIR="${PSM_EXPORT_DIR:-${ROOT}/data/official_psm_inputs/av2}"
OFFICIAL_OUT_DIR="${OFFICIAL_OUT_DIR:-${ROOT}/results_official_psm_baselines_av2}"
DEVICE="${DEVICE:-cuda}"
LIMIT_CASES="${LIMIT_CASES:-0}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
MAX_TEST_ROWS="${MAX_TEST_ROWS:-0}"
SMOKE="${SMOKE:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"

cd "${ROOT}"

"${PYTHON_BIN}" scripts/prepare_official_psm_dataset.py \
  --manifest "${INPUT_MANIFEST}" \
  --out-dir "${PSM_EXPORT_DIR}" \
  --max-train-rows "${MAX_TRAIN_ROWS}" \
  --max-test-rows "${MAX_TEST_ROWS}"

mapfile -t CASES < <(find "${PSM_EXPORT_DIR}" -mindepth 4 -maxdepth 4 -type f -name metadata.json | sort)

WRAPPER_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  WRAPPER_ARGS+=(--smoke)
fi
if [[ "${TIMEOUT_SEC}" != "0" ]]; then
  WRAPPER_ARGS+=(--timeout-sec "${TIMEOUT_SEC}")
fi

count=0
for meta in "${CASES[@]}"; do
  case_dir="$(dirname "${meta}")"
  rel="${case_dir#${PSM_EXPORT_DIR}/}"
  safe_rel="${rel//\//__}"

  if [[ "${LIMIT_CASES}" != "0" && "${count}" -ge "${LIMIT_CASES}" ]]; then
    break
  fi

  echo "=== official PSM case=${rel}"
  "${PYTHON_BIN}" scripts/run_official_modern_tcn.py \
    --case-dir "${case_dir}" \
    --device "${DEVICE}" \
    --out-json "${OFFICIAL_OUT_DIR}/modern_tcn/${safe_rel}.json" \
    "${WRAPPER_ARGS[@]}"

  "${PYTHON_BIN}" scripts/run_official_time_mixer.py \
    --case-dir "${case_dir}" \
    --device "${DEVICE}" \
    --out-json "${OFFICIAL_OUT_DIR}/time_mixer/${safe_rel}.json" \
    "${WRAPPER_ARGS[@]}"

  count=$((count + 1))
done
