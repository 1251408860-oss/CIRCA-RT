#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATASETS="${DATASETS:-av2 nuimages}"

for dataset in ${DATASETS}; do
  case "${dataset}" in
    av2)
      FRAME_DIR="${ROOT}/data/frames_av2"
      OUT_DIR="${ROOT}/results_autodl_perception_semantics_realframes_av2"
      DEEP_OUT_DIR="${ROOT}/results_recent_deep_baselines_realframes_av2"
      FRAME_LIMIT="${AV2_FRAME_LIMIT:-2200}"
      N="${AV2_N:-900}"
      WARMUP="${AV2_WARMUP:-30}"
      PROGRESS_EVERY="${AV2_PROGRESS_EVERY:-150}"
      ;;
    nuimages)
      FRAME_DIR="${ROOT}/data/frames_nuimages"
      OUT_DIR="${ROOT}/results_autodl_perception_semantics_realframes_nuimages"
      DEEP_OUT_DIR="${ROOT}/results_recent_deep_baselines_realframes_nuimages"
      FRAME_LIMIT="${NUIMAGES_FRAME_LIMIT:-217}"
      N="${NUIMAGES_N:-217}"
      WARMUP="${NUIMAGES_WARMUP:-20}"
      PROGRESS_EVERY="${NUIMAGES_PROGRESS_EVERY:-50}"
      ;;
    droid)
      FRAME_DIR="${ROOT}/data/frames_droid"
      OUT_DIR="${ROOT}/results_autodl_perception_semantics_realframes_droid"
      DEEP_OUT_DIR="${ROOT}/results_recent_deep_baselines_realframes_droid"
      FRAME_LIMIT="${DROID_FRAME_LIMIT:-1200}"
      N="${DROID_N:-600}"
      WARMUP="${DROID_WARMUP:-20}"
      PROGRESS_EVERY="${DROID_PROGRESS_EVERY:-75}"
      ;;
    *)
      echo "unknown dataset key: ${dataset}" >&2
      exit 2
      ;;
  esac

  echo "=== running dataset=${dataset} frame_dir=${FRAME_DIR} out_dir=${OUT_DIR}"
  FRAME_DIR="${FRAME_DIR}" \
  OUT_DIR="${OUT_DIR}" \
  DEEP_OUT_DIR="${DEEP_OUT_DIR}" \
  FRAME_LIMIT="${FRAME_LIMIT}" \
  N="${N}" \
  WARMUP="${WARMUP}" \
  PROGRESS_EVERY="${PROGRESS_EVERY}" \
  "${ROOT}/scripts/run_realframes_perception_experiment.sh"

  "${PYTHON_BIN}" "${ROOT}/scripts/report_real_dataset_quality.py" \
    --manifest "${FRAME_DIR}/manifest.csv" \
    --out "${OUT_DIR}/tables/real_dataset_quality.json"
done
