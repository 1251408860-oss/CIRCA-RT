#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

export DROID_REPO_ID="${DROID_REPO_ID:-lerobot/droid_100}"
export DROID_HF_ENDPOINT="${DROID_HF_ENDPOINT:-https://hf-mirror.com}"
export DROID_RAW_DIR="${DROID_RAW_DIR:-${ROOT}/data/raw/droid_100_subset}"
export FRAME_DIR="${FRAME_DIR:-${ROOT}/data/frames_droid}"
export OUT_DIR="${OUT_DIR:-${ROOT}/results_autodl_perception_semantics_realframes_droid}"
export DEEP_OUT_DIR="${DEEP_OUT_DIR:-${ROOT}/results_recent_deep_baselines_realframes_droid}"
export DROID_STRIDE="${DROID_STRIDE:-5}"
export DROID_MAX_FRAMES="${DROID_MAX_FRAMES:-1200}"
export DROID_PER_VIDEO_FRAMES="${DROID_PER_VIDEO_FRAMES:-400}"
export FRAME_LIMIT="${FRAME_LIMIT:-1200}"
export N="${N:-600}"
export WARMUP="${WARMUP:-20}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-75}"

if [[ ! -s "${FRAME_DIR}/manifest.csv" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/download_hf_video_frames.py" \
    --repo-id "${DROID_REPO_ID}" \
    --endpoint "${DROID_HF_ENDPOINT}" \
    --repo-file "videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4" \
    --repo-file "videos/observation.images.exterior_image_2_left/chunk-000/file-000.mp4" \
    --repo-file "videos/observation.images.wrist_image_left/chunk-000/file-000.mp4" \
    --raw-dir "${DROID_RAW_DIR}" \
    --frames-out "${FRAME_DIR}" \
    --dataset-hint droid_robot \
    --stride "${DROID_STRIDE}" \
    --max-frames "${DROID_MAX_FRAMES}" \
    --per-video-frames "${DROID_PER_VIDEO_FRAMES}" \
    --min-width 128 \
    --min-height 128
fi

"${ROOT}/scripts/run_realframes_perception_experiment.sh"

"${PYTHON_BIN}" "${ROOT}/scripts/report_real_dataset_quality.py" \
  --manifest "${FRAME_DIR}/manifest.csv" \
  --out "${OUT_DIR}/tables/real_dataset_quality.json"
