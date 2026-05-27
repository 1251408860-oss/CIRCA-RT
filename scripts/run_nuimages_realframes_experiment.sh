#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FRAME_DIR="${FRAME_DIR:-${ROOT}/data/frames_nuimages}"
export OUT_DIR="${OUT_DIR:-${ROOT}/results_autodl_perception_semantics_realframes_nuimages}"
export DEEP_OUT_DIR="${DEEP_OUT_DIR:-${ROOT}/results_recent_deep_baselines_realframes_nuimages}"
export FRAME_LIMIT="${FRAME_LIMIT:-217}"
export N="${N:-217}"
export WARMUP="${WARMUP:-20}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

exec "${ROOT}/scripts/run_realframes_perception_experiment.sh"
