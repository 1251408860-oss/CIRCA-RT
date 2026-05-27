#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_RESULT_ROOT="${BASE_RESULT_ROOT:-${ROOT}/results_agx_orin64_maximal_${STAMP}}"

if [[ -d "${ROOT}/torch_cache" ]]; then
  export TORCH_HOME="${TORCH_HOME:-${ROOT}/torch_cache}"
fi

run_one_mode() {
  local label="$1"
  local nvp_model_id="$2"
  local result_root="${BASE_RESULT_ROOT}_${label}"

  echo
  echo "=== AGX Orin 64GB maximal run: ${label} ==="
  echo "result_root=${result_root}"

  PYTHON_BIN="${PYTHON_BIN}" \
  RESULT_ROOT="${result_root}" \
  NVP_MODEL_ID="${nvp_model_id}" \
  SET_JETSON_CLOCKS="${SET_JETSON_CLOCKS:-1}" \
  AGX_DATASETS="${AGX_DATASETS:-av2 droid nuimages}" \
  DEADLINES_MS="${DEADLINES_MS:-16.667 20 25 33.333 40 50 100}" \
  PRIMARY_DEADLINE_MS="${PRIMARY_DEADLINE_MS:-33.333}" \
  MODELS="${MODELS:-squeezenet1_1 mobilenet_v2 resnet18 resnet50}" \
  METHODS="${METHODS:-CIRCA-RT CIRCA-RT-Slack ConditionalRFF-HSIC ContextAwareConformal RFF-HSIC TimingThreshold SemanticThreshold AlwaysAudit RandomBudgetAudit}" \
  SEEDS="${SEEDS:-7 8 9 10 11}" \
  WARMUP="${WARMUP:-80}" \
  DEVICE="${DEVICE:-cuda}" \
  WEIGHTS="${WEIGHTS:-default}" \
  AUDIT_MODEL="${AUDIT_MODEL:-resnet18}" \
  AUDIT_WEIGHTS="${AUDIT_WEIGHTS:-default}" \
  N_AV2="${N_AV2:-1500}" \
  N_DROID="${N_DROID:-900}" \
  N_NUIMAGES="${N_NUIMAGES:-217}" \
  FRAME_LIMIT_AV2="${FRAME_LIMIT_AV2:-2169}" \
  FRAME_LIMIT_DROID="${FRAME_LIMIT_DROID:-901}" \
  FRAME_LIMIT_NUIMAGES="${FRAME_LIMIT_NUIMAGES:-218}" \
  GPU_STRESS_SIZE="${GPU_STRESS_SIZE:-2048}" \
  GPU_STRESS_REPEATS="${GPU_STRESS_REPEATS:-2}" \
  COUPLED_STRESS_EXTRA_REPEATS="${COUPLED_STRESS_EXTRA_REPEATS:-24}" \
  CIRCA_LOW_QUANTILE="${CIRCA_LOW_QUANTILE:-0.95}" \
  CIRCA_HIGH_QUANTILE="${CIRCA_HIGH_QUANTILE:-0.99}" \
  TOKEN_AUDIT_COST_MS="${TOKEN_AUDIT_COST_MS:-4.0}" \
  BUCKET_CAPACITY="${BUCKET_CAPACITY:-12.0}" \
  REPLENISH_RATE="${REPLENISH_RATE:-0.45}" \
  RUN_REALFRAMES="${RUN_REALFRAMES:-1}" \
  RUN_BACKEND_MATRIX="${RUN_BACKEND_MATRIX:-1}" \
  RUN_DERIVED_ANALYSES="${RUN_DERIVED_ANALYSES:-1}" \
  RUN_SUMMARY="${RUN_SUMMARY:-1}" \
  BACKEND_MODELS="${BACKEND_MODELS:-squeezenet1_1 mobilenet_v2 resnet18}" \
  BACKEND_RUNTIMES="${BACKEND_RUNTIMES:-torch_cuda onnx_cuda onnx_tensorrt}" \
  BACKEND_N="${BACKEND_N:-1800}" \
  BACKEND_WARMUP="${BACKEND_WARMUP:-120}" \
  BACKEND_DEADLINE_MS="${BACKEND_DEADLINE_MS:-33.333}" \
  bash "${ROOT}/scripts/run_agx_orin64_comprehensive_experiment.sh"
}

if [[ -n "${AGX_POWER_MODE_IDS:-}" ]]; then
  for mode_id in ${AGX_POWER_MODE_IDS}; do
    run_one_mode "nvp${mode_id}" "${mode_id}"
  done
else
  run_one_mode "maxperf" ""
fi

echo
echo "Maximal AGX run roots:"
find "$(dirname "${BASE_RESULT_ROOT}")" -maxdepth 1 -type d -name "$(basename "${BASE_RESULT_ROOT}")_*" -print | sort
