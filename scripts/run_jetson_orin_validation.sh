#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
FRAME_DIR="${FRAME_DIR:-${ROOT}/data/frames}"
OUT_DIR="${OUT_DIR:-${ROOT}/results_jetson_orin}"
RAW_LOG_DIR="${RAW_LOG_DIR:-${OUT_DIR}/raw_logs}"
TABLE_DIR="${TABLE_DIR:-${OUT_DIR}/tables}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
RUN_PREFIX="${RUN_PREFIX:-jetson_orin}"
PLATFORM_LABEL="${PLATFORM_LABEL:-jetson_orin_validation}"
DEVICE="${DEVICE:-cuda}"
MODELS="${MODELS:-mobilenet_v2 resnet18}"
WEIGHTS="${WEIGHTS:-default}"
AUDIT_MODEL="${AUDIT_MODEL:-resnet18}"
AUDIT_WEIGHTS="${AUDIT_WEIGHTS:-default}"
N="${N:-900}"
SEEDS="${SEEDS:-7 8 9}"
WARMUP="${WARMUP:-30}"
DEADLINE_MS="${DEADLINE_MS:-35.0}"
GPU_STRESS_SIZE="${GPU_STRESS_SIZE:-2048}"
GPU_STRESS_REPEATS="${GPU_STRESS_REPEATS:-2}"
COUPLED_STRESS_EXTRA_REPEATS="${COUPLED_STRESS_EXTRA_REPEATS:-24}"
CIRCA_LOW_QUANTILE="${CIRCA_LOW_QUANTILE:-0.95}"
CIRCA_HIGH_QUANTILE="${CIRCA_HIGH_QUANTILE:-0.99}"
TOKEN_AUDIT_COST_MS="${TOKEN_AUDIT_COST_MS:-4.0}"
BUCKET_CAPACITY="${BUCKET_CAPACITY:-12.0}"
REPLENISH_RATE="${REPLENISH_RATE:-0.45}"
PROGRESS_EVERY="${PROGRESS_EVERY:-150}"
TEGRA_INTERVAL_MS="${TEGRA_INTERVAL_MS:-100}"

if [[ ! -d "${FRAME_DIR}" ]]; then
  echo "missing FRAME_DIR=${FRAME_DIR}; run scripts/prepare_real_frame_dataset.py first" >&2
  exit 2
fi

mkdir -p "${RAW_LOG_DIR}" "${TABLE_DIR}" "${LOG_DIR}"
cd "${ROOT}"

if [[ -f "scripts/prepare_jetson_strong_pipeline.sh" ]]; then
  bash scripts/prepare_jetson_strong_pipeline.sh >/dev/null
fi

bash scripts/jetson_env_check.sh | tee "${RAW_LOG_DIR}/env_check.log"

TEGRA_PID=""
if command -v tegrastats >/dev/null 2>&1; then
  bash scripts/run_jetson_tegrastats.sh "${RAW_LOG_DIR}" "${TEGRA_INTERVAL_MS}" >/dev/null 2>&1 &
  TEGRA_PID=$!
fi
cleanup() {
  if [[ -n "${TEGRA_PID}" ]]; then
    kill "${TEGRA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

RUN_LOG="${LOG_DIR}/jetson_validation_$(date +%Y%m%d_%H%M%S).log"
{
  "${PYTHON_BIN}" scripts/run_autodl_perception_semantics.py \
    --out-dir "${OUT_DIR}" \
    --frame-dir "${FRAME_DIR}" \
    --device "${DEVICE}" \
    --models ${MODELS} \
    --weights "${WEIGHTS}" \
    --audit-model "${AUDIT_MODEL}" \
    --audit-weights "${AUDIT_WEIGHTS}" \
    --n "${N}" \
    --seeds ${SEEDS} \
    --warmup "${WARMUP}" \
    --deadline-ms "${DEADLINE_MS}" \
    --gpu-stress-size "${GPU_STRESS_SIZE}" \
    --gpu-stress-repeats "${GPU_STRESS_REPEATS}" \
    --coupled-stress-extra-repeats "${COUPLED_STRESS_EXTRA_REPEATS}" \
    --circa-low-quantile "${CIRCA_LOW_QUANTILE}" \
    --circa-high-quantile "${CIRCA_HIGH_QUANTILE}" \
    --token-audit-cost-ms "${TOKEN_AUDIT_COST_MS}" \
    --bucket-capacity "${BUCKET_CAPACITY}" \
    --replenish-rate "${REPLENISH_RATE}" \
    --methods CIRCA-RT CIRCA-RT-Slack ContextAwareConformal ConditionalRFF-HSIC RFF-HSIC SemanticThreshold TimingThreshold AlwaysAudit RandomBudgetAudit \
    --progress-every "${PROGRESS_EVERY}"
} 2>&1 | tee "${RUN_LOG}"

trap - EXIT
cleanup

"${PYTHON_BIN}" scripts/validate_audit_bound.py \
  --input-root "${OUT_DIR}/raw" \
  --out "${TABLE_DIR}/audit_bound_validation.csv" \
  --methods CIRCA-RT CIRCA-RT-Slack \
  --cost-cols audit_token_charge_ms

RAW_LOGS=()
while IFS= read -r -d '' file; do
  RAW_LOGS+=("${file}")
done < <(find "${RAW_LOG_DIR}" -maxdepth 1 -name 'tegrastats_*.log' -print0 2>/dev/null || true)

if [[ ${#RAW_LOGS[@]} -gt 0 ]]; then
  "${PYTHON_BIN}" scripts/parse_tegrastats.py \
    --out "${TABLE_DIR}/tegrastats_samples.csv" \
    --summary-out "${TABLE_DIR}/tegrastats_summary.csv" \
    "${RAW_LOGS[@]}"
fi

"${PYTHON_BIN}" scripts/summarize_jetson_perception_results.py \
  --out-dir "${OUT_DIR}" \
  --raw-logs-dir "${RAW_LOG_DIR}" \
  --platform-label "${PLATFORM_LABEL}"

echo "jetson validation results: ${OUT_DIR}"
