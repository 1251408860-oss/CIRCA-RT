#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
FRAME_DIR="${FRAME_DIR:-${ROOT}/data/frames}"
OUT_DIR="${OUT_DIR:-${ROOT}/results_autodl_perception_semantics_realframes}"
DEEP_OUT_DIR="${DEEP_OUT_DIR:-${ROOT}/results_recent_deep_baselines_realframes}"

if [[ ! -d "${FRAME_DIR}" ]]; then
  echo "missing FRAME_DIR=${FRAME_DIR}; run scripts/prepare_real_frame_dataset.py first" >&2
  exit 2
fi

cd "${ROOT}"

EXTRA_MAIN_ARGS=()
if [[ "${FRAME_LIMIT:-0}" != "0" ]]; then
  EXTRA_MAIN_ARGS+=(--frame-limit "${FRAME_LIMIT}")
fi
if [[ -n "${CLIP_LEN:-}" ]]; then
  EXTRA_MAIN_ARGS+=(--clip-len "${CLIP_LEN}")
fi

"${PYTHON_BIN}" scripts/run_autodl_perception_semantics.py \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --frame-dir "${FRAME_DIR}" \
  --models mobilenet_v2 resnet18 \
  --weights default \
  --audit-model resnet18 \
  --audit-weights default \
  --n "${N:-1200}" \
  --seeds ${SEEDS:-7 8 9} \
  --warmup "${WARMUP:-30}" \
  --deadline-ms "${DEADLINE_MS:-35.0}" \
  --gpu-stress-size "${GPU_STRESS_SIZE:-2048}" \
  --gpu-stress-repeats "${GPU_STRESS_REPEATS:-2}" \
  --coupled-stress-extra-repeats "${COUPLED_STRESS_EXTRA_REPEATS:-24}" \
  --circa-low-quantile "${CIRCA_LOW_QUANTILE:-0.95}" \
  --circa-high-quantile "${CIRCA_HIGH_QUANTILE:-0.99}" \
  --token-audit-cost-ms "${TOKEN_AUDIT_COST_MS:-4.0}" \
  --bucket-capacity "${BUCKET_CAPACITY:-12.0}" \
  --replenish-rate "${REPLENISH_RATE:-0.45}" \
  --methods CIRCA-RT ContextAwareConformal ConditionalRFF-HSIC RFF-HSIC SemanticThreshold TimingThreshold AlwaysAudit RandomBudgetAudit \
  --progress-every "${PROGRESS_EVERY:-150}" \
  "${EXTRA_MAIN_ARGS[@]}"

"${PYTHON_BIN}" scripts/validate_audit_bound.py \
  --input-root "${OUT_DIR}/raw" \
  --out "${OUT_DIR}/tables/audit_bound_validation.csv" \
  --methods CIRCA-RT

"${PYTHON_BIN}" scripts/run_recent_deep_baselines.py \
  --inputs autodl_perception_realframes="${OUT_DIR}" \
  --out-dir "${DEEP_OUT_DIR}" \
  --device "${DEEP_DEVICE:-cuda}" \
  --backend "${DEEP_BACKEND:-torch}" \
  --methods CATCH DCdetector TranAD MOMENT TimeMixer ModernTCN \
  --epochs "${DEEP_EPOCHS:-20}" \
  --batch-size "${DEEP_BATCH_SIZE:-128}" \
  --max-train-windows "${DEEP_MAX_TRAIN_WINDOWS:-4096}" \
  --budgets ${DEEP_BUDGETS:-0.02 0.05 0.10 0.15}

echo "real-frame perception results: ${OUT_DIR}"
echo "recent deep baseline results: ${DEEP_OUT_DIR}"
