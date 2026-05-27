#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -d "${ROOT}/torch_cache" ]]; then
  export TORCH_HOME="${TORCH_HOME:-${ROOT}/torch_cache}"
fi
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results_agx_orin64_comprehensive_${STAMP}}"
ENV_DIR="${RESULT_ROOT}/env"
LOG_DIR="${RESULT_ROOT}/logs"
TABLE_DIR="${RESULT_ROOT}/tables"

AGX_DATASETS="${AGX_DATASETS:-av2 droid nuimages}"
DEADLINES_MS="${DEADLINES_MS:-33.333 50}"
PRIMARY_DEADLINE_MS="${PRIMARY_DEADLINE_MS:-${DEADLINES_MS%% *}}"
MODELS="${MODELS:-mobilenet_v2 resnet18}"
METHODS="${METHODS:-CIRCA-RT CIRCA-RT-Slack ConditionalRFF-HSIC ContextAwareConformal RFF-HSIC TimingThreshold SemanticThreshold AlwaysAudit RandomBudgetAudit}"
SEEDS="${SEEDS:-7 8 9}"
WARMUP="${WARMUP:-50}"
DEVICE="${DEVICE:-cuda}"
WEIGHTS="${WEIGHTS:-default}"
AUDIT_MODEL="${AUDIT_MODEL:-resnet18}"
AUDIT_WEIGHTS="${AUDIT_WEIGHTS:-default}"

N_AV2="${N_AV2:-900}"
N_DROID="${N_DROID:-900}"
N_NUIMAGES="${N_NUIMAGES:-217}"
FRAME_LIMIT_AV2="${FRAME_LIMIT_AV2:-1200}"
FRAME_LIMIT_DROID="${FRAME_LIMIT_DROID:-900}"
FRAME_LIMIT_NUIMAGES="${FRAME_LIMIT_NUIMAGES:-217}"

FRAME_DIR_AV2="${FRAME_DIR_AV2:-${ROOT}/data/frames_av2}"
FRAME_DIR_DROID="${FRAME_DIR_DROID:-${ROOT}/data/frames_droid}"
FRAME_DIR_NUIMAGES="${FRAME_DIR_NUIMAGES:-${ROOT}/data/frames_nuimages}"

GPU_STRESS_SIZE="${GPU_STRESS_SIZE:-2048}"
GPU_STRESS_REPEATS="${GPU_STRESS_REPEATS:-2}"
COUPLED_STRESS_EXTRA_REPEATS="${COUPLED_STRESS_EXTRA_REPEATS:-24}"
CIRCA_LOW_QUANTILE="${CIRCA_LOW_QUANTILE:-0.95}"
CIRCA_HIGH_QUANTILE="${CIRCA_HIGH_QUANTILE:-0.99}"
TOKEN_AUDIT_COST_MS="${TOKEN_AUDIT_COST_MS:-4.0}"
BUCKET_CAPACITY="${BUCKET_CAPACITY:-12.0}"
REPLENISH_RATE="${REPLENISH_RATE:-0.45}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
TEGRA_INTERVAL_MS="${TEGRA_INTERVAL_MS:-500}"
FAIL_FAST="${FAIL_FAST:-0}"

RUN_REALFRAMES="${RUN_REALFRAMES:-1}"
RUN_BACKEND_MATRIX="${RUN_BACKEND_MATRIX:-1}"
RUN_DERIVED_ANALYSES="${RUN_DERIVED_ANALYSES:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

BACKEND_MODELS="${BACKEND_MODELS:-mobilenet_v2 resnet18}"
BACKEND_RUNTIMES="${BACKEND_RUNTIMES:-torch_cuda onnx_cuda onnx_tensorrt}"
BACKEND_N="${BACKEND_N:-1200}"
BACKEND_WARMUP="${BACKEND_WARMUP:-80}"
BACKEND_DEADLINE_MS="${BACKEND_DEADLINE_MS:-33.333}"
BACKEND_INTERFERENCE_SIZE="${BACKEND_INTERFERENCE_SIZE:-512}"
BACKEND_ORT_INTERFERENCE_SIZE="${BACKEND_ORT_INTERFERENCE_SIZE:-384}"

SET_JETSON_CLOCKS="${SET_JETSON_CLOCKS:-1}"
NVP_MODEL_ID="${NVP_MODEL_ID:-}"

mkdir -p "${ENV_DIR}" "${LOG_DIR}" "${TABLE_DIR}"
cd "${ROOT}" || exit 1

FAILED_LOG="${LOG_DIR}/failed_commands.log"
: > "${FAILED_LOG}"

deadline_tag() {
  echo "$1" | sed 's/\./p/g; s/-/m/g'
}

dataset_frame_dir() {
  case "$1" in
    av2) echo "${FRAME_DIR_AV2}" ;;
    droid) echo "${FRAME_DIR_DROID}" ;;
    nuimages) echo "${FRAME_DIR_NUIMAGES}" ;;
    *) echo "" ;;
  esac
}

dataset_n() {
  case "$1" in
    av2) echo "${N_AV2}" ;;
    droid) echo "${N_DROID}" ;;
    nuimages) echo "${N_NUIMAGES}" ;;
    *) echo "0" ;;
  esac
}

dataset_frame_limit() {
  case "$1" in
    av2) echo "${FRAME_LIMIT_AV2}" ;;
    droid) echo "${FRAME_LIMIT_DROID}" ;;
    nuimages) echo "${FRAME_LIMIT_NUIMAGES}" ;;
    *) echo "0" ;;
  esac
}

run_logged() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  echo
  echo "=== ${name} ==="
  echo "$*" | tee "${log_file}.cmd"
  "$@" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  if [[ ${rc} -ne 0 ]]; then
    echo "${name} failed with exit ${rc}" | tee -a "${FAILED_LOG}"
    if [[ "${FAIL_FAST}" == "1" ]]; then
      exit "${rc}"
    fi
  fi
  return "${rc}"
}

start_tegrastats() {
  local name="$1"
  local log_file="${LOG_DIR}/tegrastats_${name}.log"
  if command -v tegrastats >/dev/null 2>&1; then
    tegrastats --interval "${TEGRA_INTERVAL_MS}" --logfile "${log_file}" >/dev/null 2>&1 &
    echo "$!"
  else
    echo ""
  fi
}

stop_tegrastats() {
  local pid="$1"
  if [[ -n "${pid}" ]]; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

sudo_cmd() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

capture_env() {
  echo "Capturing AGX environment in ${ENV_DIR}"
  {
    date
    hostname
    uname -a
    echo "TORCH_HOME=${TORCH_HOME:-}"
    cat /proc/device-tree/model 2>/dev/null || true
    cat /proc/device-tree/compatible 2>/dev/null || true
    cat /etc/nv_tegra_release 2>/dev/null || true
    lsb_release -a 2>/dev/null || true
    dpkg-query -W 'nvidia-l4t-*' 2>/dev/null || true
    df -h
    free -h
  } > "${ENV_DIR}/system_snapshot.txt" 2>&1

  PYTHON_BIN="${PYTHON_BIN}" bash scripts/jetson_env_check.sh > "${ENV_DIR}/jetson_env_check.log" 2>&1 || true
  "${PYTHON_BIN}" scripts/run_jetson_tensorrt_engine_check.py --out "${ENV_DIR}/provider_status.json" > "${ENV_DIR}/provider_status.log" 2>&1 || true
  sudo_cmd nvpmodel -q > "${ENV_DIR}/nvpmodel_before.txt" 2>&1 || true
  sudo_cmd jetson_clocks --show > "${ENV_DIR}/jetson_clocks_before.txt" 2>&1 || true
}

configure_power() {
  if [[ -n "${NVP_MODEL_ID}" ]]; then
    echo "Setting nvpmodel mode ${NVP_MODEL_ID}"
    sudo_cmd nvpmodel -m "${NVP_MODEL_ID}" > "${ENV_DIR}/nvpmodel_set.txt" 2>&1 || true
  fi
  if [[ "${SET_JETSON_CLOCKS}" == "1" ]] && command -v jetson_clocks >/dev/null 2>&1; then
    echo "Enabling jetson_clocks"
    sudo_cmd jetson_clocks > "${ENV_DIR}/jetson_clocks_set.txt" 2>&1 || true
  fi
  sudo_cmd nvpmodel -q > "${ENV_DIR}/nvpmodel_after.txt" 2>&1 || true
  sudo_cmd jetson_clocks --show > "${ENV_DIR}/jetson_clocks_after.txt" 2>&1 || true
}

capture_env
configure_power

PRIMARY_INPUTS=()

if [[ "${RUN_REALFRAMES}" == "1" ]]; then
  for dataset in ${AGX_DATASETS}; do
    frame_dir="$(dataset_frame_dir "${dataset}")"
    n_value="$(dataset_n "${dataset}")"
    frame_limit="$(dataset_frame_limit "${dataset}")"
    if [[ -z "${frame_dir}" || ! -d "${frame_dir}" ]]; then
      echo "Skipping dataset ${dataset}: missing frame dir ${frame_dir}" | tee -a "${FAILED_LOG}"
      continue
    fi
    for deadline in ${DEADLINES_MS}; do
      tag="$(deadline_tag "${deadline}")"
      run_name="realframes_${dataset}_d${tag}"
      out_dir="${RESULT_ROOT}/realframes/${dataset}_d${tag}"
      mkdir -p "${out_dir}"
      tegra_pid="$(start_tegrastats "${run_name}")"
      run_logged "${run_name}" "${PYTHON_BIN}" scripts/run_autodl_perception_semantics.py \
        --out-dir "${out_dir}" \
        --frame-dir "${frame_dir}" \
        --frame-limit "${frame_limit}" \
        --models ${MODELS} \
        --weights "${WEIGHTS}" \
        --audit-model "${AUDIT_MODEL}" \
        --audit-weights "${AUDIT_WEIGHTS}" \
        --device "${DEVICE}" \
        --n "${n_value}" \
        --seeds ${SEEDS} \
        --warmup "${WARMUP}" \
        --deadline-ms "${deadline}" \
        --gpu-stress-size "${GPU_STRESS_SIZE}" \
        --gpu-stress-repeats "${GPU_STRESS_REPEATS}" \
        --coupled-stress-extra-repeats "${COUPLED_STRESS_EXTRA_REPEATS}" \
        --circa-low-quantile "${CIRCA_LOW_QUANTILE}" \
        --circa-high-quantile "${CIRCA_HIGH_QUANTILE}" \
        --token-audit-cost-ms "${TOKEN_AUDIT_COST_MS}" \
        --bucket-capacity "${BUCKET_CAPACITY}" \
        --replenish-rate "${REPLENISH_RATE}" \
        --methods ${METHODS} \
        --progress-every "${PROGRESS_EVERY}"
      stop_tegrastats "${tegra_pid}"

      if [[ -d "${out_dir}/raw" ]]; then
        run_logged "audit_bound_${dataset}_d${tag}" "${PYTHON_BIN}" scripts/validate_audit_bound.py \
          --input-root "${out_dir}/raw" \
          --out "${out_dir}/tables/audit_bound_validation.csv" \
          --methods CIRCA-RT CIRCA-RT-Slack \
          --cost-cols audit_token_charge_ms
      fi

      if [[ "${deadline}" == "${PRIMARY_DEADLINE_MS}" ]]; then
        PRIMARY_INPUTS+=("${dataset}=${out_dir}")
      fi
    done
  done
fi

if [[ "${RUN_BACKEND_MATRIX}" == "1" ]]; then
  backend_out="${RESULT_ROOT}/backend_torch_onnx_trt"
  tegra_pid="$(start_tegrastats "backend_torch_onnx_trt")"
  run_logged "backend_torch_onnx_trt" "${PYTHON_BIN}" scripts/run_autodl_full_experiment.py \
    --out-dir "${backend_out}" \
    --models ${BACKEND_MODELS} \
    --runtimes ${BACKEND_RUNTIMES} \
    --seeds ${SEEDS} \
    --n "${BACKEND_N}" \
    --warmup "${BACKEND_WARMUP}" \
    --deadline-ms "${BACKEND_DEADLINE_MS}" \
    --interference-size "${BACKEND_INTERFERENCE_SIZE}" \
    --ort-interference-size "${BACKEND_ORT_INTERFERENCE_SIZE}"
  stop_tegrastats "${tegra_pid}"
fi

if [[ "${RUN_DERIVED_ANALYSES}" == "1" && ${#PRIMARY_INPUTS[@]} -gt 0 ]]; then
  run_logged "deadline_safe_admission" "${PYTHON_BIN}" scripts/analyze_deadline_safe_admission.py \
    --inputs "${PRIMARY_INPUTS[@]}" \
    --out-dir "${RESULT_ROOT}/deadline_safe_admission" \
    --methods RFF-HSIC ConditionalRFF-HSIC ContextAwareConformal TimingThreshold \
    --quantiles 0.95 0.99 \
    --offsets-ms 0 1 2 4

  run_logged "circa_slack_sensitivity" "${PYTHON_BIN}" scripts/derive_slack_from_circa_scored.py \
    --inputs "${PRIMARY_INPUTS[@]}" \
    --out-dir "${RESULT_ROOT}/circa_slack_sensitivity" \
    --method CIRCA-RT-Slack \
    --slack-margin-ms 0.0 \
    --deadlines-ms 33.333 50 100
fi

mapfile -t TEGRA_LOGS < <(find "${LOG_DIR}" -maxdepth 1 -name 'tegrastats_*.log' -type f | sort)
if [[ ${#TEGRA_LOGS[@]} -gt 0 ]]; then
  run_logged "parse_tegrastats" "${PYTHON_BIN}" scripts/parse_tegrastats.py \
    "${TEGRA_LOGS[@]}" \
    --out "${TABLE_DIR}/tegrastats_samples.csv" \
    --summary-out "${TABLE_DIR}/tegrastats_summary.csv"
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  run_logged "summarize_agx_comprehensive" "${PYTHON_BIN}" scripts/summarize_agx_orin64_comprehensive.py \
    --result-root "${RESULT_ROOT}"
fi

echo
echo "AGX Orin 64GB comprehensive results: ${RESULT_ROOT}"
if [[ -s "${FAILED_LOG}" ]]; then
  echo "Some commands failed; inspect ${FAILED_LOG}"
else
  echo "No command failures recorded."
fi
