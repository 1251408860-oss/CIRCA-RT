#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results_agx_orin64_ros2_closed_loop_${STAMP}}"
ENV_DIR="${RESULT_ROOT}/env"
LOG_DIR="${RESULT_ROOT}/logs"
TABLE_DIR="${RESULT_ROOT}/tables"

ROS_SETUP="${ROS_SETUP:-}"
FRAME_DIR="${FRAME_DIR:-${ROOT}/data/frames_av2}"
FRAME_LIMIT="${FRAME_LIMIT:-900}"
MODELS="${MODELS:-mobilenet_v2}"
SEEDS="${SEEDS:-7 8 9}"
N="${N:-600}"
PERIOD_MS="${PERIOD_MS:-33.333}"
DEADLINE_MS="${DEADLINE_MS:-33.333}"
METHOD="${METHOD:-CIRCA-RT-Slack}"
DEVICE="${DEVICE:-cuda}"
WEIGHTS="${WEIGHTS:-default}"
AUDIT_MODEL="${AUDIT_MODEL:-resnet18}"
AUDIT_WEIGHTS="${AUDIT_WEIGHTS:-default}"
TOKEN_AUDIT_COST_MS="${TOKEN_AUDIT_COST_MS:-4.0}"
BUCKET_CAPACITY="${BUCKET_CAPACITY:-12.0}"
REPLENISH_RATE="${REPLENISH_RATE:-0.45}"
SLACK_MARGIN_MS="${SLACK_MARGIN_MS:-0.0}"
GPU_STRESS_SIZE="${GPU_STRESS_SIZE:-512}"
GPU_STRESS_REPEATS="${GPU_STRESS_REPEATS:-1}"
COUPLED_STRESS_EXTRA_REPEATS="${COUPLED_STRESS_EXTRA_REPEATS:-4}"
ATTACK_LEN="${ATTACK_LEN:-80}"
CLIP_LEN="${CLIP_LEN:-12}"
PAYLOAD_MODE="${PAYLOAD_MODE:-jpeg_b64}"
JPEG_QUALITY="${JPEG_QUALITY:-75}"
PUBLISH_RESIZE="${PUBLISH_RESIZE:-320}"
TEGRA_INTERVAL_MS="${TEGRA_INTERVAL_MS:-500}"
SET_JETSON_CLOCKS="${SET_JETSON_CLOCKS:-1}"
NVP_MODEL_ID="${NVP_MODEL_ID:-}"

mkdir -p "${ENV_DIR}" "${LOG_DIR}" "${TABLE_DIR}"
cd "${ROOT}" || exit 1

if [[ -n "${ROS_SETUP}" && -f "${ROS_SETUP}" ]]; then
  # shellcheck source=/dev/null
  source "${ROS_SETUP}"
elif [[ -f /opt/ros/foxy/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/foxy/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
fi

FAILED_LOG="${LOG_DIR}/failed_commands.log"
: > "${FAILED_LOG}"

sudo_cmd() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
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
    exit "${rc}"
  fi
}

start_tegrastats() {
  local log_file="${LOG_DIR}/tegrastats_ros2_closed_loop.log"
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

capture_env() {
  {
    date
    hostname
    uname -a
    echo "ROS_DISTRO=${ROS_DISTRO:-}"
    echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-}"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "FRAME_DIR=${FRAME_DIR}"
    cat /proc/device-tree/model 2>/dev/null || true
    cat /etc/nv_tegra_release 2>/dev/null || true
    lsb_release -a 2>/dev/null || true
    which ros2 || true
    ros2 --version || true
    df -h
    free -h
  } > "${ENV_DIR}/system_ros2_snapshot.txt" 2>&1

  "${PYTHON_BIN}" - <<'PY' > "${ENV_DIR}/python_ros2_snapshot.txt" 2>&1
import json
import platform
out = {"python": platform.python_version()}
for name in ["rclpy", "std_msgs", "torch", "torchvision", "onnxruntime"]:
    try:
        mod = __import__(name)
        out[name] = getattr(mod, "__version__", "importable")
    except Exception as exc:
        out[name] = f"ERROR: {exc!r}"
print(json.dumps(out, indent=2))
PY

  PYTHON_BIN="${PYTHON_BIN}" bash scripts/jetson_env_check.sh > "${ENV_DIR}/jetson_env_check.log" 2>&1 || true
  sudo_cmd nvpmodel -q > "${ENV_DIR}/nvpmodel_before.txt" 2>&1 || true
  sudo_cmd jetson_clocks --show > "${ENV_DIR}/jetson_clocks_before.txt" 2>&1 || true
}

configure_power() {
  if [[ -n "${NVP_MODEL_ID}" ]]; then
    sudo_cmd nvpmodel -m "${NVP_MODEL_ID}" > "${ENV_DIR}/nvpmodel_set.txt" 2>&1 || true
  fi
  if [[ "${SET_JETSON_CLOCKS}" == "1" ]] && command -v jetson_clocks >/dev/null 2>&1; then
    sudo_cmd jetson_clocks > "${ENV_DIR}/jetson_clocks_set.txt" 2>&1 || true
  fi
  sudo_cmd nvpmodel -q > "${ENV_DIR}/nvpmodel_after.txt" 2>&1 || true
  sudo_cmd jetson_clocks --show > "${ENV_DIR}/jetson_clocks_after.txt" 2>&1 || true
}

capture_env
configure_power

tegra_pid="$(start_tegrastats)"
run_logged "agx_ros2_closed_loop" "${PYTHON_BIN}" scripts/run_agx_ros2_closed_loop.py \
  --out-dir "${RESULT_ROOT}" \
  --frame-dir "${FRAME_DIR}" \
  --frame-limit "${FRAME_LIMIT}" \
  --models ${MODELS} \
  --weights "${WEIGHTS}" \
  --audit-model "${AUDIT_MODEL}" \
  --audit-weights "${AUDIT_WEIGHTS}" \
  --node-python "${PYTHON_BIN}" \
  --device "${DEVICE}" \
  --n "${N}" \
  --seeds ${SEEDS} \
  --period-ms "${PERIOD_MS}" \
  --deadline-ms "${DEADLINE_MS}" \
  --method "${METHOD}" \
  --token-audit-cost-ms "${TOKEN_AUDIT_COST_MS}" \
  --bucket-capacity "${BUCKET_CAPACITY}" \
  --replenish-rate "${REPLENISH_RATE}" \
  --slack-margin-ms "${SLACK_MARGIN_MS}" \
  --gpu-stress-size "${GPU_STRESS_SIZE}" \
  --gpu-stress-repeats "${GPU_STRESS_REPEATS}" \
  --coupled-stress-extra-repeats "${COUPLED_STRESS_EXTRA_REPEATS}" \
  --attack-len "${ATTACK_LEN}" \
  --clip-len "${CLIP_LEN}" \
  --payload-mode "${PAYLOAD_MODE}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --publish-resize "${PUBLISH_RESIZE}"
stop_tegrastats "${tegra_pid}"

run_logged "audit_bound_ros2_closed_loop" "${PYTHON_BIN}" scripts/validate_audit_bound.py \
  --input-root "${RESULT_ROOT}/raw" \
  --out "${TABLE_DIR}/audit_bound_validation.csv" \
  --methods "${METHOD}" \
  --cost-cols audit_token_charge_ms audit_charge_ms

if compgen -G "${LOG_DIR}/tegrastats_*.log" >/dev/null; then
  # shellcheck disable=SC2046
  run_logged "parse_tegrastats_ros2_closed_loop" "${PYTHON_BIN}" scripts/parse_tegrastats.py \
    $(find "${LOG_DIR}" -maxdepth 1 -name 'tegrastats_*.log' -type f | sort) \
    --out "${TABLE_DIR}/tegrastats_samples.csv" \
    --summary-out "${TABLE_DIR}/tegrastats_summary.csv"
fi

echo
echo "AGX ROS2 closed-loop results: ${RESULT_ROOT}"
if [[ -s "${FAILED_LOG}" ]]; then
  echo "Some commands failed; inspect ${FAILED_LOG}"
else
  echo "No command failures recorded."
fi
