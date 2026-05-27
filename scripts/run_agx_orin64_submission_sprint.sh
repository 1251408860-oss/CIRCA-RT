#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_RESULT_ROOT="${BASE_RESULT_ROOT:-${ROOT}/results_agx_orin64_submission_sprint_${STAMP}}"
LOG_DIR="${BASE_RESULT_ROOT}/logs"
TABLE_DIR="${BASE_RESULT_ROOT}/tables"
PHASES="${PHASES:-env ros2_closed_loop resnet50_realframes backend_matrix pressure_short}"
FAIL_FAST="${FAIL_FAST:-0}"

mkdir -p "${LOG_DIR}" "${TABLE_DIR}"
cd "${ROOT}" || exit 1

if [[ -d "${ROOT}/torch_cache" ]]; then
  export TORCH_HOME="${TORCH_HOME:-${ROOT}/torch_cache}"
fi

if [[ -n "${ROS_SETUP:-}" && -f "${ROS_SETUP}" ]]; then
  # shellcheck source=/dev/null
  source "${ROS_SETUP}"
elif [[ -f /opt/ros/foxy/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/foxy/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
fi

echo "phase,return_code,result_hint" > "${TABLE_DIR}/stage_status.csv"

contains_phase() {
  local needle="$1"
  for phase in ${PHASES}; do
    [[ "${phase}" == "${needle}" ]] && return 0
  done
  return 1
}

run_stage() {
  local name="$1"
  local result_hint="$2"
  shift 2
  local log_file="${LOG_DIR}/${name}.log"
  echo
  echo "=== submission sprint stage: ${name} ==="
  echo "$*" | tee "${log_file}.cmd"
  "$@" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  echo "${name},${rc},${result_hint}" >> "${TABLE_DIR}/stage_status.csv"
  if [[ ${rc} -ne 0 && "${FAIL_FAST}" == "1" ]]; then
    exit "${rc}"
  fi
  return "${rc}"
}

cat > "${BASE_RESULT_ROOT}/SUBMISSION_SPRINT_PLAN.txt" <<EOF
created_at=${STAMP}
base_result_root=${BASE_RESULT_ROOT}
phases=${PHASES}
fail_fast=${FAIL_FAST}
ros_distro=${ROS_DISTRO:-}
rmw_implementation=${RMW_IMPLEMENTATION:-}

Recommended RTSS evidence stages:
1. env: AGX environment, power mode, TensorRT/ONNXRuntime provider status.
2. ros2_closed_loop: ROS2 image-frame publisher -> DDS -> CUDA perception monitor -> CIRCA-RT-Slack -> inline audit.
3. resnet50_realframes: ResNet50 real-frame replay across AV2/DROID/nuImages.
4. backend_matrix: PyTorch CUDA, ONNX CUDA, and ONNX TensorRT backend matrix.
5. pressure_short: compact deadline/pressure sweep for deadline-safety stress evidence.
EOF

if contains_phase env; then
  run_stage env_check "${BASE_RESULT_ROOT}/env" bash -lc "
    mkdir -p '${BASE_RESULT_ROOT}/env';
    date > '${BASE_RESULT_ROOT}/env/date.txt';
    uname -a > '${BASE_RESULT_ROOT}/env/uname.txt';
    (cat /proc/device-tree/model || true) > '${BASE_RESULT_ROOT}/env/device_model.txt' 2>&1;
    (cat /etc/nv_tegra_release || true) > '${BASE_RESULT_ROOT}/env/nv_tegra_release.txt' 2>&1;
    (which ros2 || true) > '${BASE_RESULT_ROOT}/env/which_ros2.txt' 2>&1;
    (ros2 --version || true) > '${BASE_RESULT_ROOT}/env/ros2_version.txt' 2>&1;
    PYTHON_BIN='${PYTHON_BIN}' bash scripts/jetson_env_check.sh > '${BASE_RESULT_ROOT}/env/jetson_env_check.log' 2>&1 || true;
    '${PYTHON_BIN}' scripts/run_jetson_tensorrt_engine_check.py --out '${BASE_RESULT_ROOT}/env/provider_status.json' > '${BASE_RESULT_ROOT}/env/provider_status.log' 2>&1 || true
  "
fi

if contains_phase ros2_closed_loop; then
  run_stage ros2_closed_loop "${BASE_RESULT_ROOT}/ros2_closed_loop" env \
    RESULT_ROOT="${BASE_RESULT_ROOT}/ros2_closed_loop" \
    FRAME_DIR="${ROS2_FRAME_DIR:-data/frames_av2}" \
    MODELS="${ROS2_MODELS:-mobilenet_v2}" \
    SEEDS="${ROS2_SEEDS:-7 8 9}" \
    N="${ROS2_N:-600}" \
    PERIOD_MS="${ROS2_PERIOD_MS:-33.333}" \
    DEADLINE_MS="${ROS2_DEADLINE_MS:-33.333}" \
    PAYLOAD_MODE="${ROS2_PAYLOAD_MODE:-jpeg_b64}" \
    JPEG_QUALITY="${ROS2_JPEG_QUALITY:-75}" \
    PUBLISH_RESIZE="${ROS2_PUBLISH_RESIZE:-320}" \
    bash scripts/run_agx_orin64_ros2_closed_loop.sh
fi

if contains_phase resnet50_realframes; then
  run_stage resnet50_realframes "${BASE_RESULT_ROOT}/realframes_resnet50" env \
    PYTHON_BIN="${PYTHON_BIN}" \
    RESULT_ROOT="${BASE_RESULT_ROOT}/realframes_resnet50" \
    AGX_DATASETS="${REALFRAME_DATASETS:-av2 droid nuimages}" \
    DEADLINES_MS="${REALFRAME_DEADLINES_MS:-33.333 50}" \
    PRIMARY_DEADLINE_MS="${REALFRAME_PRIMARY_DEADLINE_MS:-33.333}" \
    MODELS="${REALFRAME_MODELS:-resnet50}" \
    METHODS="${REALFRAME_METHODS:-CIRCA-RT CIRCA-RT-Slack ConditionalRFF-HSIC ContextAwareConformal RFF-HSIC TimingThreshold SemanticThreshold AlwaysAudit RandomBudgetAudit}" \
    SEEDS="${REALFRAME_SEEDS:-7 8 9}" \
    WARMUP="${REALFRAME_WARMUP:-50}" \
    DEVICE="${DEVICE:-cuda}" \
    WEIGHTS="${WEIGHTS:-default}" \
    AUDIT_MODEL="${AUDIT_MODEL:-resnet18}" \
    AUDIT_WEIGHTS="${AUDIT_WEIGHTS:-default}" \
    N_AV2="${N_AV2:-900}" \
    N_DROID="${N_DROID:-900}" \
    N_NUIMAGES="${N_NUIMAGES:-217}" \
    FRAME_LIMIT_AV2="${FRAME_LIMIT_AV2:-1200}" \
    FRAME_LIMIT_DROID="${FRAME_LIMIT_DROID:-900}" \
    FRAME_LIMIT_NUIMAGES="${FRAME_LIMIT_NUIMAGES:-217}" \
    GPU_STRESS_SIZE="${GPU_STRESS_SIZE:-2048}" \
    GPU_STRESS_REPEATS="${GPU_STRESS_REPEATS:-2}" \
    COUPLED_STRESS_EXTRA_REPEATS="${COUPLED_STRESS_EXTRA_REPEATS:-24}" \
    RUN_BACKEND_MATRIX=0 \
    RUN_DERIVED_ANALYSES=1 \
    RUN_SUMMARY=1 \
    bash scripts/run_agx_orin64_comprehensive_experiment.sh
fi

if contains_phase backend_matrix; then
  run_stage backend_matrix "${BASE_RESULT_ROOT}/backend_matrix" env \
    PYTHON_BIN="${PYTHON_BIN}" \
    RESULT_ROOT="${BASE_RESULT_ROOT}/backend_matrix" \
    RUN_REALFRAMES=0 \
    RUN_BACKEND_MATRIX=1 \
    RUN_DERIVED_ANALYSES=0 \
    RUN_SUMMARY=1 \
    BACKEND_MODELS="${BACKEND_MODELS:-mobilenet_v2 resnet18 resnet50 squeezenet1_1}" \
    BACKEND_RUNTIMES="${BACKEND_RUNTIMES:-torch_cuda onnx_cuda onnx_tensorrt}" \
    SEEDS="${BACKEND_SEEDS:-7 8 9}" \
    BACKEND_N="${BACKEND_N:-1200}" \
    BACKEND_WARMUP="${BACKEND_WARMUP:-80}" \
    BACKEND_DEADLINE_MS="${BACKEND_DEADLINE_MS:-33.333}" \
    BACKEND_INTERFERENCE_SIZE="${BACKEND_INTERFERENCE_SIZE:-512}" \
    BACKEND_ORT_INTERFERENCE_SIZE="${BACKEND_ORT_INTERFERENCE_SIZE:-384}" \
    bash scripts/run_agx_orin64_comprehensive_experiment.sh
fi

if contains_phase pressure_short; then
  run_stage pressure_short "${BASE_RESULT_ROOT}/pressure_short" env \
    PYTHON_BIN="${PYTHON_BIN}" \
    BASE_RESULT_ROOT="${BASE_RESULT_ROOT}/pressure_short" \
    bash scripts/run_agx_orin64_pressure_short_value.sh
fi

cat > "${BASE_RESULT_ROOT}/RETURN_RESULTS_COMMANDS.txt" <<EOF
From the package root on AGX, return compact results with:

tar -czf agx_submission_sprint_${STAMP}_results.tar.gz -C "$(dirname "${BASE_RESULT_ROOT}")" "$(basename "${BASE_RESULT_ROOT}")"
sha256sum agx_submission_sprint_${STAMP}_results.tar.gz

Primary status file:
${TABLE_DIR}/stage_status.csv
EOF

echo
echo "AGX submission sprint result root: ${BASE_RESULT_ROOT}"
echo "Stage status: ${TABLE_DIR}/stage_status.csv"
echo "Return commands: ${BASE_RESULT_ROOT}/RETURN_RESULTS_COMMANDS.txt"
