#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PRESSURE_PROFILE="${PRESSURE_PROFILE:-paper}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_RESULT_ROOT="${BASE_RESULT_ROOT:-${ROOT}/results_agx_orin64_pressure_${PRESSURE_PROFILE}_${STAMP}}"

if [[ -d "${ROOT}/torch_cache" ]]; then
  export TORCH_HOME="${TORCH_HOME:-${ROOT}/torch_cache}"
fi

case "${PRESSURE_PROFILE}" in
  smoke)
    : "${PRESSURE_LEVELS:=idle:0:0:0 medium:1024:2:12}"
    : "${AGX_DATASETS:=av2}"
    : "${DEADLINES_MS:=25 33.333}"
    : "${MODELS:=mobilenet_v2}"
    : "${SEEDS:=7}"
    : "${N_AV2:=120}"
    : "${N_DROID:=120}"
    : "${N_NUIMAGES:=120}"
    : "${FRAME_LIMIT_AV2:=240}"
    : "${FRAME_LIMIT_DROID:=240}"
    : "${FRAME_LIMIT_NUIMAGES:=217}"
    ;;
  paper)
    : "${PRESSURE_LEVELS:=idle:0:0:0 mild:512:1:4 medium:1024:2:12 high:1536:3:24}"
    : "${AGX_DATASETS:=av2 droid nuimages}"
    : "${DEADLINES_MS:=16.667 20 25 33.333 50}"
    : "${MODELS:=mobilenet_v2}"
    : "${SEEDS:=7 8 9}"
    : "${N_AV2:=300}"
    : "${N_DROID:=300}"
    : "${N_NUIMAGES:=217}"
    : "${FRAME_LIMIT_AV2:=600}"
    : "${FRAME_LIMIT_DROID:=600}"
    : "${FRAME_LIMIT_NUIMAGES:=217}"
    ;;
  max)
    : "${PRESSURE_LEVELS:=idle:0:0:0 mild:512:1:4 medium:1024:2:12 high:1536:3:24 saturation:2048:4:36}"
    : "${AGX_DATASETS:=av2 droid nuimages}"
    : "${DEADLINES_MS:=16.667 20 25 33.333 40 50 100}"
    : "${MODELS:=squeezenet1_1 mobilenet_v2 resnet18}"
    : "${SEEDS:=7 8 9 10 11}"
    : "${N_AV2:=600}"
    : "${N_DROID:=600}"
    : "${N_NUIMAGES:=217}"
    : "${FRAME_LIMIT_AV2:=1200}"
    : "${FRAME_LIMIT_DROID:=900}"
    : "${FRAME_LIMIT_NUIMAGES:=217}"
    ;;
  *)
    echo "Unknown PRESSURE_PROFILE=${PRESSURE_PROFILE}; use smoke, paper, max, or set all variables explicitly." >&2
    exit 2
    ;;
esac

: "${METHODS:=CIRCA-RT CIRCA-RT-Slack ConditionalRFF-HSIC ContextAwareConformal RFF-HSIC TimingThreshold SemanticThreshold AlwaysAudit RandomBudgetAudit}"
: "${WARMUP:=60}"
: "${DEVICE:=cuda}"
: "${WEIGHTS:=default}"
: "${AUDIT_MODEL:=resnet18}"
: "${AUDIT_WEIGHTS:=default}"
: "${TOKEN_AUDIT_COST_MS:=4.0}"
: "${BUCKET_CAPACITY:=12.0}"
: "${REPLENISH_RATE:=0.45}"
: "${RUN_BACKEND_MATRIX:=0}"
: "${RUN_DERIVED_ANALYSES:=1}"
: "${RUN_SUMMARY:=1}"
: "${SET_JETSON_CLOCKS:=1}"
: "${TEGRA_INTERVAL_MS:=250}"
: "${FAIL_FAST:=0}"
: "${PRESSURE_MISS_THRESHOLD:=0.01}"

mkdir -p "${BASE_RESULT_ROOT}/logs" "${BASE_RESULT_ROOT}/tables"
cd "${ROOT}" || exit 1

safe_name() {
  echo "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

write_plan() {
  cat > "${BASE_RESULT_ROOT}/pressure_plan.txt" <<EOF
created_at=${STAMP}
profile=${PRESSURE_PROFILE}
base_result_root=${BASE_RESULT_ROOT}
pressure_levels=${PRESSURE_LEVELS}
datasets=${AGX_DATASETS}
deadlines_ms=${DEADLINES_MS}
models=${MODELS}
seeds=${SEEDS}
methods=${METHODS}
n_av2=${N_AV2}
n_droid=${N_DROID}
n_nuimages=${N_NUIMAGES}
frame_limit_av2=${FRAME_LIMIT_AV2}
frame_limit_droid=${FRAME_LIMIT_DROID}
frame_limit_nuimages=${FRAME_LIMIT_NUIMAGES}
run_backend_matrix=${RUN_BACKEND_MATRIX}
run_derived_analyses=${RUN_DERIVED_ANALYSES}
tegrastats_interval_ms=${TEGRA_INTERVAL_MS}
EOF
}

capture_pressure_env() {
  {
    date
    hostname
    uname -a
    cat /proc/device-tree/model 2>/dev/null || true
    cat /etc/nv_tegra_release 2>/dev/null || true
    "${PYTHON_BIN}" --version || true
    sudo nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || true
    sudo nvpmodel -q --verbose 2>/dev/null || true
    sudo jetson_clocks --show 2>/dev/null || jetson_clocks --show 2>/dev/null || true
    df -h
    free -h
  } > "${BASE_RESULT_ROOT}/logs/pressure_system_snapshot.log" 2>&1

  PYTHON_BIN="${PYTHON_BIN}" bash scripts/jetson_env_check.sh \
    > "${BASE_RESULT_ROOT}/logs/pressure_jetson_env_check.log" 2>&1 || true
}

run_level() {
  local level_index="$1"
  local spec="$2"
  local label size repeats extra
  IFS=':' read -r label size repeats extra <<< "${spec}"
  if [[ -z "${label:-}" || -z "${size:-}" || -z "${repeats:-}" || -z "${extra:-}" ]]; then
    echo "Invalid pressure level '${spec}', expected label:gpu_stress_size:gpu_stress_repeats:coupled_extra_repeats" >&2
    return 2
  fi

  local safe_label
  safe_label="$(safe_name "${label}")"
  local result_root="${BASE_RESULT_ROOT}/level_${level_index}_${safe_label}"
  local run_log="${BASE_RESULT_ROOT}/logs/level_${level_index}_${safe_label}.log"
  mkdir -p "${result_root}"

  cat > "${result_root}/pressure_meta.json" <<EOF
{
  "pressure_profile": "${PRESSURE_PROFILE}",
  "pressure_level_index": ${level_index},
  "pressure_label": "${label}",
  "gpu_stress_size": ${size},
  "gpu_stress_repeats": ${repeats},
  "coupled_stress_extra_repeats": ${extra},
  "datasets": "${AGX_DATASETS}",
  "deadlines_ms": "${DEADLINES_MS}",
  "models": "${MODELS}",
  "seeds": "${SEEDS}",
  "n_av2": ${N_AV2},
  "n_droid": ${N_DROID},
  "n_nuimages": ${N_NUIMAGES}
}
EOF

  echo
  echo "=== pressure level ${level_index}: ${label} size=${size} repeats=${repeats} coupled_extra=${extra} ==="
  echo "result_root=${result_root}"

  PYTHON_BIN="${PYTHON_BIN}" \
  RESULT_ROOT="${result_root}" \
  AGX_DATASETS="${AGX_DATASETS}" \
  DEADLINES_MS="${DEADLINES_MS}" \
  PRIMARY_DEADLINE_MS="${PRIMARY_DEADLINE_MS:-33.333}" \
  MODELS="${MODELS}" \
  METHODS="${METHODS}" \
  SEEDS="${SEEDS}" \
  WARMUP="${WARMUP}" \
  DEVICE="${DEVICE}" \
  WEIGHTS="${WEIGHTS}" \
  AUDIT_MODEL="${AUDIT_MODEL}" \
  AUDIT_WEIGHTS="${AUDIT_WEIGHTS}" \
  N_AV2="${N_AV2}" \
  N_DROID="${N_DROID}" \
  N_NUIMAGES="${N_NUIMAGES}" \
  FRAME_LIMIT_AV2="${FRAME_LIMIT_AV2}" \
  FRAME_LIMIT_DROID="${FRAME_LIMIT_DROID}" \
  FRAME_LIMIT_NUIMAGES="${FRAME_LIMIT_NUIMAGES}" \
  GPU_STRESS_SIZE="${size}" \
  GPU_STRESS_REPEATS="${repeats}" \
  COUPLED_STRESS_EXTRA_REPEATS="${extra}" \
  TOKEN_AUDIT_COST_MS="${TOKEN_AUDIT_COST_MS}" \
  BUCKET_CAPACITY="${BUCKET_CAPACITY}" \
  REPLENISH_RATE="${REPLENISH_RATE}" \
  RUN_BACKEND_MATRIX="${RUN_BACKEND_MATRIX}" \
  RUN_DERIVED_ANALYSES="${RUN_DERIVED_ANALYSES}" \
  RUN_SUMMARY="${RUN_SUMMARY}" \
  SET_JETSON_CLOCKS="${SET_JETSON_CLOCKS}" \
  TEGRA_INTERVAL_MS="${TEGRA_INTERVAL_MS}" \
  FAIL_FAST="${FAIL_FAST}" \
  bash scripts/run_agx_orin64_comprehensive_experiment.sh 2>&1 | tee "${run_log}"

  local rc=${PIPESTATUS[0]}
  echo "${label},${rc},${result_root}" >> "${BASE_RESULT_ROOT}/tables/pressure_level_status.csv"
  if [[ ${rc} -ne 0 && "${FAIL_FAST}" == "1" ]]; then
    exit "${rc}"
  fi
}

write_plan
capture_pressure_env
echo "pressure_label,return_code,result_root" > "${BASE_RESULT_ROOT}/tables/pressure_level_status.csv"

level_index=0
for spec in ${PRESSURE_LEVELS}; do
  run_level "${level_index}" "${spec}"
  level_index=$((level_index + 1))
done

"${PYTHON_BIN}" scripts/summarize_agx_orin64_pressure_sweep.py \
  --input-root "${BASE_RESULT_ROOT}" \
  --miss-threshold "${PRESSURE_MISS_THRESHOLD}" \
  > "${BASE_RESULT_ROOT}/logs/summarize_pressure_sweep.log" 2>&1 || true

echo
echo "AGX Orin 64GB pressure sweep root: ${BASE_RESULT_ROOT}"
echo "Summary report: ${BASE_RESULT_ROOT}/PRESSURE_SWEEP_REPORT.md"
