# AGX Orin 64GB Real Device Runbook

Updated: 2026-05-24

Purpose: when a real Jetson AGX Orin 64GB device becomes available, use this
runbook to collect a clean RTSS-ready edge-device evidence package for CIRCA-RT.

This run must create a new result directory. Do not reuse historical AGX output
directories, derived stress envelopes, or any previous fake/misnamed data.

Recommended result name:

```text
results_jetson_agx_orin64_real_202605xx/
```

## 1. Paper Claim Enabled

If the run completes successfully, the paper may claim:

> We evaluate CIRCA-RT on a real Jetson AGX Orin 64GB edge device using a
> replay-driven online perception loop. The run records provider status,
> per-frame inference latency, CIRCA-RT audit decisions, deadline misses, and
> device-side tegrastats power/thermal samples.

If TensorRT is active, the paper may say `TensorRT` or `ONNXRuntime TensorRT EP`
only if the provider/engine logs prove it. If TensorRT fails and CUDA/PyTorch is
used instead, the paper must say CUDA/PyTorch Jetson execution, not TensorRT.

Do not claim:

- full autonomous-driving closed loop
- physical TSN validation
- hard real-time guarantee under arbitrary GPU contention
- AGX Orin results unless they come from this new real-device run
- TensorRT results if the runtime silently falls back to CUDA or CPU

## 2. Minimum Device Requirements

The device is usable only if all of these hold:

- SSH works.
- `sudo` works.
- `tegrastats` works.
- `nvpmodel` works.
- `jetson_clocks` works.
- CUDA or TensorRT inference works.
- At least 40 GB free disk is available.
- Results can be downloaded after the run.

No-go conditions:

- no `tegrastats`
- no provider/runtime evidence
- no real extra audit computation
- only offline CSV scoring with no device-side inference
- result directory missing raw traces or environment logs

## 3. First Commands After SSH

Run these before uploading the full project:

```bash
hostname
uname -a
cat /etc/nv_tegra_release
lsb_release -a || true
df -h
free -h
```

Check Jetson tools:

```bash
which tegrastats
which nvpmodel
which jetson_clocks
sudo nvpmodel -q
sudo jetson_clocks --show
```

Check Python, CUDA, TensorRT, PyTorch, and ONNXRuntime:

```bash
python3 --version

python3 - <<'PY'
try:
    import tensorrt as trt
    print("tensorrt", trt.__version__)
except Exception as e:
    print("tensorrt failed:", e)

try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
except Exception as e:
    print("torch failed:", e)

try:
    import onnxruntime as ort
    print("onnxruntime", ort.__version__)
    print("providers", ort.get_available_providers())
except Exception as e:
    print("onnxruntime failed:", e)
PY
```

On Jetson, `nvidia-smi` may be unavailable. Use `tegrastats`, `nvpmodel`, and
runtime provider logs as the main device evidence.

## 4. Device Power Setup

Use maximum-performance mode for the first valid run:

```bash
sudo nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --show
```

If time remains, add a separate power-capped run after the main run. Use the mode
ID shown by `sudo nvpmodel -q`; do not guess mode numbers:

```bash
sudo nvpmodel -m <MODE_ID>
sudo jetson_clocks
sudo nvpmodel -q
```

Keep power-capped results in a separate directory:

```text
results_jetson_agx_orin64_real_30w_202605xx/
```

## 5. Project Upload Layout

Use a clean working directory on the device:

```bash
mkdir -p ~/rtss_agx
cd ~/rtss_agx
```

Upload only the required project files, frame data, and scripts. Avoid uploading
old AGX result directories or unrelated large baseline folders.

Expected project root on device:

```text
~/rtss_agx/exp_begin/
```

Before running experiments:

```bash
cd ~/rtss_agx/exp_begin
bash scripts/jetson_env_check.sh | tee jetson_env_check_first.log
python3 scripts/run_jetson_tensorrt_engine_check.py \
  --out results_jetson_agx_orin64_env/provider_status.json
```

## 6. Smoke Run First

Do not start the full matrix before a smoke run passes.

```bash
cd ~/rtss_agx/exp_begin

OUT_DIR=results_jetson_agx_orin64_real_smoke \
N=300 \
SEEDS="7" \
DEADLINE_MS=33.333 \
PLATFORM_LABEL=jetson_agx_orin64_real \
PYTHON_BIN=python3 \
bash scripts/run_jetson_orin_validation.sh
```

Smoke success criteria:

- raw traces exist
- scored raw outputs exist
- `raw_logs/env_check.log` exists
- `raw_logs/tegrastats_*.log` exists
- `tables/audit_bound_validation.csv` exists
- TensorRT/CUDA provider evidence is saved
- no CPU fallback is reported as a TensorRT result

If smoke fails, fix environment first. Do not continue to the full run.

## 7. Main Run

Preferred comprehensive run:

```bash
cd ~/rtss_agx/exp_begin

PYTHON_BIN=python3 \
RESULT_ROOT=results_agx_orin64_comprehensive_202605xx \
AGX_DATASETS="av2 droid nuimages" \
DEADLINES_MS="33.333 50" \
MODELS="mobilenet_v2 resnet18" \
SEEDS="7 8 9" \
RUN_BACKEND_MATRIX=1 \
RUN_DERIVED_ANALYSES=1 \
bash scripts/run_agx_orin64_comprehensive_experiment.sh
```

This comprehensive script performs:

- environment/provider capture
- optional `jetson_clocks`
- AV2/DROID/nuImages real-frame perception monitoring
- 33.333 ms and 50 ms deadline runs
- `CIRCA-RT` and `CIRCA-RT-Slack`
- ContextAwareConformal, ConditionalRFF-HSIC, RFF-HSIC, TimingThreshold,
  SemanticThreshold, AlwaysAudit, RandomBudgetAudit
- PyTorch/ONNX CUDA/TensorRT backend matrix through
  `run_autodl_full_experiment.py`
- `tegrastats` logs per run
- audit-bound validation
- generic score + deadline-safe admission analysis
- CIRCA slack sensitivity analysis
- final summary tables and figures

The expected output is:

```text
results_agx_orin64_comprehensive_202605xx/
  env/
  logs/
  realframes/
  backend_torch_onnx_trt/
  deadline_safe_admission/
  circa_slack_sensitivity/
  tables/
  figures/
  AGX_ORIN64_COMPREHENSIVE_REPORT.md
```

For a faster smoke version of the comprehensive run:

```bash
cd ~/rtss_agx/exp_begin

PYTHON_BIN=python3 \
RESULT_ROOT=results_agx_orin64_comprehensive_smoke \
AGX_DATASETS="av2" \
DEADLINES_MS="33.333" \
MODELS="mobilenet_v2" \
SEEDS="7" \
N_AV2=300 \
FRAME_LIMIT_AV2=300 \
RUN_BACKEND_MATRIX=0 \
bash scripts/run_agx_orin64_comprehensive_experiment.sh
```

Legacy compact run:

Run the compact RTSS-ready matrix:

```bash
cd ~/rtss_agx/exp_begin

OUT_DIR=results_jetson_agx_orin64_real_202605xx \
N=1500 \
SEEDS="7 8 9" \
DEADLINE_MS=33.333 \
PLATFORM_LABEL=jetson_agx_orin64_real \
PYTHON_BIN=python3 \
bash scripts/run_jetson_orin_validation.sh
```

The compact validation script covers:

```text
nominal
mode_shift
semantic_corruption
gpu_interference
stale_replay_attack
coupled_semantic_timing_attack
```

and these policies:

```text
CIRCA-RT
CIRCA-RT-Slack
ContextAwareConformal
ConditionalRFF-HSIC
RFF-HSIC
SemanticThreshold
TimingThreshold
AlwaysAudit
RandomBudgetAudit
```

This is enough for the main Jetson evidence block. Do not expand to a much
larger baseline suite unless the first run is already complete and downloaded.

## 8. Optional Stress Run

AGX Orin 64GB is powerful, so MobileNetV2/ResNet18 may be too light. If time
remains, run one stronger stress condition:

```bash
cd ~/rtss_agx/exp_begin

OUT_DIR=results_jetson_agx_orin64_real_stress_202605xx \
N=1500 \
SEEDS="7 8 9" \
DEADLINE_MS=33.333 \
GPU_STRESS_SIZE=3072 \
GPU_STRESS_REPEATS=3 \
COUPLED_STRESS_EXTRA_REPEATS=32 \
PLATFORM_LABEL=jetson_agx_orin64_real_stress \
PYTHON_BIN=python3 \
bash scripts/run_jetson_orin_validation.sh
```

Use this to answer the reviewer concern that AGX Orin is overpowered for small
models. The correct claim is not that CIRCA-RT survives arbitrary GPU
contention. The claim is that it bounds audit demand and exposes where stress
tails break the deadline.

## 9. Required Output Package

Each valid run directory must contain:

```text
raw_logs/env_check.log
raw_logs/tegrastats_*.log
logs/jetson_validation_*.log
traces/
raw/
tables/audit_bound_validation.csv
tables/tegrastats_samples.csv
tables/tegrastats_summary.csv
tables/*summary*.csv
```

Also save or generate:

```text
manifest/device_manifest.json
manifest/provider_status.json
manifest/environment.txt
checksums.sha256
run_command_log.txt
```

If these files are missing, the run is not paper-ready.

## 10. Download Back To Windows

Download the final valid directories into:

```text
F:\RTSS\exp_begin\results_jetson_agx_orin64_real_202605xx\
F:\RTSS\exp_begin\results_jetson_agx_orin64_real_stress_202605xx\
```

Then generate local summaries and figures:

```powershell
cd F:\RTSS\exp_begin
python scripts\summarize_jetson_perception_results.py `
  --out-dir results_jetson_agx_orin64_real_202605xx `
  --raw-logs-dir results_jetson_agx_orin64_real_202605xx\raw_logs `
  --platform-label jetson_agx_orin64_real

python scripts\summarize_agx_orin64_comprehensive.py `
  --root results_jetson_agx_orin64_real_202605xx
```

Use `scripts/summarize_agx_orin64_comprehensive.py` for current AGX summaries.
Do not regenerate figures from obsolete AGX package paths.

## 11. What To Report In The Paper

Main table columns:

```text
method
attack recall
false alarm
audit rate
monitor cost p99
audit cost p99
p99 latency
p999 latency
deadline miss ratio
```

Device table columns:

```text
device
JetPack version
runtime provider
power mode
deadline
frames
seeds
tegrastats samples
mean power
p95 power
max temperature
```

Required figures:

```text
Jetson recall vs audit rate
Jetson p99/p999 latency by policy
Jetson deadline sensitivity
Jetson power/thermal summary
```

Required validation:

```text
audit_bound_validation.csv has zero failures for CIRCA-RT
provider_status.json proves the active runtime
tegrastats_summary.csv proves device-side power/thermal sampling
```

## 12. Final Paper Wording

Use this if TensorRT is proven:

> The final edge-device experiment runs a replay-driven perception-monitoring
> loop on a Jetson AGX Orin 64GB device. The run uses TensorRT execution for
> the perception path, records per-frame CIRCA-RT audit decisions and deadline
> misses, and samples device power/thermal behavior with tegrastats.

Use this if TensorRT is not proven but CUDA is valid:

> The final edge-device experiment runs a replay-driven perception-monitoring
> loop on a Jetson AGX Orin 64GB device using CUDA inference. TensorRT is not
> claimed for this run.

Limitations to state explicitly:

- the run is replay-driven, not a full autonomous-driving stack
- heavy GPU interference may still violate p99 deadlines
- detection quality is empirical; the token bucket bounds audit demand
- platform-specific latency envelopes must be remeasured on each deployment

## 13. References

- NVIDIA Jetson AGX Orin product family:
  `https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/`
- NVIDIA JetPack SDK:
  `https://developer.nvidia.com/embedded/jetpack`
- NVIDIA tegrastats utility:
  `https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html`
