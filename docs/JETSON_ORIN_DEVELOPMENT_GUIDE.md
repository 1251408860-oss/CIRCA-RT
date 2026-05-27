# Jetson Orin Development Guide For CIRCA-RT

Updated: 2026-05-19

This document describes how to run the final edge-device validation for CIRCA-RT on a real NVIDIA Jetson Orin. The goal is a small but complete RTSS-style validation, not a full rerun of all AutoDL experiments.

## 1. Target Device

Preferred device:

- Jetson AGX Orin 64GB
- JetPack 6.x
- Ubuntu 22.04
- CUDA 12.x
- TensorRT 10.x
- cuDNN installed
- SSH access
- `sudo` or root permission
- `tegrastats` available

Acceptable fallback:

- Jetson Orin NX 16GB
- JetPack 6.x
- TensorRT available

Do not use as the main edge result:

- Orin Nano
- Xavier
- TX2
- ordinary RTX cloud server
- AutoDL RTX server

## 2. What This Jetson Experiment Must Prove

The Jetson experiment should support this claim:

> CIRCA-RT preserves low audit cost and low tail-latency overhead on a real edge AI device using JetPack/TensorRT-class deployment.

It does not need to prove that CIRCA-RT has the highest recall. That is not the model's strongest claim.

## 3. Minimal Experiment Matrix

Use a compact matrix:

| Component | Choice |
|---|---|
| Models | MobileNetV2, ResNet18 |
| Runtime | ONNXRuntime, TensorRT if valid |
| Scenarios | nominal, GPU interference, coupled attack |
| Seeds | 7, 8, 9 |
| Samples per trace | 900-1500 |
| Monitors | CIRCA-RT, ContextAwareConformal, TranAD, CATCH, RandomBudgetAudit, AlwaysAudit |
| Power logs | `tegrastats` |

If time is tight, run:

- MobileNetV2 only
- ONNXRuntime only
- CIRCA-RT, ContextAwareConformal, TranAD, AlwaysAudit
- `n=900`, seeds `7,8,9`

## 4. First Login And Hardware Check

After renting the device, SSH into it:

```bash
ssh <user>@<host> -p <port>
```

Run:

```bash
uname -a
uname -m
cat /etc/nv_tegra_release
lsb_release -a
nvidia-smi || true
tegrastats --help | head
sudo nvpmodel -q || true
```

Expected:

- `uname -m` should be `aarch64`
- `/etc/nv_tegra_release` should show JetPack/L4T info
- `tegrastats` should exist
- `nvidia-smi` may not exist on Jetson; this is normal

Check CUDA/TensorRT:

```bash
nvcc --version || true
dpkg -l | grep -E 'nvinfer|tensorrt|cuda|cudnn' | head -80
python3 - <<'PY'
try:
    import tensorrt as trt
    print("TensorRT", trt.__version__)
except Exception as e:
    print("NO_TENSORRT_PYTHON", repr(e))
PY
```

## 5. Performance Mode

Set a stable power/performance mode before collecting latency:

```bash
sudo nvpmodel -q
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true
```

Record the chosen mode in the paper.

If `sudo nvpmodel -m 0` is not allowed by the rental provider, just record the default mode.

## 6. Workspace Setup

Create workspace:

```bash
mkdir -p /root/rtss_exp_begin
cd /root/rtss_exp_begin
```

Upload this repository from your local machine or AutoDL. At minimum, upload:

```text
configs/
docs/
scripts/
src/
README.md
requirements.txt
```

Also upload the final AutoDL trace results if you want to reuse offline evaluation scripts:

```text
results_autodl_final/
results_autodl_ros2_twoprocess_final/
results_recent_deep_baselines_final/
```

Install Python dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas scipy scikit-learn matplotlib
python3 -m pip install onnx onnxruntime
```

On Jetson, ONNXRuntime GPU/TensorRT wheels can be tricky. Prefer the JetPack-provided runtime or NVIDIA-compatible wheels if available. If ONNXRuntime GPU is not available, run PyTorch/ONNX CPU only as a smoke test and use TensorRT directly if installed.

## 7. Environment Check Script

Run:

```bash
cd /root/rtss_exp_begin
python3 - <<'PY'
import platform
print("python ok")
print("machine", platform.machine())
for name in ["numpy", "pandas", "sklearn", "onnxruntime", "tensorrt"]:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "imported"))
    except Exception as e:
        print("missing", name, repr(e))
PY
```

## 8. TensorRT Validation

TensorRT must be validated before reporting it.

Run:

```bash
python3 - <<'PY'
try:
    import tensorrt as trt
    print("TensorRT version:", trt.__version__)
except Exception as e:
    raise SystemExit(f"TensorRT Python import failed: {e}")
PY
```

If TensorRT Python is not available but `trtexec` exists:

```bash
which trtexec
trtexec --version || true
```

Do not report TensorRT latency unless either:

- TensorRT Python engine execution is actually used, or
- `trtexec` is used to build and run a real TensorRT engine.

Do not repeat the AutoDL mistake where TensorRT provider is listed but falls back to CPU.

## 9. Recommended Jetson Experiment Script

If the current repository does not yet have a dedicated Jetson runner, create one based on:

- `scripts/run_autodl_full_experiment.py`
- `scripts/run_recent_deep_baselines.py`
- `scripts/run_ros2_dds_twoprocess_capture.py`
- `scripts/run_ros2_dds_eval.py`

Expected Jetson output directory:

```text
results_jetson_orin/
```

Recommended structure:

```text
results_jetson_orin/
  logs/
  traces/
  raw/
  tables/
  tegrastats/
  engines/
```

## 10. Power And Thermal Logging

Start `tegrastats` before each run:

```bash
mkdir -p results_jetson_orin/tegrastats
tegrastats --interval 1000 > results_jetson_orin/tegrastats/run_$(date +%Y%m%d_%H%M%S).log &
TEGRA_PID=$!
echo $TEGRA_PID
```

After the run:

```bash
kill $TEGRA_PID
```

Keep these logs for the paper:

- average power if available
- peak temperature
- GPU/CPU frequency
- throttling signs

## 11. ONNXRuntime/TensorRT Latency Run

Preferred command shape:

```bash
python3 scripts/run_jetson_orin_experiment.py \
  --out-dir results_jetson_orin \
  --models mobilenet_v2 resnet18 \
  --runtimes onnxruntime tensorrt \
  --seeds 7 8 9 \
  --n 1200 \
  --warmup 120 \
  --deadline-ms 25.0
```

If TensorRT fails:

```bash
python3 scripts/run_jetson_orin_experiment.py \
  --out-dir results_jetson_orin \
  --models mobilenet_v2 resnet18 \
  --runtimes onnxruntime \
  --seeds 7 8 9 \
  --n 1200 \
  --warmup 120 \
  --deadline-ms 25.0
```

Record TensorRT as failed only if:

- library missing
- engine build fails
- active execution provider falls back to CPU

## 12. ROS2/DDS Optional Jetson Run

ROS2 on Jetson is useful but not mandatory if time is tight, because AutoDL already has ROS2 two-process traces.

If you can install ROS2 Humble:

```bash
sudo apt update
sudo apt install -y software-properties-common curl gnupg lsb-release
```

Follow the existing installer logic:

```bash
bash scripts/install_ros2_humble_autodl.sh
```

Then run:

```bash
set +u
source /opt/ros/humble/setup.bash
set -u

python3 scripts/run_ros2_dds_twoprocess_capture.py \
  --out-dir results_jetson_orin_ros2_twoprocess \
  --seeds 7 8 9 \
  --n 900 \
  --period-ms 20 \
  --deadline-ms 50 \
  --node-python /usr/bin/python3

python3 scripts/run_ros2_dds_eval.py --root results_jetson_orin_ros2_twoprocess
```

If ROS2 install is painful, skip it and use the AutoDL ROS2/DDS result as middleware evidence. Jetson's required contribution is edge inference/runtime validation.

## 13. Required Tables

Create these final tables:

```text
results_jetson_orin/tables/jetson_latency_table.csv
results_jetson_orin/tables/jetson_monitoring_main_table.csv
results_jetson_orin/tables/jetson_monitoring_main_table_ci.csv
results_jetson_orin/tables/jetson_power_summary.csv
results_jetson_orin/tables/provider_status.csv
```

Minimum columns:

```text
runtime
model
seed
scenario
mean_ms
p50_ms
p95_ms
p99_ms
p999_ms
deadline_miss_ratio
method
attack_recall
false_alarm_rate
audit_rate
latency_inflation_mean_ms
monitor_cost_mean_ms
```

Power summary:

```text
runtime
model
scenario
mean_power_w
peak_temp_c
mean_gpu_freq_mhz
mean_cpu_freq_mhz
```

If power cannot be parsed reliably, include raw `tegrastats` logs and state that power is reported qualitatively.

## 14. Paper Figures

Use only 2-3 Jetson figures:

1. Jetson p99 latency by runtime/model.
2. Jetson recall vs audit/overhead Pareto.
3. Jetson power/thermal summary if reliable.

Do not overload the paper with Jetson tables. The full AutoDL and ROS2 tables are already large.

## 15. Expected Paper Claim After Jetson

Strong and safe claim:

> On Jetson Orin, CIRCA-RT preserves sub-millisecond monitoring overhead and low audit rate while maintaining competitive detection on coupled semantic-timing attacks.

Avoid:

> CIRCA-RT beats all deep anomaly detectors on Jetson.

Avoid:

> TensorRT results are available if TensorRT failed or fell back to CPU.

## 16. Failure Handling

If TensorRT fails:

- Keep ONNXRuntime/PyTorch Jetson results.
- Include `provider_status.csv`.
- State TensorRT was unavailable on the rented image.

If ROS2 fails:

- Keep AutoDL ROS2/DDS two-process results.
- Use Jetson for edge inference validation.

If `tegrastats` cannot run:

- Record provider limitation.
- Still report latency and monitoring overhead.

## 17. Final Checklist

Before ending the rental session, download:

```text
results_jetson_orin/
results_jetson_orin_ros2_twoprocess/   # if run
logs/
tegrastats/
provider_status.csv
```

Also save:

```bash
uname -a
uname -m
cat /etc/nv_tegra_release
dpkg -l | grep -E 'nvinfer|tensorrt|cuda|cudnn'
python3 -c "import tensorrt as trt; print(trt.__version__)"
sudo nvpmodel -q
```

These environment details are necessary for the paper artifact and reproducibility appendix.

