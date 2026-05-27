# AutoDL-First Mini-System Plan

Status update on 2026-05-25: this remains an AutoDL-first planning note. The
current AGX Orin 64GB-class evidence is the completed full archive documented in
`docs/AGX_ORIN64_FULL_ARCHIVE_VALIDATION_2026_05_25.md`.

更新时间：2026-05-17

目标：把能在普通 NVIDIA GPU 云主机上做的实验优先放到 AutoDL，降低 Jetson 租用时间和调试风险。

## Core Strategy

AutoDL 先完成：

- NVIDIA GPU 环境验证
- PyTorch / ONNXRuntime / TensorRT batch-1 latency
- TensorRT 脚本预调试
- ROS2/DDS 或 Python periodic 软件链路
- CIRCA-RT online monitor 集成
- selective audit vs always audit 对比
- 批量 seeds / baseline / table generation

Jetson 后做：

- edge-class TensorRT latency
- Jetson power mode
- `tegrastats`
- final selective audit tail-latency validation

这样论文证据链是：

```text
local split proof
+ local periodic middleware fallback
+ AutoDL server-GPU ROS2/TensorRT mini-system
+ Jetson edge validation
```

## What AutoDL Can Do First

### Phase A: GPU Environment

Must check:

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorrt as trt; print(trt.__version__)"
python -c "import onnxruntime as ort; print(ort.__version__)"
```

### Phase B: TensorRT Latency

Models:

- `MobileNetV2`
- `ResNet18`
- optional `YOLOv8n`

Policies:

- perception only
- perception + monitor
- perception + selective audit
- perception + always audit

Metrics:

- p50 / p95 / p99 / p999 latency
- deadline miss ratio
- monitor cost
- audit cost
- GPU utilization

### Phase C: Middleware-Style Pipeline

Preferred:

- ROS2 Humble on Ubuntu 22.04
- ROS2 Jazzy only if Ubuntu 24.04

Fallback:

- Python periodic pipeline already implemented in `run_phase5_periodic_pipeline.py`

### Phase D: Batch Experiments

AutoDL should run:

- more seeds
- larger traces
- baseline repeats
- sensitivity sweeps
- ablation table

Do not wait for Jetson to run these.

## What Jetson Should Still Do

Jetson is still needed for:

- edge device claim
- power/frequency/temperature
- `nvpmodel`
- `jetson_clocks`
- `tegrastats`
- Jetson-class p99/p999 latency

Jetson should not repeat every AutoDL baseline. It should run a smaller matrix:

- nominal
- coupled attack
- jitter attack
- stale replay
- perception only
- monitor only
- selective audit
- always audit

## Minimum Publishable Experiment Set

Minimum small-but-complete RTSS package:

1. Local `results_split/`
2. Local `results_phase5/`
3. AutoDL TensorRT server-GPU mini-system
4. Jetson edge validation subset

Current status:

- AutoDL results remain supporting server-GPU evidence.
- The final RTSS hardware evidence should use the 2026-05-25 AGX full archive.

## Output Layout

AutoDL results should be downloaded into:

```text
F:\RTSS\exp_begin\results_autodl\
  traces\
  raw\
  summaries\
  tables\
  figures\
```

Platform label:

```text
autodl_server_gpu
```

## Go / No-Go

AutoDL Go:

- `nvidia-smi` works
- CUDA visible to PyTorch
- TensorRT or ONNXRuntime GPU works
- results can be downloaded

AutoDL No-Go:

- no GPU visible
- no file download
- no Python package install
- no long-running job support

## 2026-05-18 Status

Current AutoDL instance:

- GPU: NVIDIA GeForce RTX 4080 SUPER
- PyTorch CUDA: available
- TensorRT: not installed in the current Python environment
- ONNXRuntime: not installed in the current Python environment

Completed:

- Uploaded project to `/root/rtss_exp_begin`
- Installed project dependencies: `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- Ran Phase 5 periodic pipeline on AutoDL
- Ran PyTorch CUDA latency microbenchmark
- Exported GPU latency traces into `results_autodl/traces`
- Evaluated GPU traces with CIRCA-RT and split baselines
- Ran torchvision real-model latency baselines: `ResNet18`, `MobileNetV2`
- Exported real-model traces into `results_autodl/traces_torchvision`
- Evaluated real-model traces with CIRCA-RT and split baselines
- Installed `onnxruntime-gpu` and `onnx`
- Ran ONNXRuntime CUDAExecutionProvider baselines for `ResNet18` and `MobileNetV2`
- Exported ONNXRuntime traces into `results_autodl/traces_onnxruntime`
- Generated AutoDL overview tables

Local downloaded outputs:

- `F:\RTSS\exp_begin\results_autodl\tables\autodl_latency_overview.csv`
- `F:\RTSS\exp_begin\results_autodl\tables\autodl_monitoring_overview.csv`
- `F:\RTSS\exp_begin\results_autodl\tables\torch_latency_table.csv`
- `F:\RTSS\exp_begin\results_autodl\tables\torchvision_latency_table.csv`
- `F:\RTSS\exp_begin\results_autodl\tables\onnxruntime_latency_table.csv`
- `F:\RTSS\exp_begin\results_autodl\tables\gpu_trace_main_table.csv`
- `F:\RTSS\exp_begin\results_autodl\tables\torchvision_main_table.csv`
- `F:\RTSS\exp_begin\results_autodl\tables\onnxruntime_main_table.csv`
- `F:\RTSS\exp_begin\results_autodl\tables\torchvision_scenario_table.csv`
- `F:\RTSS\exp_begin\results_autodl\tables\gpu_trace_scenario_table.csv`
- `F:\RTSS\exp_begin\results_autodl\logs\env_check_gpu_on.log`

Key real-model latency:

- `ResNet18` nominal p99: about `2.00 ms`
- `MobileNetV2` nominal p99: about `4.36 ms`
- `MobileNetV2` coupled-GPU p99: about `7.36 ms`
- `ONNXRuntime ResNet18` nominal p99: about `1.07 ms`
- `ONNXRuntime MobileNetV2` nominal p99: about `2.58 ms`

Key real-model monitoring result:

- `ResNet18 + CIRCA-RT`: attack recall about `0.488`, audit rate about `0.019`, p99 latency about `4.51 ms`
- `MobileNetV2 + CIRCA-RT`: attack recall about `0.523`, audit rate about `0.023`, p99 latency about `8.03 ms`
- `ONNXRuntime ResNet18 + CIRCA-RT`: attack recall about `0.517`, audit rate about `0.016`, p99 latency about `2.95 ms`
- `ONNXRuntime MobileNetV2 + CIRCA-RT`: attack recall about `0.571`, audit rate about `0.016`, p99 latency about `4.48 ms`

Important limitation:

- Current AutoDL result includes PyTorch CUDA and ONNXRuntime CUDA server-GPU baselines.
- ONNXRuntime reports `TensorrtExecutionProvider` as available, but TensorRT EP has not been used as the main result yet.
- TensorRT EP should be treated as optional unless it is validated with stable repeatability.
