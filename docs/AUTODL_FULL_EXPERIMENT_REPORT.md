# AutoDL Full Experiment Report

Updated: 2026-05-18

This report records the completed AutoDL experiments for the CIRCA-RT RTSS submission path. It separates verified results from failed or deferred claims so the paper does not accidentally overclaim TensorRT, TSN, or Jetson results.

Status update on 2026-05-25: this AutoDL report remains server-GPU and ROS2/DDS
evidence only. The current AGX Orin 64GB-class evidence is the completed full
archive documented in `docs/AGX_ORIN64_FULL_ARCHIVE_VALIDATION_2026_05_25.md`.

## Environment

AutoDL instance:

- GPU: NVIDIA GeForce RTX 4080 SUPER
- Driver: 595.71.05
- GPU memory: 32760 MiB
- OS: Ubuntu 22.04.5 LTS
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- ONNXRuntime: 1.26.0
- ROS2: Humble installed through Ubuntu 22.04 deb packages

Official basis:

- ROS2 Humble provides Ubuntu 22.04 Jammy deb packages, which matches the AutoDL OS: https://docs.ros.org/en/humble/Installation.html
- ONNXRuntime TensorRT EP requires NVIDIA TensorRT runtime libraries, not just the provider name in `get_available_providers()`: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- JetPack 6.2.1 includes Jetson Linux 36.4.4, Ubuntu 22.04-based rootfs, TensorRT, cuDNN, and CUDA libraries for Jetson validation: https://docs.nvidia.com/jetson/jetpack/6.2.1/introduction/index.html

## Completed Experiments

### 1. AutoDL GPU Inference Traces

Output directory:

```text
results_autodl_full/
```

Matrix:

- Runtimes: `torch_cuda`, `onnx_cuda`
- Models: `resnet18`, `mobilenet_v2`, `squeezenet1_1`
- Seeds: `7, 8, 9`
- Scenarios: `nominal`, `gpu_interference`, `coupled_gpu_attack`
- Samples: 1500 per trace, 120 warmup iterations

Key tables:

- `results_autodl_full/tables/latency_full_table.csv`
- `results_autodl_full/tables/monitoring_main_table.csv`
- `results_autodl_full/tables/monitoring_main_table_ci.csv`
- `results_autodl_full/tables/autodl_full_monitoring_overview.csv`
- `results_autodl_full/tables/provider_status.csv`

Mean p99 latency by runtime/model/scenario:

| runtime | model | nominal p99 ms | interference p99 ms | coupled p99 ms |
|---|---:|---:|---:|---:|
| ONNXRuntime CUDA | ResNet18 | 0.738 | 1.647 | 0.797 |
| ONNXRuntime CUDA | MobileNetV2 | 1.253 | 2.167 | 1.453 |
| ONNXRuntime CUDA | SqueezeNet1.1 | 0.803 | 1.540 | 0.853 |
| PyTorch CUDA | ResNet18 | 2.500 | 2.378 | 2.384 |
| PyTorch CUDA | MobileNetV2 | 5.728 | 5.193 | 6.859 |
| PyTorch CUDA | SqueezeNet1.1 | 2.396 | 2.975 | 2.387 |

Default CIRCA-RT monitoring result on real GPU traces:

| runtime | model | recall | false alarm | audit rate | p99 latency ms | mean latency overhead ms |
|---|---:|---:|---:|---:|---:|---:|
| ONNXRuntime CUDA | ResNet18 | 0.480 | 0.094 | 0.016 | 3.591 | 0.146 |
| ONNXRuntime CUDA | MobileNetV2 | 0.500 | 0.091 | 0.018 | 3.741 | 0.151 |
| ONNXRuntime CUDA | SqueezeNet1.1 | 0.477 | 0.095 | 0.018 | 3.264 | 0.150 |
| PyTorch CUDA | ResNet18 | 0.493 | 0.068 | 0.015 | 4.348 | 0.138 |
| PyTorch CUDA | MobileNetV2 | 0.510 | 0.066 | 0.015 | 7.058 | 0.140 |
| PyTorch CUDA | SqueezeNet1.1 | 0.489 | 0.071 | 0.014 | 4.380 | 0.137 |

Interpretation:

- ONNXRuntime CUDA and PyTorch CUDA baselines are now real AutoDL GPU results, not local synthetic-only results.
- CIRCA-RT keeps audit rate around 1.4-1.8% and mean latency overhead around 0.14-0.15 ms.
- The default configuration prioritizes audit cost and tail latency. It is not tuned for maximum recall.

### 2. TensorRT EP Validation

TensorRT EP was explicitly tested and rejected as a main baseline on the current AutoDL image.

Observed status:

```text
TensorrtExecutionProvider listed by ONNXRuntime, but session initialization falls back to CPUExecutionProvider.
Missing library: libnvinfer.so.10
```

Recorded file:

```text
results_autodl_full/tables/provider_status.csv
```

Paper implication:

- Do not report a TensorRT latency result from this AutoDL image.
- It is valid to report that TensorRT EP was attempted but unavailable due to missing TensorRT runtime libraries.
- TensorRT should be moved to the Jetson/JetPack phase or to an AutoDL image that already includes a matching TensorRT runtime.

### 3. ROS2/DDS Loopback Trace

Output directory:

```text
results_autodl_ros2/
```

Matrix:

- ROS2 distribution: Humble
- Transport: rclpy over the default ROS2 DDS/RMW stack
- Process model: single Python process with ROS2 publisher/subscriber node
- Seeds: `7, 8, 9`
- Scenarios: `nominal`, `mode_shift`, `jitter_attack`, `cpu_load_interference`, `coupled_timing_semantic_attack`
- Samples: 900 per trace

Mean ROS2 loopback latency:

| scenario | mean ms | p99 ms | p999 ms | deadline miss |
|---|---:|---:|---:|---:|
| nominal | 0.941 | 1.530 | 2.024 | 0.000 |
| mode shift | 0.949 | 1.616 | 2.493 | 0.000 |
| jitter attack | 1.016 | 1.468 | 2.203 | 0.000 |
| CPU-load interference | 0.967 | 1.836 | 3.024 | 0.000 |
| coupled timing-semantic attack | 0.992 | 1.668 | 2.680 | 0.000 |

Default CIRCA-RT result:

- Recall: 0.538
- False alarm: 0.133
- Audit rate: 0.0267
- p99 latency: 3.416 ms
- Mean latency overhead: 0.187 ms

### 4. ROS2/DDS Two-Process Trace

Output directory:

```text
results_autodl_ros2_twoprocess/
```

This is the stronger middleware experiment. Publisher and subscriber are separate ROS2 processes, which makes it closer to a real DDS deployment than the loopback trace.

Mean two-process ROS2 latency:

| scenario | mean ms | p99 ms | p999 ms | deadline miss |
|---|---:|---:|---:|---:|
| nominal | 0.849 | 1.095 | 2.038 | 0.000 |
| mode shift | 0.855 | 1.149 | 1.777 | 0.000 |
| jitter attack | 0.860 | 1.201 | 2.181 | 0.000 |
| CPU-load interference | 0.857 | 1.138 | 2.710 | 0.000 |
| coupled timing-semantic attack | 0.827 | 1.110 | 2.341 | 0.000 |

Default CIRCA-RT result:

- Recall: 0.387
- False alarm: 0.110
- Audit rate: 0.0181
- p99 latency: 2.671 ms
- Mean latency overhead: 0.152 ms

Interpretation:

- This is the most defensible AutoDL middleware result because it uses separate ROS2 processes.
- Default CIRCA-RT is conservative here; it gives low overhead but lower recall than some heavier baselines.
- The paper should frame this as a tradeoff result, not as a universal best-detector result.

### 5. CIRCA-RT Configuration Sweep

Output directory:

```text
results_autodl_sweep/
```

Key tables:

- `results_autodl_sweep/tables/config_sweep_main_table.csv`
- `results_autodl_sweep/tables/config_sweep_main_table_ci.csv`
- `results_autodl_sweep/tables/config_sweep_pareto_table.csv`

Best high-recall CIRCA-RT configurations:

| dataset | config | recall | false alarm | audit rate | p99 latency ms | overhead ms |
|---|---|---:|---:|---:|---:|---:|
| ROS2 loopback | `w64_d32_ql0.85_qh0.97_r0.45` | 0.665 | 0.293 | 0.036 | 4.371 | 0.223 |
| ROS2 two-process | `w16_d32_ql0.85_qh0.97_r0.45` | 0.485 | 0.188 | 0.026 | 4.210 | 0.184 |
| PyTorch MobileNetV2 | `w16_d32_ql0.85_qh0.97_r0.45` | 0.576 | 0.164 | 0.030 | 8.015 | 0.199 |
| ONNXRuntime SqueezeNet1.1 | `w32_d32_ql0.85_qh0.97_r0.45` | 0.590 | 0.217 | 0.029 | 4.926 | 0.196 |

Interpretation:

- CIRCA-RT has a clear recall-cost tradeoff.
- Lower quantile thresholds raise recall but also raise false alarms.
- The default config is a low-overhead setting; the sweep provides a stronger paper figure for sensitivity and Pareto analysis.

## Current Quality Assessment

What is now strong:

- Real AutoDL GPU inference traces with PyTorch CUDA and ONNXRuntime CUDA.
- Three real CNN models instead of toy-only latency.
- Multi-seed repeated runs and CI-ready tables.
- Real ROS2 Humble trace capture on Ubuntu 22.04.
- Stronger two-process ROS2/DDS trace.
- Explicit TensorRT EP failure record, avoiding false claims from CPU fallback.
- Unified schema across synthetic, GPU, ROS2, and two-process ROS2 traces.

What is still not covered by this AutoDL-only report:

- The AGX Orin full archive is documented separately and should be cited from
  `docs/AGX_ORIN64_FULL_ARCHIVE_VALIDATION_2026_05_25.md`.
- No real TSN hardware or physical network experiment.
- TensorRT should not be claimed from this AutoDL image.
- ROS2/DDS is on AutoDL server Linux, not Jetson.
- Default CIRCA-RT is a low-overhead detector, not always the highest-recall detector.

## Recommended Paper Claim

Safe claim:

> CIRCA-RT provides a low-overhead, budget-aware online monitoring layer for coupled semantic-timing anomalies, and it remains deployable on real GPU inference and ROS2/DDS traces while using substantially less audit budget than always-audit and heavier dependence-monitoring baselines.

Do not claim:

> CIRCA-RT beats every SOTA baseline in recall on all ROS2/DDS scenarios.

Do not derive Jetson/AGX claims from this AutoDL report; use the AGX full
archive validation instead.

## Next Phase

The remaining work is paper integration:

- Use this report for AutoDL server-GPU and ROS2/DDS evidence.
- Use the 2026-05-25 AGX full archive for AGX/Jetson evidence.
- Do not claim physical TSN hardware.
- Keep the claim on budget-aware monitoring, low audit rate, and controlled tail latency rather than universal recall dominance.
