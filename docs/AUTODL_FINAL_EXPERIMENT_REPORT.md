# AutoDL Final Experiment Report

Updated: 2026-05-19

This is the final AutoDL experiment package after strengthening the coupled attack construction and adding budget-normalized comparisons. It should be treated as server-GPU and ROS2/DDS evidence. The current AGX Orin 64GB-class evidence is documented separately in `docs/AGX_ORIN64_FULL_ARCHIVE_VALIDATION_2026_05_25.md`.

## Final Output Directories

- `results_autodl_final/`
- `results_autodl_ros2_twoprocess_final/`
- `results_budget_normalized_final/`

Previous results remain available:

- `results_autodl_full/`
- `results_autodl_ros2/`
- `results_autodl_ros2_twoprocess/`
- `results_autodl_sweep/`

## What Changed In The Final Run

The coupled attack is now closer to the paper's intended threat model:

- The marginal distributions of semantic and timing residuals are kept close to benign statistics.
- The attack primarily increases conditional semantic-timing dependence.
- This avoids an overly easy single-signal anomaly that simple residual or context-aware methods can detect.

Added final budget-normalized evaluation:

- Budgets: `0.02`, `0.05`, `0.10`, `0.15`
- All score-based methods are forced to audit the same fraction of samples.
- This makes recall/cost comparison fairer than raw threshold defaults.

## Final AutoDL GPU Matrix

Output:

```text
results_autodl_final/
```

Matrix:

- GPU: NVIDIA GeForce RTX 4080 SUPER
- Runtimes: `torch_cuda`, `onnx_cuda`
- TensorRT EP: attempted, but unavailable due to missing `libnvinfer.so.10`
- Models: `resnet18`, `mobilenet_v2`, `squeezenet1_1`
- Seeds: `7, 8, 9`
- Scenarios: `nominal`, `gpu_interference`, `coupled_gpu_attack`
- Samples per trace: `1800`
- Warmup: `150`

Main files:

- `results_autodl_final/tables/provider_status.csv`
- `results_autodl_final/tables/latency_full_table.csv`
- `results_autodl_final/tables/monitoring_main_table.csv`
- `results_autodl_final/tables/monitoring_main_table_ci.csv`
- `results_autodl_final/tables/autodl_full_monitoring_overview.csv`

Mean p99 latency:

| runtime | model | nominal | interference | coupled |
|---|---:|---:|---:|---:|
| ONNXRuntime CUDA | ResNet18 | 1.130 ms | 2.762 ms | 0.982 ms |
| ONNXRuntime CUDA | MobileNetV2 | 1.693 ms | 3.550 ms | 2.391 ms |
| ONNXRuntime CUDA | SqueezeNet1.1 | 1.260 ms | 1.664 ms | 1.222 ms |
| PyTorch CUDA | ResNet18 | 3.904 ms | 3.129 ms | 4.172 ms |
| PyTorch CUDA | MobileNetV2 | 7.888 ms | 7.596 ms | 8.737 ms |
| PyTorch CUDA | SqueezeNet1.1 | 4.120 ms | 3.116 ms | 3.553 ms |

Default CIRCA-RT on GPU traces:

| runtime | model | recall | false alarm | audit rate | p99 latency | mean overhead |
|---|---:|---:|---:|---:|---:|---:|
| ONNXRuntime CUDA | ResNet18 | 0.449 | 0.088 | 0.0147 | 3.998 ms | 0.139 ms |
| ONNXRuntime CUDA | MobileNetV2 | 0.485 | 0.074 | 0.0131 | 4.255 ms | 0.132 ms |
| ONNXRuntime CUDA | SqueezeNet1.1 | 0.472 | 0.083 | 0.0143 | 3.182 ms | 0.137 ms |
| PyTorch CUDA | ResNet18 | 0.464 | 0.063 | 0.0117 | 4.765 ms | 0.127 ms |
| PyTorch CUDA | MobileNetV2 | 0.433 | 0.059 | 0.0108 | 8.820 ms | 0.123 ms |
| PyTorch CUDA | SqueezeNet1.1 | 0.444 | 0.061 | 0.0117 | 4.670 ms | 0.127 ms |

Interpretation:

- The final GPU experiment is complete and paper-usable.
- CIRCA-RT has consistently low audit rate around `1.1%-1.5%`.
- Mean latency overhead is around `0.12-0.14 ms`.
- CIRCA-RT is competitive with RFF/ConditionalRFF recall in default settings, but not a universal recall winner.

## Final ROS2/DDS Two-Process Experiment

Output:

```text
results_autodl_ros2_twoprocess_final/
```

Matrix:

- ROS2: Humble
- Publisher and subscriber: separate processes
- DDS/RMW: default ROS2 Humble stack
- Seeds: `7, 8, 9`
- Samples per scenario: `1200`
- Scenarios: `nominal`, `mode_shift`, `jitter_attack`, `cpu_load_interference`, `coupled_timing_semantic_attack`

Main files:

- `results_autodl_ros2_twoprocess_final/tables/ros2_twoprocess_latency_table.csv`
- `results_autodl_ros2_twoprocess_final/tables/ros2_main_table.csv`
- `results_autodl_ros2_twoprocess_final/tables/ros2_main_table_ci.csv`
- `results_autodl_ros2_twoprocess_final/tables/ros2_scenario_table.csv`

Key default-method comparison:

| method | recall | false alarm | audit rate | p99 latency | mean overhead |
|---|---:|---:|---:|---:|---:|
| AlwaysAudit | 1.000 | 1.000 | 1.000 | 5.275 ms | 4.050 ms |
| LearnedFourierIndependence | 0.902 | 0.831 | 0.839 | 5.265 ms | 3.405 ms |
| ContextAwareConformal | 0.589 | 0.060 | 0.123 | 5.077 ms | 0.541 ms |
| CIRCA-RT | 0.429 | 0.077 | 0.014 | 2.638 ms | 0.138 ms |
| ConditionalRFF-HSIC | 0.363 | 0.036 | 0.074 | 3.863 ms | 0.344 ms |
| RFF-HSIC | 0.342 | 0.032 | 0.067 | 3.775 ms | 0.318 ms |

Interpretation:

- This is the strongest middleware experiment in the AutoDL package.
- CIRCA-RT is not the highest-recall method on this trace.
- CIRCA-RT is the clearest low-budget method among nontrivial detectors: `0.014` audit rate and `0.138 ms` overhead.
- ContextAwareConformal is the main competitor and currently has higher recall at moderate overhead.

## Budget-Normalized Comparison

Output:

```text
results_budget_normalized_final/
```

Main files:

- `results_budget_normalized_final/tables/budget_normalized_main_table.csv`
- `results_budget_normalized_final/tables/budget_normalized_main_table_ci.csv`
- `results_budget_normalized_final/tables/budget_normalized_top5_table.csv`

### GPU Budget-Normalized Findings

At `10%` audit budget:

- ONNX MobileNetV2: CIRCA-RT reaches `0.400` recall, tied with ConditionalRFF-HSIC, with slightly higher monitor overhead because CIRCA-RT monitor cost is set to `0.08 ms`.
- ONNX ResNet18 and SqueezeNet1.1: RFF-HSIC often edges out CIRCA-RT in recall, but CIRCA-RT remains close.
- PyTorch MobileNetV2: CIRCA-RT reaches `0.385` recall at `10%` budget, near the top group.

At `15%` audit budget:

- ONNX MobileNetV2: CIRCA-RT reaches `0.515` recall.
- PyTorch ResNet18: CIRCA-RT reaches `0.501` recall.

### ROS2 Two-Process Budget-Normalized Findings

At `10%` audit budget:

| method | recall | false alarm | p99 latency | overhead |
|---|---:|---:|---:|---:|
| ContextAwareConformal | 0.394 | 0.056 | 5.084 ms | 0.450 ms |
| RFF-HSIC | 0.219 | 0.082 | 5.156 ms | 0.450 ms |
| LearnedFourierIndependence | 0.208 | 0.084 | 5.121 ms | 0.450 ms |
| CIRCA-RT | 0.199 | 0.085 | 5.115 ms | 0.480 ms |

At `15%` audit budget:

| method | recall | false alarm | p99 latency | overhead |
|---|---:|---:|---:|---:|
| ContextAwareConformal | 0.519 | 0.095 | 5.128 ms | 0.650 ms |
| RFF-HSIC | 0.319 | 0.125 | 5.177 ms | 0.650 ms |
| LearnedFourierIndependence | 0.316 | 0.125 | 5.156 ms | 0.650 ms |
| CIRCA-RT | 0.312 | 0.126 | 5.160 ms | 0.680 ms |

Interpretation:

- Budget-normalized results are important because they prevent unfair comparisons where one method simply audits more.
- On GPU traces, CIRCA-RT is competitive under the same audit budget.
- On ROS2 two-process traces, ContextAwareConformal is stronger in recall.
- CIRCA-RT's best defensible claim is low-overhead budget-aware detection, not recall dominance.

## TensorRT Status

TensorRT EP is still not valid on the current AutoDL image.

Recorded status:

```text
TensorrtExecutionProvider listed by ONNXRuntime, but did not initialize.
Active providers fell back to CPUExecutionProvider.
Missing library: libnvinfer.so.10.
```

Do not report TensorRT latency from AutoDL. Move TensorRT to Jetson/JetPack or a known TensorRT-ready image.

## Final Quality Assessment

What is complete:

- Real server-GPU inference traces.
- Two GPU runtimes: PyTorch CUDA and ONNXRuntime CUDA.
- Three CNN models.
- Multi-seed repeated runs.
- Stronger hidden coupled attack design.
- Real ROS2 Humble two-process DDS-style experiment.
- Budget-normalized evaluation.
- TensorRT failure documented instead of overclaimed.

What remains weak:

- AGX Orin/Jetson evidence is covered by the separate 2026-05-25 full archive validation.
- A future Jetson block should be presented as edge validation, not a full rerun of all AutoDL baselines.
- No physical TSN hardware.
- CIRCA-RT is not recall-SOTA on ROS2 two-process traces.
- ContextAwareConformal is a strong competitor and must be handled honestly in the paper.

## Paper-Ready Claim

Use this:

> CIRCA-RT provides a budget-aware online monitor that preserves low audit cost and low tail-latency overhead on real GPU inference and ROS2/DDS traces, while remaining competitive on coupled semantic-timing attacks under fixed audit budgets.

Do not use this:

> CIRCA-RT outperforms all baselines across all settings.

Do not derive Jetson/AGX validation from this AutoDL report; use the separate AGX full archive validation.

## Recommended Next Step

The AutoDL stage is complete enough to support the server-GPU and ROS2/DDS parts of the paper. The remaining work is packaging and positioning:

- Use AutoDL as the server-GPU and ROS2/DDS evidence.
- Use AGX/Jetson evidence only after a fresh real-device run is collected.
- Do not claim physical TSN hardware.
- Keep the main claim on budget-aware monitoring, low audit rate, and controlled tail latency rather than universal recall dominance.
