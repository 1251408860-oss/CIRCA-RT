# AGX Orin 64GB Full Archive Validation

Date: 2026-05-25

Source archive:

```text
F:\RTSS\agx_orin64_codex_20260525_FULL_CODE_LOGS_RESULTS.tar.gz
```

Local extraction:

```text
F:\RTSS\exp_begin\agx_orin64_full_archive_20260525
```

Archive fingerprint:

```text
size: 937.14 MB
sha256: F239CB9B1599D7A731703AA8EBF6219281133F0B09597D7319F507036623B22E
members: 7204 total, 3788 files, 3413 directories
```

## Verdict

This is the strongest AGX Orin result package returned so far. It contains the
isolated project copy, execution scripts, nohup logs, environment evidence,
wheel/cache material, local library side-loads, full result tables, backend
matrix outputs, and pressure-test evidence from the 2026-05-25 AGX run.

The package is suitable as real Jetson AGX Orin 64GB-class edge-device evidence
for the paper, with these boundaries:

- Safe to claim real AGX Orin PyTorch CUDA real-frame execution.
- Safe to claim ONNXRuntime CUDA and ONNXRuntime TensorRT provider execution in
  the full backend matrix.
- Safe to claim ResNet50 real-frame evidence across AV2, DROID, and nuImages.
- Safe to claim staged ResNet50 AV2 pressure levels passed through
  `stress1536_r2_x12`.
- Do not claim the final extreme overlay completed. The AGX rebooted during the
  added 24 GB memory plus GPU 4096 matrix overlay.
- Do not call the reboot OOM or thermal failure. Logged RAM, swap, and
  temperatures do not support those explanations.
- Do not make power-efficiency claims unless VDD power is collected and parsed;
  this archive gives resource and thermal telemetry, not a clean power table.
- Full AGX ROS2 closed-loop validation is now a separate optional rerun path,
  documented in `docs/AGX_ORIN64_ROS2_CLOSED_LOOP_RUNBOOK_2026_05_25.md`. Do
  not claim that result unless its result directory and tables are present.

Windows extraction produced expected cross-platform warnings for Linux symlinks
and one member named `provider_status.log\r`. The important CSV, JSON, Markdown,
and log files extracted successfully.

## Device And Runtime Evidence

The archive records a real Jetson platform:

- Machine: `aarch64`.
- Kernel: `Linux 5.10.192-tegra`.
- L4T/JetPack line: `R35.5.0`.
- Device name in Torch: `Orin`.
- Visible RAM: about 62,802 MB.
- Power mode: `MAXN`.
- PyTorch: `2.1.0a0+41361538.nv23.06`.
- TensorRT Python package: `8.5.2.2`.
- ONNXRuntime: `1.18.0`.
- ONNXRuntime providers: `TensorrtExecutionProvider`,
  `CUDAExecutionProvider`, `CPUExecutionProvider`.

The original run was executed in an isolated AGX workspace:

```text
/home/jetson/rtss_agx_codex_20260525_001
```

## Included Result Blocks

The archive includes:

- MobileNetV2 stage-1/stable real-frame run.
- ResNet50 real-frame run.
- Full backend matrix over MobileNetV2, ResNet18, ResNet50, and SqueezeNet1.1.
- ONNXRuntime CUDA/TensorRT provider validation.
- Pressure-test results and postmortem logs.
- Run scripts and nohup logs.
- Local wheel cache, model-weight cache, and isolated library side-loads.

## MobileNetV2 Real-Frame Run

Result root:

```text
agx_orin64_run_package_20260524_full\results_agx_orin64_comprehensive_stage1_stable_codex_20260525_160051
```

Coverage:

- Datasets: AV2, DROID, nuImages.
- Deadlines: 33.333 ms and 50 ms.
- Model: MobileNetV2.
- Methods: AlwaysAudit, CIRCA-RT, CIRCA-RT-Slack, ConditionalRFF-HSIC,
  ContextAwareConformal, RFF-HSIC, RandomBudgetAudit, SemanticThreshold,
  TimingThreshold.
- Main rows: 54.
- Audit-bound validation rows: 504.
- Audit-bound failed rows: 0.

Key CIRCA-RT rows:

| Dataset | Deadline | Recall | Audit Rate | p99 | Miss |
|---|---:|---:|---:|---:|---:|
| AV2 | 33.333 | 0.692 | 0.089 | 19.275 ms | 0.000 |
| AV2 | 50.000 | 0.537 | 0.067 | 18.679 ms | 0.000 |
| DROID | 33.333 | 0.459 | 0.047 | 16.764 ms | 0.000 |
| DROID | 50.000 | 0.593 | 0.065 | 18.032 ms | 0.000 |
| nuImages | 33.333 | 0.672 | 0.082 | 18.223 ms | 0.000 |
| nuImages | 50.000 | 0.620 | 0.073 | 18.046 ms | 0.000 |

Tegrastats aggregate:

```text
samples: 1414
max RAM: 13639 MB / 62802 MB
max CPU temperature: 58 C
max GPU temperature: 51 C
swap used: 0
```

## ResNet50 Real-Frame Run

Result root:

```text
agx_orin64_run_package_20260524_full\results_agx_orin64_comprehensive_resnet50_realframes_codex_20260525_1726
```

Coverage:

- Datasets: AV2, DROID, nuImages.
- Deadlines: 33.333 ms and 50 ms.
- Model: ResNet50.
- Methods: same 9-method set as the MobileNetV2 real-frame run.
- Main rows: 54.
- Audit-bound validation rows: 504.
- Audit-bound failed rows: 0.

Key CIRCA-RT rows:

| Dataset | Deadline | Recall | Audit Rate | p99 | Miss |
|---|---:|---:|---:|---:|---:|
| AV2 | 33.333 | 0.557 | 0.067 | 20.137 ms | 0.001 |
| AV2 | 50.000 | 0.564 | 0.065 | 19.510 ms | 0.000 |
| DROID | 33.333 | 0.598 | 0.064 | 18.935 ms | 0.000 |
| DROID | 50.000 | 0.764 | 0.075 | 19.761 ms | 0.000 |
| nuImages | 33.333 | 0.677 | 0.082 | 20.092 ms | 0.000 |
| nuImages | 50.000 | 0.575 | 0.077 | 20.164 ms | 0.000 |

Important slack result:

- AV2 / ResNet50 / 33.333 ms: CIRCA-RT miss ratio is 0.001333, while
  CIRCA-RT-Slack keeps the same recall and audit rate with 0.000 miss ratio.
- AlwaysAudit has nonzero miss ratios at tight 33.333 ms in AV2, DROID, and
  nuImages, which supports the selective-audit latency argument.

Tegrastats aggregate:

```text
samples: 1454
max RAM: 20274 MB / 62802 MB
max CPU temperature: 57 C
max GPU temperature: 51 C
swap used: 0
```

## Full Backend Matrix

Result root:

```text
agx_orin64_run_package_20260524_full\results_agx_orin64_env\ortgpu_backend_matrix_full_20260525_1740
```

Coverage:

- Models: MobileNetV2, ResNet18, ResNet50, SqueezeNet1.1.
- Runtimes: PyTorch CUDA, ONNX CUDA, ONNX TensorRT.
- Seeds: 7, 8, 9.
- Samples per trace: `n=1200`.
- Latency rows: 108.
- Monitoring rows: 180.
- Provider-status rows: 9.
- Provider-status result: 8 `ok` rows plus 1 environment `available` row.
- Max deadline miss ratio in latency table: 0.0.

Latency summary:

| Runtime | Provider | Model | Mean | Mean p99 | Max p99 | Max Miss |
|---|---|---|---:|---:|---:|---:|
| ONNX CUDA | CUDAExecutionProvider | MobileNetV2 | 4.356 ms | 5.641 ms | 8.027 ms | 0.000 |
| ONNX TensorRT | TensorrtExecutionProvider | MobileNetV2 | 1.406 ms | 1.683 ms | 2.937 ms | 0.000 |
| PyTorch CUDA | PyTorch-CUDA | MobileNetV2 | 9.658 ms | 9.908 ms | 10.094 ms | 0.000 |
| ONNX CUDA | CUDAExecutionProvider | ResNet18 | 3.701 ms | 4.740 ms | 6.979 ms | 0.000 |
| ONNX TensorRT | TensorrtExecutionProvider | ResNet18 | 1.429 ms | 1.736 ms | 2.479 ms | 0.000 |
| PyTorch CUDA | PyTorch-CUDA | ResNet18 | 4.529 ms | 4.620 ms | 4.751 ms | 0.000 |
| ONNX CUDA | CUDAExecutionProvider | ResNet50 | 7.982 ms | 9.427 ms | 12.774 ms | 0.000 |
| ONNX TensorRT | TensorrtExecutionProvider | ResNet50 | 2.731 ms | 3.649 ms | 5.574 ms | 0.000 |
| PyTorch CUDA | PyTorch-CUDA | ResNet50 | 11.156 ms | 11.409 ms | 11.750 ms | 0.000 |
| ONNX CUDA | CUDAExecutionProvider | SqueezeNet1.1 | 3.573 ms | 5.146 ms | 7.572 ms | 0.000 |
| ONNX TensorRT | TensorrtExecutionProvider | SqueezeNet1.1 | 1.032 ms | 1.297 ms | 1.897 ms | 0.000 |
| PyTorch CUDA | PyTorch-CUDA | SqueezeNet1.1 | 4.151 ms | 4.212 ms | 4.450 ms | 0.000 |

This is the canonical statistical backend matrix for the AGX result package.

## Pressure-Test Evidence

Result root:

```text
agx_orin64_run_package_20260524_full\results_agx_orin64_pressure_codex_20260525_1815
```

Passed real-frame pressure levels:

| Level | CIRCA Recall | CIRCA Audit | CIRCA p99 | CIRCA Miss | Slack p99 | Slack Miss | Audit Bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| stress512_r1_x4 | 0.494 | 0.059 | 19.070 ms | 0.000 | 19.065 ms | 0.000 | 84/84 |
| stress1024_r2_x8 | 0.733 | 0.079 | 20.440 ms | 0.002 | 20.432 ms | 0.000 | 84/84 |
| stress1536_r2_x12 | 0.863 | 0.091 | 21.698 ms | 0.007 | 21.631 ms | 0.005 | 84/84 |

All staged real-frame pressure commands returned `rc=0`, and
`failed_commands.log` is empty.

Backend sustained evidence before the extreme overlay:

```text
runtime/provider/model: ONNX CUDA / CUDAExecutionProvider / ResNet50
trace: seed7 nominal
rows: 3000
deadline miss ratio: 0.0
mean latency: 9.647 ms
p95 latency: 9.880 ms
p99 latency: 10.080 ms
max latency: 11.671 ms
```

Extreme overlay outcome:

- Existing background pressure: 4 CPU workers and 8 GB memory pressure.
- Additional overlay: 24 GB memory pressure and GPU 4096 matrix pressure.
- Outcome: AGX rebooted during backend sustained execution.
- Pre-reboot telemetry from `tegrastats_backend_ort_resnet50_sustained.log`:

```text
samples: 487
max RAM: 25620 MB / 62802 MB
swap used: 0
max CPU temperature: 58 C
max GPU temperature: 52 C
max CPU utilization: 100%
max GR3D frequency parse: 78
```

Interpretation:

- The staged pressure tests are valid passed results.
- The extreme overlay is valid boundary evidence.
- The extreme overlay must be reported as a reboot boundary, not as a passed
  pressure run.
- The available telemetry argues against OOM and thermal runaway. The safer
  explanation is board-level stability, power envelope, or simultaneous
  CPU/GPU/memory stress boundary.

## Paper-Ready Claims

Safe wording:

> We evaluate CIRCA-RT on a real Jetson AGX Orin 64GB-class edge platform using
> PyTorch CUDA real-frame streams from AV2, DROID, and nuImages, and we validate
> ONNXRuntime CUDA/TensorRT execution in a full backend matrix over four CNN
> backbones. The AGX run records per-frame latency, deadline misses, audit
> decisions, audit-bound validation, provider status, and tegrastats telemetry.
> CIRCA-RT-Slack preserves detection behavior while reducing tight-deadline miss
> risk, and staged ResNet50 pressure runs pass up to `stress1536_r2_x12`.

Safe pressure wording:

> Under an additional extreme overlay of 24 GB memory pressure and GPU 4096
> matrix pressure during backend sustained execution, the AGX rebooted despite
> RAM, swap, and temperature telemetry remaining below obvious OOM or thermal
> limits. We report this as a device stability boundary rather than a passed
> stress test.

Do not write:

- The extreme backend sustained overlay passed.
- The reboot was caused by OOM.
- The reboot was caused by thermal throttling or overheating.
- The experiment measured energy or power efficiency.
- Every real-frame backbone was run with multiple seeds. The full multi-backbone
  matrix is backend-level; the real-frame tables are MobileNetV2 and ResNet50.

## Overall Quality

The AGX evidence is now strong enough to close the earlier real-edge-device gap.
Compared with the previous stage-1 archive, this full archive adds the two most
important missing pieces: ResNet50 real-frame evidence and a full ONNX
CUDA/TensorRT backend matrix.

Remaining work is mostly integration and presentation:

- Update paper tables/figures to use this full archive, not obsolete AGX
  material.
- Separate real-frame results from backend-matrix results in the writing.
- Present the pressure reboot as a boundary/postmortem result, not a failure of
  the main method.
- Keep all SHA256 hashes and extracted result paths in the artifact appendix.
