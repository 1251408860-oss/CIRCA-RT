# Jetson Strong Pipeline Plan

更新时间：2026-05-16

这是 Jetson 强验证方案。现在采用 AutoDL-first 策略：能在 AutoDL 上完成的 TensorRT、ROS2/Python pipeline、批量 baseline 和脚本调试先在 AutoDL 完成；Jetson 只保留 edge-class final validation。

## Goal

最终在 Jetson 上构建一条同机真实链路：

```text
ROS2/DDS sensor publisher
-> perception / TensorRT inference node
-> CIRCA-RT monitor node
-> selective audit node
-> logging and trace export
```

这条链路同时提供：

- ROS2/DDS message age
- callback latency
- queue depth proxy
- TensorRT batch-1 inference latency
- monitor overhead
- selective audit overhead
- always audit overhead
- p99/p999 latency
- power / frequency / temperature

## Recommended Jetson Environment

优先租：

- Jetson Orin NX 或 AGX Orin
- JetPack 6.x
- Ubuntu 22.04
- CUDA / TensorRT 已安装
- SSH 可用
- `sudo` 可用
- `tegrastats` 可用
- `nvpmodel` 可用
- `jetson_clocks` 可用

ROS2 版本建议：

- JetPack 6.x / Ubuntu 22.04: use ROS2 Humble
- Ubuntu 24.04 only: use ROS2 Jazzy

不要在 JetPack 6.x 上硬装 Jazzy 作为主路线，版本不匹配会浪费时间。

## Experiment Matrix

AutoDL-first 后，Jetson 不需要重复全量 baseline。Jetson 最小强验证矩阵：

| Scenario | ROS2/DDS | TensorRT | CIRCA-RT | Audit Policy | Purpose |
|---|---|---|---|---|---|
| perception_only | yes | yes | no | none | base latency |
| monitor_only | yes | yes | yes | none | monitor overhead |
| selective_audit | yes | yes | yes | token bucket | main method |
| always_audit | yes | yes | yes | always | upper bound |
| coupled_attack | yes | yes | yes | token bucket | main CIRCA claim |
| jitter_attack | yes | yes | yes | token bucket | DDS timing attack |
| stale_replay_attack | yes | yes | yes | token bucket | sensing freshness |

## Trace Schema

Every run must export the existing CIRCA-RT CSV schema:

- `run_id`
- `platform`
- `timestamp_ms`
- `seq`
- `mode`
- `attack_type`
- `label`
- `semantic_residual`
- `timing_residual_ms`
- `context_cpu_util`
- `context_gpu_util`
- `context_net_delay_ms`
- `context_jitter_ms`
- `context_queue_depth`
- `context_message_age_ms`
- `baseline_latency_ms`
- `e2e_latency_ms`
- `deadline_ms`
- `deadline_miss`
- `method`
- `alarm`
- `audit`
- `audit_cost_ms`
- `monitor_cost_ms`

Platform label:

```text
jetson_orin_ros2_tensorrt
```

## Commands On Jetson

Environment checks:

```bash
cat /etc/nv_tegra_release
lsb_release -a
python3 --version
python3 -c "import tensorrt as trt; print(trt.__version__)"
tegrastats --interval 100
sudo nvpmodel -q
sudo jetson_clocks --show
ros2 --version
printenv ROS_DISTRO
```

Power modes:

```bash
sudo nvpmodel -q
sudo jetson_clocks
```

Run tegrastats in a side terminal:

```bash
mkdir -p logs
tegrastats --interval 100 | tee logs/tegrastats_$(date +%Y%m%d_%H%M%S).log
```

## Result Files

Expected output layout on Jetson:

```text
jetson_runs/
  raw_logs/
  traces/
  scored/
  tables/
```

After download to this Windows project:

```text
F:\RTSS\exp_begin\results_jetson\
  traces\
  raw\
  summaries\
  tables\
  figures\
```

## Paper Claim

Strong claim enabled only after a fresh valid real-device run:

> CIRCA-RT is evaluated on a Jetson AGX Orin 64GB TensorRT edge-inference block, showing coupled timing-semantic anomaly detection with selective audit overhead and tail-latency control.

Claims still not allowed:

- hardware TSN validation
- hard real-time guarantee
- full autonomous driving stack
- full Holoscan reproduction unless actually used

AutoDL-first relation:

- AutoDL provides server-GPU and ROS2/DDS bulk baseline evidence
- Jetson provides edge-GPU tail-latency and power/resource evidence
- Both export the same CIRCA-RT CSV schema

## Go / No-Go

Go:

- Jetson exports at least `perception_only`, `selective_audit`, `always_audit`, and `coupled_attack` CSV traces
- `tegrastats` logs are captured
- p99/p999 latency and audit rate can be computed

No-Go:

- no TensorRT available
- no SSH download path
- no sudo and no tegrastats
- ROS2 cannot run and Python fallback is also blocked

Rental checklist:

- `docs/JETSON_RENTAL_CHECKLIST.md`
