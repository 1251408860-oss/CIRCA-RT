# AGX Orin 64GB ROS2 Closed-Loop Runbook

Date: 2026-05-25

This runbook adds the optional full AGX ROS2 closed-loop validation path. It is
separate from the completed AGX real-frame replay, TensorRT/CUDA backend matrix,
and pressure-test archive.

## What This Run Runs

The runner executes a real ROS2 runtime path on the AGX:

```text
ROS2 timer frame publisher
  -> DDS topic carrying compressed JPEG frame payloads and scenario labels
  -> Torch CUDA perception/monitor node
  -> online CIRCA-RT or CIRCA-RT-Slack admission
  -> inline heavy-audit model execution when admitted
  -> per-frame scored trace and summary tables
```

The implementation intentionally uses `std_msgs/String` JSON payloads with
base64 JPEG frames so it can run without building custom ROS2 message packages.
Set `PAYLOAD_MODE=path` only as a fallback if the ROS2/DDS stack cannot sustain
image payloads.

## Main Command

On the AGX workspace, source the ROS2 setup first if it is not already in the
environment. JetPack/L4T R35.5.0 is Ubuntu 20.04, so Foxy is the expected binary
ROS2 line if ROS2 was installed from packages.

```bash
cd /home/jetson/rtss_agx_codex_20260525_001/agx_orin64_run_package_20260524_full
source /opt/ros/foxy/setup.bash

FRAME_DIR=data/frames_av2 \
MODELS="mobilenet_v2" \
SEEDS="7 8 9" \
N=600 \
PERIOD_MS=33.333 \
DEADLINE_MS=33.333 \
PAYLOAD_MODE=jpeg_b64 \
PUBLISH_RESIZE=320 \
bash scripts/run_agx_orin64_ros2_closed_loop.sh
```

For a stronger but longer run:

```bash
FRAME_DIR=data/frames_av2 \
MODELS="mobilenet_v2 resnet18" \
SEEDS="7 8 9" \
N=900 \
bash scripts/run_agx_orin64_ros2_closed_loop.sh
```

If the AGX package does not contain raw frames, point `FRAME_DIR` to the copied
real-frame directory used by the completed AGX replay run.

## Outputs

The result root defaults to:

```text
results_agx_orin64_ros2_closed_loop_<timestamp>/
```

Important files:

- `tables/agx_ros2_closed_loop_main_table.csv`
- `tables/agx_ros2_closed_loop_scenario_table.csv`
- `tables/agx_ros2_closed_loop_latency_table.csv`
- `tables/audit_bound_validation.csv`
- `tables/tegrastats_summary.csv`
- `traces/<model>/seed<seed>/<scenario>.csv`
- `raw/<model>/seed<seed>/<scenario>/CIRCA-RT-Slack/scored.csv`
- `env/system_ros2_snapshot.txt`
- `env/python_ros2_snapshot.txt`
- `logs/agx_ros2_closed_loop.log`

## Claim Boundary

If this run completes on the AGX, the safe paper claim is:

> We additionally validate the complete AGX ROS2 runtime path with a timer-driven
> image-frame publisher, DDS image transport, online perception monitoring,
> slack-admissible audit admission, and measured heavy-audit execution.

Do not turn this into an autonomous-driving closed-loop claim. This run does not
include planning, control, actuators, TSN hardware, or a vehicle/robot plant.

If ROS2 is not installed or `rclpy` is not importable, the runner fails closed.
In that case, keep the existing AGX evidence as edge inference/runtime evidence
and keep AutoDL ROS2/DDS as middleware evidence.
