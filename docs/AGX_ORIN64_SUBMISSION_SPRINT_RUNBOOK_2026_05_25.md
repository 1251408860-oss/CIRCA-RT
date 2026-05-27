# AGX Orin 64GB Submission Sprint Runbook

Date: 2026-05-25

This is the command set to send with the AGX package when the goal is to collect
the remaining RTSS-facing evidence in one device session.

## Recommended Full Run

From the AGX package root:

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260525_submission_full
source /opt/ros/foxy/setup.bash  # if ROS2 is installed there

PHASES="env ros2_closed_loop resnet50_realframes backend_matrix pressure_short" \
FAIL_FAST=0 \
bash scripts/run_agx_orin64_submission_sprint.sh
```

The default stages are:

- `env`: device, L4T, power mode, CUDA/TensorRT/ONNXRuntime/ROS2 evidence.
- `ros2_closed_loop`: ROS2 image-frame publisher, DDS image transport, CUDA
  perception monitor, online CIRCA-RT-Slack, inline heavy audit.
- `resnet50_realframes`: ResNet50 real-frame replay on AV2, DROID, and
  nuImages at 33.333 ms and 50 ms.
- `backend_matrix`: PyTorch CUDA, ONNX CUDA, and ONNX TensorRT backend matrix
  over MobileNetV2, ResNet18, ResNet50, and SqueezeNet1.1.
- `pressure_short`: compact deadline/pressure sweep for stress evidence.

## Faster Fallback

If time is tight:

```bash
PHASES="env ros2_closed_loop pressure_short" \
ROS2_N=300 \
bash scripts/run_agx_orin64_submission_sprint.sh
```

If ROS2 is not available:

```bash
PHASES="env resnet50_realframes backend_matrix pressure_short" \
bash scripts/run_agx_orin64_submission_sprint.sh
```

## Return Results

The script writes a result root like:

```text
results_agx_orin64_submission_sprint_<timestamp>/
```

Return this file from the AGX:

```bash
tar -czf agx_submission_sprint_results.tar.gz results_agx_orin64_submission_sprint_*
sha256sum agx_submission_sprint_results.tar.gz
```

The first file to inspect is:

```text
results_agx_orin64_submission_sprint_<timestamp>/tables/stage_status.csv
```

## Claim Boundary

Use the ROS2 closed-loop result only if the `ros2_closed_loop` stage completes
and produces tables. If it fails because ROS2/rclpy is not installed, do not
claim AGX ROS2 closed-loop; keep the existing AGX inference/backend/pressure
evidence and use AutoDL ROS2/DDS as middleware evidence.
