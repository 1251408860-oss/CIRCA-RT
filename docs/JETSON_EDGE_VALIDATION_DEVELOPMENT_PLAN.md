# Jetson Edge Validation Development Plan

Updated: 2026-05-21

Status update on 2026-05-25: the current AGX Orin 64GB-class evidence is the
completed full archive documented in
`docs/AGX_ORIN64_FULL_ARCHIVE_VALIDATION_2026_05_25.md`. Treat this file as a
development note, not as the source of final AGX claims.

This document remains useful as the validation plan and checklist.

## Purpose

The Jetson work is not a second copy of the AutoDL experiments.
It is the final edge-device proof that CIRCA-RT is an online, bounded-overhead monitoring system.

The target is a small but defensible RTSS-style validation:

- a real Jetson Orin device
- a live frame/video replay pipeline
- online perception + CIRCA-RT monitoring
- real selective audit work
- measured tail latency, audit cost, and power/thermal behavior

The paper claim should stay narrow:

> CIRCA-RT preserves low audit overhead and controlled tail latency on a real edge device while detecting coupled semantic-timing anomalies.

Do not turn this into a recall-SOTA claim.

## Why This Order

The Jetson work should remove three reviewer objections in sequence:

1. "This is offline trace scoring."
2. "Audit cost is assigned, not measured."
3. "The result only uses one narrow validation source."

That is why the order is:

1. one Jetson online loop
2. one real heavy audit
3. one standard driving source
4. one robotics source for domain contrast

## System Shape

Use one pipeline:

```text
frame/video replay
-> perception node
-> CIRCA-RT monitor node
-> selective audit node
-> trace logger
```

The logger must export the same trace schema used by the current AutoDL/ROS2 code so that the existing summary scripts can be reused.

Required Jetson fields:

- `p50/p95/p99/p999 latency`
- `deadline miss ratio`
- `audit rate`
- `monitor cost`
- `audit cost`
- `power`
- `temperature`
- `frequency`
- `actual provider`
- `engine path`

## Phase 0: Device Bring-Up

Goal: prove that the Jetson image is usable before any experiment work.

Checklist:

- SSH works
- `sudo` works
- `tegrastats` works
- `nvpmodel` works
- `jetson_clocks` works
- `TensorRT` imports or a real engine can be executed
- ROS2 runtime matches the device image

Use the existing helpers first:

- `scripts/jetson_env_check.sh`
- `scripts/run_jetson_tegrastats.sh`
- `scripts/prepare_jetson_strong_pipeline.sh`

If TensorRT cannot be activated, do not claim TensorRT results.
Use ONNXRuntime or PyTorch only as a fallback smoke test.

## Phase 1: Minimal Online Loop

Build the smallest possible online path on Jetson:

- start from replayed frames or a short video clip
- publish frames at a fixed rate
- run one fast perception model
- compute CIRCA-RT decisions online
- log alarms, audits, and latencies

Start with one model pair only:

- fast path: MobileNetV2 or ResNet18
- heavy audit path: a stronger model, a higher resolution rerun, or a second-pass consistency check

The purpose of Phase 1 is not accuracy.
The purpose is to prove the system runs online and the logs are complete.

## Phase 2: Real Heavy Audit

This is the main missing evidence.

`audit=1` must cause a real extra computation, not a synthetic cost field.

Good audit choices:

- rerun the same model at higher input resolution
- rerun a larger model on the same frame
- run a second model and compare outputs
- run detector + temporal consistency check

Bad audit choices:

- hand-assigned costs
- fixed dummy delays
- offline post-processing that is later labeled as audit

Report the audit cost separately from the monitor cost.
Also report the aggregate effect on `p99` and deadline misses.

## Phase 3: Standard Real Sources

The first source should be a standard driving dataset that reviewers already recognize.
Preferred order:

1. `nuScenes mini`
2. `BDD100K` sample or small clip subset
3. keep `AV2` as an internal support source, not the only driving source

The second source should be a robotics or robot-camera source:

- `DROID / LeRobot`
- or another robot-cam / indoor camera sequence if available

The point is not to maximize dataset count.
The point is to show that the method survives both driving and robotics motion statistics.

## Phase 4: Experiment Matrix

Keep the first Jetson matrix small:

| Item | Choice |
|---|---|
| Source 1 | standard driving dataset |
| Source 2 | robotics dataset |
| Models | 1 fast model + 1 heavy audit model |
| Scenarios | nominal, interference, coupled attack |
| Seeds | 7, 8, 9 |
| Policies | CIRCA-RT, AlwaysAudit, ContextAwareConformal, one strong baseline |
| Logs | ROS2 trace + `tegrastats` |

Do not start with the full benchmark suite.
That expands risk without changing the story.

## Phase 5: What Must Be Measured

Every run should let you answer these questions:

- Does CIRCA-RT run online on Jetson?
- Is the active provider actually TensorRT or a fallback?
- What is the real audit cost when audit is triggered?
- How much p99 latency changes under interference?
- How much power and thermal headroom remains?
- Is the result stable across two different real sources?

## Required Scripts

Implement the Jetson part as a small set of scripts, not as ad-hoc manual commands:

- `scripts/run_jetson_orin_validation.sh`
- `scripts/run_jetson_tensorrt_engine_check.py`
- `scripts/parse_tegrastats.py`
- `scripts/summarize_jetson_perception_results.py`
- optional future work: a dedicated ROS2 node runner once the replay path is stable

These scripts should reuse the current manifest and summary format whenever possible.

## No-Go Conditions

Do not write the Jetson section as complete if any of these happen:

- TensorRT silently falls back
- audit is not a real extra computation
- no `tegrastats` log is captured
- the trace only exists offline after the fact
- the paper claim becomes universal recall dominance

## Definition of Done

The Jetson validation is done when all of the following are true:

- one Jetson online replay pipeline runs end-to-end
- one real heavy audit path is measured
- one standard driving source is completed
- one robotics source is completed
- results are exported in the same schema as the current AutoDL traces
- summary tables can be generated without manual cleanup

## Reference Material

- Jetson Orin development notes in this repo: `docs/JETSON_ORIN_DEVELOPMENT_GUIDE.md`
- Jetson pipeline sketch in this repo: `docs/JETSON_STRONG_PIPELINE.md`
- JetPack 6.1 release notes: https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-61/release-notes/index.html
- TensorRT Python API docs: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html
- Jetson Linux power management and clocking docs: https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonAgxOrinSeries.html
- nuScenes official site: https://www.nuscenes.org/
- nuScenes mini documentation / tutorials: https://www.nuscenes.org/tutorials
