# RTSS High-Probability Development Plan

Updated: 2026-05-19

Status update on 2026-05-25: treat this as a forward-looking development plan.
The current AGX Orin 64GB-class evidence is the completed full archive
documented in `docs/AGX_ORIN64_FULL_ARCHIVE_VALIDATION_2026_05_25.md`.

This document defines the development path for turning CIRCA-RT from a competitive RTSS submission into a high-confidence RTSS full-paper candidate. The target is not a literal guarantee of acceptance. RTSS 2025 accepted 44 out of 200 evaluated submissions, so `80%+` should be interpreted as an internal quality bar: after this plan, the paper should have no obvious fatal reviewer objection.

The current project already has a strong experimental base:

- Split synthetic calibration/test experiments.
- AutoDL server-GPU traces.
- ROS2/DDS two-process traces.
- Budget-normalized comparisons.
- Recent deep TSAD adapters.
- Local ablation and paper figure generation.

The missing pieces are not more tables of the same kind. The missing pieces are:

- Real perception-derived semantic signals.
- Online Jetson/ROS2/TensorRT execution rather than offline trace scoring.
- Real measured audit work rather than assigned audit cost.
- A formal timing/audit bound.
- Artifact packaging that makes the result hard to dismiss.

## External RTSS Facts

Use these facts to shape the paper and experiments.

- RTSS 2026 CFP says every submission must explicitly address real-time requirements or constraints.
- RTSS 2026 Track 2 explicitly includes real-time systems for AI, AI safety in CPS, runtime monitoring, robustness under uncertainty, AI-at-the-edge, SoC design, power, thermal behavior, timing, and predictability.
- RTSS 2026 allows supplementary artifacts up to 600 MB, including source code, data sets, formal models, and short demo videos, but reviewers are not required to inspect them.
- RTSS 2025 accepted 44 out of 200 evaluated submissions.
- RTSS 2025 already had related accepted work in ROS2, TensorRT, edge inference, Holoscan, predictable video latency, stale sensing streams, and secure system auditing.

Official references:

- RTSS 2026 CFP: https://2026.rtss.org/cfp/
- RTSS 2026 submission rules: https://2026.rtss.org/submission/
- RTSS 2025 accepted paper count: https://2025.rtss.org/news/list-of-accepted-papers/index.html
- RTSS 2025 program: https://2025.rtss.org/program/index.html

## Probability Model

Current realistic full-paper probability:

- Existing AutoDL/ROS2 package only: `10%-18%`.
- Add the current Jetson guide only: `22%-35%`.
- Add this full plan with clean results: `50%-70%`.
- `80%+` full-paper probability is only plausible if the result becomes close to best-paper-candidate quality: real online system, real semantic signal, measured audit work, strong bound, strong baselines, and no negative headline result.

For RTSS 2026 specifically:

- Abstract deadline: 2026-05-21.
- Paper deadline: 2026-05-26.
- If the hardware, data, and scripts are not already available, the full `80%+` path is not feasible before 2026-05-26.
- The emergency 2026 path is to implement the smallest real end-to-end slice and present the remaining AutoDL/ROS2 results as supporting evidence.

## Target Paper Claim

Strong claim to enable:

> CIRCA-RT is a bounded-overhead semantic-timing runtime auditing layer for real-time edge perception pipelines. On a real Jetson Orin ROS2/TensorRT deployment, it detects coupled semantic-timing anomalies while preserving tail-latency and audit-workload constraints better than always-audit and high-cost anomaly monitors.

Do not claim:

> CIRCA-RT is the highest-recall detector across all attacks and platforms.

Do not claim:

> CIRCA-RT is validated on physical TSN hardware.

Do not claim unless implemented:

> Heavy audit is real extra computation.

## Reviewer Objection Matrix

| Likely reviewer objection | Current risk | Required fix |
|---|---:|---|
| Semantic residuals are synthetic or label-derived. | Fatal | Derive semantic residuals from real model outputs on real video/image streams. |
| The system is offline trace scoring, not online monitoring. | High | Run CIRCA-RT inside a live ROS2/TensorRT pipeline and log decisions online. |
| Audit cost is assigned, not measured. | High | Heavy audit must invoke a real expensive operation and record measured cost. |
| No real-time analysis. | High | Add a token-bucket audit bound and connect it to latency/deadline miss. |
| CIRCA-RT is not recall-SOTA on ROS2. | Medium | Reframe around Pareto frontier under audit and tail-latency constraints. |
| TensorRT may silently fall back. | High | Record active provider, engine build logs, and fail closed if fallback occurs. |
| Deep baselines are weaker adapters. | Medium | Use official implementations where feasible or label adapters honestly. |
| Dataset/attack construction may be tuned to CIRCA-RT. | Medium | Add marginal-matching diagnostics, blinded scenarios, and non-coupled negative controls. |
| Artifact is too hard to reproduce. | Medium | Package a double-anonymous 600 MB supplement with scripts, small data, logs, and a demo video. |

## Acceptance Gates

Do not submit as a high-confidence full paper unless all `Must Pass` gates pass.

| Gate | Requirement | Status target |
|---|---|---|
| G1 Real semantic signal | `semantic_residual` is computed from perception outputs, never from `label`. | Must Pass |
| G2 Online execution | Main Jetson results are generated by an online pipeline, not by offline rescoring only. | Must Pass |
| G3 Real audit | `audit=1` triggers real heavier computation or cross-checking. | Must Pass |
| G4 TensorRT validity | TensorRT result is reported only when a real engine/provider is active. | Must Pass |
| G5 Timing bound | Paper includes token-bucket audit-workload and added-latency bound. | Must Pass |
| G6 Pareto result | CIRCA-RT is on or near the recall-vs-audit and recall-vs-p99 Pareto frontier in main scenarios. | Must Pass |
| G7 Benign robustness | False alarm under benign mode shifts stays within the paper's stated operating range. | Must Pass |
| G8 Artifact | Reproduction package runs one small end-to-end demo and all table-generation scripts. | Must Pass |
| G9 Multi-platform support | Jetson main result plus AutoDL/ROS2 supporting results. | Should Pass |
| G10 Official deep baselines | At least one official deep TSAD implementation or a clearly documented adapter limitation. | Should Pass |

## Workstream A: Real Perception Semantics

Goal:

Replace synthetic or label-derived semantic residuals with real perception-derived signals.

Recommended inputs:

- Primary: camera or video replay through ROS2.
- Practical dataset choices: KITTI raw, BDD100K sample, nuScenes mini, or a local camera stream.
- If download time is tight, use a small curated video set with at least day/night, indoor/outdoor, high-motion, and low-motion segments.

Recommended perception models:

- Fast model: MobileNetV2 classifier or YOLOv5n/YOLOv8n detector.
- Heavy audit model: ResNet50, YOLOv5s/YOLOv8s, higher-resolution inference, or double inference with temporal consistency check.
- Runtime: TensorRT first, ONNXRuntime second, PyTorch only as fallback.

Semantic features to log per frame:

- `confidence_entropy`: entropy of class or detection confidence distribution.
- `top1_margin`: top-1 minus top-2 probability.
- `embedding_mahalanobis`: distance from benign calibration embedding distribution.
- `temporal_feature_drift`: cosine or L2 drift from previous frame embedding.
- `object_count_delta`: frame-to-frame object-count change for detector pipelines.
- `box_temporal_iou_error`: tracking or detection consistency error when object boxes are available.
- `audit_disagreement`: disagreement between fast model and heavy audit model when audit runs.

Semantic residual definition:

```text
semantic_residual_t =
  standardized residual of semantic score after conditioning on
  scene_mode, frame_brightness, motion_proxy, cpu_util, gpu_util,
  queue_depth, and message_age.
```

Rules:

- Do not use `label` in semantic feature construction.
- Do not rewrite semantic residuals in attack post-processing for the main Jetson result.
- Labels may be used only for evaluation after the trace has been captured.
- The paper must explicitly distinguish real perception-derived residuals from synthetic residuals used in early experiments.

Files to add:

```text
src/circa_rt/perception_semantics.py
scripts/run_video_perception_trace.py
scripts/run_jetson_perception_pipeline.py
```

Expected trace output:

```text
results_jetson_orin_perception/traces/seed7/nominal.csv
results_jetson_orin_perception/traces/seed7/mode_shift.csv
results_jetson_orin_perception/traces/seed7/semantic_corruption.csv
results_jetson_orin_perception/traces/seed7/timing_interference.csv
results_jetson_orin_perception/traces/seed7/coupled_semantic_timing_attack.csv
```

## Workstream B: Online Jetson ROS2/TensorRT Pipeline

Goal:

Create one live pipeline that generates the main result.

Target graph:

```text
ROS2 video/frame publisher
-> perception node
-> CIRCA-RT monitor node
-> selective audit node
-> trace logger
```

Required online behavior:

- The perception node publishes frame-level semantic features and inference latency.
- The monitor node consumes features and emits `score`, `alarm`, `audit`, and `bucket_level` online.
- The audit node runs only when `audit=1`.
- The logger records the unified CIRCA-RT schema plus Jetson-specific fields.

Required measurements:

- ROS2 message age.
- Callback latency.
- Queue depth proxy.
- Fast inference latency.
- Monitor latency.
- Heavy audit latency.
- End-to-end latency.
- p50, p95, p99, p999.
- Deadline miss ratio.
- Power, temperature, and frequency from `tegrastats`.

Runtime validation:

- TensorRT engine path must be recorded.
- Active execution provider must be recorded.
- Engine build logs must be saved.
- If TensorRT fails or falls back, the run must be marked invalid for TensorRT claims.

Files to add:

```text
scripts/run_jetson_ros2_perception_pipeline.py
scripts/run_jetson_tensorrt_engine_check.py
scripts/parse_tegrastats.py
scripts/summarize_jetson_perception_results.py
configs/rtss80_jetson_config.json
```

Output layout:

```text
results_jetson_orin_perception/
  logs/
  raw_ros2/
  traces/
  raw/
  tables/
  figures/
  tegrastats/
  engines/
  provider_status.csv
```

## Workstream C: Real Heavy Audit

Goal:

Make selective audit a real system action.

Valid heavy audit implementations:

- Run a larger model on the same frame.
- Run the same model at higher input resolution.
- Run temporal consistency over the last `k` frames.
- Run a duplicate inference on a separate runtime and compare outputs.
- Run an object tracker consistency check for detector output.

Invalid main-result implementations:

- Setting `audit_cost_ms = 4.0` without actually doing work.
- Offline recomputation of audit latency.
- Using label information as audit output.

Required audit columns:

```text
audit_model
audit_runtime
audit_start_ns
audit_end_ns
audit_latency_ms
audit_disagreement
audit_result
```

The existing schema should keep:

```text
audit
audit_cost_ms
e2e_latency_ms
deadline_miss
```

`audit_cost_ms` must be measured from `audit_latency_ms` for main Jetson results.

## Workstream D: Attack And Scenario Design

Goal:

Make attacks realistic enough that reviewers cannot dismiss them as a constructed dependence trick.

Main scenarios:

| Scenario | Purpose | Required source |
|---|---|---|
| `nominal` | Calibration and clean test. | Real video or camera. |
| `benign_mode_shift` | Distribution shift without attack. | Different lighting, motion, or scene. |
| `gpu_interference` | Timing perturbation without semantic attack. | Real background GPU load. |
| `semantic_corruption` | Semantic perturbation without timing attack. | Blur, occlusion, brightness, compression, patch. |
| `stale_replay` | Sensing freshness attack. | Replay old frames or hold frames. |
| `drop_reorder_jitter` | Middleware timing attack. | ROS2 drop/reorder/sleep. |
| `coupled_semantic_timing_attack` | Main threat model. | Simultaneous semantic corruption and timing stress with marginal checks. |

Coupled attack rule:

- The attack should not simply increase both semantic and timing marginal magnitudes.
- The main change should be conditional dependence between semantic residual and timing residual after context residualization.
- Include a table comparing benign and attack marginals:
  - mean
  - std
  - p95
  - p99
  - KS statistic
  - residual correlation or HSIC score

Negative controls:

- Semantic-only attack.
- Timing-only attack.
- Benign mode shift.
- Random frame corruption not synchronized with timing.

The main claim is strongest only if CIRCA-RT reacts primarily to coupled attacks while not exploding on benign mode shifts.

## Workstream E: Formal Timing And Audit Bound

Goal:

Add a small but concrete real-time analysis section.

Definitions:

```text
B      bucket capacity
r      bucket replenish rate per sample or per period
C_a    measured worst-case or high-percentile audit cost
C_m    measured monitor cost
N(I)   number of frames in interval I
A(I)   number of audits in interval I
T(I)   length of interval I
```

Token-bucket audit bound:

```text
A(I) <= floor((B + r * N(I)) / C_a)
```

If replenish is time-based:

```text
A(I) <= floor((B + r_time * T(I)) / C_a)
```

Added work bound:

```text
W_added(I) <= C_m * N(I) + C_a * A(I)
```

Sequential latency bound:

```text
L_t <= L_base_t + C_m + audit_t * C_a
```

Asynchronous audit server utilization:

```text
rho_audit <= lambda_audit * C_a
```

Required experiments to validate the bound:

- Show measured `A(I)` never exceeds the bound over sliding windows.
- Show p99 added latency is below the bound-derived operating limit.
- Show AlwaysAudit violates the latency or utilization target under at least one load condition where CIRCA-RT remains acceptable.

Paper positioning:

- The theorem does not need to prove detection recall.
- It needs to prove workload and latency control under selective audit.
- Detection quality is empirical; real-time feasibility is bounded.

## Workstream F: Baselines And Fairness

Main baselines:

- `ContextAwareConformal`
- `TranAD`
- `CATCH`
- `DCdetector`
- `ConditionalRFF-HSIC`
- `RFF-HSIC`
- `SemanticThreshold`
- `TimingThreshold`
- `RandomBudgetAudit`
- `AlwaysAudit`

Fairness requirements:

- Same traces.
- Same benign calibration split.
- Same feature availability.
- Same latency accounting.
- Same audit budget when using budget-normalized tables.
- Same online/offline distinction clearly reported.

Deep baseline rule:

- If official implementation is not used, call it `CATCH-style`, `DCdetector-style`, or `TranAD-style`.
- Do not write "exact reproduction" unless it is exact.
- Main table may include adapters, but the paper must be explicit.

Budget-normalized tables:

- Budgets: `0.02`, `0.05`, `0.10`, `0.15`, `0.20`.
- For each budget, force every score-based method to audit the same fraction.
- Report recall, false alarm, p99, p999, deadline miss, mean overhead, and power.

Pareto tables:

- Recall vs audit rate.
- Recall vs p99 overhead.
- Recall vs deadline miss.
- Recall vs mean power.

## Workstream G: Required Tables

Final paper tables:

```text
results_jetson_orin_perception/tables/jetson_latency_table.csv
results_jetson_orin_perception/tables/jetson_monitoring_main_table.csv
results_jetson_orin_perception/tables/jetson_scenario_table.csv
results_jetson_orin_perception/tables/jetson_budget_main_table.csv
results_jetson_orin_perception/tables/jetson_pareto_frontier_table.csv
results_jetson_orin_perception/tables/jetson_power_summary.csv
results_jetson_orin_perception/tables/semantic_marginal_diagnostics.csv
results_jetson_orin_perception/tables/audit_bound_validation.csv
results_jetson_orin_perception/tables/provider_status.csv
```

Minimum columns for monitoring tables:

```text
platform
runtime
model
scenario
seed
method
attack_recall
false_alarm_rate
detection_delay
audit_rate
mean_audit_cost_ms
monitor_cost_mean_ms
p99_latency_ms
p999_latency_ms
deadline_miss_ratio
latency_inflation_mean_ms
mean_power_w
peak_temp_c
active_provider
online
real_audit
semantic_source
```

## Workstream H: Required Figures

Main paper figures:

1. System diagram: ROS2 frame publisher, TensorRT perception, CIRCA-RT monitor, selective audit, logger.
2. Jetson p99/p999 latency under no monitor, CIRCA-RT, strongest competitor, and AlwaysAudit.
3. Recall vs audit rate Pareto frontier.
4. Recall vs p99 overhead Pareto frontier.
5. Token-bucket audit bound validation over time.

Appendix figures:

- Semantic/timing marginal diagnostics for coupled attack.
- Ablation of residualization, dependence score, and token bucket.
- Power and thermal traces.
- Representative score timeline.

## Workstream I: Artifact Package

RTSS 2026 supplementary materials allow up to 600 MB. Prepare a double-anonymous ZIP.

Artifact contents:

```text
README_ARTIFACT.md
configs/
src/
scripts/
artifact_small_data/
artifact_expected_outputs/
results_tables/
logs_sanitized/
demo_video.mp4
run_smoke_test.sh
run_table_reproduction.sh
```

Smoke test requirements:

- Runs without Jetson hardware using a tiny included trace.
- Regenerates one table and one figure.
- Validates schema.
- Validates that no main-result table reports TensorRT if provider status is invalid.

Jetson reproduction mode:

- Separate script for hardware reproduction.
- Clear expected runtime.
- Clear hardware requirements.
- Clear failure mode when TensorRT or `tegrastats` is unavailable.

## Emergency RTSS 2026 Plan

Use this only if targeting the 2026-05-26 deadline.

Minimum feasible upgrade:

1. Implement `perception_semantics.py` for real model-output semantic features.
2. Run Jetson `MobileNetV2` with one runtime, preferably TensorRT, otherwise ONNXRuntime.
3. Use `nominal`, `benign_mode_shift`, `semantic_corruption`, `gpu_interference`, and `coupled_semantic_timing_attack`.
4. Make heavy audit real by running a larger model or higher-resolution inference.
5. Add a short token-bucket bound section.
6. Generate only the core tables and figures.
7. Package a small artifact with scripts and a demo video.

Emergency acceptance gate:

- If real semantic signal and real audit cannot be completed, do not present the work as high-confidence RTSS full-paper material.
- If TensorRT fails but ONNXRuntime works on Jetson, submit only if real semantic signal and real audit are strong.
- If Jetson cannot run online, submit as a lower-confidence full paper or redirect to a brief/demo-style track.

## Full 80-Quality Plan

Use this for RTSS 2027 or a later deadline if the 2026 schedule is too tight.

Phase 1: Real semantic trace generation

- Implement perception semantic feature extraction.
- Build video replay to unified schema.
- Validate that semantic residuals are not label-derived.

Phase 2: Online Jetson pipeline

- Implement live ROS2/TensorRT pipeline.
- Implement online CIRCA-RT monitor.
- Implement real heavy audit.
- Log all timing, audit, and power fields.

Phase 3: Scenario and attack validation

- Run all main scenarios.
- Produce marginal diagnostics.
- Add negative controls.

Phase 4: Baselines

- Run all statistical baselines.
- Run recent deep baselines.
- Add budget-normalized evaluation.
- Add official deep implementation for at least one key baseline if feasible.

Phase 5: Analysis and artifact

- Write token-bucket theorem and proof.
- Generate final tables and figures.
- Build anonymous artifact ZIP.
- Record demo video.

Phase 6: Paper writing

- Lead with real-time requirement and bounded audit.
- Present detection as budget-aware evidence.
- Keep recall dominance claims out of the abstract.
- Put older synthetic experiments in appendix only.

## Go/No-Go Checklist

Go for high-confidence RTSS full paper if all are true:

- Main Jetson results use real perception-derived semantic features.
- Main Jetson results are online.
- Heavy audit is real measured computation.
- TensorRT or ONNXRuntime provider status is valid and documented.
- CIRCA-RT is on the Pareto frontier in the main coupled scenarios.
- CIRCA-RT has substantially lower audit and p99/p999 overhead than AlwaysAudit.
- Strong competitors are reported honestly.
- Token-bucket bound is included and empirically validated.
- Artifact reproduces a small result and includes sanitized logs.

No-Go for high-confidence RTSS full paper if any are true:

- Main semantic residuals are label-derived.
- Audit cost is simulated in the main result.
- Main result is offline-only.
- TensorRT fallback is reported as TensorRT.
- ContextAwareConformal or TranAD dominates CIRCA-RT on recall, audit, and latency simultaneously.
- Benign mode shifts cause uncontrolled false alarms.
- There is no bound or explicit real-time analysis.

## Final Development Priority

The single most important change is:

> Replace synthetic semantic residuals with real perception-derived semantic residuals and make heavy audit a real measured operation.

The second most important change is:

> Move the main Jetson result from offline trace scoring to an online ROS2/TensorRT pipeline.

The third most important change is:

> Add a compact token-bucket real-time bound and validate it on the Jetson traces.

Without these three changes, the paper can still be competitive, but it should not be treated as an `80%+` full-paper candidate.
