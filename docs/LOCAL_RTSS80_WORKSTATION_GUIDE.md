# Local RTSS80 Workstation Guide

Updated: 2026-05-19

This guide describes what can be done on a local workstation before Jetson is available. The goal is to improve paper quality in ways that directly address the main reviewer objections: synthetic semantics, offline-only monitoring, simulated audit cost, and lack of a real-time bound.

## What Local Work Can Prove

Use the local workstation to prove the following:

- CIRCA-RT can run on real image or video frames, not only on synthetic CSV traces.
- Semantic residuals can be derived from image features, not from `label`.
- Heavy audit can be measured as real computation, not as a fixed constant.
- The token-bucket audit policy has an explicit workload bound.
- Baselines and ablations still run under the same trace schema.

Do not use the local workstation to claim:

- Jetson Orin validation.
- TensorRT runtime validation.
- Physical TSN support.
- Final hardware tail-latency claims.

## Local High-Value Tasks

### 1. Real Frame Semantic Traces

Run the perception trace generator on a directory of real images or on the built-in demo frames for smoke testing.

Recommended command:

```powershell
python scripts/run_local_perception_semantics.py --frame-dir <your_frame_dir> --out-dir results_local_perception_semantics --n 240 --seeds 7 8 9 --include-deep
```

If no frame directory is ready, run the demo mode first:

```powershell
python scripts/run_local_perception_semantics.py --out-dir results_local_perception_semantics --n 120
```

What this produces:

- `results_local_perception_semantics/traces/`
- `results_local_perception_semantics/raw/`
- `results_local_perception_semantics/tables/`
- `results_local_perception_semantics/figures/`

What to check:

- `semantic_residual` is computed from image-feature statistics.
- `label` is not used to build semantic residuals.
- `audit_cost_ms` is measured from a real audit function on audited frames.

### 2. Audit Bound Validation

Validate that the token-bucket policy respects its windowed audit-workload bound.

Recommended command:

```powershell
python scripts/validate_audit_bound.py --input-root results_local_perception_semantics\raw --out results_local_perception_semantics\tables\audit_bound_validation.csv --methods CIRCA-RT
```

Useful stricter check:

```powershell
python scripts/validate_audit_bound.py --input-root results_local_perception_semantics\raw --out results_local_perception_semantics\tables\audit_bound_validation.csv --methods all
```

What to check:

- `audit_count_pass` should be `True` for CIRCA-RT across the tested windows.
- `sample_bound_pass` should be `True` for the measured local audit path.
- `work_bound_pass` should not fail on the CIRCA-RT rows.

### 3. Local Ablations

Use the local frame-derived traces to support ablation claims.

Recommended checks:

- `CIRCA-RT` versus `ContextAwareConformal`.
- `CIRCA-RT` versus `RFF-HSIC` and `ConditionalRFF-HSIC`.
- `CIRCA-RT` versus `AlwaysAudit`.
- `nominal` versus `mode_shift`.
- `semantic_corruption` versus `timing_interference`.
- `coupled_semantic_timing_attack` versus the two single-factor controls.

### 4. Artifact Packaging

Prepare a compact reproducibility package from the local outputs:

- one smoke-test frame set,
- one generated summary table,
- one audit-bound validation table,
- one score timeline figure,
- one short README that explains what is real and what is not.

## Recommended Local Output Contract

Keep these files if the local run is useful:

```text
results_local_perception_semantics/
  traces/
  raw/
  summaries/
  tables/
  figures/
```

Important tables:

```text
results_local_perception_semantics/tables/local_perception_main_table.csv
results_local_perception_semantics/tables/local_perception_scenario_table.csv
results_local_perception_semantics/tables/local_perception_main_table_ci.csv
results_local_perception_semantics/tables/semantic_timing_diagnostics.csv
results_local_perception_semantics/tables/audit_bound_validation.csv
```

## Interpretation Rules

- If the local demo frames are used, treat the output as a pipeline smoke test only.
- If real image or video frames are used, the output is strong evidence that semantic residuals and audit cost do not rely on the synthetic CSV-only path.
- This local work strengthens the paper, but it does not replace Jetson validation.

## Best Local Priority Order

1. Run the perception semantic trace generator on real frames.
2. Validate audit bounds on the CIRCA-RT rows.
3. Regenerate the local summary tables and figures.
4. Use those outputs to tighten the paper's method and evaluation wording.

