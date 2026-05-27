# CIRCA-RT Split Protocol and Results

更新时间：2026-05-16

这是当前主实验版本，和早期 `same-trace calibration` 版本区分开。

## Protocol

- Calibration trace: `nominal + mode_shift`
- Test traces: `nominal`, `mode_shift`, `stealthy_coupled_attack`, `timing_only_attack`, `semantic_only_attack`, `mixed_attack`
- Split unit: one fold = one calibration seed + one test seed
- Calibration only uses benign data to set thresholds / models
- Main outputs are stored under `results_split/`

Key scripts:

```powershell
python scripts\run_phase0_4_split_pipeline.py
python scripts\run_sensitivity.py
```

## What Was Added

- Strict calibration/test separation for CIRCA-RT
- Split versions of the main statistical baselines
- Bootstrap confidence intervals for summary tables
- Sensitivity sweeps for `window_size`, `feature_dim`, and `replenish_rate`
- Scenario table now includes `audit_rate`

## Main Findings

On `stealthy_coupled_attack`:

- `CIRCA-RT`: recall `0.8917`, false alarm `0.1063`, delay `8.07`, audit rate `0.0357`, p99 latency `38.30 ms`
- `RFF-HSIC`: recall `0.7850`, false alarm `0.0527`, delay `14.67`, audit rate `0.1992`, p99 latency `40.37 ms`
- `ContextAwareConformal`: recall `0.5417`, false alarm `0.0440`, delay `1.53`, audit rate `0.1435`, p99 latency `40.12 ms`
- `LearnedFourierIndependence`: recall `0.9242`, but false alarm `0.7673` and audit rate `0.7983`
- `AlwaysAudit`: recall `1.0`, but false alarm `1.0` and p99 latency `41.23 ms`

On benign traces (`nominal` and `mode_shift`):

- `CIRCA-RT` false alarm is about `0.05`
- `RFF-HSIC` and `SemanticThreshold` are lower, but they miss more coupled attacks
- `AlwaysAudit` is high cost and not a realistic baseline for the main claim

## Sensitivity

The current sensitivity sweep shows:

- `window_size=16` gives the best recall and shortest delay among the tested values
- `feature_dim=32` is a good default tradeoff
- `replenish_rate` mainly controls audit rate and p99 inflation, not raw recall

## Interpretation

The paper-friendly claim is not "best AUROC". The better claim is:

> Under a benign-only calibration protocol, CIRCA-RT detects coupled semantic-timing anomalies with controlled false alarm and low latency overhead, while selective auditing avoids the cost of always-audit policies.

## Next Phase Implications

Phase 5 should now focus on real or semi-real timing traces, not more synthetic variants:

- ROS2/DDS timing trace if the Linux environment is stable
- Python periodic pipeline if ROS2 setup blocks progress
- Same CSV schema and same split protocol as Phase 0-4

Phase 6 should focus on GPU inference integration:

- PyTorch / ONNXRuntime / TensorRT latency
- monitor-only, selective-audit, and always-audit overhead
- optional Holoscan-style streaming baseline if the environment supports it

Phase 7 should be the main external-system evidence:

- Jetson Orin batch-1 inference
- p99/p999 latency
- power mode and frequency traces
- selective audit vs always audit

Phase 8-9 should treat `results_split/` as the main source. The older `results/` directory is now a smoke-test or appendix source only.
