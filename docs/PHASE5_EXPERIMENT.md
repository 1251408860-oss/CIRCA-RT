# Phase 5 Periodic Pipeline Experiment

更新时间：2026-05-16

Phase 5 当前采用本机 Python periodic pipeline，作为 ROS2/DDS 实验前的可复现 fallback。

## Goal

目标不是声称已经完成真实 ROS2/DDS 或 TSN 硬件实验，而是拿到 middleware-style timing traces：

- periodic publisher/subscriber
- message age
- callback latency proxy
- queue depth
- burst delay
- jitter
- stale replay
- CPU load interference

所有 trace 都转换成 CIRCA-RT 统一 CSV schema。

## Command

```powershell
python scripts\run_phase5_periodic_pipeline.py
```

## Outputs

- `results_phase5/traces/`
- `results_phase5/summaries/summary_all.csv`
- `results_phase5/tables/main_table.csv`
- `results_phase5/tables/main_table_ci.csv`
- `results_phase5/tables/scenario_table.csv`
- `results_phase5/tables/scenario_table_ci.csv`
- `results_phase5/tables/attack_only_table.csv`
- `results_phase5/figures/*.png`

## Protocol

- Calibration traces: `nominal + mode_shift`
- Test traces:
  - `nominal`
  - `mode_shift`
  - `burst_delay_attack`
  - `jitter_attack`
  - `stale_replay_attack`
  - `cpu_load_interference`
  - `coupled_timing_semantic_attack`
- Protocol: benign calibration / held-out test
- Platform label: `local_periodic_pipeline`

## Key Result

On `coupled_timing_semantic_attack`:

- `CIRCA-RT`: recall `0.9015`, false alarm `0.1398`, audit rate `0.0363`, p99 latency `40.69 ms`
- `RFF-HSIC`: recall `0.8993`, false alarm `0.0813`, audit rate `0.2653`, p99 latency `42.60 ms`
- `ConditionalRFF-HSIC`: recall `0.7867`, false alarm `0.0641`, audit rate `0.2267`, p99 latency `42.52 ms`
- `ContextAwareConformal`: recall `0.6363`, false alarm `0.0538`, audit rate `0.1848`, p99 latency `42.60 ms`
- `AlwaysAudit`: recall `1.0`, false alarm `1.0`, audit rate `1.0`, p99 latency `43.00 ms`

On benign traces:

- `CIRCA-RT` false alarm is about `0.061-0.063`
- `CIRCA-RT` audit rate is about `0.005-0.006`
- `AlwaysAudit` remains `1.0` audit rate and is only an upper-bound reference

## Interpretation

Phase 5 supports this claim:

> CIRCA-RT preserves high coupled-anomaly detection on middleware-style timing traces while using much less audit budget and lower tail latency than always-audit or non-selective dependence monitors.

Phase 5 does not support this stronger claim:

> CIRCA-RT is the best detector for every timing-only or load-interference perturbation.

That stronger claim should not be made. Timing-only perturbations are often better handled by timing-specific baselines.

## Next Step

The next strongest experiment is Phase 7 Jetson Orin:

- batch-1 edge inference latency
- perception only
- perception + monitor
- perception + selective audit
- perception + always audit
- p99/p999 and power mode table

Phase 6 AutoDL can be used before Jetson only to debug TensorRT scripts and server GPU baseline.
