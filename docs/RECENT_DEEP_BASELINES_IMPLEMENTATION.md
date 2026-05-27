# Recent Deep Baselines Implementation

Updated: 2026-05-19

This project now includes recent deep time-series baseline adapters for the CIRCA-RT traces.

Implemented methods:

- `CATCH`: channel/frequency reconstruction baseline inspired by frequency-patching TSAD.
- `DCdetector`: dual-view time/frequency contrastive baseline inspired by contrastive MTSAD.
- `TranAD`: transformer reconstruction baseline inspired by transformer TSAD.

These are deployment-compatible adapters, not verbatim copies of external repositories. They use the same CIRCA-RT trace schema, benign calibration protocol, online windows, audit-cost model, and latency accounting.

## Files Added

- `src/circa_rt/deep_baselines.py`
- `scripts/run_recent_deep_baselines.py`

Related registry:

- `src/circa_rt/baselines.py` now exposes `RECENT_DEEP_BASELINE_FUNCS`.

## Design

Each method uses the same input stream:

```text
[semantic_residual,
 timing_residual_ms,
 context_cpu_util,
 context_gpu_util,
 context_net_delay_ms,
 context_jitter_ms,
 context_queue_depth,
 context_message_age_ms]
```

The detector sees only an online history window ending at the current sample. No future samples are used.

Training/calibration:

- Fit feature normalization on benign calibration samples.
- Build online windows grouped by `run_id`.
- Train only on benign calibration windows.
- Set the alarm threshold from the benign calibration score quantile.

Evaluation:

- Score every test sample.
- Alarm/audit when the score exceeds the threshold.
- Add measured detector inference cost to `monitor_cost_ms`.
- Add audit cost when `audit=1`.
- Report default threshold and budget-normalized results.

## No-GPU Check

When PyTorch is unavailable, the code falls back to sklearn implementations:

- `CATCH`: PCA reconstruction on frequency features.
- `TranAD`: PCA reconstruction on time-window features.
- `DCdetector`: dual PCA time/frequency alignment score.

This fallback is for code validation and CPU-only checking. Use AutoDL GPU for final numbers.

Dry-run command:

```bash
python scripts/run_recent_deep_baselines.py \
  --inputs autodl_full=results_autodl_final \
  --out-dir results_recent_deep_baselines_dryrun \
  --device cpu \
  --backend sklearn \
  --epochs 1 \
  --max-groups 1 \
  --max-train-windows 512 \
  --batch-size 64
```

## Final AutoDL Command After GPU Is Open

Run this from `/root/rtss_exp_begin`:

```bash
/root/miniconda3/bin/python scripts/run_recent_deep_baselines.py \
  --inputs autodl_full=results_autodl_final autodl_ros2_twoprocess=results_autodl_ros2_twoprocess_final \
  --out-dir results_recent_deep_baselines_final \
  --device cuda \
  --backend torch \
  --epochs 30 \
  --batch-size 128 \
  --max-train-windows 4096 \
  --budgets 0.02 0.05 0.10 0.15
```

If CUDA is not visible, use:

```bash
/root/miniconda3/bin/python scripts/run_recent_deep_baselines.py \
  --inputs autodl_full=results_autodl_final autodl_ros2_twoprocess=results_autodl_ros2_twoprocess_final \
  --out-dir results_recent_deep_baselines_cpu \
  --device cpu \
  --backend sklearn \
  --epochs 30 \
  --batch-size 128 \
  --max-train-windows 4096 \
  --budgets 0.02 0.05 0.10 0.15
```

## Expected Outputs

Default-threshold results:

- `results_recent_deep_baselines_final/tables/recent_deep_main_table.csv`
- `results_recent_deep_baselines_final/tables/recent_deep_main_table_ci.csv`
- `results_recent_deep_baselines_final/tables/recent_deep_scenario_table.csv`
- `results_recent_deep_baselines_final/tables/recent_deep_summary_all.csv`

Budget-normalized results:

- `results_recent_deep_baselines_final/tables/recent_deep_budget_main_table.csv`
- `results_recent_deep_baselines_final/tables/recent_deep_budget_main_table_ci.csv`
- `results_recent_deep_baselines_final/tables/recent_deep_budget_top5_table.csv`
- `results_recent_deep_baselines_final/tables/recent_deep_budget_summary_all.csv`

Raw scored traces:

- `results_recent_deep_baselines_final/raw/.../scored.csv`

## Paper Usage

Use these names carefully:

- `CATCH-style`
- `DCdetector-style`
- `TranAD-style`

Recommended wording:

> For recent deep time-series anomaly detectors whose original artifacts do not directly target online real-time audit decisions, we implement deployment-compatible adapters using the same benign calibration, online-window, audit-budget, and latency-accounting protocol.

Avoid:

> We exactly reproduce the official CATCH/DCdetector/TranAD implementations.

## How To Combine With Existing Results

Use existing AutoDL results as the main system evidence:

- `results_autodl_final/`
- `results_autodl_ros2_twoprocess_final/`
- `results_budget_normalized_final/`

Use this new directory as the recent-deep-baseline add-on:

- `results_recent_deep_baselines_final/`

The paper's strongest fair comparison should combine:

- CIRCA-RT
- ContextAwareConformal
- ConditionalRFF-HSIC
- LearnedFourierIndependence
- CATCH-style
- DCdetector-style
- TranAD-style
- RandomBudgetAudit
- PeriodicAudit
- AlwaysAudit
