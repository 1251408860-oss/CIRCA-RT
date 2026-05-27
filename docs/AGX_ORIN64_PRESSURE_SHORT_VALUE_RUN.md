# AGX Orin 64GB Short-Value Pressure Test

Updated: 2026-05-25

This is the recommended cost-effective AGX pressure test. It is much shorter
than the full pressure sweep, but it still gives a clean RTSS-style robustness
story:

- Real AGX Orin 64GB device.
- Real-frame datasets: AV2 and nuImages.
- Deadline tightening: 20, 33.333, and 50 ms.
- Stress contrast: idle vs high GPU contention.
- CIRCA-RT-Slack compared with high-audit and statistical baselines.
- Tegrastats telemetry for each run.

Expected runtime: about 1.5 to 3 hours on the AGX device, depending on storage
speed, thermals, and whether the machine is otherwise busy.

## Run Command

On the AGX machine:

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_max

PYTHON_BIN=python3 \
bash scripts/run_agx_orin64_pressure_short_value.sh
```

The wrapper expands to:

```bash
PYTHON_BIN=python3 \
PRESSURE_PROFILE=paper \
AGX_DATASETS="av2 nuimages" \
DEADLINES_MS="20 33.333 50" \
SEEDS="7" \
PRESSURE_LEVELS="idle:0:0:0 high:1536:3:24" \
MODELS="mobilenet_v2" \
N_AV2=300 \
N_NUIMAGES=217 \
FRAME_LIMIT_AV2=600 \
FRAME_LIMIT_NUIMAGES=217 \
RUN_BACKEND_MATRIX=0 \
RUN_DERIVED_ANALYSES=1 \
bash scripts/run_agx_orin64_pressure_sweep.sh
```

## What It Produces

Default output directory:

```text
results_agx_orin64_pressure_short_value_<timestamp>/
```

Important files:

```text
PRESSURE_SWEEP_REPORT.md
tables/pressure_manifest.json
tables/pressure_realframe_main_all.csv
tables/pressure_audit_bound_validation_all.csv
tables/pressure_tegrastats_summary_all.csv
tables/pressure_deadline_breakpoints.csv
tables/pressure_circa_slack_delta.csv
tables/pressure_failed_commands.csv
figures/deadline_miss_ratio_*.png
figures/p99_latency_ms_*.png
figures/attack_recall_*.png
figures/audit_rate_*.png
```

## Success Criteria

Use the result in the paper only if:

- `tables/pressure_failed_commands.csv` reports no failed level.
- `tables/pressure_audit_bound_validation_all.csv` has zero failed rows.
- Both AV2 and nuImages are present.
- Both `idle` and `high` pressure levels are present.
- `CIRCA-RT-Slack` appears in `pressure_realframe_main_all.csv`.
- Tegrastats summaries exist for both pressure levels.

## How To Package Results

After the run finishes:

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_max

tar -czf agx_orin64_pressure_short_value_$(date +%Y%m%d_%H%M%S).tar.gz \
  results_agx_orin64_pressure_short_value_*

sha256sum agx_orin64_pressure_short_value_*.tar.gz
```

Send back the `.tar.gz` and the SHA256 line.

## Optional 3-Level Variant

If the AGX machine is available for about 3 to 5 hours, add a medium stress
level:

```bash
PYTHON_BIN=python3 \
PRESSURE_LEVELS="idle:0:0:0 medium:1024:2:12 high:1536:3:24" \
bash scripts/run_agx_orin64_pressure_short_value.sh
```

This gives a better trend line, but the two-level version is the best
time-to-value choice.

## Paper Wording

Safe wording:

> We run a compact AGX Orin pressure test that contrasts idle and high GPU
> contention over AV2 and nuImages while sweeping 20, 33.333, and 50 ms
> deadlines. The experiment records tegrastats telemetry and compares
> CIRCA-RT-Slack against high-audit and statistical baselines.

Do not claim:

- Exhaustive pressure characterization.
- Multi-backbone pressure results.
- TensorRT pressure results, because this short pressure run disables the
  backend matrix by default.
