# AGX Orin 64GB Pressure Test Runbook

Updated: 2026-05-25

Purpose: run a real-device robustness experiment on Jetson AGX Orin 64GB that
answers a real-time systems question:

> Under increasing GPU contention and tightening deadlines, does CIRCA-RT keep
> deadline misses and audit demand controlled while preserving useful attack
> recall?

This is not a burn-in test. It is a reproducible RTSS-style contention and
deadline robustness test.

## Rationale

The pressure test follows NVIDIA Jetson tooling conventions:

- `nvpmodel` records and optionally sets power mode.
- `jetson_clocks --show` verifies CPU/GPU/EMC frequency state.
- `tegrastats --interval <ms> --logfile <file>` records memory, processor,
  frequency, and temperature telemetry during each run.

Official references:

- NVIDIA Jetson tegrastats utility:
  https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html
- NVIDIA Jetson nvpmodel validation:
  https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/TestPlanValidation.html

## What This Adds

Your current AGX evidence proves that the method runs on real AGX hardware. This
pressure sweep adds evidence that the method remains useful under:

- GPU contention.
- Coupled semantic/timing attacks.
- Tight deadlines: 16.667, 20, 25, 33.333, and 50 ms.
- Multiple real-frame datasets.
- Multiple random seeds.
- Tegrastats-verified device telemetry.

The strongest expected paper plot is:

- x-axis: deadline or stress level.
- y-axis: deadline miss ratio, p99 latency, recall, and audit rate.
- lines: CIRCA-RT-Slack, CIRCA-RT, ConditionalRFF-HSIC,
  ContextAwareConformal, AlwaysAudit.

## Script

Main script:

```bash
bash scripts/run_agx_orin64_pressure_sweep.sh
```

Summary script:

```bash
python3 scripts/summarize_agx_orin64_pressure_sweep.py \
  --input-root results_agx_orin64_pressure_paper_<timestamp>
```

The wrapper runs `scripts/run_agx_orin64_comprehensive_experiment.sh` at each
stress level, then aggregates the generated reports.

## Profiles

### Smoke

Use this first. It checks whether the pressure wrapper, telemetry, and summary
work.

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_max

PYTHON_BIN=python3 \
PRESSURE_PROFILE=smoke \
bash scripts/run_agx_orin64_pressure_sweep.sh
```

Smoke defaults:

- Dataset: AV2.
- Deadlines: 25 and 33.333 ms.
- Model: MobileNetV2.
- Seed: 7.
- Pressure levels: idle and medium.
- Backend matrix: off.

### Paper

This is the recommended RTSS-quality pressure test.

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_max

PYTHON_BIN=python3 \
PRESSURE_PROFILE=paper \
BASE_RESULT_ROOT=results_agx_orin64_pressure_paper_202605xx \
bash scripts/run_agx_orin64_pressure_sweep.sh
```

Paper defaults:

- Datasets: AV2, DROID, nuImages.
- Deadlines: 16.667, 20, 25, 33.333, 50 ms.
- Model: MobileNetV2.
- Seeds: 7, 8, 9.
- Samples: AV2 300, DROID 300, nuImages 217.
- Pressure levels:
  - `idle`: `GPU_STRESS_SIZE=0`, `GPU_STRESS_REPEATS=0`, coupled extra 0.
  - `mild`: `512`, repeats 1, coupled extra 4.
  - `medium`: `1024`, repeats 2, coupled extra 12.
  - `high`: `1536`, repeats 3, coupled extra 24.
- Backend matrix: off, because backend smoke has already been returned
  separately and repeated backend collection at every stress level is wasteful.

### Max

Use only if the AGX machine is available for a long run.

```bash
PYTHON_BIN=python3 \
PRESSURE_PROFILE=max \
BASE_RESULT_ROOT=results_agx_orin64_pressure_max_202605xx \
bash scripts/run_agx_orin64_pressure_sweep.sh
```

Max defaults:

- Datasets: AV2, DROID, nuImages.
- Deadlines: 16.667, 20, 25, 33.333, 40, 50, 100 ms.
- Models: SqueezeNet1.1, MobileNetV2, ResNet18.
- Seeds: 7, 8, 9, 10, 11.
- Pressure levels: idle, mild, medium, high, saturation.

## Custom High-Value Variant

If time is limited but you want strong evidence, run this:

```bash
PYTHON_BIN=python3 \
PRESSURE_PROFILE=paper \
BASE_RESULT_ROOT=results_agx_orin64_pressure_compact_202605xx \
AGX_DATASETS="av2 nuimages" \
DEADLINES_MS="20 25 33.333 50" \
SEEDS="7 8 9" \
PRESSURE_LEVELS="idle:0:0:0 medium:1024:2:12 high:1536:3:24" \
bash scripts/run_agx_orin64_pressure_sweep.sh
```

This compact version still answers the core RTSS question with two real-frame
datasets, multiple deadlines, multiple seeds, and three stress levels.

## Short-Value One-Command Variant

For the best time-to-value tradeoff, use the standalone wrapper:

```bash
PYTHON_BIN=python3 \
bash scripts/run_agx_orin64_pressure_short_value.sh
```

It runs AV2 and nuImages, deadlines 20/33.333/50 ms, seed 7, and two pressure
levels: idle and high. See:

```text
docs/AGX_ORIN64_PRESSURE_SHORT_VALUE_RUN.md
```

## Output

The wrapper creates:

```text
results_agx_orin64_pressure_<profile>_<timestamp>/
  pressure_plan.txt
  PRESSURE_SWEEP_REPORT.md
  logs/
  tables/
    pressure_manifest.json
    pressure_realframe_main_all.csv
    pressure_audit_bound_validation_all.csv
    pressure_tegrastats_summary_all.csv
    pressure_deadline_breakpoints.csv
    pressure_circa_slack_delta.csv
    pressure_failed_commands.csv
  figures/
    deadline_miss_ratio_<stress>.png
    p99_latency_ms_<stress>.png
    attack_recall_<stress>.png
    audit_rate_<stress>.png
  level_0_idle/
  level_1_mild/
  level_2_medium/
  level_3_high/
```

Each `level_*` directory is a normal comprehensive AGX result root with its own:

- `env/`
- `logs/`
- `realframes/`
- `tables/`
- `figures/`
- `AGX_ORIN64_COMPREHENSIVE_REPORT.md`

## Success Criteria

Use the pressure result in the paper only if:

- `pressure_failed_commands.csv` has no failures.
- `pressure_audit_bound_validation_all.csv` has zero failed rows.
- Every stress level has tegrastats logs.
- CIRCA-RT-Slack appears in `pressure_realframe_main_all.csv`.
- At least AV2 and nuImages complete.
- The report contains deadline breakpoints and CIRCA-RT-Slack deltas.

## Paper Claim

Safe claim:

> We further stress-test the real AGX Orin deployment by sweeping GPU contention
> intensity and perception deadlines. The run records tegrastats telemetry and
> shows how CIRCA-RT's slack-admissible audit policy changes deadline misses,
> audit rate, and p99 latency relative to high-audit baselines.

Do not claim:

- Hard real-time guarantees under arbitrary load.
- Thermal/power superiority unless VDD power fields are present and parsed.
- TensorRT pressure results unless backend pressure runs are explicitly enabled
  and provider tables prove TensorRT execution.

## What To Send Back

After completion:

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_max
tar -czf agx_orin64_pressure_results_$(date +%Y%m%d_%H%M%S).tar.gz \
  results_agx_orin64_pressure_* \
  jetson_env_check_first.log 2>/dev/null || true
sha256sum agx_orin64_pressure_results_*.tar.gz
```

Send back the `.tar.gz` and the SHA256 line.
