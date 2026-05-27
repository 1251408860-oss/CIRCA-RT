# CIRCA-RT

Code-first GitHub upload package for the CIRCA-RT local experiment stack.

This package keeps the reproducible Python code, configuration files, and run
documentation. It intentionally leaves out large raw datasets, bulky scored
traces, GPU/AGX result archives, third-party baseline checkouts, caches, and
paper build products.

## Contents

- `src/circa_rt/`: reusable CIRCA-RT package code.
- `scripts/`: local, AutoDL, ROS2, Jetson/AGX, and figure/table runners.
- `configs/`: experiment and schema configuration.
- `docs/`: runbooks and experiment notes.
- `requirements.txt`: lightweight local CPU dependencies.
- `requirements_agx_orin64_runtime.txt`: non-NVIDIA Python utilities for AGX runs.

## Quick Start

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_phase0_4_pipeline.py
```

The local pipeline writes generated traces under `traces/` and summary tables
and figures under `results/`. Those generated outputs are ignored by Git by
default.

## Common Commands

Run the split main experiment:

```bash
python scripts/run_phase0_4_split_pipeline.py
python scripts/run_sensitivity.py
```

Run the Phase 5 periodic-pipeline experiment:

```bash
python scripts/run_phase5_periodic_pipeline.py
```

Run the local syntax check:

```bash
python -m compileall src scripts
```

## Notes For Larger Experiments

GPU, ROS2, real-frame replay, and AGX Orin scripts are included, but their raw
datasets, device logs, result archives, model caches, and third-party baselines
are not part of this small upload package. See `docs/` for the corresponding
runbooks before running those workflows.
