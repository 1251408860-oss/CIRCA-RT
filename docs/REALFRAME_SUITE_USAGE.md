# Real-Frame Suite Usage

## One Dataset

### DROID robot camera subset

```bash
cd /root/rtss_exp_begin
bash scripts/run_droid_realframes_experiment.sh
```

This downloads a small public `lerobot/droid_100` three-camera subset when
`data/frames_droid/manifest.csv` is absent, normalizes it into
`data/frames_droid` with per-camera frame balancing, and then runs the
real-frame experiment.

### nuImages mini

```bash
cd /root/rtss_exp_begin
bash scripts/run_nuimages_realframes_experiment.sh
```

Default outputs:

- `results_autodl_perception_semantics_realframes_nuimages`
- `results_recent_deep_baselines_realframes_nuimages`

## Multi-Dataset Suite

```bash
cd /root/rtss_exp_begin
DATASETS="av2 nuimages" bash scripts/run_realframe_dataset_suite.sh
```

After preparing DROID, run the three-source suite with:

```bash
cd /root/rtss_exp_begin
DATASETS="av2 nuimages droid" bash scripts/run_realframe_dataset_suite.sh
```

The suite writes a dataset-quality JSON report into each result directory:

- `tables/real_dataset_quality.json`

## Notes

- `nuImages mini` currently has `217` normalized frames across `67` sequences,
  so it should be treated as a secondary real-data validation set, not the only
  main real-frame benchmark.
- `AV2` remains the stronger main real-frame autonomous-driving dataset in the
  current environment.
- `DROID` is a robot-camera validation source. It should be reported separately
  from driving datasets because the semantics and camera motion distribution are
  different.
