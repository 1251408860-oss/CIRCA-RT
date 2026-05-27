# Real-Frame Dataset and Recent-Baseline Protocol

This project should use CIFAR-10 only as a pipeline smoke test. The RTSS-facing
evidence should be based on ordered real camera frames from robotics,
autonomous-driving, or deployed-camera sequences.

## Dataset Priority

Use at least two of the following families, with one held out from all tuning.

1. Waymo Open Dataset, especially the 2025 End-to-End Driving Dataset and
   2024 modular Perception v2 data. This is the strongest autonomous-driving
   choice, but access/download may require the official Waymo process and cloud
   storage tooling. Official entry points: https://waymo.com/open/download and
   https://waymo.com/intl/es/open/data/e2e/.
2. Argoverse 2 Sensor Dataset. This is the most convenient autonomous-driving
   source for this code path because camera images are ordinary files once the
   dataset is downloaded. Official entry points:
   https://www.argoverse.org/av2.html and
   https://argoverse.github.io/user-guide/datasets/sensor.html.
3. BDD100K. This is a useful camera-only driving source for broad weather,
   time-of-day, and scene diversity. Official/project entry points:
   https://bdd-data.berkeley.edu/ and
   https://bair.berkeley.edu/blog/2018/05/30/bdd/.
4. DROID / Open X-Embodiment / BridgeData V2. Use these for robot camera
   sequences to support a robotics claim instead of only a driving claim.
   Official/project entry points: https://droid-dataset.github.io/,
   https://robotics-transformer-x.github.io/, and
   https://rail-berkeley.github.io/bridgedata/.
5. A local USB/IP camera or robot log. This is valuable for external validity
   if the capture metadata is preserved and the exact preprocessing command is
   logged.

## Frame Preparation

The experiment runner consumes a flat or nested image tree through
`--frame-dir data/frames`. Normalize downloaded datasets with:

```bash
cd /root/rtss_exp_begin
/root/miniconda3/bin/python scripts/prepare_real_frame_dataset.py \
  --input-root /path/to/downloaded/dataset \
  --out-dir data/frames \
  --mode symlink \
  --stride 3 \
  --limit 20000 \
  --min-width 256 \
  --min-height 256
```

For video-only camera logs, add `--include-videos`. This requires `ffmpeg`.

The script writes `data/frames/manifest.csv` with source path, normalized path,
dataset hint, sequence hint, size, frame index, and SHA-1. Keep that file with
the paper artifact.

For a directly reproducible public driving-frame subset, download Argoverse 2
Sensor camera frames from the official public S3 bucket:

```bash
cd /root/rtss_exp_begin
/root/miniconda3/bin/python scripts/download_av2_sensor_frames.py \
  --split test \
  --cameras ring_front_center ring_front_left ring_front_right \
  --max-frames 3000 \
  --max-logs 12 \
  --max-frames-per-log-camera 120 \
  --out-dir data/raw/av2_sensor_subset

/root/miniconda3/bin/python scripts/prepare_real_frame_dataset.py \
  --input-root data/raw/av2_sensor_subset \
  --out-dir data/frames \
  --mode symlink \
  --limit 3000 \
  --min-width 256 \
  --min-height 256 \
  --dataset-hint argoverse2_sensor
```

## Main Real-Frame Run

After `data/frames` exists:

```bash
cd /root/rtss_exp_begin
bash scripts/run_realframes_perception_experiment.sh
```

The script runs:

1. CIRCA-RT and lightweight non-deep baselines on the real-frame traces.
2. Token-bucket audit-bound validation.
3. Recent deep time-series baselines on the same traces.

Environment overrides:

```bash
FRAME_DIR=/root/rtss_exp_begin/data/frames \
OUT_DIR=/root/rtss_exp_begin/results_autodl_perception_semantics_realframes_av2 \
DEEP_OUT_DIR=/root/rtss_exp_begin/results_recent_deep_baselines_realframes_av2 \
N=1800 \
SEEDS="7 8 9 10 11" \
bash scripts/run_realframes_perception_experiment.sh
```

## Recent Baselines

The current unified-protocol baselines are:

- CATCH adapter: frequency-domain reconstruction, matching the channel-aware
  motivation but not claiming to be the official CATCH implementation.
- DCdetector adapter: dual time/frequency representation consistency.
- TranAD adapter: transformer reconstruction.
- MOMENT adapter: masked patch reconstruction inspired by time-series
  foundation-model pretraining.
- TimeMixer adapter: multiscale temporal reconstruction.
- ModernTCN adapter: temporal convolution reconstruction.

For a paper submission, run the official repositories where possible and report
them separately as `official` baselines. Keep these local adapters in the
artifact because they use the exact same calibration/test split and timing-cost
accounting as CIRCA-RT.

Official/research links for the baseline family:

- CATCH: https://github.com/decisionintelligence/CATCH and
  https://openreview.net/forum?id=m08aK3xxdJ&noteId=vzZHRUi6eW
- MOMENT: https://github.com/moment-timeseries-foundation-model/moment
- TimeMixer: https://github.com/kwuking/TimeMixer
- ModernTCN: https://github.com/luodhhh/ModernTCN

## Acceptance Criteria Before Claiming 80%+ Submission Probability

Do not treat the experiment as RTSS-ready until all are true:

1. Real-frame results include at least two data families, e.g. Argoverse 2 plus
   DROID/OXE, or Waymo plus BDD100K.
2. The paper main table reports controlled fault injection and real frame
   sequences, not CIFAR-10.
3. CIRCA-RT has strong coupled semantic-timing recall at low audit rate and
   bounded audit cost, with false alarms competitive against ContextAware
   Conformal and recent deep baselines.
4. Jetson Orin/TensorRT or another real edge platform reproduces the timing
   conclusion. AutoDL GPU evidence alone is not sufficient for RTSS.
5. Ablations isolate semantic-only, timing-only, context residualization,
   conditional dependence score, and token bucket scheduling.
6. The controlled coupled fault offset used in the AutoDL script is disclosed
   as synthetic fault injection, not presented as naturally occurring sensor
   corruption.
