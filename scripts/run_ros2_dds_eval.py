from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.baselines import RECENT_DEEP_BASELINE_FUNCS, SPLIT_BASELINE_FUNCS, BaselineConfig
from circa_rt.core import CIRCARuntimeConfig, apply_circa_rt, fit_circa_rt
from circa_rt.metrics import summarize_detection
from circa_rt.schema import read_trace


METRIC_COLS = [
    "attack_recall",
    "false_alarm_rate",
    "detection_delay",
    "audit_rate",
    "mean_audit_cost_ms",
    "monitor_cost_mean_ms",
    "p99_latency_ms",
    "p999_latency_ms",
    "deadline_miss_ratio",
    "latency_inflation_mean_ms",
    "auroc",
    "auprc",
]


def load_configs() -> tuple[CIRCARuntimeConfig, BaselineConfig]:
    cfg = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    c = cfg["circa_rt"]
    b = cfg["baselines"]
    return (
        CIRCARuntimeConfig(
            window_size=int(c["window_size"]),
            feature_dim=int(c["feature_dim"]),
            low_quantile=float(c["low_quantile"]),
            high_quantile=float(c["high_quantile"]),
            monitor_cost_ms=float(c["monitor_cost_ms"]),
            heavy_audit_cost_ms=float(c["heavy_audit_cost_ms"]),
            bucket_capacity=float(c["bucket_capacity"]),
            replenish_rate=float(c["replenish_rate"]),
            seed=int(cfg["random_seed"]),
        ),
        BaselineConfig(
            window_size=int(b["window_size"]),
            monitor_cost_ms=float(b["monitor_cost_ms"]),
            audit_cost_ms=float(b["audit_cost_ms"]),
            periodic_audit_interval=int(b["periodic_audit_interval"]),
            random_audit_rate=float(b["random_audit_rate"]),
            seed=int(cfg["random_seed"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "results_autodl_ros2"))
    args = parser.parse_args()

    circa_cfg, baseline_cfg = load_configs()
    root = Path(args.root)
    trace_dir = root / "traces"
    raw_dir = root / "raw"
    summary_dir = root / "summaries"
    table_dir = root / "tables"
    for d in [raw_dir, summary_dir, table_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for seed_dir in sorted(trace_dir.glob("seed*")):
        if not seed_dir.is_dir() or not (seed_dir / "nominal.csv").exists():
            continue
        seed = int(seed_dir.name.replace("seed", ""))
        calibration_frames = [read_trace(seed_dir / "nominal.csv")]
        if (seed_dir / "mode_shift.csv").exists():
            calibration_frames.append(read_trace(seed_dir / "mode_shift.csv"))
        calibration = pd.concat(calibration_frames, ignore_index=True)
        model = fit_circa_rt(calibration, circa_cfg)

        for path in sorted(seed_dir.glob("*.csv")):
            trace_type = path.stem
            raw = read_trace(path)
            trace_key = f"ros2_seed{seed}_{trace_type}"

            circa = apply_circa_rt(model, raw)
            out_dir = raw_dir / f"seed{seed}" / trace_type / "CIRCA-RT"
            out_dir.mkdir(parents=True, exist_ok=True)
            circa.to_csv(out_dir / "scored.csv", index=False)
            row = summarize_detection(circa, trace_name=trace_key).to_dict()
            row.update({"seed": seed, "trace_type": trace_type, "protocol": "ros2_nominal_modeshift_calibration"})
            summaries.append(row)

            for method, func in SPLIT_BASELINE_FUNCS.items():
                if method in RECENT_DEEP_BASELINE_FUNCS:
                    continue
                result = func(calibration, raw, baseline_cfg)
                out_dir = raw_dir / f"seed{seed}" / trace_type / method
                out_dir.mkdir(parents=True, exist_ok=True)
                result.to_csv(out_dir / "scored.csv", index=False)
                row = summarize_detection(result, trace_name=trace_key).to_dict()
                row.update({"seed": seed, "trace_type": trace_type, "protocol": "ros2_nominal_modeshift_calibration"})
                summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)
    main = mean_table(summary, ["method"], METRIC_COLS).sort_values(["attack_recall", "p99_latency_ms"], ascending=[False, True])
    main.to_csv(table_dir / "ros2_main_table.csv", index=False)
    bootstrap_ci_table(summary, ["method"], METRIC_COLS).to_csv(table_dir / "ros2_main_table_ci.csv", index=False)
    scenario = mean_table(summary, ["trace_type", "method"], METRIC_COLS).sort_values(
        ["trace_type", "attack_recall"], ascending=[True, False]
    )
    scenario.to_csv(table_dir / "ros2_scenario_table.csv", index=False)
    print(main.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
