from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table
from circa_rt.core import CIRCARuntimeConfig, apply_circa_rt, fit_circa_rt
from circa_rt.metrics import summarize_detection
from circa_rt.plots import plot_metric_bars
from circa_rt.synthetic import generate_trace


CALIBRATION_TRACE_TYPES = ["nominal", "mode_shift"]
TEST_TRACE_TYPES = ["nominal", "mode_shift", "stealthy_coupled_attack"]
SWEEPS = {
    "window_size": [16, 32, 64],
    "feature_dim": [16, 32, 64],
    "replenish_rate": [0.2, 0.45, 0.8],
}


def load_configs() -> tuple[dict, CIRCARuntimeConfig]:
    cfg = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    c = cfg["circa_rt"]
    circa_cfg = CIRCARuntimeConfig(
        window_size=int(c["window_size"]),
        feature_dim=int(c["feature_dim"]),
        low_quantile=float(c["low_quantile"]),
        high_quantile=float(c["high_quantile"]),
        monitor_cost_ms=float(c["monitor_cost_ms"]),
        heavy_audit_cost_ms=float(c["heavy_audit_cost_ms"]),
        bucket_capacity=float(c["bucket_capacity"]),
        replenish_rate=float(c["replenish_rate"]),
        seed=int(cfg["random_seed"]),
    )
    return cfg, circa_cfg


def build_trace(trace_type: str, seed: int, cfg: dict) -> pd.DataFrame:
    return generate_trace(
        trace_type=trace_type,
        seed=seed,
        n=int(cfg["n_per_trace"]),
        deadline_ms=float(cfg["deadline_ms"]),
        baseline_latency_ms=float(cfg["baseline_latency_ms"]),
    )


def main() -> None:
    cfg, base_cfg = load_configs()
    out_dir = ROOT / "results_split" / "sensitivity"
    summary_dir = out_dir / "summaries"
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    for d in [summary_dir, table_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    protocol = "benign_calibration_split"

    for sweep_name, values in SWEEPS.items():
        for value in values:
            cfg_variant = replace(base_cfg, **{sweep_name: value})
            for fold in range(int(cfg["n_seeds"])):
                test_seed = int(cfg["random_seed"]) + fold
                cal_seed = int(cfg["random_seed"]) + 2000 + fold
                cal_frames = [
                    build_trace(trace_type, cal_seed, cfg)
                    for trace_type in CALIBRATION_TRACE_TYPES
                ]
                calibration = pd.concat(cal_frames, ignore_index=True)
                model = fit_circa_rt(calibration, cfg_variant)
                for trace_type in TEST_TRACE_TYPES:
                    raw = build_trace(trace_type, test_seed, cfg)
                    result = apply_circa_rt(model, raw)
                    row = summarize_detection(result, trace_name=f"{trace_type}_fold{fold}").to_dict()
                    row.update(
                        {
                            "protocol": protocol,
                            "trace_type": trace_type,
                            "sweep_name": sweep_name,
                            "sweep_value": value,
                            "fold": fold,
                        }
                    )
                    summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)

    attack_df = summary[summary["trace_type"] == "stealthy_coupled_attack"].copy()
    benign_df = summary[summary["trace_type"].isin(["nominal", "mode_shift"])].copy()

    attack_table = bootstrap_ci_table(
        attack_df,
        ["sweep_name", "sweep_value"],
        ["attack_recall", "detection_delay", "audit_rate", "p99_latency_ms", "deadline_miss_ratio"],
    )
    benign_table = bootstrap_ci_table(
        benign_df,
        ["sweep_name", "sweep_value"],
        ["false_alarm_rate", "p99_latency_ms", "deadline_miss_ratio"],
    )
    table = attack_table.merge(benign_table, on=["sweep_name", "sweep_value"], how="inner", suffixes=("", "_benign"))
    table.sort_values(["sweep_name", "sweep_value"], inplace=True)
    table.to_csv(table_dir / "sensitivity_table.csv", index=False)

    plot_metric_bars(summary, figure_dir)
    print(f"wrote sensitivity summary: {summary_dir / 'summary_all.csv'}")
    print(f"wrote sensitivity table: {table_dir / 'sensitivity_table.csv'}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
