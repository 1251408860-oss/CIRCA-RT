from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.baselines import BaselineConfig, context_aware_conformal_split, conditional_rff_hsic_split, learned_fourier_independence_split
from circa_rt.core import CIRCARuntimeConfig, apply_circa_rt, fit_circa_rt
from circa_rt.metrics import summarize_detection
from circa_rt.synthetic import generate_trace


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


def load_configs() -> tuple[dict, CIRCARuntimeConfig, BaselineConfig]:
    raw = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    c = raw["circa_rt"]
    b = raw["baselines"]
    return raw, CIRCARuntimeConfig(
        window_size=int(c["window_size"]),
        feature_dim=int(c["feature_dim"]),
        low_quantile=float(c["low_quantile"]),
        high_quantile=float(c["high_quantile"]),
        monitor_cost_ms=float(c["monitor_cost_ms"]),
        heavy_audit_cost_ms=float(c["heavy_audit_cost_ms"]),
        bucket_capacity=float(c["bucket_capacity"]),
        replenish_rate=float(c["replenish_rate"]),
        seed=int(raw["random_seed"]),
    ), BaselineConfig(
        window_size=int(b["window_size"]),
        monitor_cost_ms=float(b["monitor_cost_ms"]),
        audit_cost_ms=float(b["audit_cost_ms"]),
        periodic_audit_interval=int(b["periodic_audit_interval"]),
        random_audit_rate=float(b["random_audit_rate"]),
        seed=int(raw["random_seed"]),
    )


def coupled_strength_trace(*, seed: int, n: int, deadline_ms: float, baseline_latency_ms: float, strength: float) -> pd.DataFrame:
    df = generate_trace(
        trace_type="nominal",
        seed=seed,
        n=n,
        deadline_ms=deadline_ms,
        baseline_latency_ms=baseline_latency_ms,
    ).copy()
    rng = np.random.default_rng(seed + 9000)
    label = np.zeros(n, dtype=np.int64)
    centers = np.linspace(int(n * 0.25), int(n * 0.85), 3, dtype=int)
    length = max(40, int(70 + 25 * strength))
    for center in centers:
        start = max(0, min(n - length, int(center + rng.integers(-25, 26))))
        label[start : start + length] = 1
    idx = label == 1
    z = rng.normal(size=int(idx.sum()))
    sem = df["semantic_residual"].to_numpy(dtype=float)
    tim = df["timing_residual_ms"].to_numpy(dtype=float)
    sem[idx] += strength * (0.75 * z + 0.10 * rng.normal(size=len(z)))
    tim[idx] += strength * (4.5 * z + 0.30 * rng.normal(size=len(z)))
    df["label"] = label
    df["attack_type"] = np.where(idx, f"coupled_strength_{strength:.2f}", "benign")
    df["semantic_residual"] = sem
    df["timing_residual_ms"] = tim
    df["baseline_latency_ms"] = np.clip(baseline_latency_ms + 0.18 * tim + 1.2 * rng.normal(size=n), 1.0, None)
    df["e2e_latency_ms"] = df["baseline_latency_ms"]
    df["deadline_miss"] = (df["e2e_latency_ms"] > df["deadline_ms"]).astype(int)
    df["run_id"] = f"coupled_strength_{strength:.2f}_seed{seed}"
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    parser.add_argument("--out-dir", default=str(ROOT / "results_local_complete" / "attack_strength"))
    args = parser.parse_args()

    raw_cfg, circa_cfg, baseline_cfg = load_configs()
    out = Path(args.out_dir)
    table_dir = out / "tables"
    summary_dir = out / "summaries"
    for directory in [table_dir, summary_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for fold in range(int(raw_cfg["n_seeds"])):
        seed = int(raw_cfg["random_seed"]) + fold
        cal_seed = int(raw_cfg["random_seed"]) + 5000 + fold
        cfg = replace(circa_cfg, seed=seed)
        bcfg = replace(baseline_cfg, seed=seed)
        calibration = pd.concat(
            [
                generate_trace(
                    trace_type=trace_type,
                    seed=cal_seed,
                    n=int(raw_cfg["n_per_trace"]),
                    deadline_ms=float(raw_cfg["deadline_ms"]),
                    baseline_latency_ms=float(raw_cfg["baseline_latency_ms"]),
                )
                for trace_type in ["nominal", "mode_shift"]
            ],
            ignore_index=True,
        )
        model = fit_circa_rt(calibration, cfg)
        for strength in args.strengths:
            raw = coupled_strength_trace(
                seed=seed,
                n=int(raw_cfg["n_per_trace"]),
                deadline_ms=float(raw_cfg["deadline_ms"]),
                baseline_latency_ms=float(raw_cfg["baseline_latency_ms"]),
                strength=float(strength),
            )
            methods = {
                "CIRCA-RT": apply_circa_rt(model, raw, method="CIRCA-RT"),
                "ContextAwareConformal": context_aware_conformal_split(calibration, raw, bcfg),
                "ConditionalRFF-HSIC": conditional_rff_hsic_split(calibration, raw, bcfg),
                "LearnedFourierIndependence": learned_fourier_independence_split(calibration, raw, bcfg),
            }
            for method, scored in methods.items():
                row = summarize_detection(scored, trace_name=f"strength_{strength:.2f}_fold{fold}").to_dict()
                row.update({"method": method, "strength": strength, "fold": fold, "seed": seed})
                rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)
    main = mean_table(summary, ["strength", "method"], METRIC_COLS).sort_values(
        ["strength", "attack_recall"],
        ascending=[True, False],
    )
    main.to_csv(table_dir / "attack_strength_main_table.csv", index=False)
    bootstrap_ci_table(summary, ["strength", "method"], METRIC_COLS).to_csv(table_dir / "attack_strength_main_table_ci.csv", index=False)
    print(main.to_string(index=False))


if __name__ == "__main__":
    main()

