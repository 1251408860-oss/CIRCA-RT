from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.baselines import BASELINE_FUNCS, BaselineConfig
from circa_rt.core import CIRCARuntimeConfig, run_circa_rt
from circa_rt.metrics import summarize_detection
from circa_rt.plots import plot_metric_bars, plot_trace_scores
from circa_rt.schema import write_trace
from circa_rt.synthetic import generate_trace


TRACE_TYPES = [
    "nominal",
    "mode_shift",
    "stealthy_coupled_attack",
    "timing_only_attack",
    "semantic_only_attack",
    "mixed_attack",
]


def load_configs() -> tuple[dict, CIRCARuntimeConfig, BaselineConfig]:
    cfg = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    c = cfg["circa_rt"]
    b = cfg["baselines"]
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
    base_cfg = BaselineConfig(
        window_size=int(b["window_size"]),
        monitor_cost_ms=float(b["monitor_cost_ms"]),
        audit_cost_ms=float(b["audit_cost_ms"]),
        periodic_audit_interval=int(b["periodic_audit_interval"]),
        random_audit_rate=float(b["random_audit_rate"]),
        seed=int(cfg["random_seed"]),
    )
    return cfg, circa_cfg, base_cfg


def main() -> None:
    cfg, circa_cfg, baseline_cfg = load_configs()
    trace_dir = ROOT / "traces" / "synthetic"
    raw_dir = ROOT / "results" / "raw"
    summary_dir = ROOT / "results" / "summaries"
    table_dir = ROOT / "results" / "tables"
    figure_dir = ROOT / "results" / "figures"
    for d in [trace_dir, raw_dir, summary_dir, table_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    representative_raw = None
    representative_scored = None
    for seed_idx in range(int(cfg["n_seeds"])):
        seed = int(cfg["random_seed"]) + seed_idx
        for trace_type in TRACE_TYPES:
            raw = generate_trace(
                trace_type=trace_type,
                seed=seed,
                n=int(cfg["n_per_trace"]),
                deadline_ms=float(cfg["deadline_ms"]),
                baseline_latency_ms=float(cfg["baseline_latency_ms"]),
            )
            trace_path = trace_dir / f"{trace_type}_seed{seed_idx}.csv"
            write_trace(raw, trace_path)
            circa_result = run_circa_rt(raw, circa_cfg)
            circa_out_dir = raw_dir / trace_path.stem / "CIRCA-RT"
            circa_out_dir.mkdir(parents=True, exist_ok=True)
            circa_result.to_csv(circa_out_dir / "scored.csv", index=False)
            summaries.append(summarize_detection(circa_result, trace_name=trace_path.stem).to_dict())
            if trace_type == "stealthy_coupled_attack" and seed_idx == 0:
                representative_raw = raw
                representative_scored = circa_result

            for method, func in BASELINE_FUNCS.items():
                result = func(raw, baseline_cfg)
                out_dir = raw_dir / trace_path.stem / method
                out_dir.mkdir(parents=True, exist_ok=True)
                result.to_csv(out_dir / "scored.csv", index=False)
                summaries.append(summarize_detection(result, trace_name=trace_path.stem).to_dict())

    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)
    main_table = (
        summary.groupby("method", as_index=False)[
            [
                "attack_recall",
                "false_alarm_rate",
                "detection_delay",
                "audit_rate",
                "mean_audit_cost_ms",
                "p99_latency_ms",
                "p999_latency_ms",
                "deadline_miss_ratio",
                "latency_inflation_mean_ms",
                "auroc",
                "auprc",
            ]
        ]
        .mean()
        .sort_values(["attack_recall", "p99_latency_ms"], ascending=[False, True])
    )
    main_table.to_csv(table_dir / "main_table.csv", index=False)
    scenario_table = (
        summary.groupby(["trace_name", "method"], as_index=False)[
            ["attack_recall", "false_alarm_rate", "detection_delay", "audit_rate", "p99_latency_ms", "deadline_miss_ratio"]
        ]
        .mean()
        .sort_values(["trace_name", "attack_recall"], ascending=[True, False])
    )
    scenario_table.to_csv(table_dir / "scenario_table.csv", index=False)
    ablation_methods = [
        "CIRCA-RT",
        "ConditionalRFF-HSIC",
        "RFF-HSIC",
        "LearnedFourierIndependence",
        "OnlineConformalAnomaly",
        "ContextAwareConformal",
        "SemanticThreshold",
        "TimingThreshold",
        "AlwaysAudit",
        "RandomBudgetAudit",
    ]
    summary[summary["method"].isin(ablation_methods)].to_csv(table_dir / "ablation_table.csv", index=False)
    plot_metric_bars(summary, figure_dir)
    if representative_raw is not None and representative_scored is not None:
        plot_trace_scores(representative_raw, representative_scored, figure_dir / "circa_rt_score_timeline.png")
    print(f"wrote traces: {trace_dir}")
    print(f"wrote summary: {summary_dir / 'summary_all.csv'}")
    print(f"wrote main table: {table_dir / 'main_table.csv'}")
    print(main_table.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
