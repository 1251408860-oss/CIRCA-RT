from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.baselines import SPLIT_BASELINE_FUNCS, BaselineConfig
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


def model_name_from_path(path: Path) -> str:
    name = path.stem
    for suffix in ["_nominal", "_gpu_interference", "_coupled_gpu_attack"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.split("_")[0]


def main() -> None:
    circa_cfg, baseline_cfg = load_configs()
    root = ROOT / "results_autodl"
    trace_dir = root / "traces_onnxruntime"
    raw_dir = root / "raw_onnxruntime"
    summary_dir = root / "summaries"
    table_dir = root / "tables"
    for d in [raw_dir, summary_dir, table_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    models = sorted({model_name_from_path(p) for p in trace_dir.glob("*.csv")})
    for model_name in models:
        calibration_path = trace_dir / f"{model_name}_nominal.csv"
        if not calibration_path.exists():
            continue
        calibration = read_trace(calibration_path)
        model = fit_circa_rt(calibration, circa_cfg)
        for path in sorted(trace_dir.glob(f"{model_name}_*.csv")):
            trace_type = path.stem[len(model_name) + 1 :]
            raw = read_trace(path)
            circa = apply_circa_rt(model, raw)
            out_dir = raw_dir / model_name / trace_type / "CIRCA-RT"
            out_dir.mkdir(parents=True, exist_ok=True)
            circa.to_csv(out_dir / "scored.csv", index=False)
            row = summarize_detection(circa, trace_name=f"{model_name}_{trace_type}").to_dict()
            row.update({"model": model_name, "trace_type": trace_type, "protocol": "autodl_onnxruntime_nominal_calibration"})
            summaries.append(row)
            for method, func in SPLIT_BASELINE_FUNCS.items():
                result = func(calibration, raw, baseline_cfg)
                out_dir = raw_dir / model_name / trace_type / method
                out_dir.mkdir(parents=True, exist_ok=True)
                result.to_csv(out_dir / "scored.csv", index=False)
                row = summarize_detection(result, trace_name=f"{model_name}_{trace_type}").to_dict()
                row.update({"model": model_name, "trace_type": trace_type, "protocol": "autodl_onnxruntime_nominal_calibration"})
                summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_dir / "onnxruntime_summary_all.csv", index=False)
    summary.to_csv(table_dir / "onnxruntime_summary_all.csv", index=False)
    main = mean_table(summary, ["model", "method"], METRIC_COLS).sort_values(
        ["model", "attack_recall", "p99_latency_ms"], ascending=[True, False, True]
    )
    scenario = mean_table(summary, ["model", "trace_type", "method"], METRIC_COLS).sort_values(
        ["model", "trace_type", "attack_recall"], ascending=[True, True, False]
    )
    main.to_csv(table_dir / "onnxruntime_main_table.csv", index=False)
    scenario.to_csv(table_dir / "onnxruntime_scenario_table.csv", index=False)
    bootstrap_ci_table(summary, ["model", "method"], METRIC_COLS).to_csv(table_dir / "onnxruntime_main_table_ci.csv", index=False)
    print(main.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
