from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_autodl"


def load_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    table_dir = RESULTS / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    latency_frames: list[pd.DataFrame] = []
    torchv = load_optional(table_dir / "torchvision_latency_table.csv")
    if not torchv.empty:
        torchv = torchv.copy()
        torchv["runtime"] = "pytorch_torchvision"
        latency_frames.append(torchv)
    ort = load_optional(table_dir / "onnxruntime_latency_table.csv")
    if not ort.empty:
        ort = ort.copy()
        ort["runtime"] = "onnxruntime_cuda"
        latency_frames.append(ort)
    tiny = load_optional(table_dir / "torch_latency_table.csv")
    if not tiny.empty:
        tiny = tiny.copy()
        tiny["runtime"] = "pytorch_tinyconv"
        tiny["model"] = "tinyconv"
        latency_frames.append(tiny)

    if latency_frames:
        latency = pd.concat(latency_frames, ignore_index=True, sort=False)
        keep = ["runtime", "model", "trace_type", "device", "provider", "n", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "p999_ms", "deadline_miss_ratio"]
        latency = latency[[c for c in keep if c in latency.columns]]
        latency.to_csv(table_dir / "autodl_latency_overview.csv", index=False)

    metric_frames: list[pd.DataFrame] = []
    for runtime, path in [
        ("pytorch_torchvision", table_dir / "torchvision_main_table.csv"),
        ("onnxruntime_cuda", table_dir / "onnxruntime_main_table.csv"),
        ("pytorch_tinyconv", table_dir / "gpu_trace_main_table.csv"),
    ]:
        df = load_optional(path)
        if not df.empty:
            df = df.copy()
            df["runtime"] = runtime
            if "model" not in df.columns:
                df["model"] = "tinyconv"
            metric_frames.append(df)

    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
        key_methods = ["CIRCA-RT", "RFF-HSIC", "ConditionalRFF-HSIC", "ContextAwareConformal", "AlwaysAudit"]
        overview = metrics[metrics["method"].isin(key_methods)].copy()
        keep = [
            "runtime",
            "model",
            "method",
            "attack_recall",
            "false_alarm_rate",
            "audit_rate",
            "p99_latency_ms",
            "p999_latency_ms",
            "deadline_miss_ratio",
            "latency_inflation_mean_ms",
        ]
        overview = overview[[c for c in keep if c in overview.columns]]
        overview.sort_values(["runtime", "model", "method"], inplace=True)
        overview.to_csv(table_dir / "autodl_monitoring_overview.csv", index=False)

    print(f"wrote {table_dir / 'autodl_latency_overview.csv'}")
    print(f"wrote {table_dir / 'autodl_monitoring_overview.csv'}")


if __name__ == "__main__":
    main()
