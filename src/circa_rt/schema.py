from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "run_id",
    "platform",
    "timestamp_ms",
    "seq",
    "mode",
    "attack_type",
    "label",
    "semantic_residual",
    "timing_residual_ms",
    "context_cpu_util",
    "context_gpu_util",
    "context_net_delay_ms",
    "context_jitter_ms",
    "context_queue_depth",
    "context_message_age_ms",
    "baseline_latency_ms",
    "e2e_latency_ms",
    "deadline_ms",
    "deadline_miss",
    "method",
    "alarm",
    "audit",
    "audit_cost_ms",
    "monitor_cost_ms",
]


CONTEXT_COLUMNS = [
    "context_cpu_util",
    "context_gpu_util",
    "context_net_delay_ms",
    "context_jitter_ms",
    "context_queue_depth",
    "context_message_age_ms",
]


def validate_trace(df: pd.DataFrame, *, path: str | Path | None = None) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        where = f" in {path}" if path else ""
        raise ValueError(f"missing required columns{where}: {missing}")
    if not set(df["label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0/1 values")


def read_trace(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_trace(df, path=path)
    return df


def write_trace(df: pd.DataFrame, path: str | Path) -> None:
    validate_trace(df, path=path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
