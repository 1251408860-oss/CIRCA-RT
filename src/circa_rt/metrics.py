from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EPS = 1e-12


@dataclass(frozen=True)
class SummaryMetrics:
    method: str
    trace_name: str
    run_id: str
    platform: str
    n: int
    attack_recall: float
    false_alarm_rate: float
    detection_delay: float
    audit_rate: float
    mean_audit_cost_ms: float
    monitor_cost_mean_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    p999_latency_ms: float
    deadline_miss_ratio: float
    latency_inflation_mean_ms: float
    auroc: float
    auprc: float

    def to_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def _safe_auc(label: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(label)) < 2:
        return 0.5, float(np.mean(label)) if label.size else 0.0
    return float(roc_auc_score(label, score)), float(average_precision_score(label, score))


def detection_delay(label: np.ndarray, alarm: np.ndarray) -> float:
    starts = np.where((label == 1) & (np.concatenate([[0], label[:-1]]) == 0))[0]
    if starts.size == 0:
        return float("nan")
    delays: list[float] = []
    for start in starts:
        idx = start
        detected = False
        segment_len = 0
        while idx < len(label) and label[idx] == 1:
            segment_len += 1
            if alarm[idx] == 1:
                delays.append(float(idx - start))
                detected = True
                break
            idx += 1
        if not detected:
            delays.append(float(max(segment_len, 1)))
    return float(np.mean(delays))


def summarize_detection(df: pd.DataFrame, *, trace_name: str, score_col: str = "score") -> SummaryMetrics:
    label = df["label"].to_numpy(dtype=int)
    alarm = df["alarm"].to_numpy(dtype=int)
    audit = df["audit"].to_numpy(dtype=int)
    benign = label == 0
    attack = label == 1
    score = df[score_col].to_numpy(dtype=float) if score_col in df.columns else alarm.astype(float)
    auroc, auprc = _safe_auc(label, score)
    e2e = df["e2e_latency_ms"].to_numpy(dtype=float)
    base = df["baseline_latency_ms"].to_numpy(dtype=float)
    return SummaryMetrics(
        method=str(df["method"].iloc[0]),
        trace_name=trace_name,
        run_id=str(df["run_id"].iloc[0]),
        platform=str(df["platform"].iloc[0]),
        n=int(len(df)),
        attack_recall=float(np.mean(alarm[attack])) if np.any(attack) else float("nan"),
        false_alarm_rate=float(np.mean(alarm[benign])) if np.any(benign) else 0.0,
        detection_delay=detection_delay(label, alarm),
        audit_rate=float(np.mean(audit)) if audit.size else 0.0,
        mean_audit_cost_ms=float(np.mean(df["audit_cost_ms"].to_numpy(dtype=float))),
        monitor_cost_mean_ms=float(np.mean(df["monitor_cost_ms"].to_numpy(dtype=float))),
        p50_latency_ms=float(np.quantile(e2e, 0.50)),
        p95_latency_ms=float(np.quantile(e2e, 0.95)),
        p99_latency_ms=float(np.quantile(e2e, 0.99)),
        p999_latency_ms=float(np.quantile(e2e, 0.999)),
        deadline_miss_ratio=float(np.mean(df["deadline_miss"].to_numpy(dtype=int))),
        latency_inflation_mean_ms=float(np.mean(e2e - base)),
        auroc=auroc,
        auprc=auprc,
    )
