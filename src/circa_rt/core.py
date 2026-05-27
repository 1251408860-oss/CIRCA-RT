from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import RandomFourierFeatures, Residualizer, rolling_cross_scores_by_group
from .schema import CONTEXT_COLUMNS, validate_trace


@dataclass(frozen=True)
class CIRCARuntimeConfig:
    window_size: int = 32
    feature_dim: int = 32
    low_quantile: float = 0.95
    high_quantile: float = 0.99
    monitor_cost_ms: float = 0.08
    heavy_audit_cost_ms: float = 4.0
    bucket_capacity: float = 12.0
    replenish_rate: float = 0.45
    slack_margin_ms: float = 0.0
    seed: int = 7


@dataclass(frozen=True)
class CIRCARTModel:
    cfg: CIRCARuntimeConfig
    residualizer: Residualizer
    rff_u: RandomFourierFeatures
    rff_v: RandomFourierFeatures
    q_low: float
    q_high: float


def _token_bucket(scores: np.ndarray, q_low: float, q_high: float, cfg: CIRCARuntimeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alarm = np.zeros(len(scores), dtype=np.int64)
    audit = np.zeros(len(scores), dtype=np.int64)
    bucket_trace = np.zeros(len(scores), dtype=np.float64)
    bucket = float(cfg.bucket_capacity)
    for i, score in enumerate(scores):
        bucket = min(float(cfg.bucket_capacity), bucket + float(cfg.replenish_rate))
        if score >= q_high and bucket >= cfg.heavy_audit_cost_ms:
            alarm[i] = 1
            audit[i] = 1
            bucket -= cfg.heavy_audit_cost_ms
        elif score >= q_low:
            alarm[i] = 1
        bucket_trace[i] = bucket
    return alarm, audit, bucket_trace


def _token_bucket_with_slack(
    scores: np.ndarray,
    baseline_latency_ms: np.ndarray,
    deadline_ms: np.ndarray,
    q_low: float,
    q_high: float,
    cfg: CIRCARuntimeConfig,
    *,
    audit_cost_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alarm = np.zeros(len(scores), dtype=np.int64)
    audit = np.zeros(len(scores), dtype=np.int64)
    bucket_trace = np.zeros(len(scores), dtype=np.float64)
    slack_trace = deadline_ms.astype(float) - baseline_latency_ms.astype(float) - float(cfg.monitor_cost_ms)
    bucket = float(cfg.bucket_capacity)
    for i, score in enumerate(scores):
        bucket = min(float(cfg.bucket_capacity), bucket + float(cfg.replenish_rate))
        if score >= q_low:
            alarm[i] = 1
        has_tokens = bucket >= cfg.heavy_audit_cost_ms
        has_slack = slack_trace[i] >= float(audit_cost_ms) + float(cfg.slack_margin_ms)
        if score >= q_high and has_tokens and has_slack:
            audit[i] = 1
            bucket -= cfg.heavy_audit_cost_ms
        bucket_trace[i] = bucket
    return alarm, audit, bucket_trace, slack_trace


def fit_circa_rt(calibration_df: pd.DataFrame, cfg: CIRCARuntimeConfig) -> CIRCARTModel:
    validate_trace(calibration_df)
    context = calibration_df[CONTEXT_COLUMNS].to_numpy(dtype=np.float64)
    mode = calibration_df["mode"].astype(str).to_numpy()
    semantic = calibration_df["semantic_residual"].to_numpy(dtype=np.float64)
    timing = calibration_df["timing_residual_ms"].to_numpy(dtype=np.float64)
    label = calibration_df["label"].to_numpy(dtype=np.int64)
    benign_idx = label == 0
    if np.sum(benign_idx) < max(10, cfg.window_size):
        raise ValueError("CIRCA-RT needs enough benign calibration samples")
    residualizer = Residualizer().fit(semantic[benign_idx], timing[benign_idx], context[benign_idx], mode[benign_idx])
    u, v = residualizer.transform(semantic, timing, context, mode)
    rff_u = RandomFourierFeatures(cfg.feature_dim, seed=cfg.seed).fit(u[benign_idx])
    rff_v = RandomFourierFeatures(cfg.feature_dim, seed=cfg.seed + 1).fit(v[benign_idx])
    phi_u = rff_u.transform(u)
    phi_v = rff_v.transform(v)
    scores = rolling_cross_scores_by_group(phi_u, phi_v, calibration_df["run_id"].astype(str).to_numpy(), cfg.window_size)
    q_low = float(np.quantile(scores[benign_idx], cfg.low_quantile))
    q_high = float(np.quantile(scores[benign_idx], cfg.high_quantile))
    return CIRCARTModel(
        cfg=cfg,
        residualizer=residualizer,
        rff_u=rff_u,
        rff_v=rff_v,
        q_low=q_low,
        q_high=q_high,
    )


def apply_circa_rt(model: CIRCARTModel, df: pd.DataFrame, *, method: str = "CIRCA-RT") -> pd.DataFrame:
    validate_trace(df)
    out = df.copy()
    context = out[CONTEXT_COLUMNS].to_numpy(dtype=np.float64)
    mode = out["mode"].astype(str).to_numpy()
    semantic = out["semantic_residual"].to_numpy(dtype=np.float64)
    timing = out["timing_residual_ms"].to_numpy(dtype=np.float64)
    u, v = model.residualizer.transform(semantic, timing, context, mode)
    phi_u = model.rff_u.transform(u)
    phi_v = model.rff_v.transform(v)
    scores = rolling_cross_scores_by_group(phi_u, phi_v, out["run_id"].astype(str).to_numpy(), model.cfg.window_size)
    alarm, audit, bucket = _token_bucket(scores, model.q_low, model.q_high, model.cfg)
    out["method"] = method
    out["score"] = scores
    out["conditional_semantic_residual"] = u
    out["conditional_timing_residual"] = v
    out["alarm"] = alarm
    out["audit"] = audit
    out["audit_cost_ms"] = audit.astype(float) * model.cfg.heavy_audit_cost_ms
    out["monitor_cost_ms"] = model.cfg.monitor_cost_ms
    out["e2e_latency_ms"] = out["baseline_latency_ms"].to_numpy(dtype=float) + out["monitor_cost_ms"] + out["audit_cost_ms"]
    out["deadline_miss"] = (out["e2e_latency_ms"] > out["deadline_ms"]).astype(int)
    out["bucket_level"] = bucket
    out["q_low"] = model.q_low
    out["q_high"] = model.q_high
    return out


def apply_circa_rt_slack_admissible(
    model: CIRCARTModel,
    df: pd.DataFrame,
    *,
    method: str = "CIRCA-RT-Slack",
    audit_cost_ms: float | None = None,
) -> pd.DataFrame:
    validate_trace(df)
    out = df.copy()
    context = out[CONTEXT_COLUMNS].to_numpy(dtype=np.float64)
    mode = out["mode"].astype(str).to_numpy()
    semantic = out["semantic_residual"].to_numpy(dtype=np.float64)
    timing = out["timing_residual_ms"].to_numpy(dtype=np.float64)
    u, v = model.residualizer.transform(semantic, timing, context, mode)
    phi_u = model.rff_u.transform(u)
    phi_v = model.rff_v.transform(v)
    scores = rolling_cross_scores_by_group(phi_u, phi_v, out["run_id"].astype(str).to_numpy(), model.cfg.window_size)
    audit_cost = float(model.cfg.heavy_audit_cost_ms if audit_cost_ms is None else audit_cost_ms)
    alarm, audit, bucket, slack = _token_bucket_with_slack(
        scores,
        out["baseline_latency_ms"].to_numpy(dtype=float),
        out["deadline_ms"].to_numpy(dtype=float),
        model.q_low,
        model.q_high,
        model.cfg,
        audit_cost_ms=audit_cost,
    )
    out["method"] = method
    out["score"] = scores
    out["conditional_semantic_residual"] = u
    out["conditional_timing_residual"] = v
    out["alarm"] = alarm
    out["audit"] = audit
    out["audit_cost_ms"] = audit.astype(float) * audit_cost
    out["monitor_cost_ms"] = model.cfg.monitor_cost_ms
    out["e2e_latency_ms"] = out["baseline_latency_ms"].to_numpy(dtype=float) + out["monitor_cost_ms"] + out["audit_cost_ms"]
    out["deadline_miss"] = (out["e2e_latency_ms"] > out["deadline_ms"]).astype(int)
    out["bucket_level"] = bucket
    out["deadline_slack_before_audit_ms"] = slack
    out["slack_admissible"] = (slack >= audit_cost + float(model.cfg.slack_margin_ms)).astype(int)
    out["q_low"] = model.q_low
    out["q_high"] = model.q_high
    return out


def run_circa_rt(df: pd.DataFrame, cfg: CIRCARuntimeConfig) -> pd.DataFrame:
    return apply_circa_rt(fit_circa_rt(df, cfg), df)
