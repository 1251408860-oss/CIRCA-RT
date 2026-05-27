from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import REQUIRED_COLUMNS


PHASE5_TRACE_TYPES = [
    "nominal",
    "mode_shift",
    "burst_delay_attack",
    "jitter_attack",
    "stale_replay_attack",
    "cpu_load_interference",
    "coupled_timing_semantic_attack",
]


def _attack_segments(n: int, rng: np.random.Generator, *, count: int = 3, length: int = 90) -> np.ndarray:
    label = np.zeros(n, dtype=np.int64)
    anchors = np.linspace(int(n * 0.25), int(n * 0.82), count, dtype=int)
    for anchor in anchors:
        start = int(np.clip(anchor + int(rng.integers(-35, 36)), 0, max(n - length, 0)))
        label[start : start + length] = 1
    return label


def _queue_depth(cpu: np.ndarray, gpu: np.ndarray, net_delay: np.ndarray, rng: np.random.Generator, period_ms: float) -> np.ndarray:
    depth = np.zeros(len(cpu), dtype=np.float64)
    q = 0.0
    for i in range(len(cpu)):
        service_pressure = (0.12 * net_delay[i] + 5.5 * cpu[i] + 3.0 * gpu[i]) / max(period_ms, 1.0)
        q = max(0.0, 0.82 * q + service_pressure - 0.65 + 0.08 * rng.normal())
        depth[i] = q
    return depth


def generate_periodic_pipeline_trace(
    *,
    trace_type: str,
    seed: int,
    n: int = 1200,
    period_ms: float = 20.0,
    deadline_ms: float = 50.0,
    baseline_latency_ms: float = 30.0,
) -> pd.DataFrame:
    if trace_type not in PHASE5_TRACE_TYPES:
        raise ValueError(f"unknown Phase 5 trace_type: {trace_type}")

    rng = np.random.default_rng(seed)
    seq = np.arange(n)
    timestamp_ms = seq.astype(float) * float(period_ms)
    phase = (seq // 240) % 3
    mode = np.where(phase == 0, "dds_nominal", np.where(phase == 1, "dds_congested", "dds_switching"))

    cpu = np.clip(0.32 + 0.18 * (mode == "dds_congested") + 0.08 * rng.normal(size=n), 0.03, 0.98)
    gpu = np.clip(0.28 + 0.15 * (mode == "dds_switching") + 0.07 * rng.normal(size=n), 0.02, 0.98)
    base_net = 2.2 + 3.8 * (mode == "dds_congested") + 1.2 * (mode == "dds_switching")
    net_delay = np.clip(base_net + rng.gamma(shape=1.8, scale=0.55, size=n) + 0.35 * rng.normal(size=n), 0.0, None)

    if trace_type == "mode_shift":
        lo, hi = int(n * 0.40), int(n * 0.68)
        cpu[lo:hi] = np.clip(cpu[lo:hi] + 0.24, 0.0, 0.99)
        gpu[lo:hi] = np.clip(gpu[lo:hi] + 0.12, 0.0, 0.99)
        net_delay[lo:hi] += 5.0

    label = np.zeros(n, dtype=np.int64)
    attack_type = np.full(n, "benign", dtype=object)
    if trace_type not in {"nominal", "mode_shift"}:
        label = _attack_segments(n, rng)
        attack_type[label == 1] = trace_type

    attack_idx = label == 1
    if trace_type == "burst_delay_attack":
        hidden_delay = 10.0 + 4.0 * rng.random(np.sum(attack_idx))
        net_delay[attack_idx] += 0.35 * hidden_delay
    elif trace_type == "jitter_attack":
        net_delay[attack_idx] += rng.lognormal(mean=1.3, sigma=0.45, size=np.sum(attack_idx))
    elif trace_type == "cpu_load_interference":
        cpu[attack_idx] = np.clip(cpu[attack_idx] + 0.38 + 0.05 * rng.normal(size=np.sum(attack_idx)), 0.0, 0.99)
        gpu[attack_idx] = np.clip(gpu[attack_idx] + 0.14 + 0.04 * rng.normal(size=np.sum(attack_idx)), 0.0, 0.99)

    queue = _queue_depth(cpu, gpu, net_delay, rng, period_ms)
    callback_latency = np.clip(1.8 + 1.7 * queue + 4.2 * cpu + 0.8 * rng.normal(size=n), 0.0, None)
    jitter = np.abs(net_delay - np.concatenate([[net_delay[0]], net_delay[:-1]])) + np.abs(rng.normal(scale=0.25, size=n))
    message_age = np.clip(period_ms + net_delay + callback_latency + 0.75 * queue + rng.normal(scale=0.8, size=n), 0.0, None)

    semantic_mean = 0.32 * cpu + 0.28 * gpu + 0.018 * message_age + 0.07 * (mode == "dds_switching")
    timing_mean = 0.58 * message_age + 1.6 * jitter + 0.95 * queue + 0.8 * cpu
    semantic = semantic_mean + rng.normal(scale=0.38, size=n)
    timing = timing_mean + rng.normal(scale=1.25, size=n)

    if trace_type == "burst_delay_attack":
        timing[attack_idx] += 7.5 + 1.5 * rng.normal(size=np.sum(attack_idx))
    elif trace_type == "jitter_attack":
        timing[attack_idx] += rng.normal(loc=4.5, scale=5.0, size=np.sum(attack_idx))
        jitter[attack_idx] += 4.5 + 1.0 * rng.random(np.sum(attack_idx))
    elif trace_type == "stale_replay_attack":
        replay_age = 9.0 + 4.0 * rng.random(np.sum(attack_idx))
        message_age[attack_idx] += replay_age
        semantic[attack_idx] += 1.15 + 0.25 * rng.normal(size=np.sum(attack_idx))
        timing[attack_idx] += 2.8 + 0.8 * rng.normal(size=np.sum(attack_idx))
    elif trace_type == "cpu_load_interference":
        queue[attack_idx] += 2.0 + 0.8 * rng.random(np.sum(attack_idx))
        message_age[attack_idx] += 2.5 + 0.6 * queue[attack_idx]
        timing[attack_idx] += 5.8 + 1.4 * rng.normal(size=np.sum(attack_idx))
    elif trace_type == "coupled_timing_semantic_attack":
        z = rng.normal(size=n)
        semantic[attack_idx] += 0.80 * z[attack_idx] + 0.10 * rng.normal(size=np.sum(attack_idx))
        timing[attack_idx] += 5.2 * z[attack_idx] + 0.45 * rng.normal(size=np.sum(attack_idx))
        message_age[attack_idx] += 0.8 * np.abs(z[attack_idx])

    baseline = baseline_latency_ms - 8.0 + 0.34 * message_age + 0.18 * np.maximum(timing, 0.0) + 1.0 * rng.normal(size=n)
    baseline = np.clip(baseline, 1.0, None)

    df = pd.DataFrame(
        {
            "run_id": f"phase5_{trace_type}_seed{seed}",
            "platform": "local_periodic_pipeline",
            "timestamp_ms": timestamp_ms,
            "seq": seq,
            "mode": mode,
            "attack_type": attack_type,
            "label": label,
            "semantic_residual": semantic,
            "timing_residual_ms": timing,
            "context_cpu_util": cpu,
            "context_gpu_util": gpu,
            "context_net_delay_ms": net_delay,
            "context_jitter_ms": jitter,
            "context_queue_depth": queue,
            "context_message_age_ms": message_age,
            "baseline_latency_ms": baseline,
            "e2e_latency_ms": baseline,
            "deadline_ms": float(deadline_ms),
            "deadline_miss": (baseline > deadline_ms).astype(int),
            "method": "raw",
            "alarm": 0,
            "audit": 0,
            "audit_cost_ms": 0.0,
            "monitor_cost_ms": 0.0,
        }
    )
    return df[REQUIRED_COLUMNS]
