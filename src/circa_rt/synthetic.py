from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import REQUIRED_COLUMNS


def _attack_segments(n: int, rng: np.random.Generator, count: int = 3, length: int = 80) -> np.ndarray:
    label = np.zeros(n, dtype=np.int64)
    candidates = np.linspace(int(n * 0.25), int(n * 0.85), count, dtype=int)
    for c in candidates:
        jitter = int(rng.integers(-30, 31))
        start = max(0, min(n - length, c + jitter))
        label[start : start + length] = 1
    return label


def generate_trace(
    *,
    trace_type: str,
    seed: int,
    n: int = 1200,
    deadline_ms: float = 50.0,
    baseline_latency_ms: float = 30.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seq = np.arange(n)
    timestamp_ms = seq * 20.0
    phase = (seq // 200) % 3
    mode = np.where(phase == 0, "nominal", np.where(phase == 1, "congested", "switching"))

    cpu = np.clip(0.35 + 0.20 * (mode == "congested") + 0.08 * rng.normal(size=n), 0.05, 0.98)
    gpu = np.clip(0.30 + 0.18 * (mode == "switching") + 0.08 * rng.normal(size=n), 0.02, 0.98)
    net_delay = np.clip(2.0 + 5.0 * (mode == "congested") + 1.2 * rng.normal(size=n), 0.0, None)
    jitter = np.clip(0.6 + 1.8 * (mode == "switching") + 0.5 * rng.normal(size=n), 0.0, None)
    queue = np.clip(1.0 + 5.0 * cpu + 2.0 * (mode == "congested") + rng.normal(size=n), 0.0, None)
    msg_age = np.clip(12.0 + net_delay + 1.5 * jitter + 0.8 * queue + rng.normal(scale=1.0, size=n), 0.0, None)

    semantic_mean = 0.4 * cpu + 0.3 * gpu + 0.08 * (mode == "switching")
    timing_mean = 2.8 * net_delay + 1.5 * jitter + 0.7 * queue + 0.8 * cpu
    semantic = semantic_mean + rng.normal(scale=0.45, size=n)
    timing = timing_mean + rng.normal(scale=1.7, size=n)

    label = np.zeros(n, dtype=np.int64)
    attack_type = np.full(n, "benign", dtype=object)

    if trace_type in {"stealthy_coupled_attack", "timing_only_attack", "semantic_only_attack", "mixed_attack"}:
        label = _attack_segments(n, rng)
        attack_type[label == 1] = trace_type

    if trace_type == "mode_shift":
        # Benign context shift. No attack label; context explains most changes.
        cpu[450:700] = np.clip(cpu[450:700] + 0.25, 0.0, 0.99)
        net_delay[450:700] += 6.0
        queue[450:700] += 3.0
        semantic += 0.45 * ((seq >= 450) & (seq < 700))
        timing += 12.0 * ((seq >= 450) & (seq < 700))
    elif trace_type == "stealthy_coupled_attack":
        z = rng.normal(size=n)
        idx = label == 1
        semantic[idx] += 0.85 * z[idx] + 0.12 * rng.normal(size=np.sum(idx))
        timing[idx] += 5.0 * z[idx] + 0.35 * rng.normal(size=np.sum(idx))
    elif trace_type == "timing_only_attack":
        idx = label == 1
        timing[idx] += 8.0 + 2.5 * rng.normal(size=np.sum(idx))
    elif trace_type == "semantic_only_attack":
        idx = label == 1
        semantic[idx] += 1.4 + 0.35 * rng.normal(size=np.sum(idx))
    elif trace_type == "mixed_attack":
        idx = np.where(label == 1)[0]
        chunks = np.array_split(idx, 3)
        if len(chunks) >= 1:
            semantic[chunks[0]] += 1.2 + 0.25 * rng.normal(size=len(chunks[0]))
            attack_type[chunks[0]] = "semantic_only"
        if len(chunks) >= 2:
            timing[chunks[1]] += 8.0 + 1.5 * rng.normal(size=len(chunks[1]))
            attack_type[chunks[1]] = "timing_only"
        if len(chunks) >= 3:
            z = rng.normal(size=len(chunks[2]))
            semantic[chunks[2]] += 0.85 * z
            timing[chunks[2]] += 5.0 * z
            attack_type[chunks[2]] = "coupled"

    baseline = baseline_latency_ms + 0.18 * timing + 1.2 * rng.normal(size=n)
    baseline = np.clip(baseline, 1.0, None)
    df = pd.DataFrame(
        {
            "run_id": f"{trace_type}_seed{seed}",
            "platform": "local_synthetic",
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
            "context_message_age_ms": msg_age,
            "baseline_latency_ms": baseline,
            "e2e_latency_ms": baseline,
            "deadline_ms": deadline_ms,
            "deadline_miss": (baseline > deadline_ms).astype(int),
            "method": "raw",
            "alarm": 0,
            "audit": 0,
            "audit_cost_ms": 0.0,
            "monitor_cost_ms": 0.0,
        }
    )
    return df[REQUIRED_COLUMNS]
