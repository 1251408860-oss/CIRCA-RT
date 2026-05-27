from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import REQUIRED_COLUMNS


def gpu_latency_trace(
    *,
    latencies_ms: np.ndarray,
    trace_type: str,
    seed: int,
    deadline_ms: float,
    platform: str = "autodl_server_gpu",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latencies = np.asarray(latencies_ms, dtype=np.float64)
    n = len(latencies)
    if n == 0:
        raise ValueError("latencies_ms must not be empty")
    seq = np.arange(n)
    timestamp_ms = seq.astype(float) * 20.0
    label = np.zeros(n, dtype=np.int64)
    attack_type = np.full(n, "benign", dtype=object)
    mode = np.full(n, "gpu_nominal", dtype=object)

    if trace_type != "nominal":
        starts = np.linspace(int(n * 0.25), int(n * 0.8), 3, dtype=int)
        length = max(20, min(80, n // 12))
        for start in starts:
            lo = int(np.clip(start + int(rng.integers(-10, 11)), 0, max(n - length, 0)))
            label[lo : lo + length] = 1
        attack_type[label == 1] = trace_type

    baseline = np.clip(latencies, 0.001, None)
    normalized = (baseline - float(np.median(baseline))) / (float(np.std(baseline)) + 1e-6)
    cpu = np.clip(0.35 + 0.05 * rng.normal(size=n) + 0.03 * normalized, 0.02, 0.98)
    gpu = np.clip(0.55 + 0.10 * rng.normal(size=n) + 0.12 * normalized, 0.02, 0.99)
    net_delay = np.clip(2.0 + 0.2 * rng.normal(size=n), 0.0, None)
    jitter = np.clip(np.abs(np.diff(np.concatenate([[baseline[0]], baseline]))) + 0.05 * rng.random(n), 0.0, None)
    queue = np.clip(1.0 + 0.15 * normalized + 0.15 * rng.normal(size=n), 0.0, None)
    message_age = np.clip(20.0 + baseline + 0.4 * jitter + 0.3 * queue, 0.0, None)

    semantic = 0.25 * cpu + 0.35 * gpu + 0.02 * message_age + 0.15 * rng.normal(size=n)
    timing = baseline + 0.3 * jitter + 0.2 * queue + 0.25 * rng.normal(size=n)
    idx = label == 1
    if trace_type == "gpu_interference":
        timing[idx] += 0.25 * np.maximum(baseline[idx] - np.median(baseline), 0.0)
        gpu[idx] = np.clip(gpu[idx] + 0.20, 0.0, 0.99)
    elif trace_type == "coupled_gpu_attack":
        benign = ~idx
        z = rng.normal(size=np.sum(idx))
        eps = rng.normal(size=np.sum(idx))
        rho = 0.96
        if np.any(benign):
            sem_mu = float(np.mean(semantic[benign]))
            tim_mu = float(np.mean(timing[benign]))
            sem_scale = float(np.std(semantic[benign]) + 1e-6)
            tim_scale = float(np.std(timing[benign]) + 1e-6)
        else:
            sem_mu = float(np.mean(semantic))
            tim_mu = float(np.mean(timing))
            sem_scale = float(np.std(semantic) + 1e-6)
            tim_scale = float(np.std(timing) + 1e-6)
        semantic[idx] = sem_mu + sem_scale * z
        timing[idx] = tim_mu + tim_scale * (rho * z + np.sqrt(1.0 - rho * rho) * eps)

    df = pd.DataFrame(
        {
            "run_id": f"{platform}_{trace_type}_seed{seed}",
            "platform": platform,
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
