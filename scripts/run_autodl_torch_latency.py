from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.gpu_trace import gpu_latency_trace
from circa_rt.schema import write_trace


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def run_latency(*, n: int, warmup: int, image_size: int, device: str, interference: bool) -> np.ndarray:
    torch.set_grad_enabled(False)
    model = TinyConvNet().eval().to(device)
    x = torch.randn(1, 3, image_size, image_size, device=device)
    interferer = None
    if interference:
        interferer = torch.randn(768, 768, device=device)
    for _ in range(warmup):
        _ = model(x)
        if interferer is not None:
            _ = interferer @ interferer
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latencies: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        _ = model(x)
        if interferer is not None:
            _ = interferer @ interferer
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(latencies, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--deadline-ms", type=float, default=8.0)
    parser.add_argument("--out-dir", default=str(ROOT / "results_autodl"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this script requires an AutoDL GPU instance")
    device = "cuda"
    out = Path(args.out_dir)
    trace_dir = out / "traces"
    table_dir = out / "tables"
    log_dir = out / "logs"
    for d in [trace_dir, table_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for trace_type, interference in [("nominal", False), ("gpu_interference", True), ("coupled_gpu_attack", False)]:
        latencies = run_latency(n=args.n, warmup=args.warmup, image_size=args.image_size, device=device, interference=interference)
        if trace_type == "coupled_gpu_attack":
            latencies = latencies + np.random.default_rng(7).normal(0.0, 0.05 * np.std(latencies), size=len(latencies))
        trace = gpu_latency_trace(latencies_ms=latencies, trace_type=trace_type, seed=7, deadline_ms=args.deadline_ms)
        write_trace(trace, trace_dir / f"{trace_type}.csv")
        rows.append(
            {
                "trace_type": trace_type,
                "device": torch.cuda.get_device_name(0),
                "n": int(len(latencies)),
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "p99_ms": percentile(latencies, 0.99),
                "p999_ms": percentile(latencies, 0.999),
                "deadline_miss_ratio": float(np.mean(latencies > args.deadline_ms)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(table_dir / "torch_latency_table.csv", index=False)
    (log_dir / "torch_latency_env.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "image_size": args.image_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
