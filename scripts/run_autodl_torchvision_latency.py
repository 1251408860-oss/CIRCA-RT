from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.gpu_trace import gpu_latency_trace
from circa_rt.schema import write_trace


MODEL_BUILDERS = {
    "resnet18": lambda: models.resnet18(weights=None),
    "mobilenet_v2": lambda: models.mobilenet_v2(weights=None),
    "squeezenet1_1": lambda: models.squeezenet1_1(weights=None),
}


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def allocate_interferer(size: int, device: str) -> torch.Tensor:
    return torch.randn(size, size, device=device)


def run_latency(
    *,
    model_name: str,
    n: int,
    warmup: int,
    image_size: int,
    device: str,
    interference: bool,
) -> np.ndarray:
    torch.set_grad_enabled(False)
    model = MODEL_BUILDERS[model_name]().eval().to(device)
    x = torch.randn(1, 3, image_size, image_size, device=device)
    interferer = allocate_interferer(512, device) if interference else None
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
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenet_v2"])
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--deadline-ms", type=float, default=12.0)
    parser.add_argument("--out-dir", default=str(ROOT / "results_autodl"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this script requires an AutoDL GPU instance")
    unknown = sorted(set(args.models) - set(MODEL_BUILDERS))
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choices={sorted(MODEL_BUILDERS)}")

    device = "cuda"
    out = Path(args.out_dir)
    trace_dir = out / "traces_torchvision"
    table_dir = out / "tables"
    log_dir = out / "logs"
    for d in [trace_dir, table_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in args.models:
        for trace_type, interference in [("nominal", False), ("gpu_interference", True), ("coupled_gpu_attack", False)]:
            latencies = run_latency(
                model_name=model_name,
                n=args.n,
                warmup=args.warmup,
                image_size=args.image_size,
                device=device,
                interference=interference,
            )
            if trace_type == "coupled_gpu_attack":
                latencies = latencies + np.random.default_rng(13).normal(0.0, 0.03 * np.std(latencies), size=len(latencies))
            trace = gpu_latency_trace(
                latencies_ms=latencies,
                trace_type=trace_type,
                seed=13,
                deadline_ms=args.deadline_ms,
                platform=f"autodl_server_gpu_{model_name}",
            )
            write_trace(trace, trace_dir / f"{model_name}_{trace_type}.csv")
            rows.append(
                {
                    "model": model_name,
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
    table.to_csv(table_dir / "torchvision_latency_table.csv", index=False)
    (log_dir / "torchvision_latency_env.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "torchvision": getattr(models, "__version__", "unknown"),
                "device": torch.cuda.get_device_name(0),
                "models": args.models,
                "image_size": args.image_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
