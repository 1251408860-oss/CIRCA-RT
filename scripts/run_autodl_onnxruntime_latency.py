from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
}


def export_onnx(model_name: str, image_size: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    model = MODEL_BUILDERS[model_name]().eval()
    x = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        x,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
    )


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def make_session(path: Path, provider: str) -> ort.InferenceSession:
    providers = ort.get_available_providers()
    if provider not in providers:
        raise RuntimeError(f"provider {provider} not available; available={providers}")
    return ort.InferenceSession(str(path), providers=[provider, "CPUExecutionProvider"])


def run_latency(session: ort.InferenceSession, *, n: int, warmup: int, image_size: int, interference: bool) -> np.ndarray:
    rng = np.random.default_rng(23)
    x = rng.normal(size=(1, 3, image_size, image_size)).astype(np.float32)
    input_name = session.get_inputs()[0].name
    interference_a = rng.normal(size=(384, 384)).astype(np.float32) if interference else None
    interference_b = rng.normal(size=(384, 384)).astype(np.float32) if interference else None
    for _ in range(warmup):
        session.run(None, {input_name: x})
        if interference_a is not None and interference_b is not None:
            _ = interference_a @ interference_b
    latencies: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        if interference_a is not None and interference_b is not None:
            _ = interference_a @ interference_b
        latencies.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(latencies, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenet_v2"])
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--deadline-ms", type=float, default=12.0)
    parser.add_argument("--out-dir", default=str(ROOT / "results_autodl"))
    args = parser.parse_args()

    unknown = sorted(set(args.models) - set(MODEL_BUILDERS))
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choices={sorted(MODEL_BUILDERS)}")

    out = Path(args.out_dir)
    model_dir = out / "onnx"
    trace_dir = out / "traces_onnxruntime"
    table_dir = out / "tables"
    log_dir = out / "logs"
    for d in [model_dir, trace_dir, table_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in args.models:
        onnx_path = model_dir / f"{model_name}.onnx"
        export_onnx(model_name, args.image_size, onnx_path)
        session = make_session(onnx_path, args.provider)
        for trace_type, interference in [("nominal", False), ("gpu_interference", True), ("coupled_gpu_attack", False)]:
            latencies = run_latency(session, n=args.n, warmup=args.warmup, image_size=args.image_size, interference=interference)
            if trace_type == "coupled_gpu_attack":
                latencies = latencies + np.random.default_rng(29).normal(0.0, 0.03 * np.std(latencies), size=len(latencies))
            trace = gpu_latency_trace(
                latencies_ms=latencies,
                trace_type=trace_type,
                seed=29,
                deadline_ms=args.deadline_ms,
                platform=f"autodl_onnxruntime_{model_name}",
            )
            write_trace(trace, trace_dir / f"{model_name}_{trace_type}.csv")
            rows.append(
                {
                    "model": model_name,
                    "provider": args.provider,
                    "trace_type": trace_type,
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
    table.to_csv(table_dir / "onnxruntime_latency_table.csv", index=False)
    (log_dir / "onnxruntime_latency_env.json").write_text(
        json.dumps(
            {
                "onnxruntime": ort.__version__,
                "available_providers": ort.get_available_providers(),
                "provider": args.provider,
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
