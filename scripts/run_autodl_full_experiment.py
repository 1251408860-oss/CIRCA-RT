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

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.baselines import RECENT_DEEP_BASELINE_FUNCS, SPLIT_BASELINE_FUNCS, BaselineConfig
from circa_rt.core import CIRCARuntimeConfig, apply_circa_rt, fit_circa_rt
from circa_rt.gpu_trace import gpu_latency_trace
from circa_rt.metrics import summarize_detection
from circa_rt.schema import read_trace, write_trace


MODEL_BUILDERS = {
    "resnet18": lambda: models.resnet18(weights=None),
    "mobilenet_v2": lambda: models.mobilenet_v2(weights=None),
    "squeezenet1_1": lambda: models.squeezenet1_1(weights=None),
}

TRACE_TYPES = ["nominal", "gpu_interference", "coupled_gpu_attack"]

METRIC_COLS = [
    "attack_recall",
    "false_alarm_rate",
    "detection_delay",
    "audit_rate",
    "mean_audit_cost_ms",
    "monitor_cost_mean_ms",
    "p99_latency_ms",
    "p999_latency_ms",
    "deadline_miss_ratio",
    "latency_inflation_mean_ms",
    "auroc",
    "auprc",
]


def load_configs() -> tuple[CIRCARuntimeConfig, BaselineConfig]:
    cfg = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    c = cfg["circa_rt"]
    b = cfg["baselines"]
    return (
        CIRCARuntimeConfig(
            window_size=int(c["window_size"]),
            feature_dim=int(c["feature_dim"]),
            low_quantile=float(c["low_quantile"]),
            high_quantile=float(c["high_quantile"]),
            monitor_cost_ms=float(c["monitor_cost_ms"]),
            heavy_audit_cost_ms=float(c["heavy_audit_cost_ms"]),
            bucket_capacity=float(c["bucket_capacity"]),
            replenish_rate=float(c["replenish_rate"]),
            seed=int(cfg["random_seed"]),
        ),
        BaselineConfig(
            window_size=int(b["window_size"]),
            monitor_cost_ms=float(b["monitor_cost_ms"]),
            audit_cost_ms=float(b["audit_cost_ms"]),
            periodic_audit_interval=int(b["periodic_audit_interval"]),
            random_audit_rate=float(b["random_audit_rate"]),
            seed=int(cfg["random_seed"]),
        ),
    )


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def latency_row(
    *,
    runtime: str,
    provider: str,
    model_name: str,
    seed: int,
    trace_type: str,
    latencies: np.ndarray,
    deadline_ms: float,
    device: str,
) -> dict[str, float | int | str]:
    return {
        "runtime": runtime,
        "provider": provider,
        "model": model_name,
        "seed": seed,
        "trace_type": trace_type,
        "device": device,
        "n": int(len(latencies)),
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "p999_ms": percentile(latencies, 0.999),
        "deadline_miss_ratio": float(np.mean(latencies > deadline_ms)),
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


def make_ort_session(path: Path, provider: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"{provider} is not available; available={available}")
    session = ort.InferenceSession(str(path), providers=[provider, "CPUExecutionProvider"])
    active = session.get_providers()
    if provider not in active:
        raise RuntimeError(f"{provider} did not initialize; active providers={active}; available={available}")
    return session


def run_torch_latency(
    *,
    model_name: str,
    seed: int,
    n: int,
    warmup: int,
    image_size: int,
    trace_type: str,
    interference_size: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"
    model = MODEL_BUILDERS[model_name]().eval().to(device)
    x = torch.randn(1, 3, image_size, image_size, device=device)
    interferer = torch.randn(interference_size, interference_size, device=device) if trace_type == "gpu_interference" else None
    latencies: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
            if interferer is not None:
                _ = interferer @ interferer
        torch.cuda.synchronize()
        for _ in range(n):
            start = time.perf_counter()
            _ = model(x)
            if interferer is not None:
                _ = interferer @ interferer
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(latencies, dtype=np.float64)


def run_ort_latency(
    *,
    session: ort.InferenceSession,
    seed: int,
    n: int,
    warmup: int,
    image_size: int,
    trace_type: str,
    interference_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    input_name = session.get_inputs()[0].name
    x = rng.normal(size=(1, 3, image_size, image_size)).astype(np.float32)
    interference_a = rng.normal(size=(interference_size, interference_size)).astype(np.float32)
    interference_b = rng.normal(size=(interference_size, interference_size)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: x})
        if trace_type == "gpu_interference":
            _ = interference_a @ interference_b
    latencies: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        if trace_type == "gpu_interference":
            _ = interference_a @ interference_b
        latencies.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(latencies, dtype=np.float64)


def write_latency_trace(
    *,
    latencies: np.ndarray,
    trace_type: str,
    seed: int,
    deadline_ms: float,
    platform: str,
    out_path: Path,
) -> None:
    trace = gpu_latency_trace(
        latencies_ms=latencies,
        trace_type=trace_type,
        seed=seed,
        deadline_ms=deadline_ms,
        platform=platform,
    )
    write_trace(trace, out_path)


def run_latency_collection(args: argparse.Namespace, out: Path) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this script requires an AutoDL GPU instance")

    unknown = sorted(set(args.models) - set(MODEL_BUILDERS))
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choices={sorted(MODEL_BUILDERS)}")

    table_dir = out / "tables"
    trace_dir = out / "traces"
    model_dir = out / "onnx"
    log_dir = out / "logs"
    for d in [table_dir, trace_dir, model_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    provider_rows: list[dict[str, str]] = []
    latency_rows: list[dict[str, float | int | str]] = []

    available_providers = ort.get_available_providers()
    provider_rows.append(
        {
            "runtime": "environment",
            "provider": "onnxruntime",
            "model": "",
            "status": "available",
            "detail": json.dumps(available_providers),
        }
    )

    for runtime in args.runtimes:
        for model_name in args.models:
            onnx_path = model_dir / f"{model_name}.onnx"
            if runtime.startswith("onnx"):
                export_onnx(model_name, args.image_size, onnx_path)

            if runtime == "torch_cuda":
                provider = "PyTorch-CUDA"
            elif runtime == "onnx_cuda":
                provider = "CUDAExecutionProvider"
            else:
                provider = "TensorrtExecutionProvider"
            session = None
            if runtime.startswith("onnx"):
                try:
                    session = make_ort_session(onnx_path, provider)
                    provider_rows.append(
                        {
                            "runtime": runtime,
                            "provider": provider,
                            "model": model_name,
                            "status": "ok",
                            "detail": json.dumps(session.get_providers()),
                        }
                    )
                except Exception as exc:
                    provider_rows.append(
                        {
                            "runtime": runtime,
                            "provider": provider,
                            "model": model_name,
                            "status": "failed",
                            "detail": repr(exc),
                        }
                    )
                    continue

            for seed in args.seeds:
                for trace_type in TRACE_TYPES:
                    platform = f"autodl_full_{runtime}_{model_name}"
                    trace_path = trace_dir / runtime / provider / model_name / f"seed{seed}" / f"{trace_type}.csv"
                    if runtime == "torch_cuda":
                        latencies = run_torch_latency(
                            model_name=model_name,
                            seed=seed,
                            n=args.n,
                            warmup=args.warmup,
                            image_size=args.image_size,
                            trace_type=trace_type,
                            interference_size=args.interference_size,
                        )
                    else:
                        assert session is not None
                        latencies = run_ort_latency(
                            session=session,
                            seed=seed,
                            n=args.n,
                            warmup=args.warmup,
                            image_size=args.image_size,
                            trace_type=trace_type,
                            interference_size=args.ort_interference_size,
                        )

                    write_latency_trace(
                        latencies=latencies,
                        trace_type=trace_type,
                        seed=seed,
                        deadline_ms=args.deadline_ms,
                        platform=platform,
                        out_path=trace_path,
                    )
                    latency_rows.append(
                        latency_row(
                            runtime=runtime,
                            provider=provider,
                            model_name=model_name,
                            seed=seed,
                            trace_type=trace_type,
                            latencies=latencies,
                            deadline_ms=args.deadline_ms,
                            device=torch.cuda.get_device_name(0),
                        )
                    )
                    print(f"wrote {trace_path}")

    pd.DataFrame(provider_rows).to_csv(table_dir / "provider_status.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(table_dir / "latency_full_table.csv", index=False)
    (log_dir / "autodl_full_env.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "torchvision": getattr(models, "__version__", "unknown"),
                "device": torch.cuda.get_device_name(0),
                "onnxruntime": ort.__version__,
                "available_providers": available_providers,
                "runtimes": args.runtimes,
                "models": args.models,
                "seeds": args.seeds,
                "n": args.n,
                "warmup": args.warmup,
                "image_size": args.image_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def trace_groups(trace_dir: Path) -> list[tuple[str, str, str, int, Path]]:
    groups: list[tuple[str, str, str, int, Path]] = []
    for seed_dir in sorted(trace_dir.glob("*/*/*/seed*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.replace("seed", ""))
        except ValueError:
            continue
        model_name = seed_dir.parent.name
        provider = seed_dir.parent.parent.name
        runtime = seed_dir.parent.parent.parent.name
        if (seed_dir / "nominal.csv").exists():
            groups.append((runtime, provider, model_name, seed, seed_dir))
    return groups


def run_monitoring_eval(out: Path) -> None:
    circa_cfg, baseline_cfg = load_configs()
    trace_dir = out / "traces"
    raw_dir = out / "raw"
    summary_dir = out / "summaries"
    table_dir = out / "tables"
    for d in [raw_dir, summary_dir, table_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | int | str]] = []
    for runtime, provider, model_name, seed, group_dir in trace_groups(trace_dir):
        calibration = read_trace(group_dir / "nominal.csv")
        model = fit_circa_rt(calibration, circa_cfg)
        for trace_path in sorted(group_dir.glob("*.csv")):
            trace_type = trace_path.stem
            raw = read_trace(trace_path)
            trace_key = f"{runtime}_{model_name}_seed{seed}_{trace_type}"

            circa = apply_circa_rt(model, raw)
            out_dir = raw_dir / runtime / provider / model_name / f"seed{seed}" / trace_type / "CIRCA-RT"
            out_dir.mkdir(parents=True, exist_ok=True)
            circa.to_csv(out_dir / "scored.csv", index=False)
            row = summarize_detection(circa, trace_name=trace_key).to_dict()
            row.update(
                {
                    "runtime": runtime,
                    "provider": provider,
                    "model": model_name,
                    "seed": seed,
                    "trace_type": trace_type,
                    "protocol": "autodl_full_nominal_calibration",
                }
            )
            summaries.append(row)

            for method, func in SPLIT_BASELINE_FUNCS.items():
                if method in RECENT_DEEP_BASELINE_FUNCS:
                    continue
                result = func(calibration, raw, baseline_cfg)
                out_dir = raw_dir / runtime / provider / model_name / f"seed{seed}" / trace_type / method
                out_dir.mkdir(parents=True, exist_ok=True)
                result.to_csv(out_dir / "scored.csv", index=False)
                row = summarize_detection(result, trace_name=trace_key).to_dict()
                row.update(
                    {
                        "runtime": runtime,
                        "provider": provider,
                        "model": model_name,
                        "seed": seed,
                        "trace_type": trace_type,
                        "protocol": "autodl_full_nominal_calibration",
                    }
                )
                summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)

    group_cols = ["runtime", "provider", "model", "method"]
    main = mean_table(summary, group_cols, METRIC_COLS).sort_values(
        ["runtime", "model", "attack_recall", "p99_latency_ms"],
        ascending=[True, True, False, True],
    )
    main.to_csv(table_dir / "monitoring_main_table.csv", index=False)
    bootstrap_ci_table(summary, group_cols, METRIC_COLS).to_csv(table_dir / "monitoring_main_table_ci.csv", index=False)

    scenario = mean_table(summary, ["runtime", "provider", "model", "trace_type", "method"], METRIC_COLS).sort_values(
        ["runtime", "model", "trace_type", "attack_recall"],
        ascending=[True, True, True, False],
    )
    scenario.to_csv(table_dir / "monitoring_scenario_table.csv", index=False)
    print(main[main["method"].isin(["CIRCA-RT", "AlwaysAudit", "RFF-HSIC", "ConditionalRFF-HSIC"])].head(40).to_string(index=False))


def write_overviews(out: Path) -> None:
    table_dir = out / "tables"
    latency = pd.read_csv(table_dir / "latency_full_table.csv") if (table_dir / "latency_full_table.csv").exists() else pd.DataFrame()
    summary = pd.read_csv(table_dir / "monitoring_main_table.csv") if (table_dir / "monitoring_main_table.csv").exists() else pd.DataFrame()
    if not latency.empty:
        overview = latency.sort_values(["runtime", "model", "seed", "trace_type"]).copy()
        overview.to_csv(table_dir / "autodl_full_latency_overview.csv", index=False)
    if not summary.empty:
        key_methods = ["CIRCA-RT", "RFF-HSIC", "ConditionalRFF-HSIC", "ContextAwareConformal", "AlwaysAudit"]
        overview = summary[summary["method"].isin(key_methods)].copy()
        keep = [
            "runtime",
            "provider",
            "model",
            "method",
            "attack_recall",
            "false_alarm_rate",
            "audit_rate",
            "p99_latency_ms",
            "p999_latency_ms",
            "deadline_miss_ratio",
            "latency_inflation_mean_ms",
            "auroc",
            "auprc",
        ]
        overview = overview[[c for c in keep if c in overview.columns]]
        overview.sort_values(["runtime", "model", "method"], inplace=True)
        overview.to_csv(table_dir / "autodl_full_monitoring_overview.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results_autodl_full"))
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenet_v2", "squeezenet1_1"])
    parser.add_argument("--runtimes", nargs="+", default=["torch_cuda", "onnx_cuda", "onnx_tensorrt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--deadline-ms", type=float, default=12.0)
    parser.add_argument("--interference-size", type=int, default=512)
    parser.add_argument("--ort-interference-size", type=int, default=384)
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_collection:
        run_latency_collection(args, out)
    if not args.skip_eval:
        run_monitoring_eval(out)
    write_overviews(out)
    print(f"wrote AutoDL full results: {out}")


if __name__ == "__main__":
    main()
