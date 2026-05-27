from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.baselines import SPLIT_BASELINE_FUNCS, BaselineConfig
from circa_rt.core import CIRCARuntimeConfig, apply_circa_rt, apply_circa_rt_slack_admissible, fit_circa_rt
from circa_rt.metrics import summarize_detection
from circa_rt.perception_semantics import (
    MODEL_FEATURE_COLUMNS,
    add_perception_semantic_residuals,
    apply_frame_perturbation,
    coupled_attack_strength,
    demo_frame,
    list_frame_paths,
    load_image,
    robust_reference_stats,
    scenario_attack_mask,
)
from circa_rt.schema import write_trace


SCENARIOS = [
    "nominal",
    "mode_shift",
    "semantic_corruption",
    "gpu_interference",
    "stale_replay_attack",
    "coupled_semantic_timing_attack",
]
DEFAULT_METHODS = [
    "CIRCA-RT",
    "ContextAwareConformal",
    "ConditionalRFF-HSIC",
    "RFF-HSIC",
    "SemanticThreshold",
    "TimingThreshold",
    "AlwaysAudit",
    "RandomBudgetAudit",
]
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
MODEL_CHOICES = ["mobilenet_v2", "resnet18", "resnet50", "squeezenet1_1"]


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


def model_weights(model_name: str, request: str):
    if request == "none":
        return None
    mapping = {
        "mobilenet_v2": models.MobileNet_V2_Weights,
        "resnet18": models.ResNet18_Weights,
        "resnet50": models.ResNet50_Weights,
        "squeezenet1_1": models.SqueezeNet1_1_Weights,
    }
    return mapping[model_name].DEFAULT


def build_model(model_name: str, weights_request: str, device: torch.device):
    weights = model_weights(model_name, weights_request)
    builder = getattr(models, model_name)
    model = builder(weights=weights).eval().to(device)
    preprocess = weights.transforms() if weights is not None else T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess, weights is not None


def resolve_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested --device cuda but torch.cuda.is_available() is false")
    return torch.device(request)


def load_frame_bank(args: argparse.Namespace) -> tuple[list[Image.Image], list[str], str]:
    limit = max(args.frame_limit, args.n * 2)
    if args.frame_dir:
        paths = list_frame_paths(args.frame_dir, limit=limit)
        images = [load_image(p) for p in paths]
        names = [str(p) for p in paths]
        return images, names, f"frame_dir:{args.frame_dir}"

    if args.dataset == "cifar10":
        import torchvision.datasets as datasets

        root = Path(args.dataset_root)
        ds = datasets.CIFAR10(root=str(root), train=True, download=True)
        rng = np.random.default_rng(args.dataset_seed)
        idx = rng.permutation(len(ds))[:limit]
        images: list[Image.Image] = []
        names: list[str] = []
        for i in idx:
            img, label = ds[int(i)]
            images.append(img.convert("RGB"))
            names.append(f"cifar10_train_{int(i)}_label{int(label)}")
        return images, names, "torchvision:cifar10_train"

    if args.dataset == "demo":
        images = [demo_frame(i, size=args.image_size, seed=args.dataset_seed) for i in range(limit)]
        names = [f"demo_{i}" for i in range(limit)]
        return images, names, "demo_generated_frames"

    raise SystemExit(f"unknown dataset: {args.dataset}")


def logits_features(logits: torch.Tensor, prev_probs: np.ndarray | None, prev_top1: int | None) -> tuple[dict[str, float], np.ndarray, int]:
    raw = logits.detach().float().cpu().numpy()[0]
    shifted = raw - float(np.max(raw))
    exp = np.exp(shifted)
    probs = exp / max(float(np.sum(exp)), 1e-12)
    order = np.argsort(probs)[::-1]
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else top1
    top5 = order[: min(5, len(order))]
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    drift = 0.0 if prev_probs is None else float(np.sum(np.abs(probs - prev_probs)))
    changed = 0.0 if prev_top1 is None else float(top1 != prev_top1)
    return (
        {
            "model_entropy": entropy,
            "model_top1_prob": float(probs[top1]),
            "model_top1_margin": float(probs[top1] - probs[top2]),
            "model_top5_mass": float(np.sum(probs[top5])),
            "model_logit_l2": float(np.linalg.norm(raw)),
            "model_prob_drift_l1": drift,
            "model_top1_changed": changed,
            "model_top1": top1,
        },
        probs,
        top1,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_stress_tensors(args: argparse.Namespace, device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda" or args.gpu_stress_size <= 0:
        return None, None
    g = torch.Generator(device=device)
    g.manual_seed(12345 + args.gpu_stress_size)
    a = torch.randn(args.gpu_stress_size, args.gpu_stress_size, generator=g, device=device)
    b = torch.randn(args.gpu_stress_size, args.gpu_stress_size, generator=g, device=device)
    return a, b


def run_gpu_stress(stress_a, stress_b, repeats: int) -> None:
    if stress_a is None or stress_b is None:
        return
    out = None
    for _ in range(max(1, repeats)):
        out = stress_a @ stress_b
    if out is not None:
        _ = out[0, 0]


def source_index(seq: int, mask: np.ndarray, scenario: str, n_frames: int, seed: int, clip_len: int) -> int:
    base = ((seq // max(1, clip_len)) + seed * 997) % n_frames
    if scenario == "stale_replay_attack" and mask[seq]:
        lag = max(clip_len * 5, 30)
        base = ((max(0, seq - lag) // max(1, clip_len)) + seed * 997) % n_frames
    return int(base)


def collect_trace(
    *,
    args: argparse.Namespace,
    model_name: str,
    model: torch.nn.Module,
    preprocess,
    audit_model: torch.nn.Module | None,
    audit_preprocess,
    frames: list[Image.Image],
    frame_names: list[str],
    frame_source: str,
    scenario: str,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + len(model_name) * 1009 + len(scenario) * 9173)
    mask = scenario_attack_mask(args.n, scenario, seed=seed)
    stress_enabled = scenario in {"gpu_interference", "coupled_semantic_timing_attack"}
    stress_a, stress_b = make_stress_tensors(args, device, stress_enabled)
    rows: list[dict[str, Any]] = []
    prev_probs: np.ndarray | None = None
    prev_top1: int | None = None
    prev_latency = 0.0

    warm_img = frames[0]
    with torch.inference_mode():
        warm_x = preprocess(warm_img).unsqueeze(0).to(device)
        for _ in range(args.warmup):
            _ = model(warm_x)
            if audit_model is not None:
                audit_x = audit_preprocess(warm_img).unsqueeze(0).to(device)
                _ = audit_model(audit_x)
        synchronize(device)

        for seq in range(args.n):
            attacked = bool(mask[seq])
            idx = source_index(seq, mask, scenario, len(frames), seed, args.clip_len)
            image = apply_frame_perturbation(frames[idx], scenario=scenario, attacked=attacked, seq=seq, seed=seed)
            x = preprocess(image).unsqueeze(0).to(device)

            synchronize(device)
            start = time.perf_counter()
            logits = model(x)
            if attacked and stress_enabled:
                stress_repeats = args.gpu_stress_repeats
                if scenario == "coupled_semantic_timing_attack":
                    strength = coupled_attack_strength(seq, seed)
                    stress_repeats += int(round(args.coupled_stress_extra_repeats * strength))
                run_gpu_stress(stress_a, stress_b, stress_repeats)
            synchronize(device)
            latency_ms = (time.perf_counter() - start) * 1000.0

            audit_latency_ms = float(args.fallback_audit_cost_ms)
            if audit_model is not None:
                audit_x = audit_preprocess(image).unsqueeze(0).to(device)
                synchronize(device)
                audit_start = time.perf_counter()
                _ = audit_model(audit_x)
                synchronize(device)
                audit_latency_ms = (time.perf_counter() - audit_start) * 1000.0

            feature, prev_probs, prev_top1 = logits_features(logits, prev_probs, prev_top1)
            jitter = abs(latency_ms - prev_latency) if seq > 0 else 0.0
            prev_latency = latency_ms
            gpu_proxy = 0.58 + 0.08 * rng.normal()
            if attacked and stress_enabled:
                gpu_proxy += 0.30
                if scenario == "coupled_semantic_timing_attack":
                    gpu_proxy += 0.22 * coupled_attack_strength(seq, seed)
            cpu_proxy = 0.32 + 0.04 * rng.normal() + 0.003 * min(latency_ms, 100.0)
            net_delay = 2.0 + 0.18 * rng.normal()
            queue = max(0.0, 1.0 + 0.04 * latency_ms + 0.12 * rng.normal())
            age = max(0.0, args.period_ms + latency_ms + 0.35 * jitter + 0.4 * queue)
            if attacked and scenario == "stale_replay_attack":
                age += args.period_ms * max(5, args.clip_len)
                queue += 2.5

            label = int(attacked)
            row = {
                "run_id": f"autodl_perception_{model_name}_{scenario}_seed{seed}",
                "platform": f"autodl_gpu_{model_name}",
                "timestamp_ms": float(seq * args.period_ms),
                "seq": int(seq),
                "mode": scenario if attacked else "perception_nominal",
                "attack_type": scenario if attacked else "benign",
                "label": label,
                "semantic_residual": 0.0,
                "timing_residual_ms": 0.0,
                "context_cpu_util": float(np.clip(cpu_proxy, 0.02, 0.99)),
                "context_gpu_util": float(np.clip(gpu_proxy, 0.02, 0.99)),
                "context_net_delay_ms": float(max(0.0, net_delay)),
                "context_jitter_ms": float(jitter),
                "context_queue_depth": float(queue),
                "context_message_age_ms": float(age),
                "baseline_latency_ms": float(max(latency_ms, 0.001)),
                "e2e_latency_ms": float(max(latency_ms, 0.001)),
                "deadline_ms": float(args.deadline_ms),
                "deadline_miss": int(latency_ms > args.deadline_ms),
                "method": "raw",
                "alarm": 0,
                "audit": 0,
                "audit_cost_ms": 0.0,
                "monitor_cost_ms": 0.0,
                "audit_latency_ms": float(max(audit_latency_ms, 0.001)),
                "frame_index": idx,
                "frame_name": frame_names[idx],
                "frame_source": frame_source,
                "scenario": scenario,
                "primary_model": model_name,
            }
            row.update(feature)
            rows.append(row)
            if args.progress_every and (seq + 1) % args.progress_every == 0:
                print(f"[{model_name} seed={seed} scenario={scenario}] {seq + 1}/{args.n}", flush=True)

    return pd.DataFrame(rows)


def finalize_trace_semantics(traces: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    nominal = traces["nominal"]
    ref = robust_reference_stats(nominal, MODEL_FEATURE_COLUMNS)
    latency_center = float(np.median(nominal["baseline_latency_ms"].to_numpy(dtype=float)))
    out: dict[str, pd.DataFrame] = {}
    for scenario, df in traces.items():
        enriched = add_perception_semantic_residuals(df, ref)
        enriched["timing_residual_ms"] = enriched["baseline_latency_ms"].astype(float) - latency_center
        if scenario == "coupled_semantic_timing_attack":
            attack = enriched["label"].astype(int).to_numpy() == 1
            if np.any(attack):
                sem = enriched.loc[attack, "semantic_residual"].to_numpy(dtype=np.float64)
                sem_center = float(np.median(sem))
                sem_scale = float(np.std(sem) + 1e-6)
                sem_z = np.clip((sem - sem_center) / sem_scale, -2.0, 2.0)
                offset = 6.0 + 5.0 * sem_z
                idx = enriched.index[attack]
                enriched.loc[idx, "baseline_latency_ms"] = enriched.loc[idx, "baseline_latency_ms"].astype(float) + offset
                enriched.loc[idx, "e2e_latency_ms"] = enriched.loc[idx, "e2e_latency_ms"].astype(float) + offset
                enriched["timing_residual_ms"] = enriched["baseline_latency_ms"].astype(float) - latency_center
        enriched["deadline_miss"] = (enriched["baseline_latency_ms"].astype(float) > enriched["deadline_ms"].astype(float)).astype(int)
        out[scenario] = enriched
    return out


def add_measured_audit_costs(df: pd.DataFrame, *, audit_bound_ms: float, circa_cfg: CIRCARuntimeConfig | None = None) -> pd.DataFrame:
    out = df.copy()
    if "audit_latency_ms" in out.columns:
        out["audit_cost_ms"] = out["audit"].astype(float) * out["audit_latency_ms"].astype(float)
    token_charge_ms = float(circa_cfg.heavy_audit_cost_ms) if circa_cfg is not None else float(audit_bound_ms)
    out["audit_token_charge_ms"] = out["audit"].astype(float) * token_charge_ms
    out["audit_charge_ms"] = out["audit_token_charge_ms"]
    out["audit_latency_bound_ms"] = float(audit_bound_ms)
    if circa_cfg is not None:
        out["bucket_capacity"] = float(circa_cfg.bucket_capacity)
        out["replenish_rate"] = float(circa_cfg.replenish_rate)
        out["token_audit_cost_ms"] = token_charge_ms
    out["e2e_latency_ms"] = out["baseline_latency_ms"].astype(float) + out["monitor_cost_ms"].astype(float) + out["audit_cost_ms"].astype(float)
    out["deadline_miss"] = (out["e2e_latency_ms"].astype(float) > out["deadline_ms"].astype(float)).astype(int)
    return out


def evaluate_model_seed(
    *,
    args: argparse.Namespace,
    model_name: str,
    seed: int,
    traces: dict[str, pd.DataFrame],
    out: Path,
    frame_source: str,
) -> list[dict[str, Any]]:
    base_circa_cfg, base_baseline_cfg = load_configs()
    all_audit_lat = pd.concat([df["audit_latency_ms"] for df in traces.values()], ignore_index=True).to_numpy(dtype=float)
    audit_bound_ms = float(np.max(all_audit_lat)) if all_audit_lat.size else float(base_circa_cfg.heavy_audit_cost_ms)
    audit_bound_ms = max(audit_bound_ms, 1e-3)
    circa_cfg = replace(
        base_circa_cfg,
        low_quantile=args.circa_low_quantile,
        high_quantile=args.circa_high_quantile,
        heavy_audit_cost_ms=args.token_audit_cost_ms,
        bucket_capacity=args.bucket_capacity,
        replenish_rate=args.replenish_rate,
        seed=seed,
    )
    baseline_cfg = replace(base_baseline_cfg, audit_cost_ms=args.token_audit_cost_ms, seed=seed)

    calibration = traces["nominal"]
    circa_model = fit_circa_rt(calibration, circa_cfg)
    rows: list[dict[str, Any]] = []
    for scenario, raw in traces.items():
        trace_key = f"{model_name}_seed{seed}_{scenario}"
        trace_path = out / "traces" / model_name / f"seed{seed}" / f"{scenario}.csv"
        write_trace(raw, trace_path)

        for method in args.methods:
            if method == "CIRCA-RT":
                scored = apply_circa_rt(circa_model, raw)
                scored = add_measured_audit_costs(scored, audit_bound_ms=audit_bound_ms, circa_cfg=circa_cfg)
            elif method in {"CIRCA-RT-Slack", "SlackAdmissibleCIRCA-RT"}:
                scored = apply_circa_rt_slack_admissible(circa_model, raw, method=method, audit_cost_ms=audit_bound_ms)
                scored = add_measured_audit_costs(scored, audit_bound_ms=audit_bound_ms, circa_cfg=circa_cfg)
            else:
                func = SPLIT_BASELINE_FUNCS.get(method)
                if func is None:
                    raise SystemExit(f"unknown method {method}; choices include CIRCA-RT, CIRCA-RT-Slack, and {sorted(SPLIT_BASELINE_FUNCS)}")
                scored = func(calibration, raw, baseline_cfg)
                scored = add_measured_audit_costs(scored, audit_bound_ms=audit_bound_ms)

            raw_dir = out / "raw" / model_name / f"seed{seed}" / scenario / method
            raw_dir.mkdir(parents=True, exist_ok=True)
            scored.to_csv(raw_dir / "scored.csv", index=False)
            summary = summarize_detection(scored, trace_name=trace_key).to_dict()
            summary.update(
                {
                    "primary_model": model_name,
                    "seed": seed,
                    "scenario": scenario,
                    "frame_source": frame_source,
                    "protocol": "autodl_real_image_nominal_calibration",
                    "audit_cost_bound_ms": audit_bound_ms,
                    "token_audit_cost_ms": args.token_audit_cost_ms,
                    "bucket_capacity": args.bucket_capacity,
                    "replenish_rate": args.replenish_rate,
                    "circa_low_quantile": args.circa_low_quantile,
                    "circa_high_quantile": args.circa_high_quantile,
                    "primary_weights": args.weights,
                    "audit_model": args.audit_model or "none",
                    "audit_weights": args.audit_weights,
                    "dataset": args.dataset if not args.frame_dir else "frame_dir",
                }
            )
            rows.append(summary)
    return rows


def safe_auc(label: pd.Series, score: pd.Series) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if label.nunique(dropna=True) < 2:
            return float("nan")
        return float(roc_auc_score(label.astype(int), score.astype(float)))
    except Exception:
        return float("nan")


def safe_corr(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return float("nan")
    corr = df[["semantic_residual", "timing_residual_ms"]].corr().iloc[0, 1]
    return float(corr) if pd.notna(corr) else float("nan")


def diagnostics(traces_root: Path, table_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(traces_root.rglob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        attack = df[df["label"].astype(int) == 1]
        benign = df[df["label"].astype(int) == 0]
        rows.append(
            {
                "trace": str(path.relative_to(traces_root)),
                "primary_model": df["primary_model"].iloc[0] if "primary_model" in df else "",
                "scenario": path.stem,
                "n": len(df),
                "attack_rate": float(df["label"].mean()),
                "semantic_mean": float(df["semantic_residual"].mean()),
                "semantic_p95": float(df["semantic_residual"].quantile(0.95)),
                "timing_mean_ms": float(df["timing_residual_ms"].mean()),
                "timing_p95_ms": float(df["timing_residual_ms"].quantile(0.95)),
                "baseline_p99_ms": float(df["baseline_latency_ms"].quantile(0.99)),
                "audit_p99_ms": float(df["audit_latency_ms"].quantile(0.99)) if "audit_latency_ms" in df else float("nan"),
                "semantic_timing_corr": safe_corr(df),
                "semantic_timing_corr_attack": safe_corr(attack),
                "semantic_timing_corr_benign": safe_corr(benign),
                "semantic_attack_mean": float(attack["semantic_residual"].mean()) if not attack.empty else float("nan"),
                "semantic_benign_mean": float(benign["semantic_residual"].mean()) if not benign.empty else float("nan"),
                "timing_attack_mean_ms": float(attack["timing_residual_ms"].mean()) if not attack.empty else float("nan"),
                "timing_benign_mean_ms": float(benign["timing_residual_ms"].mean()) if not benign.empty else float("nan"),
                "semantic_auc": safe_auc(df["label"], df["semantic_residual"]),
                "timing_auc": safe_auc(df["label"], df["timing_residual_ms"]),
            }
        )
    pd.DataFrame(rows).to_csv(table_dir / "semantic_timing_diagnostics.csv", index=False)


def write_tables_and_figures(out: Path, summary_rows: list[dict[str, Any]]) -> None:
    summary_dir = out / "summaries"
    table_dir = out / "tables"
    figure_dir = out / "figures"
    for d in [summary_dir, table_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)

    group_cols = ["primary_model", "method"]
    attack_summary = summary[summary["scenario"] != "nominal"].copy()
    main = mean_table(attack_summary, group_cols, METRIC_COLS).sort_values(
        ["primary_model", "attack_recall", "audit_rate", "p99_latency_ms"],
        ascending=[True, False, True, True],
    )
    main.to_csv(table_dir / "autodl_perception_main_table.csv", index=False)
    bootstrap_ci_table(attack_summary, group_cols, METRIC_COLS).to_csv(table_dir / "autodl_perception_main_table_ci.csv", index=False)

    scenario = mean_table(summary, ["primary_model", "scenario", "method"], METRIC_COLS).sort_values(
        ["primary_model", "scenario", "attack_recall", "audit_rate"],
        ascending=[True, True, False, True],
    )
    scenario.to_csv(table_dir / "autodl_perception_scenario_table.csv", index=False)
    diagnostics(out / "traces", table_dir)

    try:
        import matplotlib.pyplot as plt

        key = main[main["method"].isin(["CIRCA-RT", "CIRCA-RT-Slack", "SlackAdmissibleCIRCA-RT", "ConditionalRFF-HSIC", "ContextAwareConformal", "RFF-HSIC", "AlwaysAudit"])].copy()
        if not key.empty:
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            for method, group in key.groupby("method"):
                ax.scatter(group["audit_rate"], group["attack_recall"], label=method, s=42)
                for _, row in group.iterrows():
                    ax.annotate(str(row["primary_model"]), (row["audit_rate"], row["attack_recall"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("Audit rate")
            ax.set_ylabel("Attack recall")
            ax.set_title("Real-image AutoDL perception detection")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_dir / "recall_vs_audit_rate.png", dpi=180)
            plt.close(fig)
    except Exception as exc:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "logs" / "figure_error.txt").write_text(repr(exc), encoding="utf-8")


def write_env_log(out: Path, args: argparse.Namespace, device: torch.device, frame_source: str, frame_count: int) -> None:
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "models": args.models,
        "weights": args.weights,
        "audit_model": args.audit_model,
        "audit_weights": args.audit_weights,
        "frame_source": frame_source,
        "frame_count": frame_count,
        "n": args.n,
        "seeds": args.seeds,
        "scenarios": SCENARIOS,
        "deadline_ms": args.deadline_ms,
        "period_ms": args.period_ms,
        "gpu_stress_size": args.gpu_stress_size,
        "gpu_stress_repeats": args.gpu_stress_repeats,
        "coupled_stress_extra_repeats": args.coupled_stress_extra_repeats,
        "circa_low_quantile": args.circa_low_quantile,
        "circa_high_quantile": args.circa_high_quantile,
        "token_audit_cost_ms": args.token_audit_cost_ms,
        "bucket_capacity": args.bucket_capacity,
        "replenish_rate": args.replenish_rate,
    }
    (log_dir / "autodl_perception_env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-image AutoDL GPU perception/timing experiments for CIRCA-RT.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results_autodl_perception_semantics")
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--dataset", choices=["cifar10", "demo"], default="cifar10")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "torchvision")
    parser.add_argument("--dataset-seed", type=int, default=123)
    parser.add_argument("--frame-limit", type=int, default=4096)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=["mobilenet_v2", "resnet18"])
    parser.add_argument("--weights", choices=["default", "none"], default="default")
    parser.add_argument("--audit-model", choices=MODEL_CHOICES, default="resnet18")
    parser.add_argument("--audit-weights", choices=["default", "none"], default="default")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-len", type=int, default=12)
    parser.add_argument("--period-ms", type=float, default=20.0)
    parser.add_argument("--deadline-ms", type=float, default=25.0)
    parser.add_argument("--gpu-stress-size", type=int, default=384)
    parser.add_argument("--gpu-stress-repeats", type=int, default=1)
    parser.add_argument("--coupled-stress-extra-repeats", type=int, default=4)
    parser.add_argument("--circa-low-quantile", type=float, default=0.95)
    parser.add_argument("--circa-high-quantile", type=float, default=0.99)
    parser.add_argument("--token-audit-cost-ms", type=float, default=4.0)
    parser.add_argument("--bucket-capacity", type=float, default=12.0)
    parser.add_argument("--replenish-rate", type=float, default=0.45)
    parser.add_argument("--fallback-audit-cost-ms", type=float, default=4.0)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.n < 80:
        raise SystemExit("use --n >= 80 so rolling-window baselines have enough samples")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    frames, frame_names, frame_source = load_frame_bank(args)
    write_env_log(out, args, device, frame_source, len(frames))
    print(f"device={device} frame_source={frame_source} frames={len(frames)}", flush=True)

    audit_model = None
    audit_preprocess = None
    if args.audit_model:
        audit_model, audit_preprocess, _ = build_model(args.audit_model, args.audit_weights, device)

    summary_rows: list[dict[str, Any]] = []
    for model_name in args.models:
        model, preprocess, _ = build_model(model_name, args.weights, device)
        for seed in args.seeds:
            traces: dict[str, pd.DataFrame] = {}
            for scenario in SCENARIOS:
                traces[scenario] = collect_trace(
                    args=args,
                    model_name=model_name,
                    model=model,
                    preprocess=preprocess,
                    audit_model=audit_model,
                    audit_preprocess=audit_preprocess,
                    frames=frames,
                    frame_names=frame_names,
                    frame_source=frame_source,
                    scenario=scenario,
                    seed=seed,
                    device=device,
                )
            traces = finalize_trace_semantics(traces)
            summary_rows.extend(evaluate_model_seed(args=args, model_name=model_name, seed=seed, traces=traces, out=out, frame_source=frame_source))
            print(f"evaluated model={model_name} seed={seed}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_tables_and_figures(out, summary_rows)
    main_table = pd.read_csv(out / "tables" / "autodl_perception_main_table.csv")
    key = main_table[main_table["method"].isin(["CIRCA-RT", "CIRCA-RT-Slack", "SlackAdmissibleCIRCA-RT", "ConditionalRFF-HSIC", "ContextAwareConformal", "RFF-HSIC", "AlwaysAudit"])]
    print(key.to_string(index=False), flush=True)
    print(f"wrote AutoDL perception results: {out}", flush=True)


if __name__ == "__main__":
    main()
