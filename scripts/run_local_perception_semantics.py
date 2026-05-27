from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.baselines import SPLIT_BASELINE_FUNCS, BaselineConfig
from circa_rt.core import CIRCARuntimeConfig, apply_circa_rt, fit_circa_rt
from circa_rt.metrics import summarize_detection
from circa_rt.perception_semantics import (
    add_perception_semantic_residuals,
    apply_frame_perturbation,
    demo_frame,
    frame_feature_row,
    heavy_audit_score,
    list_frame_paths,
    load_image,
    measure_heavy_audit,
)
from circa_rt.plots import plot_metric_bars, plot_trace_scores
from circa_rt.schema import write_trace


SCENARIOS = [
    "nominal",
    "mode_shift",
    "semantic_corruption",
    "timing_interference",
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


def load_configs() -> tuple[dict, CIRCARuntimeConfig, BaselineConfig]:
    raw = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    c = raw["circa_rt"]
    b = raw["baselines"]
    return raw, CIRCARuntimeConfig(
        window_size=int(c["window_size"]),
        feature_dim=int(c["feature_dim"]),
        low_quantile=float(c["low_quantile"]),
        high_quantile=float(c["high_quantile"]),
        monitor_cost_ms=float(c["monitor_cost_ms"]),
        heavy_audit_cost_ms=float(c["heavy_audit_cost_ms"]),
        bucket_capacity=float(c["bucket_capacity"]),
        replenish_rate=float(c["replenish_rate"]),
        seed=int(raw["random_seed"]),
    ), BaselineConfig(
        window_size=int(b["window_size"]),
        monitor_cost_ms=float(b["monitor_cost_ms"]),
        audit_cost_ms=float(b["audit_cost_ms"]),
        periodic_audit_interval=int(b["periodic_audit_interval"]),
        random_audit_rate=float(b["random_audit_rate"]),
        seed=int(raw["random_seed"]),
    )


def attack_mask(n: int, rng: np.random.Generator) -> np.ndarray:
    label = np.zeros(n, dtype=np.int64)
    length = max(16, int(round(n * 0.10)))
    centers = np.linspace(int(n * 0.28), int(n * 0.78), 3, dtype=int)
    for center in centers:
        start = int(np.clip(center + int(rng.integers(-8, 9)), 0, max(n - length, 0)))
        label[start : start + length] = 1
    return label


def load_frame_bank(frame_dir: str | None, *, n: int, seed: int) -> tuple[list[np.ndarray], str]:
    if frame_dir:
        paths = list_frame_paths(frame_dir)
        frames = [load_image(p) for p in paths]
        return frames, "real_frame_dir"
    bank_size = max(n, 180)
    frames = [demo_frame(i, seed=seed) for i in range(bank_size)]
    return frames, "demo_generated_frames"


def build_trace(
    frames: list[np.ndarray],
    *,
    scenario: str,
    seed: int,
    n: int,
    period_ms: float,
    deadline_ms: float,
    sleep_ms: float,
    reference_fraction: float,
    frame_source: str,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choices={SCENARIOS}")

    rng = np.random.default_rng(seed + 17 * SCENARIOS.index(scenario))
    label = attack_mask(n, rng) if scenario not in {"nominal", "mode_shift"} else np.zeros(n, dtype=np.int64)
    rows: list[dict[str, object]] = []
    used_frames: list[np.ndarray] = []
    previous_frame: np.ndarray | None = None
    previous_latency = 0.0
    queue = 0.0
    reference_count = max(16, int(round(n * reference_fraction)))

    for i in range(n):
        active = bool(label[i])
        source_idx = (i + 11 * seed) % len(frames)
        if scenario == "stale_replay_attack" and active:
            source_idx = max(0, source_idx - max(3, n // 20)) % len(frames)
        base_frame = frames[source_idx]
        perturb_scenario = "nominal" if scenario == "stale_replay_attack" else scenario
        transformed = apply_frame_perturbation(
            base_frame,
            scenario=perturb_scenario,
            active=active,
            step=i,
            rng=rng,
        )

        start = time.perf_counter()
        if active and scenario in {"timing_interference", "coupled_semantic_timing_attack"}:
            time.sleep(max(0.0, sleep_ms) / 1000.0)
        features = frame_feature_row(transformed, previous_frame)
        processing_latency_ms = (time.perf_counter() - start) * 1000.0

        base_latency = (
            6.0
            + processing_latency_ms
            + 6.5 * float(features["semantic_motion_l1"])
            + 2.0 * float(features["semantic_highfreq_mean"])
            + 1.2 * abs(float(features["semantic_entropy"]) - 0.60)
            + float(rng.normal(scale=0.20))
        )
        base_latency = max(0.5, base_latency)
        jitter = abs(base_latency - previous_latency) if i else 0.0
        cpu = np.clip(
            0.18
            + 0.70 * float(features["semantic_edge_mean"])
            + 0.35 * float(features["semantic_motion_l1"])
            + (0.12 if active and scenario in {"timing_interference", "coupled_semantic_timing_attack"} else 0.0)
            + rng.normal(scale=0.015),
            0.02,
            0.98,
        )
        gpu = np.clip(
            0.10
            + 0.65 * float(features["semantic_highfreq_mean"])
            + (0.10 if active and scenario == "coupled_semantic_timing_attack" else 0.0)
            + rng.normal(scale=0.015),
            0.01,
            0.98,
        )
        net_delay = max(0.0, 0.8 + 0.12 * processing_latency_ms + 1.5 * float(features["semantic_motion_l1"]) + rng.normal(scale=0.05))
        queue = max(0.0, 0.72 * queue + base_latency / max(period_ms, 1.0) - 0.52 + float(rng.normal(scale=0.035)))
        message_age = max(period_ms, period_ms + net_delay + base_latency + 0.55 * queue)
        mode = "vision_shift" if scenario == "mode_shift" else "vision_nominal"
        attack_type = scenario if active else "benign"
        row: dict[str, object] = {
            "run_id": f"local_perception_{scenario}_seed{seed}",
            "platform": "local_workstation_perception_semantics",
            "timestamp_ms": float(i) * float(period_ms),
            "seq": i,
            "mode": mode,
            "attack_type": attack_type,
            "label": int(label[i]),
            "semantic_residual": 0.0,
            "timing_residual_ms": 0.0,
            "context_cpu_util": float(cpu),
            "context_gpu_util": float(gpu),
            "context_net_delay_ms": float(net_delay),
            "context_jitter_ms": float(jitter),
            "context_queue_depth": float(queue),
            "context_message_age_ms": float(message_age),
            "baseline_latency_ms": float(base_latency),
            "e2e_latency_ms": float(base_latency),
            "deadline_ms": float(deadline_ms),
            "deadline_miss": int(base_latency > deadline_ms),
            "method": "raw",
            "alarm": 0,
            "audit": 0,
            "audit_cost_ms": 0.0,
            "monitor_cost_ms": 0.0,
            "frame_source": frame_source,
            "frame_index": int(source_idx),
            "processing_latency_ms": float(processing_latency_ms),
            "reference_count": int(reference_count),
        }
        row.update(features)
        rows.append(row)
        used_frames.append(transformed)
        previous_frame = transformed
        previous_latency = base_latency

    df = pd.DataFrame(rows)
    df = add_perception_semantic_residuals(df, reference_count=reference_count)
    timing_center = float(np.median(df["baseline_latency_ms"].iloc[:reference_count].to_numpy(dtype=float)))
    df["timing_residual_ms"] = df["baseline_latency_ms"].to_numpy(dtype=float) - timing_center
    df["e2e_latency_ms"] = df["baseline_latency_ms"]
    df["deadline_miss"] = (df["e2e_latency_ms"] > df["deadline_ms"]).astype(int)
    return df, used_frames


def annotate_real_audit_cost(scored: pd.DataFrame, frames: list[np.ndarray], *, reference_count: int) -> pd.DataFrame:
    out = scored.copy()
    ref_scores = []
    for i in range(min(reference_count, len(frames))):
        previous = frames[i - 1] if i > 0 else None
        ref_scores.append(heavy_audit_score(frames[i], previous))
    threshold = float(np.quantile(ref_scores, 0.95)) if ref_scores else float("inf")

    out["audit_model"] = "local_fft_temporal_consistency"
    out["audit_runtime"] = "numpy_cpu"
    out["audit_score"] = np.nan
    out["audit_threshold"] = threshold
    out["audit_latency_ms"] = 0.0
    out["audit_result"] = 0
    out["audit_disagreement"] = 0
    out["real_audit"] = True

    audit = out["audit"].to_numpy(dtype=int)
    audit_cost = np.zeros(len(out), dtype=np.float64)
    for i, flag in enumerate(audit):
        if flag != 1:
            continue
        previous = frames[i - 1] if i > 0 else None
        score, latency_ms = measure_heavy_audit(frames[i], previous)
        result = int(score >= threshold)
        out.at[i, "audit_score"] = score
        out.at[i, "audit_latency_ms"] = latency_ms
        out.at[i, "audit_result"] = result
        out.at[i, "audit_disagreement"] = int(result != int(out.at[i, "alarm"]))
        audit_cost[i] = latency_ms

    out["audit_cost_ms"] = audit_cost
    out["e2e_latency_ms"] = (
        out["baseline_latency_ms"].to_numpy(dtype=float)
        + out["monitor_cost_ms"].to_numpy(dtype=float)
        + out["audit_cost_ms"].to_numpy(dtype=float)
    )
    out["deadline_miss"] = (out["e2e_latency_ms"] > out["deadline_ms"]).astype(int)
    return out


def evaluate_seed(
    traces: dict[str, pd.DataFrame],
    frames_by_scenario: dict[str, list[np.ndarray]],
    *,
    seed: int,
    circa_cfg: CIRCARuntimeConfig,
    baseline_cfg: BaselineConfig,
    methods: list[str],
    raw_dir: Path,
) -> list[dict[str, object]]:
    calibration = pd.concat([traces["nominal"], traces["mode_shift"]], ignore_index=True)
    model = fit_circa_rt(calibration, circa_cfg)
    summaries: list[dict[str, object]] = []

    for scenario, raw in traces.items():
        reference_count = int(raw["reference_count"].iloc[0]) if "reference_count" in raw.columns else max(16, len(raw) // 5)
        selected: dict[str, pd.DataFrame] = {}
        if "CIRCA-RT" in methods:
            selected["CIRCA-RT"] = apply_circa_rt(model, raw, method="CIRCA-RT")
        for method in methods:
            if method == "CIRCA-RT":
                continue
            func = SPLIT_BASELINE_FUNCS.get(method)
            if func is None:
                raise ValueError(f"unknown method {method!r}; choices=['CIRCA-RT'] + {sorted(SPLIT_BASELINE_FUNCS)}")
            selected[method] = func(calibration, raw, baseline_cfg)

        for method, scored in selected.items():
            scored = annotate_real_audit_cost(scored, frames_by_scenario[scenario], reference_count=reference_count)
            out_dir = raw_dir / f"seed{seed}" / scenario / method
            out_dir.mkdir(parents=True, exist_ok=True)
            scored.to_csv(out_dir / "scored.csv", index=False)
            row = summarize_detection(scored, trace_name=f"local_perception_seed{seed}_{scenario}").to_dict()
            row.update(
                {
                    "seed": seed,
                    "trace_type": scenario,
                    "protocol": "local_real_semantic_feature_benign_calibration",
                    "real_audit": True,
                    "semantic_source": str(raw["semantic_source"].iloc[0]) if "semantic_source" in raw.columns else "",
                    "frame_source": str(raw["frame_source"].iloc[0]) if "frame_source" in raw.columns else "",
                }
            )
            summaries.append(row)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", default="", help="Directory of real image frames. If omitted, deterministic demo frames are used.")
    parser.add_argument("--out-dir", default=str(ROOT / "results_local_perception_semantics"))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--include-deep", action="store_true", help="Append CATCH/DCdetector/TranAD-style adapters.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--period-ms", type=float, default=33.3)
    parser.add_argument("--deadline-ms", type=float, default=25.0)
    parser.add_argument("--sleep-ms", type=float, default=4.0)
    parser.add_argument("--reference-fraction", type=float, default=0.20)
    args = parser.parse_args()

    _, circa_cfg, baseline_cfg = load_configs()
    out = Path(args.out_dir)
    trace_dir = out / "traces"
    raw_dir = out / "raw"
    summary_dir = out / "summaries"
    table_dir = out / "tables"
    figure_dir = out / "figures"
    for directory in [trace_dir, raw_dir, summary_dir, table_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    methods = list(dict.fromkeys(args.methods + (["CATCH", "DCdetector", "TranAD"] if args.include_deep else [])))
    all_rows: list[dict[str, object]] = []
    representative_raw: pd.DataFrame | None = None
    representative_scored: pd.DataFrame | None = None

    for seed in args.seeds:
        frames, frame_source = load_frame_bank(args.frame_dir or None, n=int(args.n), seed=int(seed))
        traces: dict[str, pd.DataFrame] = {}
        frames_by_scenario: dict[str, list[np.ndarray]] = {}
        for scenario in args.scenarios:
            trace, used_frames = build_trace(
                frames,
                scenario=scenario,
                seed=int(seed),
                n=int(args.n),
                period_ms=float(args.period_ms),
                deadline_ms=float(args.deadline_ms),
                sleep_ms=float(args.sleep_ms),
                reference_fraction=float(args.reference_fraction),
                frame_source=frame_source,
            )
            traces[scenario] = trace
            frames_by_scenario[scenario] = used_frames
            write_trace(trace, trace_dir / f"seed{seed}" / f"{scenario}.csv")

        missing_calibration = {"nominal", "mode_shift"} - set(traces)
        if missing_calibration:
            raise SystemExit(f"scenarios must include nominal and mode_shift for calibration; missing={sorted(missing_calibration)}")

        seed_rows = evaluate_seed(
            traces,
            frames_by_scenario,
            seed=int(seed),
            circa_cfg=replace(circa_cfg, seed=int(seed)),
            baseline_cfg=replace(baseline_cfg, seed=int(seed)),
            methods=methods,
            raw_dir=raw_dir,
        )
        all_rows.extend(seed_rows)

        if representative_raw is None and "coupled_semantic_timing_attack" in traces:
            representative_raw = traces["coupled_semantic_timing_attack"]
            scored_path = raw_dir / f"seed{seed}" / "coupled_semantic_timing_attack" / "CIRCA-RT" / "scored.csv"
            if scored_path.exists():
                representative_scored = pd.read_csv(scored_path)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)

    main = mean_table(summary, ["method"], METRIC_COLS).sort_values(["attack_recall", "p99_latency_ms"], ascending=[False, True])
    main.to_csv(table_dir / "local_perception_main_table.csv", index=False)
    bootstrap_ci_table(summary, ["method"], METRIC_COLS).to_csv(table_dir / "local_perception_main_table_ci.csv", index=False)

    scenario = mean_table(summary, ["trace_type", "method"], METRIC_COLS).sort_values(
        ["trace_type", "attack_recall", "p99_latency_ms"],
        ascending=[True, False, True],
    )
    scenario.to_csv(table_dir / "local_perception_scenario_table.csv", index=False)
    bootstrap_ci_table(summary, ["trace_type", "method"], METRIC_COLS).to_csv(table_dir / "local_perception_scenario_table_ci.csv", index=False)

    diagnostics = []
    for trace_path in sorted(trace_dir.glob("seed*/*.csv")):
        df = pd.read_csv(trace_path)
        attack = df["label"].to_numpy(dtype=int) == 1
        benign = ~attack
        diagnostics.append(
            {
                "trace": str(trace_path.relative_to(out)),
                "trace_type": trace_path.stem,
                "seed": trace_path.parent.name.replace("seed", ""),
                "frame_source": str(df["frame_source"].iloc[0]) if "frame_source" in df.columns else "",
                "semantic_source": str(df["semantic_source"].iloc[0]) if "semantic_source" in df.columns else "",
                "n": int(len(df)),
                "attack_fraction": float(np.mean(attack)),
                "semantic_residual_benign_mean": float(np.mean(df.loc[benign, "semantic_residual"])) if np.any(benign) else float("nan"),
                "semantic_residual_attack_mean": float(np.mean(df.loc[attack, "semantic_residual"])) if np.any(attack) else float("nan"),
                "timing_residual_benign_mean": float(np.mean(df.loc[benign, "timing_residual_ms"])) if np.any(benign) else float("nan"),
                "timing_residual_attack_mean": float(np.mean(df.loc[attack, "timing_residual_ms"])) if np.any(attack) else float("nan"),
            }
        )
    pd.DataFrame(diagnostics).to_csv(table_dir / "semantic_timing_diagnostics.csv", index=False)

    plot_metric_bars(summary, figure_dir)
    if representative_raw is not None and representative_scored is not None:
        plot_trace_scores(representative_raw, representative_scored, figure_dir / "local_perception_circa_rt_score_timeline.png")

    print(f"wrote traces: {trace_dir}")
    print(f"wrote summary: {summary_dir / 'summary_all.csv'}")
    print(f"wrote main table: {table_dir / 'local_perception_main_table.csv'}")
    print(main.to_string(index=False))


if __name__ == "__main__":
    main()
