from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SERVER_METHODS = [
    "Risk+DeferrableServer",
    "Risk+SporadicServer",
    "Risk+CBS",
    "Risk+DeferrableServer+Slack",
    "Risk+SporadicServer+Slack",
    "Risk+CBS+Slack",
]


@dataclass(frozen=True)
class TraceMeta:
    source_root: str
    pressure_label: str
    dataset: str
    deadline_name: str
    model: str
    seed: str
    scenario: str


def positive_float(series: pd.Series, default: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return float(default)
    return float(vals.iloc[0])


def quantile(values: pd.Series, q: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def parse_meta(path: Path) -> TraceMeta:
    parts = list(path.parts)
    source_root = next((p for p in parts if p.startswith("results_codex_")), "unknown")
    pressure_label = ""
    for p in parts:
        if p.startswith("level_"):
            pressure_label = p.strip("_")
            break

    dataset = "unknown"
    deadline_name = ""
    model = "unknown"
    seed = "unknown"
    scenario = "unknown"

    if "circa_slack_sensitivity" in parts:
        i = parts.index("circa_slack_sensitivity")
        if i + 1 < len(parts):
            dataset = parts[i + 1]
        if "raw" in parts[i:]:
            r = parts.index("raw", i)
            if r + 4 < len(parts):
                model, seed, scenario = parts[r + 1], parts[r + 2], parts[r + 3]
    elif "realframes" in parts:
        i = parts.index("realframes")
        if i + 1 < len(parts):
            token = parts[i + 1]
            if "_d" in token:
                dataset, deadline_name = token.split("_d", 1)
                deadline_name = "d" + deadline_name
            else:
                dataset = token
        if "raw" in parts[i:]:
            r = parts.index("raw", i)
            if r + 4 < len(parts):
                model, seed, scenario = parts[r + 1], parts[r + 2], parts[r + 3]
    elif "raw" in parts:
        r = parts.index("raw")
        if r + 4 < len(parts):
            model, seed, scenario = parts[r + 1], parts[r + 2], parts[r + 3]

    return TraceMeta(
        source_root=source_root,
        pressure_label=pressure_label,
        dataset=dataset,
        deadline_name=deadline_name,
        model=model,
        seed=seed,
        scenario=scenario,
    )


def find_inputs(input_root: Path, include_pressure: bool, include_ros2: bool) -> list[Path]:
    paths: list[Path] = []
    for dirpath, _, filenames in os.walk(input_root, topdown=True, onerror=lambda _: None):
        if "scored.csv" not in filenames:
            continue
        path = Path(dirpath) / "scored.csv"
        text = str(path)
        if "CIRCA-RT-Slack" not in text:
            continue
        if "results_codex_smoke_" in text:
            continue
        if "backend_torch_onnx_trt" in text:
            continue
        if "results_codex_ros2_closed_loop" in text and not include_ros2:
            continue
        if "results_codex_pressure_short_reduced" in text and not include_pressure:
            continue
        if not any(
            marker in text
            for marker in [
                "results_codex_full_safe_20260526_001",
                "results_codex_resnet50_realframes_20260526_002",
                "results_codex_pressure_short_reduced_20260526_001",
                "results_codex_ros2_closed_loop_20260526_003",
            ]
        ):
            continue
        paths.append(path)
    return sorted(paths)


def estimate_period_ms(df: pd.DataFrame) -> float:
    if "timestamp_ms" in df.columns:
        ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").dropna().to_numpy(dtype=float)
        if ts.size >= 3:
            diffs = np.diff(ts)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                return float(np.median(diffs))
    return positive_float(df["deadline_ms"], 33.333)


def request_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    score = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    q_low = positive_float(df.get("q_low", pd.Series([np.inf])), np.inf)
    q_high = positive_float(df.get("q_high", pd.Series([np.inf])), np.inf)
    alarm = score >= q_low
    request = score >= q_high
    return alarm, request


def apply_deferrable(request: np.ndarray, cost: np.ndarray, capacity: float, period_frames: int) -> np.ndarray:
    audit = np.zeros(request.size, dtype=np.int64)
    budget = float(capacity)
    for i, wants in enumerate(request):
        if i % period_frames == 0:
            budget = float(capacity)
        if wants and budget + 1e-9 >= cost[i]:
            audit[i] = 1
            budget -= float(cost[i])
    return audit


def apply_sporadic(request: np.ndarray, cost: np.ndarray, capacity: float, period_frames: int) -> np.ndarray:
    audit = np.zeros(request.size, dtype=np.int64)
    budget = float(capacity)
    replenishments: list[tuple[int, float]] = []
    for i, wants in enumerate(request):
        due = [x for x in replenishments if x[0] <= i]
        replenishments = [x for x in replenishments if x[0] > i]
        for _, amount in due:
            budget = min(float(capacity), budget + float(amount))
        if wants and budget + 1e-9 >= cost[i]:
            audit[i] = 1
            amount = float(cost[i])
            budget -= amount
            replenishments.append((i + period_frames, amount))
    return audit


def apply_cbs(request: np.ndarray, cost: np.ndarray, capacity: float, period_frames: int) -> np.ndarray:
    # Discrete-frame CBS replay for non-deferrable per-frame audits. If budget is
    # exhausted, the server postpones its deadline to the next period and rejects
    # the current audit request because stale audits are not useful for this frame.
    audit = np.zeros(request.size, dtype=np.int64)
    budget = float(capacity)
    server_deadline = period_frames
    for i, wants in enumerate(request):
        if i >= server_deadline:
            periods = max(1, math.floor((i - server_deadline) / period_frames) + 1)
            server_deadline += periods * period_frames
            budget = float(capacity)
        if wants and budget + 1e-9 >= cost[i]:
            audit[i] = 1
            budget -= float(cost[i])
        elif wants and budget + 1e-9 < cost[i]:
            server_deadline += period_frames
            budget = 0.0
    return audit


def build_policy(df: pd.DataFrame, method: str, capacity: float, refill: float) -> pd.DataFrame:
    out = df.copy()
    alarm, request = request_masks(out)
    cost = pd.to_numeric(out.get("token_audit_cost_ms", pd.Series([4.0] * len(out))), errors="coerce").fillna(4.0).to_numpy(dtype=float)
    # Use the measured bound for latency accounting when available.
    latency_cost = pd.to_numeric(out.get("audit_latency_bound_ms", out.get("token_audit_cost_ms", 4.0)), errors="coerce").fillna(4.0).to_numpy(dtype=float)
    monitor = pd.to_numeric(out["monitor_cost_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    baseline = pd.to_numeric(out["baseline_latency_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    deadline = pd.to_numeric(out["deadline_ms"], errors="coerce").fillna(33.333).to_numpy(dtype=float)
    period_frames = max(1, int(math.ceil(float(capacity) / max(float(refill), 1e-9))))

    if "DeferrableServer" in method:
        audit = apply_deferrable(request, cost, capacity, period_frames)
    elif "SporadicServer" in method:
        audit = apply_sporadic(request, cost, capacity, period_frames)
    elif "CBS" in method:
        audit = apply_cbs(request, cost, capacity, period_frames)
    else:
        raise ValueError(f"unknown method {method}")

    if method.endswith("+Slack"):
        slack = deadline - baseline - monitor
        audit = audit * (slack + 1e-9 >= latency_cost)

    out["method"] = method
    out["alarm"] = alarm.astype(np.int64)
    out["audit"] = audit.astype(np.int64)
    out["audit_cost_ms"] = audit.astype(float) * latency_cost
    out["audit_charge_ms"] = audit.astype(float) * cost
    out["audit_token_charge_ms"] = out["audit_charge_ms"]
    out["e2e_latency_ms"] = baseline + monitor + out["audit_cost_ms"].to_numpy(dtype=float)
    out["deadline_miss"] = (out["e2e_latency_ms"].to_numpy(dtype=float) > deadline).astype(np.int64)
    out["server_capacity_ms"] = float(capacity)
    out["server_period_frames"] = int(period_frames)
    out["server_replenish_ms_per_frame"] = float(refill)
    return out


def summarize(df: pd.DataFrame, meta: TraceMeta, source_path: Path) -> dict[str, object]:
    label = pd.to_numeric(df["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    alarm = pd.to_numeric(df["alarm"], errors="coerce").fillna(0).to_numpy(dtype=int)
    audit = pd.to_numeric(df["audit"], errors="coerce").fillna(0).to_numpy(dtype=int)
    e2e = pd.to_numeric(df["e2e_latency_ms"], errors="coerce").fillna(0.0)
    base = pd.to_numeric(df["baseline_latency_ms"], errors="coerce").fillna(0.0)
    deadline = pd.to_numeric(df["deadline_ms"], errors="coerce").fillna(33.333)
    monitor = pd.to_numeric(df["monitor_cost_ms"], errors="coerce").fillna(0.0)
    audit_bound = pd.to_numeric(df.get("audit_latency_bound_ms", df.get("audit_cost_ms", pd.Series([0] * len(df)))), errors="coerce").fillna(0.0)
    attack = label == 1
    benign = label == 0
    timestamp_period_ms = estimate_period_ms(df)
    period_ms = float(np.median(deadline.to_numpy(dtype=float)))
    carry_in = (e2e.to_numpy(dtype=float) > period_ms).astype(np.int64)
    admitted = audit == 1
    slack_margin = deadline.to_numpy(dtype=float) - base.to_numpy(dtype=float) - monitor.to_numpy(dtype=float) - audit_bound.to_numpy(dtype=float)
    admitted_margin = slack_margin[admitted]
    return {
        "source_root": meta.source_root,
        "pressure_label": meta.pressure_label,
        "dataset": meta.dataset,
        "deadline_name": meta.deadline_name,
        "model": meta.model,
        "seed": meta.seed,
        "scenario": meta.scenario,
        "method": str(df["method"].iloc[0]),
        "rows": int(len(df)),
        "admitted_audits": int(audit.sum()),
        "audit_rate": float(audit.mean()) if audit.size else 0.0,
        "alarm_attack_recall": float(alarm[attack].mean()) if np.any(attack) else float("nan"),
        "audit_attack_recall": float(audit[attack].mean()) if np.any(attack) else float("nan"),
        "false_alarm_rate": float(alarm[benign].mean()) if np.any(benign) else 0.0,
        "deadline_miss_ratio": float(pd.to_numeric(df["deadline_miss"], errors="coerce").fillna(0).mean()),
        "deadline_bound_failures": int(((e2e > deadline) & (audit == 1)).sum()),
        "period_ms_est": float(period_ms),
        "timestamp_period_ms_est": float(timestamp_period_ms),
        "carry_in_violations": int((carry_in & admitted).sum()),
        "monitor_cost_p50_ms": quantile(monitor, 0.50),
        "monitor_cost_p95_ms": quantile(monitor, 0.95),
        "monitor_cost_p99_ms": quantile(monitor, 0.99),
        "audit_cost_bound_ms": quantile(audit_bound, 0.99),
        "min_admitted_slack_margin_ms": float(np.min(admitted_margin)) if admitted_margin.size else float("nan"),
        "p99_latency_ms": quantile(e2e, 0.99),
        "latency_inflation_mean_ms": float((e2e - base).mean()),
        "source_path": str(source_path),
    }


def aggregate_mean(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [
        "rows",
        "admitted_audits",
        "audit_rate",
        "alarm_attack_recall",
        "audit_attack_recall",
        "false_alarm_rate",
        "deadline_miss_ratio",
        "deadline_bound_failures",
        "carry_in_violations",
        "monitor_cost_p50_ms",
        "monitor_cost_p95_ms",
        "monitor_cost_p99_ms",
        "audit_cost_bound_ms",
        "min_admitted_slack_margin_ms",
        "p99_latency_ms",
        "latency_inflation_mean_ms",
    ]
    agg = df.groupby(group_cols, dropna=False)[metrics].mean(numeric_only=True).reset_index()
    return agg.sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RTSS server-baseline and timing-validation replay over AGX scored traces.")
    parser.add_argument("--input-root", default=r"F:\RTSS\agx_orin64_ros2\work\agx_orin64_run_package_20260525_submission_full")
    parser.add_argument("--out-dir", default=r"F:\RTSS\exp_begin\results_agx_offline_rtss_hardening_20260526")
    parser.add_argument("--include-pressure", action="store_true")
    parser.add_argument("--include-ros2", action="store_true")
    parser.add_argument("--write-raw", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_server_replay"
    if args.write_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    inputs = find_inputs(input_root, include_pressure=args.include_pressure, include_ros2=args.include_ros2)
    if not inputs:
        raise SystemExit(f"no CIRCA-RT-Slack scored.csv files found under {input_root}")

    summary_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    for path in inputs:
        df = pd.read_csv(path)
        meta = parse_meta(path)
        capacity = positive_float(df.get("bucket_capacity", pd.Series([12.0])), 12.0)
        refill = positive_float(df.get("replenish_rate", pd.Series([0.45])), 0.45)

        original = df.copy()
        original["method"] = "CIRCA-RT-Slack"
        timing_rows.append(summarize(original, meta, path))
        summary_rows.append(summarize(original, meta, path))

        for method in SERVER_METHODS:
            replay = build_policy(df, method, capacity, refill)
            summary_rows.append(summarize(replay, meta, path))
            if args.write_raw:
                rel = Path(meta.source_root) / meta.dataset / meta.model / meta.seed / meta.scenario / f"{method.replace('+', '_')}.csv"
                target = raw_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                replay.to_csv(target, index=False)

    summary = pd.DataFrame(summary_rows)
    timing = pd.DataFrame(timing_rows)
    summary.to_csv(table_dir / "server_baseline_summary_all.csv", index=False)
    timing.to_csv(table_dir / "timing_validation_circa_slack_all.csv", index=False)

    aggregate_mean(
        summary,
        ["source_root", "pressure_label", "dataset", "model", "method"],
    ).to_csv(table_dir / "server_baseline_main_by_dataset_model.csv", index=False)
    aggregate_mean(
        summary,
        ["method"],
    ).to_csv(table_dir / "server_baseline_overall.csv", index=False)
    aggregate_mean(
        timing,
        ["source_root", "pressure_label", "dataset", "model", "method"],
    ).to_csv(table_dir / "timing_validation_main_by_dataset_model.csv", index=False)
    aggregate_mean(
        timing,
        ["method"],
    ).to_csv(table_dir / "timing_validation_overall.csv", index=False)

    manifest = {
        "input_root": str(input_root),
        "out_dir": str(out_dir),
        "num_input_scored_traces": len(inputs),
        "include_pressure": bool(args.include_pressure),
        "include_ros2": bool(args.include_ros2),
        "write_raw": bool(args.write_raw),
        "server_methods": SERVER_METHODS,
        "notes": [
            "Offline replay over existing AGX scored traces; no new AGX execution.",
            "Server baselines receive the same q_high risk requests as CIRCA-RT-Slack.",
            "Latency accounting uses audit_latency_bound_ms when available.",
        ],
    }
    (out_dir / "offline_rtss_hardening_manifest.json").write_text(
        pd.Series(manifest).to_json(indent=2),
        encoding="utf-8",
    )
    print(f"inputs={len(inputs)}")
    print(f"wrote {table_dir}")


if __name__ == "__main__":
    main()
