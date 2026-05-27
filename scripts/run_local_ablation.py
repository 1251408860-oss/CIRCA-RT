from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.core import CIRCARuntimeConfig, apply_circa_rt, apply_circa_rt_slack_admissible, fit_circa_rt
from circa_rt.features import RandomFourierFeatures, Residualizer, rolling_cross_scores_by_group
from circa_rt.metrics import summarize_detection
from circa_rt.schema import CONTEXT_COLUMNS, read_trace
from circa_rt.synthetic import generate_trace


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

SYNTHETIC_TESTS = [
    "nominal",
    "mode_shift",
    "stealthy_coupled_attack",
    "timing_only_attack",
    "semantic_only_attack",
    "mixed_attack",
]

ABLATIONS = [
    "CIRCA-RT",
    "CIRCA-RT-Slack",
    "CIRCA-NoContext",
    "CIRCA-NoTokenBucket",
    "CIRCA-NoSelectiveAudit",
    "CIRCA-HighOnly",
    "CIRCA-RawRFFTokenBucket",
]


def load_config() -> tuple[dict, CIRCARuntimeConfig]:
    raw = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    cfg = raw["circa_rt"]
    return raw, CIRCARuntimeConfig(
        window_size=int(cfg["window_size"]),
        feature_dim=int(cfg["feature_dim"]),
        low_quantile=float(cfg["low_quantile"]),
        high_quantile=float(cfg["high_quantile"]),
        monitor_cost_ms=float(cfg["monitor_cost_ms"]),
        heavy_audit_cost_ms=float(cfg["heavy_audit_cost_ms"]),
        bucket_capacity=float(cfg["bucket_capacity"]),
        replenish_rate=float(cfg["replenish_rate"]),
        seed=int(raw["random_seed"]),
    )


def token_bucket(scores: np.ndarray, q_low: float, q_high: float, cfg: CIRCARuntimeConfig) -> tuple[np.ndarray, np.ndarray]:
    alarm = np.zeros(len(scores), dtype=np.int64)
    audit = np.zeros(len(scores), dtype=np.int64)
    bucket = float(cfg.bucket_capacity)
    for i, score in enumerate(scores):
        bucket = min(float(cfg.bucket_capacity), bucket + float(cfg.replenish_rate))
        if score >= q_high and bucket >= cfg.heavy_audit_cost_ms:
            alarm[i] = 1
            audit[i] = 1
            bucket -= cfg.heavy_audit_cost_ms
        elif score >= q_low:
            alarm[i] = 1
    return alarm, audit


def apply_alarm(df: pd.DataFrame, method: str, score: np.ndarray, alarm: np.ndarray, audit: np.ndarray, cfg: CIRCARuntimeConfig) -> pd.DataFrame:
    out = df.copy()
    out["method"] = method
    out["score"] = np.asarray(score, dtype=float)
    out["alarm"] = np.asarray(alarm, dtype=np.int64)
    out["audit"] = np.asarray(audit, dtype=np.int64)
    out["monitor_cost_ms"] = float(cfg.monitor_cost_ms)
    out["audit_cost_ms"] = out["audit"].to_numpy(dtype=float) * float(cfg.heavy_audit_cost_ms)
    out["e2e_latency_ms"] = out["baseline_latency_ms"].to_numpy(dtype=float) + out["monitor_cost_ms"] + out["audit_cost_ms"]
    out["deadline_miss"] = (out["e2e_latency_ms"] > out["deadline_ms"]).astype(int)
    return out


class ScoreModel:
    def __init__(
        self,
        *,
        cfg: CIRCARuntimeConfig,
        residualizer: Residualizer | None,
        rff_u: RandomFourierFeatures,
        rff_v: RandomFourierFeatures,
        q_low: float,
        q_high: float,
    ) -> None:
        self.cfg = cfg
        self.residualizer = residualizer
        self.rff_u = rff_u
        self.rff_v = rff_v
        self.q_low = q_low
        self.q_high = q_high


def values_for_model(df: pd.DataFrame, residualizer: Residualizer | None) -> tuple[np.ndarray, np.ndarray]:
    sem = df["semantic_residual"].to_numpy(dtype=float)
    tim = df["timing_residual_ms"].to_numpy(dtype=float)
    if residualizer is None:
        return sem, tim
    context = df[CONTEXT_COLUMNS].to_numpy(dtype=float)
    mode = df["mode"].astype(str).to_numpy()
    return residualizer.transform(sem, tim, context, mode)


def fit_score_model(calibration: pd.DataFrame, cfg: CIRCARuntimeConfig, *, use_context: bool) -> ScoreModel:
    label = calibration["label"].to_numpy(dtype=int)
    benign = label == 0
    sem = calibration["semantic_residual"].to_numpy(dtype=float)
    tim = calibration["timing_residual_ms"].to_numpy(dtype=float)
    residualizer: Residualizer | None = None
    if use_context:
        context = calibration[CONTEXT_COLUMNS].to_numpy(dtype=float)
        mode = calibration["mode"].astype(str).to_numpy()
        residualizer = Residualizer().fit(sem[benign], tim[benign], context[benign], mode[benign])
    u, v = values_for_model(calibration, residualizer)
    rff_u = RandomFourierFeatures(cfg.feature_dim, seed=cfg.seed).fit(u[benign])
    rff_v = RandomFourierFeatures(cfg.feature_dim, seed=cfg.seed + 1).fit(v[benign])
    score = rolling_cross_scores_by_group(
        rff_u.transform(u),
        rff_v.transform(v),
        calibration["run_id"].astype(str).to_numpy(),
        cfg.window_size,
    )
    return ScoreModel(
        cfg=cfg,
        residualizer=residualizer,
        rff_u=rff_u,
        rff_v=rff_v,
        q_low=float(np.quantile(score[benign], cfg.low_quantile)),
        q_high=float(np.quantile(score[benign], cfg.high_quantile)),
    )


def score_with_model(model: ScoreModel, df: pd.DataFrame) -> np.ndarray:
    u, v = values_for_model(df, model.residualizer)
    return rolling_cross_scores_by_group(
        model.rff_u.transform(u),
        model.rff_v.transform(v),
        df["run_id"].astype(str).to_numpy(),
        model.cfg.window_size,
    )


def apply_variant(
    variant: str,
    raw: pd.DataFrame,
    *,
    full_model,
    contextual: ScoreModel,
    raw_model: ScoreModel,
    cfg: CIRCARuntimeConfig,
) -> pd.DataFrame:
    if variant == "CIRCA-RT":
        return apply_circa_rt(full_model, raw, method="CIRCA-RT")
    if variant == "CIRCA-RT-Slack":
        return apply_circa_rt_slack_admissible(full_model, raw, method="CIRCA-RT-Slack")
    if variant in {"CIRCA-NoTokenBucket", "CIRCA-NoSelectiveAudit", "CIRCA-HighOnly"}:
        score = score_with_model(contextual, raw)
        if variant == "CIRCA-NoTokenBucket":
            alarm = (score >= contextual.q_low).astype(np.int64)
            audit = (score >= contextual.q_high).astype(np.int64)
        elif variant == "CIRCA-NoSelectiveAudit":
            alarm = (score >= contextual.q_low).astype(np.int64)
            audit = alarm.copy()
        else:
            alarm = (score >= contextual.q_high).astype(np.int64)
            audit = alarm.copy()
        return apply_alarm(raw, variant, score, alarm, audit, cfg)
    if variant == "CIRCA-NoContext":
        score = score_with_model(raw_model, raw)
        alarm = (score >= raw_model.q_low).astype(np.int64)
        audit = (score >= raw_model.q_high).astype(np.int64)
        return apply_alarm(raw, variant, score, alarm, audit, cfg)
    if variant == "CIRCA-RawRFFTokenBucket":
        score = score_with_model(raw_model, raw)
        alarm, audit = token_bucket(score, raw_model.q_low, raw_model.q_high, cfg)
        return apply_alarm(raw, variant, score, alarm, audit, cfg)
    raise ValueError(f"unknown variant: {variant}")


def synthetic_groups(config: dict, cfg: CIRCARuntimeConfig) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for fold in range(int(config["n_seeds"])):
        test_seed = int(config["random_seed"]) + fold
        cal_seed = int(config["random_seed"]) + 4000 + fold
        calibration = pd.concat(
            [
                generate_trace(
                    trace_type=trace_type,
                    seed=cal_seed,
                    n=int(config["n_per_trace"]),
                    deadline_ms=float(config["deadline_ms"]),
                    baseline_latency_ms=float(config["baseline_latency_ms"]),
                )
                for trace_type in ["nominal", "mode_shift"]
            ],
            ignore_index=True,
        )
        tests = [
            (
                trace_type,
                generate_trace(
                    trace_type=trace_type,
                    seed=test_seed,
                    n=int(config["n_per_trace"]),
                    deadline_ms=float(config["deadline_ms"]),
                    baseline_latency_ms=float(config["baseline_latency_ms"]),
                ),
            )
            for trace_type in SYNTHETIC_TESTS
        ]
        groups.append(
            {
                "dataset": "local_synthetic",
                "runtime": "synthetic",
                "provider": "numpy",
                "model": "synthetic_trace",
                "seed": test_seed,
                "calibration": calibration,
                "tests": tests,
                "cfg": replace(cfg, seed=test_seed),
            }
        )
    return groups


def autodl_groups() -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    trace_root = ROOT / "results_autodl_final" / "traces"
    for seed_dir in sorted(trace_root.glob("*/*/*/seed*")):
        if not seed_dir.is_dir() or not (seed_dir / "nominal.csv").exists():
            continue
        groups.append(
            {
                "dataset": "autodl_full",
                "runtime": seed_dir.parent.parent.parent.name,
                "provider": seed_dir.parent.parent.name,
                "model": seed_dir.parent.name,
                "seed": int(seed_dir.name.replace("seed", "")),
                "calibration": read_trace(seed_dir / "nominal.csv"),
                "tests": [(p.stem, read_trace(p)) for p in sorted(seed_dir.glob("*.csv"))],
            }
        )
    trace_root = ROOT / "results_autodl_ros2_twoprocess_final" / "traces"
    for seed_dir in sorted(trace_root.glob("seed*")):
        if not seed_dir.is_dir() or not (seed_dir / "nominal.csv").exists():
            continue
        cal_paths = [seed_dir / "nominal.csv"]
        if (seed_dir / "mode_shift.csv").exists():
            cal_paths.append(seed_dir / "mode_shift.csv")
        groups.append(
            {
                "dataset": "autodl_ros2_twoprocess",
                "runtime": "autodl_ros2_twoprocess",
                "provider": "ros2_dds_twoprocess",
                "model": "middleware_trace",
                "seed": int(seed_dir.name.replace("seed", "")),
                "calibration": pd.concat([read_trace(p) for p in cal_paths], ignore_index=True),
                "tests": [(p.stem, read_trace(p)) for p in sorted(seed_dir.glob("*.csv"))],
            }
        )
    return groups


def run_group(group: dict[str, object], base_cfg: CIRCARuntimeConfig) -> list[dict[str, object]]:
    seed = int(group["seed"])
    cfg = replace(base_cfg, seed=seed)
    calibration = group["calibration"]
    assert isinstance(calibration, pd.DataFrame)
    full_model = fit_circa_rt(calibration, cfg)
    contextual = fit_score_model(calibration, cfg, use_context=True)
    raw_model = fit_score_model(calibration, cfg, use_context=False)
    rows: list[dict[str, object]] = []
    for trace_type, raw in group["tests"]:
        assert isinstance(raw, pd.DataFrame)
        for variant in ABLATIONS:
            scored = apply_variant(variant, raw, full_model=full_model, contextual=contextual, raw_model=raw_model, cfg=cfg)
            row = summarize_detection(scored, trace_name=f"{group['dataset']}_{group['model']}_seed{seed}_{trace_type}").to_dict()
            row.update(
                {
                    "dataset": group["dataset"],
                    "runtime": group["runtime"],
                    "provider": group["provider"],
                    "model": group["model"],
                    "seed": seed,
                    "trace_type": trace_type,
                    "variant": variant,
                }
            )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "autodl"], choices=["synthetic", "autodl"])
    parser.add_argument("--out-dir", default=str(ROOT / "results_local_complete" / "ablation"))
    args = parser.parse_args()

    config, base_cfg = load_config()
    groups: list[dict[str, object]] = []
    if "synthetic" in args.datasets:
        groups.extend(synthetic_groups(config, base_cfg))
    if "autodl" in args.datasets:
        groups.extend(autodl_groups())
    if not groups:
        raise SystemExit("no groups found")

    out = Path(args.out_dir)
    table_dir = out / "tables"
    summary_dir = out / "summaries"
    for directory in [table_dir, summary_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for group in groups:
        rows.extend(run_group(group, base_cfg))
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)

    group_cols = ["dataset", "runtime", "provider", "model", "variant"]
    main = mean_table(summary, group_cols, METRIC_COLS).sort_values(
        ["dataset", "runtime", "model", "attack_recall", "audit_rate"],
        ascending=[True, True, True, False, True],
    )
    main.to_csv(table_dir / "ablation_main_table.csv", index=False)
    bootstrap_ci_table(summary, group_cols, METRIC_COLS).to_csv(table_dir / "ablation_main_table_ci.csv", index=False)
    scenario = mean_table(summary, group_cols + ["trace_type"], METRIC_COLS).sort_values(
        ["dataset", "runtime", "model", "trace_type", "attack_recall"],
        ascending=[True, True, True, True, False],
    )
    scenario.to_csv(table_dir / "ablation_scenario_table.csv", index=False)
    print(main.head(80).to_string(index=False))


if __name__ == "__main__":
    main()
