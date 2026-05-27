from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


KEY_METHODS = [
    "CIRCA-RT",
    "CIRCA-RT-Slack",
    "ConditionalRFF-HSIC",
    "ContextAwareConformal",
    "RFF-HSIC",
    "TimingThreshold",
    "AlwaysAudit",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def level_dirs(root: Path) -> list[Path]:
    rows = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "tables" / "agx_realframe_main_all.csv").exists():
            rows.append(child)
    return rows


def meta_for(run_dir: Path) -> dict[str, Any]:
    meta = read_json(run_dir / "pressure_meta.json")
    if not meta:
        meta = {
            "pressure_label": run_dir.name,
            "pressure_level_index": 0,
            "gpu_stress_size": None,
            "gpu_stress_repeats": None,
            "coupled_stress_extra_repeats": None,
        }
    return meta


def with_meta(df: pd.DataFrame, run_dir: Path, meta: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "pressure_run", run_dir.name)
    out.insert(1, "pressure_label", meta.get("pressure_label", run_dir.name))
    out.insert(2, "pressure_level_index", meta.get("pressure_level_index", 0))
    out.insert(3, "gpu_stress_size", meta.get("gpu_stress_size"))
    out.insert(4, "gpu_stress_repeats", meta.get("gpu_stress_repeats"))
    out.insert(5, "coupled_stress_extra_repeats", meta.get("coupled_stress_extra_repeats"))
    return out


def read_all(root: Path, filename: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_dir in level_dirs(root):
        path = run_dir / "tables" / filename
        if not path.exists() or path.stat().st_size <= 1:
            continue
        meta = meta_for(run_dir)
        rows.append(with_meta(pd.read_csv(path), run_dir, meta))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def table_md(df: pd.DataFrame, cols: list[str], max_rows: int = 24) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    sub = df[[c for c in cols if c in df.columns]].head(max_rows).copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda x: f"{x:.4f}")
    widths = [max(len(str(c)), *(len(str(v)) for v in sub[c].astype(str))) for c in sub.columns]
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(sub.columns, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]).ljust(w) for c, w in zip(sub.columns, widths)) + " |")
    return lines


def robustness_delta(real: pd.DataFrame) -> pd.DataFrame:
    if real.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["pressure_label", "gpu_stress_size", "dataset", "deadline_ms_eval", "primary_model"]
    for keys, group in real.groupby(group_cols, dropna=False):
        group = group.copy()
        target = group[group["method"] == "CIRCA-RT-Slack"]
        if target.empty:
            target = group[group["method"] == "CIRCA-RT"]
        if target.empty:
            continue
        t = target.iloc[0]
        for baseline_name in ["AlwaysAudit", "ConditionalRFF-HSIC", "ContextAwareConformal", "RFF-HSIC"]:
            b = group[group["method"] == baseline_name]
            if b.empty:
                continue
            b = b.iloc[0]
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "target_method": str(t["method"]),
                    "baseline_method": baseline_name,
                    "target_recall": float(t["attack_recall"]),
                    "baseline_recall": float(b["attack_recall"]),
                    "delta_recall": float(t["attack_recall"] - b["attack_recall"]),
                    "target_audit_rate": float(t["audit_rate"]),
                    "baseline_audit_rate": float(b["audit_rate"]),
                    "delta_audit_rate": float(t["audit_rate"] - b["audit_rate"]),
                    "target_deadline_miss": float(t["deadline_miss_ratio"]),
                    "baseline_deadline_miss": float(b["deadline_miss_ratio"]),
                    "delta_deadline_miss": float(t["deadline_miss_ratio"] - b["deadline_miss_ratio"]),
                    "target_p99_ms": float(t["p99_latency_ms"]),
                    "baseline_p99_ms": float(b["p99_latency_ms"]),
                    "delta_p99_ms": float(t["p99_latency_ms"] - b["p99_latency_ms"]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def breakpoint_table(real: pd.DataFrame, miss_threshold: float) -> pd.DataFrame:
    if real.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["pressure_label", "gpu_stress_size", "dataset", "primary_model", "method"]
    for keys, group in real.groupby(group_cols, dropna=False):
        group = group.sort_values("deadline_ms_eval")
        ok = group[group["deadline_miss_ratio"] <= miss_threshold]
        row = dict(zip(group_cols, keys))
        row["miss_threshold"] = float(miss_threshold)
        row["tightest_passing_deadline_ms"] = float(ok["deadline_ms_eval"].min()) if not ok.empty else float("nan")
        row["max_deadline_miss_ratio"] = float(group["deadline_miss_ratio"].max())
        row["max_p99_latency_ms"] = float(group["p99_latency_ms"].max())
        row["mean_attack_recall"] = float(group["attack_recall"].mean())
        row["mean_audit_rate"] = float(group["audit_rate"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def failed_commands(root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in level_dirs(root):
        path = run_dir / "logs" / "failed_commands.log"
        text = path.read_text(encoding="utf-8", errors="replace").strip() if path.exists() else ""
        rows.append({"pressure_run": run_dir.name, "failed_commands": text, "has_failures": bool(text)})
    return pd.DataFrame(rows)


def write_figures(root: Path, real: pd.DataFrame) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if real.empty:
        return
    try:
        import matplotlib.pyplot as plt

        key = real[real["method"].isin(["CIRCA-RT-Slack", "CIRCA-RT", "ConditionalRFF-HSIC", "ContextAwareConformal", "AlwaysAudit"])].copy()
        if key.empty:
            return
        for metric, ylabel in [
            ("deadline_miss_ratio", "deadline miss ratio"),
            ("p99_latency_ms", "p99 latency (ms)"),
            ("attack_recall", "attack recall"),
            ("audit_rate", "audit rate"),
        ]:
            agg = key.groupby(["pressure_level_index", "pressure_label", "deadline_ms_eval", "method"], as_index=False)[metric].mean(numeric_only=True)
            for label, g in agg.groupby("pressure_label"):
                plt.figure(figsize=(8.4, 4.8))
                for method, m in g.groupby("method"):
                    m = m.sort_values("deadline_ms_eval")
                    plt.plot(m["deadline_ms_eval"], m[metric], marker="o", label=method)
                plt.xlabel("deadline (ms)")
                plt.ylabel(ylabel)
                plt.title(f"AGX pressure sweep: {ylabel}, stress={label}")
                plt.grid(True, alpha=0.25)
                plt.legend(fontsize=7)
                plt.tight_layout()
                safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(label))
                plt.savefig(fig_dir / f"{metric}_{safe}.png", dpi=220)
                plt.close()
    except Exception as exc:
        (fig_dir / "figure_error.txt").write_text(repr(exc), encoding="utf-8")


def write_report(root: Path, real: pd.DataFrame, audit: pd.DataFrame, tegra: pd.DataFrame, delta: pd.DataFrame, breakpoints: pd.DataFrame, failures: pd.DataFrame) -> None:
    report = root / "PRESSURE_SWEEP_REPORT.md"
    lines: list[str] = [
        "# AGX Orin 64GB Pressure Sweep Report",
        "",
        f"Result root: `{root}`",
        "",
        "## Coverage",
        "",
        f"- Real-frame rows: {len(real)}",
        f"- Audit-bound validation rows: {len(audit)}",
        f"- Tegrastats summary rows: {len(tegra)}",
        f"- Pressure levels: {', '.join(map(str, sorted(real['pressure_label'].dropna().unique(), key=str))) if not real.empty else 'none'}",
        f"- Datasets: {', '.join(map(str, sorted(real['dataset'].dropna().unique(), key=str))) if not real.empty and 'dataset' in real else 'none'}",
        f"- Deadlines: {', '.join(map(lambda x: f'{float(x):g}', sorted(real['deadline_ms_eval'].dropna().unique()))) if not real.empty and 'deadline_ms_eval' in real else 'none'}",
        "",
    ]

    if not failures.empty:
        fail_count = int(failures["has_failures"].sum())
        lines.extend(["## Command Failures", "", f"- Result roots with failed commands: {fail_count}", ""])
        if fail_count:
            lines.extend(table_md(failures[failures["has_failures"]], ["pressure_run", "failed_commands"], max_rows=20))
            lines.append("")

    if not audit.empty and "pass" in audit.columns:
        failed = int((~audit["pass"].astype(bool)).sum())
        lines.extend(["## Audit-Bound Validation", "", f"- Checked rows: {len(audit)}", f"- Failed rows: {failed}", ""])

    if not real.empty:
        selected = real[real["method"].isin(KEY_METHODS)].copy()
        selected = selected.sort_values(["pressure_level_index", "dataset", "deadline_ms_eval", "primary_model", "method"])
        lines.extend(["## Selected Main Results", ""])
        lines.extend(
            table_md(
                selected,
                [
                    "pressure_label",
                    "dataset",
                    "deadline_ms_eval",
                    "primary_model",
                    "method",
                    "attack_recall",
                    "audit_rate",
                    "p99_latency_ms",
                    "deadline_miss_ratio",
                ],
                max_rows=40,
            )
        )
        lines.append("")

    if not breakpoints.empty:
        key = breakpoints[breakpoints["method"].isin(["CIRCA-RT-Slack", "CIRCA-RT", "ConditionalRFF-HSIC", "ContextAwareConformal", "AlwaysAudit"])].copy()
        key = key.sort_values(["pressure_label", "dataset", "method"])
        lines.extend(["## Deadline Breakpoints", ""])
        lines.append("Tightest deadline whose miss ratio is at or below the configured threshold.")
        lines.append("")
        lines.extend(
            table_md(
                key,
                [
                    "pressure_label",
                    "dataset",
                    "primary_model",
                    "method",
                    "tightest_passing_deadline_ms",
                    "max_deadline_miss_ratio",
                    "max_p99_latency_ms",
                    "mean_attack_recall",
                    "mean_audit_rate",
                ],
                max_rows=40,
            )
        )
        lines.append("")

    if not delta.empty:
        key = delta[delta["baseline_method"].isin(["AlwaysAudit", "ConditionalRFF-HSIC", "ContextAwareConformal"])].copy()
        key = key.sort_values(["pressure_label", "dataset", "deadline_ms_eval", "baseline_method"])
        lines.extend(["## CIRCA-RT-Slack Delta", ""])
        lines.extend(
            table_md(
                key,
                [
                    "pressure_label",
                    "dataset",
                    "deadline_ms_eval",
                    "baseline_method",
                    "target_recall",
                    "baseline_recall",
                    "delta_recall",
                    "target_audit_rate",
                    "baseline_audit_rate",
                    "delta_deadline_miss",
                    "delta_p99_ms",
                ],
                max_rows=40,
            )
        )
        lines.append("")

    if not tegra.empty:
        lines.extend(["## Telemetry", ""])
        summary = tegra.copy()
        numeric_cols = [c for c in ["samples", "ram_used_mb_max", "cpu_temp_c_max", "gpu_temp_c_max", "gr3d_freq_max"] if c in summary.columns]
        group_cols = ["pressure_label"] if "pressure_label" in summary.columns else []
        if group_cols and numeric_cols:
            telem = summary.groupby(group_cols, as_index=False)[numeric_cols].max(numeric_only=True)
            lines.extend(table_md(telem, group_cols + numeric_cols, max_rows=20))
            lines.append("")

    lines.extend(
        [
            "## Claim Boundary",
            "",
            "- Use this as an AGX real-device robustness/pressure sweep only when all result roots have no failed commands.",
            "- Report TensorRT/ONNXRuntime claims only if a separate backend provider table proves the providers ran.",
            "- Do not call this a power-efficiency result unless VDD power fields are present in tegrastats and parsed.",
            "- Prefer plots of deadline miss, p99 latency, recall, and audit rate versus deadline under each stress level.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AGX Orin 64GB pressure sweep outputs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--miss-threshold", type=float, default=0.01)
    args = parser.parse_args()

    root = args.input_root
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    real = read_all(root, "agx_realframe_main_all.csv")
    audit = read_all(root, "agx_audit_bound_validation_all.csv")
    tegra = read_all(root, "tegrastats_summary.csv")
    admission = read_all(root, "agx_deadline_safe_admission_delta.csv")
    failures = failed_commands(root)
    delta = robustness_delta(real)
    breakpoints = breakpoint_table(real, args.miss_threshold)

    real.to_csv(table_dir / "pressure_realframe_main_all.csv", index=False)
    audit.to_csv(table_dir / "pressure_audit_bound_validation_all.csv", index=False)
    tegra.to_csv(table_dir / "pressure_tegrastats_summary_all.csv", index=False)
    admission.to_csv(table_dir / "pressure_deadline_safe_admission_delta_all.csv", index=False)
    failures.to_csv(table_dir / "pressure_failed_commands.csv", index=False)
    delta.to_csv(table_dir / "pressure_circa_slack_delta.csv", index=False)
    breakpoints.to_csv(table_dir / "pressure_deadline_breakpoints.csv", index=False)

    manifest = {
        "input_root": str(root),
        "levels": len(level_dirs(root)),
        "realframe_rows": int(len(real)),
        "audit_bound_rows": int(len(audit)),
        "tegrastats_rows": int(len(tegra)),
        "admission_rows": int(len(admission)),
        "failed_level_count": int(failures["has_failures"].sum()) if not failures.empty else 0,
        "miss_threshold": float(args.miss_threshold),
    }
    (table_dir / "pressure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_figures(root, real)
    write_report(root, real, audit, tegra, delta, breakpoints, failures)
    print(root / "PRESSURE_SWEEP_REPORT.md")


if __name__ == "__main__":
    main()
