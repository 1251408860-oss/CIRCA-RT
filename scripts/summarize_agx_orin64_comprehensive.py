from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def parse_deadline_from_name(name: str) -> tuple[str, float | None]:
    if "_d" not in name:
        return name, None
    dataset, tag = name.rsplit("_d", 1)
    try:
        return dataset, float(tag.replace("p", ".").replace("m", "-"))
    except ValueError:
        return dataset, None


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def collect_realframes(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    real_root = root / "realframes"
    if not real_root.exists():
        return pd.DataFrame()
    for run_dir in sorted(p for p in real_root.iterdir() if p.is_dir()):
        table = run_dir / "tables" / "autodl_perception_main_table.csv"
        if not table.exists():
            continue
        df = pd.read_csv(table)
        dataset, deadline = parse_deadline_from_name(run_dir.name)
        df.insert(0, "dataset", dataset)
        df.insert(1, "deadline_ms_eval", deadline)
        df.insert(2, "run_dir", str(run_dir))
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_realframe_ci(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    real_root = root / "realframes"
    if not real_root.exists():
        return pd.DataFrame()
    for run_dir in sorted(p for p in real_root.iterdir() if p.is_dir()):
        table = run_dir / "tables" / "autodl_perception_main_table_ci.csv"
        if not table.exists():
            continue
        df = pd.read_csv(table)
        dataset, deadline = parse_deadline_from_name(run_dir.name)
        df.insert(0, "dataset", dataset)
        df.insert(1, "deadline_ms_eval", deadline)
        df.insert(2, "run_dir", str(run_dir))
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_audit_bound(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((root / "realframes").glob("*/tables/audit_bound_validation.csv")) if (root / "realframes").exists() else []:
        df = pd.read_csv(path)
        dataset, deadline = parse_deadline_from_name(path.parents[1].name)
        df.insert(0, "dataset", dataset)
        df.insert(1, "deadline_ms_eval", deadline)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_backend(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    backend = root / "backend_torch_onnx_trt" / "tables"
    monitoring = read_csv_if_exists(backend / "monitoring_main_table.csv")
    latency = read_csv_if_exists(backend / "autodl_full_latency_overview.csv")
    return monitoring, latency


def write_figures(root: Path, real: pd.DataFrame, backend: pd.DataFrame) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        if not real.empty:
            primary_deadline = sorted([d for d in real["deadline_ms_eval"].dropna().unique()])[0]
            sub = real[(real["deadline_ms_eval"] == primary_deadline) & (real["method"].isin(KEY_METHODS))].copy()
            if not sub.empty:
                plt.figure(figsize=(8.4, 5.0))
                for method, group in sub.groupby("method"):
                    plt.scatter(group["audit_rate"], group["attack_recall"], label=method, s=58)
                plt.xlabel("audit rate")
                plt.ylabel("attack recall")
                plt.title(f"AGX Orin real-frame recall vs audit, deadline {primary_deadline:g} ms")
                plt.grid(True, alpha=0.25)
                plt.legend(fontsize=7, ncol=2)
                plt.tight_layout()
                plt.savefig(fig_dir / "agx_realframe_recall_vs_audit.png", dpi=220)
                plt.savefig(fig_dir / "agx_realframe_recall_vs_audit.pdf")
                plt.close()

                p99 = sub.pivot_table(index="method", columns=["dataset", "primary_model"], values="p99_latency_ms", aggfunc="mean")
                ax = p99.plot(kind="bar", figsize=(11.0, 5.0), width=0.78)
                ax.set_ylabel("p99 latency (ms)")
                ax.set_title("AGX Orin real-frame p99 latency")
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(fontsize=6, ncol=2)
                plt.tight_layout()
                plt.savefig(fig_dir / "agx_realframe_p99_latency.png", dpi=220)
                plt.savefig(fig_dir / "agx_realframe_p99_latency.pdf")
                plt.close()

        if not backend.empty and {"runtime", "model", "method", "p99_latency_ms"}.issubset(backend.columns):
            sub = backend[backend["method"].isin(["CIRCA-RT", "CIRCA-RT-Slack", "ConditionalRFF-HSIC", "ContextAwareConformal", "AlwaysAudit"])].copy()
            if not sub.empty:
                p = sub.pivot_table(index="method", columns=["runtime", "model"], values="p99_latency_ms", aggfunc="mean")
                ax = p.plot(kind="bar", figsize=(11.0, 5.0), width=0.78)
                ax.set_ylabel("p99 latency (ms)")
                ax.set_title("AGX Orin backend p99 latency")
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(fontsize=6, ncol=2)
                plt.tight_layout()
                plt.savefig(fig_dir / "agx_backend_p99_latency.png", dpi=220)
                plt.savefig(fig_dir / "agx_backend_p99_latency.pdf")
                plt.close()
    except Exception as exc:
        (fig_dir / "figure_error.txt").write_text(repr(exc), encoding="utf-8")


def write_report(root: Path, real: pd.DataFrame, backend: pd.DataFrame, audit: pd.DataFrame, tegra: pd.DataFrame) -> None:
    report = root / "AGX_ORIN64_COMPREHENSIVE_REPORT.md"
    lines: list[str] = [
        "# AGX Orin 64GB Comprehensive Experiment Report",
        "",
        "This report is generated from the AGX comprehensive result directory.",
        "",
        "## Result Coverage",
        "",
        f"- Real-frame main rows: {len(real)}",
        f"- Backend monitoring rows: {len(backend)}",
        f"- Audit-bound validation rows: {len(audit)}",
        f"- Tegrastats summary rows: {len(tegra)}",
        "",
    ]

    if not real.empty:
        lines.extend(["## Real-Frame Highlights", ""])
        key = real[real["method"].isin(["CIRCA-RT", "CIRCA-RT-Slack", "ConditionalRFF-HSIC", "ContextAwareConformal"])].copy()
        if not key.empty:
            primary_deadline = sorted([d for d in key["deadline_ms_eval"].dropna().unique()])[0]
            key = key[key["deadline_ms_eval"] == primary_deadline]
            grouped = (
                key.groupby(["dataset", "primary_model", "method"], as_index=False)[
                    ["attack_recall", "false_alarm_rate", "audit_rate", "p99_latency_ms", "p999_latency_ms", "deadline_miss_ratio"]
                ]
                .mean(numeric_only=True)
                .sort_values(["dataset", "primary_model", "attack_recall"], ascending=[True, True, False])
            )
            grouped.to_csv(root / "tables" / "agx_realframe_selected_main.csv", index=False)
            for _, row in grouped.head(24).iterrows():
                lines.append(
                    f"- {row.dataset}/{row.primary_model}/{row.method}: recall {row.attack_recall:.3f}, "
                    f"audit {row.audit_rate:.3f}, p99 {row.p99_latency_ms:.3f} ms, miss {row.deadline_miss_ratio:.4f}."
                )
            lines.append("")

    if not backend.empty:
        lines.extend(["## Backend Highlights", ""])
        cols = [c for c in ["runtime", "provider", "model", "method", "attack_recall", "audit_rate", "p99_latency_ms", "deadline_miss_ratio"] if c in backend.columns]
        key = backend[backend["method"].isin(["CIRCA-RT", "CIRCA-RT-Slack", "ConditionalRFF-HSIC", "ContextAwareConformal"])][cols].copy()
        key.to_csv(root / "tables" / "agx_backend_selected_main.csv", index=False)
        for _, row in key.head(24).iterrows():
            provider = getattr(row, "provider", "")
            lines.append(
                f"- {row.runtime}/{provider}/{row.model}/{row.method}: recall {row.attack_recall:.3f}, "
                f"audit {row.audit_rate:.3f}, p99 {row.p99_latency_ms:.3f} ms, miss {row.deadline_miss_ratio:.4f}."
            )
        lines.append("")

    if not audit.empty and "pass" in audit.columns:
        failed = int((~audit["pass"].astype(bool)).sum())
        lines.extend(["## Audit-Bound Validation", "", f"- Checked rows: {len(audit)}", f"- Failed rows: {failed}", ""])

    if not tegra.empty:
        lines.extend(["## Device Telemetry", ""])
        summary = tegra.copy()
        summary.to_csv(root / "tables" / "agx_tegrastats_summary_all.csv", index=False)
        all_row = summary[summary["source_file"].astype(str) == "__all__"] if "source_file" in summary.columns else pd.DataFrame()
        if not all_row.empty:
            row = all_row.iloc[0]
            power = row.get("vdd_in_mw_mean", float("nan"))
            temp_cols = [c for c in summary.columns if c.endswith("_temp_c_max")]
            temp = max([float(row.get(c, float("nan"))) for c in temp_cols if pd.notna(row.get(c, float("nan")))] or [float("nan")])
            if pd.notna(power):
                lines.append(f"- Mean VDD_IN: {power / 1000.0:.3f} W")
            if pd.notna(temp):
                lines.append(f"- Max parsed temperature: {temp:.1f} C")
        lines.append("")

    lines.extend(
        [
            "## Claim Boundary",
            "",
            "- Use TensorRT claims only when provider status and backend tables show TensorRTExecutionProvider or a TensorRT engine actually ran.",
            "- Use the real-frame tables as Jetson CUDA/PyTorch evidence unless the specific run also proves TensorRT execution.",
            "- Keep AGX results separate from old deleted/misnamed AGX artifacts.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AGX Orin 64GB comprehensive experiment outputs.")
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.result_root
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    real = collect_realframes(root)
    real_ci = collect_realframe_ci(root)
    audit = collect_audit_bound(root)
    backend, backend_latency = collect_backend(root)
    tegra = read_csv_if_exists(table_dir / "tegrastats_summary.csv")
    admission = read_csv_if_exists(root / "deadline_safe_admission" / "tables" / "admission_delta_tight_quantiles.csv")
    slack = read_csv_if_exists(root / "circa_slack_sensitivity" / "summary_tables" / "slack_deadline_sensitivity_delta_selected.csv")

    real.to_csv(table_dir / "agx_realframe_main_all.csv", index=False)
    real_ci.to_csv(table_dir / "agx_realframe_main_ci_all.csv", index=False)
    audit.to_csv(table_dir / "agx_audit_bound_validation_all.csv", index=False)
    backend.to_csv(table_dir / "agx_backend_monitoring_main.csv", index=False)
    backend_latency.to_csv(table_dir / "agx_backend_latency_overview.csv", index=False)
    admission.to_csv(table_dir / "agx_deadline_safe_admission_delta.csv", index=False)
    slack.to_csv(table_dir / "agx_circa_slack_sensitivity_delta.csv", index=False)

    manifest = {
        "result_root": str(root),
        "realframe_main_rows": int(len(real)),
        "realframe_ci_rows": int(len(real_ci)),
        "backend_rows": int(len(backend)),
        "backend_latency_rows": int(len(backend_latency)),
        "audit_bound_rows": int(len(audit)),
        "tegrastats_summary_rows": int(len(tegra)),
        "admission_delta_rows": int(len(admission)),
        "slack_delta_rows": int(len(slack)),
    }
    (table_dir / "agx_comprehensive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_figures(root, real, backend)
    write_report(root, real, backend, audit, tegra)
    print(table_dir / "agx_comprehensive_manifest.json")
    print(root / "AGX_ORIN64_COMPREHENSIVE_REPORT.md")


if __name__ == "__main__":
    main()
