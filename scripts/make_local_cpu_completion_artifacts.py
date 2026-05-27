from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CORE_METHODS = [
    "CIRCA-RT",
    "ConditionalRFF-HSIC",
    "RFF-HSIC",
    "ContextAwareConformal",
    "LearnedFourierIndependence",
    "OnlineConformalAnomaly",
]

ABLATION_VARIANTS = [
    "CIRCA-RT",
    "CIRCA-RT-Slack",
    "CIRCA-NoContext",
    "CIRCA-NoTokenBucket",
    "CIRCA-NoSelectiveAudit",
    "CIRCA-HighOnly",
    "CIRCA-RawRFFTokenBucket",
]

COLORS = {
    "CIRCA-RT": "#0B6E4F",
    "CIRCA-RT-Slack": "#227C9D",
    "ConditionalRFF-HSIC": "#1F4E79",
    "RFF-HSIC": "#4A7C59",
    "ContextAwareConformal": "#B85C38",
    "LearnedFourierIndependence": "#7A4EAB",
    "OnlineConformalAnomaly": "#606C38",
    "CIRCA-NoContext": "#777777",
    "CIRCA-NoTokenBucket": "#D99C2B",
    "CIRCA-NoSelectiveAudit": "#BC4749",
    "CIRCA-HighOnly": "#999999",
    "CIRCA-RawRFFTokenBucket": "#555555",
}


def savefig(fig_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", dpi=240)
    plt.savefig(fig_dir / f"{name}.pdf")
    plt.close()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def dataset_inventory(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("AV2", "Argoverse 2 Sensor", root / "data" / "frames_av2" / "manifest.csv"),
        ("nuImages", "nuImages", root / "data" / "frames_nuimages" / "manifest.csv"),
        ("DROID", "DROID robot manipulation", root / "data" / "frames_droid" / "manifest.csv"),
    ]
    for short, name, path in specs:
        if not path.exists():
            rows.append({"dataset": short, "source": name, "manifest": str(path), "exists": False})
            continue
        df = pd.read_csv(path)
        sizes = (df["width"].astype(str) + "x" + df["height"].astype(str)).value_counts()
        rows.append(
            {
                "dataset": short,
                "source": name,
                "manifest": str(path),
                "exists": True,
                "rows": int(len(df)),
                "sequence_count": int(df["sequence_hint"].nunique()) if "sequence_hint" in df.columns else 0,
                "unique_sha1": int(df["sha1"].nunique()) if "sha1" in df.columns else 0,
                "top_size": str(sizes.index[0]) if not sizes.empty else "",
                "top_size_count": int(sizes.iloc[0]) if not sizes.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def export_tables(out: Path, tables: Path, root: Path) -> dict[str, pd.DataFrame]:
    ablation = read_csv(out / "ablation" / "tables" / "ablation_main_table.csv")
    budget = read_csv(out / "budget_normalized" / "tables" / "budget_normalized_main_table.csv")
    attack = read_csv(out / "attack_strength" / "tables" / "attack_strength_main_table.csv")
    overhead_files = sorted((out / "overhead").glob("*_overhead.csv"))
    overhead = pd.concat([pd.read_csv(p) for p in overhead_files], ignore_index=True) if overhead_files else pd.DataFrame()
    inventory = dataset_inventory(root)

    ablation_selected = ablation[
        ablation["variant"].isin(ABLATION_VARIANTS)
        & (
            (ablation["dataset"] == "local_synthetic")
            | (ablation["dataset"] == "autodl_ros2_twoprocess")
            | ((ablation["dataset"] == "autodl_full") & (ablation["runtime"] == "onnx_cuda") & (ablation["model"] == "mobilenet_v2"))
        )
    ].copy()
    ablation_selected.to_csv(tables / "local_ablation_selected.csv", index=False)

    budget_selected = budget[
        budget["base_method"].isin(CORE_METHODS)
        & (
            (budget["dataset"] == "autodl_ros2_twoprocess")
            | ((budget["dataset"] == "autodl_full") & (budget["runtime"] == "onnx_cuda") & (budget["model"] == "mobilenet_v2"))
        )
    ].copy()
    budget_selected.to_csv(tables / "local_budget_sweep_selected.csv", index=False)

    attack.to_csv(tables / "local_attack_strength_summary.csv", index=False)
    overhead.to_csv(tables / "local_overhead_summary.csv", index=False)
    inventory.to_csv(tables / "local_dataset_inventory.csv", index=False)

    return {
        "ablation": ablation_selected,
        "budget": budget_selected,
        "attack": attack,
        "overhead": overhead,
        "inventory": inventory,
    }


def plot_ablation(df: pd.DataFrame, fig_dir: Path, dataset: str, name: str, title: str) -> None:
    sub = df[df["dataset"] == dataset].copy()
    if dataset == "autodl_full":
        sub = sub[(sub["runtime"] == "onnx_cuda") & (sub["model"] == "mobilenet_v2")]
    if sub.empty:
        return
    sub["variant"] = pd.Categorical(sub["variant"], ABLATION_VARIANTS, ordered=True)
    sub = sub.sort_values("variant")
    x = np.arange(len(sub))
    width = 0.36
    plt.figure(figsize=(11.0, 4.8))
    plt.bar(x - width / 2, sub["attack_recall"], width, label="attack recall", color="#0B6E4F")
    plt.bar(x + width / 2, sub["audit_rate"], width, label="audit rate", color="#D99C2B")
    plt.xticks(x, sub["variant"].astype(str), rotation=25, ha="right")
    plt.ylabel("rate")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    savefig(fig_dir, name)


def plot_budget(df: pd.DataFrame, fig_dir: Path, dataset: str, runtime: str, model: str, name: str, title: str) -> None:
    sub = df[(df["dataset"] == dataset) & (df["runtime"] == runtime) & (df["model"] == model)].copy()
    if sub.empty:
        return
    plt.figure(figsize=(8.8, 5.0))
    for method in CORE_METHODS:
        group = sub[sub["base_method"] == method].sort_values("budget")
        if group.empty:
            continue
        plt.plot(
            group["budget"] * 100.0,
            group["attack_recall"],
            marker="o",
            linewidth=2.8 if method == "CIRCA-RT" else 1.8,
            color=COLORS.get(method, "#555555"),
            label=method,
        )
    plt.xlabel("audit budget (%)")
    plt.ylabel("attack recall")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    savefig(fig_dir, name)


def plot_attack_strength(df: pd.DataFrame, fig_dir: Path) -> None:
    methods = ["CIRCA-RT", "ConditionalRFF-HSIC", "ContextAwareConformal", "LearnedFourierIndependence"]
    plt.figure(figsize=(8.5, 5.0))
    for method in methods:
        group = df[df["method"] == method].sort_values("strength")
        if group.empty:
            continue
        plt.plot(group["strength"], group["attack_recall"], marker="o", color=COLORS.get(method, "#555555"), label=method)
    plt.xlabel("coupled attack strength")
    plt.ylabel("attack recall")
    plt.title("Local synthetic attack-strength sensitivity")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    savefig(fig_dir, "local_attack_strength_recall")


def plot_overhead(df: pd.DataFrame, fig_dir: Path) -> None:
    if df.empty:
        return
    sub = df.copy()
    sub["trace_label"] = sub.get("note", sub["source_trace"]).astype(str)
    pivot = sub.pivot_table(index="component", columns="trace_label", values="mean_ms_per_frame", aggfunc="mean")
    order = ["residualization", "rff_transform", "rolling_score", "full_apply_per_frame"]
    pivot = pivot.reindex([x for x in order if x in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(9.5, 4.8), width=0.75)
    ax.set_ylabel("mean ms/frame")
    ax.set_title("CIRCA-RT CPU monitor overhead breakdown")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    savefig(fig_dir, "local_monitor_overhead_breakdown")


def plot_inventory(df: pd.DataFrame, fig_dir: Path) -> None:
    sub = df[df["exists"] == True].copy()  # noqa: E712
    if sub.empty:
        return
    plt.figure(figsize=(7.2, 4.4))
    plt.bar(sub["dataset"], sub["rows"], color=["#1F4E79", "#B85C38", "#0B6E4F"])
    plt.ylabel("frames")
    plt.title("Local real-frame dataset inventory")
    plt.grid(True, axis="y", alpha=0.25)
    savefig(fig_dir, "local_realframe_inventory")


def write_report(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    report = out / "LOCAL_CPU_COMPLETION_REPORT.md"
    docs_report = ROOT / "docs" / "LOCAL_CPU_EXPERIMENTS_2026_05_24.md"
    docs_report.parent.mkdir(parents=True, exist_ok=True)

    ablation = tables["ablation"]
    budget = tables["budget"]
    attack = tables["attack"]
    overhead = tables["overhead"]
    inventory = tables["inventory"]

    def row_for(df: pd.DataFrame, **query: object) -> pd.Series | None:
        sub = df.copy()
        for col, value in query.items():
            sub = sub[sub[col] == value]
        if sub.empty:
            return None
        return sub.iloc[0]

    ros_circa = row_for(ablation, dataset="autodl_ros2_twoprocess", variant="CIRCA-RT")
    ros_slack = row_for(ablation, dataset="autodl_ros2_twoprocess", variant="CIRCA-RT-Slack")
    onnx_budget = budget[
        (budget["dataset"] == "autodl_full")
        & (budget["runtime"] == "onnx_cuda")
        & (budget["model"] == "mobilenet_v2")
        & (budget["base_method"] == "CIRCA-RT")
        & (budget["budget"].round(2) == 0.10)
    ]
    attack_circa = attack[attack["method"] == "CIRCA-RT"].sort_values("strength")
    full_apply = overhead[overhead["component"] == "full_apply_per_frame"] if not overhead.empty else pd.DataFrame()
    bound_paths = [
        out / "tables" / "audit_bound_ros2_validation.csv",
        out / "tables" / "audit_bound_autodl_full_validation.csv",
    ]
    bound_tables = [pd.read_csv(path) for path in bound_paths if path.exists()]
    audit_bound = pd.concat(bound_tables, ignore_index=True) if bound_tables else pd.DataFrame()

    lines = [
        "# Local CPU Experiment Completion Report",
        "",
        "Date: 2026-05-24",
        "",
        "This report records the non-GPU local work completed before additional AutoDL or AGX Orin runs.",
        "",
        "## Completed Local Tasks",
        "",
        "- Re-ran CIRCA-RT ablations on local synthetic traces and existing AutoDL/ROS2 traces.",
        "- Added `CIRCA-RT-Slack` to the local ablation path and evaluated it against original CIRCA-RT.",
        "- Re-ran matched audit-budget sweeps at 1%, 2%, 5%, 10%, 15%, and 20%.",
        "- Re-ran local attack-strength sensitivity on synthetic coupled attacks.",
        "- Measured CPU monitor overhead breakdown on ROS2 and ONNX MobileNet traces.",
        "- Regenerated real-frame dataset inventory for AV2, nuImages, and DROID manifests.",
        "",
        "## Key Local Results",
        "",
    ]
    if ros_circa is not None and ros_slack is not None:
        lines.append(
            f"- ROS2 CIRCA-RT: recall {ros_circa.attack_recall:.3f}, false alarm {ros_circa.false_alarm_rate:.3f}, "
            f"audit {ros_circa.audit_rate:.3f}, p99 {ros_circa.p99_latency_ms:.3f} ms, miss {ros_circa.deadline_miss_ratio:.4f}."
        )
        lines.append(
            f"- ROS2 CIRCA-RT-Slack: recall {ros_slack.attack_recall:.3f}, false alarm {ros_slack.false_alarm_rate:.3f}, "
            f"audit {ros_slack.audit_rate:.3f}, p99 {ros_slack.p99_latency_ms:.3f} ms, miss {ros_slack.deadline_miss_ratio:.4f}."
        )
    if not onnx_budget.empty:
        r = onnx_budget.iloc[0]
        lines.append(
            f"- ONNX MobileNetV2 at 10% matched audit budget: CIRCA-RT recall {r.attack_recall:.3f}, "
            f"false alarm {r.false_alarm_rate:.3f}, p99 {r.p99_latency_ms:.3f} ms."
        )
    if not attack_circa.empty:
        lo = attack_circa.iloc[0]
        hi = attack_circa.iloc[-1]
        lines.append(
            f"- Synthetic attack-strength sweep: CIRCA-RT recall rises from {lo.attack_recall:.3f} at strength {lo.strength:.2f} "
            f"to {hi.attack_recall:.3f} at strength {hi.strength:.2f}, while audit rises from {lo.audit_rate:.3f} to {hi.audit_rate:.3f}."
        )
    if not full_apply.empty:
        max_mean = float(full_apply["mean_ms_per_frame"].max())
        max_p99 = float(full_apply["p99_ms_per_frame"].max())
        lines.append(f"- CPU full monitor apply overhead is at most {max_mean:.4f} ms/frame mean and {max_p99:.4f} ms/frame p99 in the two measured traces.")
    if not audit_bound.empty:
        failed = int((~audit_bound["pass"].astype(bool)).sum())
        lines.append(f"- Token-bucket audit-cost validation checked {len(audit_bound)} windows across ROS2/AutoDL scored traces with {failed} failures.")

    lines.extend(["", "## Real-Frame Inventory", ""])
    for _, row in inventory.iterrows():
        if not bool(row.get("exists", False)):
            lines.append(f"- {row.dataset}: manifest missing.")
            continue
        seq_label = "sequence" if int(row.sequence_count) == 1 else "sequences"
        lines.append(
            f"- {row.dataset}: {int(row.rows)} frames, {int(row.sequence_count)} {seq_label}, "
            f"{int(row.unique_sha1)} unique SHA1 values, dominant size {row.top_size}."
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `results_local_cpu_completion_20260524/tables/local_ablation_selected.csv`",
            "- `results_local_cpu_completion_20260524/tables/local_budget_sweep_selected.csv`",
            "- `results_local_cpu_completion_20260524/tables/local_attack_strength_summary.csv`",
            "- `results_local_cpu_completion_20260524/tables/local_overhead_summary.csv`",
            "- `results_local_cpu_completion_20260524/tables/local_dataset_inventory.csv`",
            "- `results_local_cpu_completion_20260524/tables/audit_bound_ros2_validation.csv`",
            "- `results_local_cpu_completion_20260524/tables/audit_bound_autodl_full_validation.csv`",
            "- `results_local_cpu_completion_20260524/figures/`",
            "",
            "## Interpretation Boundary",
            "",
            "These results strengthen the local algorithmic and analysis evidence. They do not replace the later AGX Orin 64GB real edge-device validation.",
            "",
        ]
    )
    text = "\n".join(lines)
    report.write_text(text, encoding="utf-8")
    docs_report.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-facing artifacts for local CPU-completable experiments.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results_local_cpu_completion_20260524")
    args = parser.parse_args()

    out = args.out_dir
    fig_dir = out / "figures"
    table_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    tables = export_tables(out, table_dir, ROOT)
    plot_ablation(tables["ablation"], fig_dir, "autodl_ros2_twoprocess", "local_ros2_slack_ablation", "ROS2 CIRCA-RT slack ablation")
    plot_ablation(tables["ablation"], fig_dir, "local_synthetic", "local_synthetic_slack_ablation", "Synthetic CIRCA-RT slack ablation")
    plot_budget(
        tables["budget"],
        fig_dir,
        "autodl_ros2_twoprocess",
        "autodl_ros2_twoprocess",
        "middleware_trace",
        "local_ros2_budget_sweep",
        "ROS2 recall under matched audit budgets",
    )
    plot_budget(
        tables["budget"],
        fig_dir,
        "autodl_full",
        "onnx_cuda",
        "mobilenet_v2",
        "local_onnx_mobilenet_budget_sweep",
        "ONNX MobileNetV2 recall under matched audit budgets",
    )
    plot_attack_strength(tables["attack"], fig_dir)
    plot_overhead(tables["overhead"], fig_dir)
    plot_inventory(tables["inventory"], fig_dir)
    write_report(out, tables)
    print(f"wrote tables to {table_dir}")
    print(f"wrote figures to {fig_dir}")
    print(f"wrote report to {out / 'LOCAL_CPU_COMPLETION_REPORT.md'}")
    print(f"wrote docs report to {ROOT / 'docs' / 'LOCAL_CPU_EXPERIMENTS_2026_05_24.md'}")


if __name__ == "__main__":
    main()
