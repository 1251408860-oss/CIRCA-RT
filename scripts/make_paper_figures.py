from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


OUT = ROOT / "results_local_complete"
FIG = OUT / "figures"
TABLE = OUT / "tables"
DOC = ROOT / "docs"
for directory in [FIG, TABLE, DOC]:
    directory.mkdir(parents=True, exist_ok=True)

CORE_METHODS = [
    "CIRCA-RT",
    "ContextAwareConformal",
    "ConditionalRFF-HSIC",
    "LearnedFourierIndependence",
    "CATCH",
    "DCdetector",
    "TranAD",
    "RandomBudgetAudit",
    "PeriodicAudit",
    "AlwaysAudit",
]

COLORS = {
    "CIRCA-RT": "#0B6E4F",
    "ContextAwareConformal": "#B85C38",
    "ConditionalRFF-HSIC": "#1F4E79",
    "LearnedFourierIndependence": "#7A4EAB",
    "CATCH": "#D99C2B",
    "DCdetector": "#606C38",
    "TranAD": "#BC4749",
    "RandomBudgetAudit": "#777777",
    "PeriodicAudit": "#999999",
    "AlwaysAudit": "#222222",
}


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=240)
    plt.savefig(FIG / f"{name}.pdf")
    plt.close()


def read_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ROOT / "results_recent_deep_baselines_final" / "tables"
    default = pd.read_csv(base / "paper_core_default_table.csv")
    budget = pd.read_csv(base / "paper_core_budget_table.csv")
    return default, budget


def scatter_pareto(df: pd.DataFrame, *, dataset: str, x_col: str, title: str, name: str) -> None:
    sub = df[(df["dataset"] == dataset) & (df["method"].isin(CORE_METHODS))].copy()
    if sub.empty:
        return
    if dataset == "autodl_full":
        sub = sub[(sub["runtime"] == "onnx_cuda") & (sub["model"] == "mobilenet_v2")]
    plt.figure(figsize=(8.5, 5.0))
    for method, group in sub.groupby("method"):
        plt.scatter(
            group[x_col],
            group["attack_recall"],
            s=90 if method == "CIRCA-RT" else 62,
            color=COLORS.get(method, "#555555"),
            edgecolor="black" if method == "CIRCA-RT" else "none",
            linewidth=1.0,
            label=method,
            alpha=0.90,
        )
    plt.xlabel(x_col.replace("_", " "))
    plt.ylabel("attack recall")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    savefig(name)


def budget_recall_plot(budget: pd.DataFrame, *, dataset: str, runtime: str, model: str, name: str, title: str) -> None:
    sub = budget[
        (budget["dataset"] == dataset)
        & (budget["runtime"] == runtime)
        & (budget["model"] == model)
        & (budget["base_method"].isin(CORE_METHODS))
    ].copy()
    if sub.empty:
        return
    methods = [m for m in CORE_METHODS if m in set(sub["base_method"])]
    plt.figure(figsize=(9.0, 5.2))
    for method in methods:
        g = sub[sub["base_method"] == method].sort_values("budget")
        plt.plot(
            g["budget"] * 100.0,
            g["attack_recall"],
            marker="o",
            linewidth=2.6 if method == "CIRCA-RT" else 1.7,
            color=COLORS.get(method, "#555555"),
            label=method,
        )
    plt.xlabel("audit budget (%)")
    plt.ylabel("attack recall")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    savefig(name)


def latency_bar(df: pd.DataFrame) -> None:
    methods = ["CIRCA-RT", "ContextAwareConformal", "TranAD", "CATCH", "DCdetector", "AlwaysAudit"]
    sub = df[(df["dataset"] == "autodl_ros2_twoprocess") & (df["method"].isin(methods))].copy()
    sub["method"] = pd.Categorical(sub["method"], methods, ordered=True)
    sub.sort_values("method", inplace=True)
    x = np.arange(len(sub))
    plt.figure(figsize=(8.5, 4.7))
    plt.bar(x, sub["p99_latency_ms"], color=[COLORS.get(m, "#555555") for m in sub["method"]])
    plt.xticks(x, sub["method"], rotation=25, ha="right")
    plt.ylabel("p99 latency (ms)")
    plt.title("ROS2/DDS p99 latency with monitoring")
    plt.grid(True, axis="y", alpha=0.25)
    savefig("ros2_p99_latency_bar")


def gpu_model_tradeoff(df: pd.DataFrame) -> None:
    sub = df[
        (df["dataset"] == "autodl_full")
        & (df["method"].isin(["CIRCA-RT", "CATCH", "DCdetector", "TranAD"]))
    ].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, runtime in zip(axes, ["onnx_cuda", "torch_cuda"]):
        r = sub[sub["runtime"] == runtime]
        for method, group in r.groupby("method"):
            ax.scatter(
                group["audit_rate"],
                group["attack_recall"],
                s=80 if method == "CIRCA-RT" else 55,
                color=COLORS.get(method, "#555555"),
                edgecolor="black" if method == "CIRCA-RT" else "none",
                label=method,
                alpha=0.90,
            )
            for _, row in group.iterrows():
                ax.annotate(str(row["model"]).replace("_", ""), (row["audit_rate"], row["attack_recall"]), fontsize=7, alpha=0.7)
        ax.set_title(runtime)
        ax.set_xlabel("audit rate")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("attack recall")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(FIG / "gpu_model_recall_audit_tradeoff.png", dpi=240)
    plt.savefig(FIG / "gpu_model_recall_audit_tradeoff.pdf")
    plt.close()


def ablation_plot() -> None:
    path = OUT / "ablation" / "tables" / "ablation_main_table.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[(df["dataset"] == "autodl_ros2_twoprocess")].copy()
    if sub.empty:
        sub = df[(df["dataset"] == "local_synthetic")].copy()
    sub = sub[sub["model"].isin(["middleware_trace", "synthetic_trace"])]
    order = ["CIRCA-RT", "CIRCA-NoContext", "CIRCA-NoTokenBucket", "CIRCA-NoSelectiveAudit", "CIRCA-HighOnly", "CIRCA-RawRFFTokenBucket"]
    sub = sub[sub["variant"].isin(order)]
    sub["variant"] = pd.Categorical(sub["variant"], order, ordered=True)
    sub.sort_values("variant", inplace=True)
    x = np.arange(len(sub))
    width = 0.36
    plt.figure(figsize=(10.0, 4.8))
    plt.bar(x - width / 2, sub["attack_recall"], width, label="recall", color="#0B6E4F")
    plt.bar(x + width / 2, sub["audit_rate"], width, label="audit rate", color="#D99C2B")
    plt.xticks(x, sub["variant"], rotation=25, ha="right")
    plt.ylabel("rate")
    plt.title("CIRCA-RT module ablation")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    savefig("circa_ablation_recall_audit")


def attack_strength_plot() -> None:
    path = OUT / "attack_strength" / "tables" / "attack_strength_main_table.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    methods = ["CIRCA-RT", "ContextAwareConformal", "ConditionalRFF-HSIC", "LearnedFourierIndependence"]
    plt.figure(figsize=(8.5, 5.0))
    for method in methods:
        g = df[df["method"] == method].sort_values("strength")
        if g.empty:
            continue
        plt.plot(g["strength"], g["attack_recall"], marker="o", label=method, color=COLORS.get(method, "#555555"))
    plt.xlabel("coupled attack strength")
    plt.ylabel("attack recall")
    plt.title("Sensitivity to coupled attack strength")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    savefig("attack_strength_recall")


def export_latex(default: pd.DataFrame, budget: pd.DataFrame) -> None:
    ros = default[
        (default["dataset"] == "autodl_ros2_twoprocess")
        & (default["method"].isin(["CIRCA-RT", "ContextAwareConformal", "TranAD", "CATCH", "DCdetector", "AlwaysAudit"]))
    ][["method", "attack_recall", "false_alarm_rate", "audit_rate", "p99_latency_ms", "latency_inflation_mean_ms"]]
    ros.to_csv(TABLE / "paper_ros2_default_selected.csv", index=False)
    ros.to_latex(TABLE / "paper_ros2_default_selected.tex", index=False, float_format="%.3f")

    b = budget[
        (budget["dataset"] == "autodl_ros2_twoprocess")
        & (budget["budget"].round(2).isin([0.10, 0.15]))
        & (budget["base_method"].isin(["CIRCA-RT", "ContextAwareConformal", "TranAD", "CATCH", "DCdetector", "ConditionalRFF-HSIC"]))
    ][["budget", "base_method", "attack_recall", "false_alarm_rate", "p99_latency_ms", "latency_inflation_mean_ms"]]
    b.to_csv(TABLE / "paper_ros2_budget_selected.csv", index=False)
    b.to_latex(TABLE / "paper_ros2_budget_selected.tex", index=False, float_format="%.3f")


def write_report(default: pd.DataFrame, budget: pd.DataFrame) -> None:
    report = DOC / "LOCAL_COMPLETE_EXPERIMENT_REPORT.md"
    ros = default[
        (default["dataset"] == "autodl_ros2_twoprocess")
        & (default["method"].isin(["CIRCA-RT", "ContextAwareConformal", "TranAD", "CATCH", "DCdetector"]))
    ].sort_values("attack_recall", ascending=False)
    lines = [
        "# Local Complete Experiment Report",
        "",
        "Updated: 2026-05-19",
        "",
        "## Local Work Completed",
        "",
        "- Re-ran local split synthetic experiments.",
        "- Re-ran local sensitivity experiments.",
        "- Added and ran CIRCA-RT module ablations on local synthetic and downloaded AutoDL/ROS2 traces.",
        "- Added and ran coupled attack-strength sensitivity.",
        "- Generated paper-ready figures, CSV tables, and LaTeX tables from AutoDL/ROS2/deep-baseline results.",
        "",
        "## Key ROS2/DDS Default Results",
        "",
    ]
    for _, row in ros.iterrows():
        lines.append(
            f"- `{row.method}`: recall {row.attack_recall:.3f}, false alarm {row.false_alarm_rate:.3f}, "
            f"audit {row.audit_rate:.3f}, p99 {row.p99_latency_ms:.3f} ms, overhead {row.latency_inflation_mean_ms:.3f} ms."
        )
    lines.extend(
        [
            "",
            "## Main Output Directories",
            "",
            "- `results_local_complete/figures/`",
            "- `results_local_complete/tables/`",
            "- `results_local_complete/ablation/`",
            "- `results_local_complete/attack_strength/`",
            "- `results_split/`",
            "- `results_split/sensitivity/`",
            "",
            "## Paper Claim Boundary",
            "",
            "CIRCA-RT should be presented as a low-audit, low-overhead real-time monitor with competitive detection under strict budget constraints. Do not claim universal recall dominance.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    default, budget = read_tables()
    scatter_pareto(
        default,
        dataset="autodl_ros2_twoprocess",
        x_col="audit_rate",
        title="ROS2/DDS recall vs audit rate",
        name="ros2_pareto_recall_vs_audit",
    )
    scatter_pareto(
        default,
        dataset="autodl_ros2_twoprocess",
        x_col="latency_inflation_mean_ms",
        title="ROS2/DDS recall vs latency overhead",
        name="ros2_pareto_recall_vs_overhead",
    )
    scatter_pareto(
        default,
        dataset="autodl_full",
        x_col="audit_rate",
        title="ONNX MobileNetV2 recall vs audit rate",
        name="onnx_mobilenet_pareto_recall_vs_audit",
    )
    budget_recall_plot(
        budget,
        dataset="autodl_ros2_twoprocess",
        runtime="autodl_ros2_twoprocess",
        model="middleware_trace",
        name="ros2_budget_recall",
        title="ROS2/DDS recall under matched audit budgets",
    )
    budget_recall_plot(
        budget,
        dataset="autodl_full",
        runtime="onnx_cuda",
        model="mobilenet_v2",
        name="onnx_mobilenet_budget_recall",
        title="ONNX MobileNetV2 recall under matched audit budgets",
    )
    latency_bar(default)
    gpu_model_tradeoff(default)
    ablation_plot()
    attack_strength_plot()
    export_latex(default, budget)
    write_report(default, budget)
    print(f"wrote figures to {FIG}")
    print(f"wrote tables to {TABLE}")
    print(f"wrote report to {DOC / 'LOCAL_COMPLETE_EXPERIMENT_REPORT.md'}")


if __name__ == "__main__":
    main()

