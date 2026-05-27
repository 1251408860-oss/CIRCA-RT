from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RTSS = ROOT.parent
AGX_TABLES = RTSS / "agx_orin64_ros2" / "final_artifacts_20260526" / "tables"
OUT = ROOT / "paper_draft" / "figures" / "line_candidates"

METHOD_ORDER = ["AlwaysAudit", "ContextAwareConformal", "ConditionalRFF-HSIC", "CIRCA-RT", "CIRCA-RT-Slack"]
METHOD_LABEL = {
    "AlwaysAudit": "Always",
    "ContextAwareConformal": "CAC",
    "ConditionalRFF-HSIC": "CRFF",
    "CIRCA-RT": "CIRCA",
    "CIRCA-RT-Slack": "CIRCA-S",
}
MODEL_ORDER = ["mobilenet_v2", "resnet18", "resnet50"]
MODEL_LABEL = {"mobilenet_v2": "MNetV2", "resnet18": "R18", "resnet50": "R50"}
DATASET_ORDER = ["av2", "droid", "nuimages"]
DATASET_LABEL = {"av2": "AV2", "droid": "DROID", "nuimages": "nuImages"}

LINE_STYLES = {
    "AlwaysAudit": ("black", "o", "-"),
    "ContextAwareConformal": ("0.35", "s", "--"),
    "ConditionalRFF-HSIC": ("0.55", "^", "-."),
    "CIRCA-RT": ("0.15", "D", ":"),
    "CIRCA-RT-Slack": ("0.0", "x", "-"),
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, ylabel: str, panel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.text(0.5, -0.34, panel, transform=ax.transAxes, ha="center", va="top", fontsize=7)


def plot_method_lines(ax: plt.Axes, data: pd.DataFrame, x_col: str, y_col: str, x_vals: list, x_labels: list[str], ylabel: str, panel: str, scale: float = 1.0) -> None:
    for method in METHOD_ORDER:
        sub = data[data["method"] == method].copy()
        if sub.empty:
            continue
        y = []
        for x in x_vals:
            row = sub[sub[x_col] == x]
            y.append(float(row[y_col].iloc[0]) * scale if not row.empty else np.nan)
        color, marker, ls = LINE_STYLES[method]
        ax.plot(np.arange(len(x_vals)), y, label=METHOD_LABEL[method], color=color, marker=marker, linestyle=ls, linewidth=1.0, markersize=3.2)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels(x_labels)
    style_axis(ax, ylabel, panel)


def load_main() -> pd.DataFrame:
    df = pd.read_csv(AGX_TABLES / "agx_realframe_main_all.csv")
    return df[df["method"].isin(METHOD_ORDER)].copy()


def deadline_baseline_lines() -> None:
    df = load_main()
    agg = df.groupby(["deadline_ms_eval", "method"], as_index=False).agg(
        attack_recall=("attack_recall", "mean"),
        audit_rate=("audit_rate", "mean"),
        p99_latency_ms=("p99_latency_ms", "mean"),
        deadline_miss_ratio=("deadline_miss_ratio", "mean"),
    )
    deadlines = sorted(float(x) for x in agg["deadline_ms_eval"].unique())
    labels = ["33 ms" if abs(x - 33.333) < 0.01 else f"{x:g} ms" for x in deadlines]

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.85))
    plot_method_lines(axs[0], agg, "deadline_ms_eval", "attack_recall", deadlines, labels, "Recall (%)", "(a) Attack recall", 100)
    plot_method_lines(axs[1], agg, "deadline_ms_eval", "audit_rate", deadlines, labels, "Audit rate (%)", "(b) Audit demand", 100)
    plot_method_lines(axs[2], agg, "deadline_ms_eval", "p99_latency_ms", deadlines, labels, "P99 latency (ms)", "(c) Tail latency", 1)
    plot_method_lines(axs[3], agg, "deadline_ms_eval", "deadline_miss_ratio", deadlines, labels, "Miss (%)", "(d) Deadline miss", 100)
    axs[0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(2.35, 1.28))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.42, wspace=0.42)
    finish(fig, "deadline_baseline_lines")


def model_baseline_lines() -> None:
    df = load_main()
    agg = df.groupby(["primary_model", "method"], as_index=False).agg(
        attack_recall=("attack_recall", "mean"),
        audit_rate=("audit_rate", "mean"),
        p99_latency_ms=("p99_latency_ms", "mean"),
        deadline_miss_ratio=("deadline_miss_ratio", "mean"),
    )
    labels = [MODEL_LABEL[x] for x in MODEL_ORDER]

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.85))
    plot_method_lines(axs[0], agg, "primary_model", "attack_recall", MODEL_ORDER, labels, "Recall (%)", "(a) Attack recall", 100)
    plot_method_lines(axs[1], agg, "primary_model", "audit_rate", MODEL_ORDER, labels, "Audit rate (%)", "(b) Audit demand", 100)
    plot_method_lines(axs[2], agg, "primary_model", "p99_latency_ms", MODEL_ORDER, labels, "P99 latency (ms)", "(c) Tail latency", 1)
    plot_method_lines(axs[3], agg, "primary_model", "deadline_miss_ratio", MODEL_ORDER, labels, "Miss (%)", "(d) Deadline miss", 100)
    axs[0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(2.35, 1.28))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.42, wspace=0.42)
    finish(fig, "model_baseline_lines")


def dataset_baseline_lines() -> None:
    df = load_main()
    agg = df.groupby(["dataset", "method"], as_index=False).agg(
        attack_recall=("attack_recall", "mean"),
        audit_rate=("audit_rate", "mean"),
        p99_latency_ms=("p99_latency_ms", "mean"),
        deadline_miss_ratio=("deadline_miss_ratio", "mean"),
    )
    labels = [DATASET_LABEL[x] for x in DATASET_ORDER]

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.85))
    plot_method_lines(axs[0], agg, "dataset", "attack_recall", DATASET_ORDER, labels, "Recall (%)", "(a) Attack recall", 100)
    plot_method_lines(axs[1], agg, "dataset", "audit_rate", DATASET_ORDER, labels, "Audit rate (%)", "(b) Audit demand", 100)
    plot_method_lines(axs[2], agg, "dataset", "p99_latency_ms", DATASET_ORDER, labels, "P99 latency (ms)", "(c) Tail latency", 1)
    plot_method_lines(axs[3], agg, "dataset", "deadline_miss_ratio", DATASET_ORDER, labels, "Miss (%)", "(d) Deadline miss", 100)
    axs[0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(2.35, 1.28))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.42, wspace=0.42)
    finish(fig, "dataset_baseline_lines")


def audit_bound_window_lines() -> None:
    main = pd.read_csv(AGX_TABLES / "audit_bound_validation_all.csv")
    pressure = pd.read_csv(AGX_TABLES / "pressure_audit_bound_validation_all.csv")
    rows = []
    for name, df in [("Main AGX", main), ("Reduced pressure", pressure)]:
        df = df[df["window"].isin([32, 64, 128, 256])].copy()
        g = df.groupby("window", as_index=False).agg(
            min_margin=("margin_ms", "min"),
            p50_margin=("margin_ms", "median"),
            p10_margin=("margin_ms", lambda s: float(np.quantile(s.astype(float), 0.10))),
        )
        g["series"] = name
        rows.append(g)
    data = pd.concat(rows, ignore_index=True)
    windows = [32, 64, 128, 256]

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.25))
    for series, color, marker, ls in [("Main AGX", "black", "o", "-"), ("Reduced pressure", "0.45", "s", "--")]:
        sub = data[data["series"] == series]
        y = [float(sub.loc[sub["window"] == w, "min_margin"].iloc[0]) for w in windows]
        ax.plot(windows, y, label=series, color=color, marker=marker, linestyle=ls, linewidth=1.0, markersize=3.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Demand-bound window")
    ax.set_ylabel("Min margin (ms)")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend(frameon=False, loc="upper left")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.24)
    finish(fig, "audit_bound_window_lines")


def main() -> None:
    setup_style()
    deadline_baseline_lines()
    model_baseline_lines()
    dataset_baseline_lines()
    audit_bound_window_lines()
    print(f"Wrote line candidates to {OUT}")


if __name__ == "__main__":
    main()
