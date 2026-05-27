from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RTSS = ROOT.parent
AGX_TABLES = RTSS / "agx_orin64_ros2" / "final_artifacts_20260526" / "tables"
AGX_WORK = RTSS / "agx_orin64_ros2" / "work" / "agx_orin64_run_package_20260525_submission_full"
OUT = ROOT / "paper_draft" / "figures"


METHOD_ORDER = [
    "AlwaysAudit",
    "ContextAwareConformal",
    "ConditionalRFF-HSIC",
    "CIRCA-RT",
    "CIRCA-RT-Slack",
]
METHOD_LABEL = {
    "AlwaysAudit": "Always",
    "ContextAwareConformal": "CAC",
    "ConditionalRFF-HSIC": "CRFF",
    "CIRCA-RT": "CIRCA",
    "CIRCA-RT-Slack": "CIRCA-S",
    "Risk+DeferrableServer": "DS",
    "Risk+DeferrableServer+Slack": "DS+S",
    "Risk+SporadicServer": "SS",
    "Risk+SporadicServer+Slack": "SS+S",
    "Risk+CBS": "CBS",
    "Risk+CBS+Slack": "CBS+S",
}
SERVER_ORDER = [
    "CIRCA-RT-Slack",
    "Risk+DeferrableServer",
    "Risk+DeferrableServer+Slack",
    "Risk+SporadicServer",
    "Risk+SporadicServer+Slack",
    "Risk+CBS",
    "Risk+CBS+Slack",
]
MODEL_ORDER = ["mobilenet_v2", "resnet18", "resnet50", "squeezenet1_1"]
MODEL_LABEL = {
    "mobilenet_v2": "MNetV2",
    "resnet18": "R18",
    "resnet50": "R50",
    "squeezenet1_1": "SqNet",
}
RUNTIME_ORDER = ["torch_cuda", "onnx_cuda", "onnx_tensorrt"]
RUNTIME_LABEL = {
    "torch_cuda": "PyTorch",
    "onnx_cuda": "ONNX-CUDA",
    "onnx_tensorrt": "TRT",
}
SCENARIO_LABEL = {
    "coupled_semantic_timing_attack": "Coupled",
    "gpu_interference": "GPU",
    "mode_shift": "Mode",
    "nominal": "Nominal",
    "semantic_corruption": "Semantic",
    "stale_replay_attack": "Replay",
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


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.5, -0.36, label, transform=ax.transAxes, ha="center", va="top", fontsize=7)


def bar_panel(ax: plt.Axes, labels: list[str], values: list[float], ylabel: str, panel: str, ylim: tuple[float, float] | None = None) -> None:
    hatches = ["", "//", "\\\\", "..", "xx", "++", "--", "oo"]
    colors = ["0.10", "0.40", "0.65", "0.82", "1.0", "0.55", "0.92", "0.72"]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors[: len(labels)], edgecolor="black", linewidth=0.6)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    style_axis(ax, ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    panel_label(ax, panel)


def plot_main_tradeoff() -> None:
    df = pd.read_csv(AGX_TABLES / "agx_realframe_main_all.csv")
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    g = df.groupby("method", as_index=False).agg(
        attack_recall=("attack_recall", "mean"),
        audit_rate=("audit_rate", "mean"),
        p99_latency_ms=("p99_latency_ms", "mean"),
        deadline_miss_ratio=("deadline_miss_ratio", "mean"),
    )
    g["method"] = pd.Categorical(g["method"], METHOD_ORDER, ordered=True)
    g = g.sort_values("method")
    labels = [METHOD_LABEL[m] for m in g["method"].astype(str)]

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.85))
    bar_panel(axs[0], labels, (100 * g["attack_recall"]).tolist(), "Recall (%)", "(a) Attack recall", (0, 105))
    bar_panel(axs[1], labels, (100 * g["audit_rate"]).tolist(), "Audit rate (%)", "(b) Audit demand", (0, 105))
    bar_panel(axs[2], labels, g["p99_latency_ms"].tolist(), "P99 latency (ms)", "(c) Tail latency", (0, max(22, float(g["p99_latency_ms"].max()) * 1.15)))
    bar_panel(axs[3], labels, (100 * g["deadline_miss_ratio"]).tolist(), "Miss (%)", "(d) Deadline miss", (0, max(0.35, float((100 * g["deadline_miss_ratio"]).max()) * 1.25)))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.96, bottom=0.42, wspace=0.42)
    finish(fig, "fig3_main_agx_tradeoff_multipanel")


def plot_server_replay() -> None:
    df = pd.read_csv(ROOT / "results_agx_offline_rtss_hardening_pressure_20260526" / "tables" / "server_baseline_overall.csv")
    df = df[df["method"].isin(SERVER_ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], SERVER_ORDER, ordered=True)
    df = df.sort_values("method")
    labels = [METHOD_LABEL[m] for m in df["method"].astype(str)]

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.85))
    bar_panel(axs[0], labels, (100 * df["audit_rate"]).tolist(), "Audit rate (%)", "(a) Audit rate", (0, 5.5))
    bar_panel(axs[1], labels, (100 * df["audit_attack_recall"]).tolist(), "Audit hit (%)", "(b) Audit hit", (0, 9.5))
    bar_panel(axs[2], labels, (100 * df["deadline_miss_ratio"]).tolist(), "Miss (%)", "(c) Deadline miss", (0, max(0.025, float((100 * df["deadline_miss_ratio"]).max()) * 1.3)))
    bar_panel(axs[3], labels, (100 * df["deadline_bound_failures"]).tolist(), "Bound fail (%)", "(d) Per-frame bound", (0, 16.5))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.96, bottom=0.42, wspace=0.42)
    finish(fig, "fig4_server_replay_slack_multipanel")


def plot_server_bound_core() -> None:
    df = pd.read_csv(ROOT / "results_agx_offline_rtss_hardening_pressure_20260526" / "tables" / "server_baseline_overall.csv")
    df = df[df["method"].isin(SERVER_ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], SERVER_ORDER, ordered=True)
    df = df.sort_values("method")
    labels = [METHOD_LABEL[m] for m in df["method"].astype(str)]
    values = (100 * df["deadline_bound_failures"]).astype(float).tolist()

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.25))
    bar_panel(ax, labels, values, "Per-frame bound failure (%)", "", (0, max(16.5, max(values) * 1.15)))
    ax.axhline(0, color="black", linewidth=0.6)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.96, bottom=0.34)
    finish(fig, "fig4_server_bound_failure_core")


def load_backend_overview() -> pd.DataFrame:
    files = [
        AGX_WORK / "results_codex_full_safe_20260526_001" / "tables" / "agx_backend_latency_overview.csv",
        AGX_WORK / "results_codex_backend_resnet50_squeezenet_20260526_001" / "tables" / "agx_backend_latency_overview.csv",
    ]
    frames = [pd.read_csv(p) for p in files if p.exists()]
    if not frames:
        raise FileNotFoundError("No backend overview CSV files found")
    return pd.concat(frames, ignore_index=True)


def grouped_runtime_bars(ax: plt.Axes, data: pd.DataFrame, value: str, ylabel: str, panel: str) -> None:
    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    hatches = ["", "//", "\\\\"]
    colors = ["0.25", "0.68", "1.0"]
    for j, runtime in enumerate(RUNTIME_ORDER):
        vals = []
        for model in MODEL_ORDER:
            sub = data[(data["runtime"] == runtime) & (data["model"] == model)]
            vals.append(float(sub[value].iloc[0]) if not sub.empty else np.nan)
        bars = ax.bar(x + (j - 1) * width, vals, width, label=RUNTIME_LABEL[runtime], color=colors[j], edgecolor="black", linewidth=0.6)
        for b in bars:
            b.set_hatch(hatches[j])
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    style_axis(ax, ylabel)
    panel_label(ax, panel)


def plot_platform_boundary() -> None:
    backend = load_backend_overview()
    backend = backend[backend["runtime"].isin(RUNTIME_ORDER) & backend["model"].isin(MODEL_ORDER)].copy()
    agg = backend.groupby(["runtime", "model"], as_index=False).agg(
        mean_p99_ms=("p99_ms", "mean"),
        max_p99_ms=("p99_ms", "max"),
        max_miss_pct=("deadline_miss_ratio", lambda s: float(100 * s.max())),
    )

    ros2 = pd.read_csv(AGX_TABLES / "agx_ros2_closed_loop_latency_table.csv")
    ros2 = ros2[(ros2["method"] == "CIRCA-RT-Slack") & (ros2["scenario"] != "nominal_calibration")].copy()
    ros2_order = ["nominal", "mode_shift", "semantic_corruption", "stale_replay_attack", "gpu_interference", "coupled_semantic_timing_attack"]
    ros2["scenario"] = pd.Categorical(ros2["scenario"], ros2_order, ordered=True)
    ros2g = ros2.groupby("scenario", as_index=False).agg(p99_latency_ms=("p99_latency_ms", "mean"))
    ros2g = ros2g.dropna().sort_values("scenario")

    pressure = pd.read_csv(AGX_TABLES / "pressure_audit_bound_validation_all.csv")
    standard_windows = [32, 64, 128, 256]
    windows = [w for w in standard_windows if w in set(int(x) for x in pressure["window"].dropna().unique())]
    margin_data = [pressure.loc[pressure["window"] == w, "margin_ms"].astype(float).to_numpy() for w in windows]

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.85))
    grouped_runtime_bars(axs[0], agg, "mean_p99_ms", "Mean P99 (ms)", "(a) Backend mean")
    axs[0].legend(frameon=False, ncol=1, loc="upper left")

    grouped_runtime_bars(axs[1], agg, "max_p99_ms", "Max P99 (ms)", "(b) Backend tail")

    labels = [SCENARIO_LABEL.get(str(s), str(s)) for s in ros2g["scenario"].astype(str)]
    bar_panel(axs[2], labels, ros2g["p99_latency_ms"].astype(float).tolist(), "ROS2 P99 (ms)", "(c) ROS2 boundary", (0, max(450, float(ros2g["p99_latency_ms"].max()) * 1.1)))
    axs[2].axhline(33.333, color="black", linewidth=0.8, linestyle=":")
    axs[2].text(0.02, 0.12, "33 ms", transform=axs[2].transAxes, fontsize=6)

    bp = axs[3].boxplot(margin_data, patch_artist=True, widths=0.55, showfliers=False)
    for box in bp["boxes"]:
        box.set(facecolor="0.88", edgecolor="black", linewidth=0.6, hatch="//")
    for key in ["whiskers", "caps", "medians"]:
        for item in bp[key]:
            item.set(color="black", linewidth=0.6)
    axs[3].set_xticks(np.arange(1, len(windows) + 1))
    axs[3].set_xticklabels([str(w) for w in windows])
    style_axis(axs[3], "Margin (ms)")
    axs[3].set_xlabel("Window")
    panel_label(axs[3], "(d) Pressure bound")

    fig.subplots_adjust(left=0.055, right=0.995, top=0.96, bottom=0.42, wspace=0.42)
    finish(fig, "fig5_backend_ros2_pressure_multipanel")


def plot_pressure_margin_core() -> None:
    pressure = pd.read_csv(AGX_TABLES / "pressure_audit_bound_validation_all.csv")
    standard_windows = [32, 64, 128, 256]
    windows = [w for w in standard_windows if w in set(int(x) for x in pressure["window"].dropna().unique())]
    margin_data = [pressure.loc[pressure["window"] == w, "margin_ms"].astype(float).to_numpy() for w in windows]

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.25))
    bp = ax.boxplot(margin_data, patch_artist=True, widths=0.55, showfliers=False)
    for box in bp["boxes"]:
        box.set(facecolor="0.88", edgecolor="black", linewidth=0.6, hatch="//")
    for key in ["whiskers", "caps", "medians"]:
        for item in bp[key]:
            item.set(color="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(np.arange(1, len(windows) + 1))
    ax.set_xticklabels([str(w) for w in windows])
    style_axis(ax, "Audit-bound margin (ms)")
    ax.set_xlabel("Demand-bound window")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.96, bottom=0.28)
    finish(fig, "fig5_pressure_margin_core")


def main() -> None:
    setup_style()
    plot_main_tradeoff()
    plot_server_replay()
    plot_platform_boundary()
    plot_server_bound_core()
    plot_pressure_margin_core()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
