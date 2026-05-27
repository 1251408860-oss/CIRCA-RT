from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "autodl_runs" / "t4_ort_trt_fixed_20260524_1138"
FULL = SOURCE / "results_t4_ort_trt_fixed_20260524_112905"
BUDGET = SOURCE / "results_t4_ort_trt_budget_20260524_113735"

OUT = ROOT / "results_t4_ort_trt_integrated"
FIG = OUT / "figures"
TABLE = OUT / "tables"
LOCAL_FIG = ROOT / "results_local_complete" / "figures"
PAPER = ROOT / "paper_draft"

for directory in [FIG, TABLE, LOCAL_FIG, PAPER]:
    directory.mkdir(parents=True, exist_ok=True)

DEADLINES_MS = [25.0, 30.0, 33.333, 35.0, 40.0, 50.0]

METHODS = [
    "CIRCA-RT",
    "RFF-HSIC",
    "ConditionalRFF-HSIC",
    "ContextAwareConformal",
    "AlwaysAudit",
]

COLORS = {
    "CIRCA-RT": "#0B6E4F",
    "RFF-HSIC": "#1F4E79",
    "ConditionalRFF-HSIC": "#7A4EAB",
    "ContextAwareConformal": "#B85C38",
    "AlwaysAudit": "#222222",
    "onnx_cuda": "#3B6EA8",
    "onnx_tensorrt": "#0B6E4F",
}

RUNTIME_LABELS = {
    "onnx_cuda": "ONNX CUDA",
    "onnx_tensorrt": "ONNX TensorRT EP",
}

MODEL_LABELS = {
    "mobilenet_v2": "MobileNetV2",
    "resnet18": "ResNet18",
}


def savefig(name: str, *, mirror: bool = True) -> None:
    plt.tight_layout()
    png = FIG / f"{name}.png"
    pdf = FIG / f"{name}.pdf"
    plt.savefig(png, dpi=240)
    plt.savefig(pdf)
    plt.close()
    if mirror:
        shutil.copy2(png, LOCAL_FIG / png.name)
        shutil.copy2(pdf, LOCAL_FIG / pdf.name)


def method_label(method: str) -> str:
    return method.replace("ConditionalRFF-HSIC", "CondRFF-HSIC").replace("ContextAwareConformal", "CtxConformal")


def parse_raw_path(path: Path) -> dict[str, object]:
    rel = path.relative_to(FULL / "raw")
    parts = rel.parts
    return {
        "runtime": parts[0],
        "provider": parts[1],
        "model": parts[2],
        "seed": int(parts[3].replace("seed", "")),
        "trace_type": parts[4],
        "method": parts[5],
    }


def read_main_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monitoring = pd.read_csv(FULL / "tables" / "monitoring_main_table.csv")
    latency = pd.read_csv(FULL / "tables" / "latency_full_table.csv")
    budget = pd.read_csv(BUDGET / "tables" / "budget_normalized_main_table.csv")
    return monitoring, latency, budget


def build_deadline_sensitivity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((FULL / "raw").rglob("scored.csv")):
        meta = parse_raw_path(path)
        if str(meta["method"]) not in METHODS:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        latency = df["e2e_latency_ms"].astype(float)
        for deadline in DEADLINES_MS:
            row = dict(meta)
            row.update(
                {
                    "deadline_ms": float(deadline),
                    "n": int(len(df)),
                    "audit_rate": float(df["audit"].astype(float).mean()),
                    "attack_recall": float(df.loc[df["label"].astype(int) == 1, "alarm"].astype(float).mean())
                    if (df["label"].astype(int) == 1).any()
                    else 0.0,
                    "false_alarm_rate": float(df.loc[df["label"].astype(int) == 0, "alarm"].astype(float).mean())
                    if (df["label"].astype(int) == 0).any()
                    else 0.0,
                    "miss_ratio_at_deadline": float((latency > deadline).mean()),
                    "p95_latency_ms": float(latency.quantile(0.95)),
                    "p99_latency_ms": float(latency.quantile(0.99)),
                    "p999_latency_ms": float(latency.quantile(0.999)),
                    "p99_slack_ms": float(deadline - latency.quantile(0.99)),
                    "p999_slack_ms": float(deadline - latency.quantile(0.999)),
                    "min_slack_ms": float((deadline - latency).min()),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    group_cols = ["runtime", "provider", "model", "trace_type", "method", "deadline_ms"]
    metrics = [
        "n",
        "audit_rate",
        "attack_recall",
        "false_alarm_rate",
        "miss_ratio_at_deadline",
        "p95_latency_ms",
        "p99_latency_ms",
        "p999_latency_ms",
        "p99_slack_ms",
        "p999_slack_ms",
        "min_slack_ms",
    ]
    summary = out.groupby(group_cols, as_index=False)[metrics].mean()
    summary.to_csv(TABLE / "t4_deadline_sensitivity_by_trace.csv", index=False)
    out.to_csv(TABLE / "t4_deadline_sensitivity_by_seed.csv", index=False)
    return summary


def export_selected_tables(monitoring: pd.DataFrame, latency: pd.DataFrame, budget: pd.DataFrame, deadline: pd.DataFrame) -> None:
    selected = monitoring[monitoring["method"].isin(METHODS)].copy()
    selected["runtime_label"] = selected["runtime"].map(RUNTIME_LABELS)
    selected["model_label"] = selected["model"].map(MODEL_LABELS)
    keep = [
        "runtime_label",
        "model_label",
        "method",
        "attack_recall",
        "false_alarm_rate",
        "audit_rate",
        "p99_latency_ms",
        "p999_latency_ms",
        "deadline_miss_ratio",
    ]
    selected[keep].to_csv(TABLE / "t4_monitoring_selected.csv", index=False)
    selected[keep].to_latex(TABLE / "t4_monitoring_selected.tex", index=False, float_format="%.3f")

    lat = latency.groupby(["runtime", "provider", "model"], as_index=False).agg(
        mean_ms=("mean_ms", "mean"),
        p95_ms=("p95_ms", "mean"),
        p99_ms=("p99_ms", "mean"),
        p999_ms=("p999_ms", "mean"),
        deadline_miss_ratio=("deadline_miss_ratio", "mean"),
    )
    lat["runtime_label"] = lat["runtime"].map(RUNTIME_LABELS)
    lat["model_label"] = lat["model"].map(MODEL_LABELS)
    lat[["runtime_label", "model_label", "mean_ms", "p95_ms", "p99_ms", "p999_ms", "deadline_miss_ratio"]].to_csv(
        TABLE / "t4_latency_runtime_summary.csv", index=False
    )
    lat[["runtime_label", "model_label", "mean_ms", "p95_ms", "p99_ms", "p999_ms", "deadline_miss_ratio"]].to_latex(
        TABLE / "t4_latency_runtime_summary.tex", index=False, float_format="%.3f"
    )

    b = budget[
        (budget["base_method"].isin(["CIRCA-RT", "RFF-HSIC", "ConditionalRFF-HSIC", "ContextAwareConformal"]))
        & (budget["runtime"].isin(["onnx_cuda", "onnx_tensorrt"]))
        & (budget["budget"].round(3).isin([0.05, 0.10, 0.15]))
    ].copy()
    b["runtime_label"] = b["runtime"].map(RUNTIME_LABELS)
    b["model_label"] = b["model"].map(MODEL_LABELS)
    b[
        [
            "runtime_label",
            "model_label",
            "budget",
            "base_method",
            "attack_recall",
            "false_alarm_rate",
            "audit_rate",
            "p99_latency_ms",
            "deadline_miss_ratio",
        ]
    ].to_csv(TABLE / "t4_budget_selected.csv", index=False)

    d = deadline[
        (deadline["trace_type"].isin(["gpu_interference", "coupled_gpu_attack"]))
        & (deadline["method"].isin(["CIRCA-RT", "AlwaysAudit", "RFF-HSIC", "ContextAwareConformal"]))
        & (deadline["deadline_ms"].isin([33.333, 35.0, 40.0, 50.0]))
    ].copy()
    d["runtime_label"] = d["runtime"].map(RUNTIME_LABELS)
    d["model_label"] = d["model"].map(MODEL_LABELS)
    d[
        [
            "runtime_label",
            "model_label",
            "trace_type",
            "method",
            "deadline_ms",
            "miss_ratio_at_deadline",
            "p99_latency_ms",
            "p99_slack_ms",
            "p999_slack_ms",
        ]
    ].to_csv(TABLE / "t4_deadline_selected.csv", index=False)
    d[
        [
            "runtime_label",
            "model_label",
            "trace_type",
            "method",
            "deadline_ms",
            "miss_ratio_at_deadline",
            "p99_latency_ms",
            "p99_slack_ms",
            "p999_slack_ms",
        ]
    ].to_latex(TABLE / "t4_deadline_selected.tex", index=False, float_format="%.3f")


def plot_latency_runtime(latency: pd.DataFrame) -> None:
    agg = latency.groupby(["runtime", "model"], as_index=False).agg(mean_ms=("mean_ms", "mean"), p99_ms=("p99_ms", "mean"))
    models = ["mobilenet_v2", "resnet18"]
    runtimes = ["onnx_cuda", "onnx_tensorrt"]
    x = np.arange(len(models))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    for ax, metric, title in zip(axes, ["mean_ms", "p99_ms"], ["Mean Latency", "p99 Latency"]):
        for i, runtime in enumerate(runtimes):
            vals = [
                float(agg[(agg["runtime"] == runtime) & (agg["model"] == model)][metric].iloc[0])
                for model in models
            ]
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width,
                label=RUNTIME_LABELS[runtime],
                color=COLORS[runtime],
            )
        ax.axhline(33.333, color="#BC4749", linestyle="--", linewidth=1.2, label="33.333 ms" if metric == "p99_ms" else None)
        ax.set_title(title)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in models])
        ax.set_ylabel("latency (ms)")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    savefig("t4_onnx_cuda_vs_tensorrt_latency")


def plot_recall_audit(monitoring: pd.DataFrame) -> None:
    sub = monitoring[(monitoring["method"].isin(METHODS)) & (monitoring["runtime"].isin(["onnx_cuda", "onnx_tensorrt"]))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, runtime in zip(axes, ["onnx_cuda", "onnx_tensorrt"]):
        r = sub[sub["runtime"] == runtime]
        for method, group in r.groupby("method"):
            ax.scatter(
                group["audit_rate"],
                group["attack_recall"],
                s=95 if method == "CIRCA-RT" else 62,
                color=COLORS.get(method, "#555555"),
                edgecolor="black" if method == "CIRCA-RT" else "none",
                linewidth=1.0,
                label=method_label(method),
                alpha=0.90,
            )
            for _, row in group.iterrows():
                ax.annotate(MODEL_LABELS.get(row["model"], row["model"]), (row["audit_rate"], row["attack_recall"]), fontsize=7, alpha=0.75)
        ax.set_title(RUNTIME_LABELS[runtime])
        ax.set_xlabel("audit rate")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("attack recall")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(FIG / "t4_recall_vs_audit_pareto.png", dpi=240)
    plt.savefig(FIG / "t4_recall_vs_audit_pareto.pdf")
    plt.close()
    shutil.copy2(FIG / "t4_recall_vs_audit_pareto.png", LOCAL_FIG / "t4_recall_vs_audit_pareto.png")
    shutil.copy2(FIG / "t4_recall_vs_audit_pareto.pdf", LOCAL_FIG / "t4_recall_vs_audit_pareto.pdf")


def plot_deadline_sensitivity(deadline: pd.DataFrame) -> None:
    sub = deadline[
        (deadline["runtime"] == "onnx_tensorrt")
        & (deadline["trace_type"] == "gpu_interference")
        & (deadline["method"].isin(["CIRCA-RT", "RFF-HSIC", "ContextAwareConformal", "AlwaysAudit"]))
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, ["mobilenet_v2", "resnet18"]):
        r = sub[sub["model"] == model]
        for method, group in r.groupby("method"):
            g = group.sort_values("deadline_ms")
            ax.plot(
                g["deadline_ms"],
                g["miss_ratio_at_deadline"],
                marker="o",
                linewidth=2.5 if method == "CIRCA-RT" else 1.7,
                color=COLORS.get(method, "#555555"),
                label=method_label(method),
            )
        ax.set_title(f"{MODEL_LABELS[model]} under GPU interference")
        ax.set_xlabel("deadline (ms)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("deadline miss ratio")
    axes[1].legend(fontsize=8)
    savefig("t4_tensorrt_deadline_sensitivity")


def plot_deadline_by_scenario(deadline: pd.DataFrame) -> None:
    sub = deadline[
        (deadline["runtime"] == "onnx_tensorrt")
        & (deadline["method"] == "CIRCA-RT")
        & (deadline["trace_type"].isin(["nominal", "coupled_gpu_attack", "gpu_interference"]))
    ].copy()
    scenarios = ["nominal", "coupled_gpu_attack", "gpu_interference"]
    labels = {
        "nominal": "nominal",
        "coupled_gpu_attack": "coupled attack",
        "gpu_interference": "GPU interference",
    }
    scenario_colors = {
        "nominal": "#606C38",
        "coupled_gpu_attack": "#0B6E4F",
        "gpu_interference": "#BC4749",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, ["mobilenet_v2", "resnet18"]):
        r = sub[sub["model"] == model]
        for scenario in scenarios:
            g = r[r["trace_type"] == scenario].sort_values("deadline_ms")
            ax.plot(
                g["deadline_ms"],
                g["miss_ratio_at_deadline"],
                marker="o",
                linewidth=2.2,
                color=scenario_colors[scenario],
                label=labels[scenario],
            )
        ax.set_title(f"{MODEL_LABELS[model]} / TensorRT CIRCA-RT")
        ax.set_xlabel("deadline (ms)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("deadline miss ratio")
    axes[1].legend(fontsize=8)
    savefig("t4_tensorrt_circa_deadline_by_scenario")


def plot_slack_heatmap(deadline: pd.DataFrame) -> None:
    sub = deadline[
        (deadline["runtime"] == "onnx_tensorrt")
        & (deadline["trace_type"] == "gpu_interference")
        & (deadline["method"].isin(["CIRCA-RT", "RFF-HSIC", "AlwaysAudit"]))
        & (deadline["deadline_ms"].isin([33.333, 35.0, 40.0, 50.0]))
    ].copy()
    rows = []
    labels = []
    for model in ["mobilenet_v2", "resnet18"]:
        for method in ["CIRCA-RT", "RFF-HSIC", "AlwaysAudit"]:
            g = sub[(sub["model"] == model) & (sub["method"] == method)].sort_values("deadline_ms")
            if g.empty:
                continue
            rows.append(g["p99_slack_ms"].to_numpy(dtype=float))
            labels.append(f"{MODEL_LABELS[model]} / {method_label(method)}")
    data = np.vstack(rows)
    plt.figure(figsize=(8.5, 4.6))
    im = plt.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-8, vmax=12)
    plt.colorbar(im, label="p99 slack (ms)")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(4), ["33.333", "35", "40", "50"])
    plt.xlabel("deadline (ms)")
    plt.title("TensorRT p99 deadline slack under GPU interference")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)
    savefig("t4_tensorrt_p99_slack_heatmap")


def plot_budget_recall(budget: pd.DataFrame) -> None:
    sub = budget[
        (budget["runtime"] == "onnx_tensorrt")
        & (budget["model"] == "mobilenet_v2")
        & (budget["base_method"].isin(["CIRCA-RT", "RFF-HSIC", "ConditionalRFF-HSIC", "ContextAwareConformal"]))
    ].copy()
    plt.figure(figsize=(8.5, 4.8))
    for method, group in sub.groupby("base_method"):
        g = group.sort_values("budget")
        plt.plot(
            g["budget"] * 100.0,
            g["attack_recall"],
            marker="o",
            linewidth=2.6 if method == "CIRCA-RT" else 1.7,
            color=COLORS.get(method, "#555555"),
            label=method_label(method),
        )
    plt.xlabel("audit budget (%)")
    plt.ylabel("attack recall")
    plt.title("TensorRT MobileNetV2 recall under matched audit budgets")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    savefig("t4_tensorrt_budget_recall")


def write_report(monitoring: pd.DataFrame, latency: pd.DataFrame, deadline: pd.DataFrame) -> None:
    trt = monitoring[(monitoring["runtime"] == "onnx_tensorrt") & (monitoring["method"] == "CIRCA-RT")].copy()
    lat = latency.groupby(["runtime", "model"], as_index=False).agg(mean_ms=("mean_ms", "mean"), p99_ms=("p99_ms", "mean"))
    stress = deadline[
        (deadline["runtime"] == "onnx_tensorrt")
        & (deadline["method"] == "CIRCA-RT")
        & (deadline["trace_type"] == "gpu_interference")
        & (deadline["deadline_ms"].isin([33.333, 35.0, 40.0, 50.0]))
    ].copy()
    scenario_at_33333 = deadline[
        (deadline["runtime"] == "onnx_tensorrt")
        & (deadline["method"] == "CIRCA-RT")
        & (deadline["deadline_ms"] == 33.333)
    ].copy()
    lines = [
        "# T4 Deadline Sensitivity And Figure Integration",
        "",
        "Date: 2026-05-24",
        "",
        "## Outputs",
        "",
        f"- Tables: `{TABLE}`",
        f"- Figures: `{FIG}`",
        "- Mirrored core figures: `results_local_complete/figures/`",
        "",
        "## Key TensorRT CIRCA-RT Rows",
        "",
    ]
    for _, row in trt.sort_values("model").iterrows():
        lines.append(
            f"- `{MODEL_LABELS.get(row.model, row.model)}`: recall {row.attack_recall:.3f}, "
            f"false alarm {row.false_alarm_rate:.3f}, audit {row.audit_rate:.3f}, "
            f"p99 {row.p99_latency_ms:.3f} ms, p999 {row.p999_latency_ms:.3f} ms, "
            f"miss {row.deadline_miss_ratio:.3f}."
        )
    lines.extend(["", "## Runtime Latency Summary", ""])
    for _, row in lat.sort_values(["model", "runtime"]).iterrows():
        lines.append(
            f"- `{RUNTIME_LABELS[row.runtime]}` / `{MODEL_LABELS[row.model]}`: "
            f"mean {row.mean_ms:.3f} ms, p99 {row.p99_ms:.3f} ms."
        )
    lines.extend(["", "## Deadline Sensitivity Summary", ""])
    lines.append("At the 33.333 ms 30 FPS deadline, TensorRT CIRCA-RT is stable on nominal and coupled-attack traces, but GPU-interference traces have heavy tails:")
    lines.append("")
    for _, row in scenario_at_33333.sort_values(["model", "trace_type"]).iterrows():
        lines.append(
            f"- `{MODEL_LABELS[row.model]}` / `{row.trace_type}`: miss {row.miss_ratio_at_deadline:.3f}, "
            f"p99 {row.p99_latency_ms:.3f} ms, p99 slack {row.p99_slack_ms:.3f} ms."
        )
    lines.extend(["", "GPU-interference stress sweep:", ""])
    for _, row in stress.sort_values(["model", "deadline_ms"]).iterrows():
        lines.append(
            f"- `{MODEL_LABELS[row.model]}` TensorRT CIRCA-RT, deadline {row.deadline_ms:g} ms: "
            f"miss {row.miss_ratio_at_deadline:.3f}, p99 slack {row.p99_slack_ms:.3f} ms."
        )
    lines.extend(
        [
            "",
            "## Paper Wording",
            "",
            "Use this as Tesla T4 ONNXRuntime TensorRT EP low-power accelerator evidence. Do not call it Jetson or embedded edge-device evidence.",
            "",
            "The sensitivity result shows that nominal and coupled-attack traces meet the 33.333 ms p99 target, while GPU-interference traces violate p99 even at relaxed 35-50 ms deadlines. The correct paper claim is therefore not hard real-time robustness under arbitrary GPU contention; it is that CIRCA-RT keeps audit demand bounded and avoids the additional always-audit cost while exposing the stress-tail limitation.",
        ]
    )
    (PAPER / "T4_DEADLINE_SENSITIVITY_AND_FIGURES_2026_05_24.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not FULL.exists():
        raise SystemExit(f"missing result directory: {FULL}")
    monitoring, latency, budget = read_main_tables()
    deadline = build_deadline_sensitivity()
    export_selected_tables(monitoring, latency, budget, deadline)
    plot_latency_runtime(latency)
    plot_recall_audit(monitoring)
    plot_deadline_sensitivity(deadline)
    plot_deadline_by_scenario(deadline)
    plot_slack_heatmap(deadline)
    plot_budget_recall(budget)
    write_report(monitoring, latency, deadline)
    print(f"wrote tables to {TABLE}")
    print(f"wrote figures to {FIG}")
    print(f"wrote report to {PAPER / 'T4_DEADLINE_SENSITIVITY_AND_FIGURES_2026_05_24.md'}")


if __name__ == "__main__":
    main()
