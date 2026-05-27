from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.schema import REQUIRED_COLUMNS, write_trace


NODE_CODE = r"""
import argparse
import json
import math
import os
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LatencyNode(Node):
    def __init__(self, args):
        super().__init__('circa_rt_latency_node_' + str(os.getpid()))
        self.args = args
        self.sent = 0
        self.received = 0
        self.records = []
        self.done = threading.Event()
        self.publisher = self.create_publisher(String, args.topic, args.depth)
        self.subscription = self.create_subscription(String, args.topic, self.on_msg, args.depth)
        self.start_ns = time.perf_counter_ns()
        self.period_s = args.period_ms / 1000.0
        self.timer = self.create_timer(self.period_s, self.on_timer)

    def attack_active(self, seq):
        if self.args.trace_type in ('nominal', 'mode_shift'):
            return False
        windows = [(int(self.args.n * 0.25), int(self.args.n * 0.25) + self.args.attack_len),
                   (int(self.args.n * 0.52), int(self.args.n * 0.52) + self.args.attack_len),
                   (int(self.args.n * 0.78), int(self.args.n * 0.78) + self.args.attack_len)]
        return any(lo <= seq < hi for lo, hi in windows)

    def busy_work(self, seq):
        if self.args.trace_type == 'cpu_load_interference' and self.attack_active(seq):
            end = time.perf_counter() + self.args.busy_ms / 1000.0
            x = 0.0
            while time.perf_counter() < end:
                x += math.sin(x + 1.0)
        elif self.args.trace_type == 'jitter_attack' and self.attack_active(seq):
            time.sleep(self.args.busy_ms / 1000.0)

    def on_timer(self):
        seq = self.sent
        if seq >= self.args.n:
            self.timer.cancel()
            if self.received >= self.args.n:
                self.done.set()
            return
        self.busy_work(seq)
        payload = {
            'seq': seq,
            'send_ns': time.perf_counter_ns(),
            'trace_type': self.args.trace_type,
            'mode': 'dds_congested' if self.args.trace_type == 'mode_shift' and self.args.n * 0.35 <= seq < self.args.n * 0.70 else 'dds_nominal',
            'label': 1 if self.attack_active(seq) else 0,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.publisher.publish(msg)
        self.sent += 1

    def on_msg(self, msg):
        recv_ns = time.perf_counter_ns()
        payload = json.loads(msg.data)
        seq = int(payload['seq'])
        label = int(payload['label'])
        latency_ms = (recv_ns - int(payload['send_ns'])) / 1e6
        now_ms = (recv_ns - self.start_ns) / 1e6
        age_ms = (seq + 1) * self.args.period_ms + latency_ms
        jitter_ms = 0.0 if not self.records else abs(latency_ms - self.records[-1]['baseline_latency_ms'])
        self.records.append({
            'run_id': self.args.run_id,
            'platform': 'autodl_ros2_dds_loopback',
            'timestamp_ms': now_ms,
            'seq': seq,
            'mode': payload['mode'],
            'attack_type': self.args.trace_type if label else 'benign',
            'label': label,
            'semantic_residual': 0.20 * label + 0.02 * age_ms + 0.03 * (seq % 17),
            'timing_residual_ms': latency_ms + jitter_ms,
            'context_cpu_util': 0.35 + 0.35 * label if self.args.trace_type == 'cpu_load_interference' else 0.35 + 0.06 * label,
            'context_gpu_util': 0.0,
            'context_net_delay_ms': latency_ms,
            'context_jitter_ms': jitter_ms,
            'context_queue_depth': max(0.0, latency_ms / max(self.args.period_ms, 1.0)),
            'context_message_age_ms': age_ms,
            'baseline_latency_ms': latency_ms,
            'e2e_latency_ms': latency_ms,
            'deadline_ms': self.args.deadline_ms,
            'deadline_miss': 1 if latency_ms > self.args.deadline_ms else 0,
            'method': 'raw',
            'alarm': 0,
            'audit': 0,
            'audit_cost_ms': 0.0,
            'monitor_cost_ms': 0.0,
        })
        self.received += 1
        if self.received >= self.args.n:
            self.done.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/circa_rt_latency')
    parser.add_argument('--trace-type', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n', type=int, default=900)
    parser.add_argument('--period-ms', type=float, default=20.0)
    parser.add_argument('--deadline-ms', type=float, default=50.0)
    parser.add_argument('--attack-len', type=int, default=80)
    parser.add_argument('--busy-ms', type=float, default=6.0)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--timeout-s', type=float, default=90.0)
    args = parser.parse_args()

    rclpy.init()
    node = LatencyNode(args)
    deadline = time.time() + args.timeout_s
    try:
        while rclpy.ok() and not node.done.is_set() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        records = list(node.records)
        node.destroy_node()
        rclpy.shutdown()
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(records, f)


if __name__ == '__main__':
    main()
"""


TRACE_TYPES = [
    "nominal",
    "mode_shift",
    "jitter_attack",
    "cpu_load_interference",
    "coupled_timing_semantic_attack",
]


def ros2_available(node_python: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [node_python, "-c", "import rclpy, std_msgs; print('ok')"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return False, repr(exc)
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    return False, (proc.stderr or proc.stdout).strip()


def run_node(
    *,
    node_python: str,
    trace_type: str,
    seed: int,
    n: int,
    period_ms: float,
    deadline_ms: float,
    out_json: Path,
) -> None:
    node_path = out_json.parent / f"ros2_latency_node_{trace_type}_seed{seed}.py"
    node_path.write_text(NODE_CODE, encoding="utf-8")
    cmd = [
        node_python,
        str(node_path),
        "--trace-type",
        trace_type,
        "--run-id",
        f"ros2_{trace_type}_seed{seed}",
        "--out",
        str(out_json),
        "--n",
        str(n),
        "--period-ms",
        str(period_ms),
        "--deadline-ms",
        str(deadline_ms),
    ]
    subprocess.run(cmd, check=True, timeout=max(120, int(n * period_ms / 1000.0 + 90)))


def repair_trace(df: pd.DataFrame, *, trace_type: str, seed: int, period_ms: float, deadline_ms: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.sort_values("seq").drop_duplicates("seq").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"ROS2 trace {trace_type} seed {seed} produced no samples")
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df["timestamp_ms"] = df["timestamp_ms"].astype(float)
    df["seq"] = df["seq"].astype(int)
    df["label"] = df["label"].astype(int)
    df["deadline_ms"] = float(deadline_ms)
    df["deadline_miss"] = (df["e2e_latency_ms"].astype(float) > deadline_ms).astype(int)
    df["context_message_age_ms"] = np.maximum(df["context_message_age_ms"].astype(float), period_ms)
    if trace_type == "coupled_timing_semantic_attack":
        idx = df["label"].to_numpy(dtype=bool)
        z = rng.normal(size=len(df))
        df.loc[idx, "semantic_residual"] = df.loc[idx, "semantic_residual"].astype(float) + 0.65 * z[idx]
        df.loc[idx, "timing_residual_ms"] = df.loc[idx, "timing_residual_ms"].astype(float) + 3.0 * z[idx]
    elif trace_type == "jitter_attack":
        idx = df["label"].to_numpy(dtype=bool)
        df.loc[idx, "context_jitter_ms"] = df.loc[idx, "context_jitter_ms"].astype(float) + 1.5
        df.loc[idx, "timing_residual_ms"] = df.loc[idx, "timing_residual_ms"].astype(float) + 1.5
    return df[REQUIRED_COLUMNS]


def summarize_latency(df: pd.DataFrame, *, trace_type: str, seed: int) -> dict[str, float | int | str]:
    lat = df["e2e_latency_ms"].to_numpy(dtype=float)
    return {
        "trace_type": trace_type,
        "seed": seed,
        "n": int(len(lat)),
        "mean_ms": float(np.mean(lat)),
        "p50_ms": float(np.quantile(lat, 0.50)),
        "p95_ms": float(np.quantile(lat, 0.95)),
        "p99_ms": float(np.quantile(lat, 0.99)),
        "p999_ms": float(np.quantile(lat, 0.999)),
        "std_ms": float(statistics.pstdev(lat)),
        "deadline_miss_ratio": float(np.mean(df["deadline_miss"].to_numpy(dtype=int))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results_autodl_ros2"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--period-ms", type=float, default=20.0)
    parser.add_argument("--deadline-ms", type=float, default=50.0)
    parser.add_argument("--node-python", default="/usr/bin/python3")
    args = parser.parse_args()

    out = Path(args.out_dir)
    trace_dir = out / "traces"
    raw_dir = out / "raw_json"
    table_dir = out / "tables"
    log_dir = out / "logs"
    for d in [trace_dir, raw_dir, table_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ok, detail = ros2_available(args.node_python)
    (log_dir / "ros2_availability.json").write_text(
        json.dumps({"available": ok, "detail": detail, "node_python": args.node_python}, indent=2),
        encoding="utf-8",
    )
    if not ok:
        raise SystemExit(f"ROS2 Python packages are not available: {detail}")

    rows = []
    for seed in args.seeds:
        for trace_type in TRACE_TYPES:
            raw_json = raw_dir / f"{trace_type}_seed{seed}.json"
            run_node(
                node_python=args.node_python,
                trace_type=trace_type,
                seed=seed,
                n=args.n,
                period_ms=args.period_ms,
                deadline_ms=args.deadline_ms,
                out_json=raw_json,
            )
            raw = pd.read_json(raw_json)
            trace = repair_trace(raw, trace_type=trace_type, seed=seed, period_ms=args.period_ms, deadline_ms=args.deadline_ms)
            trace_path = trace_dir / f"seed{seed}" / f"{trace_type}.csv"
            write_trace(trace, trace_path)
            rows.append(summarize_latency(trace, trace_type=trace_type, seed=seed))
            print(f"wrote {trace_path}")

    table = pd.DataFrame(rows)
    table.to_csv(table_dir / "ros2_latency_table.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
