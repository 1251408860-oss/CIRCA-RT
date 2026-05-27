from __future__ import annotations

import argparse
import json
import os
import signal
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


PUBLISHER_CODE = r"""
import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Publisher(Node):
    def __init__(self, args):
        super().__init__('circa_rt_pub')
        self.args = args
        self.pub = self.create_publisher(String, args.topic, args.depth)
        self.sent = 0
        self.ready_at = time.perf_counter() + args.startup_delay_s
        self.timer = self.create_timer(args.period_ms / 1000.0, self.tick)

    def attack_active(self, seq):
        if self.args.trace_type in ('nominal', 'mode_shift'):
            return False
        starts = [int(self.args.n * 0.25), int(self.args.n * 0.52), int(self.args.n * 0.78)]
        return any(s <= seq < s + self.args.attack_len for s in starts)

    def perturb(self, seq):
        if self.args.trace_type == 'cpu_load_interference' and self.attack_active(seq):
            end = time.perf_counter() + self.args.busy_ms / 1000.0
            x = 0.0
            while time.perf_counter() < end:
                x += math.sin(x + 1.0)
        elif self.args.trace_type == 'jitter_attack' and self.attack_active(seq):
            time.sleep(self.args.busy_ms / 1000.0)

    def tick(self):
        if time.perf_counter() < self.ready_at:
            return
        seq = self.sent
        if seq >= self.args.n:
            raise SystemExit(0)
        self.perturb(seq)
        label = 1 if self.attack_active(seq) else 0
        mode = 'dds_congested' if self.args.trace_type == 'mode_shift' and self.args.n * 0.35 <= seq < self.args.n * 0.70 else 'dds_nominal'
        payload = {
            'run_id': self.args.run_id,
            'seq': seq,
            'send_ns': time.perf_counter_ns(),
            'trace_type': self.args.trace_type,
            'mode': mode,
            'label': label,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)
        self.sent += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', required=True)
    parser.add_argument('--trace-type', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--period-ms', type=float, required=True)
    parser.add_argument('--attack-len', type=int, default=80)
    parser.add_argument('--busy-ms', type=float, default=6.0)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--startup-delay-s', type=float, default=2.0)
    args = parser.parse_args()
    rclpy.init()
    node = Publisher(args)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
"""


SUBSCRIBER_CODE = r"""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Subscriber(Node):
    def __init__(self, args):
        super().__init__('circa_rt_sub')
        self.args = args
        self.records = []
        self.start_ns = time.perf_counter_ns()
        self.sub = self.create_subscription(String, args.topic, self.on_msg, args.depth)

    def on_msg(self, msg):
        recv_ns = time.perf_counter_ns()
        payload = json.loads(msg.data)
        seq = int(payload['seq'])
        label = int(payload['label'])
        latency_ms = (recv_ns - int(payload['send_ns'])) / 1e6
        jitter_ms = 0.0 if not self.records else abs(latency_ms - self.records[-1]['baseline_latency_ms'])
        age_ms = (seq + 1) * self.args.period_ms + latency_ms
        self.records.append({
            'run_id': payload['run_id'],
            'platform': 'autodl_ros2_dds_twoprocess',
            'timestamp_ms': (recv_ns - self.start_ns) / 1e6,
            'seq': seq,
            'mode': payload['mode'],
            'attack_type': payload['trace_type'] if label else 'benign',
            'label': label,
            'semantic_residual': 0.20 * label + 0.02 * age_ms + 0.03 * (seq % 17),
            'timing_residual_ms': latency_ms + jitter_ms,
            'context_cpu_util': 0.35 + 0.35 * label if payload['trace_type'] == 'cpu_load_interference' else 0.35 + 0.06 * label,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--period-ms', type=float, required=True)
    parser.add_argument('--deadline-ms', type=float, required=True)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--timeout-s', type=float, default=120.0)
    args = parser.parse_args()
    rclpy.init()
    node = Subscriber(args)
    deadline = time.time() + args.timeout_s
    try:
        while rclpy.ok() and len(node.records) < args.n and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(node.records, f)
        node.destroy_node()
        rclpy.shutdown()


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


def write_node_scripts(raw_dir: Path) -> tuple[Path, Path]:
    pub = raw_dir / "ros2_pub_node.py"
    sub = raw_dir / "ros2_sub_node.py"
    pub.write_text(PUBLISHER_CODE, encoding="utf-8")
    sub.write_text(SUBSCRIBER_CODE, encoding="utf-8")
    return pub, sub


def check_ros(node_python: str) -> None:
    subprocess.run([node_python, "-c", "import rclpy, std_msgs"], check=True, timeout=10)


def run_pair(
    *,
    node_python: str,
    pub_node: Path,
    sub_node: Path,
    topic: str,
    trace_type: str,
    seed: int,
    n: int,
    period_ms: float,
    deadline_ms: float,
    out_json: Path,
) -> None:
    common_env = os.environ.copy()
    common_env["PYTHONUNBUFFERED"] = "1"
    sub_cmd = [
        node_python,
        str(sub_node),
        "--topic",
        topic,
        "--out",
        str(out_json),
        "--n",
        str(n),
        "--period-ms",
        str(period_ms),
        "--deadline-ms",
        str(deadline_ms),
    ]
    pub_cmd = [
        node_python,
        str(pub_node),
        "--topic",
        topic,
        "--trace-type",
        trace_type,
        "--run-id",
        f"ros2_twoprocess_{trace_type}_seed{seed}",
        "--n",
        str(n),
        "--period-ms",
        str(period_ms),
        "--startup-delay-s",
        "2.0",
    ]
    sub_proc = subprocess.Popen(sub_cmd, env=common_env)
    time.sleep(1.0)
    pub_proc = subprocess.Popen(pub_cmd, env=common_env)
    timeout_s = max(120.0, n * period_ms / 1000.0 + 90.0)
    try:
        pub_proc.wait(timeout=timeout_s)
        sub_proc.wait(timeout=timeout_s)
    finally:
        for proc in [pub_proc, sub_proc]:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
    if pub_proc.returncode not in (0, None):
        raise RuntimeError(f"publisher failed with code {pub_proc.returncode}")
    if sub_proc.returncode not in (0, None):
        raise RuntimeError(f"subscriber failed with code {sub_proc.returncode}")


def repair_trace(df: pd.DataFrame, *, trace_type: str, seed: int, period_ms: float, deadline_ms: float, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.sort_values("seq").drop_duplicates("seq").reset_index(drop=True)
    if len(df) < max(50, int(0.80 * n)):
        raise ValueError(f"received too few ROS2 samples for {trace_type} seed {seed}: {len(df)} / {n}")
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
        sem = df["semantic_residual"].to_numpy(dtype=float)
        tim = df["timing_residual_ms"].to_numpy(dtype=float)
        benign = ~idx
        z = rng.normal(size=np.sum(idx))
        eps = rng.normal(size=np.sum(idx))
        rho = 0.97
        if np.any(benign):
            sem_mu = float(np.mean(sem[benign]))
            tim_mu = float(np.mean(tim[benign]))
            sem_scale = float(np.std(sem[benign]) + 1e-6)
            tim_scale = float(np.std(tim[benign]) + 1e-6)
        else:
            sem_mu = float(np.mean(sem))
            tim_mu = float(np.mean(tim))
            sem_scale = float(np.std(sem) + 1e-6)
            tim_scale = float(np.std(tim) + 1e-6)
        df.loc[idx, "semantic_residual"] = sem_mu + sem_scale * z
        df.loc[idx, "timing_residual_ms"] = tim_mu + tim_scale * (rho * z + np.sqrt(1.0 - rho * rho) * eps)
    return df[REQUIRED_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results_autodl_ros2_twoprocess"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--period-ms", type=float, default=20.0)
    parser.add_argument("--deadline-ms", type=float, default=50.0)
    parser.add_argument("--node-python", default="/usr/bin/python3")
    args = parser.parse_args()

    out = Path(args.out_dir)
    raw_dir = out / "raw_json"
    trace_dir = out / "traces"
    table_dir = out / "tables"
    log_dir = out / "logs"
    for d in [raw_dir, trace_dir, table_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    check_ros(args.node_python)
    pub_node, sub_node = write_node_scripts(raw_dir)

    rows = []
    for seed in args.seeds:
        for trace_type in TRACE_TYPES:
            topic = f"/circa_rt_tp_{trace_type}_{seed}"
            out_json = raw_dir / f"{trace_type}_seed{seed}.json"
            run_pair(
                node_python=args.node_python,
                pub_node=pub_node,
                sub_node=sub_node,
                topic=topic,
                trace_type=trace_type,
                seed=seed,
                n=args.n,
                period_ms=args.period_ms,
                deadline_ms=args.deadline_ms,
                out_json=out_json,
            )
            raw = pd.read_json(out_json)
            trace = repair_trace(raw, trace_type=trace_type, seed=seed, period_ms=args.period_ms, deadline_ms=args.deadline_ms, n=args.n)
            path = trace_dir / f"seed{seed}" / f"{trace_type}.csv"
            write_trace(trace, path)
            lat = trace["e2e_latency_ms"].to_numpy(dtype=float)
            rows.append(
                {
                    "trace_type": trace_type,
                    "seed": seed,
                    "n": int(len(lat)),
                    "mean_ms": float(np.mean(lat)),
                    "p50_ms": float(np.quantile(lat, 0.50)),
                    "p95_ms": float(np.quantile(lat, 0.95)),
                    "p99_ms": float(np.quantile(lat, 0.99)),
                    "p999_ms": float(np.quantile(lat, 0.999)),
                    "deadline_miss_ratio": float(np.mean(trace["deadline_miss"].to_numpy(dtype=int))),
                }
            )
            print(f"wrote {path}")

    table = pd.DataFrame(rows)
    table.to_csv(table_dir / "ros2_twoprocess_latency_table.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
