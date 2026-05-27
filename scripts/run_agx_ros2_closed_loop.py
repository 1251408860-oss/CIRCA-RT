from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.core import CIRCARuntimeConfig, fit_circa_rt
from circa_rt.metrics import summarize_detection
from circa_rt.perception_semantics import list_frame_paths
from circa_rt.schema import REQUIRED_COLUMNS, write_trace


PUBLISHER_CODE = r"""
import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageEnhance, ImageFilter, ImageOps
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def attack_active(scenario, seq, n, attack_len):
    if scenario in ("nominal", "mode_shift"):
        return False
    starts = [int(n * 0.25), int(n * 0.52), int(n * 0.78)]
    return any(s <= seq < s + attack_len for s in starts)


def frame_index(scenario, seq, attacked, frame_count, seed, clip_len):
    base = ((seq // max(1, clip_len)) + seed * 997) % frame_count
    if scenario == "stale_replay_attack" and attacked:
        lag = max(clip_len * 5, 30)
        base = ((max(0, seq - lag) // max(1, clip_len)) + seed * 997) % frame_count
    return int(base)


def perturb_image(image, scenario, attacked, seq, seed):
    if not attacked:
        return image
    if scenario not in {"semantic_corruption", "coupled_semantic_timing_attack"}:
        return image
    rng = np.random.default_rng(seed * 1000003 + seq)
    img = image
    if seq % 3 == 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.6))
    elif seq % 3 == 1:
        img = ImageOps.posterize(img, bits=3)
    else:
        arr = np.asarray(img).astype(np.int16)
        noise = rng.normal(0.0, 24.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return ImageEnhance.Contrast(img).enhance(1.4)


class FramePublisher(Node):
    def __init__(self, args):
        super().__init__("circa_rt_agx_frame_pub")
        self.args = args
        self.frames = json.loads(Path(args.frame_manifest).read_text(encoding="utf-8"))
        if not self.frames:
            raise RuntimeError("frame manifest is empty")
        self.pub = self.create_publisher(String, args.topic, args.depth)
        self.sent = 0
        self.ready_at = time.perf_counter() + args.startup_delay_s
        self.timer = self.create_timer(args.period_ms / 1000.0, self.tick)

    def image_payload(self, path, scenario, attacked, seq):
        if self.args.payload_mode == "path":
            return {}
        img = Image.open(path).convert("RGB")
        img = perturb_image(img, scenario, attacked, seq, self.args.seed)
        if self.args.publish_resize > 0:
            img.thumbnail((self.args.publish_resize, self.args.publish_resize))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.args.jpeg_quality)
        return {
            "image_encoding": "jpeg_b64",
            "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "image_perturbed_at_publisher": bool(attacked and scenario in {"semantic_corruption", "coupled_semantic_timing_attack"}),
        }

    def tick(self):
        if time.perf_counter() < self.ready_at:
            return
        seq = self.sent
        if seq >= self.args.n:
            raise SystemExit(0)
        attacked = attack_active(self.args.scenario, seq, self.args.n, self.args.attack_len)
        idx = frame_index(self.args.scenario, seq, attacked, len(self.frames), self.args.seed, self.args.clip_len)
        if self.args.scenario == "mode_shift" and self.args.n * 0.35 <= seq < self.args.n * 0.70:
            mode = "ros2_mode_shift"
        elif attacked:
            mode = self.args.scenario
        else:
            mode = "ros2_perception_nominal"
        payload = {
            "run_id": self.args.run_id,
            "seq": seq,
            "seed": self.args.seed,
            "scenario": self.args.scenario,
            "mode": mode,
            "label": int(attacked),
            "frame_index": idx,
            "frame_path": self.frames[idx],
            "send_ns": time.perf_counter_ns(),
            "payload_mode": self.args.payload_mode,
        }
        payload.update(self.image_payload(self.frames[idx], self.args.scenario, attacked, seq))
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)
        self.sent += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--period-ms", type=float, required=True)
    parser.add_argument("--attack-len", type=int, default=80)
    parser.add_argument("--clip-len", type=int, default=12)
    parser.add_argument("--payload-mode", choices=["jpeg_b64", "path"], default="jpeg_b64")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--publish-resize", type=int, default=320)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--startup-delay-s", type=float, default=2.0)
    args = parser.parse_args()
    rclpy.init()
    node = FramePublisher(args)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
"""


MONITOR_CODE = r"""
import argparse
import base64
import io
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rclpy.node import Node
from std_msgs.msg import String


CONTEXT_COLUMNS = [
    "context_cpu_util",
    "context_gpu_util",
    "context_net_delay_ms",
    "context_jitter_ms",
    "context_queue_depth",
    "context_message_age_ms",
]


def model_weights(model_name, request):
    if request == "none":
        return None
    mapping = {
        "mobilenet_v2": models.MobileNet_V2_Weights,
        "resnet18": models.ResNet18_Weights,
        "resnet50": models.ResNet50_Weights,
        "squeezenet1_1": models.SqueezeNet1_1_Weights,
    }
    return mapping[model_name].DEFAULT


def build_model(model_name, weights_request, device):
    weights = model_weights(model_name, weights_request)
    builder = getattr(models, model_name)
    model = builder(weights=weights).eval().to(device)
    preprocess = weights.transforms() if weights is not None else T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess


def resolve_device(request):
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested --device cuda but torch.cuda.is_available() is false")
    return torch.device(request)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_image(path):
    return Image.open(path).convert("RGB")


def payload_image(payload):
    if payload.get("image_encoding") == "jpeg_b64" and payload.get("image_b64"):
        data = base64.b64decode(payload["image_b64"].encode("ascii"))
        return Image.open(io.BytesIO(data)).convert("RGB")
    return load_image(payload["frame_path"])


def perturb_image(image, scenario, attacked, seq, seed):
    if not attacked:
        return image
    if scenario not in {"semantic_corruption", "coupled_semantic_timing_attack"}:
        return image
    rng = np.random.default_rng(seed * 1000003 + seq)
    img = image
    if seq % 3 == 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.6))
    elif seq % 3 == 1:
        img = ImageOps.posterize(img, bits=3)
    else:
        arr = np.asarray(img).astype(np.int16)
        noise = rng.normal(0.0, 24.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return ImageEnhance.Contrast(img).enhance(1.4)


def logits_features(logits, prev_probs, prev_top1):
    raw = logits.detach().float().cpu().numpy()[0]
    shifted = raw - float(np.max(raw))
    exp = np.exp(shifted)
    probs = exp / max(float(np.sum(exp)), 1e-12)
    order = np.argsort(probs)[::-1]
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else top1
    top5 = order[: min(5, len(order))]
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    drift = 0.0 if prev_probs is None else float(np.sum(np.abs(probs - prev_probs)))
    changed = 0.0 if prev_top1 is None else float(top1 != prev_top1)
    features = {
        "model_entropy": entropy,
        "model_top1_prob": float(probs[top1]),
        "model_top1_margin": float(probs[top1] - probs[top2]),
        "model_top5_mass": float(np.sum(probs[top5])),
        "model_logit_l2": float(np.linalg.norm(raw)),
        "model_prob_drift_l1": drift,
        "model_top1_changed": changed,
        "model_top1": top1,
    }
    return features, probs, top1


def semantic_score(features):
    return float(
        0.25 * features["model_entropy"]
        + 5.0 * (1.0 - features["model_top1_prob"])
        + 0.75 * features["model_prob_drift_l1"]
        + 0.5 * features["model_top1_changed"]
    )


class OnlineCIRCA:
    def __init__(self, model, method, audit_cost_ms, slack_margin_ms):
        self.model = model
        self.method = method
        self.audit_cost_ms = float(audit_cost_ms)
        self.slack_margin_ms = float(slack_margin_ms)
        self.bucket = float(model.cfg.bucket_capacity)
        self.xs = []
        self.ys = []
        d = int(model.cfg.feature_dim)
        self.cross = np.zeros((d, d), dtype=np.float64)
        self.sum_x = np.zeros(d, dtype=np.float64)
        self.sum_y = np.zeros(d, dtype=np.float64)

    def step(self, row):
        context = np.asarray([[float(row[c]) for c in CONTEXT_COLUMNS]], dtype=np.float64)
        mode = np.asarray([str(row["mode"])])
        semantic = np.asarray([float(row["semantic_residual"])], dtype=np.float64)
        timing = np.asarray([float(row["timing_residual_ms"])], dtype=np.float64)
        u, v = self.model.residualizer.transform(semantic, timing, context, mode)
        phi_u = self.model.rff_u.transform(u)[0]
        phi_v = self.model.rff_v.transform(v)[0]
        self.xs.append(phi_u)
        self.ys.append(phi_v)
        self.cross += np.outer(phi_u, phi_v)
        self.sum_x += phi_u
        self.sum_y += phi_v
        w = max(int(self.model.cfg.window_size), 1)
        if len(self.xs) > w:
            x_old = self.xs.pop(0)
            y_old = self.ys.pop(0)
            self.cross -= np.outer(x_old, y_old)
            self.sum_x -= x_old
            self.sum_y -= y_old
        m = float(len(self.xs))
        centered = self.cross - np.outer(self.sum_x, self.sum_y) / max(m, 1e-12)
        score = float(np.linalg.norm(centered, ord="fro") ** 2 / max(m * m, 1e-12))
        self.bucket = min(float(self.model.cfg.bucket_capacity), self.bucket + float(self.model.cfg.replenish_rate))
        alarm = 1 if score >= float(self.model.q_low) else 0
        has_tokens = self.bucket >= float(self.model.cfg.heavy_audit_cost_ms)
        slack_before = float(row["deadline_ms"]) - float(row["baseline_latency_ms"]) - float(self.model.cfg.monitor_cost_ms)
        if self.method == "CIRCA-RT":
            has_slack = True
        else:
            has_slack = slack_before >= self.audit_cost_ms + self.slack_margin_ms
        audit = 1 if score >= float(self.model.q_high) and has_tokens and has_slack else 0
        if audit:
            self.bucket -= float(self.model.cfg.heavy_audit_cost_ms)
        return {
            "score": score,
            "conditional_semantic_residual": float(u[0]),
            "conditional_timing_residual": float(v[0]),
            "alarm": alarm,
            "audit": audit,
            "bucket_level": float(self.bucket),
            "deadline_slack_before_audit_ms": slack_before,
            "slack_admissible": int(has_slack),
            "q_low": float(self.model.q_low),
            "q_high": float(self.model.q_high),
        }


class PerceptionMonitor(Node):
    def __init__(self, args):
        super().__init__("circa_rt_agx_perception_monitor")
        self.args = args
        self.device = resolve_device(args.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        torch.set_grad_enabled(False)
        self.primary_model, self.primary_preprocess = build_model(args.model, args.weights, self.device)
        self.audit_model = None
        self.audit_preprocess = None
        if args.audit_model != "none":
            self.audit_model, self.audit_preprocess = build_model(args.audit_model, args.audit_weights, self.device)
        self.stress_a = None
        self.stress_b = None
        if self.device.type == "cuda" and args.gpu_stress_size > 0:
            g = torch.Generator(device=self.device)
            g.manual_seed(12345 + args.gpu_stress_size)
            self.stress_a = torch.randn(args.gpu_stress_size, args.gpu_stress_size, generator=g, device=self.device)
            self.stress_b = torch.randn(args.gpu_stress_size, args.gpu_stress_size, generator=g, device=self.device)
        self.online = None
        if args.monitor_mode == "online":
            payload = pickle.loads(Path(args.circa_model).read_bytes())
            model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            self.online = OnlineCIRCA(model, args.method, args.token_audit_cost_ms, args.slack_margin_ms)
        self.records = []
        self.prev_probs = None
        self.prev_top1 = None
        self.prev_latency = 0.0
        self.start_ns = time.perf_counter_ns()
        self.sub = self.create_subscription(String, args.topic, self.on_msg, args.depth)

    def run_stress(self, scenario, attacked):
        if not attacked or self.stress_a is None or self.stress_b is None:
            return
        if scenario not in {"gpu_interference", "coupled_semantic_timing_attack"}:
            return
        repeats = max(1, int(self.args.gpu_stress_repeats))
        if scenario == "coupled_semantic_timing_attack":
            repeats += int(self.args.coupled_stress_extra_repeats)
        out = None
        for _ in range(repeats):
            out = self.stress_a @ self.stress_b
        if out is not None:
            _ = out[0, 0]

    def run_audit(self, image):
        if self.audit_model is None:
            return float(self.args.fallback_audit_cost_ms)
        x = self.audit_preprocess(image).unsqueeze(0).to(self.device)
        synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            _ = self.audit_model(x)
        synchronize(self.device)
        return float((time.perf_counter() - start) * 1000.0)

    def on_msg(self, msg):
        recv_ns = time.perf_counter_ns()
        payload = json.loads(msg.data)
        seq = int(payload["seq"])
        attacked = bool(payload["label"])
        scenario = str(payload["scenario"])
        image = payload_image(payload)
        if not bool(payload.get("image_perturbed_at_publisher", False)):
            image = perturb_image(image, scenario, attacked, seq, int(payload["seed"]))
        x = self.primary_preprocess(image).unsqueeze(0).to(self.device)

        synchronize(self.device)
        infer_start = time.perf_counter()
        with torch.inference_mode():
            logits = self.primary_model(x)
            self.run_stress(scenario, attacked)
        synchronize(self.device)
        primary_latency_ms = float((time.perf_counter() - infer_start) * 1000.0)
        after_primary_ns = time.perf_counter_ns()

        features, self.prev_probs, self.prev_top1 = logits_features(logits, self.prev_probs, self.prev_top1)
        ros2_latency_ms = float((recv_ns - int(payload["send_ns"])) / 1e6)
        raw_loop_latency_ms = float((after_primary_ns - int(payload["send_ns"])) / 1e6)
        jitter_ms = abs(raw_loop_latency_ms - self.prev_latency) if self.records else 0.0
        self.prev_latency = raw_loop_latency_ms
        semantic_residual = semantic_score(features)
        if attacked and scenario == "coupled_semantic_timing_attack":
            semantic_residual += 0.35 + 0.15 * math.sin(seq * 0.13)
        timing_residual_ms = raw_loop_latency_ms
        gpu_proxy = 0.55 + min(primary_latency_ms / max(self.args.deadline_ms, 1.0), 2.0) * 0.20
        if attacked and scenario in {"gpu_interference", "coupled_semantic_timing_attack"}:
            gpu_proxy += 0.25
        cpu_proxy = 0.30 + min(ros2_latency_ms / max(self.args.period_ms, 1.0), 2.0) * 0.15
        queue_depth = max(0.0, ros2_latency_ms / max(self.args.period_ms, 1.0))
        message_age = max(0.0, self.args.period_ms + ros2_latency_ms + primary_latency_ms)

        row = {
            "run_id": payload["run_id"],
            "platform": "agx_orin64_ros2_closed_loop",
            "timestamp_ms": float((after_primary_ns - self.start_ns) / 1e6),
            "seq": seq,
            "mode": payload["mode"],
            "attack_type": scenario if attacked else "benign",
            "label": int(attacked),
            "semantic_residual": float(semantic_residual),
            "timing_residual_ms": float(timing_residual_ms),
            "context_cpu_util": float(np.clip(cpu_proxy, 0.0, 1.0)),
            "context_gpu_util": float(np.clip(gpu_proxy, 0.0, 1.0)),
            "context_net_delay_ms": float(max(0.0, ros2_latency_ms)),
            "context_jitter_ms": float(jitter_ms),
            "context_queue_depth": float(queue_depth),
            "context_message_age_ms": float(message_age),
            "baseline_latency_ms": float(max(raw_loop_latency_ms, 0.001)),
            "e2e_latency_ms": float(max(raw_loop_latency_ms, 0.001)),
            "deadline_ms": float(self.args.deadline_ms),
            "deadline_miss": int(raw_loop_latency_ms > self.args.deadline_ms),
            "method": "raw" if self.args.monitor_mode == "calibration" else self.args.method,
            "alarm": 0,
            "audit": 0,
            "audit_cost_ms": 0.0,
            "monitor_cost_ms": 0.0,
            "score": 0.0,
            "bucket_level": float("nan"),
            "deadline_slack_before_audit_ms": float(self.args.deadline_ms - raw_loop_latency_ms),
            "slack_admissible": 0,
            "audit_token_charge_ms": 0.0,
            "audit_charge_ms": 0.0,
            "audit_latency_bound_ms": float(self.args.token_audit_cost_ms),
            "bucket_capacity": float(self.args.bucket_capacity),
            "replenish_rate": float(self.args.replenish_rate),
            "token_audit_cost_ms": float(self.args.token_audit_cost_ms),
            "ros2_transport_latency_ms": ros2_latency_ms,
            "primary_inference_latency_ms": primary_latency_ms,
            "audit_latency_ms": 0.0,
            "frame_index": int(payload["frame_index"]),
            "frame_name": payload["frame_path"],
            "scenario": scenario,
            "primary_model": self.args.model,
        }
        row.update(features)

        if self.online is not None:
            monitor_start = time.perf_counter()
            decision = self.online.step(row)
            monitor_latency_ms = float((time.perf_counter() - monitor_start) * 1000.0)
            row.update(decision)
            row["monitor_cost_ms"] = max(monitor_latency_ms, 0.001)
            if row["audit"]:
                audit_latency_ms = self.run_audit(image)
                row["audit_latency_ms"] = float(max(audit_latency_ms, 0.001))
                row["audit_cost_ms"] = row["audit_latency_ms"]
                row["audit_token_charge_ms"] = float(self.args.token_audit_cost_ms)
                row["audit_charge_ms"] = float(self.args.token_audit_cost_ms)
            row["e2e_latency_ms"] = (
                float(row["baseline_latency_ms"]) + float(row["monitor_cost_ms"]) + float(row["audit_cost_ms"])
            )
            row["deadline_miss"] = int(float(row["e2e_latency_ms"]) > float(row["deadline_ms"]))

        self.records.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--period-ms", type=float, required=True)
    parser.add_argument("--deadline-ms", type=float, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", choices=["default", "none"], default="default")
    parser.add_argument("--audit-model", default="resnet18")
    parser.add_argument("--audit-weights", choices=["default", "none"], default="default")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--monitor-mode", choices=["calibration", "online"], default="online")
    parser.add_argument("--method", choices=["CIRCA-RT", "CIRCA-RT-Slack"], default="CIRCA-RT-Slack")
    parser.add_argument("--circa-model", default="")
    parser.add_argument("--token-audit-cost-ms", type=float, default=4.0)
    parser.add_argument("--fallback-audit-cost-ms", type=float, default=4.0)
    parser.add_argument("--bucket-capacity", type=float, default=12.0)
    parser.add_argument("--replenish-rate", type=float, default=0.45)
    parser.add_argument("--slack-margin-ms", type=float, default=0.0)
    parser.add_argument("--gpu-stress-size", type=int, default=512)
    parser.add_argument("--gpu-stress-repeats", type=int, default=1)
    parser.add_argument("--coupled-stress-extra-repeats", type=int, default=4)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.root) / "src"))
    rclpy.init()
    node = PerceptionMonitor(args)
    deadline = time.time() + args.timeout_s
    try:
        while rclpy.ok() and len(node.records) < args.n and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(node.records), encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
"""


SCENARIOS = [
    "nominal",
    "mode_shift",
    "semantic_corruption",
    "gpu_interference",
    "stale_replay_attack",
    "coupled_semantic_timing_attack",
]

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

MODEL_CHOICES = ["mobilenet_v2", "resnet18", "resnet50", "squeezenet1_1"]


def load_base_circa_config() -> CIRCARuntimeConfig:
    cfg = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    c = cfg["circa_rt"]
    return CIRCARuntimeConfig(
        window_size=int(c["window_size"]),
        feature_dim=int(c["feature_dim"]),
        low_quantile=float(c["low_quantile"]),
        high_quantile=float(c["high_quantile"]),
        monitor_cost_ms=float(c["monitor_cost_ms"]),
        heavy_audit_cost_ms=float(c["heavy_audit_cost_ms"]),
        bucket_capacity=float(c["bucket_capacity"]),
        replenish_rate=float(c["replenish_rate"]),
        seed=int(cfg["random_seed"]),
    )


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    py_parts = [str(ROOT / "src"), str(ROOT)]
    if env.get("PYTHONPATH"):
        py_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_parts)
    return env


def write_node_scripts(raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pub = raw_dir / "agx_ros2_frame_publisher.py"
    mon = raw_dir / "agx_ros2_perception_monitor.py"
    pub.write_text(PUBLISHER_CODE, encoding="utf-8")
    mon.write_text(MONITOR_CODE, encoding="utf-8")
    return pub, mon


def check_runtime(node_python: str) -> None:
    cmd = [
        node_python,
        "-c",
        "import rclpy, std_msgs, torch, torchvision, PIL; print('agx_ros2_closed_loop_runtime_ok')",
    ]
    subprocess.run(cmd, check=True, timeout=20, env=child_env())


def run_pair(
    *,
    node_python: str,
    publisher_node: Path,
    monitor_node: Path,
    frame_manifest: Path,
    log_dir: Path,
    out_json: Path,
    topic: str,
    run_id: str,
    scenario: str,
    seed: int,
    n: int,
    period_ms: float,
    deadline_ms: float,
    model: str,
    weights: str,
    audit_model: str,
    audit_weights: str,
    device: str,
    monitor_mode: str,
    method: str,
    circa_model: Path | None,
    token_audit_cost_ms: float,
    fallback_audit_cost_ms: float,
    bucket_capacity: float,
    replenish_rate: float,
    slack_margin_ms: float,
    gpu_stress_size: int,
    gpu_stress_repeats: int,
    coupled_stress_extra_repeats: int,
    attack_len: int,
    clip_len: int,
    payload_mode: str,
    jpeg_quality: int,
    publish_resize: int,
) -> None:
    env = child_env()
    timeout_s = max(180.0, n * period_ms / 1000.0 + 180.0)
    monitor_cmd = [
        node_python,
        str(monitor_node),
        "--root",
        str(ROOT),
        "--topic",
        topic,
        "--out-json",
        str(out_json),
        "--n",
        str(n),
        "--period-ms",
        str(period_ms),
        "--deadline-ms",
        str(deadline_ms),
        "--model",
        model,
        "--weights",
        weights,
        "--audit-model",
        audit_model,
        "--audit-weights",
        audit_weights,
        "--device",
        device,
        "--monitor-mode",
        monitor_mode,
        "--method",
        method,
        "--token-audit-cost-ms",
        str(token_audit_cost_ms),
        "--fallback-audit-cost-ms",
        str(fallback_audit_cost_ms),
        "--bucket-capacity",
        str(bucket_capacity),
        "--replenish-rate",
        str(replenish_rate),
        "--slack-margin-ms",
        str(slack_margin_ms),
        "--gpu-stress-size",
        str(gpu_stress_size),
        "--gpu-stress-repeats",
        str(gpu_stress_repeats),
        "--coupled-stress-extra-repeats",
        str(coupled_stress_extra_repeats),
        "--timeout-s",
        str(timeout_s),
    ]
    if circa_model is not None:
        monitor_cmd.extend(["--circa-model", str(circa_model)])

    publisher_cmd = [
        node_python,
        str(publisher_node),
        "--topic",
        topic,
        "--frame-manifest",
        str(frame_manifest),
        "--scenario",
        scenario,
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--n",
        str(n),
        "--period-ms",
        str(period_ms),
        "--attack-len",
        str(attack_len),
        "--clip-len",
        str(clip_len),
        "--payload-mode",
        payload_mode,
        "--jpeg-quality",
        str(jpeg_quality),
        "--publish-resize",
        str(publish_resize),
        "--startup-delay-s",
        "2.0",
    ]

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{run_id}.monitor.cmd").write_text(" ".join(monitor_cmd), encoding="utf-8")
    (log_dir / f"{run_id}.publisher.cmd").write_text(" ".join(publisher_cmd), encoding="utf-8")
    with (log_dir / f"{run_id}.monitor.log").open("w", encoding="utf-8") as mon_log, (
        log_dir / f"{run_id}.publisher.log"
    ).open("w", encoding="utf-8") as pub_log:
        monitor_proc = subprocess.Popen(monitor_cmd, env=env, stdout=mon_log, stderr=subprocess.STDOUT)
        time.sleep(1.0)
        publisher_proc = subprocess.Popen(publisher_cmd, env=env, stdout=pub_log, stderr=subprocess.STDOUT)
        try:
            publisher_proc.wait(timeout=timeout_s)
            monitor_proc.wait(timeout=timeout_s)
        finally:
            for proc in [publisher_proc, monitor_proc]:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        proc.kill()
    if publisher_proc.returncode not in (0, None):
        raise RuntimeError(f"publisher failed for {run_id} with code {publisher_proc.returncode}")
    if monitor_proc.returncode not in (0, None):
        raise RuntimeError(f"monitor failed for {run_id} with code {monitor_proc.returncode}")


def repair_trace(df: pd.DataFrame, *, scenario: str, seed: int, n: int, deadline_ms: float) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"empty ROS2 closed-loop trace for {scenario} seed {seed}")
    df = df.sort_values("seq").drop_duplicates("seq").reset_index(drop=True)
    if len(df) < max(50, int(0.80 * n)):
        raise ValueError(f"received too few ROS2 closed-loop samples for {scenario} seed {seed}: {len(df)} / {n}")
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df["seq"] = df["seq"].astype(int)
    df["label"] = df["label"].astype(int)
    df["deadline_ms"] = float(deadline_ms)
    df["deadline_miss"] = (df["e2e_latency_ms"].astype(float) > float(deadline_ms)).astype(int)
    if "audit_token_charge_ms" not in df.columns:
        df["audit_token_charge_ms"] = df["audit_cost_ms"]
    if "audit_charge_ms" not in df.columns:
        df["audit_charge_ms"] = df["audit_token_charge_ms"]
    return df


def write_tables(out: Path, summary_rows: list[dict[str, Any]]) -> None:
    summary_dir = out / "summaries"
    table_dir = out / "tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)
    if summary.empty:
        return
    attack_summary = summary[summary["trace_type"].isin(["semantic_corruption", "gpu_interference", "stale_replay_attack", "coupled_semantic_timing_attack"])].copy()
    main = mean_table(attack_summary, ["primary_model", "method"], METRIC_COLS).sort_values(
        ["primary_model", "attack_recall", "audit_rate", "p99_latency_ms"],
        ascending=[True, False, True, True],
    )
    main.to_csv(table_dir / "agx_ros2_closed_loop_main_table.csv", index=False)
    bootstrap_ci_table(attack_summary, ["primary_model", "method"], METRIC_COLS).to_csv(
        table_dir / "agx_ros2_closed_loop_main_table_ci.csv", index=False
    )
    scenario = mean_table(summary, ["primary_model", "trace_type", "method"], METRIC_COLS).sort_values(
        ["primary_model", "trace_type", "attack_recall", "audit_rate"],
        ascending=[True, True, False, True],
    )
    scenario.to_csv(table_dir / "agx_ros2_closed_loop_scenario_table.csv", index=False)


def write_latency_table(out: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted((out / "traces").rglob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        lat = df["e2e_latency_ms"].to_numpy(dtype=float)
        rows.append(
            {
                "trace": str(path.relative_to(out / "traces")),
                "primary_model": df["primary_model"].iloc[0] if "primary_model" in df else "",
                "scenario": path.stem,
                "method": df["method"].iloc[0],
                "n": int(len(df)),
                "ros2_transport_p99_ms": float(df["ros2_transport_latency_ms"].quantile(0.99))
                if "ros2_transport_latency_ms" in df
                else float("nan"),
                "primary_inference_p99_ms": float(df["primary_inference_latency_ms"].quantile(0.99))
                if "primary_inference_latency_ms" in df
                else float("nan"),
                "audit_rate": float(df["audit"].mean()),
                "mean_audit_latency_ms": float(df["audit_latency_ms"].mean()) if "audit_latency_ms" in df else float("nan"),
                "p50_latency_ms": float(np.quantile(lat, 0.50)),
                "p95_latency_ms": float(np.quantile(lat, 0.95)),
                "p99_latency_ms": float(np.quantile(lat, 0.99)),
                "deadline_miss_ratio": float(df["deadline_miss"].mean()),
            }
        )
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_dir / "agx_ros2_closed_loop_latency_table.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AGX ROS2 closed-loop frame publisher, perception monitor, and audit loop.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results_agx_orin64_ros2_closed_loop")
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--frame-limit", type=int, default=900)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=["mobilenet_v2"])
    parser.add_argument("--weights", choices=["default", "none"], default="default")
    parser.add_argument("--audit-model", choices=MODEL_CHOICES + ["none"], default="resnet18")
    parser.add_argument("--audit-weights", choices=["default", "none"], default="default")
    parser.add_argument("--node-python", default="python3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--period-ms", type=float, default=33.333)
    parser.add_argument("--deadline-ms", type=float, default=33.333)
    parser.add_argument("--method", choices=["CIRCA-RT", "CIRCA-RT-Slack"], default="CIRCA-RT-Slack")
    parser.add_argument("--circa-low-quantile", type=float, default=0.95)
    parser.add_argument("--circa-high-quantile", type=float, default=0.99)
    parser.add_argument("--token-audit-cost-ms", type=float, default=4.0)
    parser.add_argument("--fallback-audit-cost-ms", type=float, default=4.0)
    parser.add_argument("--bucket-capacity", type=float, default=12.0)
    parser.add_argument("--replenish-rate", type=float, default=0.45)
    parser.add_argument("--slack-margin-ms", type=float, default=0.0)
    parser.add_argument("--gpu-stress-size", type=int, default=512)
    parser.add_argument("--gpu-stress-repeats", type=int, default=1)
    parser.add_argument("--coupled-stress-extra-repeats", type=int, default=4)
    parser.add_argument("--attack-len", type=int, default=80)
    parser.add_argument("--clip-len", type=int, default=12)
    parser.add_argument("--payload-mode", choices=["jpeg_b64", "path"], default="jpeg_b64")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--publish-resize", type=int, default=320)
    args = parser.parse_args()

    if args.n < 80:
        raise SystemExit("use --n >= 80 so CIRCA-RT has enough online window samples")
    if not args.frame_dir.exists():
        raise SystemExit(f"missing frame directory: {args.frame_dir}")

    out = args.out_dir
    raw_json_dir = out / "raw_json"
    trace_dir = out / "traces"
    raw_dir = out / "raw"
    model_dir = out / "models"
    table_dir = out / "tables"
    log_dir = out / "logs"
    for d in [raw_json_dir, trace_dir, raw_dir, model_dir, table_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    check_runtime(args.node_python)
    publisher_node, monitor_node = write_node_scripts(raw_json_dir / "nodes")

    frames = [str(p) for p in list_frame_paths(args.frame_dir, limit=max(args.frame_limit, args.n * 2))]
    if len(frames) < max(10, min(args.n, args.frame_limit)):
        raise SystemExit(f"too few frames in {args.frame_dir}: {len(frames)}")
    frame_manifest = raw_json_dir / "frame_manifest.json"
    frame_manifest.write_text(json.dumps(frames, indent=2), encoding="utf-8")

    base_cfg = load_base_circa_config()
    summary_rows: list[dict[str, Any]] = []
    run_config = vars(args).copy()
    run_config["frame_dir"] = str(args.frame_dir)
    run_config["out_dir"] = str(args.out_dir)
    run_config["frame_count"] = len(frames)
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    for model_name in args.models:
        for seed in args.seeds:
            prefix = f"{model_name}_seed{seed}"
            topic = f"/circa_rt/agx_closed_loop/{model_name}/seed{seed}"
            calibration_run_id = f"agx_ros2_{prefix}_nominal_calibration"
            calibration_json = raw_json_dir / model_name / f"seed{seed}" / "nominal_calibration.json"
            print(f"running calibration {calibration_run_id}", flush=True)
            run_pair(
                node_python=args.node_python,
                publisher_node=publisher_node,
                monitor_node=monitor_node,
                frame_manifest=frame_manifest,
                log_dir=log_dir,
                out_json=calibration_json,
                topic=topic,
                run_id=calibration_run_id,
                scenario="nominal",
                seed=seed,
                n=args.n,
                period_ms=args.period_ms,
                deadline_ms=args.deadline_ms,
                model=model_name,
                weights=args.weights,
                audit_model=args.audit_model,
                audit_weights=args.audit_weights,
                device=args.device,
                monitor_mode="calibration",
                method=args.method,
                circa_model=None,
                token_audit_cost_ms=args.token_audit_cost_ms,
                fallback_audit_cost_ms=args.fallback_audit_cost_ms,
                bucket_capacity=args.bucket_capacity,
                replenish_rate=args.replenish_rate,
                slack_margin_ms=args.slack_margin_ms,
                gpu_stress_size=args.gpu_stress_size,
                gpu_stress_repeats=args.gpu_stress_repeats,
                coupled_stress_extra_repeats=args.coupled_stress_extra_repeats,
                attack_len=args.attack_len,
                clip_len=args.clip_len,
                payload_mode=args.payload_mode,
                jpeg_quality=args.jpeg_quality,
                publish_resize=args.publish_resize,
            )
            calibration = repair_trace(
                pd.read_json(calibration_json),
                scenario="nominal",
                seed=seed,
                n=args.n,
                deadline_ms=args.deadline_ms,
            )
            calibration_path = trace_dir / model_name / f"seed{seed}" / "nominal_calibration.csv"
            write_trace(calibration, calibration_path)

            cfg = replace(
                base_cfg,
                low_quantile=args.circa_low_quantile,
                high_quantile=args.circa_high_quantile,
                heavy_audit_cost_ms=args.token_audit_cost_ms,
                bucket_capacity=args.bucket_capacity,
                replenish_rate=args.replenish_rate,
                slack_margin_ms=args.slack_margin_ms,
                seed=seed,
            )
            circa_model = fit_circa_rt(calibration, cfg)
            circa_model_path = model_dir / f"{prefix}_circa_model.pkl"
            circa_model_path.write_bytes(pickle.dumps({"model": circa_model, "cfg": cfg, "calibration": str(calibration_path)}))

            for scenario in SCENARIOS:
                run_id = f"agx_ros2_{prefix}_{scenario}"
                out_json = raw_json_dir / model_name / f"seed{seed}" / f"{scenario}.json"
                print(f"running online {run_id}", flush=True)
                run_pair(
                    node_python=args.node_python,
                    publisher_node=publisher_node,
                    monitor_node=monitor_node,
                    frame_manifest=frame_manifest,
                    log_dir=log_dir,
                    out_json=out_json,
                    topic=topic,
                    run_id=run_id,
                    scenario=scenario,
                    seed=seed,
                    n=args.n,
                    period_ms=args.period_ms,
                    deadline_ms=args.deadline_ms,
                    model=model_name,
                    weights=args.weights,
                    audit_model=args.audit_model,
                    audit_weights=args.audit_weights,
                    device=args.device,
                    monitor_mode="online",
                    method=args.method,
                    circa_model=circa_model_path,
                    token_audit_cost_ms=args.token_audit_cost_ms,
                    fallback_audit_cost_ms=args.fallback_audit_cost_ms,
                    bucket_capacity=args.bucket_capacity,
                    replenish_rate=args.replenish_rate,
                    slack_margin_ms=args.slack_margin_ms,
                    gpu_stress_size=args.gpu_stress_size,
                    gpu_stress_repeats=args.gpu_stress_repeats,
                    coupled_stress_extra_repeats=args.coupled_stress_extra_repeats,
                    attack_len=args.attack_len,
                    clip_len=args.clip_len,
                    payload_mode=args.payload_mode,
                    jpeg_quality=args.jpeg_quality,
                    publish_resize=args.publish_resize,
                )
                trace = repair_trace(
                    pd.read_json(out_json),
                    scenario=scenario,
                    seed=seed,
                    n=args.n,
                    deadline_ms=args.deadline_ms,
                )
                trace_path = trace_dir / model_name / f"seed{seed}" / f"{scenario}.csv"
                write_trace(trace, trace_path)
                scored_dir = raw_dir / model_name / f"seed{seed}" / scenario / args.method
                scored_dir.mkdir(parents=True, exist_ok=True)
                trace.to_csv(scored_dir / "scored.csv", index=False)
                row = summarize_detection(trace, trace_name=f"{prefix}_{scenario}").to_dict()
                row.update(
                    {
                        "primary_model": model_name,
                        "seed": seed,
                        "trace_type": scenario,
                        "protocol": "agx_ros2_frame_publisher_perception_monitor_inline_audit",
                        "period_ms": args.period_ms,
                        "deadline_ms": args.deadline_ms,
                        "circa_low_quantile": args.circa_low_quantile,
                        "circa_high_quantile": args.circa_high_quantile,
                        "token_audit_cost_ms": args.token_audit_cost_ms,
                        "bucket_capacity": args.bucket_capacity,
                        "replenish_rate": args.replenish_rate,
                        "slack_margin_ms": args.slack_margin_ms,
                    }
                )
                summary_rows.append(row)

    write_tables(out, summary_rows)
    write_latency_table(out)
    main_table = table_dir / "agx_ros2_closed_loop_main_table.csv"
    if main_table.exists():
        print(pd.read_csv(main_table).to_string(index=False), flush=True)
    print(f"wrote AGX ROS2 closed-loop results: {out}", flush=True)


if __name__ == "__main__":
    main()
