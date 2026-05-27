from __future__ import annotations

import argparse
import os
import json
import re
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TIME_MIXER_ROOT = ROOT / "third_party" / "baselines" / "TimeMixer"
FULL_ARGS = {
    "seq_len": 64,
    "label_len": 0,
    "pred_len": 0,
    "d_model": 8,
    "d_ff": 16,
    "e_layers": 1,
    "d_layers": 1,
    "down_sampling_layers": 1,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "enc_in": None,
    "c_out": None,
    "itr": 1,
    "batch_size": 64,
    "train_epochs": 2,
    "patience": 2,
    "learning_rate": 0.001,
}

SMOKE_ARGS = {
    "seq_len": 8,
    "label_len": 0,
    "pred_len": 0,
    "d_model": 4,
    "d_ff": 8,
    "e_layers": 1,
    "d_layers": 1,
    "down_sampling_layers": 1,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "itr": 1,
    "batch_size": 32,
    "train_epochs": 2,
    "patience": 2,
    "learning_rate": 0.001,
}

TIMEOUT_RETURN_CODE = 124


def _read_meta(case_dir: Path) -> dict[str, object]:
    return json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))


def _build_cmd(case_dir: Path, meta: dict[str, object], device: str, *, smoke: bool) -> list[str]:
    feat_dim = int(meta["feature_dim"])
    anomaly_ratio = max(0.01, float(meta["anomaly_ratio_pct"]))
    model_id = f"{meta['model']}_seed{meta['seed']}_{meta['scenario']}"
    cfg = SMOKE_ARGS if smoke else FULL_ARGS
    argv = [
        "run.py",
        "--task_name",
        "anomaly_detection",
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        "TimeMixer",
        "--data",
        "PSM",
        "--root_path",
        str(case_dir.resolve()),
        "--seq_len",
        str(cfg["seq_len"]),
        "--label_len",
        str(cfg["label_len"]),
        "--pred_len",
        str(cfg["pred_len"]),
        "--enc_in",
        str(feat_dim),
        "--dec_in",
        str(feat_dim),
        "--c_out",
        str(feat_dim),
        "--d_model",
        str(cfg["d_model"]),
        "--d_ff",
        str(cfg["d_ff"]),
        "--e_layers",
        str(cfg["e_layers"]),
        "--d_layers",
        str(cfg["d_layers"]),
        "--down_sampling_layers",
        str(cfg["down_sampling_layers"]),
        "--down_sampling_window",
        str(cfg["down_sampling_window"]),
        "--down_sampling_method",
        str(cfg["down_sampling_method"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--num_workers",
        "0",
        "--train_epochs",
        str(cfg["train_epochs"]),
        "--patience",
        str(cfg["patience"]),
        "--learning_rate",
        str(cfg["learning_rate"]),
        "--itr",
        str(cfg["itr"]),
        "--anomaly_ratio",
        f"{anomaly_ratio}",
        "--des",
        "RTSS",
    ]
    if device == "cuda":
        argv.extend(["--use_gpu", "True"])
    else:
        argv.extend(["--use_gpu", "False"])
    return [
        sys.executable,
        "-c",
        (
            "import runpy, sys, numpy as np; "
            "np.Inf = np.inf; "
            f"sys.argv = {argv!r}; "
            "runpy.run_path('run.py', run_name='__main__')"
        ),
    ]


def _parse_metrics(text: str) -> dict[str, float] | None:
    m = re.search(
        r"Accuracy\s*:\s*([0-9.]+),\s*Precision\s*:\s*([0-9.]+),\s*Recall\s*:\s*([0-9.]+),\s*F-score\s*:\s*([0-9.]+)",
        text,
    )
    if not m:
        return None
    return {
        "accuracy": float(m.group(1)),
        "precision": float(m.group(2)),
        "recall": float(m.group(3)),
        "f1": float(m.group(4)),
    }


def _run_command(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: int | None) -> tuple[int, bool, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return int(proc.returncode), False, stdout or "", stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stderr = stderr + f"\nTimed out after {timeout} seconds."
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait()
        return TIMEOUT_RETURN_CODE, True, stdout, stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official TimeMixer anomaly-detection code on PSM-style exports.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=0,
        help="Terminate the official run after this many seconds. Use 0 to disable.",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    meta = _read_meta(case_dir)
    cmd = _build_cmd(case_dir, meta, device=args.device, smoke=args.smoke)
    env = os.environ.copy()
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    timeout = args.timeout_sec if args.timeout_sec > 0 else None
    returncode, timed_out, stdout, stderr = _run_command(cmd, cwd=TIME_MIXER_ROOT, env=env, timeout=timeout)
    combined = stdout + "\n" + stderr
    metrics = _parse_metrics(combined)
    result = {
        "case_dir": str(case_dir),
        "returncode": returncode,
        "timed_out": timed_out,
        "smoke": bool(args.smoke),
        "metrics": metrics,
        "command": cmd,
        "stdout_tail": stdout[-8000:],
        "stderr_tail": stderr[-8000:],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
