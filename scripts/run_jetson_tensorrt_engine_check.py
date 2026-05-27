from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "cmd": cmd,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"cmd": cmd, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Jetson TensorRT/ONNXRuntime execution environment.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "uname": run_cmd(["uname", "-a"]),
        "nv_tegra_release": run_cmd(["bash", "-lc", "cat /etc/nv_tegra_release 2>/dev/null || true"]),
        "trtexec_version": run_cmd(["trtexec", "--version"]),
        "nvpmodel_query": run_cmd(["bash", "-lc", "sudo nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || true"]),
        "jetson_clocks": run_cmd(["bash", "-lc", "sudo jetson_clocks --show 2>/dev/null || jetson_clocks --show 2>/dev/null || true"]),
    }

    try:
        import tensorrt as trt  # type: ignore

        report["tensorrt"] = {"available": True, "version": getattr(trt, "__version__", "unknown")}
    except Exception as exc:  # pragma: no cover - environment dependent
        report["tensorrt"] = {"available": False, "error": repr(exc)}

    try:
        import onnxruntime as ort  # type: ignore

        report["onnxruntime"] = {
            "available": True,
            "version": getattr(ort, "__version__", "unknown"),
            "providers": getattr(ort, "get_available_providers", lambda: [])(),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        report["onnxruntime"] = {"available": False, "error": repr(exc)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
