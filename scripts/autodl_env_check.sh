#!/usr/bin/env bash
set -euo pipefail

echo "=== system ==="
uname -a
lsb_release -a || true
df -h

echo "=== gpu ==="
nvidia-smi || true

echo "=== python ==="
PYTHON_BIN=""
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [ -n "${PYTHON_BIN}" ]; then
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" - <<'PY' || true
mods = ["torch", "onnxruntime", "tensorrt", "numpy", "pandas", "sklearn"]
for name in mods:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        print(name, version)
        if name == "torch":
            print("torch cuda available", mod.cuda.is_available())
            print("torch cuda version", mod.version.cuda)
    except Exception as exc:
        print(name, "IMPORT_FAILED", exc)
PY
else
  echo "python IMPORT_FAILED no python/python3 in PATH"
fi

echo "=== ros2 ==="
which ros2 || true
ros2 --version || true
printenv ROS_DISTRO || true
