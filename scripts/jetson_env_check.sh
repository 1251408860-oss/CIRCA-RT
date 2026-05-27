#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== system ==="
uname -a
lsb_release -a || true
cat /etc/nv_tegra_release || true
cat /proc/device-tree/model 2>/dev/null || true
cat /proc/device-tree/compatible 2>/dev/null || true
dpkg-query -W 'nvidia-l4t-*' 2>/dev/null || true

echo "=== python ==="
"${PYTHON_BIN}" --version
echo "TORCH_HOME=${TORCH_HOME:-}"
if [[ -n "${TORCH_HOME:-}" ]]; then
    find "${TORCH_HOME}/hub/checkpoints" -maxdepth 1 -type f -printf "%f %s\n" 2>/dev/null || true
fi
"${PYTHON_BIN}" - <<'PY' || true
try:
    import tensorrt as trt
    print("tensorrt", trt.__version__)
except Exception as exc:
    print("tensorrt import failed:", exc)

try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch device", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch import failed:", exc)

try:
    import torchvision
    print("torchvision", torchvision.__version__)
except Exception as exc:
    print("torchvision import failed:", exc)

try:
    import onnxruntime as ort
    print("onnxruntime", ort.__version__)
    print("providers", ort.get_available_providers())
except Exception as exc:
    print("onnxruntime import failed:", exc)
PY

echo "=== jetson tools ==="
which tegrastats || true
which nvpmodel || true
which jetson_clocks || true
sudo nvpmodel -q || true
sudo jetson_clocks --show || true

echo "=== ros2 ==="
which ros2 || true
ros2 --version || true
printenv ROS_DISTRO || true

echo "=== cuda/tensorrt libraries ==="
ldconfig -p | grep -E "libnvinfer|libcudart" || true

echo "=== disk ==="
df -h
free -h || true
