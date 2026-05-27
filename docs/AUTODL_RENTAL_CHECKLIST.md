# AutoDL Rental Checklist

更新时间：2026-05-17

租 AutoDL 或普通 NVIDIA GPU 云主机前，优先确认这些条件。

## Recommended GPU

最低：

- RTX 3060 / RTX 4060 / RTX 4090 / A10 / L4 任意一种
- 显存 8GB 以上

更稳：

- RTX 4090
- A10
- L4

不需要：

- 多卡
- A100
- H100

## Recommended Image

优先：

- Ubuntu 22.04
- CUDA 11.8 / 12.x
- PyTorch 已安装
- TensorRT 已安装最好

ROS2：

- Ubuntu 22.04: ROS2 Humble
- Ubuntu 24.04: ROS2 Jazzy

## Must Check After Login

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python -c "import onnxruntime as ort; print(ort.__version__)"
python -c "import tensorrt as trt; print(trt.__version__)"
df -h
```

If TensorRT is missing, AutoDL can still run:

- PyTorch latency
- ONNXRuntime latency
- batch baselines
- ROS2/Python timing pipeline

But for the paper, TensorRT is strongly preferred.

## Rental Time

Recommended first rental:

- 24 hours for environment setup and smoke tests

Recommended full run:

- 48-72 hours

## Reject Conditions

Do not rent if:

- `nvidia-smi` unavailable
- cannot install Python packages
- cannot download files
- no terminal/SSH/Jupyter terminal access
- storage too small for results
