# Jetson Rental Checklist

更新时间：2026-05-16

租 Jetson Orin 前，把下面问题直接发给商家或平台客服。

## Must Have

1. 设备型号是什么？
   - 接受：Jetson Orin NX, AGX Orin
   - 不建议：Nano, Xavier NX unless no alternative

2. JetPack 版本是什么？
   - 推荐：JetPack 6.x
   - 可接受：JetPack 5.x，但脚本要调整

3. 系统版本是什么？
   - JetPack 6.x 通常是 Ubuntu 22.04
   - Ubuntu 22.04 上优先 ROS2 Humble

4. 是否允许 SSH？
   - 必须允许

5. 是否允许 sudo？
   - 必须允许，否则 `nvpmodel`, `jetson_clocks`, ROS2 环境安装都会受限

6. TensorRT 是否已经安装？
   - 必须能运行：

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

7. 是否可以运行 `tegrastats`？
   - 必须可以

8. 是否可以运行 `nvpmodel` 和 `jetson_clocks`？
   - 强烈建议可以

9. 是否可以安装 ROS2 Humble？
   - 强方案必须要 ROS2；如果不能安装，要确认是否已有 ROS2

10. 是否可以长期保存和下载实验数据？
    - 至少需要下载 CSV、日志、图片

## Nice To Have

- Docker 可用
- 网络稳定
- 允许开多个 SSH session
- 允许后台进程
- 有摄像头或 sample video，但不是必须
- 已安装 OpenCV / PyTorch / ONNXRuntime

## Reject Conditions

不要租：

- 不给 SSH
- 不给 sudo
- TensorRT 不可用且不能安装
- `tegrastats` 不可用
- 数据不能下载
- 只给 Web Notebook，不给系统权限

## First Command After Login

```bash
git clone <your_repo_or_upload_project>
cd exp_begin
bash scripts/jetson_env_check.sh | tee jetson_runs/raw_logs/env_check.log
```

如果 `jetson_runs/raw_logs` 不存在，先运行：

```bash
bash scripts/prepare_jetson_strong_pipeline.sh
```
