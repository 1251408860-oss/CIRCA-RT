# AGX Orin 64GB 外部电脑运行说明

更新日期：2026-05-24

用途：把 `F:\RTSS\exp_begin` 中的 AGX Orin 64GB 真实设备实验打包后，交给能 SSH 登录 Jetson AGX Orin 64GB 的同学或老师运行。这个包只用于生成新的真实 AGX 证据，不能复用旧的 AGX 结果目录。

## 1. 应该交付哪个包

优先交付完整包：

```text
agx_orin64_run_package_20260524_full.zip
```

如果明确要“最全”交付，优先交付 maximal 包：

```text
agx_orin64_run_package_20260524_max.zip
```

maximal 包在完整包基础上额外包含：

- `torch_cache/hub/checkpoints/`：MobileNetV2、ResNet18、ResNet50、SqueezeNet1.1 的 torchvision 预训练权重缓存。
- `data/torchvision`：本地已有 CIFAR-10 缓存，用于补充脚本或离线检查。
- `data/official_*`：官方 baseline 输入小文件，用于后续补充检查。
- `scripts/run_agx_orin64_maximal_experiment.sh`：一键最全 AGX 矩阵。

完整包包含：

- `src/`：CIRCA-RT 核心代码。
- `scripts/`：AGX 综合实验、环境检查、TensorRT/ONNXRuntime 检查、结果汇总脚本。
- `configs/`：实验配置。
- `docs/`、`paper_draft/`：说明、理论边界和已有实验背景。
- `requirements_agx_orin64_runtime.txt`：除 PyTorch/torchvision/ONNXRuntime/TensorRT 外的 Python 依赖。
- `data/frames_av2`、`data/frames_droid`、`data/frames_nuimages`：可直接运行的真实帧数据。

如果网速或磁盘空间不够，再交付轻量包：

```text
agx_orin64_run_package_20260524_code.zip
```

轻量包不含帧数据。对方需要把三套帧数据放到：

```text
data/frames_av2
data/frames_droid
data/frames_nuimages
```

## 2. 对方机器最低要求

硬件和系统：

- Jetson AGX Orin 64GB。
- 能 SSH 登录。
- 能运行 `sudo`。
- 至少 8 GB 空闲磁盘可跑 smoke，建议 40 GB 以上跑完整实验。
- `tegrastats`、`nvpmodel`、`jetson_clocks` 可用。

Python 和推理环境：

- `python3` 可用。
- PyTorch + torchvision 可用并能看到 CUDA。
- ONNXRuntime CUDA/TensorRT 和 Python TensorRT 如果可用，就跑 backend matrix。
- 如果 ONNXRuntime TensorRT 不可用，也可以先跑 PyTorch CUDA 真实帧闭环，但论文里不能写 TensorRT 结果。

## 3. 解压和进入目录

建议在 Jetson 上新建干净目录：

```bash
mkdir -p ~/rtss_agx
cd ~/rtss_agx
unzip agx_orin64_run_package_20260524_full.zip
cd agx_orin64_run_package_20260524_full
```

检查数据是否存在：

```bash
du -sh data/frames_av2 data/frames_droid data/frames_nuimages
find data/frames_av2 -type f | wc -l
find data/frames_droid -type f | wc -l
find data/frames_nuimages -type f | wc -l
```

本地打包时的参考规模：

```text
data/frames_av2       2169 files, about 797 MB
data/frames_droid      901 files, about 20 MB
data/frames_nuimages   218 files, about 38 MB
```

## 4. 环境安装

Jetson 上不要直接安装普通 x86 PyTorch wheel。优先使用当前 JetPack/L4T 对应的 NVIDIA PyTorch/torchvision wheel，或者使用已经配好的 Jetson 容器。

如果系统 Python 已经可以 import `torch` 和 `torchvision`，可以直接补项目依赖：

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_agx_orin64_runtime.txt
```

如果要建虚拟环境，并且 PyTorch 已经装在系统 Python 里，建议：

```bash
python3 -m venv --system-site-packages ~/venvs/circa_rt_agx
source ~/venvs/circa_rt_agx/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_agx_orin64_runtime.txt
```

检查推理环境：

```bash
PYTHON_BIN=python3 bash scripts/jetson_env_check.sh | tee jetson_env_check_first.log
python3 scripts/run_jetson_tensorrt_engine_check.py \
  --out results_agx_orin64_env/provider_status.json
```

必须保留 `jetson_env_check_first.log` 和 `results_agx_orin64_env/provider_status.json`，它们是证明真实设备和推理 backend 的证据。

## 5. 先跑 smoke

不要一上来跑完整矩阵。先跑一个小 smoke，确认代码、数据、CUDA、日志都正常：

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_full

PYTHON_BIN=python3 \
RESULT_ROOT=results_agx_orin64_comprehensive_smoke \
AGX_DATASETS="av2" \
DEADLINES_MS="33.333" \
MODELS="mobilenet_v2" \
SEEDS="7" \
N_AV2=300 \
FRAME_LIMIT_AV2=300 \
RUN_BACKEND_MATRIX=0 \
bash scripts/run_agx_orin64_comprehensive_experiment.sh
```

smoke 通过的最低标准：

- `results_agx_orin64_comprehensive_smoke/env/` 存在。
- `results_agx_orin64_comprehensive_smoke/logs/` 存在。
- 至少有一个 `tegrastats_*.log`。
- `realframes/av2_d33p333/raw/` 下有 scored CSV。
- `tables/agx_comprehensive_manifest.json` 存在。
- `AGX_ORIN64_COMPREHENSIVE_REPORT.md` 存在。

如果 smoke 失败，先修环境，不要继续跑完整实验。

## 6. 完整实验命令

如果 TensorRT/ONNXRuntime 环境完整，跑：

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_full

PYTHON_BIN=python3 \
RESULT_ROOT=results_agx_orin64_comprehensive_202605xx \
AGX_DATASETS="av2 droid nuimages" \
DEADLINES_MS="33.333 50" \
MODELS="mobilenet_v2 resnet18" \
SEEDS="7 8 9" \
RUN_BACKEND_MATRIX=1 \
RUN_DERIVED_ANALYSES=1 \
bash scripts/run_agx_orin64_comprehensive_experiment.sh
```

如果 ONNXRuntime CUDA/TensorRT 不可用，但 PyTorch CUDA 可用，先跑真实帧闭环：

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_full

PYTHON_BIN=python3 \
RESULT_ROOT=results_agx_orin64_comprehensive_202605xx_torch_cuda \
AGX_DATASETS="av2 droid nuimages" \
DEADLINES_MS="33.333 50" \
MODELS="mobilenet_v2 resnet18" \
SEEDS="7 8 9" \
RUN_BACKEND_MATRIX=0 \
RUN_DERIVED_ANALYSES=1 \
bash scripts/run_agx_orin64_comprehensive_experiment.sh
```

这仍然是有效的真实 AGX Orin 64GB edge-device 证据，但论文表述必须写 PyTorch/CUDA Jetson execution，不能写 TensorRT。

## 6.1 最全实验命令

如果时间充足，并且你要最大化投稿证据，跑 maximal wrapper：

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_max

PYTHON_BIN=python3 \
bash scripts/run_agx_orin64_maximal_experiment.sh
```

默认 maximal 矩阵：

- 数据集：AV2、DROID、nuImages。
- deadline：16.667、20、25、33.333、40、50、100 ms。
- perception model：SqueezeNet1.1、MobileNetV2、ResNet18、ResNet50。
- seed：7、8、9、10、11。
- 方法：CIRCA-RT、CIRCA-RT-Slack、ConditionalRFF-HSIC、ContextAwareConformal、RFF-HSIC、TimingThreshold、SemanticThreshold、AlwaysAudit、RandomBudgetAudit。
- backend matrix：PyTorch CUDA、ONNX CUDA、ONNXRuntime TensorRT，模型为 SqueezeNet1.1、MobileNetV2、ResNet18。
- 派生分析：deadline-safe admission、CIRCA slack sensitivity、audit-bound validation、tegrastats 解析和最终汇总图表。

这个 maximal run 可能需要很久。SSH 容易断开时必须用 `tmux` 或 `nohup`。

如果还想补不同功耗模式，不要猜 `nvpmodel` 编号，先运行：

```bash
sudo nvpmodel -q
```

确认模式编号后再跑：

```bash
AGX_POWER_MODE_IDS="0 1" \
PYTHON_BIN=python3 \
bash scripts/run_agx_orin64_maximal_experiment.sh
```

这样会分别生成：

```text
results_agx_orin64_maximal_<time>_nvp0/
results_agx_orin64_maximal_<time>_nvp1/
```

## 6.2 抗压力测试命令

如果要补 RTSS 更看重的实时鲁棒性证据，跑 pressure sweep：

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_max

PYTHON_BIN=python3 \
PRESSURE_PROFILE=paper \
BASE_RESULT_ROOT=results_agx_orin64_pressure_paper_202605xx \
bash scripts/run_agx_orin64_pressure_sweep.sh
```

先 smoke：

```bash
PYTHON_BIN=python3 \
PRESSURE_PROFILE=smoke \
bash scripts/run_agx_orin64_pressure_sweep.sh
```

详细说明见：

```text
docs/AGX_ORIN64_PRESSURE_TEST_RUNBOOK.md
```

性价比最高的短压力测试可以直接跑：

```bash
PYTHON_BIN=python3 \
bash scripts/run_agx_orin64_pressure_short_value.sh
```

短版单独说明见：

```text
docs/AGX_ORIN64_PRESSURE_SHORT_VALUE_RUN.md
```

## 7. 长时间运行建议

推荐用 `tmux` 或 `screen` 防止 SSH 断开：

```bash
tmux new -s agx
# 在 tmux 里运行完整实验命令
```

如果没有 `tmux`，用 `nohup`：

```bash
nohup bash -lc 'PYTHON_BIN=python3 RESULT_ROOT=results_agx_orin64_comprehensive_202605xx AGX_DATASETS="av2 droid nuimages" DEADLINES_MS="33.333 50" MODELS="mobilenet_v2 resnet18" SEEDS="7 8 9" RUN_BACKEND_MATRIX=1 RUN_DERIVED_ANALYSES=1 bash scripts/run_agx_orin64_comprehensive_experiment.sh' \
  > agx_full_run.nohup.log 2>&1 &
```

## 8. 跑完后需要发回什么

跑完后打包结果目录：

```bash
cd ~/rtss_agx/agx_orin64_run_package_20260524_full
tar -czf agx_orin64_results_$(date +%Y%m%d_%H%M%S).tar.gz \
  results_agx_orin64_comprehensive_* \
  results_agx_orin64_env \
  jetson_env_check_first.log \
  agx_full_run.nohup.log 2>/dev/null || true
sha256sum agx_orin64_results_*.tar.gz
```

必须发回：

- `results_agx_orin64_comprehensive_*/env/`
- `results_agx_orin64_comprehensive_*/logs/`
- `results_agx_orin64_comprehensive_*/realframes/`
- `results_agx_orin64_comprehensive_*/tables/`
- `results_agx_orin64_comprehensive_*/figures/`
- `results_agx_orin64_comprehensive_*/AGX_ORIN64_COMPREHENSIVE_REPORT.md`
- 如果跑了 backend matrix，还要发回 `backend_torch_onnx_trt/`

## 9. 结果是否能用于论文的判断标准

可以用于论文的 AGX 结果必须满足：

- 环境日志显示设备是 Jetson AGX Orin 64GB 或明确的 AGX Orin 64GB module。
- `tegrastats` 日志覆盖主要实验。
- PyTorch CUDA 或 TensorRT/ONNXRuntime provider 证据明确。
- 真实帧数据三套场景至少跑出 AV2；高质量版本应包含 AV2、DROID、nuImages。
- 每个主表有 `CIRCA-RT` 和 `CIRCA-RT-Slack`。
- audit-bound validation 没有失败行。
- 不能出现 CPU fallback 被写成 TensorRT 的情况。

如果只跑 smoke，只能证明代码能在 AGX 上跑，不能作为主论文完整表格。完整投稿证据需要完整实验命令跑完。
