# Phase 0-4 Status

更新时间：2026-05-16

本目录已经完成本机 Phase 0-4 的可运行实验工程。

## 完成内容

### Phase 0：接口固定

已完成：

- 固定 CSV schema：`configs/schema.json`
- 固定实验配置：`configs/experiment_config.json`
- 固定项目目录结构
- 固定主指标

核心字段包括：

- `semantic_residual`
- `timing_residual_ms`
- `mode`
- `context_*`
- `label`
- `baseline_latency_ms`
- `e2e_latency_ms`
- `deadline_miss`
- `alarm`
- `audit`

### Phase 1：核心 CIRCA-RT runner

已完成：

- `src/circa_rt/core.py`
- `scripts/run_circa_rt.py`

实现内容：

- context/mode residualization
- centered rolling RFF dependence statistic
- benign quantile calibration
- token-bucket selective audit
- latency and deadline miss recomputation

### Phase 2：Synthetic / semi-synthetic trace

已完成：

- `src/circa_rt/synthetic.py`
- `scripts/generate_synthetic_trace.py`

trace 类型：

- `nominal`
- `mode_shift`
- `stealthy_coupled_attack`
- `timing_only_attack`
- `semantic_only_attack`
- `mixed_attack`

### Phase 3：简单强基线

已完成：

- `SemanticThreshold`
- `TimingThreshold`
- `CUSUMTiming`
- `EWMATiming`
- `IsolationForestConcat`
- `OneClassSVMConcat`
- `RandomForestConcat`
- `MLPConcat`
- `AlwaysAudit`
- `PeriodicAudit`
- `RandomBudgetAudit`

注意：

- `RandomForestConcat` 和 `MLPConcat` 当前使用同 trace labels 训练，属于 supervised oracle / upper-bound style baseline。
- 主论文中不应把它们当作公平在线无监督 baseline。
- 可作为 appendix 或 oracle reference。

### Phase 4：近年统计 baseline

已完成 representative implementations：

- `RFF-HSIC`
- `LearnedFourierIndependence`
- `OnlineConformalAnomaly`
- `ContextAwareConformal`
- `ConditionalRFF-HSIC`

写作口径：

- 这些是可复现实验里的代表性实现。
- 不声称完整复现原论文系统。

## 如何运行

从项目根目录执行：

```powershell
python scripts\run_phase0_4_pipeline.py
```

单独生成 synthetic traces：

```powershell
python scripts\generate_synthetic_trace.py
```

单独跑 CIRCA-RT：

```powershell
python scripts\run_circa_rt.py --input traces\synthetic\stealthy_coupled_attack_seed0.csv --out-dir results\single_circa
```

单独跑 baselines：

```powershell
python scripts\run_baselines.py --input traces\synthetic\stealthy_coupled_attack_seed0.csv --out-dir results\single_baselines
```

## 当前输出

主输出：

- `traces/synthetic/*.csv`
- `results/summaries/summary_all.csv`
- `results/tables/main_table.csv`
- `results/tables/scenario_table.csv`
- `results/tables/ablation_table.csv`
- `results/figures/*.png`

## 当前结果解读

第一版结果的正确解读：

1. 在 `stealthy_coupled_attack` 场景中，CIRCA-RT 对 coupled semantic-timing anomaly 有明显优势。
2. 在 `mode_shift` 场景中，conditioning 有助于控制 benign context shift 下的误报。
3. 在 `timing_only_attack` 或 `semantic_only_attack` 中，单变量 threshold 可能更直接，这是 CIRCA-RT 的边界，不应回避。
4. `AlwaysAudit` recall 最高但 false alarm、audit cost 和 latency overhead 最大，用来证明 selective audit 的必要性。
5. `RandomForestConcat` / `MLPConcat` 是监督上界，不应作为主 claim 的公平 baseline。

## 已知限制

- 当前 synthetic trace 是机制验证，不是系统 trace。
- 当前 Phase 0-4 不包含 ROS2/DDS、AutoDL、Jetson。
- 当前 learned Fourier baseline 是 scale-search representative implementation，不是完整 NeurIPS 2024 训练目标。
- 当前 online conformal baseline 是 representative implementation，后续可换成更严格的 sequential conformal。
- MLP 有 convergence warning，不影响 Phase 0-4 主线；后续可增加 max_iter 或弱化该 baseline。

## 下一步建议

1. Phase 5：补 ROS2/DDS 或 Python periodic pipeline timing trace。
2. Phase 6：AutoDL 跑 server GPU inference latency。
3. Phase 7：线上 Jetson Orin 跑 edge latency 和 monitor/audit overhead。
4. 将 `RandomForestConcat` / `MLPConcat` 移入 oracle/appendix 表。
5. 增加 train/calibration/test split，避免监督 baseline 过强或数据泄漏争议。

## 2026-05-16 update

主实验现在采用 split protocol，结果目录变为 `results_split/`。

- calibration 使用 `nominal + mode_shift`
- test 覆盖 `nominal`, `mode_shift`, `stealthy_coupled_attack`, `timing_only_attack`, `semantic_only_attack`, `mixed_attack`
- `scenario_table.csv` 已补 `audit_rate`
- 新增 `main_table_ci.csv` 和 `scenario_table_ci.csv`
- 新增 `results_split/sensitivity/tables/sensitivity_table.csv`

当前更适合写进正文的结论：

- `CIRCA-RT` 在 `stealthy_coupled_attack` 上 recall 约 `0.892`，false alarm 约 `0.106`，audit rate 约 `0.036`
- `LearnedFourierIndependence` recall 更高，但 false alarm 约 `0.767`，不适合作为主文强 baseline
- `AlwaysAudit` 只适合做上界参考，不适合公平主 baseline
