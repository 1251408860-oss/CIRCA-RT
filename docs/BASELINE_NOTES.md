# Baseline Notes

更新时间：2026-05-16

## Baseline 分组

### 主文可用 baseline

建议主文优先使用：

- `SemanticThreshold`
- `TimingThreshold`
- `RFF-HSIC`
- `LearnedFourierIndependence`
- `OnlineConformalAnomaly`
- `ContextAwareConformal`
- `ConditionalRFF-HSIC`
- `AlwaysAudit`
- `RandomBudgetAudit`
- `CIRCA-RT`

原因：

- 它们不依赖同 trace 攻击标签训练。
- 它们更接近 deployment-time runtime monitoring。

### Appendix / oracle baseline

建议放 appendix 或 oracle table：

- `RandomForestConcat`
- `MLPConcat`

原因：

- 当前实现使用同 trace label 训练。
- 检测率会偏高。
- 可以作为 supervised upper-bound，不适合作为主要公平比较。

## 近年 baseline 对应关系

### NeurIPS 2024 learned Fourier independence test

本项目实现：

- `LearnedFourierIndependence`

实现口径：

- 使用 Fourier feature scale search 代表 learnable frequency adaptation。
- 不是完整复现 NeurIPS 2024 的 power-maximization objective。

### Online conformal anomaly detection

本项目实现：

- `OnlineConformalAnomaly`
- `ContextAwareConformal`

实现口径：

- 使用 benign calibration quantile。
- `ContextAwareConformal` 先 residualize context/mode，再做 conformal score。

### Conditional independence monitoring

本项目实现：

- `ConditionalRFF-HSIC`
- `CIRCA-RT`

区别：

- `ConditionalRFF-HSIC` 只告警/审计，不做 token-bucket selective audit。
- `CIRCA-RT` 增加 token-bucket selective audit 和 latency recomputation。

## 论文写法建议

推荐表述：

> For recent statistical monitors without a directly compatible runtime implementation, we implement representative variants under the same trace, window, and calibration protocol.

避免表述：

> We fully reproduce all recent SOTA methods.

## 当前公平性注意

所有主文 baseline 应共享：

- 相同 trace
- 相同 window size
- 相同 benign calibration protocol
- 相同 audit cost model
- 相同 deadline
监督模型如果使用攻击标签训练，必须标为 oracle。

## 2026-05-16 update

当前主实验已经切换到 split calibration/test protocol。

- `SemanticThreshold`、`TimingThreshold`、`RFF-HSIC`、`LearnedFourierIndependence`、`OnlineConformalAnomaly`、`ContextAwareConformal`、`ConditionalRFF-HSIC`、`AlwaysAudit`、`RandomBudgetAudit` 都有 split 版本
- `RandomForestConcat` / `MLPConcat` 仍只适合 oracle / appendix
- 主表建议优先引用 `results_split/tables/main_table.csv`
- 场景表建议优先引用 `results_split/tables/scenario_table.csv`
- 敏感性建议引用 `results_split/sensitivity/tables/sensitivity_table.csv`

当前更稳的主文写法是：

> Under a benign-only calibration protocol, CIRCA-RT detects coupled semantic-timing anomalies with controlled false alarm and low latency overhead.
