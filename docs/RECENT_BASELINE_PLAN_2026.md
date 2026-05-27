# Recent Baseline Plan For CIRCA-RT

Updated: 2026-05-19

This document defines the newer comparison baselines that should be used for the RTSS version of CIRCA-RT. The goal is not to reproduce every recent anomaly detector. The goal is to compare against recent, defensible, and deployment-compatible methods under the same online trace protocol, audit budget, and latency accounting.

## Recommended Claim Boundary

Do not claim:

> CIRCA-RT beats all SOTA anomaly detectors.

Use this claim instead:

> CIRCA-RT provides low-overhead, budget-aware online monitoring for coupled semantic-timing anomalies, and remains competitive with recent conformal, kernel-independence, and deep time-series detectors under matched audit budgets.

## Main Baselines To Keep In The Paper

These should be the main-table baselines because they are close to CIRCA-RT's threat model and can be run fairly on GPU/ROS2 traces.

| Baseline | Year | Role | Current status | Why it matters |
|---|---:|---|---|---|
| Context-aware conformal monitor | 2024-style | Strong online conformal competitor | Implemented as `ContextAwareConformal` | Main recall competitor on ROS2/DDS; handles context/mode shift better than simple thresholds. |
| Learnable Fourier / LF-HSIC-style independence test | 2024 | Recent independence-test competitor | Implemented as `LearnedFourierIndependence` approximation | Directly related to the NeurIPS 2024 Fourier independence-testing idea. |
| Conditional RFF-HSIC | 2024-style | Conditional dependence monitor | Implemented as `ConditionalRFF-HSIC` | Closest ablation to CIRCA-RT without selective audit/token bucket. |
| Online conformal anomaly monitor | 2022-2024 | Online statistical baseline | Implemented as `OnlineConformalAnomaly` | Clean, non-oracle, calibration-based baseline. |
| Random/periodic budget audit | timeless but necessary | Budget sanity baseline | Implemented | Shows whether CIRCA-RT uses audit budget intelligently. |
| Always audit | timeless but necessary | Cost upper bound | Implemented | Shows maximum recall and maximum latency overhead. |

## Newer Deep Time-Series Baselines To Add

These are useful because reviewers may ask why we do not compare with recent multivariate time-series anomaly detection methods. They should be reported with strict online/no-lookahead settings and explicit runtime overhead.

| Baseline | Venue/year | Priority | Suggested use |
|---|---:|---:|---|
| CATCH | ICLR 2025 | High | Newest strong frequency/channel-aware multivariate TSAD baseline. Use as the main "latest deep TSAD" comparison if implementation cost is acceptable. |
| DCdetector | KDD 2023 | High | Contrastive multivariate TSAD baseline. Good fit for semantic/timing feature streams. |
| TranAD | PVLDB 2022 | Medium-high | Canonical transformer TSAD baseline with public artifact; still widely recognized. |
| TimesNet | ICLR 2023 | Medium | General time-series model with anomaly-detection support; useful if we need a recognized deep sequence model. |
| Anomaly Transformer | ICLR 2022 | Medium/appendix | Strong but older transformer TSAD; use in appendix if table becomes too large. |
| MOMENT | ICML 2024 | Optional/stretch | Foundation time-series model; include only if runtime is acceptable and the protocol is clearly marked as heavier nearline detection. |
| SPIE-AD / unknown-unknown dependency detection | ICLR 2026 | Optional/stretch | Very relevant threat model because it targets changes in dependency structure without obvious marginal shift; include only if a faithful lightweight adapter is feasible. |

Recommended minimal upgrade:

1. Add `CATCH` and `DCdetector`.
2. Add `TranAD` if implementation time allows.
3. Keep `TimesNet`, `Anomaly Transformer`, `MOMENT`, and SPIE-AD-style unknown-unknown detection as appendix/stretch baselines.

## System Runtime Baselines

These are not anomaly detectors, but they strengthen the systems story.

| Baseline | Status | Use |
|---|---|---|
| PyTorch CUDA | Done on AutoDL | Framework baseline for GPU inference latency. |
| ONNXRuntime CUDA | Done on AutoDL | Optimized runtime baseline available on current AutoDL image. |
| TensorRT | Not valid on current AutoDL image | Must be run on Jetson/JetPack or a TensorRT-ready image. Current AutoDL image lacks `libnvinfer.so.10`. |
| Holoscan | Optional Jetson/IGX baseline | Use only if the environment supports it; otherwise mention as deployment motivation, not a measured detector baseline. |
| ROS2/DDS no-monitor trace | Done through ROS2 two-process experiments | Required latency baseline for middleware path. |

RT-Swap-style or DARIS-style real-time DNN scheduling papers can be cited as related systems work, but they are not direct anomaly-monitor baselines unless we implement their scheduling behavior. Do not put them in the main accuracy table unless the experimental behavior is reproduced.

## Baselines To Move To Appendix

These are still useful but should not be the center of the RTSS comparison because they are older or too generic.

| Baseline | Reason |
|---|---|
| SemanticThreshold | Simple sanity check. |
| TimingThreshold | Simple sanity check. |
| CUSUMTiming | Classical change detector. |
| EWMATiming | Classical smoothing detector. |
| OneClassSVMConcat | Generic non-online anomaly detection. |
| IsolationForestConcat | Generic non-online anomaly detection. |
| RandomForestConcat / MLPConcat | Only acceptable as oracle/upper-bound if trained with attack labels. |

## Fair Protocol For New Baselines

Every baseline must satisfy the same rules:

1. Use the same trace split: benign calibration/training, held-out test with nominal/interference/attack traces.
2. No future leakage: detection at time `t` may only use samples `<= t`.
3. Use the same feature stream: timing residuals, semantic residuals, context/mode features, and optional short history windows.
4. Report both default threshold and budget-normalized results at audit budgets `2%`, `5%`, `10%`, and `15%`.
5. Count detector overhead in the latency model, including window construction and model inference.
6. Report `recall`, `false_alarm_rate`, `audit_rate`, `detection_delay`, `p99`, `p999`, `deadline_miss_ratio`, and `mean_overhead_ms`.
7. If a method needs attack labels, mark it as `oracle` and keep it out of the main fair-comparison table.

## Implementation Priority

Phase A, low risk:

1. Keep current implemented baselines as the paper's statistical and budget-aware core.
2. Add budget-normalized tables to all main plots.
3. In the paper, explicitly state that `ContextAwareConformal` is the strongest current ROS2 recall competitor.

Phase B, stronger recent-baseline upgrade:

1. Implement a windowed deep-TSAD adapter that converts each trace into windows of `[timing_residual, semantic_residual, context/mode features]`.
2. Add CATCH and DCdetector first.
3. Add TranAD if training/inference time is stable on AutoDL.
4. Report these methods both with raw anomaly scores and with matched audit budgets.
5. Add one 2026 method only if its code is stable and the online/no-lookahead protocol is credible. The best 2026 candidate is SPIE-AD-style unknown-unknown dependency detection, not a generic large foundation model.

Phase C, edge validation:

1. Re-run the best subset on Jetson Orin: CIRCA-RT, ContextAwareConformal, ConditionalRFF-HSIC, CATCH/DCdetector if feasible, RandomBudgetAudit, AlwaysAudit.
2. Add TensorRT only if the environment has a valid TensorRT runtime.
3. Add power/thermal logs with `tegrastats`.

## Recommended Main-Table Set

Use this final main-table set if page space is tight:

| Category | Methods |
|---|---|
| Proposed | CIRCA-RT |
| Closest statistical competitors | ContextAwareConformal, ConditionalRFF-HSIC, LearnedFourierIndependence |
| Recent deep TSAD | CATCH, DCdetector, TranAD |
| Budget controls | RandomBudgetAudit, PeriodicAudit, AlwaysAudit |
| Simple appendix controls | TimingThreshold, SemanticThreshold, CUSUMTiming, EWMATiming |

This gives a much stronger comparison story than only using classical baselines, while keeping the experiment aligned with RTSS concerns: online behavior, overhead, tail latency, deadline misses, and audit budget.

## 2026-05-19 Implementation Status

Implemented in this repository:

- `CATCH` deployment adapter
- `DCdetector` deployment adapter
- `TranAD` deployment adapter

Main files:

- `src/circa_rt/deep_baselines.py`
- `scripts/run_recent_deep_baselines.py`
- `docs/RECENT_DEEP_BASELINES_IMPLEMENTATION.md`

Important writing boundary:

- Treat these as `CATCH-style`, `DCdetector-style`, and `TranAD-style` online adapters.
- Do not claim exact reproduction of the original external repositories unless we later import and validate the official code artifacts.
