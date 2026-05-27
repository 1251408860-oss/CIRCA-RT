from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .features import RandomFourierFeatures, Residualizer, rolling_corr_scores, rolling_cross_scores, rolling_cross_scores_by_group
from .schema import CONTEXT_COLUMNS, validate_trace


@dataclass(frozen=True)
class BaselineConfig:
    window_size: int = 32
    monitor_cost_ms: float = 0.05
    audit_cost_ms: float = 4.0
    periodic_audit_interval: int = 20
    random_audit_rate: float = 0.05
    seed: int = 7


def _apply_alarm(df: pd.DataFrame, method: str, score: np.ndarray, threshold: float, cfg: BaselineConfig, *, audit_mode: str = "on_alarm") -> pd.DataFrame:
    out = df.copy()
    alarm = (score >= threshold).astype(np.int64)
    if audit_mode == "always":
        audit = np.ones(len(out), dtype=np.int64)
        alarm = np.ones(len(out), dtype=np.int64)
    elif audit_mode == "periodic":
        audit = (np.arange(len(out)) % max(cfg.periodic_audit_interval, 1) == 0).astype(np.int64)
        alarm = audit.copy()
    elif audit_mode == "random":
        rng = np.random.default_rng(cfg.seed)
        audit = (rng.random(len(out)) < cfg.random_audit_rate).astype(np.int64)
        alarm = audit.copy()
    else:
        audit = alarm.copy()
    out["method"] = method
    out["score"] = np.asarray(score, dtype=float)
    out["alarm"] = alarm
    out["audit"] = audit
    out["monitor_cost_ms"] = cfg.monitor_cost_ms
    out["audit_cost_ms"] = audit.astype(float) * cfg.audit_cost_ms
    out["e2e_latency_ms"] = out["baseline_latency_ms"].to_numpy(dtype=float) + out["monitor_cost_ms"] + out["audit_cost_ms"]
    out["deadline_miss"] = (out["e2e_latency_ms"] > out["deadline_ms"]).astype(int)
    return out


def _benign_quantile(score: np.ndarray, label: np.ndarray, q: float = 0.99) -> float:
    benign = label == 0
    if np.any(benign):
        return float(np.quantile(score[benign], q))
    return float(np.quantile(score, q))


def semantic_threshold(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    label = df["label"].to_numpy(dtype=int)
    score = np.abs(df["semantic_residual"].to_numpy(dtype=float))
    threshold = _benign_quantile(score, label, 0.99)
    return _apply_alarm(df, "SemanticThreshold", score, threshold, cfg)


def timing_threshold(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    label = df["label"].to_numpy(dtype=int)
    score = np.abs(df["timing_residual_ms"].to_numpy(dtype=float))
    threshold = _benign_quantile(score, label, 0.99)
    return _apply_alarm(df, "TimingThreshold", score, threshold, cfg)


def ewma_timing(df: pd.DataFrame, cfg: BaselineConfig, alpha: float = 0.15) -> pd.DataFrame:
    x = np.abs(df["timing_residual_ms"].to_numpy(dtype=float))
    label = df["label"].to_numpy(dtype=int)
    score = np.zeros_like(x)
    state = x[0] if x.size else 0.0
    for i, value in enumerate(x):
        state = alpha * value + (1.0 - alpha) * state
        score[i] = state
    threshold = _benign_quantile(score, label, 0.99)
    return _apply_alarm(df, "EWMATiming", score, threshold, cfg)


def cusum_timing(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    x = np.abs(df["timing_residual_ms"].to_numpy(dtype=float))
    label = df["label"].to_numpy(dtype=int)
    benign = label == 0
    mu = float(np.mean(x[benign])) if np.any(benign) else float(np.mean(x))
    sigma = float(np.std(x[benign]) + 1e-6) if np.any(benign) else float(np.std(x) + 1e-6)
    score = np.zeros_like(x)
    s = 0.0
    for i, value in enumerate(x):
        s = max(0.0, s + (value - mu - 0.25 * sigma) / sigma)
        score[i] = s
    threshold = _benign_quantile(score, label, 0.99)
    return _apply_alarm(df, "CUSUMTiming", score, threshold, cfg)


def _concat_features(df: pd.DataFrame) -> np.ndarray:
    cols = ["semantic_residual", "timing_residual_ms"] + CONTEXT_COLUMNS
    return df[cols].to_numpy(dtype=np.float64)


def isolation_forest(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    x = _concat_features(df)
    label = df["label"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(x[label == 0] if np.any(label == 0) else x)
    xs = scaler.transform(x)
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=cfg.seed)
    model.fit(xs[label == 0] if np.any(label == 0) else xs)
    score = -model.score_samples(xs)
    threshold = _benign_quantile(score, label, 0.99)
    return _apply_alarm(df, "IsolationForestConcat", score, threshold, cfg)


def one_class_svm(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    x = _concat_features(df)
    label = df["label"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(x[label == 0] if np.any(label == 0) else x)
    xs = scaler.transform(x)
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    model.fit(xs[label == 0] if np.any(label == 0) else xs)
    score = -model.score_samples(xs)
    threshold = _benign_quantile(score, label, 0.99)
    return _apply_alarm(df, "OneClassSVMConcat", score, threshold, cfg)


def random_forest_concat(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    x = _concat_features(df)
    label = df["label"].to_numpy(dtype=int)
    if len(np.unique(label)) < 2:
        score = np.zeros(len(df), dtype=float)
    else:
        model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=cfg.seed, class_weight="balanced")
        model.fit(x, label)
        score = model.predict_proba(x)[:, 1]
    threshold = 0.5
    return _apply_alarm(df, "RandomForestConcat", score, threshold, cfg)


def mlp_concat(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    x = _concat_features(df)
    label = df["label"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    if len(np.unique(label)) < 2:
        score = np.zeros(len(df), dtype=float)
    else:
        model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=300, random_state=cfg.seed)
        model.fit(xs, label)
        score = model.predict_proba(xs)[:, 1]
    threshold = 0.5
    return _apply_alarm(df, "MLPConcat", score, threshold, cfg)


def always_audit(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    score = np.ones(len(df), dtype=float)
    return _apply_alarm(df, "AlwaysAudit", score, 0.0, cfg, audit_mode="always")


def periodic_audit(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    score = np.ones(len(df), dtype=float)
    return _apply_alarm(df, "PeriodicAudit", score, 0.0, cfg, audit_mode="periodic")


def random_budget_audit(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    score = np.ones(len(df), dtype=float)
    return _apply_alarm(df, "RandomBudgetAudit", score, 0.0, cfg, audit_mode="random")


def rff_hsic(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    label = df["label"].to_numpy(dtype=int)
    sem = df["semantic_residual"].to_numpy(dtype=float)
    tim = df["timing_residual_ms"].to_numpy(dtype=float)
    benign = label == 0
    rff_s = RandomFourierFeatures(32, seed=cfg.seed).fit(sem[benign] if np.any(benign) else sem)
    rff_t = RandomFourierFeatures(32, seed=cfg.seed + 1).fit(tim[benign] if np.any(benign) else tim)
    score = rolling_cross_scores(rff_s.transform(sem), rff_t.transform(tim), cfg.window_size)
    threshold = _benign_quantile(score, label, 0.99)
    return _apply_alarm(df, "RFF-HSIC", score, threshold, cfg)


def learned_fourier_independence(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    label = df["label"].to_numpy(dtype=int)
    sem = df["semantic_residual"].to_numpy(dtype=float)
    tim = df["timing_residual_ms"].to_numpy(dtype=float)
    benign = label == 0
    best_score = None
    best_gap = -np.inf
    for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
        s = sem * scale
        t = tim * scale
        rff_s = RandomFourierFeatures(32, seed=cfg.seed).fit(s[benign] if np.any(benign) else s)
        rff_t = RandomFourierFeatures(32, seed=cfg.seed + 1).fit(t[benign] if np.any(benign) else t)
        score = rolling_cross_scores(rff_s.transform(s), rff_t.transform(t), cfg.window_size)
        if np.any(label == 1) and np.any(label == 0):
            gap = float(np.mean(score[label == 1]) - np.mean(score[label == 0]))
        else:
            gap = float(np.std(score))
        if gap > best_gap:
            best_gap = gap
            best_score = score
    assert best_score is not None
    threshold = _benign_quantile(best_score, label, 0.99)
    return _apply_alarm(df, "LearnedFourierIndependence", best_score, threshold, cfg)


def online_conformal(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    label = df["label"].to_numpy(dtype=int)
    x = _concat_features(df)
    benign = label == 0
    scaler = StandardScaler().fit(x[benign] if np.any(benign) else x)
    xs = scaler.transform(x)
    center = np.mean(xs[benign], axis=0) if np.any(benign) else np.mean(xs, axis=0)
    score = np.linalg.norm(xs - center.reshape(1, -1), axis=1)
    threshold = _benign_quantile(score, label, 0.95)
    return _apply_alarm(df, "OnlineConformalAnomaly", score, threshold, cfg)


def context_aware_conformal(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    label = df["label"].to_numpy(dtype=int)
    context = df[CONTEXT_COLUMNS].to_numpy(dtype=float)
    mode = df["mode"].astype(str).to_numpy()
    sem = df["semantic_residual"].to_numpy(dtype=float)
    tim = df["timing_residual_ms"].to_numpy(dtype=float)
    benign = label == 0
    residualizer = Residualizer().fit(sem[benign], tim[benign], context[benign], mode[benign]) if np.any(benign) else Residualizer().fit(sem, tim, context, mode)
    u, v = residualizer.transform(sem, tim, context, mode)
    score = np.sqrt(u * u + v * v)
    threshold = _benign_quantile(score, label, 0.95)
    return _apply_alarm(df, "ContextAwareConformal", score, threshold, cfg)


def conditional_rff_hsic_no_audit(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    label = df["label"].to_numpy(dtype=int)
    context = df[CONTEXT_COLUMNS].to_numpy(dtype=float)
    mode = df["mode"].astype(str).to_numpy()
    sem = df["semantic_residual"].to_numpy(dtype=float)
    tim = df["timing_residual_ms"].to_numpy(dtype=float)
    benign = label == 0
    residualizer = Residualizer().fit(sem[benign], tim[benign], context[benign], mode[benign]) if np.any(benign) else Residualizer().fit(sem, tim, context, mode)
    u, v = residualizer.transform(sem, tim, context, mode)
    rff_u = RandomFourierFeatures(32, seed=cfg.seed).fit(u[benign] if np.any(benign) else u)
    rff_v = RandomFourierFeatures(32, seed=cfg.seed + 1).fit(v[benign] if np.any(benign) else v)
    score = rolling_cross_scores(rff_u.transform(u), rff_v.transform(v), cfg.window_size)
    threshold = _benign_quantile(score, label, 0.99)
    return _apply_alarm(df, "ConditionalRFF-HSIC", score, threshold, cfg)


def _benign_calibration_mask(df: pd.DataFrame) -> np.ndarray:
    label = df["label"].to_numpy(dtype=int)
    benign = label == 0
    if not np.any(benign):
        raise ValueError("split baselines require benign calibration data")
    return benign


def _score_by_group(phi_x: np.ndarray, phi_y: np.ndarray, df: pd.DataFrame, window_size: int) -> np.ndarray:
    return rolling_cross_scores_by_group(phi_x, phi_y, df["run_id"].astype(str).to_numpy(), window_size)


def _ewma_stream(x: np.ndarray, alpha: float, initial: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    score = np.zeros_like(x)
    if x.size == 0:
        return score
    state = float(x[0] if initial is None else initial)
    for i, value in enumerate(x):
        state = alpha * float(value) + (1.0 - alpha) * state
        score[i] = state
    return score


def _cusum_stream(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    score = np.zeros_like(x)
    s = 0.0
    sigma = max(float(sigma), 1e-6)
    for i, value in enumerate(x):
        s = max(0.0, s + (float(value) - mu - 0.25 * sigma) / sigma)
        score[i] = s
    return score


def semantic_threshold_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    benign = _benign_calibration_mask(calibration_df)
    cal_score = np.abs(calibration_df["semantic_residual"].to_numpy(dtype=float))
    threshold = float(np.quantile(cal_score[benign], 0.99))
    score = np.abs(test_df["semantic_residual"].to_numpy(dtype=float))
    return _apply_alarm(test_df, "SemanticThreshold", score, threshold, cfg)


def timing_threshold_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    benign = _benign_calibration_mask(calibration_df)
    cal_score = np.abs(calibration_df["timing_residual_ms"].to_numpy(dtype=float))
    threshold = float(np.quantile(cal_score[benign], 0.99))
    score = np.abs(test_df["timing_residual_ms"].to_numpy(dtype=float))
    return _apply_alarm(test_df, "TimingThreshold", score, threshold, cfg)


def ewma_timing_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig, alpha: float = 0.15) -> pd.DataFrame:
    cal_x = np.abs(calibration_df["timing_residual_ms"].to_numpy(dtype=float))
    benign = _benign_calibration_mask(calibration_df)
    cal_score = _ewma_stream(cal_x, alpha, initial=float(np.mean(cal_x[benign])))
    threshold = float(np.quantile(cal_score[benign], 0.99))
    test_x = np.abs(test_df["timing_residual_ms"].to_numpy(dtype=float))
    score = _ewma_stream(test_x, alpha, initial=float(np.mean(cal_x[benign])))
    return _apply_alarm(test_df, "EWMATiming", score, threshold, cfg)


def cusum_timing_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    cal_x = np.abs(calibration_df["timing_residual_ms"].to_numpy(dtype=float))
    benign = _benign_calibration_mask(calibration_df)
    mu = float(np.mean(cal_x[benign]))
    sigma = float(np.std(cal_x[benign]) + 1e-6)
    cal_score = _cusum_stream(cal_x, mu, sigma)
    threshold = float(np.quantile(cal_score[benign], 0.99))
    test_x = np.abs(test_df["timing_residual_ms"].to_numpy(dtype=float))
    score = _cusum_stream(test_x, mu, sigma)
    return _apply_alarm(test_df, "CUSUMTiming", score, threshold, cfg)


def isolation_forest_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    cal_x = _concat_features(calibration_df)
    test_x = _concat_features(test_df)
    benign = _benign_calibration_mask(calibration_df)
    scaler = StandardScaler().fit(cal_x[benign])
    cal_xs = scaler.transform(cal_x)
    test_xs = scaler.transform(test_x)
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=cfg.seed)
    model.fit(cal_xs[benign])
    cal_score = -model.score_samples(cal_xs)
    threshold = float(np.quantile(cal_score[benign], 0.99))
    score = -model.score_samples(test_xs)
    return _apply_alarm(test_df, "IsolationForestConcat", score, threshold, cfg)


def one_class_svm_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    cal_x = _concat_features(calibration_df)
    test_x = _concat_features(test_df)
    benign = _benign_calibration_mask(calibration_df)
    scaler = StandardScaler().fit(cal_x[benign])
    cal_xs = scaler.transform(cal_x)
    test_xs = scaler.transform(test_x)
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    model.fit(cal_xs[benign])
    cal_score = -model.score_samples(cal_xs)
    threshold = float(np.quantile(cal_score[benign], 0.99))
    score = -model.score_samples(test_xs)
    return _apply_alarm(test_df, "OneClassSVMConcat", score, threshold, cfg)


def rff_hsic_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    benign = _benign_calibration_mask(calibration_df)
    cal_sem = calibration_df["semantic_residual"].to_numpy(dtype=float)
    cal_tim = calibration_df["timing_residual_ms"].to_numpy(dtype=float)
    test_sem = test_df["semantic_residual"].to_numpy(dtype=float)
    test_tim = test_df["timing_residual_ms"].to_numpy(dtype=float)
    rff_s = RandomFourierFeatures(32, seed=cfg.seed).fit(cal_sem[benign])
    rff_t = RandomFourierFeatures(32, seed=cfg.seed + 1).fit(cal_tim[benign])
    cal_score = _score_by_group(rff_s.transform(cal_sem), rff_t.transform(cal_tim), calibration_df, cfg.window_size)
    threshold = float(np.quantile(cal_score[benign], 0.99))
    score = _score_by_group(rff_s.transform(test_sem), rff_t.transform(test_tim), test_df, cfg.window_size)
    return _apply_alarm(test_df, "RFF-HSIC", score, threshold, cfg)


def learned_fourier_independence_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    benign = _benign_calibration_mask(calibration_df)
    cal_sem = calibration_df["semantic_residual"].to_numpy(dtype=float)
    cal_tim = calibration_df["timing_residual_ms"].to_numpy(dtype=float)
    test_sem = test_df["semantic_residual"].to_numpy(dtype=float)
    test_tim = test_df["timing_residual_ms"].to_numpy(dtype=float)
    best = None
    best_metric = -np.inf
    for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
        s_cal = cal_sem * scale
        t_cal = cal_tim * scale
        rff_s = RandomFourierFeatures(32, seed=cfg.seed).fit(s_cal[benign])
        rff_t = RandomFourierFeatures(32, seed=cfg.seed + 1).fit(t_cal[benign])
        cal_score = _score_by_group(rff_s.transform(s_cal), rff_t.transform(t_cal), calibration_df, cfg.window_size)
        metric = float(np.std(cal_score[benign]))
        if metric > best_metric:
            best_metric = metric
            best = (scale, rff_s, rff_t, float(np.quantile(cal_score[benign], 0.99)))
    assert best is not None
    _, rff_s, rff_t, threshold = best
    score = _score_by_group(rff_s.transform(test_sem), rff_t.transform(test_tim), test_df, cfg.window_size)
    return _apply_alarm(test_df, "LearnedFourierIndependence", score, threshold, cfg)


def online_conformal_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    benign = _benign_calibration_mask(calibration_df)
    cal_x = _concat_features(calibration_df)
    test_x = _concat_features(test_df)
    scaler = StandardScaler().fit(cal_x[benign])
    cal_xs = scaler.transform(cal_x)
    test_xs = scaler.transform(test_x)
    center = np.mean(cal_xs[benign], axis=0)
    cal_score = np.linalg.norm(cal_xs - center.reshape(1, -1), axis=1)
    threshold = float(np.quantile(cal_score[benign], 0.95))
    score = np.linalg.norm(test_xs - center.reshape(1, -1), axis=1)
    return _apply_alarm(test_df, "OnlineConformalAnomaly", score, threshold, cfg)


def context_aware_conformal_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    benign = _benign_calibration_mask(calibration_df)
    cal_context = calibration_df[CONTEXT_COLUMNS].to_numpy(dtype=float)
    cal_mode = calibration_df["mode"].astype(str).to_numpy()
    cal_sem = calibration_df["semantic_residual"].to_numpy(dtype=float)
    cal_tim = calibration_df["timing_residual_ms"].to_numpy(dtype=float)
    test_context = test_df[CONTEXT_COLUMNS].to_numpy(dtype=float)
    test_mode = test_df["mode"].astype(str).to_numpy()
    test_sem = test_df["semantic_residual"].to_numpy(dtype=float)
    test_tim = test_df["timing_residual_ms"].to_numpy(dtype=float)
    residualizer = Residualizer().fit(cal_sem[benign], cal_tim[benign], cal_context[benign], cal_mode[benign])
    cal_u, cal_v = residualizer.transform(cal_sem, cal_tim, cal_context, cal_mode)
    test_u, test_v = residualizer.transform(test_sem, test_tim, test_context, test_mode)
    cal_score = np.sqrt(cal_u * cal_u + cal_v * cal_v)
    threshold = float(np.quantile(cal_score[benign], 0.95))
    score = np.sqrt(test_u * test_u + test_v * test_v)
    return _apply_alarm(test_df, "ContextAwareConformal", score, threshold, cfg)


def conditional_rff_hsic_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    benign = _benign_calibration_mask(calibration_df)
    cal_context = calibration_df[CONTEXT_COLUMNS].to_numpy(dtype=float)
    cal_mode = calibration_df["mode"].astype(str).to_numpy()
    cal_sem = calibration_df["semantic_residual"].to_numpy(dtype=float)
    cal_tim = calibration_df["timing_residual_ms"].to_numpy(dtype=float)
    test_context = test_df[CONTEXT_COLUMNS].to_numpy(dtype=float)
    test_mode = test_df["mode"].astype(str).to_numpy()
    test_sem = test_df["semantic_residual"].to_numpy(dtype=float)
    test_tim = test_df["timing_residual_ms"].to_numpy(dtype=float)
    residualizer = Residualizer().fit(cal_sem[benign], cal_tim[benign], cal_context[benign], cal_mode[benign])
    cal_u, cal_v = residualizer.transform(cal_sem, cal_tim, cal_context, cal_mode)
    test_u, test_v = residualizer.transform(test_sem, test_tim, test_context, test_mode)
    rff_u = RandomFourierFeatures(32, seed=cfg.seed).fit(cal_u[benign])
    rff_v = RandomFourierFeatures(32, seed=cfg.seed + 1).fit(cal_v[benign])
    cal_score = _score_by_group(rff_u.transform(cal_u), rff_v.transform(cal_v), calibration_df, cfg.window_size)
    threshold = float(np.quantile(cal_score[benign], 0.99))
    score = _score_by_group(rff_u.transform(test_u), rff_v.transform(test_v), test_df, cfg.window_size)
    return _apply_alarm(test_df, "ConditionalRFF-HSIC", score, threshold, cfg)


def always_audit_split(_: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    score = np.ones(len(test_df), dtype=float)
    return _apply_alarm(test_df, "AlwaysAudit", score, 0.0, cfg, audit_mode="always")


def periodic_audit_split(_: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    score = np.ones(len(test_df), dtype=float)
    return _apply_alarm(test_df, "PeriodicAudit", score, 0.0, cfg, audit_mode="periodic")


def random_budget_audit_split(_: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    score = np.ones(len(test_df), dtype=float)
    return _apply_alarm(test_df, "RandomBudgetAudit", score, 0.0, cfg, audit_mode="random")


def _deep_cfg_from_baseline(cfg: BaselineConfig):
    from .deep_baselines import DeepBaselineConfig

    return DeepBaselineConfig(
        window_size=cfg.window_size,
        audit_cost_ms=cfg.audit_cost_ms,
        seed=cfg.seed,
    )


def catch_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    from .deep_baselines import run_recent_deep_baseline_split

    return run_recent_deep_baseline_split(calibration_df, test_df, "CATCH", _deep_cfg_from_baseline(cfg))


def dcdetector_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    from .deep_baselines import run_recent_deep_baseline_split

    return run_recent_deep_baseline_split(calibration_df, test_df, "DCdetector", _deep_cfg_from_baseline(cfg))


def tranad_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    from .deep_baselines import run_recent_deep_baseline_split

    return run_recent_deep_baseline_split(calibration_df, test_df, "TranAD", _deep_cfg_from_baseline(cfg))


def moment_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    from .deep_baselines import run_recent_deep_baseline_split

    return run_recent_deep_baseline_split(calibration_df, test_df, "MOMENT", _deep_cfg_from_baseline(cfg))


def timemixer_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    from .deep_baselines import run_recent_deep_baseline_split

    return run_recent_deep_baseline_split(calibration_df, test_df, "TimeMixer", _deep_cfg_from_baseline(cfg))


def moderntcn_split(calibration_df: pd.DataFrame, test_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    from .deep_baselines import run_recent_deep_baseline_split

    return run_recent_deep_baseline_split(calibration_df, test_df, "ModernTCN", _deep_cfg_from_baseline(cfg))


RECENT_DEEP_BASELINE_FUNCS = {
    "CATCH": catch_split,
    "DCdetector": dcdetector_split,
    "TranAD": tranad_split,
    "MOMENT": moment_split,
    "TimeMixer": timemixer_split,
    "ModernTCN": moderntcn_split,
}


SPLIT_BASELINE_FUNCS = {
    "SemanticThreshold": semantic_threshold_split,
    "TimingThreshold": timing_threshold_split,
    "EWMATiming": ewma_timing_split,
    "CUSUMTiming": cusum_timing_split,
    "IsolationForestConcat": isolation_forest_split,
    "OneClassSVMConcat": one_class_svm_split,
    "AlwaysAudit": always_audit_split,
    "PeriodicAudit": periodic_audit_split,
    "RandomBudgetAudit": random_budget_audit_split,
    "RFF-HSIC": rff_hsic_split,
    "LearnedFourierIndependence": learned_fourier_independence_split,
    "OnlineConformalAnomaly": online_conformal_split,
    "ContextAwareConformal": context_aware_conformal_split,
    "ConditionalRFF-HSIC": conditional_rff_hsic_split,
    **RECENT_DEEP_BASELINE_FUNCS,
}


BASELINE_FUNCS = {
    "SemanticThreshold": semantic_threshold,
    "TimingThreshold": timing_threshold,
    "EWMATiming": ewma_timing,
    "CUSUMTiming": cusum_timing,
    "IsolationForestConcat": isolation_forest,
    "OneClassSVMConcat": one_class_svm,
    "RandomForestConcat": random_forest_concat,
    "MLPConcat": mlp_concat,
    "AlwaysAudit": always_audit,
    "PeriodicAudit": periodic_audit,
    "RandomBudgetAudit": random_budget_audit,
    "RFF-HSIC": rff_hsic,
    "LearnedFourierIndependence": learned_fourier_independence,
    "OnlineConformalAnomaly": online_conformal,
    "ContextAwareConformal": context_aware_conformal,
    "ConditionalRFF-HSIC": conditional_rff_hsic_no_audit,
}
