from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


EPS = 1e-12


def mode_one_hot(mode: np.ndarray) -> np.ndarray:
    values = sorted(str(v) for v in np.unique(mode))
    out = np.zeros((len(mode), len(values)), dtype=np.float64)
    m = mode.astype(str)
    for j, value in enumerate(values):
        out[:, j] = (m == value).astype(np.float64)
    return out


class ConditionEncoder:
    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.mode_values: list[str] = []

    def fit(self, context: np.ndarray, mode: np.ndarray) -> "ConditionEncoder":
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        self.scaler = StandardScaler().fit(context)
        self.mode_values = sorted(str(v) for v in np.unique(mode.astype(str)))
        return self

    def transform(self, context: np.ndarray, mode: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("ConditionEncoder is not fitted")
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        x = np.nan_to_num(self.scaler.transform(context), nan=0.0, posinf=0.0, neginf=0.0)
        m = mode.astype(str)
        mode_hot = np.zeros((len(mode), len(self.mode_values)), dtype=np.float64)
        for j, value in enumerate(self.mode_values):
            mode_hot[:, j] = (m == value).astype(np.float64)
        return np.concatenate([x, x**2, mode_hot, np.ones((len(mode), 1), dtype=np.float64)], axis=1)


class Residualizer:
    def __init__(self) -> None:
        self.encoder = ConditionEncoder()
        self.sem_model = Ridge(alpha=1.0)
        self.tim_model = Ridge(alpha=1.0)

    def fit(self, semantic: np.ndarray, timing: np.ndarray, context: np.ndarray, mode: np.ndarray) -> "Residualizer":
        self.encoder.fit(context, mode)
        z = self.encoder.transform(context, mode)
        self.sem_model.fit(z, semantic)
        self.tim_model.fit(z, timing)
        return self

    def transform(self, semantic: np.ndarray, timing: np.ndarray, context: np.ndarray, mode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.encoder.transform(context, mode)
        u = semantic - self.sem_model.predict(z)
        v = timing - self.tim_model.predict(z)
        return np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)


class RandomFourierFeatures:
    def __init__(self, feature_dim: int = 32, seed: int = 0) -> None:
        if feature_dim % 2 != 0:
            raise ValueError("feature_dim must be even")
        self.feature_dim = int(feature_dim)
        self.seed = int(seed)
        self.scale = np.ones(1, dtype=np.float64)
        self.omega: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "RandomFourierFeatures":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        rng = np.random.default_rng(self.seed)
        self.omega = rng.normal(size=(self.feature_dim // 2, x.shape[1])).astype(np.float64)
        std = np.std(x, axis=0)
        self.scale = 1.0 / np.maximum(std, 1e-3)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.omega is None:
            raise RuntimeError("RandomFourierFeatures is not fitted")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        proj = (x * self.scale.reshape(1, -1)) @ self.omega.T
        feats = np.concatenate([np.cos(proj), np.sin(proj)], axis=1)
        return math.sqrt(2.0 / self.feature_dim) * feats


def rolling_cross_scores(phi_x: np.ndarray, phi_y: np.ndarray, window_size: int) -> np.ndarray:
    """Centered rolling RFF dependence statistic.

    This approximates a windowed HSIC-style covariance norm. Centering is
    essential because raw RFF means otherwise dominate the statistic and make
    benign windows look spuriously dependent.
    """
    n, d = phi_x.shape
    scores = np.zeros(n, dtype=np.float64)
    cross = np.zeros((d, d), dtype=np.float64)
    sum_x = np.zeros(d, dtype=np.float64)
    sum_y = np.zeros(d, dtype=np.float64)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    w = max(int(window_size), 1)
    for i in range(n):
        x_i = phi_x[i]
        y_i = phi_y[i]
        xs.append(x_i)
        ys.append(y_i)
        cross += np.outer(x_i, y_i)
        sum_x += x_i
        sum_y += y_i
        if len(xs) > w:
            x_old = xs.pop(0)
            y_old = ys.pop(0)
            cross -= np.outer(x_old, y_old)
            sum_x -= x_old
            sum_y -= y_old
        m = float(len(xs))
        centered_cross = cross - np.outer(sum_x, sum_y) / max(m, EPS)
        scores[i] = float(np.linalg.norm(centered_cross, ord="fro") ** 2 / max(m * m, EPS))
    return scores


def rolling_cross_scores_by_group(phi_x: np.ndarray, phi_y: np.ndarray, group_ids: np.ndarray, window_size: int) -> np.ndarray:
    """Compute rolling cross scores independently within each contiguous group."""
    phi_x = np.asarray(phi_x, dtype=np.float64)
    phi_y = np.asarray(phi_y, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    if len(phi_x) != len(phi_y) or len(phi_x) != len(group_ids):
        raise ValueError("phi_x, phi_y, and group_ids must have the same length")
    if len(phi_x) == 0:
        return np.zeros(0, dtype=np.float64)
    scores = np.zeros(len(phi_x), dtype=np.float64)
    start = 0
    while start < len(group_ids):
        end = start + 1
        while end < len(group_ids) and group_ids[end] == group_ids[start]:
            end += 1
        scores[start:end] = rolling_cross_scores(phi_x[start:end], phi_y[start:end], window_size)
        start = end
    return scores


def rolling_corr_scores(x: np.ndarray, y: np.ndarray, window_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    scores = np.zeros_like(x)
    w = max(int(window_size), 2)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        xx = x[lo : i + 1]
        yy = y[lo : i + 1]
        if len(xx) < 3 or np.std(xx) < EPS or np.std(yy) < EPS:
            scores[i] = 0.0
        else:
            scores[i] = abs(float(np.corrcoef(xx, yy)[0, 1]))
    return scores
