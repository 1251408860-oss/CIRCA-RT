from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .schema import CONTEXT_COLUMNS, validate_trace


DEEP_BASELINE_METHODS = ("CATCH", "DCdetector", "TranAD", "MOMENT", "TimeMixer", "ModernTCN")


@dataclass(frozen=True)
class DeepBaselineConfig:
    window_size: int = 32
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    hidden_dim: int = 64
    latent_dim: int = 16
    threshold_quantile: float = 0.99
    monitor_cost_floor_ms: float = 0.12
    audit_cost_ms: float = 4.0
    seed: int = 7
    device: str = "auto"
    backend: str = "auto"
    max_train_windows: int = 4096


@dataclass
class DeepBaselineModel:
    method: str
    cfg: DeepBaselineConfig
    scaler: StandardScaler
    threshold: float
    backend: str
    payload: dict[str, Any]
    monitor_cost_ms: float
    train_seconds: float


def feature_columns() -> list[str]:
    return ["semantic_residual", "timing_residual_ms"] + CONTEXT_COLUMNS


def _normalize_method(method: str) -> str:
    aliases = {
        "catch": "CATCH",
        "dc": "DCdetector",
        "dcdetector": "DCdetector",
        "tranad": "TranAD",
        "moment": "MOMENT",
        "timemixer": "TimeMixer",
        "time_mixer": "TimeMixer",
        "moderntcn": "ModernTCN",
        "modern_tcn": "ModernTCN",
    }
    key = method.replace("-", "").replace("_", "").lower()
    normalized = aliases.get(key, method)
    if normalized not in DEEP_BASELINE_METHODS:
        raise ValueError(f"unknown deep baseline {method!r}; choices={DEEP_BASELINE_METHODS}")
    return normalized


def _torch_module():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        return None, None, None
    return torch, nn, F


def _resolve_device(requested: str, backend: str):
    if backend == "sklearn":
        return None
    torch, _, _ = _torch_module()
    if torch is None:
        if backend == "torch":
            raise RuntimeError("requested torch backend for deep baselines, but PyTorch is not installed")
        return None
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if backend == "torch":
            return torch.device("cpu")
        return None
    if requested == "cpu" and backend != "torch":
        return None
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA for deep baselines, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch, _, _ = _torch_module()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _features(df: pd.DataFrame, scaler: StandardScaler | None = None, *, fit: bool = False) -> tuple[np.ndarray, StandardScaler]:
    x = df[feature_columns()].to_numpy(dtype=np.float32)
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        x = scaler.fit_transform(x).astype(np.float32)
    else:
        x = scaler.transform(x).astype(np.float32)
    return x, scaler


def _windows_by_group(x: np.ndarray, groups: np.ndarray, window_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    groups = np.asarray(groups)
    n, channels = x.shape
    windows = np.empty((n, window_size, channels), dtype=np.float32)
    for group in pd.unique(groups):
        idx = np.where(groups == group)[0]
        gx = x[idx]
        for local_pos, global_pos in enumerate(idx):
            start = max(0, local_pos - window_size + 1)
            chunk = gx[start : local_pos + 1]
            if len(chunk) < window_size:
                pad = np.repeat(chunk[:1], window_size - len(chunk), axis=0)
                chunk = np.concatenate([pad, chunk], axis=0)
            windows[global_pos] = chunk
    return windows


def _make_windows(df: pd.DataFrame, scaler: StandardScaler, window_size: int) -> np.ndarray:
    x, _ = _features(df, scaler, fit=False)
    return _windows_by_group(x, df["run_id"].astype(str).to_numpy(), window_size)


def _fit_scaler(calibration_df: pd.DataFrame) -> StandardScaler:
    label = calibration_df["label"].to_numpy(dtype=int)
    benign = label == 0
    if not np.any(benign):
        raise ValueError("deep baselines require benign calibration samples")
    scaler = StandardScaler().fit(calibration_df.loc[benign, feature_columns()].to_numpy(dtype=np.float32))
    return scaler


def _sample_train_windows(windows: np.ndarray, mask: np.ndarray, cfg: DeepBaselineConfig) -> np.ndarray:
    train = windows[mask]
    if len(train) == 0:
        raise ValueError("no benign windows available for deep baseline training")
    max_n = int(max(cfg.batch_size, cfg.max_train_windows))
    if len(train) > max_n:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(train), size=max_n, replace=False)
        train = train[np.sort(idx)]
    return train.astype(np.float32)


def _numpy_freq_features(windows: np.ndarray) -> np.ndarray:
    spec = np.fft.rfft(windows, axis=1)
    mag = np.log1p(np.abs(spec))
    return mag.reshape(len(windows), -1).astype(np.float32)


def _numpy_time_features(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(len(windows), -1).astype(np.float32)


def _numpy_patch_features(windows: np.ndarray, patch_size: int = 8) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    n, steps, channels = windows.shape
    patch_size = max(1, min(int(patch_size), steps))
    pad = (-steps) % patch_size
    if pad:
        tail = np.repeat(windows[:, -1:, :], pad, axis=1)
        windows = np.concatenate([windows, tail], axis=1)
    patches = windows.reshape(n, -1, patch_size, channels)
    mean = patches.mean(axis=2)
    std = patches.std(axis=2)
    last = patches[:, :, -1, :]
    slope = last - patches[:, :, 0, :]
    return np.concatenate([mean, std, slope], axis=2).reshape(n, -1).astype(np.float32)


def _numpy_multiscale_features(windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    feats = [_numpy_time_features(windows), _numpy_freq_features(windows)]
    for width in [2, 4, 8]:
        if windows.shape[1] < width:
            continue
        kernel = np.ones(width, dtype=np.float32) / float(width)
        smooth_channels = []
        for ch in range(windows.shape[2]):
            smooth = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, windows[:, :, ch])
            smooth_channels.append(smooth[:, :, None])
        smooth_x = np.concatenate(smooth_channels, axis=2)
        residual = windows - smooth_x
        feats.append(smooth_x.reshape(len(windows), -1).astype(np.float32))
        feats.append(residual.reshape(len(windows), -1).astype(np.float32))
    return np.concatenate(feats, axis=1).astype(np.float32)


def _numpy_temporal_conv_features(windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    diff1 = np.diff(windows, axis=1, prepend=windows[:, :1, :])
    diff2 = np.diff(diff1, axis=1, prepend=diff1[:, :1, :])
    stats = []
    for x in [windows, diff1, diff2]:
        stats.extend(
            [
                x.mean(axis=1),
                x.std(axis=1),
                x.max(axis=1),
                x.min(axis=1),
                x[:, -1, :],
            ]
        )
    return np.concatenate(stats, axis=1).astype(np.float32)


def _pca_reconstruction_payload(kind: str, x: np.ndarray, cfg: DeepBaselineConfig, latent: int) -> dict[str, Any]:
    components = min(latent, x.shape[1], len(x) - 1)
    model = PCA(n_components=max(1, components), random_state=cfg.seed).fit(x)
    return {"kind": kind, "model": model}


def _safe_quantile(scores: np.ndarray, mask: np.ndarray, q: float) -> float:
    base = scores[mask] if np.any(mask) else scores
    if len(base) == 0:
        return 0.0
    return float(np.quantile(base, q))


def _measured_score(model: DeepBaselineModel, windows: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    scores = _score_windows(model, windows)
    torch, _, _ = _torch_module()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    elapsed = time.perf_counter() - start
    per_sample_ms = (elapsed * 1000.0 / max(len(windows), 1)) if len(windows) else model.cfg.monitor_cost_floor_ms
    return scores, max(float(model.cfg.monitor_cost_floor_ms), float(per_sample_ms))


def _apply_deep_alarm(test_df: pd.DataFrame, model: DeepBaselineModel, score: np.ndarray) -> pd.DataFrame:
    out = test_df.copy()
    alarm = (score >= model.threshold).astype(np.int64)
    audit = alarm.copy()
    out["method"] = model.method
    out["score"] = np.asarray(score, dtype=float)
    out["alarm"] = alarm
    out["audit"] = audit
    out["monitor_cost_ms"] = float(model.monitor_cost_ms)
    out["audit_cost_ms"] = audit.astype(float) * float(model.cfg.audit_cost_ms)
    out["e2e_latency_ms"] = out["baseline_latency_ms"].to_numpy(dtype=float) + out["monitor_cost_ms"] + out["audit_cost_ms"]
    out["deadline_miss"] = (out["e2e_latency_ms"] > out["deadline_ms"]).astype(int)
    out["deep_backend"] = model.backend
    out["deep_threshold"] = float(model.threshold)
    out["deep_train_seconds"] = float(model.train_seconds)
    return out


def fit_recent_deep_baseline(calibration_df: pd.DataFrame, method: str, cfg: DeepBaselineConfig) -> DeepBaselineModel:
    validate_trace(calibration_df)
    method = _normalize_method(method)
    _seed_everything(cfg.seed)
    scaler = _fit_scaler(calibration_df)
    windows = _make_windows(calibration_df, scaler, cfg.window_size)
    label = calibration_df["label"].to_numpy(dtype=int)
    benign = label == 0
    train_windows = _sample_train_windows(windows, benign, cfg)

    start = time.perf_counter()
    device = _resolve_device(cfg.device, cfg.backend)
    if device is not None:
        payload = _fit_torch_model(method, train_windows, cfg, device)
        backend = f"torch:{device.type}"
    else:
        payload = _fit_sklearn_model(method, train_windows, cfg)
        backend = "sklearn"
    train_seconds = time.perf_counter() - start

    provisional = DeepBaselineModel(
        method=method,
        cfg=cfg,
        scaler=scaler,
        threshold=0.0,
        backend=backend,
        payload=payload,
        monitor_cost_ms=cfg.monitor_cost_floor_ms,
        train_seconds=train_seconds,
    )
    cal_scores, monitor_cost_ms = _measured_score(provisional, windows)
    threshold = _safe_quantile(cal_scores, benign, cfg.threshold_quantile)
    provisional.threshold = threshold
    provisional.monitor_cost_ms = monitor_cost_ms
    return provisional


def apply_recent_deep_baseline(model: DeepBaselineModel, test_df: pd.DataFrame) -> pd.DataFrame:
    validate_trace(test_df)
    windows = _make_windows(test_df, model.scaler, model.cfg.window_size)
    score = _score_windows(model, windows)
    return _apply_deep_alarm(test_df, model, score)


def run_recent_deep_baseline_split(
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    cfg: DeepBaselineConfig,
) -> pd.DataFrame:
    model = fit_recent_deep_baseline(calibration_df, method, cfg)
    return apply_recent_deep_baseline(model, test_df)


def _fit_sklearn_model(method: str, train_windows: np.ndarray, cfg: DeepBaselineConfig) -> dict[str, Any]:
    latent = max(2, min(int(cfg.latent_dim), len(train_windows) - 1))
    if method == "CATCH":
        return _pca_reconstruction_payload("pca_freq", _numpy_freq_features(train_windows), cfg, latent)
    if method == "TranAD":
        return _pca_reconstruction_payload("pca_time", _numpy_time_features(train_windows), cfg, latent)
    if method == "MOMENT":
        return _pca_reconstruction_payload("pca_patch", _numpy_patch_features(train_windows), cfg, latent)
    if method == "TimeMixer":
        return _pca_reconstruction_payload("pca_multiscale", _numpy_multiscale_features(train_windows), cfg, latent)
    if method == "ModernTCN":
        return _pca_reconstruction_payload("pca_temporal_conv", _numpy_temporal_conv_features(train_windows), cfg, latent)
    time_x = _numpy_time_features(train_windows)
    freq_x = _numpy_freq_features(train_windows)
    time_proj = PCA(n_components=min(latent, time_x.shape[1], len(time_x) - 1), random_state=cfg.seed).fit(time_x)
    freq_proj = PCA(n_components=min(latent, freq_x.shape[1], len(freq_x) - 1), random_state=cfg.seed + 1).fit(freq_x)
    zt = time_proj.transform(time_x)
    zf = freq_proj.transform(freq_x)
    center_t = zt.mean(axis=0)
    center_f = zf.mean(axis=0)
    return {"kind": "dual_pca", "time_proj": time_proj, "freq_proj": freq_proj, "center_t": center_t, "center_f": center_f}


def _score_sklearn(payload: dict[str, Any], windows: np.ndarray) -> np.ndarray:
    kind = payload["kind"]
    if kind == "pca_freq":
        x = _numpy_freq_features(windows)
        model = payload["model"]
        recon = model.inverse_transform(model.transform(x))
        return np.mean((x - recon) ** 2, axis=1)
    if kind == "pca_time":
        x = _numpy_time_features(windows)
        model = payload["model"]
        recon = model.inverse_transform(model.transform(x))
        return np.mean((x - recon) ** 2, axis=1)
    if kind == "pca_patch":
        x = _numpy_patch_features(windows)
        model = payload["model"]
        recon = model.inverse_transform(model.transform(x))
        return np.mean((x - recon) ** 2, axis=1)
    if kind == "pca_multiscale":
        x = _numpy_multiscale_features(windows)
        model = payload["model"]
        recon = model.inverse_transform(model.transform(x))
        return np.mean((x - recon) ** 2, axis=1)
    if kind == "pca_temporal_conv":
        x = _numpy_temporal_conv_features(windows)
        model = payload["model"]
        recon = model.inverse_transform(model.transform(x))
        return np.mean((x - recon) ** 2, axis=1)
    time_x = _numpy_time_features(windows)
    freq_x = _numpy_freq_features(windows)
    zt = payload["time_proj"].transform(time_x)
    zf = payload["freq_proj"].transform(freq_x)
    dim = min(zt.shape[1], zf.shape[1])
    zt = zt[:, :dim]
    zf = zf[:, :dim]
    cos = np.sum(zt * zf, axis=1) / (np.linalg.norm(zt, axis=1) * np.linalg.norm(zf, axis=1) + 1e-9)
    center_t = payload["center_t"][:dim]
    center_f = payload["center_f"][:dim]
    center_dist = np.linalg.norm(zt - center_t, axis=1) + np.linalg.norm(zf - center_f, axis=1)
    return (1.0 - cos) + 0.05 * center_dist


def _fit_torch_model(method: str, train_windows: np.ndarray, cfg: DeepBaselineConfig, device) -> dict[str, Any]:
    torch, nn, F = _torch_module()
    assert torch is not None and nn is not None and F is not None

    class MLPAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class TransformerAutoencoder(nn.Module):
        def __init__(self, channels: int, hidden_dim: int) -> None:
            super().__init__()
            nhead = 4 if hidden_dim % 4 == 0 else 2
            self.input_proj = nn.Linear(channels, hidden_dim)
            self.pos = nn.Parameter(torch.zeros(1, cfg.window_size, hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 2,
                dropout=0.05,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.output_proj = nn.Linear(hidden_dim, channels)

        def forward(self, x):
            h = self.input_proj(x) + self.pos[:, : x.shape[1]]
            h = self.encoder(h)
            return self.output_proj(h)

    class DualContrastiveNet(nn.Module):
        def __init__(self, time_dim: int, freq_dim: int, hidden_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.time_proj = nn.Sequential(
                nn.Linear(time_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.freq_proj = nn.Sequential(
                nn.Linear(freq_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, time_x, freq_x):
            zt = F.normalize(self.time_proj(time_x), dim=1)
            zf = F.normalize(self.freq_proj(freq_x), dim=1)
            return zt, zf

    class PatchAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.net = MLPAutoencoder(input_dim, hidden_dim, latent_dim)

        def forward(self, x):
            return self.net(x)

    class TimeMixerAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.GELU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class TemporalConvAutoencoder(nn.Module):
        def __init__(self, channels: int, hidden_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(channels, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
            )
            self.decoder = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, channels, kernel_size=3, padding=1),
            )

        def forward(self, x):
            h = self.encoder(x.transpose(1, 2))
            return self.decoder(h).transpose(1, 2)

    def batches(x: np.ndarray):
        rng = np.random.default_rng(cfg.seed)
        for _ in range(max(1, int(cfg.epochs))):
            order = rng.permutation(len(x))
            for start in range(0, len(order), max(1, int(cfg.batch_size))):
                yield x[order[start : start + int(cfg.batch_size)]]

    if method == "CATCH":
        train_x = _numpy_freq_features(train_windows)
        net = MLPAutoencoder(train_x.shape[1], cfg.hidden_dim, cfg.latent_dim).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
        net.train()
        for batch in batches(train_x):
            xb = torch.as_tensor(batch, dtype=torch.float32, device=device)
            loss = F.mse_loss(net(xb), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return {"kind": "torch_catch", "net": net}

    if method == "TranAD":
        net = TransformerAutoencoder(train_windows.shape[2], cfg.hidden_dim).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
        net.train()
        for batch in batches(train_windows):
            xb = torch.as_tensor(batch, dtype=torch.float32, device=device)
            recon = net(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return {"kind": "torch_tranad", "net": net}

    if method == "MOMENT":
        train_x = _numpy_patch_features(train_windows)
        net = PatchAutoencoder(train_x.shape[1], cfg.hidden_dim, cfg.latent_dim).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
        net.train()
        for batch in batches(train_x):
            xb = torch.as_tensor(batch, dtype=torch.float32, device=device)
            mask = (torch.rand_like(xb) > 0.15).float()
            masked = xb * mask
            loss = F.mse_loss(net(masked), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return {"kind": "torch_moment", "net": net}

    if method == "TimeMixer":
        train_x = _numpy_multiscale_features(train_windows)
        net = TimeMixerAutoencoder(train_x.shape[1], cfg.hidden_dim, cfg.latent_dim).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
        net.train()
        for batch in batches(train_x):
            xb = torch.as_tensor(batch, dtype=torch.float32, device=device)
            loss = F.smooth_l1_loss(net(xb), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return {"kind": "torch_timemixer", "net": net}

    if method == "ModernTCN":
        net = TemporalConvAutoencoder(train_windows.shape[2], cfg.hidden_dim).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
        net.train()
        for batch in batches(train_windows):
            xb = torch.as_tensor(batch, dtype=torch.float32, device=device)
            recon = net(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return {"kind": "torch_moderntcn", "net": net}

    time_x = _numpy_time_features(train_windows)
    freq_x = _numpy_freq_features(train_windows)
    net = DualContrastiveNet(time_x.shape[1], freq_x.shape[1], cfg.hidden_dim, cfg.latent_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    net.train()
    rng = np.random.default_rng(cfg.seed)
    for _ in range(max(1, int(cfg.epochs))):
        order = rng.permutation(len(time_x))
        for start in range(0, len(order), max(1, int(cfg.batch_size))):
            idx = order[start : start + int(cfg.batch_size)]
            tb = torch.as_tensor(time_x[idx], dtype=torch.float32, device=device)
            fb = torch.as_tensor(freq_x[idx], dtype=torch.float32, device=device)
            zt, zf = net(tb, fb)
            align_loss = 1.0 - F.cosine_similarity(zt, zf, dim=1).mean()
            var_loss = 0.5 * (torch.relu(0.5 - zt.std(dim=0)).mean() + torch.relu(0.5 - zf.std(dim=0)).mean())
            loss = align_loss + 0.05 * var_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        tb = torch.as_tensor(time_x, dtype=torch.float32, device=device)
        fb = torch.as_tensor(freq_x, dtype=torch.float32, device=device)
        zt, zf = net(tb, fb)
        center_t = zt.mean(dim=0).detach()
        center_f = zf.mean(dim=0).detach()
    return {"kind": "torch_dc", "net": net, "center_t": center_t, "center_f": center_f}


def _score_torch(payload: dict[str, Any], windows: np.ndarray, cfg: DeepBaselineConfig) -> np.ndarray:
    torch, _, F = _torch_module()
    assert torch is not None and F is not None
    kind = payload["kind"]
    net = payload["net"]
    device = next(net.parameters()).device
    net.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), max(1, int(cfg.batch_size))):
            batch = windows[start : start + int(cfg.batch_size)]
            if kind == "torch_catch":
                xb = torch.as_tensor(_numpy_freq_features(batch), dtype=torch.float32, device=device)
                recon = net(xb)
                score = torch.mean((recon - xb) ** 2, dim=1)
            elif kind == "torch_tranad":
                xb = torch.as_tensor(batch, dtype=torch.float32, device=device)
                recon = net(xb)
                last_step = torch.mean((recon[:, -1, :] - xb[:, -1, :]) ** 2, dim=1)
                full_window = torch.mean((recon - xb) ** 2, dim=(1, 2))
                score = last_step + 0.25 * full_window
            elif kind == "torch_moment":
                xb = torch.as_tensor(_numpy_patch_features(batch), dtype=torch.float32, device=device)
                recon = net(xb)
                score = torch.mean((recon - xb) ** 2, dim=1)
            elif kind == "torch_timemixer":
                xb = torch.as_tensor(_numpy_multiscale_features(batch), dtype=torch.float32, device=device)
                recon = net(xb)
                score = torch.mean((recon - xb) ** 2, dim=1)
            elif kind == "torch_moderntcn":
                xb = torch.as_tensor(batch, dtype=torch.float32, device=device)
                recon = net(xb)
                last_step = torch.mean((recon[:, -1, :] - xb[:, -1, :]) ** 2, dim=1)
                derivatives = torch.mean((torch.diff(recon, dim=1) - torch.diff(xb, dim=1)) ** 2, dim=(1, 2))
                score = last_step + 0.25 * derivatives
            else:
                tb = torch.as_tensor(_numpy_time_features(batch), dtype=torch.float32, device=device)
                fb = torch.as_tensor(_numpy_freq_features(batch), dtype=torch.float32, device=device)
                zt, zf = net(tb, fb)
                align = 1.0 - F.cosine_similarity(zt, zf, dim=1)
                center = torch.linalg.norm(zt - payload["center_t"], dim=1) + torch.linalg.norm(zf - payload["center_f"], dim=1)
                score = align + 0.05 * center
            scores.append(score.detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float64) if scores else np.zeros(0, dtype=np.float64)


def _score_windows(model: DeepBaselineModel, windows: np.ndarray) -> np.ndarray:
    if model.backend.startswith("torch:"):
        return _score_torch(model.payload, windows, model.cfg)
    return _score_sklearn(model.payload, windows)
