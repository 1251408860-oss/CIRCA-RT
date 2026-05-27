from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".webp"}
MODEL_FEATURE_COLUMNS = [
    "model_entropy",
    "model_top1_prob",
    "model_top1_margin",
    "model_top5_mass",
    "model_logit_l2",
    "model_prob_drift_l1",
    "model_top1_changed",
]


@dataclass(frozen=True)
class RobustReferenceStats:
    columns: tuple[str, ...]
    center: dict[str, float]
    scale: dict[str, float]


def list_frame_paths(frame_dir: str | Path, *, limit: int | None = None) -> list[Path]:
    root = Path(frame_dir)
    if not root.exists():
        raise FileNotFoundError(f"frame directory does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"no image files found under {root}")
    return paths[:limit] if limit is not None else paths


def load_image(path: str | Path | Image.Image) -> Image.Image:
    if isinstance(path, Image.Image):
        return path.convert("RGB")
    with Image.open(path) as img:
        return img.convert("RGB")


def demo_frame(index: int, *, size: int = 224, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed + index * 1009)
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = np.stack(
        [
            0.35 + 0.35 * np.sin(2 * np.pi * (xx + 0.07 * index)),
            0.30 + 0.45 * yy,
            0.25 + 0.35 * np.cos(2 * np.pi * (xx * yy + 0.03 * index)),
        ],
        axis=-1,
    )
    noise = rng.normal(0.0, 0.025, size=(size, size, 3)).astype(np.float32)
    arr = np.clip(base + noise, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    w = max(8, size // 12)
    x0 = int((index * 17) % max(size - w, 1))
    y0 = int((index * 29) % max(size - w, 1))
    draw.rectangle([x0, y0, x0 + 2 * w, y0 + w], outline=(230, 230, 230), width=max(1, size // 96))
    return img


def apply_frame_perturbation(
    image: Image.Image,
    *,
    scenario: str,
    attacked: bool,
    seq: int,
    seed: int,
) -> Image.Image:
    """Apply deterministic frame perturbations for scenario replay and audit measurement."""
    img = image.convert("RGB")
    if not attacked:
        factor = 1.0 + 0.015 * np.sin(0.17 * seq + seed)
        return ImageEnhance.Brightness(img).enhance(float(factor))

    rng = np.random.default_rng(seed * 1000003 + seq * 9176 + len(scenario))
    if scenario == "mode_shift":
        img = ImageEnhance.Color(img).enhance(0.55)
        img = ImageEnhance.Contrast(img).enhance(1.65)
        return ImageEnhance.Brightness(img).enhance(0.82)

    if scenario in {"semantic_corruption", "coupled_semantic_timing_attack"}:
        strength = coupled_attack_strength(seq, seed) if scenario == "coupled_semantic_timing_attack" else 0.5
        severity = 0.85 + 2.25 * strength if scenario == "coupled_semantic_timing_attack" else 1.0
        arr = np.asarray(img).astype(np.float32) / 255.0
        noise = rng.normal(0.0, 0.055 * severity, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        if scenario == "coupled_semantic_timing_attack":
            yy, xx = np.indices(arr.shape[:2])
            checker = (((xx // 8 + yy // 8 + seq) % 2) * 2.0 - 1.0).astype(np.float32)
            arr = np.clip(arr + (0.035 + 0.060 * strength) * checker[..., None], 0.0, 1.0)
        out = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
        out = out.filter(ImageFilter.GaussianBlur(radius=0.55 + 0.70 * severity))
        draw = ImageDraw.Draw(out)
        w, h = out.size
        box_w = int(w * min(0.62, (0.18 + 0.14 * rng.random()) * severity))
        box_h = int(h * min(0.58, (0.15 + 0.14 * rng.random()) * severity))
        x0 = int(rng.integers(0, max(w - box_w, 1)))
        y0 = int(rng.integers(0, max(h - box_h, 1)))
        fill = (12, 12, 12) if scenario == "semantic_corruption" else (int(30 + 170 * strength), 18, int(210 - 130 * strength))
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=fill)
        return out

    if scenario == "stale_replay_attack":
        img = ImageEnhance.Brightness(img).enhance(0.92)
        return ImageEnhance.Contrast(img).enhance(0.88)

    return img


def robust_reference_stats(df, columns: Iterable[str] = MODEL_FEATURE_COLUMNS) -> RobustReferenceStats:
    center: dict[str, float] = {}
    scale: dict[str, float] = {}
    cols = tuple(columns)
    for col in cols:
        values = np.asarray(df[col], dtype=np.float64)
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        robust_sigma = 1.4826 * mad
        std = float(np.std(values))
        sigma = max(robust_sigma, 0.25 * std, 1e-6)
        center[col] = med
        scale[col] = sigma
    return RobustReferenceStats(columns=cols, center=center, scale=scale)


def add_perception_semantic_residuals(df, ref: RobustReferenceStats):
    out = df.copy()
    z_cols: list[str] = []
    for col in ref.columns:
        z_col = f"{col}_robust_z"
        out[z_col] = (out[col].astype(float) - ref.center[col]) / ref.scale[col]
        z_cols.append(z_col)
    z = out[z_cols].to_numpy(dtype=np.float64)
    out["semantic_residual"] = np.sqrt(np.mean(z * z, axis=1))
    return out


def scenario_attack_mask(n: int, scenario: str, *, seed: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if scenario == "nominal":
        return mask
    rng = np.random.default_rng(seed + len(scenario) * 4099)
    length = max(24, min(120, n // 8))
    starts = np.linspace(int(n * 0.22), int(n * 0.78), 3, dtype=int)
    for start in starts:
        lo = int(np.clip(start + rng.integers(-max(4, length // 4), max(5, length // 4 + 1)), 0, max(n - length, 0)))
        mask[lo : lo + length] = True
    return mask


def coupled_attack_strength(seq: int, seed: int) -> float:
    """Shared latent used to couple semantic corruption severity and GPU pressure."""
    phase = 0.071 * float(seq) + 0.37 * float(seed)
    carrier = 0.5 + 0.5 * np.sin(phase)
    ripple = 0.5 + 0.5 * np.sin(0.173 * float(seq) + 1.31 * float(seed))
    return float(np.clip(0.72 * carrier + 0.28 * ripple, 0.0, 1.0))


def max_rolling_sum(values: np.ndarray, window: int) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    w = int(max(1, min(window, arr.size)))
    prefix = np.concatenate([[0.0], np.cumsum(arr)])
    return float(np.max(prefix[w:] - prefix[:-w]))
