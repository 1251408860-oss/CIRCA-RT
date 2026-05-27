from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@dataclass(frozen=True)
class FrameRecord:
    src: Path
    dst: Path
    dataset_hint: str
    sequence_hint: str
    frame_index: int
    width: int
    height: int
    sha1: str


def _safe_name(text: str) -> str:
    cleaned = []
    for ch in str(text):
        cleaned.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    out = "".join(cleaned).strip("._")
    return out or "frame"


def _guess_dataset(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    joined = "/".join(path.parts).lower()
    if "nuimages" in joined:
        return "nuimages"
    if "argoverse" in joined or "av2" in parts:
        return "argoverse2_sensor"
    if "waymo" in joined:
        return "waymo_open"
    if "bdd" in joined or "bdd100k" in joined:
        return "bdd100k"
    if "nuscenes" in joined:
        return "nuscenes"
    if "droid" in joined:
        return "droid_robot"
    if "open_x_embodiment" in joined or "oxe" in parts:
        return "open_x_embodiment"
    if "bridge" in joined:
        return "bridgedata_v2"
    return "generic_camera"


def _sequence_from_camera_filename(path: Path, rel: Path) -> str | None:
    stem = path.stem
    if "__" not in stem or len(rel.parts) < 2:
        return None
    parts = stem.split("__")
    if len(parts) < 2:
        return None
    scene_hint = _safe_name(parts[0])
    camera_hint = _safe_name(parts[1])
    split_hint = _safe_name(rel.parts[0])
    return f"{scene_hint}_{camera_hint}_{split_hint}"


def _guess_sequence(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) <= 1:
        return root.name or "sequence"
    parsed = _sequence_from_camera_filename(path, rel)
    if parsed is not None:
        return parsed
    lower_parts = [p.lower() for p in rel.parts[:-1]]
    chunk_parts = [p for p in rel.parts if p.lower().startswith("chunk-")]
    camera_parts = [
        p
        for p in rel.parts
        if "observation.images" in p.lower() or p.lower().startswith("camera") or "wrist" in p.lower()
    ]
    if chunk_parts and camera_parts:
        return _safe_name(f"{chunk_parts[0]}_{camera_parts[0]}")
    camera_markers = {"ring_front_center", "ring_front_left", "ring_front_right", "cam_front", "camera", "images"}
    for idx, part in enumerate(lower_parts):
        if part in camera_markers and idx > 0:
            return _safe_name("_".join(rel.parts[:idx]))
    if len(rel.parts) >= 3:
        return _safe_name("_".join(rel.parts[:-2]))
    return _safe_name(rel.parts[0])


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None


def _sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def discover_images(input_root: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        p
        for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions and not any(part.startswith(".") for part in p.parts)
    )


def discover_videos(input_root: Path) -> list[Path]:
    return sorted(p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        return False
    return True


def _extract_video_frames_ffmpeg(video: Path, out_dir: Path, *, stride: int, limit: int | None) -> list[Path]:
    vf = f"select=not(mod(n\\,{max(1, stride)}))"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", vf, "-vsync", "vfr"]
    if limit is not None and limit > 0:
        cmd.extend(["-frames:v", str(limit)])
    cmd.append(str(out_dir / "frame_%08d.jpg"))
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("*.jpg"))


def _extract_video_frames_imageio(video: Path, out_dir: Path, *, stride: int, limit: int | None) -> list[Path]:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio and imageio-ffmpeg are required when ffmpeg CLI is unavailable") from exc

    written: list[Path] = []
    frame_stride = max(1, stride)
    reader = imageio.get_reader(str(video), format="ffmpeg")
    try:
        for raw_idx, frame in enumerate(reader):
            if raw_idx % frame_stride != 0:
                continue
            dst = out_dir / f"frame_{len(written) + 1:08d}.jpg"
            Image.fromarray(frame).convert("RGB").save(dst, quality=95)
            written.append(dst)
            if limit is not None and limit > 0 and len(written) >= limit:
                break
    finally:
        reader.close()
    return written


def _extract_video_frames_cv2(video: Path, out_dir: Path, *, stride: int, limit: int | None) -> list[Path]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python-headless is required for cv2 video extraction") from exc

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")
    written: list[Path] = []
    frame_stride = max(1, stride)
    raw_idx = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if raw_idx % frame_stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                dst = out_dir / f"frame_{len(written) + 1:08d}.jpg"
                Image.fromarray(frame_rgb).save(dst, quality=95)
                written.append(dst)
                if limit is not None and limit > 0 and len(written) >= limit:
                    break
            raw_idx += 1
    finally:
        capture.release()
    return written


def extract_video_frames(
    video: Path,
    scratch_dir: Path,
    *,
    stride: int,
    limit: int | None,
    cache_hint: str | None = None,
) -> list[Path]:
    out_dir = scratch_dir / _safe_name(cache_hint or video.with_suffix("").name)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.jpg"))
    if existing and (limit is None or limit <= 0 or len(existing) >= limit):
        return existing[:limit] if limit is not None and limit > 0 else existing
    if existing:
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    if ffmpeg_available():
        try:
            return _extract_video_frames_ffmpeg(video, out_dir, stride=stride, limit=limit)
        except Exception as exc:
            errors.append(f"ffmpeg: {exc}")
    for extractor_name, extractor in (("imageio", _extract_video_frames_imageio), ("cv2", _extract_video_frames_cv2)):
        try:
            frames = extractor(video, out_dir, stride=stride, limit=limit)
            if frames:
                return sorted(frames)
            errors.append(f"{extractor_name}: produced no frames")
        except Exception as exc:
            errors.append(f"{extractor_name}: {exc}")
    detail = "; ".join(errors) if errors else "no extractor was available"
    raise RuntimeError(f"failed to extract frames from {video}: {detail}")


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")


def prepare_frames(
    *,
    input_root: Path,
    out_dir: Path,
    mode: str,
    stride: int,
    limit: int | None,
    min_width: int,
    min_height: int,
    extensions: tuple[str, ...],
    include_videos: bool,
    scratch_dir: Path,
    dataset_hint: str | None,
    per_video_limit: int | None,
) -> list[FrameRecord]:
    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_images(input_root, extensions)
    extracted_metadata: dict[Path, tuple[str, str]] = {}
    if stride > 1:
        candidates = candidates[::stride]
    if include_videos:
        remaining = None if limit is None else max(0, limit - len(candidates))
        for video in discover_videos(input_root):
            if remaining == 0:
                break
            video_limit = remaining
            if per_video_limit is not None and per_video_limit > 0:
                video_limit = per_video_limit if remaining is None else min(remaining, per_video_limit)
            video_hint = dataset_hint or _guess_dataset(video)
            video_seq = _guess_sequence(video, input_root)
            rel_video = video.relative_to(input_root)
            cache_hint = f"{video_seq}__{rel_video.with_suffix('')}__stride-{max(1, stride)}"
            extracted = extract_video_frames(
                video,
                scratch_dir,
                stride=max(1, stride),
                limit=video_limit,
                cache_hint=cache_hint,
            )
            candidates.extend(extracted)
            for frame in extracted:
                extracted_metadata[frame] = (video_hint, video_seq)
            if remaining is not None:
                remaining = max(0, limit - len(candidates))

    if limit is not None and limit > 0:
        candidates = candidates[:limit]

    records: list[FrameRecord] = []
    per_sequence_counts: dict[str, int] = {}
    for src in candidates:
        size = _image_size(src)
        if size is None:
            continue
        width, height = size
        if width < min_width or height < min_height:
            continue
        if src in extracted_metadata:
            hint, seq = extracted_metadata[src]
        else:
            hint = dataset_hint or _guess_dataset(src)
            seq_root = input_root if _is_relative_to(src, input_root) else src.parent.parent
            seq = _guess_sequence(src, seq_root)
        idx = per_sequence_counts.get(seq, 0)
        per_sequence_counts[seq] = idx + 1
        ext = ".jpg" if src.suffix.lower() in (".jpeg", ".jpg") else src.suffix.lower()
        dst = out_dir / _safe_name(hint) / _safe_name(seq) / f"{idx:08d}{ext}"
        materialize(src, dst, mode)
        records.append(
            FrameRecord(
                src=src,
                dst=dst,
                dataset_hint=hint,
                sequence_hint=seq,
                frame_index=idx,
                width=width,
                height=height,
                sha1=_sha1(src),
            )
        )
    return records


def write_manifest(records: list[FrameRecord], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dst",
                "src",
                "dataset_hint",
                "sequence_hint",
                "frame_index",
                "width",
                "height",
                "sha1",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "dst": str(r.dst),
                    "src": str(r.src),
                    "dataset_hint": r.dataset_hint,
                    "sequence_hint": r.sequence_hint,
                    "frame_index": r.frame_index,
                    "width": r.width,
                    "height": r.height,
                    "sha1": r.sha1,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize real robot/driving/camera frames into data/frames.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing images or videos from AV/robot datasets.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "frames")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit after stride.")
    parser.add_argument("--min-width", type=int, default=128)
    parser.add_argument("--min-height", type=int, default=128)
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS))
    parser.add_argument("--include-videos", action="store_true")
    parser.add_argument("--scratch-dir", type=Path, default=ROOT / "data" / "frame_extract_scratch")
    parser.add_argument("--dataset-hint", default=None, help="Override auto-detected dataset name in the manifest.")
    parser.add_argument(
        "--per-video-limit",
        type=int,
        default=0,
        help="Maximum extracted frames per video before applying the global --limit; 0 disables per-video balancing.",
    )
    args = parser.parse_args()

    try:
        records = prepare_frames(
            input_root=args.input_root.resolve(),
            out_dir=args.out_dir.resolve(),
            mode=args.mode,
            stride=max(1, int(args.stride)),
            limit=None if int(args.limit) <= 0 else int(args.limit),
            min_width=max(1, int(args.min_width)),
            min_height=max(1, int(args.min_height)),
            extensions=tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions),
            include_videos=bool(args.include_videos),
            scratch_dir=args.scratch_dir.resolve(),
            dataset_hint=args.dataset_hint,
            per_video_limit=None if int(args.per_video_limit) <= 0 else int(args.per_video_limit),
        )
    except Exception as exc:
        raise SystemExit(f"failed to prepare frames: {exc}") from exc

    manifest = args.manifest or (args.out_dir / "manifest.csv")
    write_manifest(records, manifest.resolve())
    by_dataset: dict[str, int] = {}
    by_sequence: dict[str, int] = {}
    for r in records:
        by_dataset[r.dataset_hint] = by_dataset.get(r.dataset_hint, 0) + 1
        key = f"{r.dataset_hint}/{r.sequence_hint}"
        by_sequence[key] = by_sequence.get(key, 0) + 1
    print(f"prepared_frames={len(records)} out_dir={args.out_dir.resolve()} manifest={manifest.resolve()}")
    print("datasets=" + ", ".join(f"{k}:{v}" for k, v in sorted(by_dataset.items())))
    top_sequences = sorted(by_sequence.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("top_sequences=" + ", ".join(f"{k}:{v}" for k, v in top_sequences))
    if len(records) == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
