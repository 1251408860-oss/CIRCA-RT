from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MIN_VALID_BYTES = 1024


@dataclass(frozen=True)
class SelectedFile:
    repo_path: str
    size: int | None


def _path_of(item: object) -> str:
    path = getattr(item, "path", None) or getattr(item, "rfilename", None)
    if path is None:
        raise ValueError(f"cannot read HuggingFace path from {type(item).__name__}")
    return str(path)


def _size_of(item: object) -> int | None:
    size = getattr(item, "size", None)
    return int(size) if isinstance(size, int) else None


def _is_file(item: object) -> bool:
    item_type = getattr(item, "type", None)
    return item_type in (None, "file")


def _hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _direct_download_url(*, repo_id: str, repo_path: str) -> str:
    # Build the URL explicitly so mirrors work even when huggingface_hub falls back to the default endpoint.
    return f"{_hf_endpoint()}/datasets/{repo_id}/resolve/main/{quote(repo_path, safe='/')}"


def select_files(
    *,
    repo_id: str,
    path_prefixes: list[str],
    extensions: tuple[str, ...],
    max_files: int,
) -> list[SelectedFile]:
    from huggingface_hub import HfApi

    endpoint = os.environ.get("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    selected: list[SelectedFile] = []
    seen: set[str] = set()
    prefixes = path_prefixes or [""]
    for prefix in prefixes:
        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=prefix,
            recursive=True,
        ):
            if not _is_file(item):
                continue
            repo_path = _path_of(item)
            if repo_path in seen or not repo_path.lower().endswith(extensions):
                continue
            selected.append(SelectedFile(repo_path=repo_path, size=_size_of(item)))
            seen.add(repo_path)
            if max_files > 0 and len(selected) >= max_files:
                return selected
    return selected


def download_files(*, repo_id: str, selected: list[SelectedFile], raw_dir: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, item in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] download {item.repo_path}", flush=True)
        target = raw_dir / item.repo_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size >= MIN_VALID_BYTES:
            paths.append(target)
            continue
        if target.exists():
            target.unlink()

        try:
            local = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=item.repo_path,
                local_dir=raw_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            local_path = Path(local)
            if local_path.exists() and local_path.stat().st_size >= MIN_VALID_BYTES:
                paths.append(local_path)
                continue
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"downloaded file is empty or too small: {local_path}")
        except Exception as exc:
            print(f"  hf_hub_download failed, falling back to direct URL: {exc}", flush=True)

        url = _direct_download_url(repo_id=repo_id, repo_path=item.repo_path)
        ok = False
        if shutil.which("wget"):
            cmd = ["wget", "-c", "--show-progress", "--progress=dot:giga", "-O", str(target), url]
            try:
                subprocess.run(cmd, check=True)
                ok = target.exists() and target.stat().st_size >= MIN_VALID_BYTES
            except Exception as exc:
                print(f"  wget failed: {exc}", flush=True)
        if not ok and shutil.which("curl"):
            cmd = ["curl", "-L", "--fail", "--silent", "--show-error", "-C", "-", "--output", str(target), url]
            try:
                subprocess.run(cmd, check=True)
                ok = target.exists() and target.stat().st_size >= MIN_VALID_BYTES
            except Exception as exc:
                print(f"  curl failed: {exc}", flush=True)
        if ok:
            paths.append(target)
            continue
        if target.exists():
            target.unlink()
        if not shutil.which("wget") and not shutil.which("curl"):
            raise RuntimeError("neither wget nor curl is available for direct dataset download")
        raise RuntimeError(f"failed to download a valid non-empty file: {item.repo_path}")
    return paths


def write_download_manifest(*, manifest: Path, repo_id: str, selected: list[SelectedFile], local_paths: list[Path]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["repo_id", "repo_path", "local_path", "size"])
        writer.writeheader()
        for item, local in zip(selected, local_paths):
            writer.writerow(
                {
                    "repo_id": repo_id,
                    "repo_path": item.repo_path,
                    "local_path": str(local),
                    "size": "" if item.size is None else item.size,
                }
            )


def prepare_frames(
    *,
    raw_dir: Path,
    frames_out: Path,
    dataset_hint: str,
    stride: int,
    limit: int,
    per_video_limit: int,
    mode: str,
    min_width: int,
    min_height: int,
) -> None:
    script = ROOT / "scripts" / "prepare_real_frame_dataset.py"
    cmd = [
        sys.executable,
        str(script),
        "--input-root",
        str(raw_dir),
        "--out-dir",
        str(frames_out),
        "--mode",
        mode,
        "--stride",
        str(max(1, stride)),
        "--limit",
        str(max(0, limit)),
        "--min-width",
        str(max(1, min_width)),
        "--min-height",
        str(max(1, min_height)),
        "--dataset-hint",
        dataset_hint,
        "--include-videos",
    ]
    if per_video_limit > 0:
        cmd.extend(["--per-video-limit", str(per_video_limit)])
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a small HuggingFace dataset video/image subset and normalize it into data/frames_*."
    )
    parser.add_argument("--repo-id", default="lerobot/droid_100", help="HuggingFace dataset repo id.")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional HuggingFace endpoint or mirror, e.g. https://hf-mirror.com. Also honors HF_ENDPOINT.",
    )
    parser.add_argument(
        "--repo-file",
        action="append",
        default=[],
        help="Exact file path inside the dataset repo. Repeatable; bypasses repository listing.",
    )
    parser.add_argument("--path-prefix", action="append", default=[], help="Optional repo path prefix. Repeatable.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "droid_100_subset")
    parser.add_argument("--frames-out", type=Path, default=ROOT / "data" / "frames_droid")
    parser.add_argument("--dataset-hint", default="droid_robot")
    parser.add_argument("--max-files", type=int, default=12, help="Maximum videos/images to download; 0 means no cap.")
    parser.add_argument("--max-frames", type=int, default=2400, help="Maximum normalized frames; 0 means no cap.")
    parser.add_argument(
        "--per-video-frames",
        type=int,
        default=0,
        help="Maximum frames extracted from each video before the global --max-frames cap; 0 disables balancing.",
    )
    parser.add_argument("--stride", type=int, default=10, help="Frame stride inside videos and image lists.")
    parser.add_argument("--min-width", type=int, default=128)
    parser.add_argument("--min-height", type=int, default=128)
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--extensions", nargs="+", default=list(VIDEO_EXTENSIONS + IMAGE_EXTENSIONS))
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")

    extensions = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions)
    if args.repo_file:
        selected = [SelectedFile(repo_path=p, size=None) for p in args.repo_file if p.lower().endswith(extensions)]
    else:
        selected = select_files(
            repo_id=args.repo_id,
            path_prefixes=args.path_prefix,
            extensions=extensions,
            max_files=max(0, int(args.max_files)),
        )
    if not selected:
        raise SystemExit(f"no matching files found in {args.repo_id}; check --path-prefix and --extensions")

    raw_dir = args.raw_dir.resolve()
    local_paths = download_files(repo_id=args.repo_id, selected=selected, raw_dir=raw_dir)
    write_download_manifest(
        manifest=raw_dir / "download_manifest.csv",
        repo_id=args.repo_id,
        selected=selected,
        local_paths=local_paths,
    )
    prepare_frames(
        raw_dir=raw_dir,
        frames_out=args.frames_out.resolve(),
        dataset_hint=args.dataset_hint,
        stride=max(1, int(args.stride)),
        limit=max(0, int(args.max_frames)),
        per_video_limit=max(0, int(args.per_video_frames)),
        mode=args.mode,
        min_width=max(1, int(args.min_width)),
        min_height=max(1, int(args.min_height)),
    )


if __name__ == "__main__":
    main()
