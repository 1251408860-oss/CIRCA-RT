from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        shutil.unpack_archive(str(archive), str(dest), "zip")
        return
    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar.bz2"):
        shutil.unpack_archive(str(archive), str(dest))
        return
    raise ValueError(f"unsupported archive format: {archive}")


def _existing_or_extract(input_paths: list[Path], work_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for item in input_paths:
        item = item.resolve()
        if item.is_dir():
            roots.append(item)
            continue
        if item.is_file():
            target = work_dir / item.stem
            if not target.exists():
                _extract_archive(item, target)
            roots.append(target)
            continue
        raise FileNotFoundError(item)
    return roots


def _prepare_frames(
    *,
    input_root: Path,
    out_dir: Path,
    dataset_hint: str,
    include_videos: bool,
    stride: int,
    limit: int,
    min_width: int,
    min_height: int,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_real_frame_dataset.py"),
        "--input-root",
        str(input_root),
        "--out-dir",
        str(out_dir),
        "--mode",
        "symlink",
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
    ]
    if include_videos:
        cmd.append("--include-videos")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import manually downloaded public real datasets into the unified data/frames_* layout."
    )
    parser.add_argument(
        "--dataset",
        choices=["bdd100k", "nuscenes", "nuimages", "robot_camera", "generic"],
        required=True,
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Input directory or archive(s) from official dataset downloads.",
    )
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data" / "raw" / "prepared_imports")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "frames_imported")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    roots = _existing_or_extract(list(args.input), work_dir)

    dataset_configs = {
        "bdd100k": {"hint": "bdd100k", "include_videos": False},
        "nuscenes": {"hint": "nuscenes", "include_videos": False},
        "nuimages": {"hint": "nuimages", "include_videos": False},
        "robot_camera": {"hint": "robot_camera", "include_videos": True},
        "generic": {"hint": "generic_camera", "include_videos": True},
    }
    config = dataset_configs[args.dataset]
    for root in roots:
        _prepare_frames(
            input_root=root,
            out_dir=args.out_dir.resolve(),
            dataset_hint=config["hint"],
            include_videos=bool(config["include_videos"]),
            stride=max(1, int(args.stride)),
            limit=max(0, int(args.limit)),
            min_width=max(1, int(args.min_width)),
            min_height=max(1, int(args.min_height)),
        )


if __name__ == "__main__":
    main()
