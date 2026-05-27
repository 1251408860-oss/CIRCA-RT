from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BASE_PATHS = [
    "configs",
    "src",
    "scripts",
    "docs",
    "paper_draft",
    "traces",
    "requirements.txt",
    "requirements_agx_orin64_runtime.txt",
]

DATA_PATHS = [
    "data/frames_av2",
    "data/frames_droid",
    "data/frames_nuimages",
]

OPTIONAL_DATA_PATHS = [
    "data/torchvision",
    "data/official_baseline_inputs",
    "data/official_psm_inputs",
    "data/official_psm_inputs_smoke",
    "data/official_psm_inputs_tiny",
    "data/frames_smoke",
]

TORCH_CACHE_PATH = "torch_cache"

AGX_SUPPORT_SOURCE = ROOT.parent / "agx_orin64"
AGX_SUPPORT_PATHS = [
    "wheels",
    "local_libs",
    "local_assets",
    "python_ortgpu",
]

IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
}


def should_ignore(name: str) -> bool:
    return name in IGNORE_NAMES or name.endswith(".pyc") or name.endswith(".pyo")


def ignore_func(_dir: str, names: list[str]) -> list[str]:
    return [name for name in names if should_ignore(name)]


def copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=ignore_func)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_inventory(root: Path, rel_paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rel in rel_paths:
        path = root / rel
        if not path.exists():
            rows.append({"path": rel, "exists": False, "files": 0, "bytes": 0})
            continue
        if path.is_file():
            rows.append({"path": rel, "exists": True, "files": 1, "bytes": path.stat().st_size})
            continue
        files = [p for p in path.rglob("*") if p.is_file()]
        rows.append(
            {
                "path": rel,
                "exists": True,
                "files": len(files),
                "bytes": sum(p.stat().st_size for p in files),
            }
        )
    return rows


def write_inventory_csv(out: Path, rows: list[dict[str, object]]) -> None:
    with (out / "PACKAGE_INVENTORY.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "exists", "files", "bytes"])
        writer.writeheader()
        writer.writerows(rows)


def write_readme(out: Path) -> None:
    source = ROOT / "docs" / "AGX_ORIN64_EXTERNAL_RUN_PACKAGE.md"
    if source.exists():
        text = source.read_text(encoding="utf-8")
    else:
        text = "# AGX Orin 64GB Run Package\n\nSee docs/AGX_ORIN64_REAL_DEVICE_RUNBOOK.md.\n"
    (out / "README.md").write_text(text, encoding="utf-8")
    (out / "README_AGX_ORIN64_RUN.md").write_text(text, encoding="utf-8")


def copy_project_readme(out: Path) -> None:
    source = ROOT / "README.md"
    if source.exists():
        dst = out / "docs" / "PROJECT_README_SOURCE.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst)


def build(
    out: Path,
    include_data: bool,
    include_optional_data: bool,
    include_torch_cache: bool,
    include_agx_support: bool,
) -> dict[str, object]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for rel in BASE_PATHS:
        src = ROOT / rel
        if src.exists():
            copy_path(src, out / rel)
            copied.append(rel)

    data_paths: list[str] = []
    if include_data:
        for rel in DATA_PATHS:
            src = ROOT / rel
            if src.exists():
                copy_path(src, out / rel)
                data_paths.append(rel)

    optional_data_paths: list[str] = []
    if include_optional_data:
        for rel in OPTIONAL_DATA_PATHS:
            src = ROOT / rel
            if src.exists():
                copy_path(src, out / rel)
                optional_data_paths.append(rel)

    torch_cache_paths: list[str] = []
    if include_torch_cache:
        src = ROOT / TORCH_CACHE_PATH
        if src.exists():
            copy_path(src, out / TORCH_CACHE_PATH)
            torch_cache_paths.append(TORCH_CACHE_PATH)

    agx_support_paths: list[str] = []
    if include_agx_support and AGX_SUPPORT_SOURCE.exists():
        for rel in AGX_SUPPORT_PATHS:
            src = AGX_SUPPORT_SOURCE / rel
            if src.exists():
                dst_rel = f"agx_support/{rel}"
                copy_path(src, out / dst_rel)
                agx_support_paths.append(dst_rel)

    write_readme(out)
    copy_project_readme(out)

    inventory_paths = copied + data_paths + optional_data_paths + torch_cache_paths + agx_support_paths
    inventory = dir_inventory(out, inventory_paths)
    write_inventory_csv(out, inventory)

    file_count = 0
    byte_count = 0
    for path in out.rglob("*"):
        if path.is_file():
            file_count += 1
            byte_count += path.stat().st_size

    manifest: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "package_root": out.name,
        "include_data": include_data,
        "include_optional_data": include_optional_data,
        "include_torch_cache": include_torch_cache,
        "include_agx_support": include_agx_support,
        "copied_base_paths": copied,
        "copied_data_paths": data_paths,
        "copied_optional_data_paths": optional_data_paths,
        "copied_torch_cache_paths": torch_cache_paths,
        "copied_agx_support_paths": agx_support_paths,
        "file_count": file_count,
        "bytes": byte_count,
        "inventory": inventory,
        "primary_commands": [
            "PYTHON_BIN=python3 bash scripts/jetson_env_check.sh | tee jetson_env_check_first.log",
            "python3 scripts/run_jetson_tensorrt_engine_check.py --out results_agx_orin64_env/provider_status.json",
            "PYTHON_BIN=python3 RESULT_ROOT=results_agx_orin64_comprehensive_smoke AGX_DATASETS=\"av2\" DEADLINES_MS=\"33.333\" MODELS=\"mobilenet_v2\" SEEDS=\"7\" N_AV2=300 FRAME_LIMIT_AV2=300 RUN_BACKEND_MATRIX=0 bash scripts/run_agx_orin64_comprehensive_experiment.sh",
            "PYTHON_BIN=python3 RESULT_ROOT=results_agx_orin64_comprehensive_202605xx AGX_DATASETS=\"av2 droid nuimages\" DEADLINES_MS=\"33.333 50\" MODELS=\"mobilenet_v2 resnet18\" SEEDS=\"7 8 9\" RUN_BACKEND_MATRIX=1 RUN_DERIVED_ANALYSES=1 bash scripts/run_agx_orin64_comprehensive_experiment.sh",
            "PYTHON_BIN=python3 FRAME_DIR=data/frames_av2 MODELS=\"mobilenet_v2\" SEEDS=\"7 8 9\" N=600 PERIOD_MS=33.333 DEADLINE_MS=33.333 PAYLOAD_MODE=jpeg_b64 PUBLISH_RESIZE=320 bash scripts/run_agx_orin64_ros2_closed_loop.sh",
            "PHASES=\"env ros2_closed_loop resnet50_realframes backend_matrix pressure_short\" FAIL_FAST=0 bash scripts/run_agx_orin64_submission_sprint.sh",
            "PYTHON_BIN=python3 bash scripts/run_agx_orin64_maximal_experiment.sh",
        ],
    }
    (out / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def make_zip(out: Path) -> Path:
    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out.parent))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AGX Orin 64GB external run package.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--include-optional-data", action="store_true")
    parser.add_argument("--include-torch-cache", action="store_true")
    parser.add_argument("--include-agx-support", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    out = args.out_dir
    if not out.is_absolute():
        out = ROOT / out

    manifest = build(
        out,
        include_data=args.include_data,
        include_optional_data=args.include_optional_data,
        include_torch_cache=args.include_torch_cache,
        include_agx_support=args.include_agx_support,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    if args.zip:
        zip_path = make_zip(out)
        print(f"wrote {zip_path}")
        print(f"sha256 {sha256_file(zip_path)}")


if __name__ == "__main__":
    main()
