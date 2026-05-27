from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path("/root/rtss_exp_begin")
DATA_ROOT = ROOT / "data" / "torchvision"
LOG_ROOT = ROOT / "autodl_runs" / "logs"
TORCH_CACHE = Path("/root/.cache/torch/hub/checkpoints")

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"
WEIGHTS = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    log("run: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_checked(url: str, path: Path, expected_md5: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        if path.exists() and expected_md5 and md5(path) == expected_md5:
            log(f"already valid: {path}")
            return
        if path.exists() and expected_md5:
            log(f"existing file has wrong md5, retrying with resume: {path} got={md5(path)} expected={expected_md5}")
        run(["wget", "-c", "-O", str(path), url])
        if expected_md5 is None:
            return
        got = md5(path)
        if got == expected_md5:
            log(f"download verified: {path} md5={got}")
            return
        log(f"md5 mismatch attempt={attempt}: got={got} expected={expected_md5}")
        path.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download valid file after retries: {url}")


def safe_extract_tar_gz(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as tar:
        dest_resolved = dest.resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest_resolved) + os.sep):
                raise RuntimeError(f"unsafe tar member: {member.name}")
        tar.extractall(dest)


def prepare_cifar() -> dict[str, object]:
    archive = DATA_ROOT / "cifar-10-python.tar.gz"
    download_checked(CIFAR_URL, archive, CIFAR_MD5)
    extracted = DATA_ROOT / "cifar-10-batches-py"
    if not extracted.exists():
        log(f"extracting {archive}")
        safe_extract_tar_gz(archive, DATA_ROOT)
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root=str(DATA_ROOT), train=True, download=False)
    test = CIFAR10(root=str(DATA_ROOT), train=False, download=False)
    return {
        "archive": str(archive),
        "archive_size": archive.stat().st_size,
        "archive_md5": md5(archive),
        "train_len": len(train),
        "test_len": len(test),
        "extracted": str(extracted),
    }


def prepare_weights() -> dict[str, object]:
    import torch
    import torchvision.models as models
    from torch.hub import load_state_dict_from_url

    TORCH_CACHE.mkdir(parents=True, exist_ok=True)
    rows: dict[str, object] = {}
    for name, url in WEIGHTS.items():
        filename = Path(urlparse(url).path).name
        path = TORCH_CACHE / filename
        for attempt in range(1, 3):
            try:
                state = load_state_dict_from_url(url, model_dir=str(TORCH_CACHE), check_hash=True, progress=True, map_location="cpu")
                break
            except Exception:
                if attempt >= 2:
                    raise
                log(f"removing possibly corrupt weight file and retrying: {path}")
                path.unlink(missing_ok=True)
        builder = getattr(models, name)
        weight_enum = {
            "mobilenet_v2": models.MobileNet_V2_Weights.DEFAULT,
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "squeezenet1_1": models.SqueezeNet1_1_Weights.DEFAULT,
        }[name]
        model = builder(weights=weight_enum).eval()
        rows[name] = {
            "url": url,
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256(path),
            "state_dict_keys": len(state),
            "loaded_class": model.__class__.__name__,
        }
        del model
    rows["torch"] = torch.__version__
    rows["torch_cache"] = str(TORCH_CACHE)
    return rows


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "cifar10": prepare_cifar(),
        "weights": prepare_weights(),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = LOG_ROOT / "download_asset_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"wrote manifest: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {exc!r}")
        raise
