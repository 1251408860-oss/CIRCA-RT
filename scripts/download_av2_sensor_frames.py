from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S3_LIST_URL = "https://argoverse.s3.amazonaws.com/"
S3_OBJECT_URL = "https://argoverse.s3.amazonaws.com/{key}"
XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int


def list_s3_page(prefix: str, continuation_token: str | None = None, *, max_keys: int = 1000) -> tuple[list[S3Object], str | None]:
    params = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
    if continuation_token:
        params["continuation-token"] = continuation_token
    url = S3_LIST_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    objects: list[S3Object] = []
    for node in root.findall("s3:Contents", XML_NS):
        key = node.findtext("s3:Key", default="", namespaces=XML_NS)
        size = int(node.findtext("s3:Size", default="0", namespaces=XML_NS))
        if key:
            objects.append(S3Object(key=key, size=size))
    token = root.findtext("s3:NextContinuationToken", default=None, namespaces=XML_NS)
    return objects, token


def parse_av2_key(key: str) -> tuple[str, str, str] | None:
    parts = key.split("/")
    try:
        split_idx = parts.index("sensor") + 1
        split = parts[split_idx]
        log_id = parts[split_idx + 1]
        camera_idx = parts.index("cameras") + 1
        camera = parts[camera_idx]
    except (ValueError, IndexError):
        return None
    return split, log_id, camera


def discover_frame_keys(
    *,
    split: str,
    cameras: set[str],
    log_ids: list[str],
    max_frames: int,
    max_logs: int,
    max_frames_per_log_camera: int,
    list_page_size: int,
) -> list[S3Object]:
    if log_ids:
        selected: list[S3Object] = []
        for log_id in log_ids:
            for camera in sorted(cameras):
                prefix = f"datasets/av2/sensor/{split}/{log_id}/sensors/cameras/{camera}/"
                token: str | None = None
                per_camera = 0
                while True:
                    objects, token = list_s3_page(prefix, token, max_keys=list_page_size)
                    for obj in objects:
                        if not obj.key.lower().endswith(".jpg"):
                            continue
                        if max_frames_per_log_camera > 0 and per_camera >= max_frames_per_log_camera:
                            break
                        selected.append(obj)
                        per_camera += 1
                        if len(selected) >= max_frames:
                            return selected
                    if not token or (max_frames_per_log_camera > 0 and per_camera >= max_frames_per_log_camera):
                        break
        return selected

    prefix = f"datasets/av2/sensor/{split}/"
    selected: list[S3Object] = []
    per_log_camera: dict[tuple[str, str], int] = {}
    seen_logs: set[str] = set()
    token: str | None = None
    while True:
        objects, token = list_s3_page(prefix, token, max_keys=list_page_size)
        for obj in objects:
            if not obj.key.lower().endswith(".jpg"):
                continue
            parsed = parse_av2_key(obj.key)
            if parsed is None:
                continue
            _, log_id, camera = parsed
            if camera not in cameras:
                continue
            if max_logs > 0 and log_id not in seen_logs and len(seen_logs) >= max_logs:
                continue
            count_key = (log_id, camera)
            if max_frames_per_log_camera > 0 and per_log_camera.get(count_key, 0) >= max_frames_per_log_camera:
                continue
            seen_logs.add(log_id)
            per_log_camera[count_key] = per_log_camera.get(count_key, 0) + 1
            selected.append(obj)
            if len(selected) >= max_frames:
                return selected
        if not token:
            return selected


def local_path_for_key(out_dir: Path, key: str) -> Path:
    parts = key.split("/")
    split_idx = parts.index("sensor") + 1
    split = parts[split_idx]
    log_id = parts[split_idx + 1]
    camera = parts[parts.index("cameras") + 1]
    name = parts[-1]
    return out_dir / split / log_id / camera / name


def _download_with_curl(url: str, tmp: Path, *, connect_timeout: int, max_time: int, retries: int) -> None:
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        str(connect_timeout),
        "--max-time",
        str(max_time),
        "--retry",
        str(retries),
        "--retry-delay",
        "1",
        "--output",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)


def _download_with_wget(url: str, tmp: Path, *, connect_timeout: int, max_time: int, retries: int) -> None:
    cmd = [
        "wget",
        "--quiet",
        "--tries",
        str(max(1, retries + 1)),
        "--connect-timeout",
        str(connect_timeout),
        "--read-timeout",
        str(max_time),
        "--timeout",
        str(max_time),
        "-O",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)


def _download_with_urllib(url: str, tmp: Path, *, max_time: int) -> None:
    with urllib.request.urlopen(url, timeout=max_time) as response, tmp.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def download_one(
    obj: S3Object,
    out_dir: Path,
    retries: int,
    connect_timeout: int,
    max_time: int,
) -> tuple[S3Object, Path, str]:
    dst = local_path_for_key(out_dir, obj.key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == obj.size:
        return obj, dst, "cached"
    url = S3_OBJECT_URL.format(key=urllib.parse.quote(obj.key))
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for attempt in range(retries + 1):
        try:
            if shutil.which("curl"):
                _download_with_curl(url, tmp, connect_timeout=connect_timeout, max_time=max_time, retries=0)
            elif shutil.which("wget"):
                _download_with_wget(url, tmp, connect_timeout=connect_timeout, max_time=max_time, retries=0)
            else:
                _download_with_urllib(url, tmp, max_time=max_time)
            if tmp.stat().st_size != obj.size:
                raise IOError(f"size mismatch for {obj.key}: expected {obj.size}, got {tmp.stat().st_size}")
            tmp.replace(dst)
            return obj, dst, "downloaded"
        except (subprocess.CalledProcessError, urllib.error.URLError, TimeoutError, IOError) as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt >= retries:
                return obj, dst, f"failed:{type(exc).__name__}:{exc}"
            time.sleep(1.5 * (attempt + 1))
    return obj, dst, "failed:unknown"


def write_manifest(rows: list[tuple[S3Object, Path, str]], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "local_path", "size", "status", "split", "log_id", "camera"])
        writer.writeheader()
        for obj, dst, status in rows:
            parsed = parse_av2_key(obj.key)
            split, log_id, camera = parsed if parsed is not None else ("", "", "")
            writer.writerow(
                {
                    "key": obj.key,
                    "local_path": str(dst),
                    "size": obj.size,
                    "status": status,
                    "split": split,
                    "log_id": log_id,
                    "camera": camera,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a reproducible Argoverse 2 Sensor camera-frame subset from public S3.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--log-ids", nargs="*", default=[], help="Optional AV2 log IDs. Direct-prefix listing is much faster than split-wide scanning.")
    parser.add_argument("--cameras", nargs="+", default=["ring_front_center", "ring_front_left", "ring_front_right"])
    parser.add_argument("--max-frames", type=int, default=3000)
    parser.add_argument("--max-logs", type=int, default=12, help="0 means no limit.")
    parser.add_argument("--max-frames-per-log-camera", type=int, default=120, help="0 means no limit.")
    parser.add_argument("--list-page-size", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw" / "av2_sensor_subset")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument("--download-timeout", type=int, default=90)
    parser.add_argument("--allow-failures", type=int, default=0)
    parser.add_argument("--min-success", type=int, default=0, help="0 means require max-frames minus allowed failures.")
    args = parser.parse_args()

    objects = discover_frame_keys(
        split=args.split,
        cameras=set(args.cameras),
        log_ids=list(args.log_ids),
        max_frames=max(1, int(args.max_frames)),
        max_logs=max(0, int(args.max_logs)),
        max_frames_per_log_camera=max(0, int(args.max_frames_per_log_camera)),
        list_page_size=max(1, int(args.list_page_size)),
    )
    if not objects:
        raise SystemExit("no AV2 camera frames discovered; check split/camera arguments")
    total_mb = sum(obj.size for obj in objects) / (1024 * 1024)
    print(f"selected_frames={len(objects)} total_size_mb={total_mb:.1f} out_dir={args.out_dir}", flush=True)

    rows: list[tuple[S3Object, Path, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                download_one,
                obj,
                args.out_dir,
                int(args.retries),
                max(1, int(args.connect_timeout)),
                max(5, int(args.download_timeout)),
            )
            for obj in objects
        ]
        for idx, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            rows.append(row)
            if idx == 1 or idx % 100 == 0 or idx == len(futures):
                done = sum(1 for _, _, status in rows if status in {"downloaded", "cached"})
                failed = sum(1 for _, _, status in rows if status.startswith("failed"))
                print(f"progress {idx}/{len(futures)} ok={done} failed={failed}", flush=True)

    manifest = args.manifest or (args.out_dir / "download_manifest.csv")
    rows.sort(key=lambda row: row[0].key)
    write_manifest(rows, manifest)
    failed = [row for row in rows if row[2].startswith("failed")]
    print(f"wrote_manifest={manifest} ok={len(rows) - len(failed)} failed={len(failed)}")
    min_success = int(args.min_success) if int(args.min_success) > 0 else max(0, int(args.max_frames) - int(args.allow_failures))
    if len(rows) - len(failed) < min_success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
