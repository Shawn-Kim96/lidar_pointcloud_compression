#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


URLS: dict[str, str] = {
    # KITTI Odometry Velodyne point clouds (sequences 00-21)
    "velodyne": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip",
    # KITTI Odometry calibration / timestamps
    "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip",
    # SemanticKITTI labels (+ poses) for sequences 00-10
    "labels": "https://www.semantic-kitti.org/assets/data_odometry_labels.zip",
}


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TiB"


def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        base_resolved = base.resolve()
        target_resolved = target.resolve()
    except FileNotFoundError:
        # If the path doesn't exist yet, resolve its parent
        base_resolved = base.resolve()
        target_resolved = (target.parent.resolve() / target.name)
    return os.path.commonpath([str(base_resolved)]) == os.path.commonpath(
        [str(base_resolved), str(target_resolved)]
    )


def _head_content_length(url: str, timeout_s: int = 30) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    bytes_written: int
    total_bytes: int | None
    resumed: bool


def download_with_resume(
    url: str,
    dest_zip: Path,
    *,
    retries: int = 5,
    chunk_bytes: int = 8 * 1024 * 1024,
    quiet: bool = False,
) -> DownloadResult:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_zip.with_suffix(dest_zip.suffix + ".part")

    total = _head_content_length(url)
    attempt = 0
    last_error: Exception | None = None

    while attempt <= retries:
        attempt += 1
        try:
            existing = part_path.stat().st_size if part_path.exists() else 0

            headers = {}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", None)
                can_resume = status == 206 and existing > 0
                if existing > 0 and not can_resume:
                    # Server didn't honor Range; restart.
                    existing = 0

                mode = "ab" if can_resume else "wb"
                bytes_written = existing
                t0 = time.monotonic()
                last_report = t0
                last_bytes = bytes_written

                if not quiet:
                    if total is not None:
                        resume_note = " (resume)" if can_resume else ""
                        print(
                            f"Downloading {dest_zip.name}{resume_note}: {_human_bytes(bytes_written)}/{_human_bytes(total)}",
                            flush=True,
                        )
                    else:
                        resume_note = " (resume)" if can_resume else ""
                        print(
                            f"Downloading {dest_zip.name}{resume_note}: {_human_bytes(bytes_written)}",
                            flush=True,
                        )

                with open(part_path, mode) as f:
                    while True:
                        chunk = resp.read(chunk_bytes)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_written += len(chunk)

                        now = time.monotonic()
                        if quiet:
                            continue
                        if now - last_report >= 10.0:
                            dt = max(1e-6, now - last_report)
                            speed = (bytes_written - last_bytes) / dt
                            if total is not None:
                                pct = 100.0 * bytes_written / total
                                print(
                                    f"  ... {pct:5.1f}%  {_human_bytes(bytes_written)}/{_human_bytes(total)}  ({_human_bytes(int(speed))}/s)",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"  ... {_human_bytes(bytes_written)}  ({_human_bytes(int(speed))}/s)",
                                    flush=True,
                                )
                            last_report = now
                            last_bytes = bytes_written

            # Atomic-ish finalize
            part_path.replace(dest_zip)
            if not quiet:
                if total is not None:
                    print(f"Finished {dest_zip.name}: {_human_bytes(total)}", flush=True)
                else:
                    print(f"Finished {dest_zip.name}: {_human_bytes(dest_zip.stat().st_size)}", flush=True)
            return DownloadResult(
                path=dest_zip,
                bytes_written=dest_zip.stat().st_size,
                total_bytes=total,
                resumed=can_resume,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_error = e
            if attempt > retries:
                break
            wait_s = min(60, 2**attempt)
            if not quiet:
                print(f"Download error ({type(e).__name__}): {e}. Retrying in {wait_s}s...", file=sys.stderr)
            time.sleep(wait_s)

    raise RuntimeError(f"Failed to download {url} after {retries} retries") from last_error


def _safe_extract_zip(zip_path: Path, dst_dir: Path, *, use_unzip: bool = False) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    if use_unzip:
        unzip = shutil.which("unzip")
        if unzip is None:
            raise RuntimeError("Requested --extractor unzip but `unzip` was not found in PATH.")
        subprocess.run([unzip, "-o", str(zip_path), "-d", str(dst_dir)], check=True)
        return

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.filename.startswith("/") or member.filename.startswith("\\"):
                raise RuntimeError(f"Refusing to extract absolute path from zip: {member.filename}")
            target = dst_dir / member.filename
            if not _is_within_directory(dst_dir, target):
                raise RuntimeError(f"Refusing to extract path outside destination: {member.filename}")
        zf.extractall(dst_dir)


def _marker_path(data_dir: Path, zip_path: Path) -> Path:
    # Keep it simple + stable across platforms.
    name = zip_path.name.replace(".", "_")
    return data_dir / f".extracted_{name}"


def _has_marker(data_dir: Path, zip_path: Path) -> bool:
    return _marker_path(data_dir, zip_path).exists()


def _write_marker(data_dir: Path, zip_path: Path) -> None:
    marker = _marker_path(data_dir, zip_path)
    marker.write_text(f"{zip_path.name}\n{zip_path.stat().st_size}\n{int(zip_path.stat().st_mtime)}\n")


def _default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "dataset" / "semantickitti"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download + extract SemanticKITTI (KITTI Odometry Velodyne + SemanticKITTI labels)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Where to store zip files and extracted `dataset/` (default: repo/data/dataset/semantickitti).",
    )
    parser.add_argument(
        "--only",
        choices=["all", "velodyne", "labels", "calib"],
        default="all",
        help="Download/extract only a subset (default: all).",
    )
    parser.add_argument("--skip-download", action="store_true", help="Assume zip files already exist.")
    parser.add_argument("--skip-extract", action="store_true", help="Only download zips; do not extract.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract even if files/markers already exist.",
    )
    parser.add_argument(
        "--extractor",
        choices=["python", "unzip"],
        default="python",
        help="Extraction backend (default: python).",
    )
    parser.add_argument("--retries", type=int, default=5, help="Retry count for downloads (default: 5).")
    parser.add_argument(
        "--chunk-mb", type=int, default=8, help="Download chunk size in MiB (default: 8)."
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="Delete zip files after successful extraction (saves disk space).",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    components = ["velodyne", "labels", "calib"] if args.only == "all" else [args.only]
    zip_paths: dict[str, Path] = {
        c: data_dir / Path(URLS[c]).name for c in components
    }

    if not args.quiet:
        print(f"Data dir: {data_dir}", flush=True)

    dataset_sequences = data_dir / "dataset" / "sequences"
    dataset_exists = dataset_sequences.exists()

    if args.force:
        for comp in components:
            zip_path = zip_paths[comp]
            _marker_path(data_dir, zip_path).unlink(missing_ok=True)
            zip_path.unlink(missing_ok=True)
            zip_path.with_suffix(zip_path.suffix + ".part").unlink(missing_ok=True)
        dataset_exists = dataset_sequences.exists()

    # Download
    if not args.skip_download:
        for comp in components:
            zip_path = zip_paths[comp]
            if zip_path.exists():
                if not args.quiet:
                    print(f"Already exists: {zip_path.name} ({_human_bytes(zip_path.stat().st_size)})", flush=True)
                continue
            if _has_marker(data_dir, zip_path):
                if dataset_exists:
                    if not args.quiet:
                        print(f"Already extracted (marker), skipping download: {zip_path.name}", flush=True)
                    continue
            download_with_resume(
                URLS[comp],
                zip_path,
                retries=max(0, args.retries),
                chunk_bytes=max(1, args.chunk_mb) * 1024 * 1024,
                quiet=args.quiet,
            )
    else:
        # Validate presence
        missing = [p.name for p in zip_paths.values() if not p.exists()]
        if missing:
            print(
                f"--skip-download set, but missing zip(s) in {data_dir}: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 2

    # Extract
    if not args.skip_extract:
        use_unzip = args.extractor == "unzip"
        for comp in components:
            zip_path = zip_paths[comp]
            if _has_marker(data_dir, zip_path):
                if dataset_exists:
                    if not args.quiet:
                        print(f"Already extracted (marker): {zip_path.name}", flush=True)
                    continue
            if not args.quiet:
                print(f"Extracting {zip_path.name} ...", flush=True)
            _safe_extract_zip(zip_path, data_dir, use_unzip=use_unzip)
            _write_marker(data_dir, zip_path)
            if args.delete_zips:
                zip_path.unlink(missing_ok=True)
                if not args.quiet:
                    print(f"Deleted {zip_path.name}", flush=True)

    dataset_exists = dataset_sequences.exists()
    if dataset_sequences.exists():
        if not args.quiet:
            print(f"Ready: {dataset_sequences}", flush=True)
    else:
        if not args.quiet:
            print("Note: expected extracted path `dataset/sequences` was not found.", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
