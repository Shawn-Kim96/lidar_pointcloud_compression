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


OBJECT_URLS: dict[str, str] = {
    "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
    "image_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "label_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "velodyne": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
    "image_3": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip",
    "devkit": "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip",
}


IMAGESET_URLS: dict[str, str] = {
    "train.txt": "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets/train.txt",
    "val.txt": "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets/val.txt",
    "trainval.txt": "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets/trainval.txt",
    "test.txt": "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets/test.txt",
}


def _list_bin_ids(bin_dir: Path) -> List[str]:
    if not bin_dir.exists():
        return []
    return sorted(p.stem for p in bin_dir.glob("*.bin"))


def _write_trainval_from_train_val(imagesets_dir: Path) -> None:
    train = imagesets_dir / "train.txt"
    val = imagesets_dir / "val.txt"
    if not train.exists() or not val.exists():
        return
    ids = sorted(
        {
            line.strip()
            for src in (train, val)
            for line in src.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    (imagesets_dir / "trainval.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def _write_test_from_dataset(data_dir: Path, imagesets_dir: Path) -> None:
    ids = _list_bin_ids(data_dir / "testing" / "velodyne")
    if ids:
        (imagesets_dir / "test.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")


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
    dest_path: Path,
    *,
    retries: int = 5,
    chunk_bytes: int = 8 * 1024 * 1024,
    quiet: bool = False,
) -> DownloadResult:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    total = _head_content_length(url)
    attempt = 0
    last_error: Exception | None = None

    while attempt <= retries:
        attempt += 1
        can_resume = False
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
                    existing = 0

                mode = "ab" if can_resume else "wb"
                bytes_written = existing
                last_report = time.monotonic()
                last_bytes = bytes_written

                if not quiet:
                    resume_note = " (resume)" if can_resume else ""
                    if total is not None:
                        print(
                            f"Downloading {dest_path.name}{resume_note}: {_human_bytes(bytes_written)}/{_human_bytes(total)}",
                            flush=True,
                        )
                    else:
                        print(
                            f"Downloading {dest_path.name}{resume_note}: {_human_bytes(bytes_written)}",
                            flush=True,
                        )

                with open(part_path, mode) as f:
                    while True:
                        chunk = resp.read(chunk_bytes)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_written += len(chunk)

                        if quiet:
                            continue
                        now = time.monotonic()
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

            part_path.replace(dest_path)
            if not quiet:
                size = dest_path.stat().st_size
                print(f"Finished {dest_path.name}: {_human_bytes(size)}", flush=True)
            return DownloadResult(
                path=dest_path,
                bytes_written=dest_path.stat().st_size,
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
            raise RuntimeError("Requested --extractor unzip but `unzip` is not found in PATH.")
        subprocess.run([unzip, "-o", str(zip_path), "-d", str(dst_dir)], check=True)
        return

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.filename.startswith("/") or member.filename.startswith("\\"):
                raise RuntimeError(f"Refusing to extract absolute path: {member.filename}")
            target = dst_dir / member.filename
            if not _is_within_directory(dst_dir, target):
                raise RuntimeError(f"Refusing to extract path outside destination: {member.filename}")
        zf.extractall(dst_dir)


def _marker_path(data_dir: Path, zip_path: Path) -> Path:
    name = zip_path.name.replace(".", "_")
    return data_dir / f".extracted_{name}"


def _has_marker(data_dir: Path, zip_path: Path) -> bool:
    return _marker_path(data_dir, zip_path).exists()


def _write_marker(data_dir: Path, zip_path: Path) -> None:
    marker = _marker_path(data_dir, zip_path)
    marker.write_text(f"{zip_path.name}\n{zip_path.stat().st_size}\n{int(zip_path.stat().st_mtime)}\n")


def _default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "dataset" / "kitti3dobject"


def _write_imagesets(data_dir: Path, force: bool, quiet: bool) -> None:
    imagesets_dir = data_dir / "ImageSets"
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in IMAGESET_URLS.items():
        dst = imagesets_dir / fname
        if dst.exists() and not force:
            if not quiet:
                print(f"ImageSet exists, skipping: {dst.name}", flush=True)
            continue
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read().decode("utf-8")
            dst.write_text(content, encoding="utf-8")
            if not quiet:
                print(f"Wrote ImageSet: {dst.name} ({len(content.splitlines())} lines)", flush=True)
        except urllib.error.HTTPError as e:
            # OpenPCDet upstream occasionally misses trainval.txt on raw mirror.
            # Fall back to deterministic local synthesis instead of aborting.
            if fname == "trainval.txt":
                _write_trainval_from_train_val(imagesets_dir)
                if (imagesets_dir / "trainval.txt").exists():
                    if not quiet:
                        n = len((imagesets_dir / "trainval.txt").read_text(encoding="utf-8").splitlines())
                        print(f"Generated ImageSet fallback: trainval.txt ({n} lines) [HTTP {e.code}]", flush=True)
                    continue
            if fname == "test.txt":
                _write_test_from_dataset(data_dir, imagesets_dir)
                if (imagesets_dir / "test.txt").exists():
                    if not quiet:
                        n = len((imagesets_dir / "test.txt").read_text(encoding="utf-8").splitlines())
                        print(f"Generated ImageSet fallback: test.txt ({n} lines) [HTTP {e.code}]", flush=True)
                    continue
            raise


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download + extract KITTI 3D Object Detection dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Destination root (default: repo/data/dataset/kitti3dobject).",
    )
    parser.add_argument(
        "--only",
        choices=["essentials", "all", "calib", "image_2", "label_2", "velodyne", "image_3", "devkit"],
        default="essentials",
        help="Dataset subset to download (default: essentials).",
    )
    parser.add_argument("--skip-download", action="store_true", help="Assume zip files already exist.")
    parser.add_argument("--skip-extract", action="store_true", help="Only download zips, do not extract.")
    parser.add_argument("--skip-imagesets", action="store_true", help="Do not fetch ImageSets split files.")
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract.")
    parser.add_argument("--retries", type=int, default=5, help="Retry count for downloads.")
    parser.add_argument("--chunk-mb", type=int, default=8, help="Download chunk size in MiB.")
    parser.add_argument(
        "--extractor",
        choices=["python", "unzip"],
        default="python",
        help="Extraction backend (default: python).",
    )
    parser.add_argument("--delete-zips", action="store_true", help="Delete zip files after extraction.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output.")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.only == "essentials":
        components = ["calib", "image_2", "label_2", "velodyne"]
    elif args.only == "all":
        components = ["calib", "image_2", "label_2", "velodyne", "image_3", "devkit"]
    else:
        components = [args.only]

    zip_paths = {c: data_dir / Path(OBJECT_URLS[c]).name for c in components}

    if not args.quiet:
        print(f"Data dir: {data_dir}", flush=True)
        print(f"Components: {', '.join(components)}", flush=True)

    if args.force:
        for comp in components:
            zp = zip_paths[comp]
            _marker_path(data_dir, zp).unlink(missing_ok=True)
            zp.unlink(missing_ok=True)
            zp.with_suffix(zp.suffix + ".part").unlink(missing_ok=True)

    if not args.skip_download:
        for comp in components:
            zip_path = zip_paths[comp]
            if zip_path.exists() and not args.force:
                if not args.quiet:
                    print(f"Already exists: {zip_path.name} ({_human_bytes(zip_path.stat().st_size)})", flush=True)
                continue
            download_with_resume(
                OBJECT_URLS[comp],
                zip_path,
                retries=max(0, args.retries),
                chunk_bytes=max(1, args.chunk_mb) * 1024 * 1024,
                quiet=args.quiet,
            )
    else:
        missing = [zp.name for zp in zip_paths.values() if not zp.exists()]
        if missing:
            if args.skip_extract:
                required = [
                    data_dir / "training" / "velodyne",
                    data_dir / "training" / "label_2",
                    data_dir / "training" / "calib",
                    data_dir / "testing" / "velodyne",
                    data_dir / "testing" / "calib",
                ]
                if all(p.exists() for p in required):
                    if not args.quiet:
                        print(
                            "--skip-download: zip files missing, but extracted KITTI layout exists. "
                            "Continuing without archives.",
                            flush=True,
                        )
                else:
                    print(f"--skip-download set but missing: {', '.join(missing)}", file=sys.stderr)
                    return 2
            else:
                print(f"--skip-download set but missing: {', '.join(missing)}", file=sys.stderr)
                return 2

    if not args.skip_extract:
        use_unzip = args.extractor == "unzip"
        for comp in components:
            zip_path = zip_paths[comp]
            if _has_marker(data_dir, zip_path) and not args.force:
                if not args.quiet:
                    print(f"Already extracted (marker): {zip_path.name}", flush=True)
                continue
            if not args.quiet:
                print(f"Extracting {zip_path.name} ...", flush=True)
            _safe_extract_zip(zip_path, data_dir, use_unzip=use_unzip)
            _write_marker(data_dir, zip_path)
            if args.delete_zips:
                zip_path.unlink(missing_ok=True)
                zip_path.with_suffix(zip_path.suffix + ".part").unlink(missing_ok=True)
    else:
        if not args.quiet:
            print("Skipping extraction by request.", flush=True)

    if not args.skip_imagesets:
        if not args.quiet:
            print("Fetching ImageSets split files ...", flush=True)
        _write_imagesets(data_dir, force=args.force, quiet=args.quiet)

    required = [data_dir / "training" / "velodyne", data_dir / "training" / "calib", data_dir / "training" / "label_2"]
    missing_layout = [str(p) for p in required if not p.exists()]
    if missing_layout:
        print("Warning: expected KITTI object layout not fully present.", file=sys.stderr)
        for p in missing_layout:
            print(f"  missing: {p}", file=sys.stderr)

    if not args.quiet:
        print("Done.", flush=True)
        print(f"KITTI 3D object root: {data_dir}", flush=True)
        print("Expected for OpenPCDet:", flush=True)
        print(f"  {data_dir}/training/velodyne/*.bin", flush=True)
        print(f"  {data_dir}/training/label_2/*.txt", flush=True)
        print(f"  {data_dir}/training/calib/*.txt", flush=True)
        print(f"  {data_dir}/ImageSets/train.txt", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
