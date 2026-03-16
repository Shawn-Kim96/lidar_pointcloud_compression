from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.recon_pointcloud_export import (  # noqa: E402
    load_kitti_bin,
    project_unproject_identity_points,
    save_kitti_bin,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a KITTI_Identity dataset by applying range-image projection/unprojection without a learned codec."
    )
    parser.add_argument("--source_root", type=str, required=True, help="Official KITTI detection root.")
    parser.add_argument("--output_root", type=str, required=True, help="Output root for KITTI_Identity.")
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--fov_up_deg", type=float, default=3.0)
    parser.add_argument("--fov_down_deg", type=float, default=-25.0)
    parser.add_argument("--range_threshold", type=float, default=1e-3)
    parser.add_argument(
        "--unprojection_mode",
        type=str,
        default="decoded_xyz",
        choices=("decoded_xyz", "ray"),
        help="How to reconstruct xyz from the projected range image.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing velodyne files.")
    parser.add_argument("--log_every", type=int, default=250, help="Progress logging interval.")
    return parser.parse_args()


def ensure_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        raise RuntimeError(f"Refusing to replace mismatched symlink: {dst} -> {dst.resolve()} (expected {src})")
    if dst.exists():
        if dst.resolve() == src.resolve():
            return
        raise RuntimeError(f"Refusing to overwrite existing path: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def validate_source_root(root: Path) -> None:
    required = [
        root / "ImageSets",
        root / "training",
        root / "training" / "velodyne",
        root / "training" / "label_2",
        root / "training" / "calib",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required KITTI paths:\n" + "\n".join(missing))


def replicate_static_layout(source_root: Path, output_root: Path) -> None:
    ensure_symlink(source_root / "ImageSets", output_root / "ImageSets")

    training_src = source_root / "training"
    training_out = output_root / "training"
    training_out.mkdir(parents=True, exist_ok=True)

    for name in ("label_2", "calib", "image_2", "image_3", "planes"):
        ensure_symlink(training_src / name, training_out / name)

    for top_name in ("testing",):
        ensure_symlink(source_root / top_name, output_root / top_name)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()

    validate_source_root(source_root)
    replicate_static_layout(source_root, output_root)

    src_velodyne = source_root / "training" / "velodyne"
    out_velodyne = output_root / "training" / "velodyne"
    out_velodyne.mkdir(parents=True, exist_ok=True)

    bin_paths = sorted(src_velodyne.glob("*.bin"))
    if not bin_paths:
        raise FileNotFoundError(f"No KITTI velodyne .bin files found under {src_velodyne}")

    converted = 0
    skipped = 0
    for idx, src_bin in enumerate(bin_paths, start=1):
        dst_bin = out_velodyne / src_bin.name
        if dst_bin.exists() and not args.overwrite:
            skipped += 1
            if idx % max(int(args.log_every), 1) == 0:
                print(f"[identity-export] {idx}/{len(bin_paths)} converted={converted} skipped={skipped}", flush=True)
            continue

        raw_points = load_kitti_bin(src_bin)
        identity_points = project_unproject_identity_points(
            raw_points,
            img_h=int(args.img_h),
            img_w=int(args.img_w),
            fov_up_deg=float(args.fov_up_deg),
            fov_down_deg=float(args.fov_down_deg),
            range_threshold=float(args.range_threshold),
            unprojection_mode=str(args.unprojection_mode),
        )
        save_kitti_bin(dst_bin, identity_points)
        converted += 1

        if idx % max(int(args.log_every), 1) == 0:
            print(f"[identity-export] {idx}/{len(bin_paths)} converted={converted} skipped={skipped}", flush=True)

    meta = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "img_h": int(args.img_h),
        "img_w": int(args.img_w),
        "fov_up_deg": float(args.fov_up_deg),
        "fov_down_deg": float(args.fov_down_deg),
        "range_threshold": float(args.range_threshold),
        "unprojection_mode": str(args.unprojection_mode),
        "converted": int(converted),
        "skipped": int(skipped),
        "total_training_bins": int(len(bin_paths)),
    }
    meta_path = output_root / "identity_export_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"[identity-export] done output_root={output_root} converted={converted} "
        f"skipped={skipped} total={len(bin_paths)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
