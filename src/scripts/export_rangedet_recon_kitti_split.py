#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.registry import MODELS  # noqa: E402
import models.compression  # noqa: F401,E402
import models.backbones  # noqa: F401,E402
from utils.recon_pointcloud_export import load_kitti_bin, project_points_to_range_image  # noqa: E402


class _RunConfigLoader(yaml.SafeLoader):
    """YAML loader for experiment config files with python tuple support."""


def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


_RunConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)


def _extract_state_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _latest_epoch_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_epoch_*.pth"))
    if not ckpts:
        final_ckpt = run_dir / "model_final.pth"
        if final_ckpt.exists():
            return final_ckpt
        raise FileNotFoundError(f"No model_epoch_*.pth or model_final.pth found in {run_dir}")

    def _epoch_num(path: Path) -> int:
        m = re.search(r"model_epoch_(\d+)\.pth$", path.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=_epoch_num)
    return ckpts[-1]


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in run_dir: {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=_RunConfigLoader)
    if not isinstance(config, dict):
        raise ValueError(f"Unexpected config structure in {cfg_path}")
    return config


def _load_compression_model(run_dir: Path, checkpoint: Optional[str], device: torch.device):
    config = _load_run_config(run_dir)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = (run_dir / checkpoint).resolve() if (run_dir / checkpoint).exists() else ckpt_path.resolve()
    else:
        ckpt_path = _latest_epoch_checkpoint(run_dir).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Compression checkpoint not found: {ckpt_path}")

    model = MODELS.build(config["model"]).to(device).eval()
    payload = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(_extract_state_dict(payload), strict=False)
    return model, ckpt_path


def _resolve_lidar_bin(kitti_root: Path, sample_id: str, lidar_subdir: str) -> Path:
    path = kitti_root / lidar_subdir / "velodyne" / f"{sample_id}.bin"
    if path.exists():
        return path
    alt = kitti_root / "testing" / "velodyne" / f"{sample_id}.bin"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find lidar bin for sample={sample_id} under {kitti_root}")


@torch.no_grad()
def _reconstruct_range_image(
    model: torch.nn.Module,
    device: torch.device,
    points_xyzi: np.ndarray,
    *,
    img_h: int,
    img_w: int,
    fov_up_deg: float,
    fov_down_deg: float,
    quantize: bool,
    noise_std: float,
) -> np.ndarray:
    data_5ch, _ = project_points_to_range_image(
        points_xyzi,
        img_h=img_h,
        img_w=img_w,
        fov_up_deg=fov_up_deg,
        fov_down_deg=fov_down_deg,
    )
    x = torch.from_numpy(data_5ch).unsqueeze(0).to(device)
    raw_points_t = torch.from_numpy(np.asarray(points_xyzi[:, :4], dtype=np.float32)).unsqueeze(0).to(device)
    raw_point_counts_t = torch.tensor([points_xyzi.shape[0]], device=device, dtype=torch.long)
    recon, _ = model(
        x,
        noise_std=float(noise_std),
        quantize=bool(quantize),
        raw_points=raw_points_t,
        raw_point_counts=raw_point_counts_t,
    )
    return recon.squeeze(0).detach().cpu().numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RangeDet KITTI split using reconstructed range images from a codec run."
    )
    parser.add_argument("--run_dir", type=Path, required=True, help="Compression run dir with config.yaml and checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path/name; default latest in run_dir.")
    parser.add_argument("--kitti_root", type=Path, required=True, help="Official KITTI root (training/testing folders).")
    parser.add_argument(
        "--source_rangedet_root",
        type=Path,
        default=REPO_ROOT / "data" / "dataset" / "rangedet_kitti_hq",
        help="Existing RangeDet dataset root used only to read source roidb/npz metadata.",
    )
    parser.add_argument("--split", type=str, default="validation", help="RangeDet split folder to export (e.g., validation).")
    parser.add_argument(
        "--source_roidb",
        type=Path,
        default=None,
        help="Optional explicit source roidb file. Defaults to <source_rangedet_root>/<split>/part-0000.roidb",
    )
    parser.add_argument("--output_root", type=Path, required=True, help="Output RangeDet dataset root.")
    parser.add_argument("--output_npz_subdir", type=str, default="npz_trainval")
    parser.add_argument(
        "--lidar_subdir",
        type=str,
        default="training",
        choices=("training", "testing"),
        help="KITTI subdir for velodyne lookup.",
    )
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--img_w", type=int, default=2048)
    parser.add_argument("--fov_up_deg", type=float, default=3.0)
    parser.add_argument("--fov_down_deg", type=float, default=-25.0)
    parser.add_argument("--range_threshold", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--max_frames", type=int, default=0, help="0 means full split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    kitti_root = args.kitti_root.resolve()
    source_root = args.source_rangedet_root.resolve()
    output_root = args.output_root.resolve()
    output_npz_dir = output_root / args.output_npz_subdir
    output_split_dir = output_root / args.split
    source_roidb = args.source_roidb.resolve() if args.source_roidb else (source_root / args.split / "part-0000.roidb")

    if not source_roidb.exists():
        raise FileNotFoundError(f"Source roidb not found: {source_roidb}")
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    output_npz_dir.mkdir(parents=True, exist_ok=True)
    output_split_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable.")
    device = torch.device(args.device)

    model, ckpt_path = _load_compression_model(run_dir, args.checkpoint or None, device)
    print(f"[export-rangedet] run_dir={run_dir}")
    print(f"[export-rangedet] ckpt={ckpt_path}")
    print(f"[export-rangedet] source_roidb={source_roidb}")
    print(f"[export-rangedet] output_root={output_root}")
    print(f"[export-rangedet] device={device}")

    with source_roidb.open("rb") as f:
        roidb = pickle.load(f, encoding="latin1")
    if not isinstance(roidb, list):
        raise ValueError(f"Unexpected roidb type: {type(roidb).__name__}")

    total = len(roidb)
    limit = args.max_frames if args.max_frames > 0 else total
    roidb = roidb[:limit]
    print(f"[export-rangedet] records={len(roidb)} (of {total})")

    new_roidb = []
    for idx, record in enumerate(roidb):
        src_url = Path(record["pc_url"])
        sample_id = src_url.stem
        lidar_bin = _resolve_lidar_bin(kitti_root, sample_id, args.lidar_subdir)
        raw_points = load_kitti_bin(lidar_bin)

        recon_5ch = _reconstruct_range_image(
            model,
            device,
            raw_points,
            img_h=args.img_h,
            img_w=args.img_w,
            fov_up_deg=args.fov_up_deg,
            fov_down_deg=args.fov_down_deg,
            quantize=args.quantize,
            noise_std=args.noise_std,
        )

        src_npz = np.load(str(src_url))
        inclination = src_npz["inclination"].astype(np.float32)
        azimuth = src_npz["azimuth"].astype(np.float32)

        range_image = np.stack(
            [
                recon_5ch[0],
                recon_5ch[1],
                np.zeros_like(recon_5ch[0], dtype=np.float32),
                np.zeros_like(recon_5ch[0], dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32)
        pc_vehicle_frame = np.stack([recon_5ch[2], recon_5ch[3], recon_5ch[4]], axis=-1).astype(np.float32)
        range_image_mask = (recon_5ch[0] > float(args.range_threshold))

        dst_npz = output_npz_dir / f"{sample_id}.npz"
        np.savez(
            str(dst_npz),
            range_image=range_image,
            pc_vehicle_frame=pc_vehicle_frame,
            inclination=inclination,
            azimuth=azimuth,
            range_image_mask=range_image_mask,
        )

        rec = copy.deepcopy(record)
        rec["pc_url"] = str(dst_npz.resolve())
        new_roidb.append(rec)

        if idx % 200 == 0 or idx + 1 == len(roidb):
            print(f"[export-rangedet] {idx + 1}/{len(roidb)}")

    part_path = output_split_dir / "part-0000.roidb"
    with part_path.open("wb") as f:
        pickle.dump(new_roidb, f)
    split_path = output_root / f"{args.split}.roidb"
    with split_path.open("wb") as f:
        pickle.dump(new_roidb, f)

    meta = {
        "source_roidb": str(source_roidb),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "records": len(new_roidb),
        "img_h": args.img_h,
        "img_w": args.img_w,
        "fov_up_deg": args.fov_up_deg,
        "fov_down_deg": args.fov_down_deg,
        "quantize": bool(args.quantize),
        "noise_std": float(args.noise_std),
        "output_npz_subdir": args.output_npz_subdir,
        "split": args.split,
        "lidar_subdir": args.lidar_subdir,
    }
    with (output_root / "recon_export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[export-rangedet] wrote roidb: {part_path}")
    print(f"[export-rangedet] wrote meta: {output_root / 'recon_export_meta.json'}")


if __name__ == "__main__":
    main()
