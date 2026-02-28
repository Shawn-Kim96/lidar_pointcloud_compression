from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_ROI_CLASSES = {"Car", "Pedestrian", "Cyclist"}


def _read_imageset_ids(imageset_file: Path) -> List[str]:
    ids: List[str] = []
    for line in imageset_file.read_text(encoding="utf-8").splitlines():
        sid = line.strip()
        if not sid:
            continue
        ids.append(sid)
    return ids


def _read_calib(calib_path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for line in calib_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        vals = [float(x) for x in value.strip().split()]
        data[key.strip()] = np.asarray(vals, dtype=np.float32)
    return data


def _pick_calib_matrix(calib: Dict[str, np.ndarray], key_candidates: Sequence[str], shape: tuple[int, int]) -> np.ndarray:
    for key in key_candidates:
        if key in calib:
            arr = calib[key].reshape(shape)
            return arr
    for key, value in calib.items():
        if not key_candidates:
            continue
        if key.startswith(key_candidates[0]):
            arr = value.reshape(shape)
            return arr
    raise KeyError(f"Missing calibration keys {key_candidates}")


class KittiObjectRangeDataset(Dataset):
    """
    KITTI object dataset loader that projects point clouds to a range image and
    builds a weak ROI mask from 2D detection boxes (P2 projection).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        imageset_file: Optional[str] = None,
        config: Optional[Dict[str, float]] = None,
        return_azimuth: bool = False,
        return_roi_mask: bool = True,
        roi_classes: Optional[Sequence[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.return_azimuth = return_azimuth
        self.return_roi_mask = return_roi_mask
        self.roi_classes = set(roi_classes) if roi_classes else set(DEFAULT_ROI_CLASSES)

        self.training_dir = self.root_dir / "training"
        self.velo_dir = self.training_dir / "velodyne"
        self.label_dir = self.training_dir / "label_2"
        self.calib_dir = self.training_dir / "calib"
        self.image_sets_dir = self.root_dir / "ImageSets"

        if imageset_file:
            imageset_path = Path(imageset_file)
        else:
            imageset_path = self.image_sets_dir / f"{split}.txt"
        if not imageset_path.exists():
            raise FileNotFoundError(f"ImageSets file not found: {imageset_path}")

        raw_ids = _read_imageset_ids(imageset_path)
        self.sample_ids = [sid.zfill(6) for sid in raw_ids]
        self.sample_ids = [sid for sid in self.sample_ids if (self.velo_dir / f"{sid}.bin").exists()]
        if not self.sample_ids:
            raise RuntimeError(f"No valid velodyne samples found under {self.velo_dir} for split={split}")

        if config is None:
            self.fov_up = 3.0
            self.fov_down = -25.0
            self.H = 64
            self.W = 1024
        else:
            self.fov_up = float(config.get("fov_up", 3.0))
            self.fov_down = float(config.get("fov_down", -25.0))
            self.H = int(config.get("img_height", 64))
            self.W = int(config.get("img_width", 1024))

        self._calib_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._label_cache: Dict[str, List[List[str]]] = {}

        print(f"Loaded {len(self.sample_ids)} KITTI frames from split={split} at {self.root_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_calib(self, sample_id: str) -> Dict[str, np.ndarray]:
        if sample_id in self._calib_cache:
            return self._calib_cache[sample_id]
        calib_path = self.calib_dir / f"{sample_id}.txt"
        calib = _read_calib(calib_path)
        self._calib_cache[sample_id] = calib
        return calib

    def _load_labels(self, sample_id: str) -> List[List[str]]:
        if sample_id in self._label_cache:
            return self._label_cache[sample_id]
        label_path = self.label_dir / f"{sample_id}.txt"
        if not label_path.exists():
            self._label_cache[sample_id] = []
            return []
        rows: List[List[str]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            toks = line.strip().split()
            if len(toks) >= 8:
                rows.append(toks)
        self._label_cache[sample_id] = rows
        return rows

    def _project_points_to_image(self, points_xyz: np.ndarray, calib: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p2 = _pick_calib_matrix(calib, ("P2",), (3, 4))
        tr = _pick_calib_matrix(calib, ("Tr_velo_to_cam",), (3, 4))
        r0 = _pick_calib_matrix(calib, ("R0_rect",), (3, 3))

        n = points_xyz.shape[0]
        pts_h = np.concatenate([points_xyz, np.ones((n, 1), dtype=np.float32)], axis=1)  # [N,4]

        tr4 = np.eye(4, dtype=np.float32)
        tr4[:3, :4] = tr
        r04 = np.eye(4, dtype=np.float32)
        r04[:3, :3] = r0

        pts_cam_h = (r04 @ tr4 @ pts_h.T).T  # [N,4]
        z = pts_cam_h[:, 2]
        valid = z > 1e-3

        img_h = (p2 @ pts_cam_h.T).T  # [N,3]
        u = img_h[:, 0] / np.clip(img_h[:, 2], a_min=1e-6, a_max=None)
        v = img_h[:, 1] / np.clip(img_h[:, 2], a_min=1e-6, a_max=None)
        return u, v, valid

    def _build_point_roi_mask(self, points_xyzi: np.ndarray, labels: List[List[str]], calib: Dict[str, np.ndarray]) -> np.ndarray:
        roi = np.zeros((points_xyzi.shape[0],), dtype=np.float32)
        if not labels:
            return roi

        u, v, valid = self._project_points_to_image(points_xyzi[:, :3], calib)
        for toks in labels:
            cls = toks[0]
            if cls not in self.roi_classes:
                continue
            x1, y1, x2, y2 = map(float, toks[4:8])
            inside = (
                valid
                & (u >= x1)
                & (u <= x2)
                & (v >= y1)
                & (v <= y2)
            )
            roi[inside] = 1.0
        return roi

    def do_range_projection(self, points: np.ndarray, point_roi_mask: Optional[np.ndarray] = None):
        fov_up = self.fov_up / 180.0 * np.pi
        fov_down = self.fov_down / 180.0 * np.pi
        fov = fov_up - fov_down

        depth = np.linalg.norm(points[:, :3], 2, axis=1)
        depth = np.clip(depth, a_min=1e-6, a_max=None)

        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        intensity = points[:, 3]

        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + np.abs(fov_down)) / fov

        proj_x *= self.W
        proj_y *= self.H

        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)

        order = np.argsort(depth)[::-1]
        depth = depth[order]
        points = points[order]
        intensity = intensity[order]
        yaw = yaw[order]
        if point_roi_mask is not None:
            point_roi_mask = point_roi_mask[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        proj_range = np.full((self.H, self.W), -1, dtype=np.float32)
        proj_xyz = np.full((self.H, self.W, 3), 0, dtype=np.float32)
        proj_intensity = np.full((self.H, self.W), 0, dtype=np.float32)
        proj_azimuth = np.full((self.H, self.W), 0, dtype=np.float32)
        proj_mask = np.zeros((self.H, self.W), dtype=np.float32)
        proj_roi_mask = np.zeros((self.H, self.W), dtype=np.float32)

        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points[:, :3]
        proj_intensity[proj_y, proj_x] = intensity
        proj_azimuth[proj_y, proj_x] = yaw
        proj_mask[proj_y, proj_x] = 1.0
        if point_roi_mask is not None:
            proj_roi_mask[proj_y, proj_x] = point_roi_mask.astype(np.float32)

        return proj_range, proj_xyz, proj_intensity, proj_azimuth, proj_mask, proj_roi_mask

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        velo_path = self.velo_dir / f"{sample_id}.bin"
        raw = np.fromfile(str(velo_path), dtype=np.float32)
        if raw.size % 4 == 0:
            scan = raw.reshape((-1, 4))
        elif raw.size % 3 == 0:
            xyz = raw.reshape((-1, 3))
            intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
            scan = np.concatenate([xyz, intensity], axis=1)
        else:
            raise ValueError(f"Invalid KITTI point file shape: {velo_path} (float_count={raw.size})")

        point_roi_mask = None
        if self.return_roi_mask:
            labels = self._load_labels(sample_id)
            calib = self._load_calib(sample_id)
            point_roi_mask = self._build_point_roi_mask(scan, labels, calib)

        proj_range, proj_xyz, proj_intensity, proj_azimuth, proj_mask, proj_roi_mask = self.do_range_projection(
            scan, point_roi_mask=point_roi_mask
        )

        data = np.zeros((5, self.H, self.W), dtype=np.float32)
        data[0] = proj_range
        data[1] = proj_intensity
        data[2] = proj_xyz[:, :, 0]
        data[3] = proj_xyz[:, :, 1]
        data[4] = proj_xyz[:, :, 2]

        outputs = [torch.from_numpy(data), torch.from_numpy(proj_mask)]
        if self.return_roi_mask:
            outputs.append(torch.from_numpy(proj_roi_mask).unsqueeze(0))
        if self.return_azimuth:
            outputs.append(torch.from_numpy(proj_azimuth))
        return tuple(outputs)
