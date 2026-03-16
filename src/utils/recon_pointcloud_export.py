from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


def load_kitti_bin(bin_path: str | Path) -> np.ndarray:
    path = Path(bin_path)
    if not path.exists():
        raise FileNotFoundError(f"KITTI .bin not found: {path}")
    points = np.fromfile(str(path), dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Unexpected KITTI .bin size (not divisible by 4): {path}")
    return points.reshape(-1, 4)


def save_kitti_bin(bin_path: str | Path, points_xyzi: np.ndarray) -> None:
    path = Path(bin_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points_xyzi, dtype=np.float32).reshape(-1, 4)
    pts.tofile(str(path))


def project_points_to_range_image(
    points_xyzi: np.ndarray,
    img_h: int = 64,
    img_w: int = 1024,
    fov_up_deg: float = 3.0,
    fov_down_deg: float = -25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects point cloud to range image input expected by compression model.

    Returns:
      data_5ch: [5,H,W] = [range, intensity, x, y, z]
      valid_mask: [H,W] in {0,1}
    """
    if points_xyzi.ndim != 2 or points_xyzi.shape[1] < 4:
        raise ValueError("points_xyzi must have shape [N,4+]")

    pts = np.asarray(points_xyzi[:, :4], dtype=np.float32)
    xyz = pts[:, :3]
    intensity = pts[:, 3]

    depth = np.linalg.norm(xyz, ord=2, axis=1)
    depth = np.clip(depth, 1e-6, None)

    scan_x = xyz[:, 0]
    scan_y = xyz[:, 1]
    scan_z = xyz[:, 2]

    fov_up = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    fov = fov_up - fov_down

    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    proj_x = 0.5 * (yaw / np.pi + 1.0)
    proj_y = 1.0 - (pitch + np.abs(fov_down)) / fov
    proj_x *= img_w
    proj_y *= img_h

    proj_x = np.floor(proj_x)
    proj_x = np.minimum(img_w - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(img_h - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)

    # Keep nearest point on collisions.
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    xyz = xyz[order]
    intensity = intensity[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]

    proj_range = np.full((img_h, img_w), -1.0, dtype=np.float32)
    proj_xyz = np.zeros((img_h, img_w, 3), dtype=np.float32)
    proj_intensity = np.zeros((img_h, img_w), dtype=np.float32)
    proj_mask = np.zeros((img_h, img_w), dtype=np.float32)

    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = xyz
    proj_intensity[proj_y, proj_x] = intensity
    proj_mask[proj_y, proj_x] = 1.0

    data = np.zeros((5, img_h, img_w), dtype=np.float32)
    data[0] = proj_range
    data[1] = proj_intensity
    data[2] = proj_xyz[:, :, 0]
    data[3] = proj_xyz[:, :, 1]
    data[4] = proj_xyz[:, :, 2]
    return data, proj_mask


def range_image_to_points_xyzi(
    recon_5ch: np.ndarray,
    *,
    range_threshold: float = 1e-3,
    valid_mask: Optional[np.ndarray] = None,
    unprojection_mode: str = "decoded_xyz",
    fov_up_deg: float = 3.0,
    fov_down_deg: float = -25.0,
) -> np.ndarray:
    """
    Converts reconstructed range-image tensor back to KITTI-style XYZI points.

    Modes:
    - decoded_xyz: trust reconstructed channels [x,y,z,intensity]
    - ray: recover xyz from reconstructed range + pixel angles
    """
    if recon_5ch.shape[0] != 5:
        raise ValueError(f"Expected recon_5ch shape [5,H,W], got {recon_5ch.shape}")

    mode = str(unprojection_mode).strip().lower()
    rng = recon_5ch[0]
    inten = recon_5ch[1]
    valid = (rng > float(range_threshold)) & np.isfinite(rng)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=np.float32) > 0.5

    if valid.sum() == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if mode == "decoded_xyz":
        x = recon_5ch[2]
        y = recon_5ch[3]
        z = recon_5ch[4]
        valid &= np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if valid.sum() == 0:
            return np.zeros((0, 4), dtype=np.float32)
    elif mode == "ray":
        _, img_h, img_w = recon_5ch.shape
        cols = (np.arange(img_w, dtype=np.float32) + 0.5) / float(img_w)
        rows = (np.arange(img_h, dtype=np.float32) + 0.5) / float(img_h)
        yaw = (2.0 * cols - 1.0) * np.pi
        fov_up = np.deg2rad(float(fov_up_deg))
        fov_down = np.deg2rad(float(fov_down_deg))
        pitch = fov_up - rows * (fov_up - fov_down)
        cos_pitch = np.cos(pitch)[:, None]
        sin_pitch = np.sin(pitch)[:, None]
        cos_yaw = np.cos(yaw)[None, :]
        sin_yaw = np.sin(yaw)[None, :]
        x = rng * cos_pitch * cos_yaw
        y = -rng * cos_pitch * sin_yaw
        z = rng * sin_pitch
    else:
        raise ValueError(
            f"Unsupported unprojection_mode='{unprojection_mode}'. "
            "Expected one of: decoded_xyz, ray"
        )

    pts = np.stack([x[valid], y[valid], z[valid], inten[valid]], axis=1).astype(np.float32)
    return pts


def project_unproject_identity_points(
    points_xyzi: np.ndarray,
    *,
    img_h: int = 64,
    img_w: int = 1024,
    fov_up_deg: float = 3.0,
    fov_down_deg: float = -25.0,
    range_threshold: float = 1e-3,
    unprojection_mode: str = "decoded_xyz",
) -> np.ndarray:
    """
    Applies the same projection/unprojection geometry as the codec path, without a learned model.

    This is the "identity" diagnostic baseline used to isolate projection loss from codec loss.
    """
    data_5ch, valid_mask = project_points_to_range_image(
        points_xyzi,
        img_h=img_h,
        img_w=img_w,
        fov_up_deg=fov_up_deg,
        fov_down_deg=fov_down_deg,
    )
    return range_image_to_points_xyzi(
        data_5ch,
        range_threshold=range_threshold,
        valid_mask=valid_mask,
        unprojection_mode=unprojection_mode,
        fov_up_deg=fov_up_deg,
        fov_down_deg=fov_down_deg,
    )


def _normalize_class_name(name: str) -> str:
    return str(name).strip().lower()


def _points_in_box_mask(points_xyz: np.ndarray, box_lidar: np.ndarray) -> np.ndarray:
    """
    Returns mask for points inside a single KITTI/OpenPCDet lidar box.
    box_lidar format: [x, y, z, dx, dy, dz, heading]
    """
    if points_xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    if box_lidar.shape[0] < 7:
        raise ValueError(f"Expected box shape [7], got {box_lidar.shape}")

    cx, cy, cz, dx, dy, dz, heading = [float(v) for v in box_lidar[:7]]
    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        return np.zeros((points_xyz.shape[0],), dtype=bool)

    rel = points_xyz - np.array([cx, cy, cz], dtype=np.float32)[None, :]
    cos_h = float(np.cos(-heading))
    sin_h = float(np.sin(-heading))
    local_x = rel[:, 0] * cos_h - rel[:, 1] * sin_h
    local_y = rel[:, 0] * sin_h + rel[:, 1] * cos_h
    local_z = rel[:, 2]

    hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
    inside = (
        (np.abs(local_x) <= hx)
        & (np.abs(local_y) <= hy)
        & (np.abs(local_z) <= hz)
    )
    return inside


def _dilate_binary_mask(mask_hw: np.ndarray, radius_px: int) -> np.ndarray:
    radius = int(max(0, radius_px))
    if radius == 0:
        return mask_hw.astype(np.float32)
    kernel = 2 * radius + 1
    t = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    # Max-pool acts as binary dilation for {0,1} masks.
    dilated = torch.nn.functional.max_pool2d(t, kernel_size=kernel, stride=1, padding=radius)
    return dilated.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def build_oracle_importance_map_from_gt(
    data_5ch: np.ndarray,
    valid_mask: np.ndarray,
    gt_boxes_lidar: Optional[np.ndarray],
    gt_names: Optional[Sequence[str]] = None,
    include_classes: Optional[Sequence[str]] = None,
    dilate_px: int = 0,
) -> np.ndarray:
    """
    Builds per-pixel oracle importance from GT lidar boxes by marking projected
    points that fall inside selected GT boxes.
    """
    if data_5ch.ndim != 3 or data_5ch.shape[0] != 5:
        raise ValueError(f"Expected data_5ch shape [5,H,W], got {data_5ch.shape}")
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected valid_mask shape [H,W], got {valid_mask.shape}")

    valid = valid_mask > 0.5
    imp = np.zeros(valid_mask.shape, dtype=np.float32)
    if valid.sum() == 0:
        return imp

    if gt_boxes_lidar is None:
        return imp

    boxes = np.asarray(gt_boxes_lidar, dtype=np.float32).reshape(-1, 7)
    if boxes.size == 0:
        return imp

    class_filter = None
    if include_classes is not None:
        class_filter = {_normalize_class_name(x) for x in include_classes if str(x).strip()}
        if len(class_filter) == 0:
            class_filter = None

    names: Optional[np.ndarray] = None
    if gt_names is not None:
        names = np.asarray(gt_names)
        if names.ndim == 0:
            names = names.reshape(1)

    xyz_hw3 = np.stack([data_5ch[2], data_5ch[3], data_5ch[4]], axis=-1)
    pts = xyz_hw3[valid]
    roi_flat = np.zeros((pts.shape[0],), dtype=bool)

    for bi in range(boxes.shape[0]):
        if names is not None and bi < names.shape[0] and class_filter is not None:
            cname = _normalize_class_name(str(names[bi]))
            if cname not in class_filter:
                continue
        inside = _points_in_box_mask(pts, boxes[bi])
        roi_flat |= inside

    imp[valid] = roi_flat.astype(np.float32)
    if int(dilate_px) > 0:
        imp = _dilate_binary_mask(imp, int(dilate_px))
    return imp


def _estimate_code_entropy(codes: torch.Tensor) -> float:
    q = torch.round(codes.detach()).to(torch.int64).clamp(min=0).cpu().reshape(-1)
    if q.numel() == 0:
        return float("nan")
    hist = torch.bincount(q).float()
    p = hist / hist.sum().clamp(min=1.0)
    p = p[p > 0]
    return float((-(p * torch.log2(p))).sum().item())


def estimate_rate_metrics_from_aux(
    aux: Dict[str, torch.Tensor],
    input_hw: Tuple[int, int],
    uniform_bits_fallback: int = 8,
) -> Dict[str, float]:
    """
    Computes per-sample scalar rate proxies from model aux dictionary.
    """
    in_h, in_w = int(input_hw[0]), int(input_hw[1])

    rate_proxy = float("nan")
    eq_bits = float("nan")
    code_entropy = float("nan")
    bpp_eq = float("nan")
    bpp_entropy = float("nan")
    bpp_true = float("nan")
    quantizer_mode = str(aux.get("quantizer_mode", "")).strip().lower()

    if quantizer_mode == "none":
        return {
            "rate_proxy": 0.0,
            "eq_bits": 0.0,
            "code_entropy": 0.0,
            "bpp_eq": 0.0,
            "bpp_entropy": 0.0,
            "bpp_true": 0.0,
        }

    level_map = aux.get("level_map", None)
    codes = aux.get("codes", None)

    if level_map is not None:
        lm = level_map.detach()
        rate_proxy = float(lm.mean().item())
        eq_bits = float(torch.log2(lm.clamp(min=2.0)).mean().item())
    else:
        rate_proxy = float((2 ** int(uniform_bits_fallback)) - 1)
        eq_bits = float(uniform_bits_fallback)

    if codes is not None:
        code_entropy = _estimate_code_entropy(codes)
        c_lat, h_lat, w_lat = int(codes.shape[1]), int(codes.shape[2]), int(codes.shape[3])
    else:
        c_lat, h_lat, w_lat = 5, in_h, in_w

    symbols_per_input_pixel = (c_lat * h_lat * w_lat) / float(in_h * in_w)
    bpp_eq = eq_bits * symbols_per_input_pixel if eq_bits == eq_bits else float("nan")
    bpp_entropy = (
        code_entropy * symbols_per_input_pixel if code_entropy == code_entropy else float("nan")
    )

    return {
        "rate_proxy": rate_proxy,
        "eq_bits": eq_bits,
        "code_entropy": code_entropy,
        "bpp_eq": bpp_eq,
        "bpp_entropy": bpp_entropy,
        "bpp_true": bpp_true,
    }


@torch.no_grad()
def reconstruct_kitti_points_with_model(
    model: torch.nn.Module,
    device: torch.device,
    points_xyzi: np.ndarray,
    *,
    quantize: bool = True,
    noise_std: float = 0.0,
    img_h: int = 64,
    img_w: int = 1024,
    fov_up_deg: float = 3.0,
    fov_down_deg: float = -25.0,
    range_threshold: float = 1e-3,
    uniform_bits_fallback: int = 8,
    unprojection_mode: str = "decoded_xyz",
    quant_mode: str = "native",
    oracle_gt_boxes_lidar: Optional[np.ndarray] = None,
    oracle_gt_names: Optional[Sequence[str]] = None,
    oracle_classes: Optional[Sequence[str]] = None,
    oracle_dilate_px: int = 0,
    adaptive_bg_levels_override: Optional[int] = None,
    adaptive_roi_levels_override: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Reconstructs KITTI XYZI points through the compression model.

    Returns:
      recon_points_xyzi: [M,4]
      rate_metrics: scalar dictionary
      debug_payload: {"input_5ch", "recon_5ch", "valid_mask", ...}
    """
    data_5ch, valid_mask = project_points_to_range_image(
        points_xyzi,
        img_h=img_h,
        img_w=img_w,
        fov_up_deg=fov_up_deg,
        fov_down_deg=fov_down_deg,
    )

    quant_mode_norm = str(quant_mode).strip().lower()
    importance_override_np: Optional[np.ndarray] = None
    if quant_mode_norm == "oracle_roi":
        importance_override_np = build_oracle_importance_map_from_gt(
            data_5ch=data_5ch,
            valid_mask=valid_mask,
            gt_boxes_lidar=oracle_gt_boxes_lidar,
            gt_names=oracle_gt_names,
            include_classes=oracle_classes,
            dilate_px=int(oracle_dilate_px),
        )
    elif quant_mode_norm != "native":
        raise ValueError(
            f"Unsupported quant_mode='{quant_mode}'. Expected one of: native, oracle_roi"
        )

    x = torch.from_numpy(data_5ch).unsqueeze(0).to(device)
    raw_points_t = torch.from_numpy(np.asarray(points_xyzi[:, :4], dtype=np.float32)).unsqueeze(0).to(device)
    raw_point_counts_t = torch.tensor([points_xyzi.shape[0]], device=device, dtype=torch.long)
    model_kwargs = {
        "noise_std": float(noise_std),
        "quantize": bool(quantize),
        "raw_points": raw_points_t,
        "raw_point_counts": raw_point_counts_t,
    }
    if importance_override_np is not None:
        model_kwargs["importance_map"] = (
            torch.from_numpy(importance_override_np)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

    quantizer = getattr(model, "quantizer", None)
    restore_bg = None
    restore_roi = None
    if quantizer is not None:
        if hasattr(quantizer, "bg_levels"):
            restore_bg = int(quantizer.bg_levels)
        if hasattr(quantizer, "roi_levels"):
            restore_roi = int(quantizer.roi_levels)
    if adaptive_bg_levels_override is not None:
        if quantizer is None or not hasattr(quantizer, "bg_levels"):
            raise ValueError("adaptive_bg_levels_override requires an adaptive quantizer with bg_levels.")
        bg_override = int(adaptive_bg_levels_override)
        if bg_override < 2:
            raise ValueError(f"adaptive_bg_levels_override must be >=2, got {bg_override}")
        quantizer.bg_levels = bg_override
    if adaptive_roi_levels_override is not None:
        if quantizer is None or not hasattr(quantizer, "roi_levels"):
            raise ValueError("adaptive_roi_levels_override requires an adaptive quantizer with roi_levels.")
        roi_override = int(adaptive_roi_levels_override)
        if roi_override < 2:
            raise ValueError(f"adaptive_roi_levels_override must be >=2, got {roi_override}")
        quantizer.roi_levels = roi_override

    try:
        recon, aux = model(x, **model_kwargs)
    finally:
        if quantizer is not None:
            if restore_bg is not None and hasattr(quantizer, "bg_levels"):
                quantizer.bg_levels = restore_bg
            if restore_roi is not None and hasattr(quantizer, "roi_levels"):
                quantizer.roi_levels = restore_roi

    recon_np = recon.squeeze(0).detach().cpu().numpy().astype(np.float32)

    recon_points = range_image_to_points_xyzi(
        recon_np,
        range_threshold=range_threshold,
        valid_mask=valid_mask,
        unprojection_mode=unprojection_mode,
        fov_up_deg=fov_up_deg,
        fov_down_deg=fov_down_deg,
    )
    rate_metrics = estimate_rate_metrics_from_aux(
        aux=aux,
        input_hw=(img_h, img_w),
        uniform_bits_fallback=uniform_bits_fallback,
    )
    debug_payload: Dict[str, np.ndarray] = {
        "input_5ch": data_5ch,
        "recon_5ch": recon_np,
        "valid_mask": valid_mask.astype(np.float32),
    }
    if importance_override_np is not None:
        debug_payload["importance_map_override"] = importance_override_np.astype(np.float32)
    return recon_points, rate_metrics, debug_payload
