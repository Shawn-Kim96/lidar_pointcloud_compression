from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

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
) -> np.ndarray:
    """
    Converts reconstructed range-image tensor back to KITTI-style XYZI points.

    Rules:
    - Uses channels [x,y,z,intensity] from reconstructed tensor.
    - Valid if range > threshold and xyz is finite.
    - Optional additional valid_mask can be provided.
    """
    if recon_5ch.shape[0] != 5:
        raise ValueError(f"Expected recon_5ch shape [5,H,W], got {recon_5ch.shape}")

    rng = recon_5ch[0]
    inten = recon_5ch[1]
    x = recon_5ch[2]
    y = recon_5ch[3]
    z = recon_5ch[4]

    valid = rng > float(range_threshold)
    valid &= np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=np.float32) > 0.5

    if valid.sum() == 0:
        return np.zeros((0, 4), dtype=np.float32)

    pts = np.stack([x[valid], y[valid], z[valid], inten[valid]], axis=1).astype(np.float32)
    return pts


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
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Reconstructs KITTI XYZI points through the compression model.

    Returns:
      recon_points_xyzi: [M,4]
      rate_metrics: scalar dictionary
      debug_payload: {"input_5ch", "recon_5ch", "valid_mask"}
    """
    data_5ch, valid_mask = project_points_to_range_image(
        points_xyzi,
        img_h=img_h,
        img_w=img_w,
        fov_up_deg=fov_up_deg,
        fov_down_deg=fov_down_deg,
    )

    x = torch.from_numpy(data_5ch).unsqueeze(0).to(device)
    recon, aux = model(x, noise_std=float(noise_std), quantize=bool(quantize))
    recon_np = recon.squeeze(0).detach().cpu().numpy().astype(np.float32)

    recon_points = range_image_to_points_xyzi(
        recon_np,
        range_threshold=range_threshold,
        valid_mask=valid_mask,
    )
    rate_metrics = estimate_rate_metrics_from_aux(
        aux=aux,
        input_hw=(img_h, img_w),
        uniform_bits_fallback=uniform_bits_fallback,
    )
    return recon_points, rate_metrics, {
        "input_5ch": data_5ch,
        "recon_5ch": recon_np,
        "valid_mask": valid_mask.astype(np.float32),
    }
