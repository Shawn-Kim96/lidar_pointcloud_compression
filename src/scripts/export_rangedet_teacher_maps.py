#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


HEIGHT_LUT = np.array([
    0.20966667, 0.2092, 0.2078, 0.2078, 0.2078, 0.20733333, 0.20593333, 0.20546667,
    0.20593333, 0.20546667, 0.20453333, 0.205, 0.2036, 0.20406667, 0.2036, 0.20313333,
    0.20266667, 0.20266667, 0.20173333, 0.2008, 0.2008, 0.2008, 0.20033333, 0.1994,
    0.20033333, 0.19986667, 0.1994, 0.1994, 0.19893333, 0.19846667, 0.19846667, 0.19846667,
    0.12566667, 0.1252, 0.1252, 0.12473333, 0.12473333, 0.1238, 0.12333333, 0.12286667,
    0.12286667, 0.1224, 0.12146667, 0.12146667, 0.121, 0.12053333, 0.1196, 0.11913333,
    0.11866667, 0.11866667, 0.1182, 0.11726667, 0.1168, 0.11633333, 0.11586667, 0.11493333,
    0.11446667, 0.114, 0.11353333, 0.1126, 0.11213333, 0.11166667, 0.1112, 0.11026667,
], dtype=np.float32)

EDGE_IDX = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Track2 detector-aware teacher maps from raw RangeDet outputs.")
    parser.add_argument("--source-roidb", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True, help="Raw/basic RangeDet archived output_dict pkl.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory to write <sample_id>.npy target maps.")
    parser.add_argument("--score-min", type=float, default=0.15)
    parser.add_argument("--pad-u", type=int, default=6)
    parser.add_argument("--pad-v", type=int, default=2)
    parser.add_argument("--blur", type=int, default=5, help="Odd Gaussian blur kernel. 0 disables blur.")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_archive(path: Path):
    with path.open("rb") as f:
        annotation_dict = pickle.load(f)
        output_dict = pickle.load(f)
    return annotation_dict, output_dict


def boxes_to_corners_3d_lidar(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    x, y, z, l, w, h, yaw = [boxes[:, i] for i in range(7)]
    template = np.array(
        [
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    corners = template[None, :, :].repeat(boxes.shape[0], axis=0)
    corners[:, :, 0] *= l[:, None]
    corners[:, :, 1] *= w[:, None]
    corners[:, :, 2] *= h[:, None]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.stack(
        [
            np.stack([cos_yaw, -sin_yaw, np.zeros_like(cos_yaw)], axis=-1),
            np.stack([sin_yaw, cos_yaw, np.zeros_like(cos_yaw)], axis=-1),
            np.stack([np.zeros_like(cos_yaw), np.zeros_like(cos_yaw), np.ones_like(cos_yaw)], axis=-1),
        ],
        axis=1,
    )
    corners = np.einsum("nij,nkj->nki", rot, corners)
    corners[:, :, 0] += x[:, None]
    corners[:, :, 1] += y[:, None]
    corners[:, :, 2] += z[:, None]
    return corners.astype(np.float32)


def sample_box_edges(corners: np.ndarray, samples_per_edge: int = 48) -> np.ndarray:
    samples = []
    for start, end in EDGE_IDX:
        p0 = corners[start]
        p1 = corners[end]
        t = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float32)[:, None]
        samples.append((1.0 - t) * p0[None, :] + t * p1[None, :])
    return np.concatenate(samples, axis=0)


def project_xyz_to_grid(xyz: np.ndarray, azimuth: np.ndarray, inclination: np.ndarray):
    if xyz.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z).clip(min=1e-6)
    yaw = np.arctan2(-y, x)
    pitch = np.arcsin(z / r)
    cols = np.round(((yaw + np.pi) / (2.0 * np.pi)) * (azimuth.shape[0] - 1)).astype(np.int32)
    rows = np.abs(HEIGHT_LUT[:, None] - pitch[None, :]).argmin(axis=0).astype(np.int32)
    cols = np.clip(cols, 0, azimuth.shape[0] - 1)
    rows = np.clip(rows, 0, inclination.shape[0] - 1)
    return np.stack([cols, rows], axis=-1)


def projected_envelope(points_uv: np.ndarray, width: int, height: int, pad_u: int, pad_v: int):
    if points_uv.shape[0] < 8:
        return None
    u = points_uv[:, 0]
    v = points_uv[:, 1]
    u0 = max(0, int(np.floor(u.min())) - pad_u)
    u1 = min(width - 1, int(np.ceil(u.max())) + pad_u)
    v0 = max(0, int(np.floor(v.min())) - pad_v)
    v1 = min(height - 1, int(np.ceil(v.max())) + pad_v)
    if u1 <= u0 or v1 <= v0:
        return None
    return (u0, v0, u1, v1)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.source_roidb.open("rb") as f:
        roidb = pickle.load(f, encoding="latin1")
    _, output_dict = load_archive(args.archive)

    total = len(roidb)
    limit = args.max_frames if args.max_frames > 0 else total
    roidb = roidb[:limit]

    written = 0
    for rec_id, record in enumerate(roidb):
        sample_id = Path(record["pc_url"]).stem
        npz = np.load(record["pc_url"])
        azimuth = np.asarray(npz["azimuth"], dtype=np.float32)
        inclination = np.asarray(npz["inclination"], dtype=np.float32)
        h = int(inclination.shape[0])
        w = int(azimuth.shape[0])

        boxes = np.asarray(
            output_dict.get(rec_id, {}).get("det_xyzlwhyaws", {}).get("TYPE_VEHICLE", np.zeros((0, 8), dtype=np.float32)),
            dtype=np.float32,
        )
        if boxes.shape[0] > 0:
            boxes = boxes[boxes[:, 7] >= float(args.score_min)]

        target = np.zeros((h, w), dtype=np.float32)
        if boxes.shape[0] > 0:
            corners = boxes_to_corners_3d_lidar(boxes[:, :7])
            for box, corners_i in zip(boxes, corners):
                edge_xyz = sample_box_edges(corners_i)
                proj = project_xyz_to_grid(edge_xyz, azimuth=azimuth, inclination=inclination)
                env = projected_envelope(proj, width=w, height=h, pad_u=args.pad_u, pad_v=args.pad_v)
                if env is None:
                    continue
                u0, v0, u1, v1 = env
                target[v0 : v1 + 1, u0 : u1 + 1] = np.maximum(target[v0 : v1 + 1, u0 : u1 + 1], float(box[7]))

        if args.blur and args.blur > 1:
            if cv2 is not None:
                k = int(args.blur)
                if k % 2 == 0:
                    k += 1
                target = cv2.GaussianBlur(target, (k, k), sigmaX=0.0)
                if target.max() > 0:
                    target = target / target.max()
            elif target.max() > 0:
                target = target / target.max()

        np.save(str(args.output_root / f"{sample_id}.npy"), target.astype(np.float32))
        written += 1
        if written % 200 == 0 or written == len(roidb):
            print(f"[rangedet-teacher-maps] {written}/{len(roidb)}")

    meta = {
        "source_roidb": str(args.source_roidb),
        "archive": str(args.archive),
        "score_min": float(args.score_min),
        "pad_u": int(args.pad_u),
        "pad_v": int(args.pad_v),
        "blur": int(args.blur),
        "records": int(written),
    }
    with (args.output_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[rangedet-teacher-maps] wrote {written} target maps to {args.output_root}")


if __name__ == "__main__":
    main()
