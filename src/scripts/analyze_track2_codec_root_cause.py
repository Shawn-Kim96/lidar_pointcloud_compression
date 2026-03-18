#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


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
    parser = argparse.ArgumentParser(description="Analyze Track2 codec failure modes across raw/stage0/stage1.")
    parser.add_argument("--source-roidb", type=Path, default=Path("data/dataset/rangedet_kitti_hq/validation/part-0000.roidb"))
    parser.add_argument("--raw-archive", type=Path, default=Path("logs/rangedet_eval_outputs/260315_rddecodefixfull_raw_patched_output_dict_24e.pkl"))
    parser.add_argument("--stage0-baseline-archive", type=Path, default=Path("logs/rangedet_eval_outputs/260315_rddecodefixfull_nqa_patched_output_dict_24e.pkl"))
    parser.add_argument("--stage0-enhanced-archive", type=Path, default=Path("logs/rangedet_eval_outputs/260315_rddecodefixfull_nqb_patched_output_dict_24e.pkl"))
    parser.add_argument("--stage1-baseline-archive", type=Path, default=Path("logs/rangedet_eval_outputs/260315_rddecodefixfull_uqa_patched_output_dict_24e.pkl"))
    parser.add_argument("--stage1-enhanced-archive", type=Path, default=Path("logs/rangedet_eval_outputs/260315_rddecodefixfull_uqb_patched_output_dict_24e.pkl"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/dataset/rangedet_kitti_hq"))
    parser.add_argument("--stage0-baseline-root", type=Path, default=Path("data/dataset/rangedet_kitti_recon_260315_rddecodefixfull_nqa_patched"))
    parser.add_argument("--stage0-enhanced-root", type=Path, default=Path("data/dataset/rangedet_kitti_recon_260315_rddecodefixfull_nqb_patched"))
    parser.add_argument("--stage1-baseline-root", type=Path, default=Path("data/dataset/rangedet_kitti_recon_260315_rddecodefixfull_uqa_patched"))
    parser.add_argument("--stage1-enhanced-root", type=Path, default=Path("data/dataset/rangedet_kitti_recon_260315_rddecodefixfull_uqb_patched"))
    parser.add_argument("--output-csv", type=Path, default=Path("logs/track2_codec_root_cause_summary.csv"))
    parser.add_argument("--fig-dir", type=Path, default=Path("logs/track2_codec_root_cause_figs"))
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_archive(path: Path):
    with path.open("rb") as f:
        ann = pickle.load(f)
        out = pickle.load(f)
    return ann, out


def to_range_channel(ri):
    arr = np.asarray(ri, dtype=np.float32)
    return arr[..., 0] if arr.ndim == 3 else arr


def valid_mask(ri):
    return to_range_channel(ri) > 0


def row_profile(ri):
    arr = to_range_channel(ri)
    masked = np.where(arr > 0, arr, np.nan)
    return np.nanmean(masked, axis=1)


def banding_score(ri):
    rp = row_profile(ri)
    return float(np.nanmean(np.abs(np.diff(rp))))


def gradient_mismatch(raw_ri, other_ri, axis: str):
    raw = to_range_channel(raw_ri)
    other = to_range_channel(other_ri)
    raw_valid = valid_mask(raw_ri)
    other_valid = valid_mask(other_ri)
    if axis == "row":
        raw_grad = raw[1:, :] - raw[:-1, :]
        other_grad = other[1:, :] - other[:-1, :]
        pair_mask = (raw_valid[1:, :] & raw_valid[:-1, :] & other_valid[1:, :] & other_valid[:-1, :])
    else:
        raw_grad = raw[:, 1:] - raw[:, :-1]
        other_grad = other[:, 1:] - other[:, :-1]
        pair_mask = (raw_valid[:, 1:] & raw_valid[:, :-1] & other_valid[:, 1:] & other_valid[:, :-1])
    if pair_mask.sum() == 0:
        return np.nan
    return float(np.abs(raw_grad - other_grad)[pair_mask].mean())


def high_frequency_score(ri):
    arr = to_range_channel(ri)
    mask = arr > 0
    if mask.sum() == 0:
        return np.nan
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:] = np.abs(arr[:, 1:] - arr[:, :-1])
    gy[1:, :] = np.abs(arr[1:, :] - arr[:-1, :])
    hf = gx + gy
    return float(hf[mask].mean())


def range_metrics(raw_ri, other_ri):
    raw = to_range_channel(raw_ri)
    other = to_range_channel(other_ri)
    raw_valid = valid_mask(raw_ri)
    other_valid = valid_mask(other_ri)
    inter = raw_valid & other_valid
    union = raw_valid | other_valid
    mask_iou = float(inter.sum() / max(union.sum(), 1))
    range_mae = float(np.abs(raw[inter] - other[inter]).mean()) if inter.sum() else np.nan
    row_mae = float(np.nanmean(np.abs(row_profile(raw_ri) - row_profile(other_ri))))
    return {
        "valid_mask_iou": mask_iou,
        "range_mae_valid": range_mae,
        "row_profile_mae": row_mae,
        "banding_score": banding_score(other_ri),
        "high_frequency_score": high_frequency_score(other_ri),
        "grad_row_mae": gradient_mismatch(raw_ri, other_ri, axis="row"),
        "grad_col_mae": gradient_mismatch(raw_ri, other_ri, axis="col"),
    }


def rotated_intersection_area(box1: np.ndarray, box2: np.ndarray) -> float:
    def rectangle_corners_bev(box: np.ndarray) -> np.ndarray:
        x, y, _, l, w, _, yaw = [float(v) for v in box[:7]]
        c = np.cos(yaw)
        s = np.sin(yaw)
        dx = l * 0.5
        dy = w * 0.5
        corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float64)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        return corners @ rot.T + np.array([x, y], dtype=np.float64)

    def polygon_area(poly: np.ndarray) -> float:
        if poly.shape[0] < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _cross(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def point_in_convex_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
        signs = []
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            signs.append(_cross(b - a, point - a))
        signs = np.asarray(signs)
        return bool(np.all(signs >= -1e-9) or np.all(signs <= 1e-9))

    def segment_intersection(p1, p2, q1, q2):
        r = p2 - p1
        s = q2 - q1
        denom = _cross(r, s)
        if abs(denom) < 1e-12:
            return None
        t = _cross(q1 - p1, s) / denom
        u = _cross(q1 - p1, r) / denom
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            return p1 + t * r
        return None

    def polygon_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
        pts = []
        for p in subject:
            if point_in_convex_polygon(p, clip):
                pts.append(p)
        for p in clip:
            if point_in_convex_polygon(p, subject):
                pts.append(p)
        for i in range(len(subject)):
            p1 = subject[i]
            p2 = subject[(i + 1) % len(subject)]
            for j in range(len(clip)):
                q1 = clip[j]
                q2 = clip[(j + 1) % len(clip)]
                inter = segment_intersection(p1, p2, q1, q2)
                if inter is not None:
                    pts.append(inter)
        if not pts:
            return np.zeros((0, 2), dtype=np.float64)
        pts = np.asarray(pts, dtype=np.float64)
        rounded = np.round(pts, decimals=8)
        _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
        pts = pts[np.sort(uniq_idx)]
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return pts[np.argsort(angles)]

    poly1 = rectangle_corners_bev(box1)
    poly2 = rectangle_corners_bev(box2)
    inter = polygon_intersection(poly1, poly2)
    return polygon_area(inter)


def box_iou3d(box1: np.ndarray, box2: np.ndarray) -> float:
    inter_area = rotated_intersection_area(box1, box2)
    if inter_area <= 0.0:
        return 0.0
    area1 = float(box1[3] * box1[4])
    area2 = float(box2[3] * box2[4])
    z1_min = float(box1[2] - box1[5] / 2.0)
    z1_max = float(box1[2] + box1[5] / 2.0)
    z2_min = float(box2[2] - box2[5] / 2.0)
    z2_max = float(box2[2] + box2[5] / 2.0)
    inter_h = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))
    if inter_h <= 0.0:
        return 0.0
    inter_vol = inter_area * inter_h
    vol1 = float(area1 * box1[5])
    vol2 = float(area2 * box2[5])
    return inter_vol / max(vol1 + vol2 - inter_vol, 1e-8)


def localization_metrics(gt_boxes, pred_boxes, thr=0.3):
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    best_ious = []
    matched = 0
    for gt in gt_boxes:
        best = 0.0
        for pred in pred_boxes:
            best = max(best, box_iou3d(gt, pred[:7]))
        best_ious.append(best)
        if best >= thr:
            matched += 1
    return {
        "gt_count": int(len(gt_boxes)),
        "pred_count": int(len(pred_boxes)),
        "oracle_recall3d_03": float(matched / max(len(gt_boxes), 1)),
        "mean_best_iou3d_03": float(np.mean(best_ious) if best_ious else 0.0),
        "score_mean": float(pred_boxes[:, 7].mean()) if len(pred_boxes) else 0.0,
        "score_std": float(pred_boxes[:, 7].std()) if len(pred_boxes) else 0.0,
    }


def boxes_to_corners_3d_lidar(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    x, y, z, l, w, h, yaw = [boxes[:, i] for i in range(7)]
    template = np.array([
        [0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=np.float32)
    corners = template[None, :, :].repeat(boxes.shape[0], axis=0)
    corners[:, :, 0] *= l[:, None]
    corners[:, :, 1] *= w[:, None]
    corners[:, :, 2] *= h[:, None]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.stack([
        np.stack([cos_yaw, -sin_yaw, np.zeros_like(cos_yaw)], axis=-1),
        np.stack([sin_yaw, cos_yaw, np.zeros_like(cos_yaw)], axis=-1),
        np.stack([np.zeros_like(cos_yaw), np.zeros_like(cos_yaw), np.ones_like(cos_yaw)], axis=-1),
    ], axis=1)
    corners = np.einsum("nij,nkj->nki", rot, corners)
    corners[:, :, 0] += x[:, None]
    corners[:, :, 1] += y[:, None]
    corners[:, :, 2] += z[:, None]
    return corners.astype(np.float32)


def sample_box_edges(corners, samples_per_edge=64):
    samples = []
    for start, end in EDGE_IDX:
        p0 = corners[start]
        p1 = corners[end]
        t = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float32)[:, None]
        samples.append((1.0 - t) * p0[None, :] + t * p1[None, :])
    return np.concatenate(samples, axis=0)


def project_xyz_to_rangedet_kitti_grid(xyz, azimuth, inclination):
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


def collect_projected_points(corners_batch, azimuth, inclination):
    projected = []
    for corners in corners_batch:
        edge_xyz = sample_box_edges(corners)
        projected.append(project_xyz_to_rangedet_kitti_grid(edge_xyz, azimuth=azimuth, inclination=inclination))
    return projected


def projected_envelopes(projected_list, width, height, min_points=8, pad_u=6, pad_v=2):
    envs = []
    for pts in projected_list:
        if pts.shape[0] < min_points:
            envs.append(None)
            continue
        u = pts[:, 0]
        v = pts[:, 1]
        u0 = max(0, int(np.floor(u.min())) - pad_u)
        u1 = min(width - 1, int(np.ceil(u.max())) + pad_u)
        v0 = max(0, int(np.floor(v.min())) - pad_v)
        v1 = min(height - 1, int(np.ceil(v.max())) + pad_v)
        envs.append((u0, v0, u1, v1))
    return envs


def plot_range_base(ax, range_image, title):
    arr = to_range_channel(range_image)
    valid = arr > 0
    vis = arr.copy()
    if valid.any():
        vals = vis[valid]
        lo, hi = np.percentile(vals, [2, 98])
        vis = np.clip((vis - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    ax.imshow(vis, cmap="gray", aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_projected_rectangles(ax, envelopes, color):
    for env in envelopes:
        if env is None:
            continue
        u0, v0, u1, v1 = env
        ax.plot([u0, u1, u1, u0, u0], [v0, v0, v1, v1, v0], color=color, linewidth=1.0, alpha=0.95)


def choose_ranked_frames(df: pd.DataFrame):
    base = df[df["setting"] == "stage0_baseline"].copy()
    if base.empty:
        raise RuntimeError("No stage0_baseline rows found.")
    base["failure_score"] = (
        (1.0 - base["valid_mask_iou"].fillna(0.0))
        + base["row_profile_mae"].fillna(0.0)
        + base["grad_row_mae"].fillna(0.0)
        + base["grad_col_mae"].fillna(0.0)
        + (1.0 - base["oracle_recall3d_03"].fillna(0.0))
    )
    ranked = base.sort_values("failure_score", ascending=False).reset_index(drop=True)
    worst = ranked.iloc[0]["sample_id"]
    median = ranked.iloc[len(ranked) // 2]["sample_id"]
    return str(worst), str(median)


def dominant_failure_driver(row):
    metrics = {
        "valid_mask": 1.0 - float(row.get("valid_mask_iou", 0.0)),
        "banding": float(row.get("row_profile_mae", 0.0)) + float(row.get("banding_score", 0.0)),
        "gradient": float(row.get("grad_row_mae", 0.0)) + float(row.get("grad_col_mae", 0.0)),
        "detector": 1.0 - float(row.get("oracle_recall3d_03", 0.0)),
    }
    return max(metrics.items(), key=lambda kv: kv[1])[0]


def make_case_figure(sample_id, roidb_by_sample, data_roots, pred_maps, output_path: Path):
    record = roidb_by_sample[sample_id]
    raw_npz = np.load(record["pc_url"])
    azimuth = np.asarray(raw_npz["azimuth"], dtype=np.float32)
    inclination = np.asarray(raw_npz["inclination"], dtype=np.float32)
    gt_class = np.asarray(record["gt_class"])
    gt_boxes = np.asarray(record["gt_bbox_csa"], dtype=np.float32)[gt_class == 1]
    gt_proj = collect_projected_points(boxes_to_corners_3d_lidar(gt_boxes), azimuth=azimuth, inclination=inclination)
    gt_env = projected_envelopes(gt_proj, width=azimuth.shape[0], height=inclination.shape[0])

    settings = [
        ("raw_basic", "Raw/basic", data_roots["raw_basic"]),
        ("stage0_baseline", "Stage0 baseline", data_roots["stage0_baseline"]),
        ("stage0_enhanced", "Stage0 enhanced", data_roots["stage0_enhanced"]),
        ("stage1_baseline", "Stage1 baseline", data_roots["stage1_baseline"]),
        ("stage1_enhanced", "Stage1 enhanced", data_roots["stage1_enhanced"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(28, 11), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, (setting_key, title, root) in zip(axes, settings):
        npz = np.load(root / "npz_trainval" / f"{sample_id}.npz")
        preds = pred_maps[setting_key].get(sample_id, np.zeros((0, 8), dtype=np.float32))
        pred_proj = collect_projected_points(boxes_to_corners_3d_lidar(preds[:, :7]), azimuth=azimuth, inclination=inclination)
        pred_env = projected_envelopes(pred_proj, width=azimuth.shape[0], height=inclination.shape[0])
        plot_range_base(ax, npz["range_image"], f"{title} ({sample_id})")
        draw_projected_rectangles(ax, gt_env, "#00FF80")
        draw_projected_rectangles(ax, pred_env, "#FF3366")
    axes[-1].axis("off")
    legend_handles = [
        Line2D([0], [0], color="#00FF80", linewidth=1.3, label="GT car envelopes"),
        Line2D([0], [0], color="#FF3366", linewidth=1.3, label="RangeDet predicted envelopes"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.source_roidb.open("rb") as f:
        roidb = pickle.load(f, encoding="latin1")
    if args.max_frames > 0:
        roidb = roidb[: args.max_frames]

    roidb_by_sample = {Path(rec["pc_url"]).stem: rec for rec in roidb}
    sample_ids = list(roidb_by_sample.keys())

    archives = {
        "raw_basic": args.raw_archive,
        "stage0_baseline": args.stage0_baseline_archive,
        "stage0_enhanced": args.stage0_enhanced_archive,
        "stage1_baseline": args.stage1_baseline_archive,
        "stage1_enhanced": args.stage1_enhanced_archive,
    }
    data_roots = {
        "raw_basic": args.raw_root,
        "stage0_baseline": args.stage0_baseline_root,
        "stage0_enhanced": args.stage0_enhanced_root,
        "stage1_baseline": args.stage1_baseline_root,
        "stage1_enhanced": args.stage1_enhanced_root,
    }

    pred_maps = {}
    for setting, archive_path in archives.items():
        _, out = load_archive(archive_path)
        preds = {}
        for rec_id, sample_id in enumerate(sample_ids):
            preds[sample_id] = np.asarray(
                out.get(rec_id, {}).get("det_xyzlwhyaws", {}).get("TYPE_VEHICLE", np.zeros((0, 8), dtype=np.float32)),
                dtype=np.float32,
            )
        pred_maps[setting] = preds

    rows = []
    for sample_id in sample_ids:
        record = roidb_by_sample[sample_id]
        gt_class = np.asarray(record["gt_class"])
        gt_boxes = np.asarray(record["gt_bbox_csa"], dtype=np.float32)[gt_class == 1]
        raw_npz = np.load(record["pc_url"])
        raw_ri = raw_npz["range_image"]
        for setting, root in data_roots.items():
            npz_path = root / "npz_trainval" / f"{sample_id}.npz"
            if not npz_path.exists():
                continue
            ri = np.load(npz_path)["range_image"]
            artifact = range_metrics(raw_ri, ri)
            loc = localization_metrics(gt_boxes, pred_maps[setting].get(sample_id, np.zeros((0, 8), dtype=np.float32)))
            row = {
                "sample_id": sample_id,
                "setting": setting,
                **artifact,
                **loc,
            }
            row["dominant_failure_driver"] = dominant_failure_driver(row)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[track2-root-cause] wrote summary csv: {args.output_csv}")

    worst_id, median_id = choose_ranked_frames(df)
    make_case_figure(worst_id, roidb_by_sample, data_roots, pred_maps, args.fig_dir / f"{worst_id}_worst.png")
    make_case_figure(median_id, roidb_by_sample, data_roots, pred_maps, args.fig_dir / f"{median_id}_median.png")
    print(f"[track2-root-cause] wrote figures to: {args.fig_dir}")
    print(f"[track2-root-cause] worst_sample={worst_id}")
    print(f"[track2-root-cause] median_sample={median_id}")


if __name__ == "__main__":
    main()
