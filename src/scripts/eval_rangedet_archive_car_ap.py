#!/usr/bin/env python3

import argparse
import csv
import math
import os
import pickle
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate archived RangeDet validation outputs with a simple car-only AP metric in lidar space."
    )
    parser.add_argument(
        "--source-roidb",
        type=Path,
        default=Path("data/dataset/rangedet_kitti_hq/validation/part-0000.roidb"),
        help="RangeDet validation roidb used to map rec_id -> sample id and GT boxes.",
    )
    parser.add_argument(
        "--archives",
        type=Path,
        nargs="+",
        required=True,
        help="Archived RangeDet output_dict pkl files (annotation_dict + output_dict).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the summary CSV.",
    )
    return parser.parse_args()


def load_source_roidb(path: Path):
    with path.open("rb") as f:
        roidb = pickle.load(f, encoding="latin1")

    recid_to_sample = {}
    sample_to_gt = {}
    for rec_id, record in enumerate(roidb):
        sample_id = Path(record["pc_url"]).stem
        recid_to_sample[rec_id] = sample_id
        gt_class = np.asarray(record["gt_class"])
        gt_boxes = np.asarray(record["gt_bbox_csa"], dtype=np.float32)
        sample_to_gt[sample_id] = gt_boxes[gt_class == 1]
    return recid_to_sample, sample_to_gt


def rotated_intersection_area(box1: np.ndarray, box2: np.ndarray) -> float:
    rect1 = (
        (float(box1[0]), float(box1[1])),
        (float(box1[3]), float(box1[4])),
        float(box1[6] * 180.0 / math.pi),
    )
    rect2 = (
        (float(box2[0]), float(box2[1])),
        (float(box2[3]), float(box2[4])),
        float(box2[6] * 180.0 / math.pi),
    )
    _, pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if pts is None:
        return 0.0
    hull = cv2.convexHull(pts, returnPoints=True)
    return float(cv2.contourArea(hull))


def box_iou(box1: np.ndarray, box2: np.ndarray, mode: str) -> float:
    inter_area = rotated_intersection_area(box1, box2)
    if inter_area <= 0.0:
        return 0.0

    area1 = float(box1[3] * box1[4])
    area2 = float(box2[3] * box2[4])
    if mode == "bev":
        return inter_area / max(area1 + area2 - inter_area, 1e-8)

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


def evaluate_ap(sample_to_gt, sample_to_pred, threshold: float, mode: str):
    total_gt = sum(len(v) for v in sample_to_gt.values())
    used = {sample_id: np.zeros(len(gt), dtype=bool) for sample_id, gt in sample_to_gt.items()}

    detections = []
    for sample_id, preds in sample_to_pred.items():
        for pred in preds:
            detections.append((float(pred[7]), sample_id, pred[:7]))
    detections.sort(key=lambda x: x[0], reverse=True)

    tp = []
    fp = []
    for _, sample_id, pred_box in detections:
        gt_boxes = sample_to_gt[sample_id]
        best_iou = 0.0
        best_idx = -1
        for i, gt_box in enumerate(gt_boxes):
            if used[sample_id][i]:
                continue
            iou = box_iou(pred_box, gt_box, mode)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= threshold and best_idx >= 0:
            used[sample_id][best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp, dtype=np.float64))
    recall = tp / max(total_gt, 1)
    precision = tp / np.maximum(tp + fp, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return {
        "ap": ap,
        "final_recall": float(recall[-1] if len(recall) else 0.0),
        "num_det": len(detections),
        "num_gt": int(total_gt),
    }


def evaluate_oracle(sample_to_gt, sample_to_pred, threshold: float, mode: str):
    matched = 0
    total = 0
    best_ious = []

    for sample_id, gt_boxes in sample_to_gt.items():
        preds = sample_to_pred.get(sample_id, np.zeros((0, 8), dtype=np.float32))
        for gt_box in gt_boxes:
            total += 1
            best = 0.0
            for pred in preds:
                best = max(best, box_iou(pred[:7], gt_box, mode))
            best_ious.append(best)
            if best >= threshold:
                matched += 1

    return {
        "oracle_recall": matched / max(total, 1),
        "mean_best_iou": float(np.mean(best_ious) if best_ious else 0.0),
    }


def load_predictions(path: Path, recid_to_sample, sample_to_gt):
    with path.open("rb") as f:
        _annotation_dict = pickle.load(f)
        output_dict = pickle.load(f)

    sample_to_pred = {sample_id: np.zeros((0, 8), dtype=np.float32) for sample_id in sample_to_gt}
    nonempty_frames = 0
    for rec_id, item in output_dict.items():
        sample_id = recid_to_sample.get(rec_id)
        if sample_id is None:
            continue
        boxes = np.asarray(
            item.get("det_xyzlwhyaws", {}).get("TYPE_VEHICLE", np.zeros((0, 8), dtype=np.float32)),
            dtype=np.float32,
        )
        sample_to_pred[sample_id] = boxes
        if len(boxes) > 0:
            nonempty_frames += 1

    return sample_to_pred, nonempty_frames


def main():
    args = parse_args()
    recid_to_sample, sample_to_gt = load_source_roidb(args.source_roidb)

    thresholds = (0.3, 0.5, 0.7)
    rows = []
    for archive_path in args.archives:
        sample_to_pred, nonempty_frames = load_predictions(archive_path, recid_to_sample, sample_to_gt)
        avg_det_per_frame = sum(len(v) for v in sample_to_pred.values()) / max(len(sample_to_pred), 1)
        row = {
            "archive": str(archive_path),
            "tag": archive_path.stem.replace("_output_dict_24e", ""),
            "nonempty_frames": nonempty_frames,
            "avg_det_per_frame": avg_det_per_frame,
        }
        for thr in thresholds:
            ap3d = evaluate_ap(sample_to_gt, sample_to_pred, threshold=thr, mode="3d")
            apbev = evaluate_ap(sample_to_gt, sample_to_pred, threshold=thr, mode="bev")
            oracle = evaluate_oracle(sample_to_gt, sample_to_pred, threshold=thr, mode="3d")
            key = str(thr).replace(".", "")
            row[f"ap3d_{key}"] = ap3d["ap"]
            row[f"apbev_{key}"] = apbev["ap"]
            row[f"oracle_recall3d_{key}"] = oracle["oracle_recall"]
            row[f"mean_best_iou3d_{key}"] = oracle["mean_best_iou"]
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["archive", "tag"]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[rangedet-archive-eval] wrote {args.output_csv}")
    for row in rows:
        print(
            f"{row['tag']}: "
            f"AP3D@0.3={row['ap3d_03']:.4f}, AP3D@0.5={row['ap3d_05']:.4f}, AP3D@0.7={row['ap3d_07']:.4f}, "
            f"APBEV@0.3={row['apbev_03']:.4f}, APBEV@0.5={row['apbev_05']:.4f}, APBEV@0.7={row['apbev_07']:.4f}, "
            f"meanBestIoU3D={row['mean_best_iou3d_03']:.4f}"
        )


if __name__ == "__main__":
    main()
