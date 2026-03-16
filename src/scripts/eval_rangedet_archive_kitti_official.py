#!/usr/bin/env python3

import argparse
import copy
import csv
import math
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPCDET_ROOT = REPO_ROOT / 'third_party' / 'OpenPCDet'
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate archived RangeDet outputs with official KITTI eval via OpenPCDet.')
    p.add_argument('--archives', type=Path, nargs='+', required=True)
    p.add_argument('--source-roidb', type=Path, default=REPO_ROOT / 'data/dataset/rangedet_kitti_hq/validation/part-0000.roidb')
    p.add_argument('--kitti-root', type=Path, default=REPO_ROOT / 'data/dataset/kitti3dobject')
    p.add_argument('--kitti-infos', type=Path, default=REPO_ROOT / 'data/dataset/kitti3dobject/kitti_infos_val.pkl')
    p.add_argument('--output-csv', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=None, help='Optional dir to save per-archive KITTI txt predictions and eval text.')
    p.add_argument('--score-thresh', type=float, default=0.0)
    p.add_argument('--box-mode', type=str, default='native', choices=['native', 'swap_lw', 'swap_lw_yaw90'])
    return p.parse_args()


class Calibration:
    def __init__(self, calib_file: Path):
        parsed = {}
        with calib_file.open() as f:
            for raw in f:
                if ':' not in raw:
                    continue
                key, values = raw.split(':', 1)
                parsed[key.strip()] = np.fromstring(values.strip(), sep=' ', dtype=np.float32)
        self.P2 = parsed['P2'].reshape(3, 4)
        self.R0 = parsed['R0_rect'].reshape(3, 3) if 'R0_rect' in parsed else parsed['R0'].reshape(3, 3)
        if 'Tr_velo_to_cam' in parsed:
            self.V2C = parsed['Tr_velo_to_cam'].reshape(3, 4)
        elif 'Tr_velo_to_cam_2' in parsed:
            self.V2C = parsed['Tr_velo_to_cam_2'].reshape(3, 4)
        else:
            self.V2C = parsed['Tr_velo2cam'].reshape(3, 4)
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    @staticmethod
    def cart_to_hom(pts):
        return np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))

    def lidar_to_rect(self, pts_lidar):
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        return np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))

    def rect_to_img(self, pts_rect):
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T
        pts_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]
        return pts_img, pts_depth


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib: Calibration):
    boxes = boxes3d_lidar.copy()
    xyz_lidar = boxes[:, 0:3]
    l, w, h = boxes[:, 3:4], boxes[:, 4:5], boxes[:, 5:6]
    r = boxes[:, 6:7]
    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)


def boxes3d_to_corners3d_kitti_camera(boxes3d):
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
    y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)
    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)], [zeros, ones, zeros], [np.sin(ry), zeros, np.cos(ry)]])
    R_list = np.transpose(rot_list, (2, 0, 1))
    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1), z_corners.reshape(-1, 8, 1)), axis=2)
    rotated_corners = np.matmul(temp_corners, R_list)
    x = boxes3d[:, 0].reshape(-1, 1) + rotated_corners[:, :, 0]
    y = boxes3d[:, 1].reshape(-1, 1) + rotated_corners[:, :, 1]
    z = boxes3d[:, 2].reshape(-1, 1) + rotated_corners[:, :, 2]
    return np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2).astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib: Calibration, image_shape=None):
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)
    min_uv = np.min(corners_in_image, axis=1)
    max_uv = np.max(corners_in_image, axis=1)
    boxes2d = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d[:, 0] = np.clip(boxes2d[:, 0], 0, image_shape[1] - 1)
        boxes2d[:, 1] = np.clip(boxes2d[:, 1], 0, image_shape[0] - 1)
        boxes2d[:, 2] = np.clip(boxes2d[:, 2], 0, image_shape[1] - 1)
        boxes2d[:, 3] = np.clip(boxes2d[:, 3], 0, image_shape[0] - 1)
    return boxes2d


def load_recid_to_sample(source_roidb: Path):
    with source_roidb.open('rb') as f:
        roidb = pickle.load(f, encoding='latin1')
    return {rec_id: Path(record['pc_url']).stem for rec_id, record in enumerate(roidb)}


def load_sample_to_pred(archive: Path, recid_to_sample, score_thresh: float):
    with archive.open('rb') as f:
        _annotation_dict = pickle.load(f)
        output_dict = pickle.load(f)
    sample_to_pred = {}
    for rec_id, item in output_dict.items():
        sample_id = recid_to_sample.get(rec_id)
        if sample_id is None:
            continue
        boxes = np.asarray(item.get('det_xyzlwhyaws', {}).get('TYPE_VEHICLE', np.zeros((0, 8), dtype=np.float32)), dtype=np.float32)
        if score_thresh > 0:
            boxes = boxes[boxes[:, 7] >= score_thresh]
        sample_to_pred[sample_id] = boxes
    return sample_to_pred


def apply_box_mode(boxes: np.ndarray, mode: str) -> np.ndarray:
    boxes = boxes.copy()
    if mode == 'swap_lw':
        boxes[:, [3, 4]] = boxes[:, [4, 3]]
    elif mode == 'swap_lw_yaw90':
        boxes[:, [3, 4]] = boxes[:, [4, 3]]
        boxes[:, 6] += np.pi / 2.0
        boxes[:, 6] = (boxes[:, 6] + np.pi) % (2 * np.pi) - np.pi
    return boxes


def empty_anno(frame_id: str):
    z0 = np.zeros(0, dtype=np.float64)
    return {
        'name': np.array([], dtype='<U3'),
        'truncated': z0.copy(),
        'occluded': z0.copy(),
        'alpha': z0.copy(),
        'bbox': np.zeros((0, 4), dtype=np.float64),
        'dimensions': np.zeros((0, 3), dtype=np.float64),
        'location': np.zeros((0, 3), dtype=np.float64),
        'rotation_y': z0.copy(),
        'score': z0.copy(),
        'boxes_lidar': np.zeros((0, 7), dtype=np.float64),
        'frame_id': frame_id,
    }


def build_det_anno(sample_id: str, boxes: np.ndarray, image_shape, calib: Calibration):
    if boxes.size == 0:
        return empty_anno(sample_id)

    pred_scores = boxes[:, 7].astype(np.float64)
    pred_boxes = boxes[:, :7].astype(np.float32)
    pred_boxes_camera = boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
    pred_boxes_img = boxes3d_kitti_camera_to_imageboxes(pred_boxes_camera, calib, image_shape=image_shape)

    anno = empty_anno(sample_id)
    num = pred_scores.shape[0]
    anno['name'] = np.array(['Car'] * num)
    anno['alpha'] = (-np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]).astype(np.float64)
    anno['bbox'] = pred_boxes_img.astype(np.float64)
    anno['dimensions'] = pred_boxes_camera[:, 3:6].astype(np.float64)
    anno['location'] = pred_boxes_camera[:, 0:3].astype(np.float64)
    anno['rotation_y'] = pred_boxes_camera[:, 6].astype(np.float64)
    anno['score'] = pred_scores
    anno['boxes_lidar'] = pred_boxes.astype(np.float64)
    return anno


def maybe_write_kitti_txt(out_dir: Path, anno):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{anno['frame_id']}.txt"
    with out_file.open('w') as f:
        bbox = anno['bbox']
        loc = anno['location']
        dims = anno['dimensions']
        for idx in range(len(bbox)):
            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' % (
                anno['name'][idx], anno['alpha'][idx],
                bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                dims[idx][1], dims[idx][2], dims[idx][0],
                loc[idx][0], loc[idx][1], loc[idx][2],
                anno['rotation_y'][idx], anno['score'][idx]
            ), file=f)


def main():
    args = parse_args()
    recid_to_sample = load_recid_to_sample(args.source_roidb)
    with args.kitti_infos.open('rb') as f:
        kitti_infos = pickle.load(f)

    gt_annos = [copy.deepcopy(info['annos']) for info in kitti_infos]
    rows = []
    for archive in args.archives:
        sample_to_pred = load_sample_to_pred(archive, recid_to_sample, args.score_thresh)
        det_annos = []
        pred_count = 0
        for info in kitti_infos:
            sample_id = info['point_cloud']['lidar_idx']
            image_shape = info['image']['image_shape']
            calib_path = args.kitti_root / 'training' / 'calib' / f'{sample_id}.txt'
            calib = Calibration(calib_path)
            boxes = sample_to_pred.get(sample_id, np.zeros((0, 8), dtype=np.float32))
            boxes = apply_box_mode(boxes, args.box_mode)
            pred_count += int(boxes.shape[0])
            anno = build_det_anno(sample_id, boxes, image_shape, calib)
            det_annos.append(anno)
            if args.output_dir is not None:
                maybe_write_kitti_txt(args.output_dir / archive.stem, anno)

        result_str, ap_dict = kitti_eval.get_official_eval_result(gt_annos, det_annos, ['Car'])
        row = {
            'archive': str(archive),
            'tag': archive.stem.replace('_output_dict_24e', ''),
            'box_mode': args.box_mode,
            'score_thresh': args.score_thresh,
            'pred_count': pred_count,
            'Car_image_easy_R40': ap_dict.get('Car_image/easy_R40', np.nan),
            'Car_image_moderate_R40': ap_dict.get('Car_image/moderate_R40', np.nan),
            'Car_image_hard_R40': ap_dict.get('Car_image/hard_R40', np.nan),
            'Car_bev_easy_R40': ap_dict.get('Car_bev/easy_R40', np.nan),
            'Car_bev_moderate_R40': ap_dict.get('Car_bev/moderate_R40', np.nan),
            'Car_bev_hard_R40': ap_dict.get('Car_bev/hard_R40', np.nan),
            'Car_3d_easy_R40': ap_dict.get('Car_3d/easy_R40', np.nan),
            'Car_3d_moderate_R40': ap_dict.get('Car_3d/moderate_R40', np.nan),
            'Car_3d_hard_R40': ap_dict.get('Car_3d/hard_R40', np.nan),
        }
        rows.append(row)
        if args.output_dir is not None:
            txt = args.output_dir / f'{archive.stem}_official_eval.txt'
            txt.write_text(result_str)
        print(f"[{archive.stem}] Car 3D AP R40 easy/mod/hard = {row['Car_3d_easy_R40']:.4f} / {row['Car_3d_moderate_R40']:.4f} / {row['Car_3d_hard_R40']:.4f}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[rangedet-kitti-official] wrote {args.output_csv}')


if __name__ == '__main__':
    main()
