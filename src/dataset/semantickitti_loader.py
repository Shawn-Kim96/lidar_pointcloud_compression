import os
import numpy as np
import torch
from torch.utils.data import Dataset
import argparse
from pathlib import Path


ROI_CLASSES = {10, 18, 30, 31, 32}


class SemanticKittiDataset(Dataset):
    def __init__(self, root_dir, sequences, config=None, return_azimuth=False, return_roi_mask=False):
        """
        Args:
            root_dir (str): Path to SemanticKITTI dataset (e.g., /path/to/dataset/sequences)
            sequences (list): List of sequence numbers to load (e.g., ['00', '01'])
            config (dict): Configuration dictionary containing sensor parameters
            return_azimuth (bool): If True, __getitem__ returns azimuth image as third tensor
            return_roi_mask (bool): If True, __getitem__ returns ROI mask [1, H, W]
        """
        self.root_dir = root_dir
        self.sequences = sequences
        self.return_azimuth = return_azimuth
        self.return_roi_mask = return_roi_mask
        self.files = []
        
        # Load file paths
        for seq in self.sequences:
            seq_dir = os.path.join(self.root_dir, seq, 'velodyne')
            if not os.path.exists(seq_dir):
                print(f"Warning: Sequence {seq} not found at {seq_dir}")
                continue
            
            # Sort files to ensure temporal order
            files = sorted([os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if f.endswith('.bin')])
            self.files.extend(files)
            
        print(f"Loaded {len(self.files)} frames from sequences {sequences}")

        # Sensor parameters (Defaults to Velodyne HDL-64E if not provided)
        if config is None:
            self.fov_up = 3.0
            self.fov_down = -25.0
            self.H = 64
            self.W = 1024
        else:
            self.fov_up = config.get('fov_up', 3.0)
            self.fov_down = config.get('fov_down', -25.0)
            self.H = config.get('img_height', 64)
            self.W = config.get('img_width', 1024)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Load Point Cloud (x, y, z, remission) - float32
        scan = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))

        point_labels = None
        if self.return_roi_mask:
            label_path = self._label_path_for_scan(file_path)
            if os.path.exists(label_path):
                label_raw = np.fromfile(label_path, dtype=np.uint32)
                if label_raw.shape[0] == scan.shape[0]:
                    point_labels = (label_raw & 0xFFFF).astype(np.int32)
        
        # Project to Range Image
        (
            proj_range,
            proj_xyz,
            proj_intensity,
            proj_azimuth,
            proj_mask,
            proj_roi_mask,
        ) = self.do_range_projection(scan, point_labels=point_labels)
        
        # Construct tensor: [5, H, W] -> (range, intensity, x, y, z)
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

    def _label_path_for_scan(self, scan_path):
        seq_dir = os.path.dirname(os.path.dirname(scan_path))
        scan_name = Path(scan_path).stem + ".label"
        return os.path.join(seq_dir, "labels", scan_name)

    def do_range_projection(self, points, point_labels=None):
        """
        Project 3D points to 2D range image
        """
        # Laser parameters
        fov_up = self.fov_up / 180.0 * np.pi
        fov_down = self.fov_down / 180.0 * np.pi
        fov = fov_up - fov_down
        
        # Extract depth
        depth = np.linalg.norm(points[:, :3], 2, axis=1)
        depth = np.clip(depth, a_min=1e-6, a_max=None)
        
        # Extract components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        intensity = points[:, 3]
        
        # Get angles
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)
        
        # Projections in terms of image coordinates
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + np.abs(fov_down)) / fov
        
        # Scale to image size
        proj_x *= self.W
        proj_y *= self.H
        
        # Floor to get integer indices
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)
        
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)
        
        # Order in decreasing depth (to keep closest points if collision)
        # But for compression we usually want ALL points or the closest?
        # Standard practice: sort by depth descending, so closest points overwrite distant ones
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        points = points[order]
        intensity = intensity[order]
        yaw = yaw[order]
        if point_labels is not None:
            point_labels = point_labels[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        
        # Canvas
        proj_range = np.full((self.H, self.W), -1, dtype=np.float32)
        proj_xyz = np.full((self.H, self.W, 3), 0, dtype=np.float32)
        proj_intensity = np.full((self.H, self.W), 0, dtype=np.float32)
        proj_azimuth = np.full((self.H, self.W), 0, dtype=np.float32)
        proj_mask = np.zeros((self.H, self.W), dtype=np.float32)
        proj_roi_mask = np.zeros((self.H, self.W), dtype=np.float32)
        
        # Assign
        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points[:, :3]
        proj_intensity[proj_y, proj_x] = intensity
        proj_azimuth[proj_y, proj_x] = yaw
        proj_mask[proj_y, proj_x] = 1.0
        if point_labels is not None:
            proj_roi_mask[proj_y, proj_x] = np.isin(point_labels, list(ROI_CLASSES)).astype(np.float32)
        
        return proj_range, proj_xyz, proj_intensity, proj_azimuth, proj_mask, proj_roi_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemanticKITTI loader smoke test")
    parser.add_argument(
        "--root_dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "dataset" / "semantickitti" / "dataset" / "sequences"),
        help="Path to SemanticKITTI dataset sequences root",
    )
    parser.add_argument(
        "--sequence",
        default="00",
        help="SemanticKITTI sequence id (default: 00)",
    )
    parser.add_argument(
        "--no_roi_mask",
        action="store_true",
        help="Disable ROI mask output in smoke test.",
    )
    parser.add_argument(
        "--return_azimuth",
        action="store_true",
        help="Also return azimuth image in smoke test.",
    )
    args = parser.parse_args()

    dataset = SemanticKittiDataset(
        root_dir=args.root_dir,
        sequences=[args.sequence],
        return_roi_mask=not args.no_roi_mask,
        return_azimuth=args.return_azimuth,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No frames found for sequence {args.sequence} under {args.root_dir}."
        )

    sample_tuple = dataset[0]
    sample = sample_tuple[0]
    mask = sample_tuple[1]
    print(f"Sample tensor shape: {list(sample.shape)}")
    print(f"Sample mask shape: {list(mask.shape)}")
    if not args.no_roi_mask:
        roi_mask = sample_tuple[2]
        print(f"Sample ROI mask shape: {list(roi_mask.shape)}")
    if args.return_azimuth:
        azimuth = sample_tuple[-1]
        print(f"Sample azimuth shape: {list(azimuth.shape)}")

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_tuple = next(iter(loader))
    batch = batch_tuple[0]
    batch_mask = batch_tuple[1]
    print(f"Batch tensor shape: {list(batch.shape)}")
    print(f"Batch mask shape: {list(batch_mask.shape)}")
    if not args.no_roi_mask:
        batch_roi_mask = batch_tuple[2]
        print(f"Batch ROI mask shape: {list(batch_roi_mask.shape)}")
    if args.return_azimuth:
        batch_azimuth = batch_tuple[-1]
        print(f"Batch azimuth shape: {list(batch_azimuth.shape)}")
