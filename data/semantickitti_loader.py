import os
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

class SemanticKittiDataset(Dataset):
    def __init__(self, root_dir, sequences, config=None):
        """
        Args:
            root_dir (str): Path to SemanticKITTI dataset (e.g., /path/to/dataset/sequences)
            sequences (list): List of sequence numbers to load (e.g., ['00', '01'])
            config (dict): Configuration dictionary containing sensor parameters
        """
        self.root_dir = root_dir
        self.sequences = sequences
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
        
        # Project to Range Image
        proj_range, proj_xyz, proj_remission, proj_mask = self.do_range_projection(scan)
        
        # Construct Tensor: [5, H, W] -> (range, x, y, z, remission)
        # We use a mask to indicate valid pixels (0 for empty, 1 for valid)
        data = np.zeros((5, self.H, self.W), dtype=np.float32)
        data[0] = proj_range
        data[1] = proj_xyz[:, :, 0]
        data[2] = proj_xyz[:, :, 1]
        data[3] = proj_xyz[:, :, 2]
        data[4] = proj_remission
        
        # Mask can be passed separately or used to zero out invalid data
        # data = data * proj_mask 
        
        return torch.from_numpy(data), torch.from_numpy(proj_mask)

    def do_range_projection(self, points):
        """
        Project 3D points to 2D range image
        """
        # Laser parameters
        fov_up = self.fov_up / 180.0 * np.pi
        fov_down = self.fov_down / 180.0 * np.pi
        fov = fov_up - fov_down
        
        # Extract depth
        depth = np.linalg.norm(points[:, :3], 2, axis=1)
        
        # Extract components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        remission = points[:, 3]
        
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
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = points[order]
        remission = remission[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        
        # Canvas
        proj_range = np.full((self.H, self.W), -1, dtype=np.float32)
        proj_xyz = np.full((self.H, self.W, 3), 0, dtype=np.float32)
        proj_remission = np.full((self.H, self.W), 0, dtype=np.float32)
        proj_mask = np.zeros((self.H, self.W), dtype=np.float32)
        
        # Assign
        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points[:, :3]
        proj_remission[proj_y, proj_x] = remission
        proj_mask[proj_y, proj_x] = 1.0
        
        return proj_range, proj_xyz, proj_remission, proj_mask

if __name__ == "__main__":
    # Simple test
    print("Testing SemanticKittiDataset...")
    # Mocking a path for testing (this won't run without data)
    pass
