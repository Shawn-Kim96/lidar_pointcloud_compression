import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.autoencoder import _activation, _norm


def _scatter_max(values: torch.Tensor, index: torch.Tensor, num_groups: int) -> torch.Tensor:
    out = torch.full(
        (num_groups, values.shape[1]),
        -1e9,
        device=values.device,
        dtype=values.dtype,
    )
    out.scatter_reduce_(
        0,
        index[:, None].expand(-1, values.shape[1]),
        values,
        reduce="amax",
        include_self=True,
    )
    return torch.where(out < -1e8, torch.zeros_like(out), out)


class DynamicPFNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, last_layer: bool):
        super().__init__()
        self.last_layer = bool(last_layer)
        hidden_channels = int(out_channels) if self.last_layer else int(out_channels) // 2
        self.linear = nn.Linear(in_channels, hidden_channels, bias=False)
        self.norm = nn.LayerNorm(hidden_channels)
        self.act = nn.SiLU(inplace=False)

    def forward(self, point_features: torch.Tensor, inverse: torch.Tensor, num_groups: int) -> torch.Tensor:
        x = self.act(self.norm(self.linear(point_features)))
        x_max = _scatter_max(x, inverse, num_groups)
        if self.last_layer:
            return x_max
        return torch.cat([x, x_max[inverse]], dim=1)


class BEVResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str = "batch", activation: str = "silu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(norm, out_channels)
        self.act = _activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = _norm(norm, out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _norm(norm, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.act(out + residual)
        return out


class BEVStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, norm: str = "batch", activation: str = "silu"):
        super().__init__()
        blocks = [BEVResidualBlock(in_channels, out_channels, stride=2, norm=norm, activation=activation)]
        for _ in range(max(0, int(num_blocks) - 1)):
            blocks.append(BEVResidualBlock(out_channels, out_channels, stride=1, norm=norm, activation=activation))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DynamicPillarBEVSideStream(nn.Module):
    """
    Raw-point side branch:
      raw points -> dynamic pillar PFN -> dense BEV -> BEV encoder/FPN -> gather back to RI stages
    """

    def __init__(
        self,
        point_cloud_range: Sequence[float],
        pillar_size: Sequence[float],
        pfn_hidden_channels: int,
        pfn_out_channels: int,
        bev_channels: Sequence[int],
        bev_blocks: Sequence[int],
        fpn_channels: int,
        max_raw_points: int = 150000,
        norm: str = "batch",
        activation: str = "silu",
    ):
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError("point_cloud_range must have 6 elements [xmin, ymin, zmin, xmax, ymax, zmax]")
        if len(pillar_size) != 2:
            raise ValueError("pillar_size must have 2 elements [x, y]")
        if len(bev_channels) != len(bev_blocks):
            raise ValueError("bev_channels and bev_blocks must have the same length")
        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = [float(v) for v in point_cloud_range]
        self.pillar_x, self.pillar_y = [float(v) for v in pillar_size]
        self.grid_x = int(math.ceil((self.x_max - self.x_min) / self.pillar_x))
        self.grid_y = int(math.ceil((self.y_max - self.y_min) / self.pillar_y))
        self.max_raw_points = int(max_raw_points)
        self.num_stage_outputs = len(bev_channels)
        self.fpn_channels = int(fpn_channels)

        point_feature_dim = 11  # xyz, intensity, cluster xyz, center xyz, range
        self.pfn1 = DynamicPFNLayer(point_feature_dim, int(pfn_hidden_channels), last_layer=False)
        self.pfn2 = DynamicPFNLayer(int(pfn_hidden_channels), int(pfn_out_channels), last_layer=True)

        self.stages = nn.ModuleList()
        in_channels = int(pfn_out_channels)
        for out_channels, blocks in zip(bev_channels, bev_blocks):
            self.stages.append(BEVStage(in_channels, int(out_channels), int(blocks), norm=norm, activation=activation))
            in_channels = int(out_channels)

        self.laterals = nn.ModuleList([
            nn.Conv2d(int(ch), self.fpn_channels, kernel_size=1, bias=False) for ch in bev_channels
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
                _norm(norm, self.fpn_channels),
                _activation(activation),
            )
            for _ in bev_channels
        ])

    def _build_dense_bev(self, raw_points: torch.Tensor, raw_point_counts: torch.Tensor) -> torch.Tensor:
        batch_size = raw_points.shape[0]
        bev = raw_points.new_zeros((batch_size, self.pfn2.linear.out_features, self.grid_y, self.grid_x))
        z_center = 0.5 * (self.z_min + self.z_max)

        for b in range(batch_size):
            count = int(raw_point_counts[b].item()) if raw_point_counts is not None else int(raw_points.shape[1])
            if count <= 0:
                continue
            pts = raw_points[b, :count, :4]
            xyz = pts[:, :3]
            intensity = pts[:, 3:4]
            mask = (
                (xyz[:, 0] >= self.x_min) & (xyz[:, 0] < self.x_max) &
                (xyz[:, 1] >= self.y_min) & (xyz[:, 1] < self.y_max) &
                (xyz[:, 2] >= self.z_min) & (xyz[:, 2] < self.z_max)
            )
            pts = pts[mask]
            if pts.shape[0] == 0:
                continue
            xyz = pts[:, :3]
            intensity = pts[:, 3:4]

            x_idx = torch.floor((xyz[:, 0] - self.x_min) / self.pillar_x).long()
            y_idx = torch.floor((xyz[:, 1] - self.y_min) / self.pillar_y).long()
            coords = torch.stack([x_idx, y_idx], dim=1)
            unique_coords, inverse = torch.unique(coords, return_inverse=True, dim=0)
            num_pillars = int(unique_coords.shape[0])
            if num_pillars == 0:
                continue

            counts = torch.bincount(inverse, minlength=num_pillars).clamp(min=1).to(xyz.dtype)
            xyz_sum = xyz.new_zeros((num_pillars, 3))
            xyz_sum.index_add_(0, inverse, xyz)
            xyz_mean = xyz_sum / counts[:, None]

            center_x = self.x_min + (unique_coords[:, 0].to(xyz.dtype) + 0.5) * self.pillar_x
            center_y = self.y_min + (unique_coords[:, 1].to(xyz.dtype) + 0.5) * self.pillar_y
            center = torch.stack([center_x, center_y, torch.full_like(center_x, z_center)], dim=1)

            f_cluster = xyz - xyz_mean[inverse]
            f_center = xyz - center[inverse]
            f_range = torch.norm(xyz, dim=1, keepdim=True)
            point_features = torch.cat([xyz, intensity, f_cluster, f_center, f_range], dim=1)

            pillar_features = self.pfn1(point_features, inverse, num_pillars)
            pillar_features = self.pfn2(pillar_features, inverse, num_pillars)
            bev[b, :, unique_coords[:, 1], unique_coords[:, 0]] = pillar_features.transpose(0, 1)

        return bev

    def _run_bev_backbone(self, bev: torch.Tensor) -> List[torch.Tensor]:
        stage_features: List[torch.Tensor] = []
        x = bev
        for stage in self.stages:
            x = stage(x)
            stage_features.append(x)

        fused = [None] * len(stage_features)
        top = None
        for idx in reversed(range(len(stage_features))):
            lateral = self.laterals[idx](stage_features[idx])
            if top is not None:
                top = F.interpolate(top, size=lateral.shape[-2:], mode="nearest")
                lateral = lateral + top
            top = lateral
            fused[idx] = self.smooth[idx](lateral)
        return fused

    def _gather_to_ri(
        self,
        bev_stage_features: Sequence[torch.Tensor],
        ri_xyz: torch.Tensor,
        ri_valid: torch.Tensor,
        target_stage_hws: Sequence[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        batch_size = ri_xyz.shape[0]
        for stage_idx, (bev_feat, target_hw) in enumerate(zip(bev_stage_features, target_stage_hws)):
            target_h, target_w = int(target_hw[0]), int(target_hw[1])
            xyz_ds = F.interpolate(ri_xyz, size=(target_h, target_w), mode="nearest")
            valid_ds = F.interpolate(ri_valid.float(), size=(target_h, target_w), mode="nearest") > 0.5
            scale = float(2 ** (stage_idx + 1))
            cell_x = self.pillar_x * scale
            cell_y = self.pillar_y * scale
            stage_grid_h = bev_feat.shape[-2]
            stage_grid_w = bev_feat.shape[-1]

            x_idx = torch.floor((xyz_ds[:, 0] - self.x_min) / cell_x).long()
            y_idx = torch.floor((xyz_ds[:, 1] - self.y_min) / cell_y).long()
            valid = (
                valid_ds[:, 0]
                & torch.isfinite(xyz_ds[:, 0])
                & torch.isfinite(xyz_ds[:, 1])
                & (x_idx >= 0) & (x_idx < stage_grid_w)
                & (y_idx >= 0) & (y_idx < stage_grid_h)
            )

            gathered = bev_feat.new_zeros((batch_size, bev_feat.shape[1], target_h, target_w))
            for b in range(batch_size):
                if not bool(valid[b].any()):
                    continue
                flat = bev_feat[b].reshape(bev_feat.shape[1], -1)
                flat_idx = (y_idx[b].clamp(0, stage_grid_h - 1) * stage_grid_w + x_idx[b].clamp(0, stage_grid_w - 1)).reshape(-1)
                sample = flat[:, flat_idx].reshape(bev_feat.shape[1], target_h, target_w)
                sample = sample * valid[b].to(sample.dtype).unsqueeze(0)
                gathered[b] = sample
            outputs.append(gathered)
        return outputs

    def forward(
        self,
        raw_points: torch.Tensor,
        raw_point_counts: torch.Tensor,
        ri_xyz: torch.Tensor,
        ri_valid: torch.Tensor,
        target_stage_hws: Sequence[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], dict]:
        bev = self._build_dense_bev(raw_points, raw_point_counts)
        bev_stage_features = self._run_bev_backbone(bev)
        ri_stage_features = self._gather_to_ri(bev_stage_features, ri_xyz, ri_valid, target_stage_hws)
        aux = {
            "pillar_bev_nonzero": float((bev.abs().sum(dim=1) > 0).float().mean().item()),
        }
        return ri_stage_features, aux
