from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentLinearROIHead(nn.Module):
    """
    Minimal dense objectness head: 1x1 projection on latent features, then resize to input resolution.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.logit = nn.Conv2d(int(in_channels), 1, kernel_size=1)

    def forward(self, latent: torch.Tensor, target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        logits = self.logit(latent)
        if target_hw is not None:
            logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        return logits


class LatentRefineROIHead(nn.Module):
    """
    Slightly stronger dense objectness head with latent-space refinement before full-resolution prediction.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        hidden = int(hidden_channels)
        self.latent_refine = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
        )
        self.full_res_refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        feat = self.latent_refine(latent)
        if target_hw is not None:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return self.full_res_refine(feat)


class LatentDeepROIHead(nn.Module):
    """
    Deeper dense objectness head for longer Track 2 sweeps.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        hidden = int(hidden_channels)
        self.latent_tower = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
        )
        self.full_res_tower = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        feat = self.latent_tower(latent)
        if target_hw is not None:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return self.full_res_tower(feat)


def build_range_roi_head(head_type: str, in_channels: int, hidden_channels: int = 64) -> nn.Module:
    key = str(head_type).strip().lower()
    if key == "linear":
        return LatentLinearROIHead(in_channels=in_channels)
    if key == "refine":
        return LatentRefineROIHead(in_channels=in_channels, hidden_channels=hidden_channels)
    if key == "deep":
        return LatentDeepROIHead(in_channels=in_channels, hidden_channels=hidden_channels)
    raise ValueError(f"Unsupported Track 2 head_type='{head_type}'. Expected one of: linear, refine, deep")
