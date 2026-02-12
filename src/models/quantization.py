import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveQuantizer(nn.Module):
    def __init__(self, roi_levels=256, bg_levels=16, eps=1e-6, use_ste=False):
        super(AdaptiveQuantizer, self).__init__()
        if roi_levels < 2 or bg_levels < 2:
            raise ValueError("Quantization levels must be >= 2.")
        self.roi_levels = int(roi_levels)
        self.bg_levels = int(bg_levels)
        self.eps = float(eps)
        self.use_ste = bool(use_ste)

    def _resize_importance_map(self, importance_map, target_hw):
        if importance_map.dim() == 3:
            importance_map = importance_map.unsqueeze(1)
        if importance_map.dim() != 4:
            raise ValueError("importance_map must have shape [B,1,H,W] or [B,H,W].")
        if importance_map.shape[1] != 1:
            raise ValueError("importance_map channel dimension must be 1.")
        if importance_map.shape[-2:] != target_hw:
            importance_map = F.interpolate(
                importance_map.float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        return importance_map.float().clamp(0.0, 1.0)

    def forward(self, latent, importance_map):
        if importance_map is None:
            raise ValueError("importance_map is required for AdaptiveQuantizer.")
        if latent.dim() != 4:
            raise ValueError("latent must have shape [B,C,H,W].")

        importance_map_latent = self._resize_importance_map(importance_map, latent.shape[-2:]).to(latent.device)

        x_min = latent.amin(dim=(1, 2, 3), keepdim=True)
        x_max = latent.amax(dim=(1, 2, 3), keepdim=True)
        x_norm = (latent - x_min) / (x_max - x_min + self.eps)

        # importance=0 => bg_levels, importance=1 => roi_levels.
        level_span = float(self.roi_levels - self.bg_levels)
        level_map = float(self.bg_levels) + (importance_map_latent * level_span)
        
        # Apply Straight-Through Estimator (STE) to allow gradients to flow through rounding
        level_map_rounded = level_map.round().clamp(min=2.0)
        level_map = level_map + (level_map_rounded - level_map).detach()
        
        max_code_map = level_map - 1.0

        q_code = torch.round(x_norm * max_code_map).clamp(min=0.0)
        q_norm = q_code / (max_code_map + self.eps)
        dequant = q_norm * (x_max - x_min) + x_min

        if self.use_ste:
            dequant = latent + (dequant - latent).detach()

        return dequant, q_code, level_map


if __name__ == "__main__":
    torch.manual_seed(0)

    quantizer = AdaptiveQuantizer(roi_levels=256, bg_levels=16)
    latent = torch.linspace(0.0, 1.0, steps=64, dtype=torch.float32).reshape(1, 1, 8, 8)

    importance_map = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    importance_map[:, :, :, :32] = 1.0

    _, q_code, level_map = quantizer(latent, importance_map)
    roi_mask_latent = (F.interpolate(importance_map, size=latent.shape[-2:], mode="nearest") > 0.5).expand_as(q_code)

    roi_codes = q_code[roi_mask_latent]
    bg_codes = q_code[~roi_mask_latent]

    print(f"Latent shape: {list(latent.shape)}")
    print(f"ROI levels used: {int(level_map.max().item())}")
    print(f"BG levels used: {int(level_map.min().item())}")
    print(f"ROI max code: {int(roi_codes.max().item())}, unique: {int(torch.unique(roi_codes).numel())}")
    print(f"BG max code: {int(bg_codes.max().item())}, unique: {int(torch.unique(bg_codes).numel())}")
