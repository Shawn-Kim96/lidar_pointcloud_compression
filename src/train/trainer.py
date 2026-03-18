import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.registry import MODELS
from loss.distill_loss import DistillLoss
from utils.teacher_adapter import TeacherAdapter, TeacherAdapterConfig

class Trainer:
    def __init__(
        self,
        config,
        device,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
    ):
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 1. Build Model
        self.model = MODELS.build(config["model"])
        self.model.to(device)
        
        # 2. Build Teacher (if needed)
        self.teacher = None
        if config["teacher"]["enabled"]:
            teacher_cfg = TeacherAdapterConfig(**config["teacher"]["config"])
            self.teacher = TeacherAdapter(teacher_cfg)
            
        # 3. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"]
        )
        
        # 4. Losses
        self.criterion_recon = nn.MSELoss()

        # Weights
        loss_cfg = config["loss"]
        self.w_recon = float(loss_cfg["w_recon"])
        self.w_distill = float(loss_cfg["w_distill"])
        self.w_rate = float(loss_cfg["w_rate"])
        self.w_importance = float(loss_cfg["w_importance"])
        self.w_imp_separation = float(loss_cfg.get("w_imp_separation", 0.0))
        self.recon_loss_mode = str(loss_cfg.get("recon_loss_mode", "mse")).lower()
        if self.recon_loss_mode not in ("mse", "masked_channel_weighted"):
            raise ValueError("recon_loss_mode must be one of: mse, masked_channel_weighted")
        self.recon_range_weight = float(loss_cfg.get("recon_range_weight", 1.0))
        self.recon_xyz_weight = float(loss_cfg.get("recon_xyz_weight", 1.0))
        self.recon_remission_weight = float(loss_cfg.get("recon_remission_weight", 1.0))
        self.w_ray_consistency = float(loss_cfg.get("w_ray_consistency", 0.0))
        self.w_valid_mask = float(loss_cfg.get("w_valid_mask", 0.0))
        self.w_valid_mask_dice = float(loss_cfg.get("w_valid_mask_dice", 0.0))
        self.w_range_grad_row = float(loss_cfg.get("w_range_grad_row", 0.0))
        self.w_range_grad_col = float(loss_cfg.get("w_range_grad_col", 0.0))
        self.w_row_profile = float(loss_cfg.get("w_row_profile", 0.0))
        self.w_detector_target = float(loss_cfg.get("w_detector_target", 0.0))
        data_cfg = config.get("data", {})
        self.fov_up_deg = float(data_cfg.get("fov_up_deg", 3.0))
        self.fov_down_deg = float(data_cfg.get("fov_down_deg", -25.0))
        self._ray_grid_cache = {}

        self.loss_recipe = str(loss_cfg.get("recipe", "legacy")).lower()
        if self.loss_recipe not in ("legacy", "balanced_v1", "balanced_v2"):
            raise ValueError(
                f"Unsupported loss recipe '{self.loss_recipe}'. "
                "Use one of: legacy, balanced_v1, balanced_v2"
            )

        default_rate_mode = "global_mean"
        if self.loss_recipe == "balanced_v1":
            default_rate_mode = "normalized_global"
        elif self.loss_recipe == "balanced_v2":
            default_rate_mode = "normalized_bg"
        rate_mode_raw = loss_cfg.get("rate_loss_mode", None)
        self.rate_loss_mode = default_rate_mode if rate_mode_raw in (None, "") else str(rate_mode_raw).lower()
        if self.rate_loss_mode not in ("global_mean", "normalized_global", "normalized_bg"):
            raise ValueError(
                f"Unsupported rate_loss_mode '{self.rate_loss_mode}'. "
                "Use one of: global_mean, normalized_global, normalized_bg"
            )

        default_imp_mode = "bce" if self.loss_recipe == "legacy" else "weighted_bce"
        imp_mode_raw = loss_cfg.get("importance_loss_mode", None)
        self.importance_loss_mode = default_imp_mode if imp_mode_raw in (None, "") else str(imp_mode_raw).lower()
        if self.importance_loss_mode not in ("bce", "weighted_bce"):
            raise ValueError("importance_loss_mode must be one of: bce, weighted_bce")
        self.importance_pos_weight_mode = str(loss_cfg.get("importance_pos_weight_mode", "auto")).lower()
        if self.importance_pos_weight_mode not in ("auto", "fixed"):
            raise ValueError("importance_pos_weight_mode must be one of: auto, fixed")
        self.importance_pos_weight = float(loss_cfg.get("importance_pos_weight", 1.0))
        self.importance_pos_weight_max = float(loss_cfg.get("importance_pos_weight_max", 50.0))
        self.imp_separation_margin = float(loss_cfg.get("imp_separation_margin", 0.05))

        self.criterion_distill = DistillLoss(
            feature_weight=float(loss_cfg.get("distill_feature_weight", 1.0)),
            logit_weight=float(loss_cfg.get("distill_logit_weight", 1.0)),
            temperature=float(loss_cfg.get("distill_temperature", 1.0)),
            logit_loss_type=str(loss_cfg.get("distill_logit_loss", "auto")),
            align_mode=str(loss_cfg.get("distill_align_mode", "resize")),
            align_hw=loss_cfg.get("distill_align_hw", None),
        )
        self.distill_feature_source = str(loss_cfg.get("distill_feature_source", "channel_mean")).lower()
        if self.distill_feature_source not in ("channel_mean", "energy_map", "none"):
            raise ValueError("distill_feature_source must be one of: channel_mean, energy_map, none")
        self.distill_teacher_score_min = float(loss_cfg.get("distill_teacher_score_min", 0.0))
        self.distill_teacher_score_weight = bool(loss_cfg.get("distill_teacher_score_weight", True))

        quantizer_cfg = config.get("model", {}).get("quantizer_config", {})
        self.quantizer_mode = str(quantizer_cfg.get("mode", "adaptive")).lower()
        self.uniform_bits = int(quantizer_cfg.get("uniform_bits", quantizer_cfg.get("quant_bits", 8)))
        self.roi_levels = float(quantizer_cfg.get("roi_levels", 256.0))
        self.bg_levels = float(quantizer_cfg.get("bg_levels", 16.0))
        model_cfg = config.get("model", {})
        pillar_side_cfg = model_cfg.get("pillar_side_config") or {}
        detector_aux_cfg = model_cfg.get("detector_aux_head_config") or {}
        mask_head_cfg = model_cfg.get("mask_head_config") or {}
        self.pillar_side_enabled = bool(pillar_side_cfg.get("enabled", False))
        self.detector_aux_enabled = bool(detector_aux_cfg.get("enabled", False))
        self.mask_head_enabled = bool(mask_head_cfg.get("enabled", False))

        supervision_cfg = config.get("supervision", {})
        self.supervision_type = supervision_cfg.get("type", "roi")
        self.roi_target_mode = str(supervision_cfg.get("roi_target_mode", "maxpool")).lower()
        if self.roi_target_mode not in ("nearest", "maxpool", "area"):
            raise ValueError(
                f"Unsupported roi_target_mode '{self.roi_target_mode}'. "
                "Use one of: nearest, maxpool, area"
            )

    def _feature_distill_map(self, feat: torch.Tensor):
        if feat is None:
            return None
        if self.distill_feature_source == "none":
            return None
        if self.distill_feature_source == "channel_mean":
            return feat.mean(dim=1, keepdim=True)
        # energy_map: channel-agnostic spatial response for cross-architecture matching.
        return torch.sqrt((feat ** 2).mean(dim=1, keepdim=True) + 1e-8)

    def _build_roi_target(self, roi_mask: torch.Tensor, target_hw):
        if roi_mask is None:
            return None
        if roi_mask.dim() == 3:
            roi_mask = roi_mask.unsqueeze(1)
        roi_mask = roi_mask.float()
        if roi_mask.shape[-2:] == tuple(target_hw):
            return roi_mask.clamp(0.0, 1.0)
        if self.roi_target_mode == "nearest":
            return F.interpolate(roi_mask, size=target_hw, mode="nearest").clamp(0.0, 1.0)
        if self.roi_target_mode == "area":
            return F.interpolate(roi_mask, size=target_hw, mode="area").clamp(0.0, 1.0)
        if self.roi_target_mode == "maxpool":
            return F.adaptive_max_pool2d(roi_mask, output_size=target_hw).clamp(0.0, 1.0)
        raise RuntimeError(f"Unexpected roi_target_mode '{self.roi_target_mode}'")

    def _build_importance_pos_weight(self, target: torch.Tensor):
        if target is None:
            return None
        if self.importance_loss_mode != "weighted_bce":
            return None
        if self.importance_pos_weight_mode == "fixed":
            return torch.tensor([max(self.importance_pos_weight, 1e-3)], device=target.device)
        # Auto: ratio of negative to positive pixels, clamped for stability.
        pos = float(target.sum().item())
        total = float(target.numel())
        neg = max(total - pos, 1.0)
        ratio = neg / max(pos, 1.0)
        ratio = min(max(ratio, 1.0), self.importance_pos_weight_max)
        return torch.tensor([ratio], device=target.device)

    def _compute_rate_loss(self, aux, roi_target):
        if "level_map" not in aux:
            return torch.tensor(0.0, device=self.device)

        level_map = aux["level_map"]
        if self.rate_loss_mode == "global_mean":
            return level_map.mean()

        span = max(self.roi_levels - self.bg_levels, 1.0)
        level_norm = (level_map - self.bg_levels) / span
        level_norm = level_norm.clamp(min=0.0)

        if self.rate_loss_mode == "normalized_global":
            return level_norm.mean()

        if roi_target is not None:
            if roi_target.shape[-2:] != level_norm.shape[-2:]:
                roi_target = F.interpolate(roi_target, size=level_norm.shape[-2:], mode="nearest")
            bg_weight = (1.0 - roi_target).clamp(min=0.0, max=1.0)
        elif "importance_map_pred" in aux:
            imp = aux["importance_map_pred"].detach()
            if imp.shape[-2:] != level_norm.shape[-2:]:
                imp = F.interpolate(imp, size=level_norm.shape[-2:], mode="bilinear", align_corners=False)
            bg_weight = (1.0 - imp).clamp(min=0.0, max=1.0)
        else:
            bg_weight = torch.ones_like(level_norm)

        return (level_norm * bg_weight).sum() / bg_weight.sum().clamp(min=1.0)

    def _compute_importance_loss(self, imp_logits, target):
        pos_weight = self._build_importance_pos_weight(target)
        return F.binary_cross_entropy_with_logits(imp_logits, target, pos_weight=pos_weight)

    def _compute_importance_separation(self, imp_logits, target):
        if target is None or self.w_imp_separation <= 0.0:
            return torch.tensor(0.0, device=self.device)
        probs = torch.sigmoid(imp_logits)
        if target.shape[-2:] != probs.shape[-2:]:
            target = F.interpolate(target, size=probs.shape[-2:], mode="nearest")
        roi_mask = target.clamp(min=0.0, max=1.0)
        bg_mask = (1.0 - roi_mask).clamp(min=0.0, max=1.0)
        roi_mean = (probs * roi_mask).sum() / roi_mask.sum().clamp(min=1.0)
        bg_mean = (probs * bg_mask).sum() / bg_mask.sum().clamp(min=1.0)
        return F.relu(self.imp_separation_margin - (roi_mean - bg_mean))

    def _compute_recon_loss(self, recon, data, valid_mask):
        if self.recon_loss_mode == "mse":
            return self.criterion_recon(recon, data)

        if valid_mask is None:
            valid_mask = torch.ones(
                (data.shape[0], 1, data.shape[2], data.shape[3]),
                device=data.device,
                dtype=data.dtype,
            )
        valid_mask = valid_mask.float()
        if valid_mask.shape[1] != 1:
            valid_mask = valid_mask[:, :1]

        sq = (recon - data) ** 2
        denom = valid_mask.sum().clamp(min=1.0)

        range_loss = (sq[:, 0:1] * valid_mask).sum() / denom
        xyz_loss = (sq[:, 1:4] * valid_mask).sum() / (denom * 3.0)
        rem_loss = (sq[:, 4:5] * valid_mask).sum() / denom

        return (
            self.recon_range_weight * range_loss
            + self.recon_xyz_weight * xyz_loss
            + self.recon_remission_weight * rem_loss
        )

    def _compute_valid_mask_loss(self, aux, valid_mask):
        if self.w_valid_mask <= 0.0 or "valid_mask_logits" not in aux:
            return torch.tensor(0.0, device=self.device)
        logits = aux["valid_mask_logits"]
        target = valid_mask
        if target is None:
            target = torch.ones(
                (logits.shape[0], 1, logits.shape[-2], logits.shape[-1]),
                device=logits.device,
                dtype=logits.dtype,
            )
        elif target.dim() == 3:
            target = target.unsqueeze(1)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
        target = target.float()
        return F.binary_cross_entropy_with_logits(logits, target)

    def _compute_valid_mask_dice_loss(self, aux, valid_mask):
        if self.w_valid_mask_dice <= 0.0 or "valid_mask_logits" not in aux:
            return torch.tensor(0.0, device=self.device)
        logits = aux["valid_mask_logits"]
        target = valid_mask
        if target is None:
            target = torch.ones(
                (logits.shape[0], 1, logits.shape[-2], logits.shape[-1]),
                device=logits.device,
                dtype=logits.dtype,
            )
        elif target.dim() == 3:
            target = target.unsqueeze(1)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
        target = target.float()
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        return 1.0 - dice.mean()

    @staticmethod
    def _pairwise_mask(valid_mask, axis: str):
        if axis == "row":
            return valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
        if axis == "col":
            return valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
        raise ValueError(f"Unknown axis '{axis}'")

    def _compute_range_gradient_loss(self, recon, data, valid_mask, axis: str):
        weight = self.w_range_grad_row if axis == "row" else self.w_range_grad_col
        if weight <= 0.0:
            return torch.tensor(0.0, device=self.device)
        if valid_mask is None:
            valid_mask = torch.ones(
                (data.shape[0], 1, data.shape[-2], data.shape[-1]),
                device=data.device,
                dtype=data.dtype,
            )
        elif valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.float()
        if axis == "row":
            recon_grad = recon[:, 0:1, 1:, :] - recon[:, 0:1, :-1, :]
            data_grad = data[:, 0:1, 1:, :] - data[:, 0:1, :-1, :]
        else:
            recon_grad = recon[:, 0:1, :, 1:] - recon[:, 0:1, :, :-1]
            data_grad = data[:, 0:1, :, 1:] - data[:, 0:1, :, :-1]
        pair_mask = self._pairwise_mask(valid_mask, axis=axis)
        denom = pair_mask.sum().clamp(min=1.0)
        return (((recon_grad - data_grad) ** 2) * pair_mask).sum() / denom

    def _compute_row_profile_loss(self, recon, data, valid_mask):
        if self.w_row_profile <= 0.0:
            return torch.tensor(0.0, device=self.device)
        if valid_mask is None:
            valid_mask = torch.ones(
                (data.shape[0], 1, data.shape[-2], data.shape[-1]),
                device=data.device,
                dtype=data.dtype,
            )
        elif valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.float()
        row_mass = valid_mask.sum(dim=-1).clamp(min=1.0)
        recon_profile = (recon[:, 0:1] * valid_mask).sum(dim=-1) / row_mass
        data_profile = (data[:, 0:1] * valid_mask).sum(dim=-1) / row_mass
        return F.mse_loss(recon_profile, data_profile)

    def _compute_detector_target_loss(self, aux, teacher_target, valid_mask):
        if self.w_detector_target <= 0.0 or "detector_aux_logits" not in aux or teacher_target is None:
            return torch.tensor(0.0, device=self.device)
        logits = aux["detector_aux_logits"]
        if teacher_target.dim() == 3:
            teacher_target = teacher_target.unsqueeze(1)
        if teacher_target.shape[-2:] != logits.shape[-2:]:
            teacher_target = F.interpolate(teacher_target.float(), size=logits.shape[-2:], mode="bilinear", align_corners=False)
        teacher_target = teacher_target.float().clamp(min=0.0, max=1.0)
        if valid_mask is None:
            weight = torch.ones_like(teacher_target)
        else:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            if valid_mask.shape[-2:] != logits.shape[-2:]:
                valid_mask = F.interpolate(valid_mask.float(), size=logits.shape[-2:], mode="nearest")
            weight = valid_mask.float()
        bce = F.binary_cross_entropy_with_logits(logits, teacher_target, reduction="none")
        return (bce * weight).sum() / weight.sum().clamp(min=1.0)

    def _get_ray_grid(self, height: int, width: int, device, dtype):
        key = (height, width, str(device), str(dtype), round(self.fov_up_deg, 4), round(self.fov_down_deg, 4))
        if key not in self._ray_grid_cache:
            rows = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height)
            cols = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width)
            yaw = (2.0 * cols - 1.0) * torch.pi
            fov_up = torch.deg2rad(torch.tensor(self.fov_up_deg, device=device, dtype=torch.float32))
            fov_down = torch.deg2rad(torch.tensor(self.fov_down_deg, device=device, dtype=torch.float32))
            pitch = fov_up - rows * (fov_up - fov_down)
            cos_pitch = torch.cos(pitch)[:, None]
            sin_pitch = torch.sin(pitch)[:, None]
            cos_yaw = torch.cos(yaw)[None, :]
            sin_yaw = torch.sin(yaw)[None, :]
            ray_x = cos_pitch * cos_yaw
            ray_y = -cos_pitch * sin_yaw
            ray_z = sin_pitch.expand(height, width)
            rays = torch.stack([ray_x, ray_y, ray_z], dim=0).to(dtype=dtype)
            self._ray_grid_cache[key] = rays
        return self._ray_grid_cache[key]

    def _compute_ray_consistency_loss(self, recon, valid_mask):
        if self.w_ray_consistency <= 0.0:
            return torch.tensor(0.0, device=self.device)
        if valid_mask is None:
            valid_mask = torch.ones(
                (recon.shape[0], 1, recon.shape[-2], recon.shape[-1]),
                device=recon.device,
                dtype=recon.dtype,
            )
        elif valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.float()
        rays = self._get_ray_grid(recon.shape[-2], recon.shape[-1], recon.device, recon.dtype).unsqueeze(0)
        pred_range = recon[:, 0:1]
        pred_xyz = recon[:, 2:5]
        expected_xyz = pred_range * rays
        denom = (valid_mask.sum() * 3.0).clamp(min=1.0)
        return (((pred_xyz - expected_xyz) ** 2) * valid_mask).sum() / denom

    def _estimate_rate_stats(self, aux):
        rate_proxy = float("nan")
        eq_bits = float("nan")
        code_entropy = float("nan")

        level_map = aux.get("level_map", None)
        if level_map is not None:
            lm = level_map.detach()
            rate_proxy = float(lm.mean().item())
            eq_bits = float(torch.log2(lm.clamp(min=2.0)).mean().item())
        elif self.quantizer_mode == "uniform":
            rate_proxy = float((2 ** self.uniform_bits) - 1)
            eq_bits = float(self.uniform_bits)
        elif self.quantizer_mode == "none":
            rate_proxy = 0.0
            eq_bits = 0.0

        codes = aux.get("codes", None)
        if self.quantizer_mode == "none":
            code_entropy = 0.0
        elif codes is not None:
            q = torch.round(codes.detach()).to(torch.int64).clamp(min=0)
            q_flat = q.reshape(-1).cpu()
            if q_flat.numel() > 0:
                hist = torch.bincount(q_flat).float()
                p = hist / hist.sum().clamp(min=1.0)
                p = p[p > 0]
                code_entropy = float((-(p * torch.log2(p))).sum().item())

        return {
            "rate_proxy": rate_proxy,
            "eq_bits": eq_bits,
            "code_entropy": code_entropy,
        }
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss_sum = 0
        total_rate_proxy = 0.0
        total_eq_bits = 0.0
        total_code_entropy = 0.0
        total_imp_mean = 0.0
        total_valid_mask_loss = 0.0
        total_grad_row_loss = 0.0
        total_grad_col_loss = 0.0
        total_row_profile_loss = 0.0
        total_detector_target_loss = 0.0
        stat_count = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if len(batch) >= 1:
                data = batch[0].to(self.device)
            else:
                continue
            valid_mask = None
            if len(batch) >= 2:
                valid_mask = batch[1].to(self.device)
                if valid_mask.dim() == 3:
                    valid_mask = valid_mask.unsqueeze(1)
                
            # Teacher Forward
            teacher_out = {}
            if self.teacher:
                with torch.no_grad():
                    teacher_out = self.teacher.infer(data, valid_mask=valid_mask)
            
            # Student Forward
            # Supply importance map from teacher if using distillation-based importance
            # For now, let's assume the model figures it out or uses the teacher's map if passed
            # The current AdaptiveQuantizer computes importance from latent, but later checks against teacher
            
            # Pass GT ROI mask if available and using 'roi' supervision
            # This logic mimics train_stage2_1.py
            roi_mask = None
            batch_idx = 2
            if len(batch) >= 3 and self.supervision_type == "roi":
                 roi_mask = batch[batch_idx].to(self.device)
                 batch_idx += 1
            raw_points = None
            raw_point_counts = None
            if self.pillar_side_enabled:
                if len(batch) < batch_idx + 2:
                    raise ValueError("pillar_side training expects raw point tensors in the batch.")
                raw_points = batch[batch_idx].to(self.device)
                raw_point_counts = batch[batch_idx + 1].view(-1).to(self.device)
                batch_idx += 2
            teacher_target = None
            if self.detector_aux_enabled and len(batch) > batch_idx and torch.is_tensor(batch[batch_idx]):
                teacher_target = batch[batch_idx].to(self.device)
                batch_idx += 1

            # Forward
            recon, aux = self.model(
                data,
                noise_std=self.config["train"]["noise_std"],
                quantize=True,
                importance_map=None, # or roi_mask if we want to force it? usually supervised via loss
                raw_points=raw_points,
                raw_point_counts=raw_point_counts,
            )
            
            # Losses
            loss_recon = self._compute_recon_loss(recon, data, valid_mask)
            
            loss_distill = torch.tensor(0.0, device=self.device)

            # Importance Loss
            loss_imp = torch.tensor(0.0, device=self.device)
            loss_imp_sep = torch.tensor(0.0, device=self.device)
            roi_target = None
            if "importance_logits" in aux:
                imp_logits = aux["importance_logits"]
                if roi_mask is not None:
                     # ROI Supervision
                     # Resize/aggregate ROI mask to logits shape.
                     # nearest: strict sampling
                     # area: soft occupancy target
                     # maxpool: preserve sparse positives after heavy downsampling
                     roi_target = self._build_roi_target(roi_mask, imp_logits.shape[-2:])
                     loss_imp = self._compute_importance_loss(imp_logits, roi_target)
                elif "importance_map" in teacher_out:
                     # Teacher supervision
                     target = teacher_out["importance_map"].detach()
                     target = F.interpolate(target, size=imp_logits.shape[-2:], mode="bilinear")
                     loss_imp = self._compute_importance_loss(imp_logits, target)

                loss_imp_sep = self._compute_importance_separation(imp_logits, roi_target)

            loss_rate = self._compute_rate_loss(aux, roi_target)
            loss_ray = self._compute_ray_consistency_loss(recon, valid_mask)
            loss_valid_mask = self._compute_valid_mask_loss(aux, valid_mask)
            loss_valid_mask_dice = self._compute_valid_mask_dice_loss(aux, valid_mask)
            loss_grad_row = self._compute_range_gradient_loss(recon, data, valid_mask, axis="row")
            loss_grad_col = self._compute_range_gradient_loss(recon, data, valid_mask, axis="col")
            loss_row_profile = self._compute_row_profile_loss(recon, data, valid_mask)
            loss_detector_target = self._compute_detector_target_loss(aux, teacher_target, valid_mask)

            if self.teacher and self.w_distill > 0.0:
                student_features = None
                teacher_features = None
                if "latent" in aux and "features" in teacher_out:
                    # Feature distillation map can be channel-mean (legacy) or energy-map (recommended).
                    student_features = self._feature_distill_map(aux["latent"])
                    teacher_features = self._feature_distill_map(teacher_out["features"].detach())

                student_logits = aux.get("importance_logits", None)
                teacher_logits = teacher_out.get("logits", None)
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.detach()

                if student_features is not None or student_logits is not None:
                    distill_weight = roi_target
                    if distill_weight is None and "importance_map" in teacher_out:
                        distill_weight = teacher_out["importance_map"].detach()

                    if (
                        self.distill_teacher_score_weight
                        and "score" in teacher_out
                        and self.distill_teacher_score_min > 0.0
                    ):
                        # Suppress distillation on low-confidence teacher samples.
                        score = teacher_out["score"].detach().view(-1, 1, 1, 1).to(self.device)
                        sample_gate = (score >= self.distill_teacher_score_min).float()
                        if distill_weight is None:
                            ref = student_logits if student_logits is not None else student_features
                            if ref is not None:
                                distill_weight = torch.ones(
                                    (ref.shape[0], 1, ref.shape[-2], ref.shape[-1]),
                                    device=ref.device,
                                )
                        if distill_weight is not None:
                            if distill_weight.dim() == 3:
                                distill_weight = distill_weight.unsqueeze(1)
                            distill_weight = distill_weight * sample_gate

                    loss_distill, _ = self.criterion_distill(
                        student_features=student_features,
                        teacher_features=teacher_features,
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        importance_map=distill_weight,
                        return_details=False,
                    )

            total_loss = (
                self.w_recon * loss_recon +
                self.w_distill * loss_distill +
                self.w_rate * loss_rate +
                self.w_importance * loss_imp +
                self.w_imp_separation * loss_imp_sep +
                self.w_ray_consistency * loss_ray +
                self.w_valid_mask * loss_valid_mask +
                self.w_valid_mask_dice * loss_valid_mask_dice +
                self.w_range_grad_row * loss_grad_row +
                self.w_range_grad_col * loss_grad_col +
                self.w_row_profile * loss_row_profile +
                self.w_detector_target * loss_detector_target
            )
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_loss_sum += total_loss.item()
            rate_stats = self._estimate_rate_stats(aux)
            total_rate_proxy += rate_stats["rate_proxy"]
            total_eq_bits += rate_stats["eq_bits"]
            total_code_entropy += rate_stats["code_entropy"]
            imp_map_pred = aux.get("importance_map_pred", None)
            if imp_map_pred is not None:
                total_imp_mean += float(imp_map_pred.detach().mean().item())
            else:
                total_imp_mean += float("nan")
            total_valid_mask_loss += float(loss_valid_mask.detach().item() + loss_valid_mask_dice.detach().item())
            total_grad_row_loss += float(loss_grad_row.detach().item())
            total_grad_col_loss += float(loss_grad_col.detach().item())
            total_row_profile_loss += float(loss_row_profile.detach().item())
            total_detector_target_loss += float(loss_detector_target.detach().item())
            stat_count += 1
            pbar.set_postfix(loss=total_loss.item())

        denom = max(stat_count, 1)
        return {
            "loss": total_loss_sum / max(len(self.train_loader), 1),
            "rate_proxy": total_rate_proxy / denom,
            "eq_bits": total_eq_bits / denom,
            "code_entropy": total_code_entropy / denom,
            "imp_mean": total_imp_mean / denom,
            "valid_mask_loss": total_valid_mask_loss / denom,
            "grad_row_loss": total_grad_row_loss / denom,
            "grad_col_loss": total_grad_col_loss / denom,
            "row_profile_loss": total_row_profile_loss / denom,
            "detector_target_loss": total_detector_target_loss / denom,
        }

    def run(self, epochs, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        # Save config
        import yaml
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        for epoch in range(epochs):
            train_stats = self.train_epoch(epoch)
            print(
                f"Epoch {epoch}: Loss {train_stats['loss']:.4f} "
                f"| rate_proxy={train_stats['rate_proxy']:.4f} "
                f"| eq_bits={train_stats['eq_bits']:.4f} "
                f"| code_entropy={train_stats['code_entropy']:.4f} "
                f"| imp_mean={train_stats['imp_mean']:.4f} "
                f"| valid_mask={train_stats['valid_mask_loss']:.4f} "
                f"| grad_row={train_stats['grad_row_loss']:.4f} "
                f"| grad_col={train_stats['grad_col_loss']:.4f} "
                f"| row_profile={train_stats['row_profile_loss']:.4f} "
                f"| det_target={train_stats['detector_target_loss']:.4f}"
            )
            
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))

        # Always save a final checkpoint so short pilot runs are evaluable.
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model_final.pth"))
