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
        )

        quantizer_cfg = config.get("model", {}).get("quantizer_config", {})
        self.quantizer_mode = str(quantizer_cfg.get("mode", "adaptive")).lower()
        self.uniform_bits = int(quantizer_cfg.get("uniform_bits", quantizer_cfg.get("quant_bits", 8)))
        self.roi_levels = float(quantizer_cfg.get("roi_levels", 256.0))
        self.bg_levels = float(quantizer_cfg.get("bg_levels", 16.0))

        supervision_cfg = config.get("supervision", {})
        self.supervision_type = supervision_cfg.get("type", "roi")
        self.roi_target_mode = str(supervision_cfg.get("roi_target_mode", "maxpool")).lower()
        if self.roi_target_mode not in ("nearest", "maxpool", "area"):
            raise ValueError(
                f"Unsupported roi_target_mode '{self.roi_target_mode}'. "
                "Use one of: nearest, maxpool, area"
            )

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

        codes = aux.get("codes", None)
        if codes is not None:
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
            if len(batch) >= 3 and self.supervision_type == "roi":
                 roi_mask = batch[2].to(self.device)

            # Forward
            recon, aux = self.model(
                data,
                noise_std=self.config["train"]["noise_std"],
                quantize=True,
                importance_map=None # or roi_mask if we want to force it? usually supervised via loss
            )
            
            # Losses
            loss_recon = self.criterion_recon(recon, data)
            
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

            if self.teacher and self.w_distill > 0.0:
                student_features = None
                teacher_features = None
                if "latent" in aux and "features" in teacher_out:
                    # Channel-mismatch-safe feature distillation using channel-averaged maps.
                    student_features = aux["latent"].mean(dim=1, keepdim=True)
                    teacher_features = teacher_out["features"].detach().mean(dim=1, keepdim=True)

                student_logits = aux.get("importance_logits", None)
                teacher_logits = teacher_out.get("logits", None)
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.detach()

                if student_features is not None or student_logits is not None:
                    distill_weight = roi_target
                    if distill_weight is None and "importance_map" in teacher_out:
                        distill_weight = teacher_out["importance_map"].detach()

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
                self.w_imp_separation * loss_imp_sep
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
            stat_count += 1
            pbar.set_postfix(loss=total_loss.item())

        denom = max(stat_count, 1)
        return {
            "loss": total_loss_sum / max(len(self.train_loader), 1),
            "rate_proxy": total_rate_proxy / denom,
            "eq_bits": total_eq_bits / denom,
            "code_entropy": total_code_entropy / denom,
            "imp_mean": total_imp_mean / denom,
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
                f"| imp_mean={train_stats['imp_mean']:.4f}"
            )
            
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))

        # Always save a final checkpoint so short pilot runs are evaluable.
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model_final.pth"))
