import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

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
        self.criterion_recon = nn.MSELoss() # Or MultiScaleChamfer
        self.criterion_distill = DistillLoss()
        
        # Weights
        self.w_recon = config["loss"]["w_recon"]
        self.w_distill = config["loss"]["w_distill"]
        self.w_rate = config["loss"]["w_rate"]
        self.w_importance = config["loss"]["w_importance"]
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss_sum = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if len(batch) >= 1:
                data = batch[0].to(self.device)
            else:
                continue
                
            # Teacher Forward
            teacher_out = {}
            if self.teacher:
                with torch.no_grad():
                    teacher_out = self.teacher.infer(data)
            
            # Student Forward
            # Supply importance map from teacher if using distillation-based importance
            # For now, let's assume the model figures it out or uses the teacher's map if passed
            # The current AdaptiveQuantizer computes importance from latent, but later checks against teacher
            
            # Pass GT ROI mask if available and using 'roi' supervision
            # This logic mimics train_stage2_1.py
            roi_mask = None
            if len(batch) >= 3 and self.config["supervision"]["type"] == "roi":
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
            if self.teacher:
                 # We need features from student. 
                 # Currently RangeCompressionModel returns (recon, aux).
                 # We might need to adjust model to return features for distillation.
                 # For now, assume simple recon distillation or skip feature distillation if not exposed.
                 # The previous code distilled features.
                 pass
                 
            # Rate Loss (Bpp)
            # aux['codes'] might be [B, C, H, W]
            # Simplest rate proxy: mean value of quantized levels (if scalar) or entropy
            # adaptive quantization return 'level_map' in aux usually
            loss_rate = torch.tensor(0.0, device=self.device)
            if "level_map" in aux:
                loss_rate = aux["level_map"].mean()
                
            # Importance Loss
            loss_imp = torch.tensor(0.0, device=self.device)
            if "importance_logits" in aux:
                imp_logits = aux["importance_logits"]
                if roi_mask is not None:
                     # ROI Supervision
                     # Resize mask to logits shape
                     target = F.interpolate(roi_mask, size=imp_logits.shape[-2:], mode="nearest")
                     loss_imp = F.binary_cross_entropy_with_logits(imp_logits, target)
                elif "importance_map" in teacher_out:
                     # Teacher supervision
                     target = teacher_out["importance_map"].detach()
                     target = F.interpolate(target, size=imp_logits.shape[-2:], mode="bilinear")
                     loss_imp = F.binary_cross_entropy_with_logits(imp_logits, target)

            total_loss = (
                self.w_recon * loss_recon +
                self.w_distill * loss_distill +
                self.w_rate * loss_rate +
                self.w_importance * loss_imp
            )
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_loss_sum += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())
            
        return total_loss_sum / len(self.train_loader)

    def run(self, epochs, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for epoch in range(epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
