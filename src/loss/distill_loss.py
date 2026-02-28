from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() != 4:
        raise ValueError(f"Expected 3D/4D tensor, got shape {list(x.shape)}")
    return x


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    x = _ensure_4d(x)
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def _parse_hw(value: Union[str, Tuple[int, int], list, None]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        h, w = int(value[0]), int(value[1])
        return (h, w) if h > 0 and w > 0 else None
    if isinstance(value, str):
        text = value.strip().lower().replace("x", ",")
        if not text:
            return None
        toks = [t.strip() for t in text.split(",") if t.strip()]
        if len(toks) != 2:
            return None
        h, w = int(toks[0]), int(toks[1])
        return (h, w) if h > 0 and w > 0 else None
    return None


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred = _ensure_4d(pred)
    target = _resize_like(target, pred)
    diff = (pred - target) ** 2
    if weight is None:
        return diff.mean()
    w = _resize_like(weight, pred).clamp(min=0.0)
    return (diff * w).sum() / w.sum().clamp(min=1.0)


def weighted_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    student_logits = _ensure_4d(student_logits)
    teacher_logits = _resize_like(teacher_logits, student_logits)
    t = float(max(temperature, 1e-6))

    student_log_probs = F.log_softmax(student_logits / t, dim=1)
    teacher_probs = F.softmax(teacher_logits / t, dim=1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none") * (t * t)

    if weight is None:
        return kl.mean()
    w = _resize_like(weight, student_logits).clamp(min=0.0)
    return (kl * w).sum() / w.sum().clamp(min=1.0)


def weighted_bce_with_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    student_logits = _ensure_4d(student_logits)
    teacher_logits = _resize_like(teacher_logits, student_logits)
    t = float(max(temperature, 1e-6))

    teacher_prob = torch.sigmoid(teacher_logits / t)
    bce = F.binary_cross_entropy_with_logits(student_logits / t, teacher_prob, reduction="none") * (t * t)
    if weight is None:
        return bce.mean()
    w = _resize_like(weight, student_logits).clamp(min=0.0)
    return (bce * w).sum() / w.sum().clamp(min=1.0)


def weighted_logit_mse(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    student_logits = _ensure_4d(student_logits)
    teacher_logits = _resize_like(teacher_logits, student_logits)
    return weighted_mse(student_logits, teacher_logits, weight=weight)


class DistillLoss(nn.Module):
    """
    Composite distillation objective over teacher/student feature/logit maps.
    """

    def __init__(
        self,
        feature_weight: float = 1.0,
        logit_weight: float = 1.0,
        temperature: float = 1.0,
        loss_type: str = "mse",
        logit_loss_type: str = "auto",
        align_mode: str = "resize",
        align_hw: Optional[Union[str, Tuple[int, int], list]] = None,
    ):
        super(DistillLoss, self).__init__()
        self.feature_weight = float(feature_weight)
        self.logit_weight = float(logit_weight)
        self.temperature = float(temperature)
        self.loss_type = str(loss_type).lower()
        self.logit_loss_type = str(logit_loss_type).lower()
        self.align_mode = str(align_mode).lower()
        self.align_hw = _parse_hw(align_hw)
        if self.loss_type not in ("mse", "l1"):
            raise ValueError("loss_type must be one of: mse, l1")
        if self.logit_loss_type not in ("auto", "kl", "bce", "mse"):
            raise ValueError("logit_loss_type must be one of: auto, kl, bce, mse")
        if self.align_mode not in ("resize", "adaptive_pool"):
            raise ValueError("align_mode must be one of: resize, adaptive_pool")

    def _resolve_align_hw(self, student: torch.Tensor, teacher: torch.Tensor) -> Tuple[int, int]:
        if self.align_hw is not None:
            return self.align_hw
        return (
            min(int(student.shape[-2]), int(teacher.shape[-2])),
            min(int(student.shape[-1]), int(teacher.shape[-1])),
        )

    def _align_pair(self, student: torch.Tensor, teacher: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        student = _ensure_4d(student)
        teacher = _ensure_4d(teacher)
        if self.align_mode == "resize":
            return student, _resize_like(teacher, student)
        hw = self._resolve_align_hw(student, teacher)
        student = F.adaptive_avg_pool2d(student, output_size=hw)
        teacher = F.adaptive_avg_pool2d(teacher, output_size=hw)
        return student, teacher

    def _align_weight(self, weight: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
        if weight is None:
            return None
        w = _ensure_4d(weight)
        if self.align_mode == "adaptive_pool":
            w = F.adaptive_avg_pool2d(w, output_size=ref.shape[-2:])
            return w.clamp(min=0.0)
        return _resize_like(w, ref).clamp(min=0.0)

    def _feature_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        importance_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        student_features, teacher_features = self._align_pair(student_features, teacher_features)
        aligned_weight = self._align_weight(importance_map, student_features)
        if self.loss_type == "l1":
            diff = torch.abs(student_features - teacher_features)
            if aligned_weight is None:
                return diff.mean()
            w = aligned_weight
            return (diff * w).sum() / w.sum().clamp(min=1.0)
        return weighted_mse(student_features, teacher_features, weight=aligned_weight)

    def forward(
        self,
        *,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        importance_map: Optional[torch.Tensor] = None,
        return_details: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = None
        if student_features is not None:
            device = student_features.device
        elif student_logits is not None:
            device = student_logits.device
        else:
            raise ValueError("At least one distillation target must be provided.")
        total = torch.tensor(0.0, device=device)
        details: Dict[str, float] = {
            "feature_distill": 0.0,
            "logit_distill": 0.0,
            "distill_total": 0.0,
        }

        if student_features is not None and teacher_features is not None and self.feature_weight > 0.0:
            f_loss = self._feature_loss(student_features, teacher_features, importance_map)
            total = total + (self.feature_weight * f_loss)
            details["feature_distill"] = float(f_loss.detach().item())

        if student_logits is not None and teacher_logits is not None and self.logit_weight > 0.0:
            student_logits, teacher_logits = self._align_pair(student_logits, teacher_logits)
            aligned_weight = self._align_weight(importance_map, student_logits)

            logit_loss_type = self.logit_loss_type
            if logit_loss_type == "auto":
                # For one-channel logits, KL over channel dim is degenerate; use BCE distillation.
                logit_loss_type = "bce" if student_logits.shape[1] == 1 else "kl"

            if logit_loss_type == "kl":
                l_loss = weighted_kl(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    temperature=self.temperature,
                    weight=aligned_weight,
                )
            elif logit_loss_type == "bce":
                l_loss = weighted_bce_with_logits(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    temperature=self.temperature,
                    weight=aligned_weight,
                )
            else:
                l_loss = weighted_logit_mse(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    weight=aligned_weight,
                )
            total = total + (self.logit_weight * l_loss)
            details["logit_distill"] = float(l_loss.detach().item())

        details["distill_total"] = float(total.detach().item())
        if return_details:
            return total, details
        return total, {}


if __name__ == "__main__":
    torch.manual_seed(0)
    loss_fn = DistillLoss(feature_weight=1.0, logit_weight=0.5, temperature=2.0)
    student_feat = torch.randn(2, 32, 8, 16, requires_grad=True)
    teacher_feat = torch.randn(2, 32, 8, 16)
    student_logits = torch.randn(2, 4, 8, 16, requires_grad=True)
    teacher_logits = torch.randn(2, 4, 8, 16)
    importance = torch.rand(2, 1, 8, 16)

    total, info = loss_fn(
        student_features=student_feat,
        teacher_features=teacher_feat,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        importance_map=importance,
        return_details=True,
    )
    total.backward()
    print(f"Distill total: {float(total.item()):.6f}")
    print(info)
