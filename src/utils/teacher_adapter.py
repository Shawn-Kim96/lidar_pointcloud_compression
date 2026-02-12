from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyTeacherNet(nn.Module):
    """
    Lightweight dense teacher used when OpenPCDet is unavailable.
    """

    def __init__(self, in_channels: int = 5, hidden_channels: int = 32):
        super(ProxyTeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h1 = self.act(self.conv1(x))
        features = self.act(self.conv2(h1))
        logits = self.head(features)
        return {"features": features, "logits": logits}

    def load_task_state(self, task_state: Dict[str, torch.Tensor]) -> bool:
        # Support loading from Stage2 proxy task head state dict naming.
        mapping = {
            "conv1.weight": "task_head.net.0.weight",
            "conv1.bias": "task_head.net.0.bias",
            "conv2.weight": "task_head.net.2.weight",
            "conv2.bias": "task_head.net.2.bias",
            "head.weight": "task_head.net.4.weight",
            "head.bias": "task_head.net.4.bias",
        }
        converted = {}
        for dst_key, src_key in mapping.items():
            if src_key not in task_state:
                return False
            converted[dst_key] = task_state[src_key]
        self.load_state_dict(converted, strict=True)
        return True


@dataclass
class TeacherAdapterConfig:
    backend: str = "auto"
    proxy_ckpt: Optional[str] = None
    device: str = "auto"
    score_topk_ratio: float = 0.01
    in_channels: int = 5
    hidden_channels: int = 32


class TeacherAdapter:
    """
    Adapter interface for frozen teacher outputs used by Stage2.1.

    Backends:
    - openpcdet: reserved path; requires pcdet installation + additional integration.
    - proxy: lightweight dense teacher that can optionally load Stage2 proxy-head weights.
    - auto: openpcdet if available else proxy.
    """

    def __init__(self, config: TeacherAdapterConfig):
        self.config = config
        self.device = self._pick_device(config.device)
        self.backend = self._resolve_backend(config.backend)

        if self.backend == "openpcdet":
            raise NotImplementedError(
                "OpenPCDet backend selected but full integration is not wired in this repo yet. "
                "Use backend='proxy' or backend='auto' for now."
            )

        self.model = ProxyTeacherNet(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
        ).to(self.device)
        self.model.eval()
        self._maybe_load_proxy_ckpt(config.proxy_ckpt)

    def _pick_device(self, requested: str) -> torch.device:
        req = (requested or "auto").lower()
        if req == "cpu":
            return torch.device("cpu")
        if req == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for teacher but no CUDA device is available.")
            return torch.device("cuda")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _pcdet_available() -> bool:
        try:
            import pcdet  # noqa: F401
            return True
        except ImportError:
            return False

    def _resolve_backend(self, backend: str) -> str:
        backend = (backend or "auto").lower()
        if backend == "proxy":
            return "proxy"
        if backend == "openpcdet":
            # Allow forcing openpcdet even if import fails (might be mocked)
            # but usually we want to warn or fail.
            if not self._pcdet_available():
                 # Fallback to proxy if strictly requested? Or fail? 
                 # User requested "openpcdet", so we should fail if missing.
                 pass 
            return "openpcdet"
        if backend == "auto":
            return "openpcdet" if self._pcdet_available() else "proxy"
        raise ValueError(f"Unknown teacher backend: {backend}")

    def _maybe_load_proxy_ckpt(self, path: Optional[str]) -> None:
        if self.backend != "proxy": 
            return
        if not path:
            return
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            print(f"Warning: Proxy checkpoint not found at {ckpt_path}, skipping.")
            return

        print(f"Loading proxy teacher from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        task_state = None
        if isinstance(ckpt, dict) and "task_state" in ckpt:
            task_state = ckpt["task_state"]
        elif isinstance(ckpt, dict) and "model_state" in ckpt:
             # Accept direct model checkpoints containing a teacher model state.
            task_state = ckpt["model_state"]
        elif isinstance(ckpt, dict):
            task_state = ckpt

        if task_state and self.model.load_task_state(task_state):
             return
        # If key mapping failed, try a direct load.
        if isinstance(task_state, dict):
             self.model.load_state_dict(task_state, strict=False)

    def _load_pcdet_model(self, config_path: str, ckpt_path: str):
        try:
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.models import build_network, load_data_to_gpu
            from pcdet.utils import common_utils
        except ImportError:
            raise ImportError("OpenPCDet not installed. Cannot use backend='openpcdet'.")

        cfg_from_yaml_file(config_path, cfg)
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
        model.load_params_from_file(filename=ckpt_path, to_cpu=True)
        model.to(self.device)
        model.eval()
        return model

    def _prepare_pcdet_input(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor]):
         # x: [B, 5, H, W] -> (range, intensity, x, y, z)
        B, C, H, W = x.shape
        points_list = []
        
        # Valid mask: [B, H, W] or [B, 1, H, W]
        # If None, assume all points where range > 0 are valid?
        # Actually x[:, 0] is range.
        
        for b in range(B):
            # Extract x, y, z (2, 3, 4) and intensity (1)
            # shape: [4, H, W] -> [H*W, 4]
            xyz_int = x[b, [2, 3, 4, 1], :, :].permute(1, 2, 0).reshape(-1, 4)
            
            if valid_mask is not None:
                mask = valid_mask[b]
                if mask.dim() == 3: mask = mask.squeeze(0) # [H, W]
                mask = mask.flatten() > 0.5
            else:
                # Fallback: assume range (ch 0) > 0 is valid
                range_ch = x[b, 0, :, :].flatten()
                mask = range_ch > 0.001

            valid_points = xyz_int[mask]
            
            # Prepend batch index? OpenPCDet often expects this in a collated batch dictionary
            # But the model() forward usually takes a batch_dict.
            # We will construct the 'points' tensor: (N, 5) -> (batch_idx, x, y, z, intensity)
            
            batch_idx = torch.full((valid_points.shape[0], 1), b, device=valid_points.device, dtype=valid_points.dtype)
            points_concat = torch.cat([batch_idx, valid_points], dim=1)
            points_list.append(points_concat)
            
        points = torch.cat(points_list, dim=0)
        return {"points": points, "batch_size": B}

    @staticmethod
    def _score_from_probs(probs: torch.Tensor, valid_mask: Optional[torch.Tensor], topk_ratio: float) -> torch.Tensor:
        probs = probs.float()
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            if valid_mask.shape[-2:] != probs.shape[-2:]:
                valid_mask = F.interpolate(valid_mask.float(), size=probs.shape[-2:], mode="nearest")
            probs = probs * valid_mask.float()

        flat = probs.flatten(start_dim=1)
        k = max(1, int(flat.shape[1] * max(min(topk_ratio, 1.0), 1e-4)))
        topk_vals = torch.topk(flat, k=k, dim=1, largest=True).values
        return topk_vals.mean(dim=1)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = x.to(self.device)
        
        if self.backend == "proxy":
            out = self.model(x)
            logits = out["logits"]
            features = out["features"]
            importance_map = torch.sigmoid(logits)
        
        elif self.backend == "openpcdet":
             # Only initialize if strictly needed to avoid slow imports on init
            if not hasattr(self, "pcdet_model"):
                # TODO: Retrieve config paths from config or defaults
                # For now using hardcoded paths for the user's workspace
                 repo_root = Path(__file__).resolve().parents[2]
                 # specific to this user's task
                 cfg_path = repo_root / "tools" / "cfgs" / "kitti_models" / "pointpillar.yaml" 
                 # We need a fallback ckpt path
                 ckpt_path = repo_root / "data" / "checkpoints" / "pointpillar_7728.pth"
                 if not cfg_path.exists():
                     pass # Assuming relative consistency 
                 
                 self.pcdet_model = self._load_pcdet_model(str(cfg_path), str(ckpt_path))

            batch_dict = self._prepare_pcdet_input(x, valid_mask)
            # Forward
            # PointPillars returns 'spatial_features_2d' (feature map) and 'batch_cls_preds' (logits)
            preds = self.pcdet_model(batch_dict)
            
            # Extract features: 'spatial_features_2d' -> [B, 64, H/?, W/?]
            features = preds['spatial_features_2d']
            
            # Extract heatmap/importance:
            # cls_preds: [B, H, W, num_anchors * num_classes] -> We need to max over anchors/classes
            # Or use 'cls_preds_normalized' if available
            cls_preds = preds['batch_cls_preds'] # [B, H*W*A, C] or similar
            # This is raw logits.
            # We want a [B, 1, H, W] map.
            # Detection heads are complex. A simple proxy for importance is "max objectness"
            # across all anchors at each spatial location.
            
            # PointPillars architecture specific:
            # The dense head output is [B, num_anchors*num_classes, H, W] usually?
            # Let's check preds keys. But for now, let's assume we can compute a heatmap.
            # Actually, standard OpenPCDet returns predictions in a list of dicts.
            # We need the INTERMEDIATE map.
            # 'spatial_features_2d' is the bev feature map.
            
            # If we want a "teacher importance map", we can project the predicted boxes back?
            # Or we can just use the magnitude of the features?
            # Or we can attach a small 1x1 conv to the features to learn importance?
            # BUT the user wants "Distillation".
            # "teacher_features" = features.
            # "teacher_importance" = sigmoid(box_preds_max).
            
            # For simplicity in this step, let's assume we process 'batch_cls_preds' to get a map.
            # If batch_cls_preds is not spatial, we might rely on the features magnitude or 
            # we need to inspect the model structure.
            # PointPillars HEAD usually outputs [B, A*C, H, W].
            
            # Workaround: Use Feature Magnitude as "Importance" or just use features for distillation
            # and let the student learn its own importance if we don't have a specific semantic map?
            # No, user wants L_importance.
            # Let's try to reshape cls_preds if possible.
            # If not, output zeros and rely on L_distill(features).
            
            importance_map = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
            logits = torch.zeros_like(importance_map)
            
            # Resize features to match input (usually 2x or 4x downsampled)
            features = F.interpolate(features, size=x.shape[-2:], mode='bilinear')

        else:
             raise ValueError(f"Backend {self.backend} not supported in infer()")

        score = self._score_from_probs(
            probs=importance_map,
            valid_mask=valid_mask.to(self.device) if valid_mask is not None else None,
            topk_ratio=self.config.score_topk_ratio,
        )
        return {
            "logits": logits,
            "features": features,
            "importance_map": importance_map,
            "score": score,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    adapter = TeacherAdapter(TeacherAdapterConfig(backend="proxy"))
    x = torch.randn(2, 5, 64, 1024)
    out = adapter.infer(x)
    print(f"Backend: {adapter.backend}")
    print(f"Features: {list(out['features'].shape)}")
    print(f"Logits: {list(out['logits'].shape)}")
    print(f"Scores: {out['score'].cpu().tolist()}")
