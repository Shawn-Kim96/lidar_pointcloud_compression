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

    @staticmethod
    def _pcdet_available() -> bool:
        try:
            import pcdet  # noqa: F401

            return True
        except Exception:
            return False

    def _resolve_backend(self, backend: str) -> str:
        backend = (backend or "auto").lower()
        if backend == "proxy":
            return "proxy"
        if backend == "openpcdet":
            if not self._pcdet_available():
                raise RuntimeError("backend='openpcdet' requested but pcdet is not installed.")
            return "openpcdet"
        if backend == "auto":
            return "openpcdet" if self._pcdet_available() else "proxy"
        raise ValueError(f"Unknown teacher backend: {backend}")

    def _maybe_load_proxy_ckpt(self, path: Optional[str]) -> None:
        if not path:
            return
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

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
        out = self.model(x)
        logits = out["logits"]
        features = out["features"]
        importance_map = torch.sigmoid(logits)
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
