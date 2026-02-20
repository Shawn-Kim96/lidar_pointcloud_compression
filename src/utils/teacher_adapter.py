from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyTeacherNet(nn.Module):
    """
    Lightweight dense teacher used when full detector teacher integration is unavailable.
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


class _PPBackboneBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, num_layers: int, stride: int):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, bias=False, padding=1),
            nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers):
            blocks.extend(
                [
                    nn.Conv2d(out_channel, out_channel, 3, bias=False, padding=1),
                    nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ]
            )
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _PPNeckBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, stride, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ZhulfPointPillarsTeacherNet(nn.Module):
    """
    Dense distillation adapter that reuses zhulf0804/PointPillars checkpoint
    weights (backbone/neck/head) on engineered range-image features.
    """

    def __init__(self):
        super().__init__()
        self.input_hw = (496, 432)  # KITTI BEV map size from the original model.

        # checkpoint key: pillar_encoder.conv.weight [64, 9, 1]
        self.stem_conv = nn.Conv2d(9, 64, kernel_size=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.stem_act = nn.ReLU(inplace=True)

        self.backbone_blocks = nn.ModuleList(
            [
                _PPBackboneBlock(64, 64, num_layers=3, stride=2),
                _PPBackboneBlock(64, 128, num_layers=5, stride=2),
                _PPBackboneBlock(128, 256, num_layers=5, stride=2),
            ]
        )
        self.neck_blocks = nn.ModuleList(
            [
                _PPNeckBlock(64, 128, stride=1),
                _PPNeckBlock(128, 128, stride=2),
                _PPNeckBlock(256, 128, stride=4),
            ]
        )
        self.head_cls = nn.Conv2d(384, 18, kernel_size=1)  # 6 anchors * 3 classes
        self.head_reg = nn.Conv2d(384, 42, kernel_size=1)  # 6 anchors * 7
        self.head_dir = nn.Conv2d(384, 12, kernel_size=1)  # 6 anchors * 2

    def _to_9ch(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B, 5, H, W] => engineered 9ch tensor.
        rng = x[:, 0:1]
        inten = x[:, 1:2]
        xyz = x[:, 2:5]
        dist = torch.norm(xyz, dim=1, keepdim=True)

        if valid_mask is None:
            valid = (rng > 1e-3).float()
        else:
            valid = valid_mask.unsqueeze(1).float() if valid_mask.dim() == 3 else valid_mask.float()

        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([rng, inten, xyz, dist, valid, yy, xx], dim=1)

    def load_zhulf_checkpoint(self, ckpt_path: Path):
        payload = torch.load(str(ckpt_path), map_location="cpu")
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected checkpoint format for zhulf PointPillars: {type(payload)}")

        with torch.no_grad():
            conv_w = state.get("pillar_encoder.conv.weight", None)
            if conv_w is not None and tuple(conv_w.shape) == (64, 9, 1):
                self.stem_conv.weight.copy_(conv_w.unsqueeze(-1))
            for attr in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                key = f"pillar_encoder.bn.{attr}"
                if key in state and hasattr(self.stem_bn, attr):
                    getattr(self.stem_bn, attr).copy_(state[key])

            # Backbone
            for bi, block in enumerate(self.backbone_blocks):
                for li, layer in enumerate(block.block):
                    if isinstance(layer, nn.Conv2d):
                        key = f"backbone.multi_blocks.{bi}.{li}.weight"
                        if key in state:
                            layer.weight.copy_(state[key])
                    elif isinstance(layer, nn.BatchNorm2d):
                        for attr in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                            key = f"backbone.multi_blocks.{bi}.{li}.{attr}"
                            if key in state and hasattr(layer, attr):
                                getattr(layer, attr).copy_(state[key])

            # Neck
            for ni, block in enumerate(self.neck_blocks):
                for li, layer in enumerate(block.block):
                    prefix = f"neck.decoder_blocks.{ni}.{li}"
                    if isinstance(layer, nn.ConvTranspose2d):
                        key = f"{prefix}.weight"
                        if key in state:
                            layer.weight.copy_(state[key])
                    elif isinstance(layer, nn.BatchNorm2d):
                        for attr in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                            key = f"{prefix}.{attr}"
                            if key in state and hasattr(layer, attr):
                                getattr(layer, attr).copy_(state[key])

            # Head
            for name, layer in [
                ("conv_cls", self.head_cls),
                ("conv_reg", self.head_reg),
                ("conv_dir_cls", self.head_dir),
            ]:
                w_key = f"head.{name}.weight"
                b_key = f"head.{name}.bias"
                if w_key in state:
                    layer.weight.copy_(state[w_key])
                if b_key in state:
                    layer.bias.copy_(state[b_key])

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat9 = self._to_9ch(x, valid_mask=valid_mask)
        feat9 = F.interpolate(feat9, size=self.input_hw, mode="bilinear", align_corners=False)

        h = self.stem_act(self.stem_bn(self.stem_conv(feat9)))
        xs = []
        for blk in self.backbone_blocks:
            h = blk(h)
            xs.append(h)

        up = [blk(xi) for blk, xi in zip(self.neck_blocks, xs)]
        fused = torch.cat(up, dim=1)  # [B,384,248,216]
        cls_logits_full = self.head_cls(fused)  # [B,18,248,216]

        b, _, hh, ww = cls_logits_full.shape
        cls_logits_view = cls_logits_full.view(b, 6, 3, hh, ww)
        cls_logits = cls_logits_view.amax(dim=2).amax(dim=1, keepdim=True)  # [B,1,hh,ww]
        return {"features": fused, "logits": cls_logits}


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
    Adapter interface for frozen teacher outputs used by Stage2 training.

    Backends:
    - pointpillars_zhulf: uses zhulf0804/PointPillars checkpoint in dense adapter mode.
    - proxy: lightweight dense teacher with optional proxy task-head checkpoint.
    - openpcdet: reserved path (not wired in this repo).
    - auto: pointpillars_zhulf if pointpillars-style ckpt exists, else proxy.
    """

    def __init__(self, config: TeacherAdapterConfig):
        self.config = config
        self.device = self._pick_device(config.device)
        self.backend = self._resolve_backend(config.backend)

        if self.backend == "proxy":
            self.model = ProxyTeacherNet(
                in_channels=config.in_channels,
                hidden_channels=config.hidden_channels,
            ).to(self.device)
            self.model.eval()
            self._maybe_load_proxy_ckpt(config.proxy_ckpt)

        elif self.backend == "pointpillars_zhulf":
            self.model = ZhulfPointPillarsTeacherNet().to(self.device)
            self.model.eval()
            self._load_zhulf_ckpt(config.proxy_ckpt)

        elif self.backend == "openpcdet":
            raise NotImplementedError(
                "OpenPCDet backend selected but full integration is not wired in this repo yet. "
                "Use backend='pointpillars_zhulf' or backend='proxy'."
            )

        else:
            raise ValueError(f"Unknown teacher backend: {self.backend}")

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
        except ImportError:
            return False

    def _resolve_backend(self, backend: str) -> str:
        backend = (backend or "auto").lower()
        if backend in ("proxy", "pointpillars_zhulf", "openpcdet"):
            return backend
        if backend == "auto":
            if self.config.proxy_ckpt and "pointpillars" in str(self.config.proxy_ckpt).lower():
                return "pointpillars_zhulf"
            return "proxy"
        raise ValueError(f"Unknown teacher backend: {backend}")

    def _maybe_load_proxy_ckpt(self, path: Optional[str]) -> None:
        if self.backend != "proxy" or not path:
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
            task_state = ckpt["model_state"]
        elif isinstance(ckpt, dict):
            task_state = ckpt

        if task_state and self.model.load_task_state(task_state):
            return
        if isinstance(task_state, dict):
            self.model.load_state_dict(task_state, strict=False)

    def _load_zhulf_ckpt(self, path: Optional[str]) -> None:
        if self.backend != "pointpillars_zhulf":
            return
        if not path:
            raise ValueError("pointpillars_zhulf backend requires --teacher_proxy_ckpt path.")

        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"PointPillars checkpoint not found at {ckpt_path}")

        print(f"Loading zhulf PointPillars teacher from {ckpt_path}")
        self.model.load_zhulf_checkpoint(ckpt_path)

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
        vm = valid_mask.to(self.device) if valid_mask is not None else None

        if self.backend == "proxy":
            out = self.model(x)
            logits = out["logits"]
            features = out["features"]
            importance_map = torch.sigmoid(logits)

        elif self.backend == "pointpillars_zhulf":
            out = self.model(x, valid_mask=vm)
            logits = out["logits"]
            features = out["features"]
            importance_map = torch.sigmoid(logits)

        else:
            raise ValueError(f"Backend {self.backend} not supported in infer()")

        score = self._score_from_probs(
            probs=importance_map,
            valid_mask=vm,
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
