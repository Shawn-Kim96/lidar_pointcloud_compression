import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import BACKBONES, MODELS
from models.quantization import AdaptiveQuantizer
from models.autoencoder import (
    CoordConditionedDecoder,
    Decoder,
    Encoder,
    QuantizationLayer,
    SkipCoordConditionedDecoder,
    SkipDecoder,
    _activation,
    _norm,
)
from models.importance_head import ImportanceHead
from models.pillar_side import DynamicPillarBEVSideStream
from models.stage3_necks import Stage3MultiScaleFusion

# Register ResNet Encoder (temporarily until migrated)
BACKBONES.register("resnet")(Encoder)

@MODELS.register("lidar_compression")
class LidarCompressionModel(nn.Module):
    def __init__(
        self,
        backbone_config,
        quantizer_config,
        decoder_config,
        head_config=None,
        feature_fusion_config=None,
        position_branch_config=None,
        pillar_side_config=None,
    ):
        super().__init__()
        
        # 1. Build Backbone
        # Config example: {name: "darknet", in_channels: 5, ...}
        self.backbone = BACKBONES.build(backbone_config)
        
        latent_channels = quantizer_config.get("latent_channels", 64)
        if "latent_channels" not in quantizer_config:
             # If not in quantizer config (it shouldn't be for adaptive), check decoder?
             # Or assume 64.
             latent_channels = decoder_config.get("latent_channels", 64)
        self.latent_channels = int(latent_channels)

        # 2. Project Features if needed
        # DarkNet ends with 1024 channels, we might want 64
        if hasattr(self.backbone, "out_channels") and self.backbone.out_channels != latent_channels:
            print(f"Projecting backbone output {self.backbone.out_channels} -> {latent_channels}")
            self.feature_projection = nn.Sequential(
                nn.Conv2d(self.backbone.out_channels, latent_channels, kernel_size=1, bias=False),
                _norm("batch", latent_channels),
                _activation("relu")
            )
        else:
            self.feature_projection = nn.Identity()

        # 2.25 Optional position side-branch for xyz-aware latent context.
        self.position_backbone = None
        self.position_projection = None
        self.position_fuse = None
        self.position_latent_channels = 0
        self.position_branch_enabled = False
        if position_branch_config is not None and position_branch_config.get("enabled", False):
            pos_cfg = dict(position_branch_config)
            pos_in_channels = int(pos_cfg.get("in_channels", 3))
            pos_latent_channels = int(pos_cfg.get("latent_channels", max(16, self.latent_channels // 2)))
            pos_backbone_cfg = dict(backbone_config)
            pos_backbone_cfg["in_channels"] = pos_in_channels
            if pos_backbone_cfg.get("name") == "resnet":
                pos_backbone_cfg["latent_channels"] = pos_latent_channels
            self.position_backbone = BACKBONES.build(pos_backbone_cfg)
            if hasattr(self.position_backbone, "out_channels") and self.position_backbone.out_channels != pos_latent_channels:
                self.position_projection = nn.Sequential(
                    nn.Conv2d(self.position_backbone.out_channels, pos_latent_channels, kernel_size=1, bias=False),
                    _norm("batch", pos_latent_channels),
                    _activation("relu"),
                )
            else:
                self.position_projection = nn.Identity()
            self.position_fuse = nn.Sequential(
                nn.Conv2d(self.latent_channels + pos_latent_channels, self.latent_channels, kernel_size=1, bias=False),
                _norm("batch", self.latent_channels),
                _activation("relu"),
            )
            self.position_latent_channels = pos_latent_channels
            self.position_branch_enabled = True

        # 2.75 Optional raw-point pillar/BEV side stream.
        self.pillar_side = None
        self.pillar_side_enabled = False
        self.pillar_stage_proj = None
        self.pillar_stage_gate = None
        self.pillar_stage_channels = []
        if pillar_side_config is not None and pillar_side_config.get("enabled", False):
            if backbone_config.get("name") != "resnet":
                raise ValueError("pillar_side_config currently supports the resnet encoder only.")
            pillar_cfg = dict(pillar_side_config)
            self.pillar_side = DynamicPillarBEVSideStream(
                point_cloud_range=pillar_cfg.get("point_cloud_range", [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]),
                pillar_size=pillar_cfg.get("pillar_size", [0.24, 0.24]),
                pfn_hidden_channels=int(pillar_cfg.get("pfn_hidden_channels", 64)),
                pfn_out_channels=int(pillar_cfg.get("pfn_out_channels", 128)),
                bev_channels=pillar_cfg.get("bev_channels", [128, 128, 192, 256]),
                bev_blocks=pillar_cfg.get("bev_blocks", [2, 2, 2, 3]),
                fpn_channels=int(pillar_cfg.get("fpn_channels", 128)),
                max_raw_points=int(pillar_cfg.get("max_raw_points", 150000)),
                norm=str(pillar_cfg.get("norm", "batch")),
                activation=str(pillar_cfg.get("activation", "silu")),
            )
            self.pillar_stage_channels = list(getattr(self.backbone, "stage_channels", []))
            side_ch = int(pillar_cfg.get("fpn_channels", 128))
            self.pillar_stage_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(side_ch, int(stage_ch), kernel_size=1, bias=False),
                    _norm("batch", int(stage_ch)),
                )
                for stage_ch in self.pillar_stage_channels
            ])
            self.pillar_stage_gate = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(side_ch, int(stage_ch), kernel_size=1, bias=False),
                    nn.Sigmoid(),
                )
                for stage_ch in self.pillar_stage_channels
            ])
            self.pillar_side_enabled = True

        # 2.5 Optional multiscale latent fusion before quantization.
        self.feature_fusion = None
        if feature_fusion_config is not None:
            fusion_cfg = dict(feature_fusion_config)
            if not fusion_cfg.get("enabled", True):
                fusion_cfg = None
            if fusion_cfg is not None:
                variant = str(fusion_cfg.get("variant", "rangeformer")).lower()
                hidden_channels = int(fusion_cfg.get("hidden_channels", latent_channels))
                stage_channels = fusion_cfg.get("in_channels_list", None)
                if stage_channels is None:
                    stage_channels = getattr(self.backbone, "stage_channels", None)
                if not stage_channels:
                    raise ValueError("feature_fusion_config requires multiscale stage_channels from backbone.")
                self.feature_fusion = Stage3MultiScaleFusion(
                    in_channels_list=list(stage_channels),
                    hidden_channels=hidden_channels,
                    variant=variant,
                    activation=fusion_cfg.get("activation", "relu"),
                )
                if hidden_channels != latent_channels:
                    self.feature_fusion_out = nn.Sequential(
                        nn.Conv2d(hidden_channels, latent_channels, kernel_size=1, bias=False),
                        _norm(fusion_cfg.get("norm", "batch"), latent_channels),
                        _activation(fusion_cfg.get("activation", "relu")),
                    )
                else:
                    self.feature_fusion_out = nn.Identity()
            else:
                self.feature_fusion_out = nn.Identity()
        else:
            self.feature_fusion_out = nn.Identity()

        # 3. Build Quantizer
        self.quantizer_mode = str(quantizer_config.get("mode", "adaptive")).lower()
        if self.quantizer_mode == "adaptive":
            self.quantizer = AdaptiveQuantizer(
                roi_levels=quantizer_config.get("roi_levels", 256),
                bg_levels=quantizer_config.get("bg_levels", 16),
                eps=quantizer_config.get("eps", 1e-6),
                use_ste=quantizer_config.get("use_ste", False),
            )
        elif self.quantizer_mode == "uniform":
            bits = int(quantizer_config.get("uniform_bits", quantizer_config.get("quant_bits", 8)))
            self.quantizer = QuantizationLayer(
                bits=bits,
                eps=quantizer_config.get("eps", 1e-6),
                use_ste=quantizer_config.get("use_ste", False),
            )
        elif self.quantizer_mode == "none":
            self.quantizer = None
        else:
            raise ValueError(
                f"Unsupported quantizer mode '{self.quantizer_mode}'. "
                "Use one of: adaptive, uniform, none"
            )
        
        # 4. Importance Head
        self.importance_head = None
        if head_config is not None and self.quantizer_mode == "adaptive":
            head_cfg = dict(head_config)
            if "multiscale_in_channels" not in head_cfg:
                stage_channels = getattr(self.backbone, "stage_channels", None)
                if stage_channels is not None:
                    head_cfg["multiscale_in_channels"] = list(stage_channels)
            # Head takes latent features, and can optionally consume multi-stage taps.
            self.importance_head = ImportanceHead(in_channels=latent_channels, **head_cfg)
        
        # 5. Build Decoder
        self.decoder_type = str(decoder_config.get("decoder_type", "deconv")).lower()
        decoder_cfg = dict(decoder_config)
        encoder_stage_channels = list(getattr(self.backbone, "stage_channels", []))
        if self.decoder_type == "coord_conditioned":
            decoder_cfg.pop("decoder_type", None)
            decoder_cfg.pop("fov_up_deg", None)
            decoder_cfg.pop("fov_down_deg", None)
            self.decoder = CoordConditionedDecoder(
                **decoder_cfg,
                position_condition_channels=self.position_latent_channels,
            )
        elif self.decoder_type == "skip_unet":
            decoder_cfg.pop("decoder_type", None)
            decoder_cfg.pop("coord_channels", None)
            decoder_cfg.pop("implicit_hidden_channels", None)
            decoder_cfg.pop("fov_up_deg", None)
            decoder_cfg.pop("fov_down_deg", None)
            self.decoder = SkipDecoder(
                **decoder_cfg,
                encoder_stage_channels=encoder_stage_channels,
            )
        elif self.decoder_type == "skip_coord_conditioned":
            decoder_cfg.pop("decoder_type", None)
            decoder_cfg.pop("fov_up_deg", None)
            decoder_cfg.pop("fov_down_deg", None)
            self.decoder = SkipCoordConditionedDecoder(
                **decoder_cfg,
                encoder_stage_channels=encoder_stage_channels,
                position_condition_channels=self.position_latent_channels,
            )
        else:
            decoder_cfg.pop("decoder_type", None)
            decoder_cfg.pop("coord_channels", None)
            decoder_cfg.pop("implicit_hidden_channels", None)
            decoder_cfg.pop("fov_up_deg", None)
            decoder_cfg.pop("fov_down_deg", None)
            self.decoder = Decoder(**decoder_cfg)
        self.fov_up_deg = float(decoder_config.get("fov_up_deg", 3.0))
        self.fov_down_deg = float(decoder_config.get("fov_down_deg", -25.0))
        self.coord_channels = int(decoder_config.get("coord_channels", 5))
        self.register_buffer("_coord_cache", torch.empty(0), persistent=False)
        self._coord_cache_key = None

    def _build_coord_features(self, batch_size: int, height: int, width: int, device, dtype):
        key = (height, width, str(device), str(dtype), round(self.fov_up_deg, 4), round(self.fov_down_deg, 4), self.coord_channels)
        if self._coord_cache_key != key or self._coord_cache.numel() == 0:
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
            row_norm = 2.0 * rows[:, None].expand(height, width) - 1.0
            col_norm = 2.0 * cols[None, :].expand(height, width) - 1.0
            coord = torch.stack([ray_x, ray_y, ray_z, row_norm, col_norm], dim=0)
            if self.coord_channels < coord.shape[0]:
                coord = coord[: self.coord_channels]
            self._coord_cache = coord.to(dtype=dtype)
            self._coord_cache_key = key
        coord = self._coord_cache.unsqueeze(0)
        if coord.shape[0] != batch_size:
            coord = coord.expand(batch_size, -1, -1, -1)
        return coord

    @staticmethod
    def _encoder_stage_hws(height: int, width: int, num_stages: int):
        h, w = int(height), int(width)
        out = []
        for _ in range(int(num_stages)):
            h = (h + 1) // 2
            w = (w + 1) // 2
            out.append((h, w))
        return out
        
    def forward(self, x, noise_std=0.0, quantize=None, **kwargs):
        # x: [B, 5, H, W]
        stage_features = None
        pillar_stage_additions = None
        need_skip_features = self.decoder_type in ("skip_unet", "skip_coord_conditioned")
        need_multiscale = need_skip_features or self.pillar_side_enabled or self.feature_fusion is not None or (
            self.importance_head is not None and getattr(self.importance_head, "requires_multiscale", False)
        )
        if self.pillar_side_enabled:
            raw_points = kwargs.get("raw_points", None)
            raw_point_counts = kwargs.get("raw_point_counts", None)
            if raw_points is None:
                raise ValueError("pillar_side branch requires raw_points tensor in model forward.")
            if raw_point_counts is None:
                raw_point_counts = torch.full(
                    (raw_points.shape[0],),
                    raw_points.shape[1],
                    device=raw_points.device,
                    dtype=torch.long,
                )
            stage_hws = self._encoder_stage_hws(x.shape[-2], x.shape[-1], len(self.pillar_stage_channels))
            side_feats, side_aux = self.pillar_side(
                raw_points=raw_points,
                raw_point_counts=raw_point_counts,
                ri_xyz=x[:, 2:5],
                ri_valid=(x[:, 0:1] > 0.0),
                target_stage_hws=stage_hws,
            )
            pillar_stage_additions = []
            for idx, feat in enumerate(side_feats):
                proj = self.pillar_stage_proj[idx](feat)
                gate = self.pillar_stage_gate[idx](feat)
                pillar_stage_additions.append(proj * gate)
        else:
            side_aux = None
        if need_multiscale:
            try:
                backbone_out = self.backbone(x, return_features=True, stage_additions=pillar_stage_additions)
            except TypeError:
                backbone_out = self.backbone(x)
            if isinstance(backbone_out, tuple):
                features, stage_features = backbone_out
            else:
                features = backbone_out
        else:
            features = self.backbone(x)
        latent = self.feature_projection(features)
        position_latent = None
        if self.position_branch_enabled:
            pos_input = x[:, 2:5]
            pos_out = self.position_backbone(pos_input)
            position_latent = self.position_projection(pos_out)
            if position_latent.shape[-2:] != latent.shape[-2:]:
                position_latent = F.interpolate(
                    position_latent,
                    size=latent.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            latent = self.position_fuse(torch.cat([latent, position_latent], dim=1))
        if self.feature_fusion is not None:
            latent = self.feature_fusion(latent, multiscale_features=stage_features)
            latent = self.feature_fusion_out(latent)
        
        aux = {
            "latent": latent,
            "quantizer_mode": self.quantizer_mode,
        }
        if side_aux is not None:
            aux.update(side_aux)
        if position_latent is not None:
            aux["position_latent"] = position_latent
        importance_map = kwargs.get("importance_map", None)
        
        # If we have a head, predict importance
        if self.importance_head is not None:
            # We predict from latent
            pred_imp, pred_logits = self.importance_head(latent, multiscale_features=stage_features)
            aux["importance_logits"] = pred_logits
            aux["importance_map_pred"] = pred_imp
            
            # If no external importance provided, use predicted
            if importance_map is None:
                importance_map = pred_imp
        
        # Quantization
        if self.training and noise_std > 0.0:
             noise = torch.randn_like(latent) * noise_std
             latent_noisy = latent + noise
        else:
             latent_noisy = latent

        if quantize is None:
            quantize = not self.training

        if quantize:
             if self.quantizer_mode == "adaptive":
                 # AdaptiveQuantizer requires importance_map
                 if importance_map is None:
                     # Fallback if no head and no external map.
                     importance_map = torch.ones((latent.shape[0], 1, latent.shape[2], latent.shape[3]), device=latent.device)
                 
                 latent_deq, codes, level_map = self.quantizer(latent_noisy, importance_map)
                 aux["codes"] = codes
                 aux["level_map"] = level_map
                 aux["importance_map_used"] = importance_map
             elif self.quantizer_mode == "uniform":
                 # Uniform per-sample quantization.
                 latent_deq, q = self.quantizer(latent_noisy)
                 aux["codes"] = q
             else:
                 # No quantizer path: keep latent as-is.
                 latent_deq = latent_noisy
                 aux["quantization_bypassed"] = True
        else:
             latent_deq = latent_noisy

        if self.decoder_type == "coord_conditioned":
            coord_features = self._build_coord_features(
                batch_size=x.shape[0],
                height=x.shape[-2],
                width=x.shape[-1],
                device=x.device,
                dtype=latent_deq.dtype,
            )
            recon = self.decoder(
                latent_deq,
                coord_features=coord_features,
                position_context=position_latent,
            )
            aux["coord_features"] = coord_features
        elif self.decoder_type == "skip_coord_conditioned":
            coord_features = self._build_coord_features(
                batch_size=x.shape[0],
                height=x.shape[-2],
                width=x.shape[-1],
                device=x.device,
                dtype=latent_deq.dtype,
            )
            recon = self.decoder(
                latent_deq,
                coord_features=coord_features,
                position_context=position_latent,
                skip_features=stage_features,
            )
            aux["coord_features"] = coord_features
        elif self.decoder_type == "skip_unet":
            recon = self.decoder(latent_deq, skip_features=stage_features)
        else:
            recon = self.decoder(latent_deq)
        return recon, aux

def build_model(config):
    return MODELS.build(config)
