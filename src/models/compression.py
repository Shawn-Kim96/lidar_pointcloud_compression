import torch
import torch.nn as nn
from models.registry import BACKBONES, MODELS
from models.quantization import AdaptiveQuantizer
from models.autoencoder import Decoder, _norm, _activation, Encoder, QuantizationLayer
from models.importance_head import ImportanceHead

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
        else:
            raise ValueError(
                f"Unsupported quantizer mode '{self.quantizer_mode}'. "
                "Use one of: adaptive, uniform"
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
        self.decoder = Decoder(**decoder_config)
        
    def forward(self, x, noise_std=0.0, quantize=None, **kwargs):
        # x: [B, 5, H, W]
        stage_features = None
        if self.importance_head is not None and getattr(self.importance_head, "requires_multiscale", False):
            try:
                backbone_out = self.backbone(x, return_features=True)
            except TypeError:
                backbone_out = self.backbone(x)
            if isinstance(backbone_out, tuple):
                features, stage_features = backbone_out
            else:
                features = backbone_out
        else:
            features = self.backbone(x)
        latent = self.feature_projection(features)
        
        aux = {
            "latent": latent,
            "quantizer_mode": self.quantizer_mode,
        }
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
             else:
                 # Uniform per-sample quantization.
                 latent_deq, q = self.quantizer(latent_noisy)
                 aux["codes"] = q
        else:
             latent_deq = latent_noisy

        recon = self.decoder(latent_deq)
        return recon, aux

def build_model(config):
    return MODELS.build(config)
