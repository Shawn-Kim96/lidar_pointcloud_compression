import torch
import torch.nn as nn
from models.registry import BACKBONES, MODELS
from models.quantization import AdaptiveQuantizer
from models.autoencoder import Decoder, _norm, _activation, Encoder
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
        self.quantizer = AdaptiveQuantizer(**quantizer_config)
        
        # 4. Importance Head
        self.importance_head = None
        if head_config is not None:
            # Head takes latent features or projected features?
            # Usually projected features (latent_channels)
            self.importance_head = ImportanceHead(in_channels=latent_channels, **head_config)
        
        # 5. Build Decoder
        self.decoder = Decoder(**decoder_config)
        
    def forward(self, x, noise_std=0.0, quantize=None, **kwargs):
        # x: [B, 5, H, W]
        features = self.backbone(x)
        latent = self.feature_projection(features)
        
        aux = {}
        importance_map = kwargs.get("importance_map", None)
        
        # If we have a head, predict importance
        if self.importance_head is not None:
            # We predict from latent
            pred_imp, pred_logits = self.importance_head(latent)
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
             # AdaptiveQuantizer requires importance_map
             if isinstance(self.quantizer, AdaptiveQuantizer):
                 if importance_map is None:
                     # Fallback if no head and no external map: Uniform/BG?
                     # Or raise error (as it did). 
                     # For verify, let's create ones.
                     importance_map = torch.ones((latent.shape[0], 1, latent.shape[2], latent.shape[3]), device=latent.device)
                 
                 latent_deq, codes, level_map = self.quantizer(latent_noisy, importance_map)
                 aux["codes"] = codes
                 aux["level_map"] = level_map
             else:
                 # Standard layer
                 latent_deq, q = self.quantizer(latent_noisy)
                 aux["codes"] = q
        else:
             latent_deq = latent_noisy

        recon = self.decoder(latent_deq)
        return recon, aux

def build_model(config):
    return MODELS.build(config)
