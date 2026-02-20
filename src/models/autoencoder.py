import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    raise ValueError(f"Unknown activation: {name}")


def _norm(kind: str, channels: int) -> nn.Module:
    kind = (kind or "batch").lower()
    if kind in ("batch", "bn"):
        return nn.BatchNorm2d(channels)
    if kind in ("group", "gn"):
        # 32 groups is a common default; clamp to valid range.
        groups = 32
        while groups > 1 and channels % groups != 0:
            groups //= 2
        return nn.GroupNorm(num_groups=max(1, groups), num_channels=channels)
    if kind in ("none", "identity", "no"):
        return nn.Identity()
    raise ValueError(f"Unknown norm: {kind}")


class QuantizationLayer(nn.Module):
    def __init__(self, bits=8, eps=1e-6, use_ste=False):
        super(QuantizationLayer, self).__init__()
        self.bits = bits
        self.eps = eps
        self.use_ste = bool(use_ste)

    def forward(self, x):
        # Per-sample affine quantization into [0, 2^bits - 1]
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        levels = float((2 ** self.bits) - 1)
        scale = levels / (x_max - x_min + self.eps)
        q = torch.round((x - x_min) * scale).clamp(0, levels)
        deq = q / scale + x_min
        if self.use_ste:
            deq = x + (deq - x).detach()
        return deq, q

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm="batch", activation="relu", dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _norm(norm, out_channels)
        self.act = _activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _norm(norm, out_channels)
        self.drop = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _norm(norm, out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=5,
        latent_channels=64,
        base_channels=64,
        num_stages=4,
        blocks_per_stage=1,
        norm="batch",
        activation="relu",
        dropout=0.0,
    ):
        """
        Compresses HxW image to H/(2^num_stages) x W/(2^num_stages) feature map.
        """
        super(Encoder, self).__init__()
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.num_stages = int(num_stages)
        self.blocks_per_stage = int(blocks_per_stage)
        self.norm = norm
        self.activation = activation
        self.dropout = float(dropout)

        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(self.norm, self.base_channels),
            _activation(self.activation),
        )

        stage_layers = []
        self.stage_channels = []
        in_ch = self.base_channels
        for stage in range(self.num_stages):
            if stage < (self.num_stages - 1):
                out_ch = self.base_channels * (2 ** stage)
            else:
                out_ch = self.latent_channels

            # Downsample block
            blocks = [
                ResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=2,
                    norm=self.norm,
                    activation=self.activation,
                    dropout=self.dropout,
                )
            ]
            # Extra blocks at same resolution
            for _ in range(max(0, self.blocks_per_stage - 1)):
                blocks.append(
                    ResidualBlock(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        stride=1,
                        norm=self.norm,
                        activation=self.activation,
                        dropout=self.dropout,
                    )
                )
            stage_layers.append(nn.Sequential(*blocks))
            self.stage_channels.append(out_ch)
            in_ch = out_ch

        self.stage_layers = nn.ModuleList(stage_layers)
        self.layers = nn.Sequential(*stage_layers)
        
    def forward(self, x, return_features: bool = False):
        x = self.initial_conv(x)
        features = []
        for stage in self.stage_layers:
            x = stage(x)
            features.append(x)
        if return_features:
            return x, features
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        latent_channels=64,
        out_channels=5,
        base_channels=64,
        num_stages=4,
        norm="batch",
        activation="relu",
        dropout=0.0,
    ):
        """
        Reconstructs HxW image from H/(2^num_stages) x W/(2^num_stages) feature map.
        """
        super(Decoder, self).__init__()
        self.latent_channels = int(latent_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.num_stages = int(num_stages)
        self.norm = norm
        self.activation = activation
        self.dropout = float(dropout)

        layers = []
        in_ch = self.latent_channels
        for stage in range(self.num_stages):
            remaining = self.num_stages - stage
            if remaining <= 1:
                out_ch = self.base_channels
            else:
                out_ch = self.base_channels * (2 ** (remaining - 2))

            layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                )
            )
            layers.append(_norm(self.norm, out_ch))
            layers.append(_activation(self.activation))
            if self.dropout > 0.0:
                layers.append(nn.Dropout2d(p=self.dropout))
            in_ch = out_ch

        self.deconvs = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.deconvs(x)
        x = self.final_conv(x)
        return x # Range, x, y, z, remission

from models.backbones import DarkNetEncoder

class RangeCompressionModel(nn.Module):
    def __init__(
        self,
        quant_bits=8,
        in_channels=5,
        latent_channels=64,
        base_channels=64,
        num_stages=4,
        blocks_per_stage=1,
        norm="batch",
        activation="relu",
        dropout=0.0,
        backbone_type="resnet", # "resnet" or "darknet"
    ):
        super(RangeCompressionModel, self).__init__()
        
        if backbone_type == "darknet":
            # DarkNet-21 like structure: 5 stages of downsampling (32x total)
            self.encoder = DarkNetEncoder(
                in_channels=in_channels,
                base_channels=32, # Starts smaller but doubles effectively
                layers=(1, 1, 2, 2, 1), # Standard DarkNet-21 (tiny/variant)
                norm=norm,
                activation=activation
            )
            # DarkNet Output Channels: 32 * 2^5 = 1024
            # We need to project this to latent_channels
            self.feature_projection = nn.Sequential(
                nn.Conv2d(1024, latent_channels, kernel_size=1, bias=False),
                _norm(norm, latent_channels),
                _activation(activation)
            )
            self.backbone_type = "darknet"
            # Decoder needs to upsample 5 times (32x)
            num_stages = 5 
        else:
            self.encoder = Encoder(
                in_channels=in_channels,
                latent_channels=latent_channels,
                base_channels=base_channels,
                num_stages=num_stages,
                blocks_per_stage=blocks_per_stage,
                norm=norm,
                activation=activation,
                dropout=dropout,
            )
            self.feature_projection = nn.Identity()
            self.backbone_type = "resnet"

        self.decoder = Decoder(
            latent_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        self.quantizer = QuantizationLayer(bits=quant_bits)
        
    def forward(self, x, noise_std=0.0, quantize=None):
        # x: [B, 5, H, W]
        if self.backbone_type == "darknet":
            features = self.encoder(x) # [B, 1024, H/32, W/32]
            latent = self.feature_projection(features)
        else:
            latent = self.encoder(x)
        
        # Add quantization-aware noise or transmission noise
        if self.training and noise_std > 0.0:
            noise = torch.randn_like(latent) * noise_std
            latent = latent + noise

        if quantize is None:
            quantize = not self.training

        if quantize:
            latent_deq, latent_codes = self.quantizer(latent)
        else:
            latent_deq, latent_codes = latent, latent

        recon = self.decoder(latent_deq)
        return recon, latent_codes

if __name__ == "__main__":
    # Test dims
    model = RangeCompressionModel()
    dummy_input = torch.randn(1, 5, 64, 1024)
    model.eval()
    recon, latent_codes = model(dummy_input, quantize=True)
    max_fractional = (latent_codes - torch.round(latent_codes)).abs().max().item()
    print(f"Input: {dummy_input.shape}")
    print(f"Latent codes: {latent_codes.shape}")
    print(f"Latent max fractional part: {max_fractional:.6f}")
    print(f"Recon: {recon.shape}")
