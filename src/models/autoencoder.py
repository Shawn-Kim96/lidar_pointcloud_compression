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
        
    def forward(self, x, return_features: bool = False, stage_additions=None):
        x = self.initial_conv(x)
        features = []
        for idx, stage in enumerate(self.stage_layers):
            x = stage(x)
            if stage_additions is not None and idx < len(stage_additions):
                addition = stage_additions[idx]
                if addition is not None:
                    if addition.shape[-2:] != x.shape[-2:]:
                        addition = F.interpolate(addition, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    x = x + addition
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
        post_refine_blocks=0,
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
        self.post_refine_blocks = int(post_refine_blocks)

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
        refine_layers = []
        for _ in range(max(0, self.post_refine_blocks)):
            refine_layers.append(
                ResidualBlock(
                    in_channels=self.base_channels,
                    out_channels=self.base_channels,
                    stride=1,
                    norm=self.norm,
                    activation=self.activation,
                    dropout=self.dropout,
                )
            )
        self.post_refine = nn.Sequential(*refine_layers) if refine_layers else nn.Identity()
        self.final_conv = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.deconvs(x)
        x = self.post_refine(x)
        x = self.final_conv(x)
        return x # Range, x, y, z, remission


class BilinearSkipBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        norm="batch",
        activation="relu",
        dropout=0.0,
    ):
        super().__init__()
        self.skip_channels = int(skip_channels)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(norm, out_channels),
            _activation(activation),
        )
        self.drop = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        if self.skip_channels > 0:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(self.skip_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                _norm(norm, out_channels),
                _activation(activation),
            )
            merge_in = out_channels * 2
        else:
            self.skip_proj = None
            merge_in = out_channels
        self.merge = nn.Sequential(
            nn.Conv2d(merge_in, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(norm, out_channels),
            _activation(activation),
            ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                norm=norm,
                activation=activation,
                dropout=dropout,
            ),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up(x)
        x = self.drop(x)
        if self.skip_proj is not None and skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            skip = self.skip_proj(skip)
            x = torch.cat([x, skip], dim=1)
        return self.merge(x)


class SkipDecoderCore(nn.Module):
    def __init__(
        self,
        latent_channels=64,
        base_channels=64,
        num_stages=4,
        encoder_stage_channels=None,
        norm="batch",
        activation="relu",
        dropout=0.0,
        post_refine_blocks=0,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.num_stages = int(num_stages)
        self.norm = norm
        self.activation = activation
        self.dropout = float(dropout)
        self.post_refine_blocks = int(post_refine_blocks)
        enc_stage_channels = list(encoder_stage_channels or [])
        skip_channels = list(reversed(enc_stage_channels[:-1]))
        if len(skip_channels) < self.num_stages:
            skip_channels.extend([0] * (self.num_stages - len(skip_channels)))
        self.skip_channels = skip_channels[: self.num_stages]

        blocks = []
        in_ch = self.latent_channels
        for stage in range(self.num_stages):
            remaining = self.num_stages - stage
            if remaining <= 1:
                out_ch = self.base_channels
            else:
                out_ch = self.base_channels * (2 ** (remaining - 2))
            blocks.append(
                BilinearSkipBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    skip_channels=self.skip_channels[stage],
                    norm=self.norm,
                    activation=self.activation,
                    dropout=self.dropout,
                )
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        refine_layers = []
        for _ in range(max(0, self.post_refine_blocks)):
            refine_layers.append(
                ResidualBlock(
                    in_channels=self.base_channels,
                    out_channels=self.base_channels,
                    stride=1,
                    norm=self.norm,
                    activation=self.activation,
                    dropout=self.dropout,
                )
            )
        self.post_refine = nn.Sequential(*refine_layers) if refine_layers else nn.Identity()

    def _prepare_skip_list(self, skip_features):
        if not skip_features:
            return [None] * self.num_stages
        skip_list = list(reversed(skip_features[:-1]))
        if len(skip_list) < self.num_stages:
            skip_list.extend([None] * (self.num_stages - len(skip_list)))
        return skip_list[: self.num_stages]

    def forward(self, x, skip_features=None):
        skip_list = self._prepare_skip_list(skip_features)
        for block, skip in zip(self.blocks, skip_list):
            x = block(x, skip=skip)
        x = self.post_refine(x)
        return x


class SkipDecoder(nn.Module):
    def __init__(
        self,
        latent_channels=64,
        out_channels=5,
        base_channels=64,
        num_stages=4,
        encoder_stage_channels=None,
        norm="batch",
        activation="relu",
        dropout=0.0,
        post_refine_blocks=0,
    ):
        super().__init__()
        self.core = SkipDecoderCore(
            latent_channels=latent_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            encoder_stage_channels=encoder_stage_channels,
            norm=norm,
            activation=activation,
            dropout=dropout,
            post_refine_blocks=post_refine_blocks,
        )
        self.final_conv = nn.Conv2d(int(base_channels), int(out_channels), kernel_size=3, stride=1, padding=1)

    def forward(self, x, skip_features=None):
        x = self.core(x, skip_features=skip_features)
        return self.final_conv(x)


class SkipCoordConditionedDecoder(nn.Module):
    def __init__(
        self,
        latent_channels=64,
        out_channels=5,
        base_channels=64,
        num_stages=4,
        encoder_stage_channels=None,
        norm="batch",
        activation="relu",
        dropout=0.0,
        post_refine_blocks=0,
        coord_channels=5,
        implicit_hidden_channels=128,
        position_condition_channels=0,
    ):
        super().__init__()
        self.core = SkipDecoderCore(
            latent_channels=latent_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            encoder_stage_channels=encoder_stage_channels,
            norm=norm,
            activation=activation,
            dropout=dropout,
            post_refine_blocks=post_refine_blocks,
        )
        self.base_channels = int(base_channels)
        self.coord_channels = int(coord_channels)
        self.implicit_hidden_channels = int(implicit_hidden_channels)
        self.position_condition_channels = int(position_condition_channels)
        self.norm = norm
        self.activation = activation

        if self.position_condition_channels > 0:
            self.position_proj = nn.Sequential(
                nn.Conv2d(self.position_condition_channels, self.base_channels, kernel_size=1, bias=False),
                _norm(self.norm, self.base_channels),
                _activation(self.activation),
            )
        else:
            self.position_proj = None

        head_in_ch = self.base_channels + self.coord_channels
        if self.position_proj is not None:
            head_in_ch += self.base_channels
        self.implicit_head = nn.Sequential(
            nn.Conv2d(head_in_ch, self.implicit_hidden_channels, kernel_size=1, bias=False),
            _norm(self.norm, self.implicit_hidden_channels),
            _activation(self.activation),
            nn.Conv2d(self.implicit_hidden_channels, self.implicit_hidden_channels, kernel_size=1, bias=False),
            _norm(self.norm, self.implicit_hidden_channels),
            _activation(self.activation),
            nn.Conv2d(self.implicit_hidden_channels, int(out_channels), kernel_size=1),
        )

    def forward(self, x, coord_features=None, position_context=None, skip_features=None):
        x = self.core(x, skip_features=skip_features)
        cond = [x]
        if self.position_proj is not None and position_context is not None:
            pos = self.position_proj(position_context)
            if pos.shape[-2:] != x.shape[-2:]:
                pos = F.interpolate(pos, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cond.append(pos)
        if coord_features is None:
            raise ValueError("SkipCoordConditionedDecoder requires coord_features.")
        if coord_features.shape[-2:] != x.shape[-2:]:
            coord_features = F.interpolate(coord_features, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cond.append(coord_features)
        return self.implicit_head(torch.cat(cond, dim=1))


class CoordConditionedDecoder(nn.Module):
    """
    Decoder that combines low-resolution latent features with explicit sensor-ray
    coordinates and an optional position side-context before pixel-wise prediction.
    """

    def __init__(
        self,
        latent_channels=64,
        out_channels=5,
        base_channels=64,
        num_stages=4,
        norm="batch",
        activation="relu",
        dropout=0.0,
        post_refine_blocks=0,
        coord_channels=5,
        implicit_hidden_channels=128,
        position_condition_channels=0,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.num_stages = int(num_stages)
        self.norm = norm
        self.activation = activation
        self.dropout = float(dropout)
        self.post_refine_blocks = int(post_refine_blocks)
        self.coord_channels = int(coord_channels)
        self.implicit_hidden_channels = int(implicit_hidden_channels)
        self.position_condition_channels = int(position_condition_channels)

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
        refine_layers = []
        for _ in range(max(0, self.post_refine_blocks)):
            refine_layers.append(
                ResidualBlock(
                    in_channels=self.base_channels,
                    out_channels=self.base_channels,
                    stride=1,
                    norm=self.norm,
                    activation=self.activation,
                    dropout=self.dropout,
                )
            )
        self.post_refine = nn.Sequential(*refine_layers) if refine_layers else nn.Identity()

        if self.position_condition_channels > 0:
            self.position_proj = nn.Sequential(
                nn.Conv2d(self.position_condition_channels, self.base_channels, kernel_size=1, bias=False),
                _norm(self.norm, self.base_channels),
                _activation(self.activation),
            )
        else:
            self.position_proj = None

        head_in_ch = self.base_channels + self.coord_channels
        if self.position_proj is not None:
            head_in_ch += self.base_channels
        self.implicit_head = nn.Sequential(
            nn.Conv2d(head_in_ch, self.implicit_hidden_channels, kernel_size=1, bias=False),
            _norm(self.norm, self.implicit_hidden_channels),
            _activation(self.activation),
            nn.Conv2d(self.implicit_hidden_channels, self.implicit_hidden_channels, kernel_size=1, bias=False),
            _norm(self.norm, self.implicit_hidden_channels),
            _activation(self.activation),
            nn.Conv2d(self.implicit_hidden_channels, self.out_channels, kernel_size=1),
        )

    def forward(self, x, coord_features=None, position_context=None):
        x = self.deconvs(x)
        x = self.post_refine(x)
        cond = [x]
        if self.position_proj is not None and position_context is not None:
            pos = self.position_proj(position_context)
            if pos.shape[-2:] != x.shape[-2:]:
                pos = F.interpolate(pos, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cond.append(pos)
        if coord_features is None:
            raise ValueError("CoordConditionedDecoder requires coord_features.")
        if coord_features.shape[-2:] != x.shape[-2:]:
            coord_features = F.interpolate(coord_features, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cond.append(coord_features)
        return self.implicit_head(torch.cat(cond, dim=1))

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
