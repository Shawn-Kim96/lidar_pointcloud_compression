import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    # Duplicate from autoencoder.py to be self-contained or import
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
        groups = 32
        while groups > 1 and channels % groups != 0:
            groups //= 2
        return nn.GroupNorm(num_groups=max(1, groups), num_channels=channels)
    if kind in ("none", "identity", "no"):
        return nn.Identity()
    raise ValueError(f"Unknown norm: {kind}")


class DarkNetBlock(nn.Module):
    def __init__(self, in_channels, norm="batch", activation="leaky_relu"):
        super().__init__()
        # DarkNet Residual Block: 1x1 (C/2) -> 3x3 (C)
        mid_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = _norm(norm, mid_channels)
        self.act1 = _activation(activation)

        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = _norm(norm, in_channels)
        self.act2 = _activation(activation)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out += residual
        return out


from models.registry import BACKBONES

@BACKBONES.register("darknet")
class DarkNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels=5,
        base_channels=32,
        layers=(1, 1, 2, 2, 1),
        norm="batch",
        activation="leaky_relu",
    ):
        """
        DarkNet-like Encoder.
        Total downsampling = 2^len(layers). For (1,1,2,2,1) it is 32x.
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.layers_config = layers

        # Initial Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            _norm(norm, base_channels),
            _activation(activation),
        )

        self.stages = nn.ModuleList()
        current_channels = base_channels

        for i, num_blocks in enumerate(layers):
            stage = []
            # Downsample (Stride 2 conv)
            next_channels = current_channels * 2
            stage.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    _norm(norm, next_channels),
                    _activation(activation),
                )
            )
            # Blocks
            for _ in range(num_blocks):
                stage.append(DarkNetBlock(next_channels, norm=norm, activation=activation))
            
            self.stages.append(nn.Sequential(*stage))
            current_channels = next_channels

        self.out_channels = current_channels

    def forward(self, x):
        x = self.conv1(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return x

if __name__ == "__main__":
    # Smoke test
    model = DarkNetEncoder(layers=(1, 1, 2, 2, 1))
    x = torch.randn(1, 5, 64, 1024)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    # Params
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
