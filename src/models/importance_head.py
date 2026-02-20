import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stage3_necks import Stage3MultiScaleFusion


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            _activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, activation: str = "relu"):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3, stride=1, dilation=1, activation=activation)
        self.conv2 = ConvBNAct(channels, channels, kernel_size=3, stride=1, dilation=1, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class PointPillarsLiteImportanceHead(nn.Module):
    """
    FPN-like dense head inspired by PointPillars backbone/neck topology.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 96, activation: str = "relu"):
        super().__init__()
        c1 = int(hidden_channels)
        c2 = int(round(c1 * 1.25))
        c3 = int(round(c1 * 1.5))
        if c1 < 64:
            raise ValueError(
                "For pp_lite head, hidden_channels should be >= 64 "
                "to satisfy the PP20 capacity target."
            )

        self.stem = ConvBNAct(in_channels, c1, kernel_size=3, stride=1, activation=activation)
        self.enc1 = nn.Sequential(
            ResidualConvBlock(c1, activation=activation),
            ResidualConvBlock(c1, activation=activation),
        )
        self.down2 = ConvBNAct(c1, c2, kernel_size=3, stride=2, activation=activation)
        self.enc2 = nn.Sequential(
            ResidualConvBlock(c2, activation=activation),
            ResidualConvBlock(c2, activation=activation),
        )
        self.down3 = ConvBNAct(c2, c3, kernel_size=3, stride=2, activation=activation)
        self.enc3 = nn.Sequential(
            ResidualConvBlock(c3, activation=activation),
            ResidualConvBlock(c3, activation=activation),
        )

        self.up3_to_2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=False)
        self.lat2 = nn.Conv2d(c2, c2, kernel_size=1, bias=False)
        self.fuse2 = ConvBNAct(c2, c2, kernel_size=3, stride=1, activation=activation)

        self.up2_to_1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=False)
        self.lat1 = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
        self.fuse1 = ConvBNAct(c1, c1, kernel_size=3, stride=1, activation=activation)

        self.out = nn.Conv2d(c1, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(self.down2(x1))
        x3 = self.enc3(self.down3(x2))

        u2 = self.up3_to_2(x3)
        if u2.shape[-2:] != x2.shape[-2:]:
            u2 = F.interpolate(u2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.fuse2(u2 + self.lat2(x2))

        u1 = self.up2_to_1(u2)
        if u1.shape[-2:] != x1.shape[-2:]:
            u1 = F.interpolate(u1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.fuse1(u1 + self.lat1(x1))
        return self.out(u1)


class ImportanceHead(nn.Module):
    """
    Predicts a soft importance map in [0, 1] used by the adaptive quantizer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        activation: str = "relu",
        head_type: str = "basic",
        multiscale_in_channels=None,
        min_importance: float = 0.01,
        max_importance: float = 0.99,
    ):
        super(ImportanceHead, self).__init__()
        self.min_importance = float(min_importance)
        self.max_importance = float(max_importance)
        self.head_type = str(head_type or "basic").lower()
        self.requires_multiscale = self.head_type in ("bifpn", "deformable_msa", "dynamic", "rangeformer", "frnet")
        if not (0.0 <= self.min_importance < self.max_importance <= 1.0):
            raise ValueError("Expected 0 <= min_importance < max_importance <= 1.")

        self.net = None
        self.stem = None
        self.branches = None
        self.fuse = None
        self.out = None
        self.pp_lite = None
        self.stage3_fusion = None

        if self.head_type == "basic":
            act = _activation(activation)
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                act,
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                _activation(activation),
                nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0),
            )
        elif self.head_type == "multiscale":
            self.stem = ConvBNAct(in_channels, hidden_channels, kernel_size=3, dilation=1, activation=activation)
            self.branches = nn.ModuleList(
                [
                    ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, dilation=1, activation=activation),
                    ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, dilation=2, activation=activation),
                    ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, dilation=4, activation=activation),
                ]
            )
            self.fuse = ConvBNAct(hidden_channels * 3, hidden_channels, kernel_size=1, dilation=1, activation=activation)
            self.out = nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0)
        elif self.head_type == "pp_lite":
            self.pp_lite = PointPillarsLiteImportanceHead(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                activation=activation,
            )
        elif self.head_type in ("bifpn", "deformable_msa", "dynamic", "rangeformer", "frnet"):
            if multiscale_in_channels is None:
                multiscale_in_channels = [in_channels, in_channels, in_channels]
            ch_list = list(multiscale_in_channels)
            if len(ch_list) > 3:
                ch_list = ch_list[-3:]
            while len(ch_list) < 3:
                ch_list.insert(0, ch_list[0])
            self.stage3_fusion = Stage3MultiScaleFusion(
                in_channels_list=ch_list,
                hidden_channels=hidden_channels,
                variant=self.head_type,
                activation=activation,
            )
            self.out = nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0)
        else:
            raise ValueError(
                f"Unsupported head_type '{head_type}'. "
                "Use one of: basic, multiscale, pp_lite, bifpn, deformable_msa, dynamic, rangeformer, frnet."
            )

    def forward(self, x: torch.Tensor, target_hw=None, return_logits: bool = True, multiscale_features=None):
        if self.head_type == "basic":
            logits = self.net(x)
        elif self.head_type == "pp_lite":
            logits = self.pp_lite(x)
        elif self.head_type in ("bifpn", "deformable_msa", "dynamic", "rangeformer", "frnet"):
            fused = self.stage3_fusion(x, multiscale_features=multiscale_features)
            logits = self.out(fused)
        else:
            base = self.stem(x)
            feats = [b(base) for b in self.branches]
            fused = self.fuse(torch.cat(feats, dim=1))
            logits = self.out(fused)
        if target_hw is not None and logits.shape[-2:] != tuple(target_hw):
            logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)
        probs = probs * (self.max_importance - self.min_importance) + self.min_importance
        if return_logits:
            return probs, logits
        return probs


if __name__ == "__main__":
    torch.manual_seed(0)
    features = torch.randn(2, 64, 4, 64)
    ms_feats = [torch.randn(2, 64, 32, 512), torch.randn(2, 128, 16, 256), torch.randn(2, 256, 8, 128)]
    for head_type, hidden in (
        ("basic", 32),
        ("multiscale", 32),
        ("pp_lite", 96),
        ("bifpn", 64),
        ("deformable_msa", 64),
        ("dynamic", 64),
        ("rangeformer", 64),
        ("frnet", 64),
    ):
        head = ImportanceHead(
            in_channels=64,
            hidden_channels=hidden,
            head_type=head_type,
            multiscale_in_channels=[64, 128, 256],
        )
        probs, logits = head(features, multiscale_features=ms_feats)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"[{head_type}] params={n_params}")
        print(f"Importance probs: {list(probs.shape)} min={probs.min().item():.4f} max={probs.max().item():.4f}")
        print(f"Importance logits: {list(logits.shape)}")
