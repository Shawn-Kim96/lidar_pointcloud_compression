import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, activation: str = "relu"):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            _activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _pick_num_heads(channels: int) -> int:
    for h in (8, 4, 2, 1):
        if channels % h == 0:
            return h
    return 1


class _ScaleAdapter(nn.Module):
    def __init__(self, in_channels_list, hidden_channels: int, activation: str = "relu"):
        super().__init__()
        self.in_channels_list = list(in_channels_list)
        self.proj = nn.ModuleList(
            [ConvBNAct(int(c), int(hidden_channels), kernel_size=1, activation=activation) for c in self.in_channels_list]
        )

    def _fallback_features(self, x: torch.Tensor):
        feats = [x]
        for s in (2, 4):
            k_h = min(s, x.shape[-2])
            k_w = min(s, x.shape[-1])
            feats.append(F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w)))
        # use only required scale count
        feats = feats[: len(self.proj)]
        while len(feats) < len(self.proj):
            feats.insert(0, feats[0])
        return feats

    def forward(self, x: torch.Tensor, multiscale_features=None):
        if multiscale_features is None or len(multiscale_features) == 0:
            src = self._fallback_features(x)
        else:
            src = list(multiscale_features)
            if len(src) > len(self.proj):
                src = src[-len(self.proj) :]
            while len(src) < len(self.proj):
                src.insert(0, src[0])

        target_hw = x.shape[-2:]
        out = []
        for feat, proj in zip(src, self.proj):
            z = proj(feat)
            if z.shape[-2:] != target_hw:
                z = F.interpolate(z, size=target_hw, mode="bilinear", align_corners=False)
            out.append(z)
        return out


class Stage3MultiScaleFusion(nn.Module):
    """
    Stage3 multi-scale ROI fusion block.
    Variants:
      - bifpn
      - deformable_msa
      - dynamic
      - rangeformer
      - frnet
    """

    def __init__(self, in_channels_list, hidden_channels: int, variant: str, activation: str = "relu"):
        super().__init__()
        self.variant = str(variant).lower()
        self.hidden_channels = int(hidden_channels)
        self.adapter = _ScaleAdapter(in_channels_list, hidden_channels=hidden_channels, activation=activation)
        self.scale_count = len(self.adapter.proj)
        self.fuse_weights = nn.Parameter(torch.ones(self.scale_count))
        self.post = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation)

        if self.variant == "deformable_msa":
            self.offset = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)
            nheads = _pick_num_heads(hidden_channels)
            self.attn = nn.MultiheadAttention(hidden_channels, nheads, batch_first=True)
            self.ffn = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation)
        elif self.variant == "dynamic":
            self.scale_mlp = nn.Sequential(
                nn.Linear(hidden_channels * self.scale_count, hidden_channels),
                _activation(activation),
                nn.Linear(hidden_channels, self.scale_count),
            )
            red = max(hidden_channels // 4, 8)
            self.channel_se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_channels, red, kernel_size=1),
                _activation(activation),
                nn.Conv2d(red, hidden_channels, kernel_size=1),
                nn.Sigmoid(),
            )
        elif self.variant == "rangeformer":
            nheads = _pick_num_heads(hidden_channels)
            self.row_attn = nn.MultiheadAttention(hidden_channels, nheads, batch_first=True)
            self.col_attn = nn.MultiheadAttention(hidden_channels, nheads, batch_first=True)
            self.range_ffn = nn.Sequential(
                ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            )
        elif self.variant == "frnet":
            red = max(hidden_channels // 4, 8)
            self.freq_mlp = nn.Sequential(
                nn.Linear(hidden_channels, red),
                _activation(activation),
                nn.Linear(red, hidden_channels),
                nn.Sigmoid(),
            )
            self.fr_post = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, activation=activation)
        elif self.variant == "bifpn":
            # weights + post conv are sufficient lightweight BiFPN-style fusion in latent resolution.
            pass
        else:
            raise ValueError(
                f"Unsupported stage3 fusion variant '{variant}'. "
                "Use one of: bifpn, deformable_msa, dynamic, rangeformer, frnet."
            )

    def _weighted_sum(self, feats):
        w = F.relu(self.fuse_weights)
        w = w / (w.sum() + 1e-6)
        fused = 0.0
        for wi, fi in zip(w, feats):
            fused = fused + wi * fi
        return fused

    def _deformable_msa(self, feats):
        fused = self.post(self._weighted_sum(feats))
        b, c, h, w = fused.shape
        offset = self.offset(fused)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=fused.device),
            torch.linspace(-1.0, 1.0, w, device=fused.device),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        norm = torch.tensor([max(w - 1, 1), max(h - 1, 1)], device=fused.device, dtype=fused.dtype).view(1, 1, 1, 2)
        offset_norm = 2.0 * offset.permute(0, 2, 3, 1) / norm
        sampled = F.grid_sample(fused, (grid + offset_norm).clamp(-1.0, 1.0), align_corners=False)

        tokens = sampled.flatten(2).transpose(1, 2)  # [B, HW, C]
        tokens, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        out = tokens.transpose(1, 2).reshape(b, c, h, w)
        return self.ffn(out + sampled)

    def _dynamic(self, feats):
        pooled = [f.mean(dim=(2, 3)) for f in feats]
        stacked = torch.cat(pooled, dim=1)
        scale_logits = self.scale_mlp(stacked)
        scale_w = torch.softmax(scale_logits, dim=1)

        fused = 0.0
        for i, f in enumerate(feats):
            fused = fused + scale_w[:, i].view(-1, 1, 1, 1) * f
        fused = self.post(fused)
        gate = self.channel_se(fused)
        return fused * gate

    def _rangeformer(self, feats):
        fused = self.post(self._weighted_sum(feats))
        b, c, h, w = fused.shape

        row_in = fused.permute(0, 2, 3, 1).reshape(b * h, w, c)
        row_out, _ = self.row_attn(row_in, row_in, row_in, need_weights=False)
        row_out = row_out.reshape(b, h, w, c).permute(0, 3, 1, 2)

        col_in = row_out.permute(0, 3, 2, 1).reshape(b * w, h, c)
        col_out, _ = self.col_attn(col_in, col_in, col_in, need_weights=False)
        col_out = col_out.reshape(b, w, h, c).permute(0, 3, 2, 1)

        return col_out + self.range_ffn(col_out)

    def _frnet(self, feats):
        fused = self.post(self._weighted_sum(feats))
        freq = torch.fft.rfft2(fused, norm="ortho")
        mag = torch.abs(freq).mean(dim=(2, 3))  # [B, C]
        gate = self.freq_mlp(mag).unsqueeze(-1).unsqueeze(-1)
        out = fused * gate
        return self.fr_post(out)

    def forward(self, x: torch.Tensor, multiscale_features=None):
        feats = self.adapter(x, multiscale_features=multiscale_features)
        if self.variant == "bifpn":
            return self.post(self._weighted_sum(feats))
        if self.variant == "deformable_msa":
            return self._deformable_msa(feats)
        if self.variant == "dynamic":
            return self._dynamic(feats)
        if self.variant == "rangeformer":
            return self._rangeformer(feats)
        if self.variant == "frnet":
            return self._frnet(feats)
        raise RuntimeError(f"Unexpected variant {self.variant}")
