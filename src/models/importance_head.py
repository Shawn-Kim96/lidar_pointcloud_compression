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


class ImportanceHead(nn.Module):
    """
    Predicts a soft importance map in [0, 1] used by the adaptive quantizer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        activation: str = "relu",
        min_importance: float = 0.01,
        max_importance: float = 0.99,
    ):
        super(ImportanceHead, self).__init__()
        self.min_importance = float(min_importance)
        self.max_importance = float(max_importance)
        if not (0.0 <= self.min_importance < self.max_importance <= 1.0):
            raise ValueError("Expected 0 <= min_importance < max_importance <= 1.")

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

    def forward(self, x: torch.Tensor, target_hw=None, return_logits: bool = True):
        logits = self.net(x)
        if target_hw is not None and logits.shape[-2:] != tuple(target_hw):
            logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)
        probs = probs * (self.max_importance - self.min_importance) + self.min_importance
        if return_logits:
            return probs, logits
        return probs


if __name__ == "__main__":
    torch.manual_seed(0)
    head = ImportanceHead(in_channels=64, hidden_channels=32)
    features = torch.randn(2, 64, 4, 64)
    probs, logits = head(features)
    print(f"Importance probs: {list(probs.shape)} min={probs.min().item():.4f} max={probs.max().item():.4f}")
    print(f"Importance logits: {list(logits.shape)}")
