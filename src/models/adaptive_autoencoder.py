import torch
import torch.nn as nn

from models.autoencoder import Decoder, Encoder
from models.importance_head import ImportanceHead
from models.quantization import AdaptiveQuantizer


class AdaptiveRangeCompressionModel(nn.Module):
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
        roi_levels=256,
        bg_levels=16,
        quant_use_ste=True,
        importance_hidden_channels=64,
        importance_from_latent=True,
        importance_min=0.01,
        importance_max=0.99,
    ):
        super(AdaptiveRangeCompressionModel, self).__init__()
        self.importance_from_latent = bool(importance_from_latent)
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
        self.decoder = Decoder(
            latent_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        self.quantizer = AdaptiveQuantizer(
            roi_levels=roi_levels,
            bg_levels=bg_levels,
            use_ste=quant_use_ste,
        )
        importance_in_channels = latent_channels if self.importance_from_latent else in_channels
        self.importance_head = ImportanceHead(
            in_channels=importance_in_channels,
            hidden_channels=importance_hidden_channels,
            activation=activation,
            min_importance=importance_min,
            max_importance=importance_max,
        )

    def _predict_importance(self, x, latent):
        source = latent if self.importance_from_latent else x
        return self.importance_head(source, return_logits=True)

    def forward(self, x, roi_mask=None, importance_map=None, noise_std=0.0, quantize=True):
        latent = self.encoder(x)
        if self.training and noise_std > 0.0:
            latent = latent + (torch.randn_like(latent) * noise_std)

        importance_pred, importance_logits = self._predict_importance(x, latent)
        effective_importance = importance_map
        if effective_importance is None:
            if roi_mask is not None:
                effective_importance = roi_mask
            else:
                effective_importance = importance_pred

        if quantize:
            latent_dequant, latent_codes, level_map = self.quantizer(latent, effective_importance)
        else:
            latent_dequant, latent_codes, level_map = latent, latent, None

        recon = self.decoder(latent_dequant)
        aux = {
            "latent": latent,
            "latent_dequant": latent_dequant,
            "latent_codes": latent_codes,
            "level_map": level_map,
            "importance_map": effective_importance,
            "importance_pred": importance_pred,
            "importance_logits": importance_logits,
        }
        return recon, aux


if __name__ == "__main__":
    torch.manual_seed(0)
    model = AdaptiveRangeCompressionModel()
    model.eval()
    x = torch.randn(2, 5, 64, 1024)
    roi = torch.zeros(2, 1, 64, 1024)
    roi[:, :, 20:45, 300:650] = 1.0
    recon, aux = model(x, roi_mask=roi, quantize=True)
    print(f"Input: {list(x.shape)}")
    print(f"Recon: {list(recon.shape)}")
    print(f"Latent codes: {list(aux['latent_codes'].shape)}")
    print(f"Importance map: {list(aux['importance_map'].shape)}")
