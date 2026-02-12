import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def average_precision_binary(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.size == 0 or labels.size == 0:
        return 0.0
    positives = int((labels == 1).sum())
    if positives == 0:
        return 0.0

    order = np.argsort(-scores)
    labels_sorted = labels[order]

    tp = np.cumsum(labels_sorted == 1).astype(np.float64)
    fp = np.cumsum(labels_sorted == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(positives)

    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


class ProxyTaskHead(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32):
        super(ProxyTaskHead, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, recon):
        return self.net(recon)


class TaskLossModule(nn.Module):
    def __init__(self, backend="auto", in_channels=5, hidden_channels=32):
        super(TaskLossModule, self).__init__()
        self.backend = backend
        self.task_head = None
        self.openpcdet_ready = False

        if backend in ("auto", "openpcdet"):
            try:
                import pcdet  # noqa: F401

                self.openpcdet_ready = True
            except Exception:
                if backend == "openpcdet":
                    warnings.warn("OpenPCDet backend requested but unavailable. Falling back to proxy.")

        if backend == "proxy" or not self.openpcdet_ready:
            self.backend = "proxy"
            self.task_head = ProxyTaskHead(in_channels=in_channels, hidden_channels=hidden_channels)
        else:
            # Placeholder: full OpenPCDet loss path can be integrated here when dependency is available.
            self.backend = "openpcdet"
            self.task_head = ProxyTaskHead(in_channels=in_channels, hidden_channels=hidden_channels)
            warnings.warn("OpenPCDet import succeeded, but proxy head is used until full adapter is wired.")

    @staticmethod
    def _prepare_masks(roi_mask, valid_mask, target_hw, device):
        if roi_mask.dim() == 3:
            roi_mask = roi_mask.unsqueeze(1)
        if roi_mask.shape[-2:] != target_hw:
            roi_mask = F.interpolate(roi_mask.float(), size=target_hw, mode="nearest")
        roi_mask = roi_mask.float().to(device)

        if valid_mask is None:
            valid_mask = torch.ones_like(roi_mask, device=device)
        else:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            if valid_mask.shape[-2:] != target_hw:
                valid_mask = F.interpolate(valid_mask.float(), size=target_hw, mode="nearest")
            valid_mask = valid_mask.float().to(device)
        return roi_mask, valid_mask

    def forward(self, recon, roi_mask, valid_mask=None, return_details=False):
        logits = self.task_head(recon)
        roi_mask, valid_mask = self._prepare_masks(
            roi_mask=roi_mask,
            valid_mask=valid_mask,
            target_hw=logits.shape[-2:],
            device=logits.device,
        )

        bce = F.binary_cross_entropy_with_logits(logits, roi_mask, reduction="none")
        weighted_bce = (bce * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        with torch.no_grad():
            scores = torch.sigmoid(logits)
            valid = valid_mask > 0.5
            ap = average_precision_binary(
                scores[valid].detach().cpu().numpy(),
                roi_mask[valid].detach().cpu().numpy(),
            )

        if return_details:
            details = {
                "task_bce": float(weighted_bce.detach().item()),
                "map_proxy": float(ap),
            }
            return weighted_bce, details
        return weighted_bce


if __name__ == "__main__":
    torch.manual_seed(0)

    module = TaskLossModule(backend="auto")
    recon = torch.randn(2, 5, 64, 1024, requires_grad=True)
    roi_mask = torch.zeros(2, 1, 64, 1024)
    roi_mask[:, :, 20:45, 200:500] = 1.0
    valid_mask = torch.ones(2, 1, 64, 1024)

    loss, details = module(recon, roi_mask, valid_mask=valid_mask, return_details=True)
    loss.backward()
    print(f"Task loss scalar: {float(loss.item()):.6f}")
    print(f"Proxy mAP: {details['map_proxy']:.6f}")
