from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset.kitti_object_loader import KittiObjectRangeDataset  # noqa: E402
from models.registry import MODELS  # noqa: E402
import models.backbones  # noqa: F401,E402
import models.compression  # noqa: F401,E402
from utils.range_det_head import build_range_roi_head  # noqa: E402


class _RunConfigLoader(yaml.SafeLoader):
    """YAML loader with limited python tuple support for legacy experiment configs."""


def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


_RunConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track 2 pilot: train a dense range-view ROI/objectness head on top of a frozen compression backbone."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Compression experiment directory with config.yaml and checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional specific compression checkpoint path or filename.")
    parser.add_argument("--kitti_root", type=str, required=True, help="KITTI detection root.")
    parser.add_argument("--split_train", type=str, default="train")
    parser.add_argument("--split_val", type=str, default="val")
    parser.add_argument("--head_type", type=str, default="linear", choices=("linear", "refine", "deep"))
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=1024, help="0 means full split.")
    parser.add_argument("--max_val_samples", type=int, default=512, help="0 means full split.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the imported compression backbone during training.")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos_weight_cap", type=float, default=50.0)
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--fov_up_deg", type=float, default=3.0)
    parser.add_argument("--fov_down_deg", type=float, default=-25.0)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(kind: str) -> torch.device:
    key = str(kind).strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device=cuda but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_checkpoint(run_dir: Path, checkpoint_arg: str) -> Path:
    if checkpoint_arg:
        ckpt = Path(checkpoint_arg)
        if not ckpt.is_absolute():
            local = run_dir / checkpoint_arg
            if local.exists():
                return local
            return (Path.cwd() / checkpoint_arg).resolve()
        return ckpt

    final_ckpt = run_dir / "model_final.pth"
    if final_ckpt.exists():
        return final_ckpt

    ckpts = sorted(run_dir.glob("model_epoch_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No model_final.pth or model_epoch_*.pth found under {run_dir}")

    def _epoch_key(path: Path) -> int:
        m = re.search(r"model_epoch_(\d+)\.pth$", path.name)
        return int(m.group(1)) if m else -1

    return sorted(ckpts, key=_epoch_key)[-1]


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
    raise RuntimeError("Unsupported checkpoint format for compression model.")


def _load_compression_model(run_dir: Path, checkpoint_arg: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any], Path]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml under {run_dir}")
    config = yaml.load(cfg_path.read_text(encoding="utf-8"), Loader=_RunConfigLoader)

    if "model" not in config:
        raise RuntimeError(f"Compression config missing 'model' section: {cfg_path}")

    ckpt_path = _find_checkpoint(run_dir, checkpoint_arg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Compression checkpoint not found: {ckpt_path}")

    model = MODELS.build(config["model"]).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(_extract_state_dict(payload), strict=False)
    model.eval()
    return model, config, ckpt_path


def _limit_dataset(dataset: KittiObjectRangeDataset, max_samples: int) -> Sequence[int]:
    if int(max_samples) <= 0 or len(dataset) <= int(max_samples):
        return list(range(len(dataset)))
    return list(range(int(max_samples)))


class Track2RangeROIPilotNet(nn.Module):
    """
    Pragmatic Track 2 pilot:
    - train a dense ROI/objectness head on raw range images
    - evaluate the same detector on compression-reconstructed range images
    - reuse the compression backbone as the feature extractor
    """

    def __init__(
        self,
        compression_model: nn.Module,
        latent_channels: int,
        head_type: str,
        hidden_channels: int,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.compression_model = compression_model
        self.freeze_backbone = bool(freeze_backbone)
        self.head = build_range_roi_head(head_type=head_type, in_channels=int(latent_channels), hidden_channels=int(hidden_channels))

        if self.freeze_backbone:
            for param in self.compression_model.parameters():
                param.requires_grad = False

    def _encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.compression_model.backbone(x)
                latent = self.compression_model.feature_projection(features)
            return latent

        features = self.compression_model.backbone(x)
        return self.compression_model.feature_projection(features)

    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon, _ = self.compression_model(x, quantize=True)
        return recon.detach()

    def forward(self, x: torch.Tensor, use_compressed_input: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        detector_input = self._reconstruct(x) if use_compressed_input else x
        latent = self._encode_latent(detector_input)
        logits = self.head(latent, target_hw=(x.shape[-2], x.shape[-1]))
        return logits, detector_input


def _masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight_cap: float,
) -> Tuple[torch.Tensor, float]:
    valid = valid_mask > 0.5
    target = target.float()
    pos = (target[valid]).sum().item()
    total = float(valid.sum().item())
    neg = max(total - float(pos), 0.0)
    if pos <= 0.0:
        pos_weight = 1.0
    else:
        pos_weight = min(max(neg / max(pos, 1.0), 1.0), float(pos_weight_cap))
    weight_tensor = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=weight_tensor)
    denom = valid.float().sum().clamp(min=1.0)
    loss = (per_pixel * valid.float()).sum() / denom
    return loss, float(pos_weight)


@torch.no_grad()
def _evaluate(
    model: Track2RangeROIPilotNet,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
    pos_weight_cap: float,
    use_compressed_input: bool,
) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0.0
    loss_sum = 0.0
    steps = 0

    for batch in loader:
        x, proj_mask, roi_mask = batch
        x = x.to(device, non_blocking=True)
        valid = proj_mask.unsqueeze(1).to(device, non_blocking=True)
        target = roi_mask.to(device, non_blocking=True)

        logits, _ = model(x, use_compressed_input=use_compressed_input)
        loss, _ = _masked_bce_with_logits(logits, target, valid, pos_weight_cap)
        prob = torch.sigmoid(logits)
        pred = (prob >= float(threshold)) & (valid > 0.5)
        tgt = (target > 0.5) & (valid > 0.5)

        tp += float((pred & tgt).sum().item())
        fp += float((pred & (~tgt)).sum().item())
        fn += float(((~pred) & tgt).sum().item())
        loss_sum += float(loss.item())
        steps += 1

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": loss_sum / max(steps, 1),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    device = _select_device(args.device)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    kitti_root = Path(args.kitti_root)
    if not kitti_root.is_absolute():
        kitti_root = (Path.cwd() / kitti_root).resolve()
    if not kitti_root.exists():
        raise FileNotFoundError(f"KITTI root not found: {kitti_root}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compression_model, config, ckpt_path = _load_compression_model(run_dir, args.checkpoint, device)
    model_cfg = config.get("model", {})
    quant_cfg = model_cfg.get("quantizer_config", {})
    decoder_cfg = model_cfg.get("decoder_config", {})
    latent_channels = int(quant_cfg.get("latent_channels", decoder_cfg.get("latent_channels", 64)))

    pilot_model = Track2RangeROIPilotNet(
        compression_model=compression_model,
        latent_channels=latent_channels,
        head_type=str(args.head_type),
        hidden_channels=int(args.hidden_channels),
        freeze_backbone=bool(args.freeze_backbone),
    ).to(device)

    train_dataset = KittiObjectRangeDataset(
        root_dir=str(kitti_root),
        split=str(args.split_train),
        config={
            "img_height": int(args.img_h),
            "img_width": int(args.img_w),
            "fov_up": float(args.fov_up_deg),
            "fov_down": float(args.fov_down_deg),
        },
        return_roi_mask=True,
    )
    val_dataset = KittiObjectRangeDataset(
        root_dir=str(kitti_root),
        split=str(args.split_val),
        config={
            "img_height": int(args.img_h),
            "img_width": int(args.img_w),
            "fov_up": float(args.fov_up_deg),
            "fov_down": float(args.fov_down_deg),
        },
        return_roi_mask=True,
    )

    train_subset = Subset(train_dataset, _limit_dataset(train_dataset, int(args.max_train_samples)))
    val_subset = Subset(val_dataset, _limit_dataset(val_dataset, int(args.max_val_samples)))

    train_loader = DataLoader(
        train_subset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    trainable_params = [p for p in pilot_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for Track 2 pilot.")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    metrics_history = []
    best_compressed_iou = -1.0
    best_state = None
    ckpt_out = output_dir / "track2_roi_head_best.pth"
    metrics_csv = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"

    print("============================================================", flush=True)
    print("[Track2 ROI Pilot]", flush=True)
    print(f"device: {device}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"compression_ckpt: {ckpt_path}", flush=True)
    print(f"head_type: {args.head_type}", flush=True)
    print(f"freeze_backbone: {bool(args.freeze_backbone)}", flush=True)
    print(f"epochs: {int(args.epochs)}", flush=True)
    print(f"batch_size: {int(args.batch_size)}", flush=True)
    print(f"max_train_samples: {len(train_subset)}", flush=True)
    print(f"max_val_samples: {len(val_subset)}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print("============================================================", flush=True)

    for epoch in range(int(args.epochs)):
        pilot_model.train()
        if bool(args.freeze_backbone):
            pilot_model.compression_model.eval()

        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            x, proj_mask, roi_mask = batch
            x = x.to(device, non_blocking=True)
            valid = proj_mask.unsqueeze(1).to(device, non_blocking=True)
            target = roi_mask.to(device, non_blocking=True)

            logits, _ = pilot_model(x, use_compressed_input=False)
            loss, pos_weight = _masked_bce_with_logits(logits, target, valid, float(args.pos_weight_cap))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_steps += 1

            if step % max(int(args.log_interval), 1) == 0:
                print(
                    f"[epoch {epoch:02d} step {step:04d}] train_loss={loss.item():.4f} pos_weight={pos_weight:.2f}",
                    flush=True,
                )

        raw_metrics = _evaluate(
            pilot_model,
            val_loader,
            device,
            threshold=float(args.threshold),
            pos_weight_cap=float(args.pos_weight_cap),
            use_compressed_input=False,
        )
        compressed_metrics = _evaluate(
            pilot_model,
            val_loader,
            device,
            threshold=float(args.threshold),
            pos_weight_cap=float(args.pos_weight_cap),
            use_compressed_input=True,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_steps, 1),
            "raw_loss": raw_metrics["loss"],
            "raw_precision": raw_metrics["precision"],
            "raw_recall": raw_metrics["recall"],
            "raw_iou": raw_metrics["iou"],
            "raw_f1": raw_metrics["f1"],
            "compressed_loss": compressed_metrics["loss"],
            "compressed_precision": compressed_metrics["precision"],
            "compressed_recall": compressed_metrics["recall"],
            "compressed_iou": compressed_metrics["iou"],
            "compressed_f1": compressed_metrics["f1"],
            "epoch_sec": time.time() - epoch_start,
        }
        metrics_history.append(epoch_record)
        print(
            (
                f"[epoch {epoch:02d}] train_loss={epoch_record['train_loss']:.4f} "
                f"raw_iou={epoch_record['raw_iou']:.4f} raw_f1={epoch_record['raw_f1']:.4f} "
                f"compressed_iou={epoch_record['compressed_iou']:.4f} compressed_f1={epoch_record['compressed_f1']:.4f}"
            ),
            flush=True,
        )

        if epoch_record["compressed_iou"] > best_compressed_iou:
            best_compressed_iou = float(epoch_record["compressed_iou"])
            best_state = {
                "head_state": pilot_model.head.state_dict(),
                "epoch": epoch,
                "metrics": epoch_record,
            }
            torch.save(best_state, ckpt_out)

        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_history[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_history)

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "run_dir": str(run_dir),
            "compression_ckpt": str(ckpt_path),
            "head_type": str(args.head_type),
            "freeze_backbone": bool(args.freeze_backbone),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_train_samples": int(len(train_subset)),
            "max_val_samples": int(len(val_subset)),
            "best_epoch": int(best_state["epoch"]) if best_state is not None else -1,
            "best_metrics": best_state["metrics"] if best_state is not None else {},
            "latest_epoch": int(epoch),
            "latest_metrics": epoch_record,
            "metrics_csv": str(metrics_csv),
            "head_ckpt": str(ckpt_out),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best_state is None:
        raise RuntimeError("Track 2 pilot finished without any recorded state.")

    print(f"[track2] saved_ckpt={ckpt_out}", flush=True)
    print(f"[track2] saved_metrics={metrics_csv}", flush=True)
    print(f"[track2] saved_summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
