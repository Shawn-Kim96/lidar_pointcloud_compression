import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset.semantickitti_loader import SemanticKittiDataset
from models.registry import MODELS
import models.compression  # noqa: F401
import models.backbones  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate native vs oracle-ROI quantization behavior on a trained run."
    )
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Experiment run dir containing config.yaml and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, latest model_epoch_*.pth in run_dir is used.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(repo_root / "data" / "dataset" / "semantickitti" / "dataset" / "sequences"),
    )
    parser.add_argument("--val_seq", type=str, default="08")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold for ROI TP/FP/FN metrics.")
    parser.add_argument(
        "--output_summary_csv",
        type=str,
        default=str(repo_root / "notebooks" / "oracle_eval_summary.csv"),
    )
    parser.add_argument(
        "--output_detail_csv",
        type=str,
        default=str(repo_root / "notebooks" / "oracle_eval_detail.csv"),
    )
    return parser.parse_args()


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _latest_epoch_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_epoch_*.pth"))
    if not ckpts:
        final_ckpt = run_dir / "model_final.pth"
        if final_ckpt.exists():
            return final_ckpt
        raise FileNotFoundError(f"No model_epoch_*.pth or model_final.pth found in {run_dir}")

    def _epoch_num(path: Path) -> int:
        m = re.search(r"model_epoch_(\d+)\.pth$", path.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=_epoch_num)
    return ckpts[-1]


def _load_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in run_dir: {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _estimate_code_entropy(codes: torch.Tensor) -> List[float]:
    # codes: [B,C,H,W]
    if codes is None:
        return []
    out: List[float] = []
    q = torch.round(codes.detach()).to(torch.int64).clamp(min=0).cpu()
    for i in range(q.shape[0]):
        flat = q[i].reshape(-1)
        if flat.numel() == 0:
            out.append(float("nan"))
            continue
        hist = torch.bincount(flat).float()
        p = hist / hist.sum().clamp(min=1.0)
        p = p[p > 0]
        ent = float((-(p * torch.log2(p))).sum().item())
        out.append(ent)
    return out


@torch.no_grad()
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = _load_config(run_dir)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_epoch_checkpoint(run_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS.build(config["model"]).to(device).eval()

    payload = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(payload)
    model.load_state_dict(state, strict=False)

    qcfg = config.get("model", {}).get("quantizer_config", {})
    quantizer_mode = str(qcfg.get("mode", "adaptive")).lower()
    quant_bits = int(qcfg.get("uniform_bits", qcfg.get("quant_bits", 8)))

    dataset = SemanticKittiDataset(
        root_dir=args.data_root,
        sequences=[args.val_seq],
        return_roi_mask=True,
    )
    if args.max_frames > 0:
        dataset = Subset(dataset, list(range(min(args.max_frames, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    modes = ["native"]
    if quantizer_mode == "adaptive":
        modes.append("oracle_roi")

    detail_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for mode in modes:
        total_all_mse = 0.0
        total_roi_mse = 0.0
        total_bg_mse = 0.0
        total_rate_proxy = 0.0
        total_eq_bits = 0.0
        total_entropy = 0.0
        total_bpp_eq = 0.0
        total_bpp_entropy = 0.0
        total_imp_roi = 0.0
        total_imp_bg = 0.0
        imp_gap_count = 0
        sample_count = 0
        tp = fp = fn = tn = 0
        frame_idx = 0

        for batch in loader:
            data, valid_mask, roi_mask = batch
            data = data.to(device)
            valid_mask = valid_mask.to(device).unsqueeze(1).float()
            roi_mask = roi_mask.to(device).float()

            importance_override = None
            if mode == "oracle_roi":
                importance_override = roi_mask

            recon, aux = model(
                data,
                noise_std=0.0,
                quantize=True,
                importance_map=importance_override,
            )

            err = (recon - data) ** 2
            all_den = (valid_mask.sum(dim=(1, 2, 3)) * data.shape[1]).clamp(min=1.0)
            all_mse = (err * valid_mask).sum(dim=(1, 2, 3)) / all_den

            roi_valid = (roi_mask * valid_mask).clamp(min=0.0, max=1.0)
            bg_valid = ((1.0 - roi_mask) * valid_mask).clamp(min=0.0, max=1.0)
            roi_den = (roi_valid.sum(dim=(1, 2, 3)) * data.shape[1]).clamp(min=1.0)
            bg_den = (bg_valid.sum(dim=(1, 2, 3)) * data.shape[1]).clamp(min=1.0)
            roi_mse = (err * roi_valid).sum(dim=(1, 2, 3)) / roi_den
            bg_mse = (err * bg_valid).sum(dim=(1, 2, 3)) / bg_den

            level_map = aux.get("level_map", None)
            codes = aux.get("codes", None)
            if level_map is not None:
                rate_proxy = level_map.mean(dim=(1, 2, 3))
                eq_bits = torch.log2(level_map.clamp(min=2.0)).mean(dim=(1, 2, 3))
            else:
                rate_proxy = torch.full((data.shape[0],), float((2 ** quant_bits) - 1), device=device)
                eq_bits = torch.full((data.shape[0],), float(quant_bits), device=device)

            entropy_vals = _estimate_code_entropy(codes) if codes is not None else [float("nan")] * data.shape[0]

            if codes is not None:
                c_lat, h_lat, w_lat = int(codes.shape[1]), int(codes.shape[2]), int(codes.shape[3])
            else:
                c_lat, h_lat, w_lat = int(data.shape[1]), int(data.shape[2]), int(data.shape[3])
            symbols_per_input_pixel = (c_lat * h_lat * w_lat) / float(data.shape[2] * data.shape[3])

            imp_pred = aux.get("importance_map_pred", None)
            imp_up = None
            if imp_pred is not None:
                imp_up = F.interpolate(
                    imp_pred.detach(),
                    size=roi_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).clamp(min=0.0, max=1.0)
                imp_roi_num = (imp_up * roi_valid).sum().item()
                imp_roi_den = roi_valid.sum().item()
                imp_bg_num = (imp_up * bg_valid).sum().item()
                imp_bg_den = bg_valid.sum().item()
                if imp_roi_den > 0 and imp_bg_den > 0:
                    total_imp_roi += imp_roi_num / imp_roi_den
                    total_imp_bg += imp_bg_num / imp_bg_den
                    imp_gap_count += 1

                pred_bin = (imp_up >= float(args.thr)) & (valid_mask > 0.5)
                gt_bin = (roi_mask >= 0.5) & (valid_mask > 0.5)
                tp += int((pred_bin & gt_bin).sum().item())
                fp += int((pred_bin & (~gt_bin)).sum().item())
                fn += int(((~pred_bin) & gt_bin).sum().item())
                tn += int(((~pred_bin) & (~gt_bin) & (valid_mask > 0.5)).sum().item())

            for b in range(data.shape[0]):
                entropy_b = float(entropy_vals[b])
                detail_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "checkpoint": str(ckpt_path),
                        "mode": mode,
                        "frame_idx": frame_idx,
                        "all_mse": float(all_mse[b].item()),
                        "roi_mse": float(roi_mse[b].item()),
                        "bg_mse": float(bg_mse[b].item()),
                        "rate_proxy": float(rate_proxy[b].item()),
                        "eq_bits": float(eq_bits[b].item()),
                        "code_entropy": entropy_b,
                        "bpp_eq": float(eq_bits[b].item()) * symbols_per_input_pixel,
                        "bpp_entropy": entropy_b * symbols_per_input_pixel if entropy_b == entropy_b else float("nan"),
                        "imp_mean": float(imp_up[b].mean().item()) if imp_up is not None else float("nan"),
                    }
                )
                frame_idx += 1

            total_all_mse += float(all_mse.sum().item())
            total_roi_mse += float(roi_mse.sum().item())
            total_bg_mse += float(bg_mse.sum().item())
            total_rate_proxy += float(rate_proxy.sum().item())
            total_eq_bits += float(eq_bits.sum().item())
            total_entropy += float(sum(entropy_vals))
            total_bpp_eq += float(eq_bits.sum().item()) * symbols_per_input_pixel
            total_bpp_entropy += float(sum(entropy_vals)) * symbols_per_input_pixel
            sample_count += data.shape[0]

        if sample_count == 0:
            continue

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        iou = _safe_div(tp, tp + fp + fn)
        imp_roi_mean = _safe_div(total_imp_roi, imp_gap_count)
        imp_bg_mean = _safe_div(total_imp_bg, imp_gap_count)
        imp_gap = imp_roi_mean - imp_bg_mean if imp_roi_mean == imp_roi_mean and imp_bg_mean == imp_bg_mean else float("nan")

        summary_rows.append(
            {
                "run_dir": str(run_dir),
                "checkpoint": str(ckpt_path),
                "quantizer_mode": quantizer_mode,
                "quant_bits": quant_bits,
                "mode": mode,
                "frames": sample_count,
                "all_mse_mean": total_all_mse / sample_count,
                "roi_mse_mean": total_roi_mse / sample_count,
                "bg_mse_mean": total_bg_mse / sample_count,
                "rate_proxy_mean": total_rate_proxy / sample_count,
                "eq_bits_mean": total_eq_bits / sample_count,
                "code_entropy_mean": total_entropy / sample_count,
                "bpp_eq_mean": total_bpp_eq / sample_count,
                "bpp_entropy_mean": total_bpp_entropy / sample_count,
                "importance_roi_mean": imp_roi_mean,
                "importance_bg_mean": imp_bg_mean,
                "importance_gap": imp_gap,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "iou": iou,
            }
        )

    out_summary = Path(args.output_summary_csv)
    out_detail = Path(args.output_detail_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_detail.parent.mkdir(parents=True, exist_ok=True)

    if summary_rows:
        with out_summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    if detail_rows:
        with out_detail.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {ckpt_path}")
    for row in summary_rows:
        print(
            f"[{row['mode']}] frames={row['frames']} "
            f"all_mse={row['all_mse_mean']:.6f} roi_mse={row['roi_mse_mean']:.6f} bg_mse={row['bg_mse_mean']:.6f} "
            f"eq_bits={row['eq_bits_mean']:.3f} entropy={row['code_entropy_mean']:.3f} "
            f"bpp_eq={row['bpp_eq_mean']:.3f} bpp_entropy={row['bpp_entropy_mean']:.3f} "
            f"iou={row['iou']:.4f} precision={row['precision']:.4f} recall={row['recall']:.4f}"
        )
    print(f"Saved summary CSV: {out_summary}")
    print(f"Saved detail CSV: {out_detail}")


if __name__ == "__main__":
    main()
