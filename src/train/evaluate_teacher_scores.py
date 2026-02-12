import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.semantickitti_loader import SemanticKittiDataset
from models.adaptive_autoencoder import AdaptiveRangeCompressionModel
from models.autoencoder import RangeCompressionModel
from utils.teacher_adapter import TeacherAdapter, TeacherAdapterConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate teacher scores on original vs reconstructed frames")
    repo_root = Path(__file__).resolve().parents[2]

    parser.add_argument(
        "--data_dir",
        default=str(repo_root / "data" / "dataset" / "semantickitti" / "dataset" / "sequences"),
    )
    parser.add_argument("--val_seq", default="08")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=512)

    parser.add_argument("--model_stage", choices=["stage1", "stage2", "stage2_1"], default="stage2_1")
    parser.add_argument(
        "--model_ckpt",
        default=str(repo_root / "data" / "results" / "checkpoints" / "stage2_adaptive.pth"),
    )
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--roi_levels", type=int, default=256)
    parser.add_argument("--bg_levels", type=int, default=16)
    parser.add_argument("--no_labels", action="store_true", help="Run without loading ROI labels.")

    parser.add_argument("--teacher_backend", choices=["auto", "proxy", "openpcdet"], default="auto")
    parser.add_argument(
        "--teacher_ckpt",
        default=str(repo_root / "data" / "results" / "checkpoints" / "stage2_adaptive.pth"),
    )
    parser.add_argument("--teacher_score_topk_ratio", type=float, default=0.01)
    parser.add_argument(
        "--output_csv",
        default=str(repo_root / "logs" / "stage2_1_teacher_scores_seq08.csv"),
    )
    return parser.parse_args()


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        if "model_state" in payload:
            return payload["model_state"]
    return payload


def _load_model(args, device):
    ckpt_path = Path(args.model_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if args.model_stage == "stage1":
        model = RangeCompressionModel(quant_bits=args.quant_bits).to(device)
        state = _extract_state_dict(ckpt)
        model.load_state_dict(state, strict=False)
        return model

    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = AdaptiveRangeCompressionModel(
        in_channels=int(config.get("in_channels", 5)),
        latent_channels=int(config.get("latent_channels", 64)),
        base_channels=int(config.get("base_channels", 64)),
        num_stages=int(config.get("num_stages", 4)),
        blocks_per_stage=int(config.get("blocks_per_stage", 1)),
        norm=str(config.get("norm", "batch")),
        activation=str(config.get("activation", "relu")),
        dropout=float(config.get("dropout", 0.0)),
        roi_levels=int(config.get("roi_levels", args.roi_levels)),
        bg_levels=int(config.get("bg_levels", args.bg_levels)),
        quant_use_ste=False,
        importance_hidden_channels=int(config.get("importance_hidden_channels", 64)),
        importance_from_latent=not bool(config.get("importance_from_input", False))
        if "importance_from_input" in config
        else bool(config.get("importance_from_latent", True)),
        importance_min=float(config.get("importance_min", 0.01)),
        importance_max=float(config.get("importance_max", 0.99)),
    ).to(device)
    model.load_state_dict(_extract_state_dict(ckpt), strict=False)
    return model


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args, device).eval()

    return_roi = not args.no_labels
    dataset = SemanticKittiDataset(
        root_dir=args.data_dir,
        sequences=[args.val_seq],
        return_roi_mask=return_roi,
    )
    if args.max_frames is not None:
        dataset = Subset(dataset, list(range(min(args.max_frames, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    teacher = TeacherAdapter(
        TeacherAdapterConfig(
            backend=args.teacher_backend,
            proxy_ckpt=args.teacher_ckpt if args.teacher_ckpt else None,
            device="cuda" if torch.cuda.is_available() else "cpu",
            score_topk_ratio=args.teacher_score_topk_ratio,
            in_channels=5,
            hidden_channels=32,
        )
    )
    print(f"Teacher backend: {teacher.backend}")

    rows = []
    frame_index = 0
    for batch in loader:
        if len(batch) == 3:
            data, valid_mask, roi_mask = batch
            roi_mask = roi_mask.to(device)
        else:
            data, valid_mask = batch
            roi_mask = None
        data = data.to(device)
        valid_mask = valid_mask.to(device).unsqueeze(1)

        if args.model_stage == "stage1":
            recon, _ = model(data, noise_std=0.0, quantize=True)
        else:
            recon, _ = model(data, roi_mask=roi_mask, noise_std=0.0, quantize=True)

        orig_out = teacher.infer(data, valid_mask=valid_mask)
        recon_out = teacher.infer(recon, valid_mask=valid_mask)
        orig_scores = orig_out["score"].detach().cpu().numpy()
        recon_scores = recon_out["score"].detach().cpu().numpy()

        for i in range(orig_scores.shape[0]):
            o = float(orig_scores[i])
            r = float(recon_scores[i])
            rows.append(
                {
                    "frame_idx": frame_index,
                    "teacher_score_original": o,
                    "teacher_score_reconstructed": r,
                    "teacher_score_drop": o - r,
                }
            )
            frame_index += 1

    if not rows:
        raise RuntimeError("No frames evaluated.")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "teacher_score_original",
                "teacher_score_reconstructed",
                "teacher_score_drop",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    orig_mean = float(np.mean([r["teacher_score_original"] for r in rows]))
    recon_mean = float(np.mean([r["teacher_score_reconstructed"] for r in rows]))
    drop_mean = float(np.mean([r["teacher_score_drop"] for r in rows]))
    print(f"Frames: {len(rows)}")
    print(f"teacher_score_original_mean: {orig_mean:.6f}")
    print(f"teacher_score_reconstructed_mean: {recon_mean:.6f}")
    print(f"teacher_score_drop_mean: {drop_mean:.6f}")
    print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
    main()
