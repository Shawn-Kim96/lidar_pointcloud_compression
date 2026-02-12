import argparse
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Subset

from dataset.semantickitti_loader import SemanticKittiDataset
from models.autoencoder import RangeCompressionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 evaluation with Chamfer Distance")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--quant_bits", type=int, default=8, help="Quantization bit depth used during inference")
    parser.add_argument(
        "--data_dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "dataset" / "semantickitti" / "dataset" / "sequences"),
        help="Path to SemanticKITTI sequences root",
    )
    parser.add_argument("--val_seq", default="08", help="Validation sequence id")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=16,
        help="Maximum number of frames to evaluate (default: 16 for quick sanity runs)",
    )
    parser.add_argument(
        "--log_path",
        default=str(Path(__file__).resolve().parents[2] / "logs" / "stage1_eval.out"),
        help="Output log path for evaluation metrics",
    )
    return parser.parse_args()


def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return float("inf")

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    dist_a_to_b, _ = tree_b.query(points_a, k=1, workers=-1)
    dist_b_to_a, _ = tree_a.query(points_b, k=1, workers=-1)

    return float(dist_a_to_b.mean() + dist_b_to_a.mean())


def masked_xyz(tensor: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    xyz = tensor[2:5]
    valid = mask > 0
    xyz_points = xyz[:, valid].transpose(0, 1).contiguous()
    return xyz_points.detach().cpu().numpy()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask > 0
    if not torch.any(valid):
        return float("inf")
    diff = pred[valid] - target[valid]
    return float(torch.mean(diff * diff).item())


def psnr_from_mse(mse: float, peak: float) -> float:
    if mse <= 0.0:
        return float("inf")
    peak = max(peak, 1e-6)
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def estimate_bpp(latent_codes: torch.Tensor, valid_points: int) -> float:
    # Approximate entropy-coded rate: H(codes) * number_of_codes / valid_points
    if valid_points <= 0:
        return float("inf")
    codes = torch.round(latent_codes).to(torch.int64).detach().cpu().numpy().reshape(-1)
    if codes.size == 0:
        return float("inf")
    min_code = int(codes.min())
    shifted = codes - min_code
    hist = np.bincount(shifted)
    probs = hist[hist > 0].astype(np.float64)
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log2(probs))
    total_bits = entropy * float(codes.size)
    return float(total_bits / valid_points)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RangeCompressionModel(quant_bits=args.quant_bits).to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dataset = SemanticKittiDataset(root_dir=args.data_dir, sequences=[args.val_seq])
    if len(dataset) == 0:
        raise RuntimeError(f"No validation frames found in {args.data_dir} for sequence {args.val_seq}")

    if args.max_frames is not None:
        max_frames = max(1, min(args.max_frames, len(dataset)))
        dataset = Subset(dataset, list(range(max_frames)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cds = []
    bpps = []
    psnr_ranges = []
    psnr_intensities = []
    with torch.no_grad():
        for data, mask in loader:
            data = data.to(device)
            mask = mask.to(device)
            recon, latent_codes = model(data, noise_std=0.0, quantize=True)

            batch_size = data.shape[0]
            for batch_idx in range(batch_size):
                sample_mask = mask[batch_idx] > 0
                valid_points = int(sample_mask.sum().item())

                range_gt = data[batch_idx, 0]
                range_recon = recon[batch_idx, 0]
                intensity_gt = data[batch_idx, 1]
                intensity_recon = recon[batch_idx, 1]

                mse_range = masked_mse(range_recon, range_gt, sample_mask)
                mse_intensity = masked_mse(intensity_recon, intensity_gt, sample_mask)

                peak_range = float(torch.max(range_gt[sample_mask]).item()) if valid_points > 0 else 1.0
                peak_intensity = float(torch.max(intensity_gt[sample_mask]).item()) if valid_points > 0 else 1.0

                psnr_ranges.append(psnr_from_mse(mse_range, peak_range))
                psnr_intensities.append(psnr_from_mse(mse_intensity, peak_intensity))
                bpps.append(estimate_bpp(latent_codes[batch_idx], valid_points))

                gt_points = masked_xyz(data[batch_idx], mask[batch_idx])
                recon_points = masked_xyz(recon[batch_idx], mask[batch_idx])
                cd = chamfer_distance(gt_points, recon_points)
                cds.append(cd)

    if not cds or not bpps or not psnr_ranges or not psnr_intensities:
        raise RuntimeError("Evaluation metrics could not be computed: no valid samples.")

    mean_bpp = float(np.mean(bpps))
    mean_psnr_range = float(np.mean(psnr_ranges))
    mean_psnr_intensity = float(np.mean(psnr_intensities))
    mean_cd = float(np.mean(cds))
    print(f"BPP: {mean_bpp:.6f}")
    print(f"PSNR (Range): {mean_psnr_range:.6f}")
    print(f"PSNR (Intensity): {mean_psnr_intensity:.6f}")
    print(f"Chamfer Distance: {mean_cd:.6f}")
    # Keep final line machine-parsable and float-only for quick checks.
    print(f"{mean_bpp:.6f} {mean_psnr_range:.6f} {mean_psnr_intensity:.6f} {mean_cd:.6f}")

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"model={args.model} qbits={args.quant_bits} seq={args.val_seq} frames={len(cds)} "
            f"bpp={mean_bpp:.6f} psnr_range={mean_psnr_range:.6f} "
            f"psnr_intensity={mean_psnr_intensity:.6f} cd={mean_cd:.6f}\n"
        )


if __name__ == "__main__":
    main()
