import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Subset

from dataset.semantickitti_loader import SemanticKittiDataset
from loss.task_loss import TaskLossModule
from models.adaptive_autoencoder import AdaptiveRangeCompressionModel
from models.autoencoder import RangeCompressionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 evaluation and Stage1/Stage2 comparison")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--stage1_model",
        default=str(repo_root / "data" / "results" / "checkpoints" / "stage1_baseline.pth"),
    )
    parser.add_argument(
        "--stage2_ckpt",
        default=str(repo_root / "data" / "results" / "checkpoints" / "stage2_adaptive.pth"),
    )
    parser.add_argument(
        "--data_dir",
        default=str(repo_root / "data" / "dataset" / "semantickitti" / "dataset" / "sequences"),
    )
    parser.add_argument("--val_seq", default="08")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=512)
    parser.add_argument(
        "--output_plot",
        default=str(repo_root / "data" / "results" / "visualization" / "comparison_stage1_stage2.png"),
    )
    parser.add_argument(
        "--output_log",
        default=str(repo_root / "logs" / "stage2_eval_comparison.out"),
    )
    return parser.parse_args()


def masked_xyz(tensor, mask):
    xyz = tensor[2:5]
    valid = mask > 0
    return xyz[:, valid].transpose(0, 1).contiguous().detach().cpu().numpy()


def chamfer_distance(points_a, points_b):
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return float("inf")
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    dist_a_to_b, _ = tree_b.query(points_a, k=1, workers=-1)
    dist_b_to_a, _ = tree_a.query(points_b, k=1, workers=-1)
    return float(dist_a_to_b.mean() + dist_b_to_a.mean())


def masked_mse(pred, target, mask):
    valid = mask > 0
    if not torch.any(valid):
        return float("inf")
    diff = pred[valid] - target[valid]
    return float(torch.mean(diff * diff).item())


def psnr_from_mse(mse, peak):
    if mse <= 0.0:
        return float("inf")
    peak = max(float(peak), 1e-6)
    return 20.0 * math.log10(peak) - 10.0 * math.log10(float(mse))


def estimate_bpp(latent_codes, valid_points):
    if valid_points <= 0:
        return float("inf")
    codes = torch.round(latent_codes).to(torch.int64).detach().cpu().numpy().reshape(-1)
    if codes.size == 0:
        return float("inf")
    shifted = codes - int(codes.min())
    hist = np.bincount(shifted)
    probs = hist[hist > 0].astype(np.float64)
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return float((entropy * float(codes.size)) / valid_points)


def load_partial_state(model, state_dict):
    target = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in target and target[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    skipped = len(state_dict) - len(compatible)
    return missing, unexpected, skipped


@torch.no_grad()
def evaluate_model(name, model, task_module, loader, device, quant_bits, is_stage2):
    bpps = []
    maps = []
    cds = []
    psnr_ranges = []
    psnr_intensities = []

    model.eval()
    task_module.eval()

    for batch in loader:
        data, valid_mask, roi_mask = batch
        data = data.to(device)
        valid_mask = valid_mask.to(device)
        valid_mask_4d = valid_mask.unsqueeze(1)
        roi_mask = roi_mask.to(device)

        if is_stage2:
            recon, aux = model(data, roi_mask=roi_mask, noise_std=0.0, quantize=True)
            latent_codes = aux["latent_codes"]
        else:
            recon, latent_codes = model(data, noise_std=0.0, quantize=True)

        _, details = task_module(recon, roi_mask, valid_mask=valid_mask_4d, return_details=True)
        maps.append(details["map_proxy"])

        for batch_idx in range(data.shape[0]):
            vp = int(valid_mask[batch_idx].sum().item())
            bpps.append(estimate_bpp(latent_codes[batch_idx], vp))

            range_gt = data[batch_idx, 0]
            range_recon = recon[batch_idx, 0]
            intensity_gt = data[batch_idx, 1]
            intensity_recon = recon[batch_idx, 1]
            mask_sample = valid_mask[batch_idx] > 0

            mse_range = masked_mse(range_recon, range_gt, mask_sample)
            mse_intensity = masked_mse(intensity_recon, intensity_gt, mask_sample)
            peak_range = float(torch.max(range_gt[mask_sample]).item()) if vp > 0 else 1.0
            peak_intensity = float(torch.max(intensity_gt[mask_sample]).item()) if vp > 0 else 1.0
            psnr_ranges.append(psnr_from_mse(mse_range, peak_range))
            psnr_intensities.append(psnr_from_mse(mse_intensity, peak_intensity))

            gt_points = masked_xyz(data[batch_idx], valid_mask[batch_idx])
            recon_points = masked_xyz(recon[batch_idx], valid_mask[batch_idx])
            cds.append(chamfer_distance(gt_points, recon_points))

    metrics = {
        "name": name,
        "quant_bits": quant_bits,
        "mAP_3d": float(np.mean(maps)) if maps else 0.0,
        "p_BPP": float(np.mean(bpps)) if bpps else float("inf"),
        "PSNR_Range": float(np.mean(psnr_ranges)) if psnr_ranges else float("inf"),
        "PSNR_Intensity": float(np.mean(psnr_intensities)) if psnr_intensities else float("inf"),
        "CD": float(np.mean(cds)) if cds else float("inf"),
    }
    return metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SemanticKittiDataset(
        root_dir=args.data_dir,
        sequences=[args.val_seq],
        return_roi_mask=True,
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

    stage2_ckpt_path = Path(args.stage2_ckpt)
    if not stage2_ckpt_path.exists():
        raise FileNotFoundError(f"Stage2 checkpoint not found: {stage2_ckpt_path}")
    stage2_ckpt = torch.load(stage2_ckpt_path, map_location=device)

    task_module = TaskLossModule(backend="auto").to(device)
    if "task_state" in stage2_ckpt:
        task_module.load_state_dict(stage2_ckpt["task_state"], strict=False)

    rows = []
    for quant_bits in [8, 4]:
        stage1 = RangeCompressionModel(quant_bits=quant_bits).to(device)
        stage1_state = torch.load(args.stage1_model, map_location=device)
        if isinstance(stage1_state, dict) and "model_state" in stage1_state:
            stage1_state = stage1_state["model_state"]
        missing, unexpected, skipped = load_partial_state(stage1, stage1_state)
        if missing or unexpected or skipped:
            print(f"Stage1 checkpoint compatibility: missing={len(missing)} unexpected={len(unexpected)} skipped={skipped}")
        rows.append(
            evaluate_model(
                name="Stage1",
                model=stage1,
                task_module=task_module,
                loader=loader,
                device=device,
                quant_bits=quant_bits,
                is_stage2=False,
            )
        )

        roi_levels = 2 ** quant_bits
        bg_levels = max(2, roi_levels // 16)
        stage2 = AdaptiveRangeCompressionModel(
            roi_levels=roi_levels,
            bg_levels=bg_levels,
            quant_use_ste=False,
        ).to(device)
        missing, unexpected, skipped = load_partial_state(stage2, stage2_ckpt["model_state"])
        if missing or unexpected or skipped:
            print(f"Stage2 checkpoint compatibility: missing={len(missing)} unexpected={len(unexpected)} skipped={skipped}")
        rows.append(
            evaluate_model(
                name="Stage2",
                model=stage2,
                task_module=task_module,
                loader=loader,
                device=device,
                quant_bits=quant_bits,
                is_stage2=True,
            )
        )

    for row in rows:
        print(
            f"{row['name']} q{row['quant_bits']} "
            f"mAP_3d={row['mAP_3d']:.6f} p_BPP={row['p_BPP']:.6f} "
            f"PSNR_Range={row['PSNR_Range']:.6f} PSNR_Intensity={row['PSNR_Intensity']:.6f} CD={row['CD']:.6f}"
        )

    output_plot = Path(args.output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, marker in [("Stage1", "o"), ("Stage2", "s")]:
        subset = [r for r in rows if r["name"] == name]
        subset = sorted(subset, key=lambda r: r["p_BPP"])
        x = [r["p_BPP"] for r in subset]
        y_map = [r["mAP_3d"] for r in subset]
        y_cd = [r["CD"] for r in subset]
        labels = [f"q{r['quant_bits']}" for r in subset]

        axes[0].plot(x, y_map, marker=marker, label=name)
        axes[1].plot(x, y_cd, marker=marker, label=name)
        for x_i, y_i, label in zip(x, y_map, labels):
            axes[0].annotate(label, (x_i, y_i))
        for x_i, y_i, label in zip(x, y_cd, labels):
            axes[1].annotate(label, (x_i, y_i))

    axes[0].set_title("mAP_3d vs p_BPP")
    axes[0].set_xlabel("p_BPP")
    axes[0].set_ylabel("mAP_3d")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("CD vs p_BPP")
    axes[1].set_xlabel("p_BPP")
    axes[1].set_ylabel("Chamfer Distance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_plot, dpi=160)
    plt.close(fig)
    print(f"Saved comparison plot: {output_plot}")

    output_log = Path(args.output_log)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    with output_log.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(
                f"name={row['name']} qbits={row['quant_bits']} "
                f"mAP_3d={row['mAP_3d']:.6f} p_BPP={row['p_BPP']:.6f} "
                f"PSNR_Range={row['PSNR_Range']:.6f} PSNR_Intensity={row['PSNR_Intensity']:.6f} "
                f"CD={row['CD']:.6f}\n"
            )


if __name__ == "__main__":
    main()
