import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.autoencoder import RangeCompressionModel
from dataset.semantickitti_loader import SemanticKittiDataset
import argparse
from pathlib import Path
from torch.utils.data import Subset

from utils.experiment import default_run_paths, make_run_id, write_manifest

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 baseline training")
    parser.add_argument(
        "--data_dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "dataset" / "semantickitti" / "dataset" / "sequences"),
        help="Path to SemanticKITTI sequences root",
    )
    parser.add_argument(
        "--train_seq",
        default=None,
        help="Single training sequence id (legacy option).",
    )
    parser.add_argument(
        "--train_seqs",
        nargs="+",
        default=["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
        help="Training sequence ids. Default follows common SemanticKITTI train split.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Latent Gaussian noise std")
    parser.add_argument("--quant_bits", type=int, default=8, help="Latent quantization bits for model config")
    parser.add_argument("--in_channels", type=int, default=5, help="Input channel count (default: 5)")
    parser.add_argument("--base_channels", type=int, default=64, help="Encoder base channels (default: 64)")
    parser.add_argument("--latent_channels", type=int, default=64, help="Latent channels (default: 64)")
    parser.add_argument("--num_stages", type=int, default=4, help="Downsample stages (default: 4)")
    parser.add_argument("--blocks_per_stage", type=int, default=1, help="Residual blocks per stage (default: 1)")
    parser.add_argument("--norm", type=str, default="batch", choices=["batch", "group", "none"], help="Norm type")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "silu", "gelu", "leaky_relu"],
        help="Activation function",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout2d probability (default: 0.0)")
    parser.add_argument("--range_w", type=float, default=1.0, help="Weight for range reconstruction loss")
    parser.add_argument("--intensity_w", type=float, default=1.0, help="Weight for intensity reconstruction loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Checkpoint filename saved under data/results/checkpoints/ (default: stage1_baseline.pth, debug.pth in --debug)",
    )
    parser.add_argument(
        "--alias_name",
        type=str,
        default=None,
        help="Optional alias checkpoint name (copied into data/results/checkpoints/).",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run id. If set, writes run artifacts under data/results/runs/<run_id>/.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="stage1",
        help="Human name used in auto run-id generation when --run_id is not provided.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a short dry-run and save data/results/checkpoints/debug.pth",
    )
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device, noise_std, range_w, intensity_w):
    model.train()
    total_total = 0.0
    total_range = 0.0
    total_intensity = 0.0
    for i, (data, mask) in enumerate(loader):
        data = data.to(device) # [B, 5, H, W]
        mask = mask.to(device).unsqueeze(1) # [B, 1, H, W]
        
        optimizer.zero_grad()
        recon, _ = model(data, noise_std=noise_std)

        range_loss = criterion(recon[:, 0:1] * mask, data[:, 0:1] * mask)
        intensity_loss = criterion(recon[:, 1:2] * mask, data[:, 1:2] * mask)
        loss = (range_w * range_loss) + (intensity_w * intensity_loss)
        
        loss.backward()
        optimizer.step()
        
        total_total += loss.item()
        total_range += range_loss.item()
        total_intensity += intensity_loss.item()
        
        if i % 10 == 0:
            print(
                f"Step {i}, Total: {loss.item():.4f}, "
                f"Range: {range_loss.item():.4f}, Intensity: {intensity_loss.item():.4f}"
            )

    num_batches = len(loader)
    return (
        total_total / num_batches,
        total_range / num_batches,
        total_intensity / num_batches,
    )

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    repo_root = Path(__file__).resolve().parents[2]
    
    model = RangeCompressionModel(
        quant_bits=args.quant_bits,
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        blocks_per_stage=args.blocks_per_stage,
        norm=args.norm,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    train_sequences = [args.train_seq] if args.train_seq is not None else args.train_seqs
    print(f"Training sequences: {train_sequences}")

    dataset = SemanticKittiDataset(root_dir=args.data_dir, sequences=train_sequences)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No training frames found in {args.data_dir} for sequences {train_sequences}"
        )

    if args.debug:
        debug_count = min(16, len(dataset))
        dataset = Subset(dataset, list(range(debug_count)))
        epochs = 4
        num_workers = 0
    else:
        epochs = args.epochs
        num_workers = args.num_workers

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Starting training: epochs={epochs}, batches={len(loader)}")
    history_range = []
    history_intensity = []
    for epoch in range(epochs):
        total_loss, range_loss, intensity_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            noise_std=args.noise_std,
            range_w=args.range_w,
            intensity_w=args.intensity_w,
        )
        history_range.append(range_loss)
        history_intensity.append(intensity_loss)
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Total: {total_loss:.6f}, Range: {range_loss:.6f}, Intensity: {intensity_loss:.6f}"
        )

    if args.debug and history_range and history_intensity:
        range_reduced = history_range[-1] < history_range[0]
        intensity_reduced = history_intensity[-1] < history_intensity[0]
        print(f"Debug reduction check - Range: {history_range[0]:.6f} -> {history_range[-1]:.6f} ({range_reduced})")
        print(
            f"Debug reduction check - Intensity: {history_intensity[0]:.6f} -> "
            f"{history_intensity[-1]:.6f} ({intensity_reduced})"
        )

    checkpoint_dir = repo_root / "data" / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    model_info = {
        "model_class": type(model).__name__,
        "encoder_class": type(model.encoder).__name__,
        "decoder_class": type(model.decoder).__name__,
        "quantizer_class": type(model.quantizer).__name__,
        "in_channels": args.in_channels,
        "base_channels": args.base_channels,
        "latent_channels": args.latent_channels,
        "num_stages": args.num_stages,
        "blocks_per_stage": args.blocks_per_stage,
        "norm": args.norm,
        "activation": args.activation,
        "dropout": args.dropout,
    }
    run_id = args.run_id
    if run_id is None:
        run_id = make_run_id(stage="1", name=args.run_name, config=config)
    run_paths = default_run_paths(run_id)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "model_info": model_info,
        },
        run_paths.checkpoint_path,
    )
    write_manifest(
        run_paths.manifest_path,
        stage="stage1",
        run_id=run_id,
        command=" ".join(os.sys.argv),
        config=config,
        model_info=model_info,
        notes="Auto-generated manifest; use this to track hyperparameters for checkpoints/logs.",
    )
    print(f"Run dir: {run_paths.run_dir}")
    print(f"Run checkpoint: {run_paths.checkpoint_path}")
    print(f"Run manifest: {run_paths.manifest_path}")

    # Save a conventional checkpoint file name as well (optional).
    if args.save_name is None or args.save_name == "auto":
        save_name = f"{run_id}.pth"
    else:
        save_name = args.save_name
    ckpt_path = checkpoint_dir / save_name
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    if args.alias_name:
        alias_path = checkpoint_dir / args.alias_name
        torch.save(model.state_dict(), alias_path)
        print(f"Saved alias checkpoint: {alias_path}")

if __name__ == "__main__":
    main()
