import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset.semantickitti_loader import SemanticKittiDataset
from loss.task_loss import TaskLossModule
from models.adaptive_autoencoder import AdaptiveRangeCompressionModel
from utils.experiment import default_run_paths, make_run_id, write_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 joint training (rate-distortion-task)")
    parser.add_argument(
        "--data_dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "dataset" / "semantickitti" / "dataset" / "sequences"),
        help="Path to SemanticKITTI sequences root",
    )
    parser.add_argument(
        "--train_seqs",
        nargs="+",
        default=["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
        help="Train sequence ids",
    )
    parser.add_argument("--val_seq", default="08", help="Validation sequence id")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--lambda_task", type=float, default=1.0)
    parser.add_argument("--beta_entropy", type=float, default=0.01)
    parser.add_argument("--roi_recon_weight", type=float, default=10.0)
    parser.add_argument("--xyz_weight", type=float, default=0.5)
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
    parser.add_argument("--roi_levels", type=int, default=256, help="ROI quantization levels (default: 256)")
    parser.add_argument("--bg_levels", type=int, default=16, help="BG quantization levels (default: 16)")
    parser.add_argument(
        "--val_eval_batches",
        type=int,
        default=64,
        help="Number of validation batches for per-epoch proxy mAP check",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init_stage1",
        default=str(Path(__file__).resolve().parents[2] / "data" / "results" / "checkpoints" / "stage1_baseline.pth"),
    )
    parser.add_argument("--save_name", default="stage2_adaptive.pth")
    parser.add_argument(
        "--alias_name",
        type=str,
        default=None,
        help="Optional alias checkpoint name (copied into data/results/checkpoints/).",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Optional run id. If set, writes run artifacts under data/results/runs/<run_id>/.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="stage2",
        help="Human name used in auto run-id generation when --run_id is not provided.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def weighted_l1(pred, target, valid_mask, roi_mask=None, roi_weight=1.0):
    weights = valid_mask.float()
    if roi_mask is not None:
        weights = weights * (1.0 + (roi_weight - 1.0) * roi_mask.float())
    diff = torch.abs(pred - target) * weights
    return diff.sum() / weights.sum().clamp(min=1.0)


def _load_partial_state(model, state_dict):
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return missing, unexpected, skipped


@torch.no_grad()
def evaluate_proxy_map(model, task_module, loader, device, max_batches=None):
    model.eval()
    task_module.eval()

    map_values = []
    for batch_index, batch in enumerate(loader):
        data, valid_mask, roi_mask = batch
        data = data.to(device)
        valid_mask = valid_mask.to(device).unsqueeze(1)
        roi_mask = roi_mask.to(device)

        recon, _ = model(data, roi_mask=roi_mask, noise_std=0.0, quantize=True)
        _, details = task_module(recon, roi_mask, valid_mask=valid_mask, return_details=True)
        map_values.append(details["map_proxy"])

        if max_batches is not None and (batch_index + 1) >= max_batches:
            break

    if not map_values:
        return 0.0
    return float(sum(map_values) / len(map_values))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train sequences: {args.train_seqs}")
    print(f"Val sequence: {args.val_seq}")

    model = AdaptiveRangeCompressionModel(
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        blocks_per_stage=args.blocks_per_stage,
        norm=args.norm,
        activation=args.activation,
        dropout=args.dropout,
        roi_levels=args.roi_levels,
        bg_levels=args.bg_levels,
        quant_use_ste=True,
    ).to(device)
    task_module = TaskLossModule(backend="auto").to(device)

    init_path = Path(args.init_stage1)
    if init_path.exists():
        stage1_state = torch.load(init_path, map_location=device)
        if isinstance(stage1_state, dict) and "model_state" in stage1_state:
            stage1_state = stage1_state["model_state"]
        missing, unexpected, skipped = _load_partial_state(model, stage1_state)
        print(f"Loaded Stage1 init from {init_path}")
        print(
            f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}, "
            f"Skipped shape mismatch: {len(skipped)}"
        )
    else:
        print(f"Stage1 init not found at {init_path}, training from scratch.")

    train_dataset = SemanticKittiDataset(
        root_dir=args.data_dir,
        sequences=args.train_seqs,
        return_roi_mask=True,
    )
    val_dataset = SemanticKittiDataset(
        root_dir=args.data_dir,
        sequences=[args.val_seq],
        return_roi_mask=True,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Train/val dataset is empty. Check data path and sequences.")

    epochs = args.epochs
    num_workers = args.num_workers
    val_eval_batches = args.val_eval_batches
    if args.debug:
        train_dataset = Subset(train_dataset, list(range(min(128, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(128, len(val_dataset)))))
        epochs = min(6, args.epochs)
        num_workers = 0
        val_eval_batches = min(8, args.val_eval_batches)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if args.debug else max(1, num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = optim.Adam(
        list(model.parameters()) + list(task_module.parameters()),
        lr=args.lr,
    )

    print(f"Starting Stage2 training: epochs={epochs}, train_batches={len(train_loader)}")
    map_history = []
    total_history = []

    for epoch in range(epochs):
        model.train()
        task_module.train()

        sum_total = 0.0
        sum_recon = 0.0
        sum_task = 0.0
        sum_entropy = 0.0

        for step, batch in enumerate(train_loader):
            data, valid_mask, roi_mask = batch
            data = data.to(device)
            valid_mask = valid_mask.to(device).unsqueeze(1)
            roi_mask = roi_mask.to(device)

            optimizer.zero_grad()
            recon, aux = model(
                data,
                roi_mask=roi_mask,
                noise_std=args.noise_std,
                quantize=True,
            )

            loss_range = weighted_l1(
                recon[:, 0:1],
                data[:, 0:1],
                valid_mask=valid_mask,
                roi_mask=roi_mask,
                roi_weight=args.roi_recon_weight,
            )
            loss_intensity = weighted_l1(
                recon[:, 1:2],
                data[:, 1:2],
                valid_mask=valid_mask,
                roi_mask=roi_mask,
                roi_weight=args.roi_recon_weight,
            )
            loss_xyz = weighted_l1(
                recon[:, 2:5],
                data[:, 2:5],
                valid_mask=valid_mask,
                roi_mask=roi_mask,
                roi_weight=args.roi_recon_weight,
            )
            recon_loss = loss_range + loss_intensity + (args.xyz_weight * loss_xyz)

            task_loss, task_details = task_module(
                recon,
                roi_mask=roi_mask,
                valid_mask=valid_mask,
                return_details=True,
            )
            entropy_loss = aux["latent_dequant"].abs().mean()
            total_loss = recon_loss + (args.lambda_task * task_loss) + (args.beta_entropy * entropy_loss)

            total_loss.backward()
            optimizer.step()

            sum_total += float(total_loss.item())
            sum_recon += float(recon_loss.item())
            sum_task += float(task_loss.item())
            sum_entropy += float(entropy_loss.item())

            if step % 200 == 0:
                print(
                    f"Epoch {epoch+1} Step {step}: "
                    f"total={total_loss.item():.4f} "
                    f"recon={recon_loss.item():.4f} "
                    f"task={task_loss.item():.4f} "
                    f"entropy={entropy_loss.item():.4f} "
                    f"map_proxy={task_details['map_proxy']:.4f}"
                )

        num_batches = max(1, len(train_loader))
        avg_total = sum_total / num_batches
        avg_recon = sum_recon / num_batches
        avg_task = sum_task / num_batches
        avg_entropy = sum_entropy / num_batches
        total_history.append(avg_total)

        val_map = evaluate_proxy_map(
            model=model,
            task_module=task_module,
            loader=val_loader,
            device=device,
            max_batches=val_eval_batches,
        )
        map_history.append(val_map)

        print(
            f"Epoch {epoch+1}/{epochs}: total={avg_total:.6f} "
            f"recon={avg_recon:.6f} task={avg_task:.6f} entropy={avg_entropy:.6f} "
            f"val_map_proxy={val_map:.6f}"
        )

    if map_history:
        map_improved = map_history[-1] > map_history[0]
        print(f"mAP(proxy) improvement: {map_history[0]:.6f} -> {map_history[-1]:.6f} ({map_improved})")
    if total_history:
        loss_decreased = total_history[-1] < total_history[0]
        print(f"Total loss trend: {total_history[0]:.6f} -> {total_history[-1]:.6f} ({loss_decreased})")

    repo_root = Path(__file__).resolve().parents[2]
    checkpoints_dir = repo_root / "data" / "results" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    model_info = {
        "model_class": type(model).__name__,
        "encoder_class": type(model.encoder).__name__,
        "decoder_class": type(model.decoder).__name__,
        "quantizer_class": type(model.quantizer).__name__,
        "roi_levels": args.roi_levels,
        "bg_levels": args.bg_levels,
        "in_channels": args.in_channels,
        "base_channels": args.base_channels,
        "latent_channels": args.latent_channels,
        "num_stages": args.num_stages,
        "blocks_per_stage": args.blocks_per_stage,
        "norm": args.norm,
        "activation": args.activation,
        "dropout": args.dropout,
        "task_backend": getattr(task_module, "backend", "unknown"),
    }
    run_id = args.run_id
    if run_id is None:
        run_id = make_run_id(stage="2", name=args.run_name, config=config)
    run_paths = default_run_paths(run_id)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "task_state": task_module.state_dict(),
            "map_history": map_history,
            "loss_history": total_history,
            "config": config,
            "model_info": model_info,
        },
        run_paths.checkpoint_path,
    )
    write_manifest(
        run_paths.manifest_path,
        stage="stage2",
        run_id=run_id,
        command=" ".join(__import__("sys").argv),
        config=config,
        model_info=model_info,
        notes="Auto-generated manifest; use this to track hyperparameters for checkpoints/logs.",
    )
    print(f"Run dir: {run_paths.run_dir}")
    print(f"Run checkpoint: {run_paths.checkpoint_path}")
    print(f"Run manifest: {run_paths.manifest_path}")

    # Save checkpoint using run_id-based naming by default.
    if args.save_name is None or args.save_name == "auto":
        save_name = f"{run_id}.pth"
    else:
        save_name = args.save_name
    ckpt_path = checkpoints_dir / save_name
    torch.save(
        {
            "model_state": model.state_dict(),
            "task_state": task_module.state_dict(),
            "map_history": map_history,
            "loss_history": total_history,
            "config": config,
            "model_info": model_info,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

    if args.alias_name:
        alias_path = checkpoints_dir / args.alias_name
        torch.save(
            {
                "model_state": model.state_dict(),
                "task_state": task_module.state_dict(),
                "map_history": map_history,
                "loss_history": total_history,
                "config": config,
                "model_info": model_info,
            },
            alias_path,
        )
        print(f"Saved alias checkpoint: {alias_path}")


if __name__ == "__main__":
    main()
