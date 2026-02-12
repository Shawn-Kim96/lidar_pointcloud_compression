import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset.semantickitti_loader import SemanticKittiDataset
from loss.distill_loss import DistillLoss
from models.adaptive_autoencoder import AdaptiveRangeCompressionModel
from utils.experiment import default_run_paths, make_run_id, write_manifest
from utils.teacher_adapter import TeacherAdapter, TeacherAdapterConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2.1 training: deployable adaptive + teacher distillation")
    repo_root = Path(__file__).resolve().parents[2]

    parser.add_argument(
        "--data_dir",
        default=str(repo_root / "data" / "dataset" / "semantickitti" / "dataset" / "sequences"),
    )
    parser.add_argument(
        "--train_seqs",
        nargs="+",
        default=["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    )
    parser.add_argument("--val_seq", default="08")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--beta_rate", type=float, default=0.005)
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    parser.add_argument("--lambda_importance", type=float, default=0.2)
    parser.add_argument("--importance_supervision", choices=["teacher", "roi", "none"], default="teacher")
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--feature_distill_weight", type=float, default=1.0)
    parser.add_argument("--logit_distill_weight", type=float, default=1.0)

    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--num_stages", type=int, default=4)
    parser.add_argument("--blocks_per_stage", type=int, default=1)
    parser.add_argument("--norm", choices=["batch", "group", "none"], default="batch")
    parser.add_argument("--activation", choices=["relu", "silu", "gelu", "leaky_relu"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--roi_levels", type=int, default=256)
    parser.add_argument("--bg_levels", type=int, default=16)
    parser.add_argument("--importance_hidden_channels", type=int, default=64)
    parser.add_argument(
        "--importance_from_input",
        action="store_true",
        help="Use input tensor (instead of latent feature map) as importance-head source.",
    )
    parser.add_argument("--importance_min", type=float, default=0.01)
    parser.add_argument("--importance_max", type=float, default=0.99)

    parser.add_argument("--teacher_backend", choices=["auto", "proxy", "openpcdet"], default="auto")
    parser.add_argument(
        "--teacher_ckpt",
        default=str(repo_root / "data" / "results" / "checkpoints" / "stage2_adaptive.pth"),
        help="Optional checkpoint for proxy teacher warm start (e.g., Stage2 task head).",
    )
    parser.add_argument("--teacher_score_topk_ratio", type=float, default=0.01)

    parser.add_argument("--val_eval_batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_name", default="stage2_1")
    parser.add_argument("--save_name", default="auto")
    parser.add_argument("--alias_name", default=None)
    parser.add_argument(
        "--init_stage1",
        default=str(repo_root / "data" / "results" / "checkpoints" / "stage1_baseline.pth"),
    )
    parser.add_argument("--no_labels", action="store_true", help="Do not load SemanticKITTI labels for ROI masks.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _weighted_l1(pred, target, valid_mask):
    valid = valid_mask.float()
    diff = torch.abs(pred - target) * valid
    return diff.sum() / valid.sum().clamp(min=1.0)


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


def _importance_supervision_loss(logits, target_map):
    if target_map is None:
        return logits.new_tensor(0.0)
    if target_map.dim() == 3:
        target_map = target_map.unsqueeze(1)
    if target_map.shape[-2:] != logits.shape[-2:]:
        target_map = F.interpolate(target_map.float(), size=logits.shape[-2:], mode="bilinear", align_corners=False)
    return F.binary_cross_entropy_with_logits(logits, target_map.float().clamp(0.0, 1.0))


@torch.no_grad()
def evaluate_teacher_drop(model, teacher, loader, device, max_batches=16):
    model.eval()
    drops = []
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            data, valid_mask, roi_mask = batch
            roi_mask = roi_mask.to(device)
        else:
            data, valid_mask = batch
            roi_mask = None
        data = data.to(device)
        valid_mask = valid_mask.to(device).unsqueeze(1)

        recon, _ = model(data, roi_mask=roi_mask, noise_std=0.0, quantize=True)
        t_orig = teacher.infer(data, valid_mask=valid_mask)
        t_recon = teacher.infer(recon, valid_mask=valid_mask)
        drop = (t_orig["score"] - t_recon["score"]).mean().item()
        drops.append(drop)
        if batch_idx + 1 >= max_batches:
            break
    if not drops:
        return 0.0
    return float(sum(drops) / len(drops))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return_roi_mask = not args.no_labels
    train_dataset = SemanticKittiDataset(
        root_dir=args.data_dir,
        sequences=args.train_seqs,
        return_roi_mask=return_roi_mask,
    )
    val_dataset = SemanticKittiDataset(
        root_dir=args.data_dir,
        sequences=[args.val_seq],
        return_roi_mask=return_roi_mask,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Train/val dataset is empty. Check data paths.")

    epochs = args.epochs
    num_workers = args.num_workers
    val_eval_batches = args.val_eval_batches
    if args.debug:
        train_dataset = Subset(train_dataset, list(range(min(128, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(128, len(val_dataset)))))
        epochs = min(4, args.epochs)
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
        importance_hidden_channels=args.importance_hidden_channels,
        importance_from_latent=not args.importance_from_input,
        importance_min=args.importance_min,
        importance_max=args.importance_max,
    ).to(device)
    init_path = Path(args.init_stage1)
    if init_path.exists():
        stage1_state = torch.load(init_path, map_location=device)
        if isinstance(stage1_state, dict) and "model_state" in stage1_state:
            stage1_state = stage1_state["model_state"]
        missing, unexpected, skipped = _load_partial_state(model, stage1_state)
        print(
            f"Loaded Stage1 init from {init_path} "
            f"(missing={len(missing)} unexpected={len(unexpected)} skipped_shape_mismatch={len(skipped)})"
        )
    else:
        print(f"Stage1 init not found at {init_path}; training from scratch.")

    teacher = TeacherAdapter(
        TeacherAdapterConfig(
            backend=args.teacher_backend,
            proxy_ckpt=args.teacher_ckpt if args.teacher_ckpt else None,
            device="cuda" if torch.cuda.is_available() else "cpu",
            score_topk_ratio=args.teacher_score_topk_ratio,
            in_channels=args.in_channels,
            hidden_channels=32,
        )
    )
    print(f"Teacher backend: {teacher.backend}")

    distill_loss_fn = DistillLoss(
        feature_weight=args.feature_distill_weight,
        logit_weight=args.logit_distill_weight,
        temperature=args.distill_temperature,
        loss_type="mse",
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    teacher_drop_history = []
    total_history = []

    print(f"Starting Stage2.1 training: epochs={epochs}, train_batches={len(train_loader)}")
    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_rate = 0.0
        running_distill = 0.0
        running_importance = 0.0

        for step, batch in enumerate(train_loader):
            if len(batch) == 3:
                data, valid_mask, roi_mask = batch
                roi_mask = roi_mask.to(device)
            else:
                data, valid_mask = batch
                roi_mask = None

            data = data.to(device)
            valid_mask = valid_mask.to(device).unsqueeze(1)

            optimizer.zero_grad()
            recon, aux = model(data, roi_mask=roi_mask, noise_std=args.noise_std, quantize=True)

            recon_range = _weighted_l1(recon[:, 0:1], data[:, 0:1], valid_mask)
            recon_intensity = _weighted_l1(recon[:, 1:2], data[:, 1:2], valid_mask)
            recon_xyz = _weighted_l1(recon[:, 2:5], data[:, 2:5], valid_mask)
            recon_loss = recon_range + recon_intensity + (0.5 * recon_xyz)

            teacher_orig = teacher.infer(data, valid_mask=valid_mask)
            teacher_recon = teacher.infer(recon, valid_mask=valid_mask)
            distill_loss, distill_details = distill_loss_fn(
                student_features=teacher_recon["features"],
                teacher_features=teacher_orig["features"],
                student_logits=teacher_recon["logits"],
                teacher_logits=teacher_orig["logits"],
                importance_map=teacher_orig["importance_map"],
                return_details=True,
            )

            importance_target = None
            if args.importance_supervision == "teacher":
                importance_target = teacher_orig["importance_map"].detach()
            elif args.importance_supervision == "roi" and roi_mask is not None:
                importance_target = roi_mask
            importance_loss = _importance_supervision_loss(aux["importance_logits"], importance_target)

            rate_loss = aux["latent_dequant"].abs().mean()
            total_loss = (
                recon_loss
                + (args.beta_rate * rate_loss)
                + (args.lambda_distill * distill_loss)
                + (args.lambda_importance * importance_loss)
            )
            total_loss.backward()
            optimizer.step()

            running_total += float(total_loss.item())
            running_recon += float(recon_loss.item())
            running_rate += float(rate_loss.item())
            running_distill += float(distill_details["distill_total"])
            running_importance += float(importance_loss.item())

            if step % 200 == 0:
                score_drop = (teacher_orig["score"] - teacher_recon["score"]).mean().item()
                print(
                    f"Epoch {epoch+1} Step {step}: "
                    f"total={total_loss.item():.4f} recon={recon_loss.item():.4f} "
                    f"distill={distill_details['distill_total']:.4f} "
                    f"rate={rate_loss.item():.4f} imp={importance_loss.item():.4f} "
                    f"teacher_drop={score_drop:.4f}"
                )

        denom = max(1, len(train_loader))
        avg_total = running_total / denom
        avg_recon = running_recon / denom
        avg_rate = running_rate / denom
        avg_distill = running_distill / denom
        avg_importance = running_importance / denom
        total_history.append(avg_total)

        val_teacher_drop = evaluate_teacher_drop(
            model=model,
            teacher=teacher,
            loader=val_loader,
            device=device,
            max_batches=val_eval_batches,
        )
        teacher_drop_history.append(val_teacher_drop)
        print(
            f"Epoch {epoch+1}/{epochs}: total={avg_total:.6f} recon={avg_recon:.6f} "
            f"distill={avg_distill:.6f} rate={avg_rate:.6f} importance={avg_importance:.6f} "
            f"val_teacher_drop={val_teacher_drop:.6f}"
        )

    repo_root = Path(__file__).resolve().parents[2]
    checkpoints_dir = repo_root / "data" / "results" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    model_info = {
        "model_class": type(model).__name__,
        "encoder_class": type(model.encoder).__name__,
        "decoder_class": type(model.decoder).__name__,
        "quantizer_class": type(model.quantizer).__name__,
        "importance_head_class": type(model.importance_head).__name__,
        "teacher_backend": teacher.backend,
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
    }

    run_id = args.run_id
    if run_id is None:
        run_id = make_run_id(stage="2_1", name=args.run_name, config=config)
    run_paths = default_run_paths(run_id)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "teacher_drop_history": teacher_drop_history,
            "loss_history": total_history,
            "config": config,
            "model_info": model_info,
        },
        run_paths.checkpoint_path,
    )
    write_manifest(
        run_paths.manifest_path,
        stage="stage2_1",
        run_id=run_id,
        command=" ".join(__import__("sys").argv),
        config=config,
        model_info=model_info,
        metrics={"val_teacher_drop_final": teacher_drop_history[-1] if teacher_drop_history else None},
        notes="Stage2.1 teacher-distilled adaptive run.",
    )
    print(f"Run dir: {run_paths.run_dir}")
    print(f"Run checkpoint: {run_paths.checkpoint_path}")
    print(f"Run manifest: {run_paths.manifest_path}")

    if args.save_name is None or args.save_name == "auto":
        save_name = f"{run_id}.pth"
    else:
        save_name = args.save_name
    ckpt_path = checkpoints_dir / save_name
    torch.save(
        {
            "model_state": model.state_dict(),
            "teacher_drop_history": teacher_drop_history,
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
                "teacher_drop_history": teacher_drop_history,
                "loss_history": total_history,
                "config": config,
                "model_info": model_info,
            },
            alias_path,
        )
        print(f"Saved alias checkpoint: {alias_path}")


if __name__ == "__main__":
    main()
