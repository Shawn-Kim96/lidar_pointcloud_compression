import argparse
import sys
import os
import time
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from train.trainer import Trainer
from dataset.semantickitti_loader import SemanticKittiDataset
from dataset.kitti_object_loader import KittiObjectRangeDataset
import models.compression
import models.backbones

def _safe_token(value):
    text = str(value).strip().replace(" ", "")
    text = text.replace("/", "-")
    return "".join(ch for ch in text if ch.isalnum() or ch in ("-", "_", "."))


def _parse_int_list(csv_text: str):
    return [int(v.strip()) for v in str(csv_text).split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="LiDAR Compression Training")
    
    # Model Config
    parser.add_argument("--backbone", type=str, default="darknet", help="resnet or darknet")
    parser.add_argument("--quantizer_mode", type=str, default="adaptive", choices=("adaptive", "uniform", "none"))
    parser.add_argument("--quant_bits", type=int, default=8, help="Uniform quantizer bits when quantizer_mode=uniform.")
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--blocks_per_stage", type=int, default=1)
    parser.add_argument("--position_branch", action="store_true", help="Enable xyz side-branch encoder.")
    parser.add_argument("--position_latent_channels", type=int, default=32)
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="deconv",
        choices=("deconv", "coord_conditioned", "skip_unet", "skip_coord_conditioned"),
        help="Decoder family. skip_* uses bilinear upsampling with encoder skip features.",
    )
    parser.add_argument("--implicit_hidden_channels", type=int, default=128)
    parser.add_argument("--pillar_side_branch", action="store_true", help="Enable raw-point pillar/BEV side stream.")
    parser.add_argument("--pillar_max_raw_points", type=int, default=150000)
    parser.add_argument("--pillar_size_x", type=float, default=0.24)
    parser.add_argument("--pillar_size_y", type=float, default=0.24)
    parser.add_argument("--pillar_pfn_hidden_channels", type=int, default=64)
    parser.add_argument("--pillar_pfn_out_channels", type=int, default=128)
    parser.add_argument("--pillar_bev_channels", type=str, default="128,128,192,256")
    parser.add_argument("--pillar_bev_blocks", type=str, default="2,2,2,3")
    parser.add_argument("--pillar_fpn_channels", type=int, default=128)
    
    # Teacher Config
    parser.add_argument(
        "--teacher_backend",
        type=str,
        default="proxy",
        help="proxy, pointpillars_zhulf, or openpcdet",
    )
    parser.add_argument("--teacher_proxy_ckpt", type=str, default=None, help="Optional proxy teacher checkpoint path.")
    parser.add_argument("--teacher_score_topk_ratio", type=float, default=0.01)
    
    # Training Config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="semantickitti",
        choices=("semantickitti", "kitti3dobject"),
        help="Training dataset backend.",
    )
    parser.add_argument(
        "--kitti_split",
        type=str,
        default="train",
        choices=("train", "val", "trainval", "test"),
        help="ImageSets split used when dataset_type=kitti3dobject.",
    )
    parser.add_argument(
        "--kitti_imageset_file",
        type=str,
        default="",
        help="Optional explicit ImageSets file path for KITTI object.",
    )
    parser.add_argument(
        "--kitti_roi_classes",
        type=str,
        default="Car,Pedestrian,Cyclist",
        help="Comma-separated classes to define ROI for KITTI object.",
    )
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_train_frames", type=int, default=0, help="If >0, train on first N frames for fast ablations.")
    parser.add_argument("--img_h", type=int, default=64)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--fov_up_deg", type=float, default=3.0)
    parser.add_argument("--fov_down_deg", type=float, default=-25.0)
    
    # Quantizer Config
    parser.add_argument("--roi_levels", type=int, default=256)
    parser.add_argument("--bg_levels", type=int, default=16)
    parser.add_argument(
        "--roi_target_mode",
        type=str,
        default="maxpool",
        choices=("nearest", "maxpool", "area"),
        help="How to downsample ROI mask for importance supervision.",
    )

    # Loss Weights
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_distill", type=float, default=0.5)
    parser.add_argument("--lambda_rate", type=float, default=0.1)
    parser.add_argument("--lambda_importance", type=float, default=0.5)
    parser.add_argument("--lambda_imp_separation", type=float, default=0.0)
    parser.add_argument("--imp_separation_margin", type=float, default=0.05)
    parser.add_argument(
        "--recon_loss_mode",
        type=str,
        default="mse",
        choices=("mse", "masked_channel_weighted"),
        help="Reconstruction loss type. masked_channel_weighted emphasizes valid xyz/range fidelity.",
    )
    parser.add_argument("--recon_range_weight", type=float, default=1.0)
    parser.add_argument("--recon_xyz_weight", type=float, default=1.0)
    parser.add_argument("--recon_remission_weight", type=float, default=1.0)
    parser.add_argument("--lambda_ray_consistency", type=float, default=0.0)

    parser.add_argument("--loss_recipe", type=str, default="legacy", choices=("legacy", "balanced_v1", "balanced_v2"))
    parser.add_argument(
        "--rate_loss_mode",
        type=str,
        default=None,
        choices=("global_mean", "normalized_global", "normalized_bg"),
        help="Override rate loss mode. If omitted, selected automatically by loss_recipe.",
    )
    parser.add_argument("--importance_loss_mode", type=str, default=None, choices=("bce", "weighted_bce"))
    parser.add_argument("--importance_pos_weight_mode", type=str, default="auto", choices=("auto", "fixed"))
    parser.add_argument("--importance_pos_weight", type=float, default=1.0)
    parser.add_argument("--importance_pos_weight_max", type=float, default=50.0)

    parser.add_argument("--distill_logit_loss", type=str, default="auto", choices=("auto", "kl", "bce", "mse"))
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--distill_feature_weight", type=float, default=1.0)
    parser.add_argument("--distill_logit_weight", type=float, default=1.0)
    parser.add_argument("--distill_align_mode", type=str, default="resize", choices=("resize", "adaptive_pool"))
    parser.add_argument(
        "--distill_align_hw",
        type=str,
        default="0,0",
        help="Target hw for adaptive_pool alignment (e.g., 16,32). 0,0 means auto(min dims).",
    )
    parser.add_argument(
        "--distill_feature_source",
        type=str,
        default="channel_mean",
        choices=("channel_mean", "energy_map", "none"),
        help="Feature distillation map type.",
    )
    parser.add_argument(
        "--distill_teacher_score_min",
        type=float,
        default=0.0,
        help="If >0, only distill from teacher samples with score >= threshold.",
    )
    parser.add_argument(
        "--distill_teacher_score_weight",
        action="store_true",
        help="Enable per-sample teacher score gate for distill weighting.",
    )

    # Importance head config
    parser.add_argument(
        "--importance_head_type",
        type=str,
        default="basic",
        choices=("basic", "multiscale", "pp_lite", "bifpn", "deformable_msa", "dynamic", "rangeformer", "frnet"),
    )
    parser.add_argument("--importance_hidden_channels", type=int, default=32)

    parser.add_argument(
        "--feature_fusion_variant",
        type=str,
        default="none",
        choices=("none", "bifpn", "deformable_msa", "dynamic", "rangeformer", "frnet"),
        help="Optional multiscale latent fusion block before quantization.",
    )
    parser.add_argument("--feature_fusion_hidden_channels", type=int, default=128)
    parser.add_argument("--decoder_post_refine_blocks", type=int, default=0)
    
    # Toggles
    parser.add_argument("--no_teacher", action="store_true", help="Disable teacher for Stage 1 training")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run identifier")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume/finetune from")

    return parser.parse_args()

def _print_experiment_summary(args, device, save_dir):
    if args.quantizer_mode == "uniform":
        stage_label = "0"
        mode_label = "uniform"
    elif args.quantizer_mode == "none":
        stage_label = "0.5"
        mode_label = "noquant"
    else:
        stage_label = "1" if args.no_teacher else "2"
        mode_label = "baseline" if args.no_teacher else "distill"
    teacher_backend = "none" if args.no_teacher else args.teacher_backend
    print("============================================================")
    print("[Experiment Metadata]")
    print(f"stage: {stage_label}")
    print(f"training_mode: {mode_label}")
    print(f"backbone: {args.backbone}")
    print(f"quantizer_mode: {args.quantizer_mode}")
    print(f"quant_bits: {args.quant_bits}")
    print(f"teacher_backend: {teacher_backend}")
    print(f"run_id: {args.run_id}")
    print(f"save_dir: {save_dir}")
    print(f"device: {device}")
    print(f"dataset_root: {args.data_root}")
    print(f"dataset_type: {args.dataset_type}")
    if args.dataset_type == "kitti3dobject":
        print(f"kitti_split: {args.kitti_split}")
        print(f"kitti_imageset_file: {args.kitti_imageset_file or 'auto(ImageSets/<split>.txt)'}")
        print(f"kitti_roi_classes: {args.kitti_roi_classes}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"lr: {args.lr}")
    print(f"roi_levels: {args.roi_levels}")
    print(f"bg_levels: {args.bg_levels}")
    print(f"roi_target_mode: {args.roi_target_mode}")
    print(f"latent_channels: {args.latent_channels}")
    print(f"base_channels: {args.base_channels}")
    print(f"blocks_per_stage: {args.blocks_per_stage}")
    print(f"position_branch: {args.position_branch}")
    print(f"position_latent_channels: {args.position_latent_channels}")
    print(f"decoder_type: {args.decoder_type}")
    print(f"implicit_hidden_channels: {args.implicit_hidden_channels}")
    print(f"pillar_side_branch: {args.pillar_side_branch}")
    print(f"pillar_max_raw_points: {args.pillar_max_raw_points}")
    print(f"pillar_size: ({args.pillar_size_x}, {args.pillar_size_y})")
    print(f"pillar_pfn_hidden_channels: {args.pillar_pfn_hidden_channels}")
    print(f"pillar_pfn_out_channels: {args.pillar_pfn_out_channels}")
    print(f"pillar_bev_channels: {args.pillar_bev_channels}")
    print(f"pillar_bev_blocks: {args.pillar_bev_blocks}")
    print(f"pillar_fpn_channels: {args.pillar_fpn_channels}")
    print(f"max_train_frames: {args.max_train_frames if args.max_train_frames > 0 else 'all'}")
    print(f"range_image_hw: {args.img_h}x{args.img_w}")
    print(f"range_image_fov: up={args.fov_up_deg}, down={args.fov_down_deg}")
    print(f"loss_recipe: {args.loss_recipe}")
    print(f"recon_loss_mode: {args.recon_loss_mode}")
    print(
        "recon_channel_weights: "
        f"range={args.recon_range_weight}, "
        f"xyz={args.recon_xyz_weight}, "
        f"remission={args.recon_remission_weight}"
    )
    print(f"lambda_ray_consistency: {args.lambda_ray_consistency}")
    print(f"rate_loss_mode: {args.rate_loss_mode or 'auto_by_recipe'}")
    print(f"importance_loss_mode: {args.importance_loss_mode or 'auto_by_recipe'}")
    print(f"importance_pos_weight_mode: {args.importance_pos_weight_mode}")
    print(f"importance_pos_weight: {args.importance_pos_weight}")
    print(f"importance_pos_weight_max: {args.importance_pos_weight_max}")
    print(f"lambda_imp_separation: {args.lambda_imp_separation}")
    print(f"imp_separation_margin: {args.imp_separation_margin}")
    print(f"distill_logit_loss: {args.distill_logit_loss}")
    print(f"distill_temperature: {args.distill_temperature}")
    print(f"distill_feature_weight: {args.distill_feature_weight}")
    print(f"distill_logit_weight: {args.distill_logit_weight}")
    print(f"distill_align_mode: {args.distill_align_mode}")
    print(f"distill_align_hw: {args.distill_align_hw}")
    print(f"distill_feature_source: {args.distill_feature_source}")
    print(f"distill_teacher_score_min: {args.distill_teacher_score_min}")
    print(f"distill_teacher_score_weight: {args.distill_teacher_score_weight}")
    print(f"importance_head_type: {args.importance_head_type}")
    print(f"importance_hidden_channels: {args.importance_hidden_channels}")
    print(f"feature_fusion_variant: {args.feature_fusion_variant}")
    print(f"feature_fusion_hidden_channels: {args.feature_fusion_hidden_channels}")
    print(f"decoder_post_refine_blocks: {args.decoder_post_refine_blocks}")
    print(f"teacher_proxy_ckpt: {args.teacher_proxy_ckpt if args.teacher_proxy_ckpt else 'none'}")
    print(f"teacher_score_topk_ratio: {args.teacher_score_topk_ratio}")
    print(f"started_at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(
        "loss_weights: "
        f"recon={args.lambda_recon}, "
        f"rate={args.lambda_rate}, "
        f"distill={args.lambda_distill}, "
        f"importance={args.lambda_importance}, "
        f"imp_separation={args.lambda_imp_separation}"
    )
    print(f"checkpoint: {args.checkpoint if args.checkpoint else 'none'}")
    print("============================================================")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataloader
    if args.dataset_type == "semantickitti":
        dataset_cfg = {
            "fov_up": args.fov_up_deg,
            "fov_down": args.fov_down_deg,
            "img_height": args.img_h,
            "img_width": args.img_w,
        }
        train_dataset = SemanticKittiDataset(
            root_dir=args.data_root,
            sequences=["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            config=dataset_cfg,
            return_roi_mask=True
        )
    elif args.dataset_type == "kitti3dobject":
        roi_classes = [c.strip() for c in args.kitti_roi_classes.split(",") if c.strip()]
        dataset_cfg = {
            "fov_up": args.fov_up_deg,
            "fov_down": args.fov_down_deg,
            "img_height": args.img_h,
            "img_width": args.img_w,
        }
        train_dataset = KittiObjectRangeDataset(
            root_dir=args.data_root,
            split=args.kitti_split,
            imageset_file=args.kitti_imageset_file or None,
            config=dataset_cfg,
            return_roi_mask=True,
            roi_classes=roi_classes,
            return_raw_points=args.pillar_side_branch,
            max_raw_points=args.pillar_max_raw_points,
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")
    if args.max_train_frames and args.max_train_frames > 0:
        max_frames = min(int(args.max_train_frames), len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(max_frames)))
        print(f"Using subset for training: {max_frames} frames")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Save Dir Logic
    if args.save_dir is None:
        date_str = time.strftime("%y%m%d")
        run_id = _safe_token(args.run_id or time.strftime("%H%M%S"))
        mode = "solo" if args.no_teacher else "distill"
        exp_tokens = [
            date_str,
            _safe_token(args.backbone),
            mode,
            f"lr{_safe_token(args.lr)}",
            f"bs{args.batch_size}",
            run_id,
        ]
        args.save_dir = os.path.join("data", "results", "experiments", "_".join(exp_tokens))
    
    print(f"Saving checkpoints to: {args.save_dir}")
    _print_experiment_summary(args, device, args.save_dir)

    if args.backbone not in ("darknet", "resnet"):
        raise ValueError(f"Unsupported backbone '{args.backbone}'. Use one of: darknet, resnet")

    backbone_config = {
        "name": args.backbone,
        "in_channels": 5,
    }
    decoder_stages = 5 if args.backbone == "darknet" else 4
    if args.backbone == "darknet":
        backbone_config["layers"] = (1, 1, 2, 2, 1)
        backbone_config["base_channels"] = args.base_channels
    else:
        # Encoder in autoencoder.py expects stage count/blocks, not "layers".
        backbone_config["latent_channels"] = args.latent_channels
        backbone_config["base_channels"] = args.base_channels
        backbone_config["num_stages"] = 4
        backbone_config["blocks_per_stage"] = args.blocks_per_stage

    # Config Construction
    pillar_bev_channels = _parse_int_list(args.pillar_bev_channels)
    pillar_bev_blocks = _parse_int_list(args.pillar_bev_blocks)
    config = {
        "model": {
            "name": "lidar_compression",
            "backbone_config": backbone_config,
            "quantizer_config": {
                "mode": args.quantizer_mode,
                "uniform_bits": args.quant_bits,
                "latent_channels": args.latent_channels,
                "roi_levels": args.roi_levels,
                "bg_levels": args.bg_levels,
                "use_ste": True
            },
            "decoder_config": {
                "latent_channels": args.latent_channels,
                "out_channels": 5,
                "base_channels": args.base_channels,
                "num_stages": decoder_stages,
                "post_refine_blocks": args.decoder_post_refine_blocks,
                "decoder_type": args.decoder_type,
                "coord_channels": 5,
                "implicit_hidden_channels": args.implicit_hidden_channels,
                "fov_up_deg": args.fov_up_deg,
                "fov_down_deg": args.fov_down_deg,
            },
            "position_branch_config": (
                {
                    "enabled": True,
                    "in_channels": 3,
                    "latent_channels": args.position_latent_channels,
                }
                if args.position_branch
                else None
            ),
            "head_config": (
                None
                if args.quantizer_mode in ("uniform", "none")
                else {
                    "hidden_channels": args.importance_hidden_channels,
                    "activation": "relu",
                    "head_type": args.importance_head_type,
                }
            ),
            "feature_fusion_config": (
                None
                if args.feature_fusion_variant == "none"
                else {
                    "enabled": True,
                    "variant": args.feature_fusion_variant,
                    "hidden_channels": args.feature_fusion_hidden_channels,
                    "activation": "silu",
                    "norm": "group",
                }
            ),
            "pillar_side_config": (
                None
                if not args.pillar_side_branch
                else {
                    "enabled": True,
                    "point_cloud_range": [0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
                    "pillar_size": [args.pillar_size_x, args.pillar_size_y],
                    "pfn_hidden_channels": args.pillar_pfn_hidden_channels,
                    "pfn_out_channels": args.pillar_pfn_out_channels,
                    "bev_channels": pillar_bev_channels,
                    "bev_blocks": pillar_bev_blocks,
                    "fpn_channels": args.pillar_fpn_channels,
                    "max_raw_points": args.pillar_max_raw_points,
                    "norm": "batch",
                    "activation": "silu",
                }
            ),
        },
        "teacher": {
            "enabled": not args.no_teacher,
            "config": {
                "backend": args.teacher_backend,
                "device": str(device),
                "proxy_ckpt": args.teacher_proxy_ckpt,
                "score_topk_ratio": args.teacher_score_topk_ratio,
            }
        },
        "data": {
            "img_height": args.img_h,
            "img_width": args.img_w,
            "fov_up_deg": args.fov_up_deg,
            "fov_down_deg": args.fov_down_deg,
        },
        "train": {
            "lr": args.lr,
            "weight_decay": 1e-4,
            "noise_std": 0.01
        },
        "loss": {
            "w_recon": args.lambda_recon,
            "w_distill": args.lambda_distill,
            "w_rate": args.lambda_rate,
            "w_importance": args.lambda_importance,
            "w_imp_separation": args.lambda_imp_separation,
            "recon_loss_mode": args.recon_loss_mode,
            "recon_range_weight": args.recon_range_weight,
            "recon_xyz_weight": args.recon_xyz_weight,
            "recon_remission_weight": args.recon_remission_weight,
            "w_ray_consistency": args.lambda_ray_consistency,
            "recipe": args.loss_recipe,
            "rate_loss_mode": args.rate_loss_mode,
            "importance_loss_mode": args.importance_loss_mode,
            "importance_pos_weight_mode": args.importance_pos_weight_mode,
            "importance_pos_weight": args.importance_pos_weight,
            "importance_pos_weight_max": args.importance_pos_weight_max,
            "imp_separation_margin": args.imp_separation_margin,
            "distill_logit_loss": args.distill_logit_loss,
            "distill_temperature": args.distill_temperature,
            "distill_feature_weight": args.distill_feature_weight,
            "distill_logit_weight": args.distill_logit_weight,
            "distill_align_mode": args.distill_align_mode,
            "distill_align_hw": args.distill_align_hw,
            "distill_feature_source": args.distill_feature_source,
            "distill_teacher_score_min": args.distill_teacher_score_min,
            "distill_teacher_score_weight": args.distill_teacher_score_weight,
        },
        "supervision": {
            "type": "roi",
            "roi_target_mode": args.roi_target_mode,
        }
    }
    
    # 3. Training
    trainer = Trainer(
        config=config,
        device=device,
        train_loader=train_loader
    )
    
    # Load Checkpoint if provided
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}...")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            # Handle both state_dict or full checkpoint dict
            state_dict = checkpoint if not "model_state_dict" in checkpoint else checkpoint["model_state_dict"]
            try:
                trainer.model.load_state_dict(state_dict, strict=False)
                print("Checkpoint loaded successfully.")
            except RuntimeError as e:
                print(f"Warning: Error loading state_dict: {e}")
        else:
            print(f"Warning: Checkpoint file {args.checkpoint} not found. Training from scratch.")
    
    print("Starting Training...")
    trainer.run(args.epochs, args.save_dir)

if __name__ == "__main__":
    main()
