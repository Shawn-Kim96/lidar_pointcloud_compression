import argparse
import sys
import os
import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from train.trainer import Trainer
from dataset.semantickitti_loader import SemanticKittiDataset
import models.compression
import models.backbones

def parse_args():
    parser = argparse.ArgumentParser(description="LiDAR Compression Training")
    
    # Model Config
    parser.add_argument("--backbone", type=str, default="darknet", help="resnet or darknet")
    parser.add_argument("--quant_bits", type=int, default=8)
    
    # Teacher Config
    parser.add_argument("--teacher_backend", type=str, default="openpcdet", help="proxy or openpcdet")
    
    # Training Config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    
    # Quantizer Config
    parser.add_argument("--roi_levels", type=int, default=256)
    parser.add_argument("--bg_levels", type=int, default=16)

    # Loss Weights
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_distill", type=float, default=0.5)
    parser.add_argument("--lambda_rate", type=float, default=0.1)
    parser.add_argument("--lambda_importance", type=float, default=0.5)
    
    # Toggles
    parser.add_argument("--no_teacher", action="store_true", help="Disable teacher for Stage 1 training")
    parser.add_argument("--no_teacher", action="store_true", help="Disable teacher for Stage 1 training")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run identifier")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume/finetune from")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataloader
    train_dataset = SemanticKittiDataset(
        root_dir=args.data_root,
        sequences=["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
        return_roi_mask=True
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Save Dir Logic
    if args.save_dir is None:
        import time
        date_str = time.strftime("%Y%m%d")
        run_id = args.run_id or time.strftime("%H%M%S")
        exp_name = f"stage1_{args.backbone}" if args.no_teacher else f"stage2_{args.backbone}_distill"
        args.save_dir = os.path.join("data", "results", "experiments", f"{date_str}_{exp_name}_{run_id}")
    
    print(f"Saving checkpoints to: {args.save_dir}")

    # Config Construction
    config = {
        "model": {
            "name": "lidar_compression",
            "backbone_config": {
                "name": args.backbone, 
                "in_channels": 5,
                "layers": (1, 1, 2, 2, 1) if args.backbone == "darknet" else (1,1,1,1)
            },
            "quantizer_config": {
                "roi_levels": args.roi_levels,
                "bg_levels": args.bg_levels,
                "use_ste": True
            },
            "decoder_config": {
                "latent_channels": 64,
                "out_channels": 5,
                "num_stages": 5 if args.backbone == "darknet" else 4
            },
            "head_config": {
                "hidden_channels": 32,
                "activation": "relu"
            }
        },
        "teacher": {
            "enabled": not args.no_teacher,
            "config": {
                "backend": args.teacher_backend,
                "device": str(device)
            }
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
            "w_importance": args.lambda_importance
        },
        "supervision": {
            "type": "roi" 
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
