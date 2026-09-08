"""URCD-YOLO Training Script on UAV-PDD2023.

Trains the URCD-YOLO11s architecture on aerial pavement distress imagery
using Ultralytics trainer, custom loss calculation, and dataset.yaml.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.urcd_yolo.model import URCDYOLO


def parse_args() -> argparse.Namespace:
    # This function parses command-line arguments for URCD-YOLO training.
    # Input: CLI args -> Intermediate: argparse argument definitions -> Output: parsed Namespace.
    parser = argparse.ArgumentParser(description="Train URCD-YOLO on UAV PDD2023 dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="data/uav_pdd2023/dataset.yaml",
        help="Path to dataset YAML configuration",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="src/urcd_yolo/urcd_yolo11s.yaml",
        help="Path to URCD-YOLO model YAML specification",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cuda', 'cpu', or device ID)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/urcd_yolo",
        help="Directory to save training run logs & weights",
    )
    parser.add_argument("--name", type=str, default="exp", help="Run experiment name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 epoch with batch size 2 for plumbing and convergence smoke testing",
    )
    return parser.parse_args()


def main() -> None:
    # This function executes the end-to-end training pipeline for URCD-YOLO.
    # Input: None -> Intermediate: instantiates URCDYOLO, validates data path, executes .train() -> Output: None.
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found at '{data_path}'. Please run 'scripts/export_yolo.py' first."
        )

    model_cfg = Path(args.model_cfg)
    if not model_cfg.is_absolute():
        model_cfg = PROJECT_ROOT / model_cfg

    epochs = 1 if args.dry_run else args.epochs
    batch_size = 2 if args.dry_run else args.batch

    print("=" * 70)
    print("🚀 Initializing URCD-YOLO Training Pipeline")
    print(f"   Architecture Config: {model_cfg}")
    print(f"   Dataset Config:      {data_path}")
    print(f"   Device:              {args.device}")
    print(f"   Epochs:              {epochs} {'(Dry-run mode)' if args.dry_run else ''}")
    print(f"   Batch Size:          {batch_size}")
    print(f"   Image Size:          {args.imgsz}")
    print(f"   Project Runs Dir:    {args.project}/{args.name}")
    print("=" * 70)

    # Initialize URCD-YOLO model with custom module registration
    model = URCDYOLO(model=str(model_cfg))

    # Execute training via Ultralytics engine
    train_kwargs = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "workers": 2 if args.device == "cpu" else 4,
        "verbose": True,
    }
    if args.dry_run:
        train_kwargs["fraction"] = 0.002  # Train on a tiny subset (~9 images) for rapid verification

    _ = model.train(**train_kwargs)

    print("\n✅ URCD-YOLO Training run completed successfully!")
    print(f"   Results saved to: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
