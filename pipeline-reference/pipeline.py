"""
Single entry point chaining the stages together.

    python pipeline.py prepare
    python pipeline.py train [--no-augment] [--epochs 30] [--batch-size 2]
    python pipeline.py evaluate --checkpoint runs/uav-pdd-detr
    python pipeline.py infer --checkpoint runs/uav-pdd-detr --image road.jpg --output pred.jpg

Every stage also runs standalone via its own file (train.py, evaluate.py,
inference.py) -- this wrapper is just a convenience/scripting entry point,
not a hidden extra layer of logic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

pipeline_dir = Path(__file__).resolve().parent
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

from config import CFG

from data import prepare_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cmd_prepare(args):
    splits, class_map = prepare_dataset(
        dataset_name=CFG.dataset_name,
        split=CFG.train_split,
        val_fraction=CFG.val_fraction,
        seed=CFG.seed,
        canonical_class_map=CFG.canonical_class_map,
    )
    logger.info("Classes (%d): %s", len(class_map), class_map)
    logger.info("Train rows: %d | Validation rows: %d", len(splits["train"]), len(splits["validation"]))


def cmd_train(args):
    from train import run_training

    if args.epochs is not None:
        CFG.num_train_epochs = args.epochs
    if args.max_steps is not None:
        CFG.max_train_steps = args.max_steps
    if args.batch_size is not None:
        CFG.per_device_train_batch_size = args.batch_size
    if args.output_dir is not None:
        CFG.output_dir = args.output_dir
    run_training(CFG, use_augmentation=not args.no_augment)


def cmd_evaluate(args):
    from evaluate import evaluate

    evaluate(args.checkpoint, CFG, score_threshold=args.score_threshold)


def cmd_infer(args):
    from inference import run_inference_on_path

    run_inference_on_path(args.checkpoint, args.image, args.output, args.score_threshold)


def main():
    parser = argparse.ArgumentParser(description="UAV-PDD2023 detection transformer pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="load the dataset and report class map / split sizes")

    p_train = sub.add_parser("train", help="fine-tune the detection transformer")
    p_train.add_argument("--no-augment", action="store_true")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--max-steps", type=int, default=None, help="limit training to N steps for quick test")
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--output-dir", type=str, default=None)

    p_eval = sub.add_parser("evaluate", help="compute mAP on the validation split")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--score-threshold", type=float, default=None)

    p_infer = sub.add_parser("infer", help="run inference on a single image")
    p_infer.add_argument("--checkpoint", type=str, required=True)
    p_infer.add_argument("--image", type=str, required=True)
    p_infer.add_argument("--output", type=str, default="prediction.jpg")
    p_infer.add_argument("--score-threshold", type=float, default=0.5)

    args = parser.parse_args()
    dispatch = {"prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate, "infer": cmd_infer}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
