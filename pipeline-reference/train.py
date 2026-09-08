"""
Stage 5 -- Train

Fine-tunes the detection transformer (config.model_checkpoint, default
facebook/detr-resnet-50) on UAV-PDD2023 via HF `Trainer`.

Requires network access to the Hugging Face Hub (to download the dataset and
pretrained weights) and, in practice, a GPU -- DETR-family models are slow
to fine-tune on CPU. Run from the project root:

    python train.py
    python train.py --no-augment
    python train.py --epochs 30 --batch-size 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

pipeline_dir = Path(__file__).resolve().parent
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import torch
from config import CFG, Config
from model import build_model_and_processor, build_param_groups
from transformers import Trainer, TrainingArguments

from data import (
    DetectionDataset,
    build_train_transform,
    make_collate_fn,
    prepare_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training(cfg: Config = CFG, use_augmentation: bool = True):
    splits, class_map = prepare_dataset(
        dataset_name=cfg.dataset_name,
        split=cfg.train_split,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        canonical_class_map=cfg.canonical_class_map,
    )
    id2label = {i: name for name, i in class_map.items()}
    label2id = class_map
    logger.info("Classes: %s", class_map)

    model, image_processor = build_model_and_processor(
        cfg.model_checkpoint,
        id2label,
        label2id,
        cfg.image_shortest_edge,
        cfg.image_longest_edge,
    )

    train_transform = build_train_transform() if use_augmentation else None
    train_dataset = DetectionDataset(splits["train"], class_map, transform=train_transform)
    eval_dataset = DetectionDataset(splits["validation"], class_map, transform=None)
    collate_fn = make_collate_fn(image_processor)

    param_groups = build_param_groups(model, cfg.learning_rate, cfg.backbone_learning_rate)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_train_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps if cfg.max_train_steps <= 0 else 1,
        eval_strategy="no" if cfg.max_train_steps > 0 else "epoch",
        save_strategy="no" if cfg.max_train_steps > 0 else "epoch",
        save_total_limit=cfg.save_total_limit,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,  # our Dataset isn't a datasets.Dataset; keep as-is regardless
        fp16=cfg.mixed_precision and torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=image_processor,
        optimizers=(optimizer, None),  # Trainer builds a default LR scheduler around this optimizer
    )

    trainer.train()

    trainer.save_model(cfg.output_dir)
    image_processor.save_pretrained(cfg.output_dir)
    logger.info("Saved fine-tuned model + image processor to %s", cfg.output_dir)

    return trainer, class_map


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a detection transformer on UAV-PDD2023.")
    parser.add_argument("--no-augment", action="store_true", help="disable train-time augmentation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="limit training to N steps for quick test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = CFG
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.max_steps is not None:
        cfg.max_train_steps = args.max_steps
    if args.batch_size is not None:
        cfg.per_device_train_batch_size = args.batch_size
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    run_training(cfg, use_augmentation=not args.no_augment)


if __name__ == "__main__":
    main()
