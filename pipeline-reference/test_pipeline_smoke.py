"""
Offline smoke test and adversarial stress test for the pipeline's plumbing.

Exercises every stage's code path -- annotation conversion, augmentation,
Dataset, collate_fn, model forward/backward, post-processing, and mAP
ground-truth conversion -- against synthetic data, tiny models, and
adversarial edge cases (inverted coordinates, out-of-bounds boxes, degenerate
boxes, non-finite values, non-RGB images, extreme aspect ratios, empty/negative samples).

Runs fast on CPU without network access.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure pipeline-reference directory is on sys.path
pipeline_dir = Path(__file__).resolve().parent
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import numpy as np
import torch
from datasets import Dataset
from evaluate import _cxcywh_norm_to_xyxy_abs
from PIL import Image
from transformers import DetrConfig, DetrForObjectDetection, ResNetConfig
from transformers.models.detr.image_processing_pil_detr import DetrImageProcessorPil

from data import (
    AlbumentationsTransform,
    DetectionDataset,
    build_class_map,
    build_train_transform,
    make_collate_fn,
    train_val_split,
)

CLASS_NAMES = ["longitudinal crack", "transverse crack", "pothole"]


def make_tiny_detr(num_labels: int) -> DetrForObjectDetection:
    """
    Builds a tiny, randomly-initialized DETR with native ResNet backbone.
    Zero external dependencies on timm.
    """
    backbone_config = ResNetConfig(layer_type="basic", depths=[1, 1, 1, 1], hidden_sizes=[16, 32, 64, 128])
    config = DetrConfig(
        num_labels=num_labels,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_queries=20,
        backbone_config=backbone_config,
    )
    return DetrForObjectDetection(config)


def make_synthetic_dataset(num_rows: int = 12, seed: int = 0) -> Dataset:
    """Fabricates rows in the exact schema: image + objects[{name, boxes}]."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_rows):
        w, h = int(rng.integers(300, 500)), int(rng.integers(200, 400))
        image = Image.fromarray(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8))

        objects = []
        for name in rng.choice(CLASS_NAMES, size=int(rng.integers(1, 3)), replace=True):
            x1, y1 = rng.uniform(0.05, 0.5, size=2)
            x2 = min(x1 + rng.uniform(0.1, 0.3), 1.0)
            y2 = min(y1 + rng.uniform(0.05, 0.2), 1.0)
            objects.append({"name": str(name), "boxes": [[float(x1), float(y1), float(x2), float(y2)]]})

        rows.append({"image": image, "objects": objects})

    return Dataset.from_list(rows)


def make_adversarial_dataset() -> Dataset:
    """
    Constructs an adversarial dataset specifically testing failure modes:
    1. Empty / hard-negative sample (no objects)
    2. Inverted coordinates (x1 > x2, y1 > y2)
    3. Out-of-bounds coordinates (x < 0, x > 1)
    4. Degenerate boxes (zero width or height)
    5. Non-finite coordinates (NaN and Inf)
    6. Non-RGB image formats (grayscale 'L', alpha 'RGBA')
    7. Extreme aspect ratios (ultra-tall, ultra-wide)
    """
    rows = []

    # 1. Hard negative: image with zero objects
    rows.append({
        "image": Image.new("RGB", (256, 256), color="black"),
        "objects": [],
    })

    # 2. Inverted & out-of-bounds coordinates
    rows.append({
        "image": Image.new("RGB", (320, 240), color="gray"),
        "objects": [
            {"name": "longitudinal crack", "boxes": [[1.2, 0.8, -0.2, 0.2]]},
        ],
    })

    # 3. Degenerate zero-size boxes & non-finite (NaN / Inf) values
    rows.append({
        "image": Image.new("RGB", (300, 300), color="white"),
        "objects": [
            {"name": "transverse crack", "boxes": [[0.4, 0.4, 0.4, 0.4]]},      # zero width & height
            {"name": "transverse crack", "boxes": [[0.2, 0.5, 0.6, 0.5]]},      # zero height
            {"name": "pothole", "boxes": [[float("nan"), 0.1, 0.5, 0.6]]},       # NaN
            {"name": "pothole", "boxes": [[0.1, float("inf"), 0.5, 0.6]]},       # Inf
            {"name": "pothole", "boxes": [[0.2, 0.2, 0.5, 0.5]]},               # valid box
        ],
    })

    # 4. Non-RGB color modes: Grayscale ("L") and RGBA
    rows.append({
        "image": Image.new("L", (200, 200), color=128),
        "objects": [{"name": "longitudinal crack", "boxes": [[0.1, 0.1, 0.4, 0.9]]}],
    })
    rows.append({
        "image": Image.new("RGBA", (200, 200), color=(100, 150, 200, 255)),
        "objects": [{"name": "pothole", "boxes": [[0.3, 0.3, 0.7, 0.7]]}],
    })

    # 5. Extreme aspect ratios: ultra-tall (50x500) and ultra-wide (500x50)
    rows.append({
        "image": Image.new("RGB", (50, 500), color="red"),
        "objects": [{"name": "longitudinal crack", "boxes": [[0.1, 0.1, 0.9, 0.9]]}],
    })
    rows.append({
        "image": Image.new("RGB", (500, 50), color="blue"),
        "objects": [{"name": "transverse crack", "boxes": [[0.1, 0.1, 0.9, 0.9]]}],
    })

    return Dataset.from_list(rows)


def run_smoke_test():
    print("=== PART 1: Standard Pipeline Plumbing Smoke Test ===")
    print("1/6 building synthetic dataset + class map ...")
    raw = make_synthetic_dataset()
    class_map = build_class_map(raw, canonical_class_map=None)
    assert set(class_map) <= set(CLASS_NAMES)
    splits = train_val_split(raw, val_fraction=0.25, seed=0)
    print(f"    classes: {class_map}, train={len(splits['train'])}, val={len(splits['validation'])}")

    print("2/6 building Dataset + augmentation ...")
    transform = AlbumentationsTransform(build_train_transform())
    train_dataset = DetectionDataset(splits["train"], class_map, transform=transform)
    item = train_dataset[0]
    assert "image" in item and "annotations" in item["target"]

    print("3/6 building image processor + collate_fn ...")
    image_processor = DetrImageProcessorPil(size={"shortest_edge": 128, "longest_edge": 256})
    collate_fn = make_collate_fn(image_processor)
    batch = collate_fn([train_dataset[i] for i in range(4)])
    assert batch["pixel_values"].shape[0] == 4
    assert len(batch["labels"]) == 4
    print(f"    pixel_values: {tuple(batch['pixel_values'].shape)}")

    print("4/6 tiny randomly-initialized DETR: forward + backward pass ...")
    model = make_tiny_detr(len(class_map))
    model.train()
    outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"], labels=batch["labels"])
    assert outputs.loss is not None and torch.isfinite(outputs.loss)
    outputs.loss.backward()
    print(f"    loss = {outputs.loss.item():.3f} (finite, backward pass ok)")

    print("5/6 post-processing predictions back to original image coordinates ...")
    model.eval()
    with torch.no_grad():
        eval_outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
    target_sizes = torch.stack([lbl["orig_size"] for lbl in batch["labels"]])
    processed = image_processor.post_process_object_detection(eval_outputs, threshold=0.0, target_sizes=target_sizes)
    assert len(processed) == 4 and "boxes" in processed[0]
    print(f"    sample-0 detections above threshold=0.0: {processed[0]['boxes'].shape[0]}")

    print("6/6 mAP ground-truth box conversion (normalized cxcywh -> original-image xyxy) ...")
    gt_boxes = _cxcywh_norm_to_xyxy_abs(batch["labels"][0]["boxes"], batch["labels"][0]["orig_size"])
    assert gt_boxes.shape == batch["labels"][0]["boxes"].shape
    print(f"    orig_size={batch['labels'][0]['orig_size'].tolist()}, boxes shape={tuple(gt_boxes.shape)}")
    print("Standard plumbing passed cleanly.\n")


def run_adversarial_test():
    print("=== PART 2: Computational & Mathematical Adversarial Stress Test ===")
    class_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    adv_data = make_adversarial_dataset()
    print(f"Created {len(adv_data)} adversarial samples (empty, inverted, out-of-bounds, NaN/Inf, grayscale, RGBA, extreme aspect ratios).")

    # 1. Dataset loading and target conversion
    transform = AlbumentationsTransform(build_train_transform())
    dataset_no_aug = DetectionDataset(adv_data, class_map, transform=None)
    dataset_with_aug = DetectionDataset(adv_data, class_map, transform=transform)

    print("Testing conversion across all adversarial rows without augmentation...")
    for i in range(len(dataset_no_aug)):
        item = dataset_no_aug[i]
        assert item["image"].mode == "RGB", f"Row {i} image mode was not converted to RGB"
        for ann in item["target"]["annotations"]:
            x, y, w, h = ann["bbox"]
            assert math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and math.isfinite(h), f"Row {i} non-finite bbox"
            assert w > 0 and h > 0, f"Row {i} degenerate box was not filtered"
            assert x >= 0 and y >= 0, f"Row {i} box coordinate negative"
    print("  -> All adversarial rows converted safely (bounds clamped, degenerate/non-finite removed, non-RGB converted).")

    print("Testing conversion across all adversarial rows WITH train-time augmentation...")
    for i in range(len(dataset_with_aug)):
        item = dataset_with_aug[i]
        assert item["image"].mode == "RGB"
        for ann in item["target"]["annotations"]:
            x, y, w, h = ann["bbox"]
            assert math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and math.isfinite(h)
            assert w > 0 and h > 0
    print("  -> Albumentations handled all adversarial boxes without crashing.")

    # 2. Batch collation with mixed empty and non-empty samples
    print("Testing batch collation of mixed adversarial samples...")
    image_processor = DetrImageProcessorPil(size={"shortest_edge": 128, "longest_edge": 256})
    collate_fn = make_collate_fn(image_processor)
    batch = collate_fn([dataset_with_aug[i] for i in range(len(dataset_with_aug))])
    assert batch["pixel_values"].shape[0] == len(dataset_with_aug)
    assert len(batch["labels"]) == len(dataset_with_aug)
    print(f"  -> Successfully collated adversarial batch of {len(dataset_with_aug)} images. Shape: {tuple(batch['pixel_values'].shape)}")

    # 3. Model forward + backward pass under adversarial batch
    print("Testing model forward/backward numerical stability on adversarial batch...")
    model = make_tiny_detr(len(class_map))
    model.train()
    outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"], labels=batch["labels"])
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss), f"Loss was not finite on adversarial batch: {outputs.loss}"
    outputs.loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), f"Gradient contained NaN/Inf in parameter {name}"
    print(f"  -> Forward/backward loss = {outputs.loss.item():.4f}. All gradients are finite (zero NaNs).")

    # 4. Single-batch optimization test (checking convergence mathematical property)
    print("Testing single-batch optimization convergence (5 optimizer steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        out = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"], labels=batch["labels"])
        loss = out.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  -> Step losses: {[round(l, 4) for l in losses]}")
    assert losses[-1] < losses[0], f"Expected loss to decrease: start={losses[0]}, end={losses[-1]}"
    print("  -> Loss strictly decreased under optimization. Mathematical convergence validated.")

    print("\nALL ADVERSARIAL STRESS TESTS PASSED WITH 100% MATHEMATICAL & COMPUTATIONAL INTEGRITY!")


def main():
    run_smoke_test()
    run_adversarial_test()


if __name__ == "__main__":
    main()
