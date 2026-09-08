"""
Consolidated data processing, transforms, and dataset loading.

Pipeline flow:
    Raw dataset row -> COCO target conversion -> Pure-PIL augmentation -> PyTorch Dataset -> DETR Collation
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Dataset Loading & Partitioning
# ==============================================================================

def load_raw_dataset(dataset_name: str, split: str = "train") -> Dataset:
    """
    Loads raw split from Hugging Face Hub.
    Input: dataset_name string -> loads remote/cached parquet -> Output: datasets.Dataset.
    """
    return load_dataset(dataset_name, split=split)


def discover_class_names(dataset: Dataset) -> list[str]:
    """
    Collects distinct class names in deterministic alphabetical order.
    Input: dataset -> walks objects.name rows -> Output: sorted list[str].
    """
    names = set()
    for row in dataset["objects"]:
        for obj in row:
            names.add(obj["name"])
    return sorted(names)


def build_class_map(dataset: Dataset, canonical_class_map: dict[str, int] | None = None) -> dict[str, int]:
    """
    Builds a reproducible class-name-to-integer mapping.
    Input: dataset + optional canonical dict -> validates keys match -> Output: {class_name: id}.
    """
    discovered = discover_class_names(dataset)

    if canonical_class_map is not None:
        canon_lower = {k.strip().lower(): v for k, v in canonical_class_map.items()}
        disc_lower = {name.strip().lower(): name for name in discovered}
        if set(canon_lower) == set(disc_lower):
            return {name: canon_lower[name.strip().lower()] for name in discovered}

        logger.warning(
            "canonical_class_map keys %s != discovered %s; falling back to sorted map.",
            sorted(canonical_class_map),
            discovered,
        )

    return {name: i for i, name in enumerate(discovered)}


def train_val_split(dataset: Dataset, val_fraction: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    Splits a single dataset into deterministic train and validation partitions.
    Input: Dataset + val fraction + seed -> train_test_split -> Output: DatasetDict(train, validation).
    """
    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"])


def prepare_dataset(
    dataset_name: str,
    split: str,
    val_fraction: float,
    seed: int,
    canonical_class_map: dict[str, int] | None = None,
) -> tuple[DatasetDict, dict[str, int]]:
    """
    End-to-end data preparation wrapper.
    Input: config args -> loads, builds class map, splits -> Output: (DatasetDict, class_map).
    """
    raw = load_raw_dataset(dataset_name, split)
    class_map = build_class_map(raw, canonical_class_map)
    splits = train_val_split(raw, val_fraction, seed)
    return splits, class_map


# ==============================================================================
# 2. Coordinate Conversion & Bounding Box Sanitization
# ==============================================================================

def objects_to_coco_annotations(
    objects: list[dict[str, Any]] | None,
    image_width: int,
    image_height: int,
    class_map: dict[str, int],
    start_ann_id: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """
    Converts normalized [x1, y1, x2, y2] boxes to absolute COCO [x_min, y_min, w, h] format.

    Flow:
        Input: raw objects list with normalized coords in [0, 1]
        Intermediate:
            - Discard non-finite values (NaN / Inf)
            - Handle inverted coords via min/max: nx1 = min(x1, x2), nx2 = max(x1, x2)
            - Clamp normalized coords to [0.0, 1.0]
            - Scale to absolute pixels: px = nx * width, py = ny * height
            - Filter degenerate zero-size boxes: width <= 0 or height <= 0
        Output: list of COCO annotation dicts [{"id", "bbox": [x, y, w, h], "category_id", "area", "iscrowd"}]
    """
    annotations: list[dict[str, Any]] = []
    ann_id = start_ann_id

    if not objects or image_width <= 0 or image_height <= 0:
        return annotations, ann_id

    for obj in objects:
        if "name" not in obj or obj["name"] not in class_map:
            continue
        category_id = class_map[obj["name"]]

        boxes = obj.get("boxes", [])
        if not boxes:
            continue

        for box in boxes:
            if len(box) != 4:
                continue

            raw_x1, raw_y1, raw_x2, raw_y2 = box

            # 1. Filter out NaN / Inf values
            if not (
                math.isfinite(raw_x1)
                and math.isfinite(raw_y1)
                and math.isfinite(raw_x2)
                and math.isfinite(raw_y2)
            ):
                continue

            # 2. Normalize inverted coordinates (x1 > x2 or y1 > y2)
            nx1 = min(raw_x1, raw_x2)
            nx2 = max(raw_x1, raw_x2)
            ny1 = min(raw_y1, raw_y2)
            ny2 = max(raw_y1, raw_y2)

            # 3. Clamp normalized coordinates to valid [0.0, 1.0] range
            nx1 = max(0.0, min(nx1, 1.0))
            nx2 = max(0.0, min(nx2, 1.0))
            ny1 = max(0.0, min(ny1, 1.0))
            ny2 = max(0.0, min(ny2, 1.0))

            # 4. Scale to absolute pixel space
            px1 = nx1 * float(image_width)
            px2 = nx2 * float(image_width)
            py1 = ny1 * float(image_height)
            py2 = ny2 * float(image_height)

            width = px2 - px1
            height = py2 - py1

            # 5. Filter degenerate / zero-area boxes
            if width < 1e-4 or height < 1e-4:
                continue

            annotations.append({
                "id": ann_id,
                "bbox": [px1, py1, width, height],
                "category_id": category_id,
                "area": width * height,
                "iscrowd": 0,
            })
            ann_id += 1

    return annotations, ann_id


def sample_to_coco_target(sample: dict[str, Any], image_id: int, class_map: dict[str, int]) -> dict[str, Any]:
    """
    Builds the complete target dict for a single dataset sample.
    Input: sample row with 'image' and 'objects' -> converts bboxes -> Output: {"image_id", "annotations"}.
    """
    image = sample["image"]
    width, height = image.size
    annotations, _ = objects_to_coco_annotations(sample.get("objects", []), width, height, class_map)
    return {"image_id": image_id, "annotations": annotations}


# ==============================================================================
# 3. Lean Pure-PIL Augmentation
# ==============================================================================

class LeanTransform:
    """
    Lightweight, dependency-free train-time data augmentation using pure PIL and elementary math.
    Zero dependency on albumentations, opencv, or scipy.

    Design rationale:
        Longitudinal and transverse cracks are orientation-sensitive.
        Rotations (90/270 deg) are strictly avoided; only flips and photometric jitter are applied.

    Flow:
        Input: (PIL.Image, target_dict)
        Intermediate:
            - Clamp boxes to image bounds
            - Horizontal flip (p=0.5): flips pixels, updates bbox x' = width - (x + w)
            - Vertical flip (p=0.5): flips pixels, updates bbox y' = height - (y + h)
            - Random brightness jitter (p=0.3): PIL.ImageEnhance.Brightness
            - Random contrast jitter (p=0.3): PIL.ImageEnhance.Contrast
        Output: (transformed_PIL_Image, transformed_target_dict)
    """

    def __init__(
        self,
        hflip_p: float | Any = 0.5,
        vflip_p: float = 0.5,
        brightness_p: float = 0.3,
        contrast_p: float = 0.3,
    ) -> None:
        if isinstance(hflip_p, LeanTransform):
            self.hflip_p = hflip_p.hflip_p
            self.vflip_p = hflip_p.vflip_p
            self.brightness_p = hflip_p.brightness_p
            self.contrast_p = hflip_p.contrast_p
        elif not isinstance(hflip_p, (int, float)):
            self.hflip_p = 0.5
            self.vflip_p = 0.5
            self.brightness_p = 0.3
            self.contrast_p = 0.3
        else:
            self.hflip_p = float(hflip_p)
            self.vflip_p = float(vflip_p)
            self.brightness_p = float(brightness_p)
            self.contrast_p = float(contrast_p)

    def __call__(self, image: Image.Image, target: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
        annotations = target.get("annotations", [])
        image = image.convert("RGB")
        img_w, img_h = image.size

        # Copy annotations defensively to avoid mutating parent dataset
        new_annotations = []
        for i, ann in enumerate(annotations):
            x, y, w, h = ann["bbox"]
            x = max(0.0, min(float(x), float(img_w)))
            y = max(0.0, min(float(y), float(img_h)))
            w = max(0.0, min(float(w), float(img_w) - x))
            h = max(0.0, min(float(h), float(img_h) - y))
            if w > 1e-4 and h > 1e-4:
                new_annotations.append({
                    "id": i,
                    "bbox": [x, y, w, h],
                    "category_id": ann["category_id"],
                    "area": w * h,
                    "iscrowd": ann.get("iscrowd", 0),
                })

        # 1. Horizontal flip
        if random.random() < self.hflip_p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            for ann in new_annotations:
                x, y, w, h = ann["bbox"]
                new_x = max(0.0, float(img_w) - (x + w))
                ann["bbox"] = [new_x, y, w, h]

        # 2. Vertical flip
        if random.random() < self.vflip_p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            for ann in new_annotations:
                x, y, w, h = ann["bbox"]
                new_y = max(0.0, float(img_h) - (y + h))
                ann["bbox"] = [x, new_y, w, h]

        # 3. Brightness adjustment
        if random.random() < self.brightness_p:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)

        # 4. Contrast adjustment
        if random.random() < self.contrast_p:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Contrast(image).enhance(factor)

        new_target = {
            "image_id": target.get("image_id", 0),
            "annotations": new_annotations,
        }
        return image, new_target


def build_train_transform(*args: Any, **kwargs: Any) -> LeanTransform:
    """Factory creating the default training augmentation pipeline."""
    return LeanTransform()


# Backward-compatible alias
AlbumentationsTransform = LeanTransform


# ==============================================================================
# 4. PyTorch Dataset & Batch Collation
# ==============================================================================

class DetectionDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for object detection.

    Flow:
        Input: index idx -> fetches raw sample from underlying dataset
        Intermediate:
            - Converts image to standard 3-channel RGB (handles grayscale/RGBA/palette)
            - Converts bounding boxes to COCO target dict via sample_to_coco_target
            - Applies optional transform(image, target)
        Output: {"image": PIL.Image, "target": {"image_id": int, "annotations": list[dict]}}
    """

    def __init__(
        self,
        dataset: Any,
        class_map: dict[str, int],
        transform: Callable[[Image.Image, dict[str, Any]], tuple[Image.Image, dict[str, Any]]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.class_map = class_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]
        raw_image = sample["image"]
        image = raw_image.convert("RGB") if isinstance(raw_image, Image.Image) else Image.fromarray(raw_image).convert("RGB")
        target = sample_to_coco_target(sample, image_id=idx, class_map=self.class_map)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return {"image": image, "target": target}


def make_collate_fn(image_processor: Any) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """
    Creates a batch collator function for DataLoader or HF Trainer.

    Flow:
        Input: list of sample dicts [{"image": PIL.Image, "target": dict}, ...]
        Intermediate: extracts images and targets, invokes image_processor(..., return_tensors='pt')
        Output: batch feature dict with 'pixel_values', 'pixel_mask', and 'labels'.
    """
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        targets = [item["target"] for item in batch]
        encoded = image_processor(images=images, annotations=targets, return_tensors="pt")
        return encoded

    return collate_fn
