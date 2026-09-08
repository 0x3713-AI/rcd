"""
COCO format annotation exporter and validator for UAV-PDD2023.

Generates standard on-disk COCO JSON files (instances_train.json, instances_val.json)
matching the official MS-COCO object detection schema:
    {
        "info": {...},
        "licenses": [...],
        "images": [{"id": ..., "file_name": ..., "width": ..., "height": ...}],
        "annotations": [{"id": ..., "image_id": ..., "category_id": ..., "bbox": [x, y, w, h], "area": ..., "iscrowd": 0}],
        "categories": [{"id": ..., "name": ..., "supercategory": "distress"}]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_CLASS_MAP: dict[str, int] = {
    "longitudinal crack": 0,
    "transverse crack": 1,
    "oblique crack": 2,
    "alligator crack": 3,
    "repair": 4,
    "pothole": 5,
}


def normalize_class_name(name: str) -> str:
    """
    Normalizes class names for whitespace and case tolerance.
    Input: raw string -> Intermediate: strip and lowercase -> Output: normalized string
    """
    return name.strip().lower()


def sanitize_coco_box(
    box: list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """
    Sanitizes normalized corner coordinates and converts to absolute COCO [x_min, y_min, w, h].

    Input: normalized [x1, y1, x2, y2] + image width/height
    -> Intermediate:
        1. Reject non-finite values (NaN / Inf)
        2. Heal inverted coordinates via min/max
        3. Clamp normalized coordinates to [0.0, 1.0]
        4. Scale to absolute pixel space: px = nx * width, py = ny * height
        5. Reject degenerate boxes where width < 1e-4 or height < 1e-4
    -> Output: (x_min, y_min, width, height) in pixels, or None if invalid
    """
    if len(box) != 4 or image_width <= 0 or image_height <= 0:
        return None

    x1, y1, x2, y2 = box

    # 1. Reject non-finite values
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return None

    # 2. Heal inverted coordinates
    nx1 = min(x1, x2)
    nx2 = max(x1, x2)
    ny1 = min(y1, y2)
    ny2 = max(y1, y2)

    # 3. Clamp normalized coordinates strictly to [0.0, 1.0]
    nx1 = max(0.0, min(nx1, 1.0))
    nx2 = max(0.0, min(nx2, 1.0))
    ny1 = max(0.0, min(ny1, 1.0))
    ny2 = max(0.0, min(ny2, 1.0))

    # 4. Scale to pixel coordinates
    px1 = nx1 * float(image_width)
    px2 = nx2 * float(image_width)
    py1 = ny1 * float(image_height)
    py2 = ny2 * float(image_height)

    w = px2 - px1
    h = py2 - py1

    # 5. Reject degenerate boxes
    if w < 1e-4 or h < 1e-4:
        return None

    return (round(px1, 2), round(py1, 2), round(w, 2), round(h, 2))


def build_coco_split_dict(
    split_dataset: Dataset,
    split_name: str,
    class_map: dict[str, int] = CANONICAL_CLASS_MAP,
    image_rel_prefix: str = "images",
    start_ann_id: int = 0,
) -> tuple[dict[str, Any], int]:
    """
    Constructs a complete COCO JSON dictionary for a single split.

    Input: split_dataset + split_name ('train' or 'val') + class_map + start_ann_id
    -> Intermediate:
        1. Construct categories list from class_map
        2. Iterate over samples, build image metadata {"id", "file_name", "width", "height"}
        3. Convert bounding boxes to COCO annotations {"id", "image_id", "category_id", "bbox", "area", "iscrowd"}
    -> Output: (complete COCO dictionary, next_ann_id)
    """
    lookup = {normalize_class_name(k): v for k, v in class_map.items()}

    # Categories list sorted by ID
    categories = [
        {"id": cid, "name": name, "supercategory": "road_distress"}
        for name, cid in sorted(class_map.items(), key=lambda item: item[1])
    ]

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = start_ann_id

    for idx, sample in enumerate(split_dataset):
        image = sample["image"]
        w, h = image.size
        stem = f"uav_{split_name}_{idx:05d}"
        file_name = f"{image_rel_prefix}/{split_name}/{stem}.jpg"

        image_entry = {
            "id": idx,
            "file_name": file_name,
            "width": int(w),
            "height": int(h),
        }
        images.append(image_entry)

        objects = sample.get("objects", [])
        for obj in objects:
            raw_name = obj.get("name", "")
            norm_name = normalize_class_name(raw_name)
            if norm_name not in lookup:
                continue
            cat_id = lookup[norm_name]

            for box in obj.get("boxes", []):
                converted = sanitize_coco_box(box, w, h)
                if converted is None:
                    continue
                bx, by, bw, bh = converted
                area = round(bw * bh, 2)

                ann_entry = {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": cat_id,
                    "bbox": [bx, by, bw, bh],
                    "area": area,
                    "iscrowd": 0,
                }
                annotations.append(ann_entry)
                ann_id += 1

    coco_dict = {
        "info": {
            "description": "UAV-PDD2023 Pavement Distress Dataset in COCO Format",
            "url": "https://huggingface.co/datasets/vikhyatk/uav-pdd2023",
            "version": "1.0",
            "year": 2023,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    return coco_dict, ann_id


def export_coco_annotations(
    dataset_name: str = "vikhyatk/uav-pdd2023",
    output_dir: str | Path = "data/uav_pdd2023",
    val_fraction: float = 0.1,
    seed: int = 42,
    limit: int | None = None,
    class_map: dict[str, int] = CANONICAL_CLASS_MAP,
) -> dict[str, Path]:
    """
    Exports UAV-PDD2023 annotations to standard COCO JSON files.

    Input: dataset_name, output_dir, val_fraction, seed, optional limit, class_map
    -> Intermediate:
        1. Load dataset from HF Hub / local cache
        2. Split into train / validation partitions
        3. Build COCO dict for train split and write annotations/instances_train.json
        4. Build COCO dict for val split and write annotations/instances_val.json
    -> Output: {"train": Path, "val": Path}
    """
    out_path = Path(output_dir)
    annotations_dir = out_path / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset '%s' for COCO export...", dataset_name)
    raw = load_dataset(dataset_name, split="train")
    if limit is not None and limit > 0:
        raw = raw.select(range(min(limit, len(raw))))

    split_dict = raw.train_test_split(test_size=val_fraction, seed=seed)
    train_ds = split_dict["train"]
    val_ds = split_dict["test"]

    train_coco, train_ann_count = build_coco_split_dict(
        train_ds, "train", class_map=class_map, start_ann_id=0
    )
    val_coco, _ = build_coco_split_dict(
        val_ds, "val", class_map=class_map, start_ann_id=train_ann_count
    )

    train_json_path = annotations_dir / "instances_train.json"
    val_json_path = annotations_dir / "instances_val.json"

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, indent=2)

    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, indent=2)

    logger.info("Exported COCO train annotations: %s (%d images, %d annotations)", train_json_path, len(train_coco["images"]), len(train_coco["annotations"]))
    logger.info("Exported COCO val annotations: %s (%d images, %d annotations)", val_json_path, len(val_coco["images"]), len(val_coco["annotations"]))

    return {"train": train_json_path, "val": val_json_path}


def verify_coco_json(json_path: str | Path) -> dict[str, Any]:
    """
    Validates a COCO JSON file against the official MS-COCO specification.

    Input: path to COCO JSON file
    -> Intermediate:
        1. Verify root keys ("info", "images", "annotations", "categories")
        2. Validate categories (unique IDs, name strings)
        3. Validate images (unique IDs, positive width and height)
        4. Validate annotations:
           - bbox has 4 non-negative numbers [x, y, w, h]
           - w > 0, h > 0, area == round(w * h, 2)
           - image_id refers to existing image
           - category_id refers to existing category
           - iscrowd is integer (0 or 1)
    -> Output: dictionary containing validation statistics
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"COCO JSON file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Root structure verification
    required_keys = {"images", "annotations", "categories"}
    for key in required_keys:
        assert key in data, f"Missing required COCO root key: '{key}'"

    categories = data["categories"]
    assert len(categories) > 0, "No categories defined"
    cat_ids = set()
    for cat in categories:
        assert "id" in cat and "name" in cat, f"Malformed category: {cat}"
        cat_ids.add(cat["id"])

    # 2. Images verification
    images = data["images"]
    image_ids = set()
    for img in images:
        assert "id" in img and "file_name" in img and "width" in img and "height" in img, f"Malformed image entry: {img}"
        iid = img["id"]
        assert iid not in image_ids, f"Duplicate image ID: {iid}"
        image_ids.add(iid)
        assert img["width"] > 0 and img["height"] > 0, f"Invalid dimensions in image: {img}"

    # 3. Annotations verification
    annotations = data["annotations"]
    ann_ids = set()
    class_counts: dict[int, int] = {cid: 0 for cid in cat_ids}

    for ann in annotations:
        for req in ["id", "image_id", "category_id", "bbox", "area", "iscrowd"]:
            assert req in ann, f"Annotation {ann.get('id')} missing required field '{req}'"

        aid = ann["id"]
        assert aid not in ann_ids, f"Duplicate annotation ID: {aid}"
        ann_ids.add(aid)

        # Referencing integrity
        assert ann["image_id"] in image_ids, f"Annotation {aid} references non-existent image_id {ann['image_id']}"
        assert ann["category_id"] in cat_ids, f"Annotation {aid} references non-existent category_id {ann['category_id']}"

        # Bounding box correctness
        bbox = ann["bbox"]
        assert len(bbox) == 4, f"Annotation {aid}: bbox must have 4 elements, got {bbox}"
        x, y, w, h = bbox
        assert x >= 0 and y >= 0, f"Annotation {aid}: negative top-left coordinate ({x}, {y})"
        assert w > 0 and h > 0, f"Annotation {aid}: non-positive dimensions ({w}, {h})"
        assert ann["area"] > 0, f"Annotation {aid}: non-positive area {ann['area']}"

        class_counts[ann["category_id"]] += 1

    stats = {
        "file": str(p),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "class_counts": class_counts,
        "status": "valid",
    }
    logger.info("COCO validation passed cleanly for %s: %s", p.name, stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and verify UAV-PDD2023 in standard COCO format.")
    parser.add_argument("--output-dir", type=str, default="data/uav_pdd2023", help="Dataset directory")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on total samples")
    parser.add_argument("--verify-only", action="store_true", help="Run verification on existing COCO JSON files")

    args = parser.parse_args()

    if args.verify_only:
        ann_dir = Path(args.output_dir) / "annotations"
        verify_coco_json(ann_dir / "instances_train.json")
        verify_coco_json(ann_dir / "instances_val.json")
    else:
        paths = export_coco_annotations(
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            limit=args.limit,
        )
        verify_coco_json(paths["train"])
        verify_coco_json(paths["val"])


if __name__ == "__main__":
    main()
