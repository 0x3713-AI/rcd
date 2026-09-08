"""
Ultralytics YOLO dataset exporter for the UAV-PDD2023 dataset.

Converts Hugging Face parquet dataset (vikhyatk/uav-pdd2023) into the standard
Ultralytics YOLO disk layout:
    data/uav_pdd2023/
    ├── dataset.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
"""

from __future__ import annotations

import argparse
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical class mapping matching UAV-PDD2023 benchmark literature
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
    Normalizes a distress class name for case-insensitive and whitespace-tolerant matching.
    Input: raw string name -> Intermediate: strip whitespace and convert to lowercase -> Output: normalized string
    """
    return name.strip().lower()


def sanitize_and_convert_box(
    box: list[float] | tuple[float, ...],
    class_id: int,
) -> tuple[int, float, float, float, float] | None:
    """
    Sanitizes raw corner coordinates and converts them to normalized YOLO center format.

    This function enforces boundary clamping, inverted coordinate healing, non-finite rejection,
    and degenerate box filtering.
    Input: raw [x1, y1, x2, y2] normalized corner coordinates + class_id
    -> Intermediate:
        1. Reject non-finite values (NaN / Inf)
        2. Heal inverted coordinates: nx1 = min(x1, x2), nx2 = max(x1, x2), ny1 = min(y1, y2), ny2 = max(y1, y2)
        3. Clamp coordinates strictly to [0.0, 1.0]
        4. Compute box width, height, and center: w = nx2 - nx1, h = ny2 - ny1, xc = nx1 + w/2, yc = ny1 + h/2
        5. Reject degenerate boxes where width < 1e-5 or height < 1e-5
    -> Output: (class_id, xc, yc, w, h) tuple, or None if invalid
    """
    if len(box) != 4:
        return None

    x1, y1, x2, y2 = box

    # 1. Filter out non-finite values (NaN / Inf)
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return None

    # 2. Heal inverted coordinates (e.g. x1 > x2)
    nx1 = min(x1, x2)
    nx2 = max(x1, x2)
    ny1 = min(y1, y2)
    ny2 = max(y1, y2)

    # 3. Clamp normalized coordinates strictly to [0.0, 1.0]
    nx1 = max(0.0, min(nx1, 1.0))
    nx2 = max(0.0, min(nx2, 1.0))
    ny1 = max(0.0, min(ny1, 1.0))
    ny2 = max(0.0, min(ny2, 1.0))

    # 4. Compute dimensions
    w = nx2 - nx1
    h = ny2 - ny1

    # 5. Filter degenerate / zero-area boxes
    if w < 1e-5 or h < 1e-5:
        return None

    xc = nx1 + (w / 2.0)
    yc = ny1 + (h / 2.0)

    return (class_id, xc, yc, w, h)


def objects_to_yolo_lines(
    objects: list[dict[str, Any]] | None,
    class_map: dict[str, int] = CANONICAL_CLASS_MAP,
) -> list[str]:
    """
    Extracts distress boxes from raw objects and formats them into YOLO label lines.

    This function matches class names case-insensitively and applies coordinate conversion.
    Input: raw objects list [{'name': ..., 'boxes': [...]}] + class_map
    -> Intermediate:
        1. Resolve class name against normalized class_map keys
        2. For each box, call sanitize_and_convert_box
        3. Format valid boxes as '<class_id> <xc:.6f> <yc:.6f> <w:.6f> <h:.6f>'
    -> Output: list of formatted string lines for .txt label file
    """
    lines: list[str] = []
    if not objects:
        return lines

    # Map normalized names to class id
    lookup = {normalize_class_name(k): v for k, v in class_map.items()}

    for obj in objects:
        raw_name = obj.get("name", "")
        norm_name = normalize_class_name(raw_name)
        if norm_name not in lookup:
            logger.warning("Encountered unrecognized class '%s'; skipping object.", raw_name)
            continue

        class_id = lookup[norm_name]
        boxes = obj.get("boxes", [])

        for box in boxes:
            converted = sanitize_and_convert_box(box, class_id)
            if converted is None:
                continue
            cid, xc, yc, w, h = converted
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return lines


def write_dataset_yaml(
    output_dir: Path,
    class_map: dict[str, int] = CANONICAL_CLASS_MAP,
) -> Path:
    """
    Generates the Ultralytics dataset.yaml configuration file.

    This function creates the manifest specifying image split locations and class names.
    Input: output_dir Path + class_map
    -> Intermediate:
        1. Invert class_map to {id: name}
        2. Construct dictionary with dataset root path, train/val image paths, and class dictionary
        3. Serialize dictionary to YAML format
    -> Output: Path to generated dataset.yaml
    """
    yaml_path = output_dir / "dataset.yaml"

    # Invert class map to index: name
    id_to_name = {v: k for k, v in sorted(class_map.items(), key=lambda item: item[1])}

    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": id_to_name,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    logger.info("Generated dataset YAML: %s", yaml_path)
    return yaml_path


def _export_single_sample(
    args: tuple[int, dict[str, Any], Path, Path, str, dict[str, int]],
) -> int:
    """
    Helper to save an image and its corresponding YOLO label file.
    Input: tuple of (index, sample, images_dir, labels_dir, split_name, class_map)
    -> Intermediate: convert PIL image to RGB, save JPEG (quality=95), format YOLO label text
    -> Output: number of bounding boxes written
    """
    idx, sample, images_dir, labels_dir, split_name, class_map = args
    stem = f"uav_{split_name}_{idx:05d}"
    img_path = images_dir / f"{stem}.jpg"
    lbl_path = labels_dir / f"{stem}.txt"

    # Save image
    img: Image.Image = sample["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(img_path, format="JPEG", quality=95)

    # Save labels (empty file if 0 boxes, acting as hard negative)
    lines = objects_to_yolo_lines(sample.get("objects", []), class_map)
    content = "\n".join(lines) + ("\n" if lines else "")
    lbl_path.write_text(content, encoding="utf-8")

    return len(lines)


def export_split(
    split_dataset: Dataset,
    split_name: str,
    output_dir: Path,
    class_map: dict[str, int] = CANONICAL_CLASS_MAP,
    num_workers: int = 4,
) -> tuple[int, int]:
    """
    Exports a single split (train or val) to disk.

    This function creates image and label files concurrently for efficiency.
    Input: split_dataset + split_name ('train' or 'val') + output_dir + class_map + num_workers
    -> Intermediate:
        1. Ensure output_dir/images/{split_name} and output_dir/labels/{split_name} exist
        2. Build parameter tuples for each sample
        3. Concurrently write JPEG images and .txt label files using ThreadPoolExecutor
    -> Output: (total_images_written, total_boxes_written)
    """
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(split_dataset)
    logger.info("Exporting %s split (%d samples)...", split_name, total_samples)

    tasks = [
        (i, split_dataset[i], images_dir, labels_dir, split_name, class_map)
        for i in range(total_samples)
    ]

    total_boxes = 0
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for count in pool.map(_export_single_sample, tasks):
                total_boxes += count
    else:
        for task in tasks:
            total_boxes += _export_single_sample(task)

    logger.info(
        "Finished %s split: %d images, %d boxes saved.",
        split_name,
        total_samples,
        total_boxes,
    )
    return total_samples, total_boxes


def export_yolo_dataset(
    dataset_name: str = "vikhyatk/uav-pdd2023",
    output_dir: str | Path = "data/uav_pdd2023",
    val_fraction: float = 0.1,
    seed: int = 42,
    limit: int | None = None,
    num_workers: int = 4,
    class_map: dict[str, int] = CANONICAL_CLASS_MAP,
) -> dict[str, Any]:
    """
    Full end-to-end pipeline to export the UAV-PDD2023 dataset to YOLO format.

    This function partitions data into train/val, saves all artifacts, writes YAML, and returns statistics.
    Input: dataset_name, output_dir, val_fraction, seed, optional limit, num_workers, class_map
    -> Intermediate:
        1. Load dataset from HF Hub / local cache
        2. Apply optional limit for smoke testing
        3. Deterministically split into train and val partitions using seed
        4. Export train split images and labels
        5. Export val split images and labels
        6. Generate dataset.yaml
    -> Output: summary statistics dictionary
    """
    out_path = Path(output_dir)
    logger.info("Loading dataset '%s'...", dataset_name)
    raw = load_dataset(dataset_name, split="train")

    if limit is not None and limit > 0:
        logger.info("Applying limit: selecting first %d samples.", limit)
        raw = raw.select(range(min(limit, len(raw))))

    # Deterministic train / val split
    split_dict = raw.train_test_split(test_size=val_fraction, seed=seed)
    train_ds = split_dict["train"]
    val_ds = split_dict["test"]

    train_imgs, train_boxes = export_split(
        train_ds, "train", out_path, class_map=class_map, num_workers=num_workers
    )
    val_imgs, val_boxes = export_split(
        val_ds, "val", out_path, class_map=class_map, num_workers=num_workers
    )

    yaml_path = write_dataset_yaml(out_path, class_map=class_map)

    stats = {
        "dataset_name": dataset_name,
        "output_dir": str(out_path.resolve()),
        "yaml_path": str(yaml_path.resolve()),
        "train_images": train_imgs,
        "val_images": val_imgs,
        "total_images": train_imgs + val_imgs,
        "train_boxes": train_boxes,
        "val_boxes": val_boxes,
        "total_boxes": train_boxes + val_boxes,
    }
    logger.info("Export completed successfully: %s", stats)
    return stats


def verify_yolo_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """
    Rigorous verification and smoke test of an on-disk YOLO dataset.

    This function audits all bounding boxes, coordinates, file existence, and YAML configuration.
    Input: path to exported YOLO dataset directory
    -> Intermediate:
        1. Verify dataset.yaml presence and parse YAML
        2. Verify existence of images/{train,val} and labels/{train,val}
        3. Verify 1-to-1 match between image files and label files
        4. Validate every bounding box line:
           - exactly 5 tokens
           - valid integer class_id in range [0, num_classes - 1]
           - valid float normalized coordinates in [0.0, 1.0]
           - positive width and height
           - box boundaries within image margins [0.0, 1.0]
        5. Accumulate per-class instance distributions
    -> Output: verification statistics dictionary
    """
    root = Path(dataset_dir)
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset.yaml at {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", {})
    num_classes = len(names)
    assert num_classes > 0, "No classes specified in dataset.yaml"

    stats: dict[str, Any] = {
        "splits": {},
        "total_images": 0,
        "total_boxes": 0,
        "class_counts": {int(k): 0 for k in names},
    }

    for split in ["train", "val"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing labels directory: {lbl_dir}")

        image_files = sorted(img_dir.glob("*.jpg"))
        label_files = sorted(lbl_dir.glob("*.txt"))

        # Verify 1:1 image and label count
        assert len(image_files) == len(
            label_files
        ), f"Split '{split}': image count ({len(image_files)}) != label count ({len(label_files)})"

        image_stems = {p.stem for p in image_files}
        label_stems = {p.stem for p in label_files}
        missing_labels = image_stems - label_stems
        assert not missing_labels, f"Split '{split}': missing labels for stems: {missing_labels}"

        split_boxes = 0
        eps = 1e-5

        for lbl_file in label_files:
            content = lbl_file.read_text(encoding="utf-8").strip()
            if not content:
                continue  # valid hard negative / empty image

            for line_idx, line in enumerate(content.splitlines()):
                tokens = line.strip().split()
                assert len(tokens) == 5, (
                    f"{lbl_file}:{line_idx + 1}: Expected 5 tokens, got {len(tokens)}: '{line}'"
                )

                # Class ID verification
                cid_str, xc_str, yc_str, w_str, h_str = tokens
                cid = int(cid_str)
                assert 0 <= cid < num_classes, (
                    f"{lbl_file}:{line_idx + 1}: Class ID {cid} out of range [0, {num_classes - 1}]"
                )

                # Float and bounds verification
                xc = float(xc_str)
                yc = float(yc_str)
                w = float(w_str)
                h = float(h_str)

                assert 0.0 <= xc <= 1.0, f"{lbl_file}:{line_idx + 1}: xc={xc} not in [0.0, 1.0]"
                assert 0.0 <= yc <= 1.0, f"{lbl_file}:{line_idx + 1}: yc={yc} not in [0.0, 1.0]"
                assert 0.0 < w <= 1.0, f"{lbl_file}:{line_idx + 1}: width={w} not in (0.0, 1.0]"
                assert 0.0 < h <= 1.0, f"{lbl_file}:{line_idx + 1}: height={h} not in (0.0, 1.0]"

                # Corner boundary sanity checks
                x1 = xc - (w / 2.0)
                x2 = xc + (w / 2.0)
                y1 = yc - (h / 2.0)
                y2 = yc + (h / 2.0)

                assert x1 >= -eps and x2 <= 1.0 + eps, (
                    f"{lbl_file}:{line_idx + 1}: x bounds [{x1}, {x2}] exceed [0, 1]"
                )
                assert y1 >= -eps and y2 <= 1.0 + eps, (
                    f"{lbl_file}:{line_idx + 1}: y bounds [{y1}, {y2}] exceed [0, 1]"
                )

                stats["class_counts"][cid] += 1
                split_boxes += 1

        stats["splits"][split] = {
            "images": len(image_files),
            "boxes": split_boxes,
        }
        stats["total_images"] += len(image_files)
        stats["total_boxes"] += split_boxes

    logger.info("Verification passed cleanly: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export UAV-PDD2023 to Ultralytics YOLO format.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="vikhyatk/uav-pdd2023",
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/uav_pdd2023",
        help="Target directory for exported YOLO dataset",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of dataset allocated to validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on total samples (useful for rapid smoke tests)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads for saving files",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run verification on an existing export directory without re-exporting",
    )

    args = parser.parse_args()

    if args.verify_only:
        logger.info("Running verification only on %s...", args.output_dir)
        verify_yolo_dataset(args.output_dir)
    else:
        export_yolo_dataset(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            limit=args.limit,
            num_workers=args.workers,
        )
        logger.info("Running automatic post-export verification...")
        verify_yolo_dataset(args.output_dir)


if __name__ == "__main__":
    main()
