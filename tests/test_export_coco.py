"""
Unit and verification tests for COCO format export and validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset
from PIL import Image

from scripts.export_coco import (
    CANONICAL_CLASS_MAP,
    build_coco_split_dict,
    sanitize_coco_box,
    verify_coco_json,
)


def test_sanitize_coco_box_normal() -> None:
    """Tests normal bounding box conversion to pixel space."""
    # Normalized [0.1, 0.2, 0.5, 0.8] on 640x640
    # px1 = 64, py1 = 128, px2 = 320, py2 = 512 -> w = 256, h = 384
    res = sanitize_coco_box([0.1, 0.2, 0.5, 0.8], image_width=640, image_height=640)
    assert res is not None
    x, y, w, h = res
    assert pytest.approx(x, abs=0.1) == 64.0
    assert pytest.approx(y, abs=0.1) == 128.0
    assert pytest.approx(w, abs=0.1) == 256.0
    assert pytest.approx(h, abs=0.1) == 384.0


def test_sanitize_coco_box_inverted_healing() -> None:
    """Tests that inverted normalized coordinates are healed in COCO export."""
    res = sanitize_coco_box([0.5, 0.8, 0.1, 0.2], image_width=640, image_height=640)
    assert res is not None
    x, y, w, h = res
    assert pytest.approx(x, abs=0.1) == 64.0
    assert pytest.approx(y, abs=0.1) == 128.0
    assert pytest.approx(w, abs=0.1) == 256.0
    assert pytest.approx(h, abs=0.1) == 384.0


def test_sanitize_coco_box_degenerate_and_non_finite() -> None:
    """Tests that degenerate and non-finite boxes return None."""
    assert sanitize_coco_box([0.2, 0.2, 0.2, 0.8], 640, 640) is None  # w = 0
    assert sanitize_coco_box([float("nan"), 0.2, 0.5, 0.8], 640, 640) is None
    assert sanitize_coco_box([0.1, float("inf"), 0.5, 0.8], 640, 640) is None


def test_synthetic_coco_export_and_verification(tmp_path: Path) -> None:
    """Tests end-to-end COCO dictionary generation and verification."""
    dummy_img = Image.new("RGB", (640, 640), color=(0, 0, 0))
    ds = Dataset.from_dict({
        "image": [dummy_img, dummy_img],
        "objects": [
            [{"name": "Longitudinal crack", "boxes": [[0.1, 0.2, 0.3, 0.4]]}],
            [],  # empty image
        ],
    })

    coco_dict, ann_count = build_coco_split_dict(ds, "train", CANONICAL_CLASS_MAP, start_ann_id=0)
    assert len(coco_dict["images"]) == 2
    assert len(coco_dict["annotations"]) == 1
    assert ann_count == 1
    assert coco_dict["annotations"][0]["category_id"] == 0

    json_path = tmp_path / "test_coco.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f)

    stats = verify_coco_json(json_path)
    assert stats["num_images"] == 2
    assert stats["num_annotations"] == 1
    assert stats["status"] == "valid"


def test_on_disk_coco_val_json_is_valid() -> None:
    """Verifies that the generated on-disk instances_val.json passes full COCO validation."""
    val_json = Path("data/uav_pdd2023/annotations/instances_val.json")
    if val_json.exists():
        stats = verify_coco_json(val_json)
        assert stats["num_images"] == 509
        assert stats["num_annotations"] == 2340
        assert stats["status"] == "valid"
