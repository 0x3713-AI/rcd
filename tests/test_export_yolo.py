"""
Smoke and mathematical verification tests for YOLO dataset export.

Tests coordinate math, boundary clamping, inverted coordinate healing,
degenerate filtering, non-finite guards, and end-to-end export verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset
from PIL import Image

from scripts.export_yolo import (
    CANONICAL_CLASS_MAP,
    export_split,
    normalize_class_name,
    objects_to_yolo_lines,
    sanitize_and_convert_box,
    verify_yolo_dataset,
    write_dataset_yaml,
)


def test_normalize_class_name() -> None:
    """Tests whitespace stripping and lowercase normalization."""
    assert normalize_class_name("  Longitudinal Crack  ") == "longitudinal crack"
    assert normalize_class_name("Transverse crack") == "transverse crack"
    assert normalize_class_name("POTHOLE") == "pothole"


def test_sanitize_and_convert_box_normal() -> None:
    """
    Tests standard valid bounding box conversion.
    Input: [0.1, 0.2, 0.5, 0.8], class_id=1
    Expected: xc=0.3, yc=0.5, w=0.4, h=0.6
    """
    result = sanitize_and_convert_box([0.1, 0.2, 0.5, 0.8], class_id=1)
    assert result is not None
    cid, xc, yc, w, h = result
    assert cid == 1
    assert pytest.approx(xc, abs=1e-6) == 0.3
    assert pytest.approx(yc, abs=1e-6) == 0.5
    assert pytest.approx(w, abs=1e-6) == 0.4
    assert pytest.approx(h, abs=1e-6) == 0.6


def test_sanitize_and_convert_box_inverted_healing() -> None:
    """
    Tests that inverted coordinates (x1 > x2 or y1 > y2) are healed.
    Input: [0.5, 0.8, 0.1, 0.2], class_id=0
    Expected: nx1=0.1, nx2=0.5, ny1=0.2, ny2=0.8 -> xc=0.3, yc=0.5, w=0.4, h=0.6
    """
    result = sanitize_and_convert_box([0.5, 0.8, 0.1, 0.2], class_id=0)
    assert result is not None
    cid, xc, yc, w, h = result
    assert cid == 0
    assert pytest.approx(xc, abs=1e-6) == 0.3
    assert pytest.approx(yc, abs=1e-6) == 0.5
    assert pytest.approx(w, abs=1e-6) == 0.4
    assert pytest.approx(h, abs=1e-6) == 0.6


def test_sanitize_and_convert_box_out_of_bounds_clamping() -> None:
    """
    Tests that out-of-bounds coordinates (<0 or >1) are clamped strictly to [0.0, 1.0].
    Input: [-0.2, -0.5, 1.2, 1.5], class_id=3
    Expected: nx1=0.0, ny1=0.0, nx2=1.0, ny2=1.0 -> xc=0.5, yc=0.5, w=1.0, h=1.0
    """
    result = sanitize_and_convert_box([-0.2, -0.5, 1.2, 1.5], class_id=3)
    assert result is not None
    cid, xc, yc, w, h = result
    assert cid == 3
    assert pytest.approx(xc, abs=1e-6) == 0.5
    assert pytest.approx(yc, abs=1e-6) == 0.5
    assert pytest.approx(w, abs=1e-6) == 1.0
    assert pytest.approx(h, abs=1e-6) == 1.0


def test_sanitize_and_convert_box_degenerate_filtering() -> None:
    """Tests that zero-width, zero-height, or microscopic boxes are rejected."""
    # Zero width
    assert sanitize_and_convert_box([0.2, 0.2, 0.2, 0.6], class_id=0) is None
    # Zero height
    assert sanitize_and_convert_box([0.2, 0.5, 0.6, 0.5], class_id=0) is None
    # Sub-epsilon width (< 1e-5)
    assert sanitize_and_convert_box([0.2, 0.2, 0.2 + 1e-6, 0.6], class_id=0) is None


def test_sanitize_and_convert_box_non_finite() -> None:
    """Tests that NaN, Inf, and -Inf coordinates are rejected."""
    assert sanitize_and_convert_box([float("nan"), 0.2, 0.5, 0.8], class_id=0) is None
    assert sanitize_and_convert_box([0.1, float("inf"), 0.5, 0.8], class_id=0) is None
    assert sanitize_and_convert_box([0.1, 0.2, float("-inf"), 0.8], class_id=0) is None
    assert sanitize_and_convert_box([0.1, 0.2, 0.5], class_id=0) is None  # invalid length


def test_objects_to_yolo_lines_case_tolerance() -> None:
    """Tests case-insensitive class name matching in objects_to_yolo_lines."""
    objects = [
        {"name": "Transverse crack", "boxes": [[0.1, 0.2, 0.3, 0.4]]},
        {"name": "LONGITUDINAL CRACK", "boxes": [[0.5, 0.5, 0.7, 0.8]]},
        {"name": "Unknown Distress", "boxes": [[0.1, 0.1, 0.2, 0.2]]},  # ignored
    ]
    lines = objects_to_yolo_lines(objects, CANONICAL_CLASS_MAP)
    assert len(lines) == 2

    # Transverse crack is class 1
    tokens0 = lines[0].split()
    assert tokens0[0] == "1"
    assert pytest.approx(float(tokens0[1]), abs=1e-5) == 0.2  # xc = (0.1+0.3)/2
    assert pytest.approx(float(tokens0[2]), abs=1e-5) == 0.3  # yc = (0.2+0.4)/2

    # Longitudinal crack is class 0
    tokens1 = lines[1].split()
    assert tokens1[0] == "0"
    assert pytest.approx(float(tokens1[1]), abs=1e-5) == 0.6  # xc = (0.5+0.7)/2
    assert pytest.approx(float(tokens1[2]), abs=1e-5) == 0.65  # yc = (0.5+0.8)/2


def test_objects_to_yolo_lines_empty() -> None:
    """Tests that empty objects or objects without boxes yield empty list."""
    assert objects_to_yolo_lines([], CANONICAL_CLASS_MAP) == []
    assert objects_to_yolo_lines(None, CANONICAL_CLASS_MAP) == []
    assert objects_to_yolo_lines([{"name": "pothole", "boxes": []}], CANONICAL_CLASS_MAP) == []


def test_write_dataset_yaml(tmp_path: Path) -> None:
    """Tests writing dataset.yaml configuration file."""
    yaml_file = write_dataset_yaml(tmp_path, CANONICAL_CLASS_MAP)
    assert yaml_file.exists()
    content = yaml_file.read_text(encoding="utf-8")
    assert "images/train" in content
    assert "images/val" in content
    assert "longitudinal crack" in content
    assert "pothole" in content


def test_end_to_end_mini_export_and_verification(tmp_path: Path) -> None:
    """
    Creates a synthetic mini dataset with normal, inverted, out-of-bounds,
    and empty (hard negative) images, exports it, and verifies it with verify_yolo_dataset.
    """
    dummy_img = Image.new("RGB", (640, 640), color=(128, 128, 128))

    synthetic_data = {
        "image": [dummy_img, dummy_img, dummy_img],
        "objects": [
            # Sample 0: normal box + inverted box
            [
                {"name": "Longitudinal crack", "boxes": [[0.1, 0.2, 0.4, 0.6]]},
                {"name": "Pothole", "boxes": [[0.8, 0.8, 0.6, 0.6]]},
            ],
            # Sample 1: hard negative (no objects)
            [],
            # Sample 2: out-of-bounds box + degenerate box
            [
                {"name": "Transverse crack", "boxes": [[-0.1, 0.2, 1.1, 0.8]]},
                {"name": "Repair", "boxes": [[0.3, 0.3, 0.3, 0.5]]},  # degenerate w=0
            ],
        ],
    }

    ds = Dataset.from_dict(synthetic_data)
    out_dir = tmp_path / "test_yolo_dataset"

    # Export split
    total_imgs, total_boxes = export_split(
        ds, "train", out_dir, class_map=CANONICAL_CLASS_MAP, num_workers=1
    )
    assert total_imgs == 3
    assert total_boxes == 3  # 2 from sample 0, 0 from sample 1, 1 valid from sample 2

    # Export empty val split just to satisfy YOLO structure
    val_ds = Dataset.from_dict({"image": [dummy_img], "objects": [[]]})
    export_split(val_ds, "val", out_dir, class_map=CANONICAL_CLASS_MAP, num_workers=1)

    # Write YAML
    write_dataset_yaml(out_dir, CANONICAL_CLASS_MAP)

    # Verify hard negative label file is empty
    hard_neg_lbl = out_dir / "labels" / "train" / "uav_train_00001.txt"
    assert hard_neg_lbl.exists()
    assert hard_neg_lbl.read_text(encoding="utf-8").strip() == ""

    # Verify sample 0 label file has 2 boxes
    sample0_lbl = out_dir / "labels" / "train" / "uav_train_00000.txt"
    assert sample0_lbl.exists()
    lines = sample0_lbl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    # Run full verification
    stats = verify_yolo_dataset(out_dir)
    assert stats["total_images"] == 4
    assert stats["total_boxes"] == 3
    assert stats["class_counts"][0] == 1  # Longitudinal crack
    assert stats["class_counts"][5] == 1  # Pothole
    assert stats["class_counts"][1] == 1  # Transverse crack
    assert stats["class_counts"][4] == 0  # Repair (degenerate skipped)


def test_randomized_mathematical_invariants() -> None:
    """
    Stress-tests sanitize_and_convert_box with 1,000 diverse random boxes.
    Verifies that whenever a box is accepted, all bounding invariants hold exactly.
    """
    import random

    rng = random.Random(42)
    test_cases = [
        # Deterministic boundary edge cases
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-100.0, -50.0, 200.0, 300.0],
        [0.5, 0.5, 0.5 + 1e-6, 0.5 + 1e-6],
        [float("nan"), 0.5, 0.6, 0.7],
        [0.5, float("-inf"), 0.6, 0.7],
    ]

    # Generate 1,000 random boxes with values spanning negative to beyond 1.0
    for _ in range(1000):
        coords = [rng.uniform(-2.0, 3.0) for _ in range(4)]
        test_cases.append(coords)

    eps = 1e-6
    accepted_count = 0

    for box in test_cases:
        res = sanitize_and_convert_box(box, class_id=0)
        if res is not None:
            cid, xc, yc, w, h = res
            accepted_count += 1
            assert cid == 0
            assert 0.0 <= xc <= 1.0
            assert 0.0 <= yc <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0
            # Strict boundary invariants
            x1 = xc - (w / 2.0)
            x2 = xc + (w / 2.0)
            y1 = yc - (h / 2.0)
            y2 = yc + (h / 2.0)
            assert x1 >= -eps, f"x1={x1} violates lower bound"
            assert x2 <= 1.0 + eps, f"x2={x2} violates upper bound"
            assert y1 >= -eps, f"y1={y1} violates lower bound"
            assert y2 <= 1.0 + eps, f"y2={y2} violates upper bound"

    assert accepted_count > 0, "At least some randomized boxes should be accepted"


def test_verify_dataset_detects_corruptions(tmp_path: Path) -> None:
    """Verifies that verify_yolo_dataset correctly rejects invalid class ids or out-of-bounds coordinates."""
    dummy_img = Image.new("RGB", (640, 640), color=(0, 0, 0))
    ds = Dataset.from_dict({"image": [dummy_img], "objects": [[]]})
    out_dir = tmp_path / "corrupt_dataset"
    export_split(ds, "train", out_dir, CANONICAL_CLASS_MAP, num_workers=1)
    export_split(ds, "val", out_dir, CANONICAL_CLASS_MAP, num_workers=1)
    write_dataset_yaml(out_dir, CANONICAL_CLASS_MAP)

    # Corrupt a label with invalid class ID
    bad_label_file = out_dir / "labels" / "train" / "uav_train_00000.txt"
    bad_label_file.write_text("99 0.5 0.5 0.2 0.2\n")

    with pytest.raises(AssertionError, match="Class ID 99 out of range"):
        verify_yolo_dataset(out_dir)

    # Corrupt with out-of-bounds xc
    bad_label_file.write_text("0 1.5 0.5 0.2 0.2\n")
    with pytest.raises(AssertionError, match="not in \\[0.0, 1.0\\]"):
        verify_yolo_dataset(out_dir)

