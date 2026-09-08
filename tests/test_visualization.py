"""
Unit and property tests for visualization and attention overlay utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

pipeline_dir = Path(__file__).resolve().parent.parent / "pipeline-reference"
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

from visualization import (
    compute_attention_overlay,
    draw_bounding_boxes,
    find_feature_grid,
    load_yolo_ground_truth,
)


def test_find_feature_grid() -> None:
    """Tests feature grid factorization across square and non-square token counts."""
    # 625 tokens at 1:1 aspect ratio -> 25x25
    assert find_feature_grid(625, 1.0) == (25, 25)
    # 400 tokens at 1:1 aspect ratio -> 20x20
    assert find_feature_grid(400, 1.0) == (20, 20)
    # 500 tokens at 1.25 aspect ratio (w/h = 25/20 = 1.25)
    assert find_feature_grid(500, 1.25) == (20, 25)
    # 100 tokens at 1:1 -> 10x10
    assert find_feature_grid(100, 1.0) == (10, 10)


def test_load_yolo_ground_truth(tmp_path: Path) -> None:
    """Tests parsing and clamping of YOLO ground truth text annotations."""
    label_file = tmp_path / "sample.txt"
    # Class 0: xc=0.5, yc=0.5, w=0.2, h=0.4 on 640x640 -> [256, 192, 384, 448]
    # Class 1: xc=0.95 (partially out of bounds) -> clamped
    label_file.write_text("0 0.5 0.5 0.2 0.4\n1 0.95 0.5 0.2 0.2\n")

    anns = load_yolo_ground_truth(label_file, 640, 640)
    assert len(anns) == 2
    assert anns[0]["class_id"] == 0
    assert anns[0]["name"] == "longitudinal crack"
    x1, y1, x2, y2 = anns[0]["box"]
    assert pytest.approx(x1, abs=1.0) == 256.0
    assert pytest.approx(y1, abs=1.0) == 192.0
    assert pytest.approx(x2, abs=1.0) == 384.0
    assert pytest.approx(y2, abs=1.0) == 448.0

    # Clamped box
    assert anns[1]["box"][2] <= 640.0


def test_draw_bounding_boxes() -> None:
    """Tests drawing bounding boxes onto PIL images."""
    img = Image.new("RGB", (640, 640), color=(100, 100, 100))
    boxes = [[100.0, 100.0, 200.0, 200.0]]
    classes = [0]
    scores = [0.85]

    annotated = draw_bounding_boxes(img, boxes, classes, scores)
    assert annotated.size == (640, 640)
    assert annotated.mode == "RGB"


def test_compute_attention_overlay() -> None:
    """Tests thermal heatmap generation and blending for transformer attention."""
    img = Image.new("RGB", (320, 240), color=(50, 50, 50))
    # Synthetic cross-attention tensor: [num_heads=8, num_queries=100, num_tokens=400]
    cross_attn = torch.rand(8, 100, 400)

    overlay = compute_attention_overlay(img, cross_attn, query_idx=0, alpha=0.5)
    assert overlay.size == (320, 240)
    assert overlay.mode == "RGB"
