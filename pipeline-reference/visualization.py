"""
Visualization utilities for Ground Truth annotations, Detection Transformer predictions,
and Transformer Decoder Cross-Attention Heatmaps.

Provides reusable, tested drawing and tensor-to-heatmap interpolation functions.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Canonical class mapping and distinct visual color palette
CANONICAL_CLASS_MAP: dict[str, int] = {
    "longitudinal crack": 0,
    "transverse crack": 1,
    "oblique crack": 2,
    "alligator crack": 3,
    "repair": 4,
    "pothole": 5,
}

CLASS_COLORS: dict[int, str] = {
    0: "#FF3838",  # Red - Longitudinal crack
    1: "#2B7FFF",  # Blue - Transverse crack
    2: "#00D154",  # Green - Oblique crack
    3: "#FFA500",  # Orange - Alligator crack
    4: "#9B51E0",  # Purple - Repair
    5: "#FFD700",  # Gold - Pothole
}


def load_yolo_ground_truth(
    label_path: str | Path,
    image_width: int,
    image_height: int,
    id2label: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Parses a YOLO-format text label file and converts normalized boxes to absolute pixels.

    Input: label_path (.txt) + image_width + image_height
    -> Intermediate:
        1. Read text lines '<class_id> <xc> <yc> <w> <h>'
        2. Convert center-normalized to absolute corners [x1, y1, x2, y2]
        3. Clamp strictly to image boundaries [0, width] and [0, height]
    -> Output: list of dicts [{"box": [x1, y1, x2, y2], "class_id": int, "name": str}]
    """
    path = Path(label_path)
    annotations: list[dict[str, Any]] = []
    if not path.exists() or image_width <= 0 or image_height <= 0:
        return annotations

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return annotations

    inverted_map = {v: k for k, v in CANONICAL_CLASS_MAP.items()}
    lookup = id2label or inverted_map

    for line in content.splitlines():
        tokens = line.strip().split()
        if len(tokens) != 5:
            continue
        try:
            cid = int(tokens[0])
            xc, yc, w, h = (float(v) for v in tokens[1:])
        except ValueError:
            continue

        # Convert normalized center-width-height to absolute corners
        px1 = (xc - (w / 2.0)) * image_width
        py1 = (yc - (h / 2.0)) * image_height
        px2 = (xc + (w / 2.0)) * image_width
        py2 = (yc + (h / 2.0)) * image_height

        # Clamp to image frame
        x1 = max(0.0, min(px1, float(image_width)))
        y1 = max(0.0, min(py1, float(image_height)))
        x2 = max(0.0, min(px2, float(image_width)))
        y2 = max(0.0, min(py2, float(image_height)))

        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue

        name = lookup.get(cid, f"Class {cid}")
        annotations.append({
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "class_id": cid,
            "name": name,
        })

    return annotations


def draw_bounding_boxes(
    image: Image.Image,
    boxes: list[list[float]] | np.ndarray,
    class_ids: list[int] | np.ndarray,
    scores: list[float] | np.ndarray | None = None,
    id2label: dict[int, str] | None = None,
    color_map: dict[int, str] | None = None,
    box_width: int = 3,
) -> Image.Image:
    """
    Renders styled bounding boxes and class badges onto an image copy.

    Input: PIL Image + boxes [N, 4] + class_ids [N] + optional scores [N]
    -> Intermediate:
        1. Create an RGB copy of the image
        2. Iterate over boxes, draw colored rectangles with contrasting labels
    -> Output: new PIL Image with annotated boxes
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    inverted_map = {v: k for k, v in CANONICAL_CLASS_MAP.items()}
    lookup = id2label or inverted_map
    palette = color_map or CLASS_COLORS

    for idx, (box, cid) in enumerate(zip(boxes, class_ids)):
        x1, y1, x2, y2 = (float(v) for v in box)
        color = palette.get(int(cid), "#FFCC00")
        name = lookup.get(int(cid), f"Class {cid}")

        label_text = name
        if scores is not None and idx < len(scores):
            label_text = f"{name} {scores[idx]:.2f}"

        # Draw bounding rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=box_width)

        # Draw label background badge for visibility
        badge_y = max(0.0, y1 - 18.0)
        draw.rectangle([(x1, badge_y), (x1 + (len(label_text) * 8.5), badge_y + 16)], fill=color)
        draw.text((x1 + 3, badge_y + 1), label_text, fill="black")

    return annotated


def find_feature_grid(num_tokens: int, aspect_ratio: float = 1.0) -> tuple[int, int]:
    """
    Determines optimal 2D feature grid dimensions (H, W) from 1D spatial token count.

    Input: total token count + target aspect ratio (W / H)
    -> Intermediate: Factorize num_tokens into pairs minimizing aspect ratio difference
    -> Output: (height_tokens, width_tokens)
    """
    if num_tokens <= 0:
        return 1, 1

    best_h, best_w = 1, num_tokens
    best_diff = float("inf")
    limit = math.isqrt(num_tokens) + 1

    for h in range(1, limit):
        if num_tokens % h == 0:
            w = num_tokens // h
            for cand_h, cand_w in [(h, w), (w, h)]:
                diff = abs((cand_w / cand_h) - aspect_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best_h, best_w = cand_h, cand_w

    return best_h, best_w


def compute_attention_overlay(
    image: Image.Image,
    cross_attention: torch.Tensor,
    query_idx: int = 0,
    alpha: float = 0.5,
    colormap_name: str = "magma",
) -> Image.Image:
    """
    Extracts DETR decoder cross-attention weights for a specific object query
    and composites a thermal heatmap overlay onto the original image.

    Input: PIL Image + cross_attention tensor [B, num_heads, num_queries, num_tokens] or [num_heads, num_queries, num_tokens]
    -> Intermediate:
        1. Extract query attention across all attention heads
        2. Average heads and normalize to range [0.0, 1.0]
        3. Factorize spatial tokens into (H_f, W_f) grid
        4. Bilinearly interpolate to original image size (H, W)
        5. Colorize using matplotlib colormap and alpha-blend with original image
    -> Output: blended PIL Image with highlighted attention focus
    """
    if cross_attention.dim() == 4:
        attn = cross_attention[0]  # [num_heads, num_queries, num_tokens]
    else:
        attn = cross_attention

    _, num_queries, num_tokens = attn.shape
    q_idx = max(0, min(query_idx, num_queries - 1))

    # Average attention across multi-head channels for the selected query
    query_weights = attn[:, q_idx, :].mean(dim=0)  # [num_tokens]

    # Normalize weights to [0.0, 1.0]
    min_val = query_weights.min()
    max_val = query_weights.max()
    norm_weights = (query_weights - min_val) / (max_val - min_val + 1e-8)

    # Determine 2D spatial grid corresponding to backbone aspect ratio
    img_w, img_h = image.size
    aspect = float(img_w) / float(img_h) if img_h > 0 else 1.0
    grid_h, grid_w = find_feature_grid(num_tokens, aspect)

    # Reshape and bilinearly interpolate to full image resolution
    grid_tensor = norm_weights.view(1, 1, grid_h, grid_w).float()
    upsampled = F.interpolate(grid_tensor, size=(img_h, img_w), mode="bilinear", align_corners=False)[0, 0]
    heatmap_np = upsampled.cpu().numpy()

    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap_name)
    colored_map = cmap(heatmap_np)[:, :, :3]  # drop alpha channel, shape (H, W, 3) in [0, 1]
    heatmap_img = Image.fromarray((colored_map * 255.0).astype(np.uint8))

    # Blend original image and attention heatmap
    blended = Image.blend(image.convert("RGB"), heatmap_img, alpha=alpha)
    return blended
