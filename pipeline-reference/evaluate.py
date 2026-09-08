"""
Stage 6 -- Evaluate

Computes COCO-style mAP (overall + per-class) on the held-out validation
split for a fine-tuned checkpoint.

Self-contained pure PyTorch/NumPy evaluation -- zero dependency on
torchmetrics or pycocotools.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

pipeline_dir = Path(__file__).resolve().parent
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import numpy as np
import torch
from config import CFG, Config
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from data import DetectionDataset, make_collate_fn, prepare_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, orig_size: torch.Tensor) -> torch.Tensor:
    """boxes: (N, 4) normalized [cx, cy, w, h]. orig_size: (2,) [height, width]."""
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device)
    orig_h, orig_w = orig_size[0].item(), orig_size[1].item()
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


class MeanAveragePrecision:
    """Lightweight, self-contained mAP evaluator. Zero torchmetrics or pycocotools."""

    def __init__(self, box_format: str = "xyxy", iou_type: str = "bbox", class_metrics: bool = True):
        self.box_format = box_format
        self.class_metrics = class_metrics
        self.predictions: list[dict[str, torch.Tensor]] = []
        self.targets: list[dict[str, torch.Tensor]] = []

    def update(self, preds: list[dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]]) -> None:
        for p, t in zip(preds, targets):
            self.predictions.append({k: v.detach().cpu() for k, v in p.items()})
            self.targets.append({k: v.detach().cpu() for k, v in t.items()})

    @staticmethod
    def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((len(boxes1), len(boxes2)), dtype=torch.float32)
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter
        return inter / union.clamp(min=1e-6)

    def compute(self) -> dict[str, Any]:
        iou_thresholds = torch.linspace(0.5, 0.95, 10)
        all_classes = set()
        for t in self.targets:
            all_classes.update(t["labels"].tolist())
        for p in self.predictions:
            all_classes.update(p["labels"].tolist())

        classes = sorted(all_classes)
        if not classes:
            return {
                "map": 0.0,
                "map_50": 0.0,
                "map_75": 0.0,
                "mar_100": 0.0,
                "classes": torch.tensor([]),
                "map_per_class": torch.tensor([]),
            }

        aps_per_class = []
        ap50_per_class = []
        ap75_per_class = []

        for c in classes:
            c_preds = []
            c_targets = []
            for p, t in zip(self.predictions, self.targets):
                mask_p = p["labels"] == c
                mask_t = t["labels"] == c
                c_preds.append((p["boxes"][mask_p], p["scores"][mask_p]))
                c_targets.append(t["boxes"][mask_t])

            total_gt = sum(len(t_boxes) for t_boxes in c_targets)
            if total_gt == 0:
                continue

            flat_scores = []
            flat_boxes = []
            flat_img_idx = []
            for img_idx, (boxes, scores) in enumerate(c_preds):
                for b, s in zip(boxes, scores):
                    flat_boxes.append(b)
                    flat_scores.append(s.item())
                    flat_img_idx.append(img_idx)

            if not flat_scores:
                aps_per_class.append(0.0)
                ap50_per_class.append(0.0)
                ap75_per_class.append(0.0)
                continue

            sort_idx = sorted(range(len(flat_scores)), key=lambda k: flat_scores[k], reverse=True)
            flat_boxes = torch.stack([flat_boxes[k] for k in sort_idx])
            flat_img_idx = [flat_img_idx[k] for k in sort_idx]

            class_iou_aps = []
            for iou_thresh in iou_thresholds:
                thresh_val = float(iou_thresh.item())
                tp = torch.zeros(len(flat_boxes))
                fp = torch.zeros(len(flat_boxes))
                matched: list[set[int]] = [set() for _ in range(len(c_targets))]

                for pred_idx, (p_box, img_id) in enumerate(zip(flat_boxes, flat_img_idx)):
                    gt_boxes = c_targets[img_id]
                    if len(gt_boxes) == 0:
                        fp[pred_idx] = 1.0
                        continue

                    ious = self._box_iou(p_box.unsqueeze(0), gt_boxes).squeeze(0)
                    best_iou, best_gt_idx = ious.max(dim=0)
                    best_iou_val = float(best_iou.item())
                    best_gt_val = int(best_gt_idx.item())

                    if best_iou_val >= thresh_val and best_gt_val not in matched[img_id]:
                        tp[pred_idx] = 1.0
                        matched[img_id].add(best_gt_val)
                    else:
                        fp[pred_idx] = 1.0

                cum_tp = torch.cumsum(tp, dim=0)
                cum_fp = torch.cumsum(fp, dim=0)
                recall = cum_tp / float(total_gt)
                precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-6)

                # 101-point COCO interpolation
                rec_levels = torch.linspace(0.0, 1.0, 101)
                prec_interp = torch.zeros(101)
                for idx, r in enumerate(rec_levels):
                    prec_interp[idx] = precision[recall >= r].max() if (recall >= r).any() else 0.0
                class_iou_aps.append(float(prec_interp.mean().item()))

            aps_per_class.append(float(np.mean(class_iou_aps)))
            ap50_per_class.append(class_iou_aps[0])
            ap75_per_class.append(class_iou_aps[5])

        mean_map = float(np.mean(aps_per_class)) if aps_per_class else 0.0
        mean_map50 = float(np.mean(ap50_per_class)) if ap50_per_class else 0.0
        mean_map75 = float(np.mean(ap75_per_class)) if ap75_per_class else 0.0

        return {
            "map": mean_map,
            "map_50": mean_map50,
            "map_75": mean_map75,
            "mar_100": 0.0,
            "classes": torch.tensor(classes),
            "map_per_class": torch.tensor(aps_per_class),
        }


@torch.no_grad()
def evaluate(
    checkpoint: str,
    cfg: Config = CFG,
    score_threshold: float | None = None,
    device: str | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    score_threshold = cfg.eval_score_threshold if score_threshold is None else score_threshold

    splits, class_map = prepare_dataset(
        dataset_name=cfg.dataset_name,
        split=cfg.train_split,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        canonical_class_map=cfg.canonical_class_map,
    )
    id2label = {i: name for name, i in class_map.items()}

    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device).eval()

    eval_dataset = DetectionDataset(splits["validation"], class_map, transform=None)
    collate_fn = make_collate_fn(image_processor)
    loader = DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = torch.stack([lbl["orig_size"] for lbl in labels]).to(device)
        processed = image_processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )
        preds = [
            {"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()}
            for p in processed
        ]

        targets = [
            {
                "boxes": _cxcywh_norm_to_xyxy_abs(lbl["boxes"], lbl["orig_size"]),
                "labels": lbl["class_labels"],
            }
            for lbl in labels
        ]

        metric.update(preds, targets)

    results = metric.compute()

    summary = {
        "map": float(results["map"]),
        "map_50": float(results["map_50"]),
        "map_75": float(results["map_75"]),
        "mar_100": float(results["mar_100"]),
    }

    if "map_per_class" in results:
        summary["map_per_class"] = {
            id2label.get(class_id, str(class_id)): ap
            for class_id, ap in zip(results["classes"].tolist(), results["map_per_class"].tolist())
        }

    logger.info("Evaluation results: %s", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned detection transformer on UAV-PDD2023.")
    parser.add_argument("--checkpoint", type=str, required=True, help="path or hub id of the fine-tuned model")
    parser.add_argument("--score-threshold", type=float, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint, score_threshold=args.score_threshold)


if __name__ == "__main__":
    main()
