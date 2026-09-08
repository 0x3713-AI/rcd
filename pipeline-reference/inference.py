"""
Stage 7 -- Inference

Runs a fine-tuned checkpoint on new UAV imagery and draws predicted boxes in
the same visual style as the notebook's `annotate_image` (yellow outline +
class-name label), so ground-truth and prediction visualizations look
consistent side by side.

    python inference.py --checkpoint runs/uav-pdd-detr --image road.jpg --output pred.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def load_model(checkpoint: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device).eval()
    return model, image_processor, device


@torch.no_grad()
def predict(
    model,
    image_processor,
    image: Image.Image,
    device: str,
    score_threshold: float = 0.5,
) -> dict:
    inputs = image_processor(images=[image], return_tensors="pt").to(device)
    outputs = model(**inputs)

    height, width = image.size[1], image.size[0]  # PIL .size is (width, height)
    target_sizes = torch.tensor([[height, width]], device=device)

    results = image_processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=target_sizes
    )[0]
    return results  # {"boxes": xyxy abs, "scores": ..., "labels": ...}


def annotate_predictions(image: Image.Image, results: dict, id2label: dict[int, str]) -> Image.Image:
    """Draw predicted boxes; visual style mirrors the notebook's annotate_image."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        name = id2label.get(int(label_id), str(int(label_id)))
        draw.rectangle(((x1, y1), (x2, y2)), outline="yellow", width=3)
        draw.text((x1, max(y1 - 12, 0)), f"{name} {score:.2f}", fill="black", font_size=16)

    return annotated


def run_inference_on_path(
    checkpoint: str,
    image_path: str,
    output_path: str,
    score_threshold: float = 0.5,
) -> dict:
    model, image_processor, device = load_model(checkpoint)
    id2label = model.config.id2label

    image = Image.open(image_path).convert("RGB")
    results = predict(model, image_processor, image, device, score_threshold)
    annotated = annotate_predictions(image, results, id2label)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    print(f"{len(results['boxes'])} detections >= {score_threshold} -> saved {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run detection transformer inference on a single image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="path or hub id of the fine-tuned model")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="prediction.jpg")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_inference_on_path(args.checkpoint, args.image, args.output, args.score_threshold)


if __name__ == "__main__":
    main()
