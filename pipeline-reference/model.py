"""
Stage 4 -- Model

Builds the detection transformer and its matching image processor from a
checkpoint name. Uses the `Auto*` classes so config.model_checkpoint can be
swapped between DETR / Deformable DETR / RT-DETR without touching this file
or any other stage -- they all expose the same
(pixel_values, pixel_mask, labels) -> loss interface used in dataset.py and
train.py, and the same post_process_object_detection(...) used in
evaluate.py / inference.py.

`ignore_mismatched_sizes=True` is required because we're replacing the
pretrained classification head (COCO's 91 classes) with our 6 pavement
distress classes -- the box/class prediction heads get freshly initialized
while the backbone + transformer encoder/decoder keep their pretrained
weights.
"""

from __future__ import annotations

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel


def build_model_and_processor(
    checkpoint: str,
    id2label: dict[int, str],
    label2id: dict[str, int],
    image_shortest_edge: int = 480,
    image_longest_edge: int = 800,
) -> tuple[PreTrainedModel, BaseImageProcessor]:
    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        size={"shortest_edge": image_shortest_edge, "longest_edge": image_longest_edge},
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model, image_processor


def build_param_groups(model: PreTrainedModel, learning_rate: float, backbone_learning_rate: float):
    """
    Two LR groups: a low LR for the pretrained CNN backbone, the base LR for
    everything else (transformer encoder/decoder + fresh prediction heads).
    This is the standard DETR fine-tuning recipe -- the backbone is already
    well-trained and a full-size LR on it tends to destroy those features
    early in fine-tuning.
    """
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    return [
        {"params": backbone_params, "lr": backbone_learning_rate},
        {"params": other_params, "lr": learning_rate},
    ]
