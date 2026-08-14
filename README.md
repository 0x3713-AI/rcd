# RCD - Road Crack Detection
> Deep learning-based road crack detection for UAV imagery.

This repository is an open-ended, reproducible exploration of
**URCD-YOLO**, the enhanced small-object detector proposed in
[*Deep learning-based road crack detection for UAV imagery*][paper], and of
the problem space around it: detecting tiny pavement cracks in low-altitude
UAV imagery under the conflicting constraints of accuracy, model size, and
edge-deployability.

The reference paper is just for a starting point. See [papers-and-datasets.md](papers-and-datasets.md)
for the full list of suggested literature review and dataset catalogue; see
[CONTRIBUTING.md](CONTRIBUTING.md) for collaboration guidelines.

[paper]: https://doi.org/10.1016/j.eij.2026.100985

## The problem

UAV aerial imagery-based road crack detection faces three core bottlenecks:

1. **High miss rate of tiny cracks** — cracks are fine-grained, low-contrast,
   and easily lost during multi-scale downsampling.
2. **Insufficient fine-grained localization accuracy** — bounding boxes on
   thin, elongated, often fragmented cracks are hard to regress precisely.
3. **Excessive parameters** — heavy models hinder deployment on UAVs and
   other resource-constrained edge terminals (e.g. Jetson, RK3588).

## The reference approach: URCD-YOLO

URCD-YOLO is an enhanced small-object detection algorithm built on
**YOLO11s**, balancing accuracy against a lightweight design via four
targeted optimizations:

| Module | Where | What it does |
|---|---|---|
| **Improved BiFPN** | Neck | Replaces vanilla concatenation with **Softmax-normalized weighted fusion** for bidirectional feature fusion, reducing tiny-crack feature loss and fusion deviation during multi-scale aggregation. |
| **WADown** (Weighted Adaptive Dual-path Downsampling) | Backbone & neck | Replaces vanilla convolutional downsampling, cutting parameter overhead while preserving micro-crack edge and texture detail. |
| **WTConv** (Wavelet Transform Convolution) | `C3k2` | Employs cascaded wavelet decomposition to enlarge the receptive field at logarithmic parameter cost, enhancing perception of low-frequency global road features and fine-grained linear crack textures. |
| **MSCA** (Multi-scale Channel Attention) | Cross-domain | A multi-scale channel attention mechanism forming a spatial + channel + frequency collaborative optimization framework, boosting feature capture for small cracks in complex backgrounds. |

### Reported results (UAV-PDD2023)

Compared to the YOLO11s baseline, the paper reports:

- **+9.1%** improvement in `mAP@0.5`
- **−9.5%** parameters

positioning URCD-YOLO as a lightweight, high-precision small-object
detection solution for UAV and edge terminals.

## What we're doing in this repo

This is an open-ended exploration of the premise. We aim to:

- **Reproduce** the URCD-YOLO results on the public **UAV-PDD2023** dataset
  (`vikhyatk/uav-pdd2023` on Hugging Face).
- **Ablate** each proposed module (BiFPN-Softmax, WADown, WTConv, MSCA)
  independently and in combination to understand *why* it helps.
- **Push the design space** along the directions below.

### Directions worth exploring

Beyond the techniques used in the paper, we consider these promising
avenues:

- **Vision Transformers & modern detector backbones**: e.g. hybrid
  CNN-transformer detectors (RT-DETR, DETR family), Swin-based backbones,
  and attention-centric architectures that may capture long-range crack
  context better than pure CNN stacks.
- **Other strong/better architectures**: including, but not limited to, newer YOLO generations, anchor-free
  detectors, DETR-style query-based detectors, and segment-anything-style
  foundations
- **Larger, more diverse datasets** — e.g. the
  [**Unified Road Defect Dataset**][unified] on Hugging Face, which merges
  RDD-2022 + UAV-PDD2023 + RoadDamageVision into a 4-class YOLO schema, or
  the multi-national **RDD-2022**. Cross-dataset generalization is a real
  test of any crack detector.
- **Further optimizations** — quantization/ONNX/TensorRT for edge
  deployment, knowledge distillation, pruning, and NAS-based width/depth
  search in service of the edge-deployment goal.
- **Robustness** — weather/illumination augmentation, synthetic-to-real
  transfer, and evaluation on unseen road surfaces and countries.

All of these are explicitly listed in [papers-and-datasets.md](papers-and-datasets.md)
with citations.

## Project layout

```
notebooks/      one marimo notebook per pipeline stage (data_prep, train, eval, ...)
outputs/        exported run snapshots worth preserving
pyproject.toml  project dependencies + uv.lock
```
note: src/            shared, reused code (pinned in pyproject.toml), src/ will be here when project reaches a stage of going outside notebook experimentations.


## Getting started

```bash
# one-time: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install project deps from pyproject.toml + uv.lock
uv sync

# run a notebook
marimo edit notebooks/data_prep.py
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md), it covers
environment reproducibility (uv + PEP 723 sandboxing) and git workflow
for marimo notebooks.
