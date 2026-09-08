# RCD — Road Crack Detection
> Deep learning-based pavement distress detection for UAV aerial imagery.

This repository provides a lean, mathematically rigorous research and engineering framework for detecting road cracks and surface defects in low-altitude drone imagery. It focuses on solving the primary challenges in aerial pavement inspection: **extreme small-object miss rates**, **fine-grained localization of thin cracks**, and **efficient edge deployment**.

---

## 1. Project Overview & Current Status

The project is architected along two synergistic tracks:

1. **Track 1: Modern Detection Transformers (Primary Focus)**
   - Implements an end-to-end detection transformer pipeline in [`pipeline-reference/`](pipeline-reference/) using the canonical COCO format.
   - Native support for **DETR** (`facebook/detr-resnet-50`), **Deformable DETR** (`SenseTime/deformable-detr`), and **RT-DETR** (`PekingU/rtdetr_r50vd`).
   - Eliminates heavy third-party wrappers: features pure-PIL orientation-safe augmentations (`LeanTransform`) and a vectorized pure-PyTorch COCO mAP evaluator.
   - Complete synthetic plumbing and adversarial stress tests (`test_pipeline_smoke.py`).

2. **Track 2: URCD-YOLO & Lightweight Edge Models**
   - Complete implementation of the anchor paper [*Deep learning-based road crack detection for UAV imagery*](https://doi.org/10.1016/j.eij.2026.100985) (Yi et al., 2026) in [`src/urcd_yolo/`](src/urcd_yolo/).
   - 4 specialized architectural innovations: **Improved BiFPN** (Softmax-weighted feature fusion), **WADown** (dual-path edge-preserving downsampling), **WTConv2d** (2D Haar wavelet convolutions in `C3k2`), and **MSCA** (multi-scale directional channel attention).
   - High-performance exporter in [`scripts/export_yolo.py`](scripts/export_yolo.py) generating on-disk datasets at [`data/uav_pdd2023/`](data/uav_pdd2023/).
   - Complete CLI trainer in [`scripts/train_urcd_yolo.py`](scripts/train_urcd_yolo.py).

---

## 2. Project Directory Structure

```text
rcd/
├── data/
│   ├── uav_pdd2023/           # Exported Ultralytics YOLO dataset (images/, labels/, dataset.yaml)
│   └── unified_road_defect/   # Extracted 4-class multi-national road defect dataset
├── demo.py                    # Root shortcut to interactive Marimo transformer demo
├── notebooks/                 # Interactive Marimo visual inspection notebooks (.py)
│   ├── README.md
│   ├── demo.py                # Interactive DETR demo (predictions, attention maps, sliders)
│   ├── uav_pdd2023.py         # Visual ground-truth inspector for UAV-PDD2023
│   └── unified_road_defect.py # Visual inspector for Unified Road Defect Dataset
├── pipeline-reference/        # Consolidated Detection Transformer pipeline
│   ├── README.md
│   ├── config.py              # Central hyperparameters & canonical class map
│   ├── data.py                # Consolidated data loader, COCO conversion, & LeanTransform
│   ├── model.py               # Hugging Face DETR / RT-DETR model builder
│   ├── train.py               # Hugging Face Trainer fine-tuning loop (CPU/CUDA)
│   ├── evaluate.py            # Vectorized, pure-PyTorch COCO mAP calculator
│   ├── inference.py           # Single-image & batch inference visualizer
│   ├── visualization.py       # PIL box renderer & Cross-Attention heatmap overlay
│   ├── pipeline.py            # Unified CLI runner for transformer pipeline
│   └── test_pipeline_smoke.py # Synthetic & adversarial pipeline stress test
├── src/
│   └── urcd_yolo/             # Production URCD-YOLO model & custom PyTorch modules
│       ├── README.md          # Architecture breakdown & mathematical formulas
│       ├── modules.py         # BiFPN, WADown, Haar DWT/IDWT, WTConv2d, MSCA
│       ├── model.py           # URCDYOLO high-level model & Ultralytics integration
│       └── urcd_yolo11s.yaml  # 190-layer architecture definition (5.17M params)
├── scripts/                   # Standalone data processing & training utilities
│   ├── README.md
│   ├── export_yolo.py         # Multi-threaded UAV-PDD2023 YOLO exporter & verifier
│   ├── export_coco.py         # MS-COCO JSON format exporter & verifier
│   └── train_urcd_yolo.py     # URCD-YOLO training script (CPU/CUDA/dry-run)
├── tests/                     # Unit & mathematical property test suite (29 tests)
│   ├── test_export_yolo.py    # Invariants for YOLO data exports
│   ├── test_export_coco.py    # Invariants for MS-COCO exports
│   ├── test_urcd_modules.py   # Haar invertibility, Softmax simplex, & WADown tests
│   └── test_visualization.py  # Box renderer & attention heatmap tests
├── papers-and-datasets.md     # Literature catalogue & dataset benchmarks
├── CONTRIBUTING.md            # Engineering standards, commenting guidelines, & PR checklist
├── pyproject.toml             # Lean project dependencies & configuration
└── uv.lock                    # Deterministic dependency lockfile
```

---

## 3. Quickstart & Environment Setup

We use [`uv`](https://docs.astral.sh/uv/) for fast, deterministic dependency management. Loose `requirements.txt` files and bare `pip install` commands are strictly avoided.

### 1. Install `uv` & Sync Dependencies
```bash
# Install uv (if not already present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install core dependencies (CPU-first wheel index, ~187 MB vs ~2.5 GB)
uv sync

# (Optional) Install Detection Transformer extras (transformers, scipy)
uv sync --extra detr
```

---

## 4. Hardware Execution: CPU vs. GPU (CUDA)

This repository is built to be **device-neutral**:

- **CPU Mode (Default):**
  - Uses the official PyTorch CPU wheel index (`https://download.pytorch.org/whl/cpu`) specified in `pyproject.toml`.
  - Enables instant local development, debugging, data export, and CI testing without downloading heavy GPU binaries.
- **GPU Acceleration (NVIDIA CUDA):**
  - When training on a GPU-enabled machine (e.g. NVIDIA RTX 5050 / Ampere / Hopper), install the CUDA-accelerated PyTorch wheels into your active virtual environment:
    ```bash
    uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```
  - `train.py` automatically detects CUDA (`torch.cuda.is_available()`), enables `fp16` mixed precision, and routes model weights and tensors to the GPU.
- **Reverting to CPU Mode:**
  - To restore the lightweight CPU environment, simply re-sync:
    ```bash
    uv sync
    ```

---

## 5. How to Test, Export, and Train

### A. Run Test Suites
```bash
# 1. Run unit & mathematical invariant property tests (pytest)
uv run pytest

# 2. Run DETR plumbing & adversarial numerical stress tests
uv run python pipeline-reference/test_pipeline_smoke.py

# 3. Check code formatting & linting (Ruff)
uv run ruff check .
```

### B. Export & Verify Datasets (YOLO & COCO)
```bash
# 1. Export full UAV-PDD2023 dataset to Ultralytics YOLO format
uv run python scripts/export_yolo.py --output-dir data/uav_pdd2023 --workers 4

# 2. Export full UAV-PDD2023 annotations to standard MS-COCO JSON format
uv run python scripts/export_coco.py --output-dir data/uav_pdd2023

# 3. Verify on-disk datasets without re-exporting
uv run python scripts/export_yolo.py --output-dir data/uav_pdd2023 --verify-only
uv run python scripts/export_coco.py --output-dir data/uav_pdd2023 --verify-only
```

### C. Run the Detection Transformer Pipeline
```bash
# 1. Prepare data & display canonical class distribution
uv run python pipeline-reference/pipeline.py prepare

# 2. Quick smoke test: train 1 step on real data (takes ~2s on CPU)
uv run python pipeline-reference/pipeline.py train --max-steps 1 --batch-size 2

# 3. Full fine-tuning (default facebook/detr-resnet-50)
uv run python pipeline-reference/pipeline.py train --epochs 30 --batch-size 4

# 4. Evaluate mAP metrics on held-out validation split
uv run python pipeline-reference/pipeline.py evaluate --checkpoint runs/uav-pdd-detr

# 5. Run inference on a test road image
uv run python pipeline-reference/pipeline.py infer --checkpoint runs/uav-pdd-detr --image path/to/road.jpg --output pred.jpg
```

### D. Launch Interactive Visual Demonstration (Marimo)
```bash
# Launch interactive Detection Transformer demonstration app
uv run marimo edit notebooks/demo.py

# Or run in full presentation mode
uv run marimo run notebooks/demo.py
```

### E. Train URCD-YOLO (Anchor Paper Architecture)
```bash
# 1. Rapid 1-epoch dry-run verification on CPU
uv run python scripts/train_urcd_yolo.py --dry-run

# 2. Full production training on GPU (CUDA)
uv run python scripts/train_urcd_yolo.py --epochs 50 --batch 16 --imgsz 640 --device cuda

# 3. High-level Python usage
python -c "
from src.urcd_yolo import URCDYOLO
model = URCDYOLO('runs/detect/runs/urcd_yolo/exp/weights/best.pt')
results = model('data/uav_pdd2023/images/val/uav_val_00000.jpg')
"
```

---

## 6. Mathematical & Engineering Standards

- **Orientation Sensitivity Domain Rule:** Pavement distress classes distinguish `longitudinal crack` (class 0) and `transverse crack` (class 1) strictly by their orientation relative to the road. Rotation augmentations (90° / 270°) are **strictly forbidden** to prevent ground-truth label inversion.
- **Defensive Coordinate Guards:** All coordinate transformations sanitize inputs: non-finite coordinates are discarded, inverted boxes are healed via $\min/\max$, coordinates are clamped to $[0.0, 1.0]$, and zero-area boxes ($< 10^{-5}$) are filtered out.
- **Locked Canonical Class Map:**
  ```python
  {
      0: "longitudinal crack",
      1: "transverse crack",
      2: "oblique crack",
      3: "alligator crack",
      4: "repair",
      5: "pothole",
  }
  ```
- **Code Contribution Guidelines:** Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed coding, testing, and commenting protocols.
