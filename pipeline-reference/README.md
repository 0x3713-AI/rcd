# UAV-PDD2023 Detection Transformer Pipeline

A lean, mathematically rigorous detection transformer pipeline fine-tuning DETR-family models on the **UAV-PDD2023** aerial pavement distress dataset (`vikhyatk/uav-pdd2023`).

Features 5,088 drone images and 23,426 labeled defect instances across six canonical classes:
`longitudinal crack`, `transverse crack`, `oblique crack`, `alligator crack`, `repair`, and `pothole`.

---

## 1. Architecture & Pipeline Structure

The pipeline is organized into modular, consolidated stages:

| Stage | Module | Description |
| :--- | :--- | :--- |
| **Config** | `config.py` | Central `Config` dataclass specifying hyperparameters, canonical class maps, and model checkpoints. |
| **Data & Transforms** | `data.py` | Consolidated data loader, coordinate sanitization (healing inverted boxes, boundary clamping), pure-PIL orientation-safe augmentations (`LeanTransform`), `DetectionDataset`, and collator. |
| **Model** | `model.py` | Model and image processor builder supporting `facebook/detr-resnet-50`, `SenseTime/deformable-detr`, and `PekingU/rtdetr_r50vd` with dual-learning-rate backbone parameter groups. |
| **Training** | `train.py` | Hugging Face `Trainer` fine-tuning loop with automated device detection (CUDA/MPS/CPU) and mixed precision (`fp16`). |
| **Evaluation** | `evaluate.py` | Vectorized, pure PyTorch/NumPy COCO mAP calculator (overall and per-class metrics without heavy dependencies). |
| **Inference** | `inference.py` | Single-image and batch inference script with bounding box visualization. |
| **CLI Runner** | `pipeline.py` | Command-line orchestrator exposing all pipeline stages. |
| **Smoke & Stress Test** | `test_pipeline_smoke.py` | End-to-end synthetic pipeline validation and adversarial stress tests (non-finite values, inverted boxes, out-of-bounds coords, gradient flow). |

---

## 2. Setup & Environment

```bash
# Sync core dependencies
uv sync

# Install DETR transformer dependencies (transformers, scipy)
uv sync --extra detr
```

---

## 3. Running & Verifying the Pipeline

### A. Run Smoke & Adversarial Stress Tests
Verify the entire pipeline end-to-end without requiring network access or a GPU:
```bash
uv run python pipeline-reference/test_pipeline_smoke.py
```
This runs synthetic data through coordinate conversion, PIL augmentation, collation, a forward/backward pass, and vectorized mAP calculation.

### B. Prepare Dataset & Print Class Map
```bash
uv run python pipeline-reference/pipeline.py prepare
```

### C. Fine-Tuning the Detection Transformer
```bash
# Run training with default configuration (facebook/detr-resnet-50)
uv run python pipeline-reference/pipeline.py train

# Custom arguments
uv run python pipeline-reference/pipeline.py train --epochs 30 --batch-size 2 --no-augment
```

### D. Evaluate Checkpoint
```bash
uv run python pipeline-reference/pipeline.py evaluate --checkpoint runs/uav-pdd-detr
```

### E. Run Inference on an Image
```bash
uv run python pipeline-reference/pipeline.py infer --checkpoint runs/uav-pdd-detr --image path/to/road.jpg --output pred.jpg
```

---

## 4. Hardware Execution: CPU vs. GPU (CUDA)

- **CPU Mode (Default):** Runs out-of-the-box on standard CPU architectures. Ideal for pipeline verification, data preparation, and unit testing.
- **GPU Acceleration (NVIDIA CUDA):** For rapid fine-tuning on an NVIDIA GPU, install CUDA-enabled PyTorch wheels into your virtual environment:
  ```bash
  uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
  `train.py` will automatically detect the GPU (`torch.cuda.is_available() == True`), enable `fp16` mixed precision, and accelerate training.

---

## 5. Key Domain & Design Rules

1. **Orientation Sensitivity:** `longitudinal crack` (class 0) and `transverse crack` (class 1) are defined relative to the roadway direction. 90° and 270° rotation transforms are strictly excluded to avoid flipping ground-truth semantics.
2. **Lean Transforms:** Data augmentation uses pure `PIL` and basic arithmetic (`LeanTransform`), eliminating heavy dependencies like `albumentations` and `opencv-python-headless`.
3. **Pure-Torch Vectorized mAP:** Evaluation calculates COCO-style average precision directly in PyTorch/NumPy, removing dependencies on `pycocotools` and `torchmetrics`.
4. **Alternative Transformer Backbones:** To test advanced transformer architectures on small cracks, change `model_checkpoint` in `config.py` to:
   - `"SenseTime/deformable-detr"` (deformable attention for multi-scale crack detection)
   - `"PekingU/rtdetr_r50vd"` (Real-Time DETR, strong small-object localization)
