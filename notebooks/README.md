# Interactive Notebooks (Marimo)

Exploratory and visual inspection notebooks built with [Marimo](https://marimo.io/).
All notebooks are standard, versionable `.py` files.

---

## Available Notebooks

### 1. `demo.py` — Interactive Detection Transformer Demonstration
Full interactive visual demonstration of the **DETR Detection Transformer** on UAV pavement distress imagery.
- Browse through all 509 validation drone images.
- Dynamic confidence threshold slider (0.01 to 0.95).
- Four switchable view modes:
  - **Side-by-Side (Ground Truth vs. DETR Predictions)**
  - **DETR Predictions Only**
  - **Ground Truth Only**
  - **Transformer Decoder Cross-Attention Heatmap** (visualizes spatial query attention).
- Interactive detection summary table displaying class name, confidence, and bounding box coordinates.

**Launch:**
```bash
uv run marimo edit notebooks/demo.py
# or run in read-only presentation mode:
uv run marimo run notebooks/demo.py
```

### 2. `uav_pdd2023.py`
Visual inspection and bounding-box annotation explorer for the **UAV-PDD2023** aerial pavement distress dataset.
- Interactive slider to browse drone images.
- Visual overlay of ground-truth crack bounding boxes and class labels.
- YOLO `<class> cx cy w h` format conversion inspector.

**Launch:**
```bash
uv run marimo edit notebooks/uav_pdd2023.py
```

### 2. `unified_road_defect.py`
Visual inspection and annotation tool for the **Unified Road Defect Dataset** (merging RDD-2022, UAV-PDD2023, and RoadDamageVision into a 4-class CRDDC schema: D00, D10, D20, D40).

**Launch:**
```bash
uv run marimo edit notebooks/unified_road_defect.py
```
