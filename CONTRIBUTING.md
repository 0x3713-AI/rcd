# Contributing to RCD

## All contributors must adhere to these standards before opening a pull request.

---

## 1. Core Engineering Philosophy

### A. Lean, Zero-Bloat Dependencies
- **Avoid heavy wrapper packages.** Prefer standard library, `Pillow` (PIL), and native PyTorch operations over heavy third-party libraries (e.g., avoid `opencv-python-headless`, `albumentations`, or `scipy` unless strictly required).
- **Consolidate architectures.** Avoid micro-file fragmentation. Keep related data loading, coordinate conversion, and transformations co-located in unified modules (e.g., `pipeline-reference/data.py`).
- **Dependency Isolation.** Heavy optional frameworks (such as HuggingFace `transformers` for DETR) are segregated into optional dependency groups (`uv sync --extra detr`) to keep core data preparation and YOLO tooling lean and fast.

### B. Mathematical & Computational Rigor
Every bounding box transformation, dataset parser, and model layer must enforce mathematical safety:
1. **Non-Finite Value Guards:** Explicitly reject `NaN`, `Inf`, and `-Inf` using `math.isfinite()`.
2. **Inverted Coordinate Healing:** Heal reversed coordinates ($x_1 > x_2$ or $y_1 > y_2$) via $\min(x_1, x_2)$ and $\max(x_1, x_2)$. Never allow negative box widths or heights.
3. **Unit-Interval Clamping:** Strictly clamp all normalized coordinates to $[0.0, 1.0]$.
4. **Degenerate Box Filtering:** Filter out degenerate zero-area or sub-pixel boxes ($w < 10^{-5}$ or $h < 10^{-5}$).
5. **Physical Domain Rules (Orientation Sensitivity):** Pavement distress types distinguish *longitudinal cracks* (class 0) and *transverse cracks* (class 1) strictly by their orientation relative to the road. **Never apply rotation transforms** (e.g. 90° or 270°), as this would invert the ground-truth distress labels. Only horizontal/vertical flips and photometric adjustments are permissible.
6. **Strict Canonical Class Mapping:** All loaders, exporters, and evaluators must adhere to the locked canonical class map:
   ```python
   CANONICAL_CLASS_MAP = {
       "longitudinal crack": 0,
       "transverse crack": 1,
       "oblique crack": 2,
       "alligator crack": 3,
       "repair": 4,
       "pothole": 5,
   }
   ```
   Class names must be parsed case-insensitively and whitespace-tolerantly (`name.strip().lower()`).

### C. Standardized Commenting Format
Every public function, transformation, and significant block of logic must include clean, concise comments matching the pattern:
```python
"""
This function does [action].
Input: [input type and properties]
-> Intermediate: [step 1, step 2, ...]
-> Output: [output type and guarantees]
"""
```

---

## 2. Environment Management with `uv`

We use [`uv`](https://docs.astral.sh/uv/) for high-speed, deterministic package management.

> [!WARNING]
> **Never run bare `pip install` or create loose `requirements.txt` files.** Always use `uv`.

### Setup
```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync base dependencies (CPU-first wheel index, ~187 MB)
uv sync

# 3. (Optional) Sync with Detection Transformer dependencies (transformers, scipy)
uv sync --extra detr
```

### Adding New Packages
```bash
# Add runtime dependency
uv add <package_name>

# Add development / testing dependency
uv add --dev <package_name>
```
This updates `pyproject.toml` and `uv.lock` deterministically. Always commit both files together.

---

## 3. Hardware Execution: CPU vs. GPU (CUDA)

This repository is designed to be **device-neutral**:
- **CPU Default (Lightweight Development & CI):** By default, `pyproject.toml` pins PyTorch against the official CPU wheel index (`https://download.pytorch.org/whl/cpu`). This enables developers and CI runners to install and test the entire project without downloading 2.3 GB of CUDA binary wheels.
- **GPGPU / CUDA Acceleration (Training & Large-Scale Inference):** When training on an NVIDIA GPU (e.g., RTX 5050 or cluster node), install the CUDA-accelerated PyTorch wheels into your virtual environment:
  ```bash
  uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
  Once installed, all training pipelines (`train.py`) automatically detect CUDA via `torch.cuda.is_available()`, activate `fp16=True` mixed precision, and route batches directly to the GPU.

To revert back to the lean CPU environment:
```bash
uv sync
```

---

## 4. Testing & Code Quality Standards

Every pull request must pass all three test tiers locally:

### 1. Unit & Property Tests (`pytest`)
```bash
uv run pytest
```
All new exporters, modules, and utilities must include unit tests in the `tests/` directory verifying standard behavior, boundary clamping, and adversarial edge cases.

### 2. DETR Plumbing & Adversarial Stress Test
```bash
uv run python pipeline-reference/test_pipeline_smoke.py
```
Validates synthetic batch flows, DETR model initialization, loss computation, finite gradient flow (zero NaNs), and single-batch loss convergence under optimization.

### 3. Code Quality & Linting (`ruff`)
```bash
uv run ruff check .
```
We enforce strict linting and formatting via Ruff. Auto-fixable issues can be addressed with:
```bash
uv run ruff check --fix .
```

---

## 5. Working with Marimo Notebooks

Interactive notebooks live in `notebooks/` (e.g., `uav_pdd2023.py`, `unified_road_defect.py`).

1. **Pure Python Format:** Marimo stores notebooks as versionable, standard `.py` files.
2. **Launch Notebooks:**
   ```bash
   uv run marimo edit notebooks/uav_pdd2023.py
   ```
3. **Stage Separation:** Keep notebooks modular — one notebook per dataset or analytical milestone.
4. **Export Significant Artifacts:** If a notebook run produces visual artifacts or metrics worth archiving:
   ```bash
   uv run marimo export html notebooks/uav_pdd2023.py -o outputs/uav_pdd2023_analysis.html
   ```

---

## 6. Pre-PR Checklist

Before submitting a pull request, ensure:
- [ ] `uv run pytest` passes 100% of unit and invariant tests.
- [ ] `uv run python pipeline-reference/test_pipeline_smoke.py` passes cleanly.
- [ ] `uv run ruff check .` reports zero errors.
- [ ] No temporary files, `.DS_Store`, or raw dataset folders (`data/`) are staged for git commit.
- [ ] Comments follow the `# Input: ... -> Intermediate: ... -> Output: ...` convention.
- [ ] All coordinates are guarded against non-finite values and out-of-bounds errors.
