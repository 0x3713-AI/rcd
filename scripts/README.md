# Utility & Export Scripts

Command-line utilities for dataset formatting, verification, and data preprocessing.

---

## 1. `export_yolo.py` — Ultralytics YOLO Exporter & Verifier

Converts the Hugging Face `vikhyatk/uav-pdd2023` dataset into the standard Ultralytics YOLO layout at `data/uav_pdd2023/`:

```text
data/uav_pdd2023/
├── dataset.yaml
├── images/
│   ├── train/  (4,579 images, JPEG quality=95)
│   └── val/    (509 images, JPEG quality=95)
└── labels/
    ├── train/  (4,579 .txt files)
    └── val/    (509 .txt files)
```

### Mathematical & Boundary Guarantees
- **Inverted Box Healing:** Recovers swapped coordinates ($x_1 > x_2$ or $y_1 > y_2$) via $\min/\max$.
- **Unit Clamping:** Enforces strict $[0.0, 1.0]$ bounds on all normalized coordinates.
- **Degenerate Box Filter:** Rejects boxes with $w < 10^{-5}$ or $h < 10^{-5}$.
- **Hard Negatives:** Generates empty `.txt` files for zero-defect images to facilitate negative background training.
- **Canonical Class Map:** Preserves exact class ordering matching benchmark literature:
  `0: longitudinal crack`, `1: transverse crack`, `2: oblique crack`, `3: alligator crack`, `4: repair`, `5: pothole`.

### Usage
```bash
# Export full dataset to data/uav_pdd2023 with 4 worker threads
uv run python scripts/export_yolo.py --output-dir data/uav_pdd2023 --workers 4

# Run a rapid 10-sample dry-run export for smoke testing
uv run python scripts/export_yolo.py --output-dir data/test_dryrun --limit 10

# Verify an existing on-disk dataset without re-exporting
uv run python scripts/export_yolo.py --output-dir data/uav_pdd2023 --verify-only
```

---

## 2. `export_coco.py` — Standard MS-COCO JSON Exporter & Validator

Exports and validates official MS-COCO format JSON annotations matching the images in `data/uav_pdd2023/`:

```text
data/uav_pdd2023/annotations/
├── instances_train.json  (21,086 annotations across 4,579 images)
└── instances_val.json    (2,340 annotations across 509 images)
```

### Schema & Structural Compliance
- `info`: Dataset metadata, license, and version.
- `categories`: Canonical 6 pavement distress classes (`id: 0..5`).
- `images`: Image dimensions (`width`, `height`), `id`, and relative file paths.
- `annotations`: Precise `bbox` in `[x_min, y_min, w, h]` absolute pixel coordinates, pixel `area`, `iscrowd: 0`, and foreign keys `image_id` and `category_id`.

### Usage
```bash
# Export standard COCO JSON annotations to data/uav_pdd2023/annotations/
uv run python scripts/export_coco.py --output-dir data/uav_pdd2023

# Run schema and referential integrity audit on existing JSON files
uv run python scripts/export_coco.py --output-dir data/uav_pdd2023 --verify-only
```

