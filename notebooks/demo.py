import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    # Ensure HF doesn't use stalled XET downloads
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # Locate project root robustly (works whether notebook is in repo root or notebooks/)
    curr_dir = Path(__file__).resolve().parent
    project_root = curr_dir if (curr_dir / "pipeline-reference").exists() else curr_dir.parent
    pipeline_dir = project_root / "pipeline-reference"
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))

    import marimo as mo
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from visualization import (
        CANONICAL_CLASS_MAP,
        compute_attention_overlay,
        draw_bounding_boxes,
        load_yolo_ground_truth,
    )

    return (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        CANONICAL_CLASS_MAP,
        Image,
        compute_attention_overlay,
        draw_bounding_boxes,
        load_yolo_ground_truth,
        mo,
        project_root,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RCD: UAV Road Crack Detection — Interactive Transformer Demo
    **Pavement Distress Detection in Low-Altitude UAV Aerial Imagery**

    This demonstration showcases a **Detection Transformer (DETR)** detecting pavement distress
    across 6 canonical distress classes:
    - <span style="color:#FF3838; font-weight:bold;">■ Longitudinal Crack</span>
    - <span style="color:#2B7FFF; font-weight:bold;">■ Transverse Crack</span>
    - <span style="color:#00D154; font-weight:bold;">■ Oblique Crack</span>
    - <span style="color:#FFA500; font-weight:bold;">■ Alligator Crack</span>
    - <span style="color:#9B51E0; font-weight:bold;">■ Repair / Patching</span>
    - <span style="color:#FFD700; font-weight:bold;">■ Pothole</span>
    """)
    return


@app.cell
def _(project_root):
    val_image_dir = project_root / "data/uav_pdd2023/images/val"
    val_label_dir = project_root / "data/uav_pdd2023/labels/val"

    if val_image_dir.exists():
        val_images = sorted(val_image_dir.glob("*.jpg"))
    else:
        val_images = []
    return val_images, val_label_dir


@app.cell
def _(AutoImageProcessor, AutoModelForObjectDetection, project_root, torch):
    # Model and processor initialization
    runs_dir = project_root / "runs"
    checkpoint_candidates = sorted(runs_dir.glob("*/checkpoint-*")) + sorted(runs_dir.glob("*"))
    valid_ckpts = [
        str(p)
        for p in checkpoint_candidates
        if (p / "config.json").exists() and (p / "model.safetensors").exists()
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_name = valid_ckpts[-1] if valid_ckpts else "facebook/detr-resnet-50"

    processor = AutoImageProcessor.from_pretrained(checkpoint_name)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint_name,
        ignore_mismatched_sizes=True,
        num_labels=6,
        _attn_implementation="eager",
    )
    model.to(device).eval()
    return checkpoint_name, device, model, processor


@app.cell
def _(checkpoint_name, device, mo, val_images):
    num_samples = max(1, len(val_images))

    sample_slider = mo.ui.slider(
        start=0,
        stop=num_samples - 1,
        step=1,
        value=0,
        label="Validation Image Index",
        full_width=True,
    )

    conf_slider = mo.ui.slider(
        start=0.01,
        stop=0.95,
        step=0.02,
        value=0.15,
        label="Confidence Threshold",
    )

    view_selector = mo.ui.radio(
        options=[
            "Side-by-Side (Ground Truth | DETR Predictions)",
            "DETR Predictions Only",
            "Ground Truth Only",
            "Transformer Decoder Attention Heatmap",
        ],
        value="Side-by-Side (Ground Truth | DETR Predictions)",
        label="View Mode",
    )

    query_slider = mo.ui.slider(
        start=0,
        stop=9,
        step=1,
        value=0,
        label="Attention Query Rank (0 = Top Detection)",
    )

    heatmap_alpha = mo.ui.slider(
        start=0.1,
        stop=0.9,
        step=0.05,
        value=0.55,
        label="Attention Heatmap Opacity",
    )

    mo.md(
        f"""
        ### ⚙️ Interactive Controls
        *Active Device: `{device}` | Model Checkpoint: `{checkpoint_name}` | Available Validation Images: `{len(val_images)}`*
        """
    )
    return (
        conf_slider,
        heatmap_alpha,
        query_slider,
        sample_slider,
        view_selector,
    )


@app.cell
def _(
    conf_slider,
    heatmap_alpha,
    mo,
    query_slider,
    sample_slider,
    view_selector,
):
    controls = mo.vstack([
        sample_slider,
        mo.hstack([conf_slider, view_selector], justify="start", gap=2),
        mo.hstack([query_slider, heatmap_alpha], justify="start", gap=2),
    ])
    controls
    return


@app.cell
def _(
    CANONICAL_CLASS_MAP,
    Image,
    compute_attention_overlay,
    conf_slider,
    device,
    draw_bounding_boxes,
    heatmap_alpha,
    load_yolo_ground_truth,
    mo,
    model,
    processor,
    query_slider,
    sample_slider,
    torch,
    val_images,
    val_label_dir,
    view_selector,
):
    if not val_images:
        display_output = mo.md("**No images found in `data/uav_pdd2023/images/val`. Run `scripts/export_yolo.py` first.**")
    else:
        img_path = val_images[sample_slider.value]
        img_raw = Image.open(img_path).convert("RGB")
        img_w, img_h = img_raw.size

        # Load ground truth from YOLO label text file
        label_path = val_label_dir / f"{img_path.stem}.txt"
        gt_annotations = load_yolo_ground_truth(label_path, img_w, img_h)

        gt_boxes = [ann["box"] for ann in gt_annotations]
        gt_classes = [ann["class_id"] for ann in gt_annotations]
        gt_img = draw_bounding_boxes(img_raw, gt_boxes, gt_classes, box_width=3)

        # Run DETR Inference with Cross-Attentions enabled
        inputs = processor(images=img_raw, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        target_sizes = torch.tensor([[img_h, img_w]], device=device)
        post_results = processor.post_process_object_detection(
            outputs, threshold=conf_slider.value, target_sizes=target_sizes
        )[0]

        pred_boxes = post_results["boxes"].cpu().numpy()
        pred_scores = post_results["scores"].cpu().numpy()
        pred_labels = post_results["labels"].cpu().numpy()

        pred_img = draw_bounding_boxes(
            img_raw,
            pred_boxes,
            pred_labels,
            scores=pred_scores,
            box_width=3,
        )

        # Build attention heatmap if requested
        cross_attns = outputs.cross_attentions
        if cross_attns and len(cross_attns) > 0:
            last_layer_attn = cross_attns[-1]
            attn_img = compute_attention_overlay(
                img_raw,
                last_layer_attn,
                query_idx=query_slider.value,
                alpha=heatmap_alpha.value,
            )
        else:
            attn_img = img_raw

        # Compose visual layout based on selected view mode
        id2name = {v: k for k, v in CANONICAL_CLASS_MAP.items()}
        selected_mode = view_selector.value

        if selected_mode == "Side-by-Side (Ground Truth | DETR Predictions)":
            visual_view = mo.hstack([
                mo.vstack([mo.md(f"**Ground Truth ({len(gt_boxes)} defects)**"), mo.image(gt_img)]),
                mo.vstack([mo.md(rf"**DETR Detections ({len(pred_boxes)} at $\ge$ {conf_slider.value:.2f})**"), mo.image(pred_img)]),
            ], justify="center", gap=2)
        elif selected_mode == "DETR Predictions Only":
            visual_view = mo.vstack([
                mo.md(rf"**DETR Detections ({len(pred_boxes)} defects at threshold $\ge$ {conf_slider.value:.2f})**"),
                mo.image(pred_img),
            ])
        elif selected_mode == "Ground Truth Only":
            visual_view = mo.vstack([
                mo.md(f"**Ground Truth Annotations ({len(gt_boxes)} defects)**"),
                mo.image(gt_img),
            ])
        else:
            visual_view = mo.vstack([
                mo.md(f"**Transformer Decoder Cross-Attention Map (Query #{query_slider.value})**"),
                mo.image(attn_img),
            ])

        # Summary table of detections
        detection_rows = []
        for b, s, l in zip(pred_boxes, pred_scores, pred_labels):
            detection_rows.append({
                "Class": id2name.get(int(l), f"Class {l}"),
                "Confidence": f"{float(s):.3f}",
                "Box [x1, y1, x2, y2]": f"[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]",
            })

        table_md = mo.ui.table(detection_rows) if detection_rows else mo.md("*No predicted detections above current threshold.*")

        display_output = mo.vstack([
            visual_view,
            mo.md(f"#### 📊 Detections Summary for `{img_path.name}`"),
            table_md,
        ])

    display_output
    return


if __name__ == "__main__":
    app.run()
