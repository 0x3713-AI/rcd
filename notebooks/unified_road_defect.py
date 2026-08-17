import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Unified Road Defect Dataset — data prep & visual annotation

        **What** — a merged YOLO-format dataset (RDD-2022 + UAV-PDD2023 +
        RoadDamageVision, 4 classes). It ships as `data/*.tar.gz` webdataset
        tars in a `images/{train,val}` + `labels/{train,val}` layout, with
        labels as YOLO `<class> cx cy w h` text files.

        **How** — `load_dataset` can't serve it faithfully (the `rdv_*` files
        come back with `jpg=None` because the webdataset parser mangles their
        keys), so we extract the tars **once** into `data/unified_road_defect/`
        and read image + label pairs straight from disk. Boxes are converted to
        the same `{name, boxes}` schema as `uav_pdd2023.py`, so the annotation
        code is identical.

        **Why** — confirm this merged dataset annotates cleanly, exactly like
        UAV-PDD2023, before using it for training.
        """
    )
    return


@app.cell
def _():
    # 4-class CRDDC schema from rdd_merged.yaml
    CLASS_NAMES = {
        0: "Longitudinal Crack (D00)",
        1: "Transverse Crack (D10)",
        2: "Alligator Crack (D20)",
        3: "Pothole (D40)",
    }
    return (CLASS_NAMES,)


@app.cell
def _():
    import tarfile
    from pathlib import Path

    DATA_DIR = Path("data/unified_road_defect")
    return (DATA_DIR, tarfile)


@app.cell
def _(DATA_DIR):
    from huggingface_hub import hf_hub_download

    if any((DATA_DIR / "images/train").glob("*.jpg")):
        tar_paths = None
    else:
        tar_paths = [
            hf_hub_download(
                "TamAko783/Unified_Road_Defect_Dataset",
                f"data/{f}",
                repo_type="dataset",
            )
            for f in ["train_a.tar.gz", "train_b.tar.gz", "val.tar.gz"]
        ]
    tar_paths
    return (tar_paths,)


@app.cell
def _(DATA_DIR, tar_paths, tarfile):
    if tar_paths is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for path in tar_paths:
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(DATA_DIR)

    len(list((DATA_DIR / "images/train").glob("*.jpg")))
    return


@app.function
def yolo_to_objects(lines, class_names):
    """YOLO label lines ('class cx cy w h', normalized) -> objects schema."""
    objects = []
    boxes_by_class = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        class_id, cx, cy, w, h = map(float, line.split())
        boxes_by_class.setdefault(int(class_id), []).append((cx, cy, w, h))

    for class_id, boxes in sorted(boxes_by_class.items()):
        objects.append(
            {
                "name": class_names[class_id],
                "boxes": [
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                    for cx, cy, w, h in boxes
                ],
            }
        )
    return objects


@app.cell
def _(CLASS_NAMES, DATA_DIR):
    from PIL import Image

    image_paths = sorted((DATA_DIR / "images/train").glob("*.jpg"))

    def get_sample(i):
        img_path = image_paths[i]
        label_path = DATA_DIR / "labels/train" / (img_path.stem + ".txt")
        objects = (
            yolo_to_objects(label_path.read_text().splitlines(), CLASS_NAMES)
            if label_path.exists()
            else []
        )
        return {"image": Image.open(img_path), "objects": objects}

    return get_sample, image_paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## function to annotate dataset
    """)
    return


@app.cell
def _():
    from PIL import ImageDraw

    def annotate_image(sample):
        image = sample["image"]
        x_max, y_max = image.size
        objects = sample["objects"]
        draw = ImageDraw.Draw(image)

        for obj in objects:
            for box in obj["boxes"]:
                x1, y1, x2, y2 = box

                x1 *= x_max
                y1 *= y_max
                x2 *= x_max
                y2 *= y_max

                coords = ((x1, y1), (x2, y2))
                draw.rectangle(coords, outline="yellow", width=3)
                draw.text((x1, y1), obj["name"], fill="black", font_size=16)
        return image

    return (annotate_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Annotated dataset
    """)
    return


@app.cell
def _(image_paths, mo):
    idx = mo.ui.slider(start=0, stop=len(image_paths) - 1, step=1, full_width=True)
    idx
    return (idx,)


@app.cell
def _(annotate_image, get_sample, idx):
    image = annotate_image(get_sample(idx.value))
    image.thumbnail((1024, 1024))
    image
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw YOLO labels for the sample above
    """)
    return


@app.cell
def _(DATA_DIR, idx, image_paths):
    label_path = DATA_DIR / "labels/train" / (image_paths[idx.value].stem + ".txt")
    label_path.read_text()
    return


if __name__ == "__main__":
    app.run()
