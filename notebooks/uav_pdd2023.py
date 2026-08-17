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
        # UAV-PDD2023 — data prep & visual annotation

        **What** — `vikhyatk/uav-pdd2023` is a Hugging Face parquet dataset of
        drone pavement images with bounding-box annotations. Each row is
        `{"image", "objects"}` where `objects = [{"name", "boxes"}]` and box
        coordinates are normalized to [0, 1].

        **How** — load it with `datasets`, build a sorted (deterministic)
        class map, browse images on a slider while drawing the ground-truth
        boxes, then convert `objects` to YOLO `<class> cx cy w h` text.

        **Why** — a quick visual sanity check of the annotations before
        training, and a first step toward a YOLO-format dataset for the
        detection pipeline.
        """
    )
    return


@app.cell
def _():
    from datasets import load_dataset

    return (load_dataset,)


@app.cell
def _(load_dataset):
    dataset = load_dataset("vikhyatk/uav-pdd2023", split="train")
    return (dataset,)


@app.cell
def _(dataset):
    dataset.features
    return


@app.cell
def _(dataset):
    class_names = set()
    objects = dataset["objects"]
    for row in objects:
        for obj in row:
            class_names.add(obj["name"])

    class_names = sorted(set(class_names))

    class_map = {
        value: i for i, value in enumerate(class_names)
    }
    class_map
    return (class_map,)


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
def _(dataset, mo):
    idx = mo.ui.slider(start=0, stop=dataset.num_rows - 1, step=1, full_width=True)
    idx
    return (idx,)


@app.cell
def _(annotate_image, dataset, idx):
    annotate_image(dataset[idx.value])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Convert to a YOLO Dataset
    """)
    return


@app.cell
def _(class_map):
    def objects_to_yolo(objects):
        labels = []
        for obj in objects:
            class_id = class_map[obj["name"]]
            for box in obj["boxes"]:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                labels.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
        return labels

    return (objects_to_yolo,)


@app.cell
def _(objects_to_yolo):
    def convert_to_yolo(sample):
        sample["yolo_annotations"] = objects_to_yolo(sample["objects"])
        return sample

    return (convert_to_yolo,)


@app.cell
def _(convert_to_yolo, dataset):
    yolo_dataset = dataset.map(convert_to_yolo)
    return (yolo_dataset,)


@app.cell
def _(yolo_dataset):
    yolo_dataset[0]
    return


if __name__ == "__main__":
    app.run()
