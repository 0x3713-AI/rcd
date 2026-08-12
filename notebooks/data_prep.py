import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
    dataset[1]
    return


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
        x_max,y_max = image.size
        objects = sample["objects"]
        draw = ImageDraw.Draw(image)

        for obj in objects:
            for box in obj["boxes"]:
                x1,y1,x2,y2 = box

                x1 *= x_max
                y1 *= y_max
                x2 *= x_max
                y2 *= y_max

                coords = ((x1, y1), (x2, y2))
                draw.rectangle(coords,outline="yellow", width=3 )
                draw.text((x1,y1), obj["name"],fill="black", font_size=16)
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
    idx = mo.ui.slider(start=0, stop=dataset.num_rows-1, step=1, full_width=True)
    idx
    return (idx,)


@app.cell
def _(annotate_image, dataset, idx):

    annotate_image(dataset[idx.value])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
