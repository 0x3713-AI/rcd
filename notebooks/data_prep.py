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
def _(dataset, mo):
    img = dataset[3]["image"]
    mo.hstack([img, img.convert("L")])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
