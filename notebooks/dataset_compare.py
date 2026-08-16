import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    DATASETS = {
        "vikhyatk/uav-pdd2023":                 ["train", "valid", "test"],
        "TamAko783/Unified_Road_Defect_Dataset":  ["train", "validation"],
    }
    ds_pick = mo.ui.dropdown(list(DATASETS), label="Dataset", value=list(DATASETS.keys())[0])
    ds_pick
    return DATASETS, ds_pick


@app.cell
def _(DATASETS, ds_pick, mo):
    split_pick = mo.ui.dropdown(DATASETS[ds_pick.value], label="Split" , value=DATASETS[ds_pick.value][0])
    split_pick
    return (split_pick,)


@app.cell
def _(ds_pick, split_pick):
    from datasets import load_dataset
    ds = load_dataset(ds_pick.value, split=split_pick.value)
    ds = ds[split_pick.value] if 'train' in ds else ds
    ds = ds.filter(lambda row: row.get('jpg') is not None or row.get('image') is not None)
    return (ds,)


@app.cell
def _(ds, mo):
    data_item = mo.ui.slider(0, ds.num_rows-1,  1, full_width=True)

    data_item
    return (data_item,)


@app.cell
def _(data_item, ds):
    _img = ds[data_item.value].get('jpg') or ds[data_item.value].get('image')
    _img
    return


if __name__ == "__main__":
    app.run()
