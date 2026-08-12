# RCD - Road Crack Detection
> Deep Learning-based Road Crack Detection for UAV imagery


# Collaboration Guide

This document covers two things every contributor must follow: how we manage
environments, and how we use git with marimo notebooks. Read before your
first PR.

---

## 1. Environment Reproducibility

We use [`uv`](https://docs.astral.sh/uv/) as the package manager. Do not use
`pip install` directly in this repo — it will drift from the lockfile and
break reproducibility for everyone else.

### Why this matters for CV specifically

Computer vision stacks are fragile: `torch`, CUDA build, `opencv-python`,
`timm`, and driver versions all need to line up. A notebook that runs on
your machine can silently fail or (worse) silently produce different
numbers on a teammate's machine if versions drift. We fix this with two
layers:

1. **Project-level dependencies** — shared code in `src/`, pinned in
   `pyproject.toml`.
2. **Notebook-level sandboxing** — experimental notebooks declare their own
   dependencies inline (PEP 723) and run isolated via `marimo edit --sandbox`.

### Project-level setup

```bash
# one-time: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install project deps from pyproject.toml + uv.lock
uv sync
```

`uv.lock` is committed to git. **Never hand-edit it.** If you add a
dependency to `src/` or shared tooling:

```bash
uv add torch torchvision
uv add --dev pytest ruff
```

This updates `pyproject.toml` and `uv.lock` together — commit both in the
same PR as the code that needs the new dependency.

### Notebook-level sandboxing

For notebooks under `notebooks/` — especially exploratory or
experiment-specific ones — declare dependencies at the top of the file
using inline script metadata:

```python
# /// script
# dependencies = [
#   "torch==2.4.0",
#   "opencv-python==4.10.0.84",
#   "timm==1.0.9",
# ]
# ///

import marimo
app = marimo.App()
```

Run it isolated:

```bash
marimo edit --sandbox notebooks/train.py
```

This spins up a throwaway env matching exactly what's declared — not
whatever happens to be in your global env or `.venv`. It means:

- Two people can run the same notebook with two different torch/CUDA
  combos and neither breaks the other's setup.
- A notebook someone wrote three weeks ago still runs exactly the same way
  today.

### Rule of thumb

| Code lives in | Managed by | Command |
|---|---|---|
| `src/` (shared, reused across notebooks) | `pyproject.toml` + `uv.lock` | `uv sync`, `uv add` |
| `notebooks/*.py` (experiment-specific) | inline PEP 723 block | `marimo edit --sandbox` |

If a dependency is used in more than one notebook, promote it to
`pyproject.toml` and import from `src/` instead of duplicating version pins
across notebook headers.

---

## 2. Git Workflow

marimo notebooks are stored as plain `.py` files. Treat them like regular
Python modules in code review — because that's what they are.

### Notebook granularity

**One notebook = one logical pipeline stage.** Don't build a single
mega-notebook that does data loading, training, and eval all in one file.
Split by stage:

```
notebooks/
  data_prep.py
  train.py
  eval.py
```

Reasons:
- Smaller, focused diffs.
- Two people can work on different stages in parallel without touching the
  same file.
- Reactive execution stays predictable — a huge single notebook has a huge
  dependency graph, which makes stale-cell behavior harder to reason about.

### Ownership

No notebook is "owned" silently by one person. Since cells are just
functions in a `.py` file, anyone can open a PR against any notebook.
Review it like you'd review a module — read the diff, don't just re-run it
and eyeball the output.

### Commit messages

Describe the analytical or logical change, not the file operation.

```
✅ "Add mixup augmentation to training pipeline"
✅ "Fix off-by-one in bbox IoU calculation"
❌ "update train.py"
❌ "wip"
```

Someone should be able to read `git log --oneline` on a notebook and
understand the experiment history without opening the file.

### Preserving outputs before merging significant runs

Cell outputs are **not** stored in the `.py` file — only code is. If a run
produces results worth keeping a record of (a training curve, a
qualitative eval grid, a metrics table), export a snapshot before merging:

```bash
marimo export html notebooks/train.py -o outputs/train_run_2026-08-13.html
```

Commit the exported HTML to `outputs/` (or attach it to the PR) if it's
worth preserving. This is optional for routine runs — do it when a run
represents a checkpoint someone will want to reference later (e.g. "the run
that fixed the augmentation bug").

### Avoid concurrent edits to the same notebook

Merge conflicts on a `.py` file are resolvable with normal git tooling —
that's the whole point of this setup. But there's a subtler failure mode:
if two people reorder or restructure cells in the same notebook
concurrently, the merged file can be syntactically fine but semantically
broken — the reactive dependency graph no longer matches what either
person tested.

Mitigation:
- Split work by pipeline stage (see above) so this rarely comes up.
- If two people genuinely need to touch the same notebook at once,
  coordinate — don't just push and hope git merges it cleanly.
- After any merge that touched a notebook, **re-run it top to bottom**
  before trusting it. Don't assume a clean merge means a working notebook.

### PR checklist

- [ ] Notebook runs clean top-to-bottom (`marimo edit --sandbox` or
      `uv run`) before opening the PR
- [ ] New dependencies added via `uv add`, not hand-edited into
      `pyproject.toml`
- [ ] Commit messages describe the analytical change
- [ ] Shared logic moved to `src/`, not duplicated across notebooks
- [ ] Significant run outputs exported if worth preserving
