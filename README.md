# Opto Analysis

Analysis pipeline for C. elegans optogenetics experiments. Takes merged tracking CSVs through decay-recovery curve fitting, bootstrap analysis, and visualization.

## Pipeline Overview

```
Raw tracking CSVs (from tracking_pipeline/batch_tracking_opto.py)
  → tracking_pipeline/prep_track_data.py  (collect finished tracks as PC#_<folder>.csv)
  → tracking_pipeline/merge_files.py      (wide → long format composite.csv)
  → bootstrap_opto.py     (decay-recovery fitting + bootstrap)
  → graph_viewer_opto.py  (visualize bootstrap results)
```

---

## Scripts

### Building `composite.csv`
`merge_files.py` (and the `prep_track_data.py` step before it) now live in [tracking_pipeline](https://github.com/cbainbri/tracking_pipeline), shared with food encounter analysis. It keeps the opto-specific columns (shape metrics and `stim`) in the composite.

---

### `bootstrap_opto.py`
PyQt5 GUI for fitting a decay-recovery model to optogenetic response trajectories. Uses bootstrap bagging to estimate parameter distributions.

The model fits the observed trajectory without forcing an asymptotic plateau, using a simplified recovery curve suited to the decay-then-recovery shape of opto responses.

> **Note:** Uses PyQt5 rather than Tkinter to avoid conflicts between Tkinter's event loop and Python's multiprocessing module during parallel bootstrap fitting.

**Input:** `composite.csv` from `tracking_pipeline/merge_files.py`
**Output:** bootstrap fit results and plots

---

### `graph_viewer_opto.py`
Tkinter + Matplotlib GUI for visualizing bootstrap analysis results. Ported from the food-side Richards viewer (v5 UI patterns). Features include:
- Mean ± 95% CI trace plots
- Violin plots with bootstrap CI error bars
- Per-group color customization
- Pre-computed trace cache for instant redraws
- Figure export dialog

**Input:** bootstrap results CSVs from `bootstrap_opto.py`

---

## Installation

```
pip install -r requirements.txt
```

> **Linux users:** Tkinter (used by `merge_files.py` and `graph_viewer_opto.py`) is not pip-installable. Install via `sudo apt-get install python3-tk`.
>
> **PyQt5** is required for `bootstrap_opto.py` and is included in `requirements.txt`.
