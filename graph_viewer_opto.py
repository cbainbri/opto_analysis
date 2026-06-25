#!/usr/bin/env python3
"""
Parameter Graph Viewer v3 — Bootstrap analysis results viewer.
Ported Richards v5 UI patterns:
  - Prism-style rcParams (no top/right spine, Arial, outward ticks)
  - GroupConfig dataclass with per-group color picker
  - "Configure Groups & Display" popup dialog (singleton)
  - Pre-computed trace stat cache rebuilt only on load (instant redraws)
  - matplotlib violinplot with auto-width + KDE bandwidth controls
  - Show/hide legend + x-axis label toggles
  - Figure size W×H with "apply to this tab" / "apply to all tabs"
  - saveasfilename export dialog
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# Prism-style rcParams
# ─────────────────────────────────────────────────────────────────────────────
_PRISM_RC = {
    "figure.facecolor":     "none",
    "axes.facecolor":       "white",  # axes keep white fill; figure bg transparent
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       1.5,
    "axes.labelsize":       12,
    "axes.labelweight":     "normal",
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.width":    1.5,
    "ytick.major.width":    1.5,
    "xtick.minor.width":    1.0,
    "ytick.minor.width":    1.0,
    "xtick.major.size":     5,
    "ytick.major.size":     5,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "lines.linewidth":      2.0,
    "lines.solid_capstyle": "round",
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "legend.frameon":       False,
    "legend.fontsize":      9,
    "axes.grid":            False,
    "pdf.fonttype":         42,
    "svg.fonttype":         "none",
    "savefig.dpi":          300,
    "savefig.facecolor":    "none",   # transparent export by default
    "figure.dpi":           100,
}
plt.rcParams.update(_PRISM_RC)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FIXED_SEX_ORDER       = ["h", "m"]
FIXED_TREATMENT_ORDER = ["fed", "30min", "3hr"]

DEFAULT_PALETTE = [
    "#E8303A", "#2166AC", "#33A02C", "#8B3FA8",
    "#FF7F00", "#1B9EBF", "#F0699B", "#B15928",
    "#56B4E9", "#009E73", "#E69F00", "#CC79A7",
]

SEM_ALPHA = 0.28

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(x, default=np.nan):
    try:    return float(x)
    except: return default


def decay_recovery_trajectory_model(t, y_min, tau_decay, A_rec, tau_rec):
    decay              = y_min + (1.0 - y_min) * np.exp(-t / tau_decay)
    recovery_component = A_rec * (1 - np.exp(-t / tau_rec))
    transition_factor  = 5.0
    decay_weight       = np.exp(-t / (transition_factor * tau_decay))
    return decay_weight * decay + (1 - decay_weight) * (y_min + recovery_component)


def compute_trace_stats(df: pd.DataFrame,
                        smooth_sigma: float = 0.0) -> Dict[Tuple, Tuple]:
    """
    Pre-compute (t, mu, lo, hi) for every (sex, genotype, treatment) group.
    Uses speed_norm column (normalized traces from bootstrap script).
    smooth_sigma > 0 applies Gaussian smoothing to mu/lo/hi (reduces jaggedness).
    Rebuilt only at load time — reads from cache on every redraw.
    """
    cache = {}
    if df is None or df.empty:
        return cache

    work = df.copy()
    work["time_rel"]   = pd.to_numeric(work["time_rel"],   errors="coerce")
    work["speed_norm"] = pd.to_numeric(work["speed_norm"], errors="coerce")
    work = work.dropna(subset=["time_rel", "speed_norm"])

    group_cols = [c for c in ["sex", "genotype", "treatment"] if c in work.columns]
    if not group_cols:
        return cache

    for keys, sub in work.groupby(group_cols):
        if len(group_cols) == 3:
            sex, gen, trt = (str(k) for k in keys)
        else:
            # fallback if some columns missing
            sex, gen, trt = str(keys), "", ""

        agg = sub.groupby("time_rel")["speed_norm"].agg(["mean", "sem"]).sort_index()
        t   = agg.index.values
        mu  = agg["mean"].values
        lo  = mu - np.nan_to_num(agg["sem"].values)
        hi  = mu + np.nan_to_num(agg["sem"].values)
        if smooth_sigma > 0:
            mu = gaussian_filter1d(mu, sigma=smooth_sigma)
            lo = gaussian_filter1d(lo, sigma=smooth_sigma)
            hi = gaussian_filter1d(hi, sigma=smooth_sigma)
        cache[(sex, gen, trt)] = (t, mu, lo, hi)

    return cache


def _postprocess_svg(svg_bytes: bytes) -> bytes:
    """
    Post-process a matplotlib SVG for clean PowerPoint / LibreOffice import:
      1. Remove patch_1 / patch_2  (figure + axes background rects) so the
         background is truly transparent.
      2. Clear any residual white fills left by transparent=True being imperfect.
      3. Unwrap <g id="figure_1"> — hoist its children directly under <svg>.
         This allows PowerPoint "Convert to Shape" / LibreOffice "Ungroup" to
         decompose the import into individual editable elements.
    """
    import xml.etree.ElementTree as ET
    import io as _io

    SVG_NS  = 'http://www.w3.org/2000/svg'
    # Register all namespaces so they survive the round-trip
    ET.register_namespace('',       SVG_NS)
    ET.register_namespace('xlink',  'http://www.w3.org/1999/xlink')
    ET.register_namespace('rdf',    'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
    ET.register_namespace('dc',     'http://purl.org/dc/elements/1.1/')
    ET.register_namespace('cc',     'http://creativecommons.org/ns#')

    root = ET.fromstring(svg_bytes)

    # 1. Remove background patch groups
    def _rm_bg(parent):
        for child in list(parent):
            if child.get('id', '') in ('patch_1', 'patch_2'):
                parent.remove(child)
            else:
                _rm_bg(child)
    _rm_bg(root)

    # 2. Clear residual white fills
    for el in root.iter():
        s = el.get('style', '')
        if s:
            for white in ('fill: #ffffff', 'fill:#ffffff',
                          'fill: white',   'fill:white'):
                s = s.replace(white, 'fill: none')
            el.set('style', s)

    # 3. Unwrap figure_1 group
    for child in list(root):
        if child.get('id') == 'figure_1':
            idx = list(root).index(child)
            root.remove(child)
            for i, grandchild in enumerate(child):
                root.insert(idx + i, grandchild)
            break

    buf = _io.BytesIO()
    ET.ElementTree(root).write(buf, xml_declaration=True, encoding='utf-8')
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
_PICKER_SWATCHES = [
    # Row 1 — reds / oranges / yellows
    "#FF0000","#CC0000","#990000","#FF6600","#FF9900","#FFCC00","#FFFF00","#CCCC00",
    # Row 2 — greens
    "#00FF00","#00CC00","#009900","#006600","#33A02C","#00FF99","#00CCAA","#009966",
    # Row 3 — blues / cyans
    "#0000FF","#0000CC","#000099","#2166AC","#1B9EBF","#00CCFF","#00FFFF","#0099CC",
    # Row 4 — purples / pinks
    "#FF00FF","#CC00CC","#990099","#8B3FA8","#CC79A7","#F0699B","#FF6699","#FF99CC",
    # Row 5 — browns / oranges
    "#663300","#996633","#B15928","#CC9966","#FF7F00","#FFAA00","#E69F00","#FFCC66",
    # Row 6 — greys / black / white
    "#000000","#333333","#666666","#999999","#BBBBBB","#CCCCCC","#DDDDDD","#FFFFFF",
    # Row 7 — palette defaults
    "#E8303A","#2166AC","#33A02C","#8B3FA8","#FF7F00","#1B9EBF","#F0699B","#B15928",
    "#56B4E9","#009E73","#E69F00","#CC79A7","#888888","#444444","#222222","#EEEEEE",
    # Sampled from your data traces
    "#DB7373","#DD4848","#A00000","#C0392B","#E74C3C","#922B21","#7B241C","#641E16",
]

def _ask_color(parent, initial_color: str = "#888888", title: str = "Choose colour") -> str | None:
    """
    Simple Tkinter-native color chooser.
    Returns the chosen hex string, or None if cancelled.
    """
    result = [None]

    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.resizable(False, False)
    dlg.grab_set()

    COLS = 8
    SW   = 28   # swatch size px

    # ── Swatch grid ──────────────────────────────────────────────────────
    grid = ttk.Frame(dlg, padding=8)
    grid.pack()

    current_hex = [initial_color.upper()]

    preview_var = tk.StringVar(value=current_hex[0])

    def _pick(hex_col):
        current_hex[0] = hex_col.upper()
        preview_var.set(current_hex[0])
        hex_e.delete(0, tk.END)
        hex_e.insert(0, current_hex[0])
        preview_lbl.config(bg=current_hex[0])

    for idx, col in enumerate(_PICKER_SWATCHES):
        r, c = divmod(idx, COLS)
        btn = tk.Button(grid, bg=col, width=2, relief="flat", bd=1,
                        activebackground=col,
                        command=lambda h=col: _pick(h))
        btn.grid(row=r, column=c, padx=1, pady=1, ipadx=2, ipady=4)

    # ── Hex entry + preview ───────────────────────────────────────────────
    bot = ttk.Frame(dlg, padding=(8, 0, 8, 8))
    bot.pack(fill=tk.X)

    ttk.Label(bot, text="Hex:").pack(side=tk.LEFT)
    hex_e = ttk.Entry(bot, width=9)
    hex_e.insert(0, current_hex[0])
    hex_e.pack(side=tk.LEFT, padx=(2, 6))

    preview_lbl = tk.Label(bot, bg=initial_color, width=4, relief="sunken", bd=2)
    preview_lbl.pack(side=tk.LEFT, padx=(0, 6))

    def _hex_changed(*_):
        val = hex_e.get().strip()
        if not val.startswith("#"):
            val = "#" + val
        try:
            dlg.winfo_rgb(val)   # raises if invalid
            current_hex[0] = val.upper()
            preview_lbl.config(bg=current_hex[0])
        except Exception:
            pass

    hex_e.bind("<KeyRelease>", _hex_changed)

    def _ok():
        result[0] = current_hex[0]
        dlg.destroy()

    def _cancel():
        dlg.destroy()

    ttk.Button(bot, text="OK",     command=_ok).pack(side=tk.RIGHT, padx=(4, 0))
    ttk.Button(bot, text="Cancel", command=_cancel).pack(side=tk.RIGHT)

    dlg.wait_window()
    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GroupConfig:
    sex:       str
    genotype:  str
    treatment: str
    visible:   bool = True
    color:     str  = "#888888"

    @property
    def key(self):
        return (self.sex, self.genotype, self.treatment)

    @property
    def label(self):
        return f"{self.sex.upper()} : {self.genotype} : {self.treatment}"


# ─────────────────────────────────────────────────────────────────────────────
# Parameter definitions
# ─────────────────────────────────────────────────────────────────────────────
PARAMETERS = {
    'y_min_actual':           ('Actual Minimum (trajectory)',       'fraction of baseline'),
    't_at_min':               ('Time at Minimum',                   'seconds'),
    't_half_decay_actual':    ('Decay t½ (actual trajectory)',      'seconds'),
    't_half_recovery_actual': ('Recovery t½ (actual trajectory)',   'seconds'),
    'trough_width_50':        ('Trough Width FWHM (below 50%)',     'seconds'),
    'trough_width_75':        ('Trough Width (below 75%)',          'seconds'),
    'trough_width_25':        ('Trough Width (below 25%)',          'seconds'),
    't_enter_50':             ('Time Entering 50% Threshold',       'seconds'),
    't_exit_50':              ('Time Exiting 50% Threshold',        'seconds'),
    'drop_depth':             ('Minimum Speed',                     '% of baseline'),
    'recovery_at_end':        ('Speed at 60s',                      'fraction of baseline'),
    'recovery_completeness':  ('Recovery Completeness',             'fraction (0-1)'),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────
class ParameterViewer:
    _DEFAULT_MIN_SPACING = 0.8   # inches of plot-area per violin before auto-expand kicks in

    def __init__(self, root: tk.Tk):
        self._in_refresh = False
        self.root  = root
        self.root.title("Parameter Graph Viewer v3")

        # Per-tab Y-axis state — (auto:bool, ymin:str, ymax:str)
        self._tab_ylim_state = [(True, "0", "1.2"), (True, "0", "1.2")]
        self._active_tab_idx = 0
        self.root.geometry("1400x900")

        # ── Data ──────────────────────────────────────────────────────────
        self.bootstrap_data: Optional[pd.DataFrame] = None
        self.trace_data:     Optional[pd.DataFrame] = None
        self._trace_cache:   Dict[Tuple, Tuple]     = {}
        self._loading    = False
        self._in_refresh = False

        # ── Group registry ────────────────────────────────────────────────
        self.groups: Dict[Tuple, GroupConfig] = {}

        # ── Shared Tk vars ────────────────────────────────────────────────
        self.param_var      = tk.StringVar(value='t_half_decay_actual')
        self.auto_ylim      = tk.BooleanVar(value=True)
        self.show_xlabel    = tk.BooleanVar(value=True)
        self.show_legend    = tk.BooleanVar(value=True)
        self.show_overlay   = tk.BooleanVar(value=False)
        self.show_points    = tk.BooleanVar(value=False)
        self.show_ci_sem    = tk.BooleanVar(value=False)   # False = 95% CI, True = SEM
        self.smooth_sigma   = tk.StringVar(value="0")

        # Per-tab titles, font sizes, tick spacing
        self.tab_titles    = [tk.StringVar() for _ in range(2)]
        self.font_auto     = tk.BooleanVar(value=True)
        self.font_title_sz = tk.StringVar(value="13")
        self.font_label_sz = tk.StringVar(value="12")
        self.font_tick_sz  = tk.StringVar(value="10")
        self.ytick_step    = [tk.StringVar(value="")    for _ in range(2)]
        self.xtick_step    = [tk.StringVar(value="")    for _ in range(2)]
        # Y-axis decimal places: "auto" or integer 0–4
        self.ydecimal_n    = [tk.StringVar(value="")    for _ in range(2)]
        self.violin_auto_w  = tk.BooleanVar(value=True)
        self.violin_w       = tk.StringVar(value="0.5")
        self.violin_auto_bw = tk.BooleanVar(value=False)
        self.violin_bw      = tk.StringVar(value="0.3")
        self.min_spacing    = tk.StringVar(value=str(self._DEFAULT_MIN_SPACING))

        self._cfg_win = None   # singleton Configure dialog reference

        self._build_ui()
        self.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=240)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        left.pack_propagate(False)
        self._build_left(left)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._build_tabs(right)

    def _build_left(self, p):
        # Load
        ttk.Button(p, text="Load Data Directory",
                   command=self.load_data).pack(fill=tk.X, pady=(0, 4))

        # File status
        sf = ttk.LabelFrame(p, text="Loaded files", padding=4)
        sf.pack(fill=tk.X, pady=(0, 6))
        self.lbl_params = ttk.Label(sf, text="Bootstrap params:  ✗", foreground="gray")
        self.lbl_traces = ttk.Label(sf, text="Normalized traces: ✗", foreground="gray")
        self.lbl_params.pack(anchor="w")
        self.lbl_traces.pack(anchor="w")

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, pady=6)

        # Configure dialog
        ttk.Button(p, text="⚙  Configure Groups & Display",
                   command=self._open_configure).pack(fill=tk.X, pady=(0, 6))

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, pady=6)

        # Y-axis
        yf = ttk.LabelFrame(p, text="Y-axis  (active tab)", padding=4)
        yf.pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(yf, text="Auto", variable=self.auto_ylim,
                        command=self._on_ylim_change).pack(anchor="w")
        row = ttk.Frame(yf); row.pack(fill=tk.X)
        ttk.Label(row, text="Min").pack(side=tk.LEFT)
        self.ymin_e = ttk.Entry(row, width=7)
        self.ymin_e.insert(0, "0"); self.ymin_e.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="Max").pack(side=tk.LEFT)
        self.ymax_e = ttk.Entry(row, width=7)
        self.ymax_e.insert(0, "1.2"); self.ymax_e.pack(side=tk.LEFT, padx=2)
        for w in (self.ymin_e, self.ymax_e):
            w.bind("<KeyRelease>", lambda _: self._on_ylim_change())
            w.bind("<FocusOut>",   lambda _: self._on_ylim_change())

        # Display toggles
        df_ = ttk.LabelFrame(p, text="Display", padding=4)
        df_.pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(df_, text="X-axis labels", variable=self.show_xlabel,
                        command=self.refresh).pack(anchor="w")
        ttk.Checkbutton(df_, text="Legend",        variable=self.show_legend,
                        command=self.refresh).pack(anchor="w")

        # Figure size
        ff = ttk.LabelFrame(p, text="Figure size  (inches)", padding=4)
        ff.pack(fill=tk.X, pady=(0, 4))
        self.fig_tab_lbl = ttk.Label(ff, text="", font=("", 8), foreground="gray")
        self.fig_tab_lbl.pack(anchor="w")
        row = ttk.Frame(ff); row.pack(fill=tk.X)
        ttk.Label(row, text="W").pack(side=tk.LEFT)
        self.fig_w = ttk.Entry(row, width=6); self.fig_w.insert(0, "10")
        self.fig_w.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="H").pack(side=tk.LEFT)
        self.fig_h = ttk.Entry(row, width=6); self.fig_h.insert(0, "6")
        self.fig_h.pack(side=tk.LEFT, padx=2)
        ttk.Button(ff, text="Apply to this tab",
                   command=self._apply_fig_size).pack(fill=tk.X, pady=(2, 0))
        ttk.Button(ff, text="Apply to all tabs",
                   command=self._apply_fig_size_all).pack(fill=tk.X, pady=(2, 0))

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, pady=6)

        ttk.Button(p, text="Export figure  (PNG / SVG / PDF)",
                   command=self._export_figure).pack(fill=tk.X, pady=(0, 10))

    def _build_tabs(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_params = ttk.Frame(self.nb)
        self.tab_traces = ttk.Frame(self.nb)
        self.nb.add(self.tab_params, text="Parameter Distributions")
        self.nb.add(self.tab_traces, text="Speed Traces")
        self.nb.bind("<<NotebookTabChanged>>", lambda _: self._on_tab_change())

        VIOLIN_MARGINS = dict(left=0.15, right=0.97, top=0.93, bottom=0.32)
        TRACE_MARGINS  = dict(left=0.15, right=0.97, top=0.92, bottom=0.12)
        WORKSPACE_BG   = "#ffffff"   # white workspace behind transparent figures

        self._fig_margins   = {}   # fig → margin dict
        self._fig_canvas    = {}   # fig → FigureCanvasTkAgg
        self._fig_workspace = {}   # fig → tk.Canvas (the white workspace)
        self._fig_win_id    = {}   # fig → canvas window item id

        def _make_tab(tab, margins, default_w=10, default_h=6):
            # ── White workspace canvas fills the tab ──────────────────
            workspace = tk.Canvas(tab, bg=WORKSPACE_BG, highlightthickness=0)
            workspace.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

            # ── Matplotlib figure with transparent background ─────────
            fig = Figure(figsize=(default_w, default_h), facecolor="none")
            ax  = fig.add_subplot(111)
            ax.set_facecolor("none")
            fig.subplots_adjust(**margins)

            # Place matplotlib canvas inside the workspace (not pack — place via window)
            mpl_canvas = FigureCanvasTkAgg(fig, master=workspace)
            widget     = mpl_canvas.get_tk_widget()
            dpi = fig.dpi
            px_w, px_h = int(default_w * dpi), int(default_h * dpi)
            widget.config(width=px_w, height=px_h, bg=WORKSPACE_BG)
            # Create a centered window item on the workspace canvas
            win_id = workspace.create_window(0, 0, window=widget, anchor="nw")
            # Center the figure when the workspace is resized
            def _center(event, ws=workspace, wid=win_id, fw=px_w, fh=px_h):
                x = max(0, (event.width  - fw) // 2)
                y = max(0, (event.height - fh) // 2)
                ws.coords(wid, x, y)
            workspace.bind("<Configure>", _center)

            # Toolbar below the workspace
            toolbar_frame = ttk.Frame(tab)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            NavigationToolbar2Tk(mpl_canvas, toolbar_frame)

            self._fig_margins[fig]   = margins
            self._fig_canvas[fig]    = mpl_canvas
            self._fig_workspace[fig] = workspace
            self._fig_win_id[fig]    = win_id
            return fig, ax, mpl_canvas

        self.fig_params, self.ax_params, self.canvas_params = _make_tab(
            self.tab_params, VIOLIN_MARGINS)
        self.fig_traces, self.ax_traces, self.canvas_traces = _make_tab(
            self.tab_traces, TRACE_MARGINS)

        for ax, msg in [(self.ax_params, "Load data to begin"),
                        (self.ax_traces, "Load trace data to begin")]:
            ax.text(0.5, 0.5, msg, ha='center', va='center',
                    fontsize=16, color='gray', transform=ax.transAxes)

    # ─────────────────────────────────────────────────────────────────────────
    # Configure Groups & Display dialog
    # ─────────────────────────────────────────────────────────────────────────
    def _open_configure(self):
        if self._cfg_win is not None:
            self._cfg_win.lift()
            self._cfg_win.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Configure Groups & Display")
        win.resizable(True, True)
        win.geometry("420x700")
        self._cfg_win = win
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_configure(win))

        # Scrollable interior
        outer  = ttk.Frame(win); outer.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb    = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner  = ttk.Frame(canvas)
        wid    = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(wid, width=e.width))

        def _wheel(e):
            try:
                if not canvas.winfo_exists(): return
                if   e.num == 4: canvas.yview_scroll(-1, "units")
                elif e.num == 5: canvas.yview_scroll( 1, "units")
                else:            canvas.yview_scroll(int(-1 * e.delta / 120), "units")
            except Exception:
                pass

        def _bind_wheel(_e=None):
            if canvas.winfo_exists():
                canvas.bind_all("<MouseWheel>", _wheel)
                canvas.bind_all("<Button-4>",   _wheel)
                canvas.bind_all("<Button-5>",   _wheel)

        def _unbind_wheel(_e=None):
            try:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            except Exception:
                pass

        canvas.bind("<Enter>",  _bind_wheel)
        canvas.bind("<Leave>",  _unbind_wheel)
        win.bind("<Destroy>",   lambda _: _unbind_wheel())

        p = inner

        # ── Groups ────────────────────────────────────────────────────────
        hdr = ttk.Frame(p); hdr.pack(fill=tk.X, padx=8, pady=(8, 2))
        ttk.Label(hdr, text="Groups", font=("", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(hdr, text="None", width=5,
                   command=lambda: self._select_all_groups(False)).pack(side=tk.RIGHT)
        ttk.Button(hdr, text="All",  width=4,
                   command=lambda: self._select_all_groups(True)).pack(side=tk.RIGHT, padx=(0, 2))
        ttk.Label(p, text="  ☑ = visible   ■ = colour",
                  font=("", 8), foreground="gray").pack(anchor="w", padx=8)

        self._cfg_group_frame = ttk.Frame(p)
        self._cfg_group_frame.pack(fill=tk.X, padx=8, pady=(2, 6))
        self._rebuild_cfg_groups()

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Parameter selection ───────────────────────────────────────────
        pf = ttk.LabelFrame(p, text="Parameter  (violin tab)", padding=6)
        pf.pack(fill=tk.X, padx=8, pady=(0, 6))

        avail = [k for k in PARAMETERS if
                 self.bootstrap_data is not None and k in self.bootstrap_data.columns]
        if not avail:
            avail = list(PARAMETERS.keys())

        param_cb = ttk.Combobox(pf, textvariable=self.param_var,
                                state="readonly", values=avail)
        param_cb.pack(fill=tk.X)
        param_cb.bind("<<ComboboxSelected>>", lambda _: self.refresh())

        ttk.Checkbutton(pf, text="Show individual points (10% sample)",
                        variable=self.show_points, command=self.refresh).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(pf, text="Show SEM instead of 95% CI",
                        variable=self.show_ci_sem, command=self.refresh).pack(anchor="w")
        row = ttk.Frame(p); row.pack(fill=tk.X, pady=(4,0))
        ttk.Label(row, text="Ribbon smoothing σ:", font=("",8)).pack(side=tk.LEFT)
        sm_e = ttk.Entry(row, textvariable=self.smooth_sigma, width=5)
        sm_e.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(0 = off)", font=("",7), foreground="gray").pack(side=tk.LEFT)
        for ev in ("<KeyRelease>","<FocusOut>","<Return>"):
            sm_e.bind(ev, lambda _: self._on_smooth_change())

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Trace options ─────────────────────────────────────────────────
        tf = ttk.LabelFrame(p, text="Speed Traces", padding=6)
        tf.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Checkbutton(tf, text="Overlay fit curves (median params)",
                        variable=self.show_overlay, command=self.refresh).pack(anchor="w")

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Violin options ────────────────────────────────────────────────
        vf = ttk.LabelFrame(p, text="Violin options", padding=6)
        vf.pack(fill=tk.X, padx=8, pady=(0, 6))

        ttk.Label(vf, text="Violin width:").pack(anchor="w")
        ttk.Checkbutton(vf, text="Auto", variable=self.violin_auto_w,
                        command=self.refresh).pack(anchor="w")
        row = ttk.Frame(vf); row.pack(fill=tk.X)
        ttk.Label(row, text="Scale (0.1–1.5)").pack(side=tk.LEFT)
        ve = ttk.Entry(row, textvariable=self.violin_w, width=6)
        ve.pack(side=tk.LEFT, padx=2)
        for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
            ve.bind(ev, lambda _: self.refresh())

        ttk.Label(vf, text="KDE bandwidth:", font=("", 8)).pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(vf, text="Auto (Scott's rule)", variable=self.violin_auto_bw,
                        command=self.refresh).pack(anchor="w")
        row = ttk.Frame(vf); row.pack(fill=tk.X)
        ttk.Label(row, text="BW").pack(side=tk.LEFT)
        bwe = ttk.Entry(row, textvariable=self.violin_bw, width=6)
        bwe.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(0.2 tight – 0.6 smooth)").pack(side=tk.LEFT)
        for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
            bwe.bind(ev, lambda _: self.refresh())

        ttk.Label(vf, text="Min spacing (in/violin):", font=("", 8)).pack(anchor="w", pady=(6, 0))
        row = ttk.Frame(vf); row.pack(fill=tk.X)
        mse = ttk.Entry(row, textvariable=self.min_spacing, width=6)
        mse.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(figure expands only if crowded)").pack(side=tk.LEFT)
        for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
            mse.bind(ev, lambda _: self.refresh())

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Tick Spacing ──────────────────────────────────────────────────
        # Enter a value to use it; leave blank (or 0) for matplotlib auto.
        tf = ttk.LabelFrame(p, text="Tick Spacing  (per tab)", padding=6)
        tf.pack(fill=tk.X, padx=8, pady=(0, 6))

        for col, lbl in enumerate(["Tab", "Y step", "X step", "Y decimals"]):
            ttk.Label(tf, text=lbl, font=("", 8, "bold"),
                      width=16 if col == 0 else 8).grid(
                row=0, column=col, sticky="w", padx=(0 if col == 0 else 4, 0))
        ttk.Label(tf, text="(blank/0 = auto)", font=("", 7),
                  foreground="gray").grid(row=0, column=4, sticky="w", padx=4)

        for i, tab_name in enumerate(self._TAB_NAMES):
            r = i + 1
            ttk.Label(tf, text=tab_name, font=("", 8), foreground="gray",
                      width=16).grid(row=r, column=0, sticky="w", pady=2)

            ys_e = ttk.Entry(tf, textvariable=self.ytick_step[i], width=7)
            ys_e.grid(row=r, column=1, padx=4)
            for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
                ys_e.bind(ev, lambda _: self.refresh())

            xs_e = ttk.Entry(tf, textvariable=self.xtick_step[i], width=7)
            xs_e.grid(row=r, column=2, padx=4)
            for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
                xs_e.bind(ev, lambda _: self.refresh())

            yn_e = ttk.Entry(tf, textvariable=self.ydecimal_n[i], width=4)
            yn_e.grid(row=r, column=3, padx=4)
            for ev in ("<KeyRelease>", "<FocusOut>", "<Return>"):
                yn_e.bind(ev, lambda _: self.refresh())

        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Labels & Fonts ────────────────────────────────────────────────
        lf = ttk.LabelFrame(p, text="Labels & Fonts", padding=6)
        lf.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(lf, text="Plot titles  (blank = auto):", font=("",8,"bold")).pack(anchor="w")
        tab_defaults = ["Parameter Distributions", "Speed Traces"]
        for i, (var, dflt) in enumerate(zip(self.tab_titles, tab_defaults)):
            row = ttk.Frame(lf); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{self._TAB_NAMES[i]}:", width=22,
                      font=("",8), foreground="gray").pack(side=tk.LEFT)
            e = ttk.Entry(row, textvariable=var); e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,0))
            for ev in ("<KeyRelease>","<FocusOut>","<Return>"): e.bind(ev, lambda _: self.refresh())
        ttk.Separator(lf, orient="horizontal").pack(fill=tk.X, pady=(6,4))
        ttk.Label(lf, text="Font sizes:", font=("",8,"bold")).pack(anchor="w")
        ttk.Checkbutton(lf, text="Auto-scale with figure height",
                        variable=self.font_auto, command=self.refresh).pack(anchor="w")
        ttk.Label(lf, text="(title≈h×2.1  label≈h×1.8  tick≈h×1.5)",
                  font=("",7), foreground="gray").pack(anchor="w")
        fsz = ttk.Frame(lf); fsz.pack(fill=tk.X, pady=(4,0))
        for col, (lbl, var) in enumerate([("Title", self.font_title_sz),
                                           ("Axis lbl", self.font_label_sz),
                                           ("Tick", self.font_tick_sz)]):
            ttk.Label(fsz, text=lbl, font=("",8)).grid(row=0, column=col*2, padx=(4,0), sticky="w")
            e = ttk.Entry(fsz, textvariable=var, width=4)
            e.grid(row=0, column=col*2+1, padx=(2,4))
            for ev in ("<KeyRelease>","<FocusOut>","<Return>"): e.bind(ev, lambda _: self.refresh())

        ttk.Button(p, text="Close",
                   command=lambda: self._close_configure(win)).pack(pady=(4, 10))

    def _close_configure(self, win):
        win.destroy()
        self._cfg_win = None

    def _rebuild_cfg_groups(self):
        if not hasattr(self, "_cfg_group_frame"):
            return
        for w in self._cfg_group_frame.winfo_children():
            w.destroy()

        self._cfg_group_vars = {}
        self._cfg_group_btns = {}

        # Order: sex → treatment (no fixed genotype order — depends on data)
        ordered_keys = []
        for sex in FIXED_SEX_ORDER:
            for trt in FIXED_TREATMENT_ORDER:
                for key, gc in self.groups.items():
                    if gc.sex == sex and gc.treatment == trt and key not in ordered_keys:
                        ordered_keys.append(key)
        # Catch any remaining
        for key in self.groups:
            if key not in ordered_keys:
                ordered_keys.append(key)

        for key in ordered_keys:
            gc  = self.groups[key]
            var = tk.BooleanVar(value=gc.visible)
            self._cfg_group_vars[key] = var

            row = ttk.Frame(self._cfg_group_frame)
            row.pack(fill=tk.X, pady=1)

            def _toggle(k=key, v=var):
                self.groups[k].visible = v.get()
                self.refresh()

            ttk.Checkbutton(row, text=gc.label, variable=var,
                            command=_toggle).pack(side=tk.LEFT, fill=tk.X, expand=True)
            btn = tk.Button(row, bg=gc.color, width=3, relief="raised",
                            command=lambda k=key: self._pick_color(k))
            btn.pack(side=tk.RIGHT, padx=(2, 0))
            self._cfg_group_btns[key] = btn

    def _pick_color(self, key):
        gc     = self.groups[key]
        chosen = _ask_color(self.root, initial_color=gc.color,
                            title=f"Colour — {gc.label}")
        if chosen:
            gc.color = chosen
            if key in self._cfg_group_btns:
                self._cfg_group_btns[key].configure(bg=gc.color)
            self.refresh()

    def _select_all_groups(self, state: bool):
        for key, gc in self.groups.items():
            gc.visible = state
            if key in getattr(self, "_cfg_group_vars", {}):
                self._cfg_group_vars[key].set(state)
        self.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # Group registry
    # ─────────────────────────────────────────────────────────────────────────
    def _populate_groups(self, df: pd.DataFrame):
        needed = {"sex", "genotype", "treatment"}
        if not needed.issubset(df.columns):
            return
        changed = False
        for _, row in df[["sex", "genotype", "treatment"]].drop_duplicates().iterrows():
            sex, gen, trt = str(row["sex"]), str(row["genotype"]), str(row["treatment"])
            key = (sex, gen, trt)
            if key not in self.groups:
                idx = len(self.groups) % len(DEFAULT_PALETTE)
                self.groups[key] = GroupConfig(sex=sex, genotype=gen, treatment=trt,
                                               visible=True, color=DEFAULT_PALETTE[idx])
                changed = True
        if changed:
            self._rebuild_cfg_groups()

    def _visible_groups(self) -> List[GroupConfig]:
        out = []
        for sex in FIXED_SEX_ORDER:
            for trt in FIXED_TREATMENT_ORDER:
                for key, gc in self.groups.items():
                    if gc.sex == sex and gc.treatment == trt and gc.visible:
                        out.append(gc)
        # catch any not matched by the fixed orders
        seen = {gc.key for gc in out}
        for gc in self.groups.values():
            if gc.key not in seen and gc.visible:
                out.append(gc)
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────────
    def load_data(self):
        directory = filedialog.askdirectory(title="Select Bootstrap Output Directory")
        if not directory:
            return

        self._loading = True
        try:
            d = Path(directory)

            # ── Bootstrap parameter CSVs ──────────────────────────────────
            param_files = list(d.glob("*_bootstrap_params.csv"))
            if not param_files:
                messagebox.showerror("Error", "No *_bootstrap_params.csv files found")
                return

            dfs = []
            for f in param_files:
                df = pd.read_csv(f)
                # Filename format: treatment__sex__genotype_bootstrap_params.csv
                stem = f.stem.replace('_bootstrap_params', '')
                parts = stem.split('__')
                if len(parts) >= 3:
                    df['treatment'] = parts[0]
                    df['sex']       = parts[1]
                    df['genotype']  = parts[2]
                elif 'sex' not in df.columns:
                    df['treatment'] = stem
                    df['sex']       = 'unknown'
                    df['genotype']  = 'unknown'
                dfs.append(df)

            self.bootstrap_data = pd.concat(dfs, ignore_index=True)
            self._populate_groups(self.bootstrap_data)

            # ── Normalized trace CSVs ─────────────────────────────────────
            trace_files = list(d.glob("*_normalized_traces.csv"))
            if trace_files:
                tdfs = []
                for f in trace_files:
                    df = pd.read_csv(f)
                    # Ensure genotype column present
                    if 'genotype' not in df.columns and 'sex' in df.columns:
                        stem = f.stem.replace('_normalized_traces', '')
                        parts = stem.split('__')
                        df['genotype'] = parts[2] if len(parts) >= 3 else 'unknown'
                    tdfs.append(df)
                self.trace_data = pd.concat(tdfs, ignore_index=True)
                self._populate_groups(self.trace_data)
                self._trace_cache = compute_trace_stats(self.trace_data, smooth_sigma=self._get_smooth_sigma())
            else:
                self.trace_data   = None
                self._trace_cache = {}

            self._update_status()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e}")
        finally:
            self._loading = False

        self.root.after(100, self.refresh)
        if self.bootstrap_data is not None or self.trace_data is not None:
            messagebox.showinfo("Loaded",
                f"Loaded {len(param_files) if 'param_files' in dir() else 0} parameter file(s)")

    def _update_status(self):
        def _s(lbl, ok, yes, no):
            lbl.config(text=yes if ok else no, foreground="black" if ok else "gray")
        _s(self.lbl_params, self.bootstrap_data is not None,
           "Bootstrap params:  ✓", "Bootstrap params:  ✗")
        _s(self.lbl_traces, self.trace_data is not None,
           "Normalized traces: ✓", "Normalized traces: ✗")

    # ─────────────────────────────────────────────────────────────────────────
    # Violin helpers (ported from Richards v5)
    # ─────────────────────────────────────────────────────────────────────────
    def _get_violin_width(self):
        if self.violin_auto_w.get():
            return 0.80
        return float(np.clip(safe_float(self.violin_w.get(), 0.5), 0.05, 1.5))

    def _get_violin_bw(self):
        if self.violin_auto_bw.get():
            return "scott"
        return float(np.clip(safe_float(self.violin_bw.get(), 0.3), 0.05, 2.0))

    def _ideal_violin_fig_width(self, n: int, vw: float) -> float:
        plot_area = n * self._DEFAULT_MIN_SPACING * max(vw, 0.5)
        return max(plot_area / 0.87, 6.0)

    def _auto_size_violin_fig(self, fig, n: int, vw: float):
        """Only expand figure if current width would crowd violins below min spacing.
        If the figure is already wide enough, leave it alone — preserving user-set size."""
        min_spacing  = float(np.clip(safe_float(self.min_spacing.get(),
                                                self._DEFAULT_MIN_SPACING), 0.1, 10.0))
        min_w_needed = (n * min_spacing) / 0.87   # 0.87 ≈ usable fraction (left+right margins)
        current_w    = fig.get_figwidth()
        if current_w < min_w_needed:
            self._resize_fig(fig, min_w_needed, fig.get_figheight())

    def _draw_violin(self, ax, fig, data_list, labels, colors):
        n  = len(data_list)
        vw = self._get_violin_width()
        self._auto_size_violin_fig(fig, n, vw)

        parts = ax.violinplot(data_list, showmeans=False, showextrema=False,
                              widths=vw, bw_method=self._get_violin_bw())
        for i, col in enumerate(colors):
            parts["bodies"][i].set_facecolor(col)
            parts["bodies"][i].set_alpha(0.80)
            parts["bodies"][i].set_edgecolor("none")

        ax.set_xlim(0.5, n + 0.5)
        xs     = list(range(1, n + 1))
        means  = [np.mean(d) for d in data_list]
        # Bootstrap 95% CI: 2.5/97.5 percentiles.
        # Do NOT divide by sqrt(N) — replicates are not independent.
        lo_err = [means[i] - np.percentile(d, 2.5)  for i, d in enumerate(data_list)]
        hi_err = [np.percentile(d, 97.5) - means[i] for i, d in enumerate(data_list)]
        ax.errorbar(xs, means, yerr=[lo_err, hi_err], fmt="o", color="black",
                    markersize=6, markerfacecolor="black", markeredgecolor="black",
                    linewidth=1.8, capsize=5, capthick=1.8, zorder=5,
                    label="mean \u00b1 95% CI (bootstrap)")
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")

    # ─────────────────────────────────────────────────────────────────────────
    # Shared plot helpers (ported from Richards v5)
    # ─────────────────────────────────────────────────────────────────────────
    def _get_smooth_sigma(self) -> float:
        try: s = float(self.smooth_sigma.get()); return max(0.0, s)
        except: return 0.0

    def _on_smooth_change(self):
        """Rebuild trace cache with new sigma and refresh."""
        if self.trace_data is not None:
            self._trace_cache = compute_trace_stats(
                self.trace_data, smooth_sigma=self._get_smooth_sigma())
        self.refresh()

    def _save_ylim_state(self):
        i = self._active_tab_idx
        self._tab_ylim_state[i] = (
            self.auto_ylim.get(),
            self.ymin_e.get(),
            self.ymax_e.get(),
        )

    def _load_ylim_state(self, tab_idx: int):
        auto, ymin, ymax = self._tab_ylim_state[tab_idx]
        self.auto_ylim.set(auto)
        self.ymin_e.delete(0, tk.END); self.ymin_e.insert(0, ymin)
        self.ymax_e.delete(0, tk.END); self.ymax_e.insert(0, ymax)

    def _on_ylim_change(self):
        self._save_ylim_state()
        self.refresh()

    def _apply_ylim(self, ax):
        if self.auto_ylim.get():
            return
        lo = safe_float(self.ymin_e.get(), np.nan)
        hi = safe_float(self.ymax_e.get(), np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            ax.set_ylim(lo, hi)

    def _get_font_sizes(self, fig):
        if self.font_auto.get():
            h = fig.get_figheight()
            ts = max(10, round(h * 2.1))
            ls = max(9,  round(h * 1.8))
            ks = max(8,  round(h * 1.5))
            self.font_title_sz.set(str(ts))
            self.font_label_sz.set(str(ls))
            self.font_tick_sz.set(str(ks))
            return ts, ls, ks
        def _i(v, d):
            try: return max(5, int(float(v.get())))
            except: return d
        return _i(self.font_title_sz,13), _i(self.font_label_sz,12), _i(self.font_tick_sz,10)

    def _get_tab_title(self, tab_idx, auto_title):
        ov = self.tab_titles[tab_idx].get().strip()
        return ov if ov else auto_title

    def _apply_ticks(self, ax, tab_idx):
        """Value present and > 0 → use it. Blank or 0 → matplotlib auto.
        When a step is set and decimals blank, infer decimals from step
        so 0.05 step never rounds to duplicate labels like 0.1, 0.1."""
        import matplotlib.ticker as ticker
        import math

        y_step = safe_float(self.ytick_step[tab_idx].get(), 0.0)
        if y_step > 0:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        else:
            ax.yaxis.set_major_locator(ticker.AutoLocator())

        dp_str = self.ydecimal_n[tab_idx].get().strip()
        if dp_str:
            try:
                nd = max(0, int(dp_str))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{nd}f"))
            except (ValueError, TypeError):
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.yaxis.get_major_formatter().set_useOffset(False)
        elif y_step > 0:
            # Auto-infer: 0.05 → 2dp, 0.1 → 1dp, 0.5 → 1dp, 1.0 → 0dp
            nd = max(0, math.ceil(-math.log10(y_step + 1e-12)))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{nd}f"))
        else:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_useOffset(False)

        if tab_idx == 1:
            x_step = safe_float(self.xtick_step[tab_idx].get(), 0.0)
            if x_step > 0:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
            else:
                ax.xaxis.set_major_locator(ticker.AutoLocator())

    def _apply_prism_spines(self, ax):
        ax.set_facecolor("white")   # axes always white over transparent figure
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(1.5)
        ax.tick_params(axis="both", which="major", direction="out", width=1.5, length=5)
        ax.tick_params(axis="both", which="minor", direction="out", width=1.0, length=3)

    def _apply_display(self, ax, fig=None, has_legend=False):
        self._apply_prism_spines(ax)
        if fig is not None:
            ts, ls, ks = self._get_font_sizes(fig)
            ax.tick_params(axis="both", labelsize=ks)
            ax.xaxis.label.set_fontsize(ls)
            ax.yaxis.label.set_fontsize(ls)
            ax.title.set_fontsize(ts)
        if not self.show_xlabel.get():
            ax.set_xticklabels([])
            ax.set_xlabel("")
        if has_legend:
            legend_fs = max(7, (self._get_font_sizes(fig)[2] - 1) if fig else 8)
            if self.show_legend.get():
                labeled = [a for a in ax.get_children()
                           if hasattr(a,"get_label") and not str(a.get_label()).startswith("_")]
                if labeled: ax.legend(fontsize=legend_fs)
            else:
                leg = ax.get_legend()
                if leg: leg.remove()

    # ─────────────────────────────────────────────────────────────────────────
    # Figure size controls
    # ─────────────────────────────────────────────────────────────────────────
    _TAB_NAMES = ["Parameter Distributions", "Speed Traces"]

    def _active_fig(self):
        idx = self.nb.index(self.nb.select())
        return (self.fig_params, self.fig_traces)[idx]

    def _on_tab_change(self):
        self._save_ylim_state()
        new_idx = self.nb.index(self.nb.select())
        self._active_tab_idx = new_idx
        self._load_ylim_state(new_idx)
        fig = self._active_fig()
        w, h = fig.get_size_inches()
        self.fig_w.delete(0, tk.END); self.fig_w.insert(0, f"{w:.1f}")
        self.fig_h.delete(0, tk.END); self.fig_h.insert(0, f"{h:.1f}")
        self.fig_tab_lbl.config(text=self._TAB_NAMES[new_idx])
        self.refresh()

    def _resize_fig(self, fig, w, h):
        """Resize figure (inches) + widget (pixels) + recenter on workspace."""
        dpi = fig.dpi
        fig.set_size_inches(w, h)
        margins = self._fig_margins.get(fig, {})
        if margins:
            fig.subplots_adjust(**margins)
        px_w = int(w * dpi)
        px_h = int(h * dpi)
        canvas    = self._fig_canvas.get(fig)
        workspace = self._fig_workspace.get(fig)
        win_id    = self._fig_win_id.get(fig)
        if canvas is not None:
            widget = canvas.get_tk_widget()
            widget.config(width=px_w, height=px_h)
            if workspace is not None and win_id is not None:
                workspace.bind("<Configure>",
                    lambda e, wid=win_id, fw=px_w, fh=px_h, ws=workspace:
                        ws.coords(wid, max(0,(e.width-fw)//2), max(0,(e.height-fh)//2)))
                def _do_center(ws=workspace, wid=win_id, fw=px_w, fh=px_h):
                    ws_w = ws.winfo_width()
                    ws_h = ws.winfo_height()
                    if ws_w > 1 and ws_h > 1:
                        x = max(0, (ws_w - fw) // 2)
                        y = max(0, (ws_h - fh) // 2)
                        ws.coords(wid, x, y)
                    else:
                        ws.after(50, _do_center)
                _do_center()
            canvas.draw()


    def _square_h(self, fig, w) -> float:
        """Given W, compute H so axes AREA is square (compensates for margins)."""
        m = self._fig_margins.get(fig, {})
        ax_w = m.get("right", 0.97) - m.get("left", 0.10)
        ax_h = m.get("top",   0.93) - m.get("bottom", 0.12)
        return w * ax_w / ax_h if ax_h > 0 else w

    def _resolve_h(self, fig, w, h_entered) -> float:
        """If W == H (within 0.01"), compensate margins for square axes.
        Otherwise use H as entered. Works for both trace and violin tabs."""
        if abs(w - h_entered) < 0.01:
            return self._square_h(fig, w)
        return h_entered


    def _apply_fig_size(self):
        w = safe_float(self.fig_w.get(), 10.0)
        h_raw = safe_float(self.fig_h.get(), 6.0)
        h = self._resolve_h(self._active_fig(), w, h_raw)
        if w > 0 and h > 0:
            self.fig_h.delete(0, tk.END); self.fig_h.insert(0, f"{h:.2f}")
            self._resize_fig(self._active_fig(), w, h)
            self.refresh()

    def _apply_fig_size_all(self):
        w = safe_float(self.fig_w.get(), 10.0)
        h_raw = safe_float(self.fig_h.get(), 6.0)
        if w > 0:
            for fig in (self.fig_params, self.fig_traces):
                h = self._resolve_h(fig, w, h_raw)
                self._resize_fig(fig, w, h)
            h_active = self._resolve_h(self._active_fig(), w, h_raw)
            self.fig_h.delete(0, tk.END); self.fig_h.insert(0, f"{h_active:.2f}")
            self.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # Plot: Parameter violin
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_params(self):
        ax = self.ax_params
        ax.clear()

        if self.bootstrap_data is None:
            ax.text(0.5, 0.5, "Load data to begin",
                    ha="center", va="center", fontsize=16, color="gray",
                    transform=ax.transAxes)
            self.canvas_params.draw_idle()
            return

        param = self.param_var.get()
        if param not in self.bootstrap_data.columns:
            ax.text(0.5, 0.5, f"Column '{param}' not found in data",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas_params.draw_idle()
            return

        visible = self._visible_groups()
        data_list, labels, colors = [], [], []

        for gc in visible:
            df = self.bootstrap_data
            mask = ((df["sex"]       == gc.sex) &
                    (df["genotype"]  == gc.genotype) &
                    (df["treatment"] == gc.treatment))
            vals = df[mask][param].dropna().values
            if len(vals) == 0:
                continue
            data_list.append(vals)
            labels.append(gc.label)
            colors.append(gc.color)

        if not data_list:
            ax.text(0.5, 0.5, "No data for selected groups",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas_params.draw_idle()
            return

        self._draw_violin(ax, self.fig_params, data_list, labels, colors)

        # Optional individual points (10% sample)
        if self.show_points.get():
            for i, (vals, col) in enumerate(zip(data_list, colors), start=1):
                sample = np.random.choice(vals, size=max(1, len(vals) // 10), replace=False)
                jitter = np.random.uniform(-0.08, 0.08, len(sample))
                ax.scatter(np.full(len(sample), i) + jitter, sample,
                           color="black", alpha=0.35, s=8, zorder=6)

        param_label, param_units = PARAMETERS.get(param, (param, ""))
        ax.set_title(self._get_tab_title(0, param_label))
        ax.set_ylabel(f"{param_label}\n({param_units})")
        self._apply_ylim(ax)
        self._apply_ticks(ax, 0)
        self._apply_display(ax, fig=self.fig_params, has_legend=False)
        self.canvas_params.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Plot: Speed traces  (reads from pre-built cache — instant)
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_traces(self):
        ax = self.ax_traces
        ax.clear()

        if not self._trace_cache:
            msg = ("Load data to begin" if self.trace_data is None
                   else "No trace data in cache")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    fontsize=16, color="gray", transform=ax.transAxes)
            self.canvas_traces.draw_idle()
            return

        visible = self._visible_groups()
        plotted = []

        for gc in visible:
            entry = self._trace_cache.get(gc.key)
            if entry is None:
                continue
            t, mu, lo, hi = entry
            ax.fill_between(t, lo, hi, color=gc.color, alpha=SEM_ALPHA,
                            linewidth=0.6, edgecolor=gc.color, antialiased=True)
            ax.plot(t, mu, color=gc.color, linewidth=2.0, label=gc.label)
            plotted.append(gc)

        # Fit curve overlay from median bootstrap parameters
        if self.show_overlay.get() and self.bootstrap_data is not None:
            for gc in plotted:
                df   = self.bootstrap_data
                mask = ((df["sex"]       == gc.sex) &
                        (df["genotype"]  == gc.genotype) &
                        (df["treatment"] == gc.treatment))
                sub  = df[mask]
                needed = ["y_min_param", "tau_decay", "A_rec", "tau_rec"]
                alt    = ["y_min_actual", "tau_decay", "A_rec", "tau_rec"]
                cols   = needed if all(c in sub.columns for c in needed) else alt
                if not all(c in sub.columns for c in cols):
                    continue
                y_min     = sub[cols[0]].median()
                tau_decay = sub[cols[1]].median()
                A_rec     = sub[cols[2]].median()
                tau_rec   = sub[cols[3]].median()
                t_fit     = np.linspace(0, 60, 300)
                curve     = decay_recovery_trajectory_model(t_fit, y_min, tau_decay,
                                                            A_rec, tau_rec)
                ax.plot(t_fit, curve, color=gc.color, linestyle="--",
                        linewidth=2.0, alpha=0.85, label=f"{gc.label}  (fit)")

        ax.axvline(0, color="#444444", linestyle=":", linewidth=1.2, alpha=0.6)
        ax.set_xlabel("Time relative to stimulus (s)")
        ax.set_ylabel("Normalized Speed")
        ax.set_title(self._get_tab_title(1, "Normalized Speed Traces  (mean ± SEM)"))
        self._apply_ylim(ax)
        self._apply_ticks(ax, 1)
        self._apply_display(ax, fig=self.fig_traces, has_legend=True)
        self.canvas_traces.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Refresh dispatcher
    # ─────────────────────────────────────────────────────────────────────────
    def refresh(self):
        if self._loading or self._in_refresh:
            return
        self._in_refresh = True
        try:
            idx = self.nb.index(self.nb.select())
            if hasattr(self, "fig_params"):
                fig = (self.fig_params, self.fig_traces)[idx]
                w, h = fig.get_size_inches()
                self.fig_w.delete(0, tk.END); self.fig_w.insert(0, f"{w:.1f}")
                self.fig_h.delete(0, tk.END); self.fig_h.insert(0, f"{h:.1f}")
                if hasattr(self, "fig_tab_lbl"):
                    self.fig_tab_lbl.config(text=self._TAB_NAMES[idx])
            [self._plot_params, self._plot_traces][idx]()

        finally:
            self._in_refresh = False
    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────
    def _export_figure(self):
        idx  = self.nb.index(self.nb.select())
        fig  = (self.fig_params, self.fig_traces)[idx]
        fp   = filedialog.asksaveasfilename(
                   defaultextension=".png",
                   filetypes=[("PNG", "*.png"), ("SVG", "*.svg"),
                               ("PDF", "*.pdf"), ("All files", "*.*")])
        if fp:
            import os
            # tight_layout before save ensures labels never clip
            # at any figure size; restore fixed margins after
            saved_margins = self._fig_margins.get(fig, {})
            fig.tight_layout(pad=0.6)
            if os.path.splitext(fp)[1].lower() == ".svg":
                import io as _io
                buf = _io.BytesIO()
                fig.savefig(buf, format="svg", transparent=True)
                svg_out = _postprocess_svg(buf.getvalue())
                with open(fp, "wb") as _f: _f.write(svg_out)
            else:
                fig.savefig(fp, dpi=300, transparent=True)
            if saved_margins:
                fig.subplots_adjust(**saved_margins)
            messagebox.showinfo("Exported", f"Saved:\n{fp}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.geometry("1400x900")
    ParameterViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
