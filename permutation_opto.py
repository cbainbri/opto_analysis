#!/usr/bin/env python3
"""
Opto Permutation Testing Application

Post-hoc analysis for decay-recovery curve fitting with bootstrap bagging.
Provides both Standard and Max-T classical permutation tests.
"""

import tkinter as tk
import json
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import sys
import io
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')


# =============================================================================
# Decay-recovery model (copied from bootstrap_opto.py)
# =============================================================================

TRANSITION_MODE = 'auto'


def get_transition_factor(tau_decay, tau_rec):
    if TRANSITION_MODE == 'auto':
        factor = 3.0 + (tau_rec - 5.0) * (7.0 - 3.0) / (70.0 - 5.0)
        return np.clip(factor, 3.0, 8.0)
    return float(TRANSITION_MODE)


def decay_recovery_trajectory_model(t, y_min, tau_decay, A_rec, tau_rec):
    decay = y_min + (1.0 - y_min) * np.exp(-t / tau_decay)
    recovery_component = A_rec * (1 - np.exp(-t / tau_rec))
    transition_factor = get_transition_factor(tau_decay, tau_rec)
    decay_weight = np.exp(-t / (transition_factor * tau_decay))
    return decay_weight * decay + (1 - decay_weight) * (y_min + recovery_component)


# =============================================================================
# Constants
# =============================================================================

OPTO_PARAMS = [
    'y_min_param', 'y_min_actual', 't_at_min',
    'tau_decay', 'A_rec', 'tau_rec',
    'drop_depth', 'recovery_at_end', 'recovery_completeness',
    't_half_decay_actual', 't_half_recovery_actual',
    'trough_width_50', 'trough_width_75', 'trough_width_25',
]

GROUP_COLUMNS = ['treatment', 'sex', 'genotype']
GROUP_SEP = '__'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Comparison:
    name: str
    group_a: str
    group_b: str
    observed_diff: float
    group_a_mean: float
    group_b_mean: float


@dataclass
class PermutationResult:
    comparison_name: str
    observed_diff: float
    p_value: float
    null_distribution: np.ndarray
    method: str
    n_permutations: int


@dataclass
class OneSampleResult:
    group: str
    parameter: str
    mu0: float
    mean: float
    ci_lo: float
    ci_hi: float
    mean_minus_mu0: float
    p_value: float
    significant: bool
    n_bootstrap: int


# =============================================================================
# Curve fitting helpers
# =============================================================================

def _fit_opto_curve(t: np.ndarray, y: np.ndarray) -> Optional[dict]:
    """Fit decay_recovery_trajectory_model; returns param dict or None."""
    try:
        valid = ~(np.isnan(t) | np.isnan(y))
        t, y = t[valid], y[valid]
        if len(t) < 6:
            return None
        y_min_guess = y[:5].min() if len(y) >= 5 else y.min()
        y_end = y[-10:].mean() if len(y) >= 10 else y[-1]
        p0 = [y_min_guess, 2.0, max(0.0, y_end - y_min_guess), 20.0]
        bounds = ([0, 0.1, 0, 1], [1, 20, 2, 300])
        popt, _ = curve_fit(
            decay_recovery_trajectory_model, t, y,
            p0=p0, bounds=bounds, maxfev=50000, method='trf')
        return {'y_min': popt[0], 'tau_decay': popt[1], 'A_rec': popt[2], 'tau_rec': popt[3]}
    except Exception:
        return None


def _extract_param(popt: dict, t_data: np.ndarray, parameter: str) -> float:
    """Derive a named parameter from fitted popt dict."""
    try:
        y_min = popt['y_min']
        tau_decay = popt['tau_decay']
        A_rec = popt['A_rec']
        tau_rec = popt['tau_rec']

        if parameter == 'y_min_param':
            return float(y_min)
        if parameter == 'tau_decay':
            return float(tau_decay)
        if parameter == 'A_rec':
            return float(A_rec)
        if parameter == 'tau_rec':
            return float(tau_rec)

        t_end = t_data[-1] if len(t_data) > 0 else 60.0
        t_eval = np.linspace(0, t_end, 1000)
        y_traj = decay_recovery_trajectory_model(t_eval, y_min, tau_decay, A_rec, tau_rec)

        amin_idx = np.argmin(y_traj)
        actual_y_min = y_traj[amin_idx]
        t_at_min = t_eval[amin_idx]

        if parameter == 'y_min_actual':
            return float(actual_y_min)
        if parameter == 't_at_min':
            return float(t_at_min)
        if parameter == 'drop_depth':
            return float(actual_y_min)

        y_final = y_traj[-1]

        if parameter == 'recovery_at_end':
            return float(y_final)
        if parameter == 'recovery_completeness':
            drop = 1.0 - actual_y_min
            return float((y_final - actual_y_min) / drop) if drop > 0 else 1.0

        if parameter == 't_half_decay_actual':
            target = 1.0 - (1.0 - actual_y_min) / 2
            idx = np.argmin(np.abs(y_traj[:amin_idx + 1] - target))
            return float(t_eval[idx])

        if parameter == 't_half_recovery_actual':
            target = actual_y_min + (y_final - actual_y_min) / 2
            idx = amin_idx + np.argmin(np.abs(y_traj[amin_idx:] - target))
            return float(t_eval[idx] - t_at_min)

        for thresh, name in [(0.50, 'trough_width_50'),
                              (0.75, 'trough_width_75'),
                              (0.25, 'trough_width_25')]:
            if parameter == name:
                below = y_traj < thresh
                if np.any(below):
                    idxs = np.where(below)[0]
                    return float(t_eval[idxs[-1]] - t_eval[idxs[0]])
                return 0.0

        return np.nan
    except Exception:
        return np.nan


def aggregate_all_and_fit_opto(
    animal_dataframes: List[pd.DataFrame],
    parameter: str,
    trajectory_agg: str = 'median',
) -> float:
    """
    CLASSICAL PERMUTATION: aggregate ALL animals (no sampling) onto a 1 s time grid,
    fit ONE decay-recovery curve to the post-stim portion, extract the named parameter.

    Each animal DataFrame must have columns ['time_rel', 'speed_norm'].
    Data are already normalized to pre-stim baseline = 1.0, so no denominator is needed.
    """
    try:
        if not animal_dataframes:
            return np.nan

        all_times = [df['time_rel'].to_numpy() for df in animal_dataframes]
        t_min = min(np.nanmin(t) for t in all_times if len(t) > 0)
        t_max = max(np.nanmax(t) for t in all_times if len(t) > 0)
        if not (np.isfinite(t_min) and np.isfinite(t_max) and t_max > t_min):
            return np.nan

        n_grid = int((t_max - t_min) / 1.0) + 1
        time_grid = np.linspace(t_min, t_max, n_grid)

        speeds = []
        for df in animal_dataframes:
            t = df['time_rel'].to_numpy()
            s = df['speed_norm'].to_numpy()
            if len(t) < 2:
                continue
            order = np.argsort(t)
            interp = np.interp(time_grid, t[order], s[order], left=np.nan, right=np.nan)
            speeds.append(interp)

        if not speeds:
            return np.nan

        speeds_arr = np.array(speeds)
        agg = (trajectory_agg or 'median').strip().lower()
        agg_traj = np.nanmean(speeds_arr, axis=0) if agg == 'mean' \
            else np.nanmedian(speeds_arr, axis=0)

        post_mask = time_grid >= 0
        t_post = time_grid[post_mask]
        y_post = agg_traj[post_mask]

        valid = ~np.isnan(y_post)
        if valid.sum() < 6:
            return np.nan

        popt = _fit_opto_curve(t_post[valid], y_post[valid])
        if popt is None:
            return np.nan

        return _extract_param(popt, t_post[valid], parameter)

    except Exception:
        return np.nan


# =============================================================================
# Parallel workers — must be module-level for multiprocessing
# =============================================================================

def _permutation_worker_standard_opto(args):
    """Single-comparison classical permutation worker."""
    perm_idx, all_animals_copy, n_a, parameter, trajectory_agg = args
    np.random.shuffle(all_animals_copy)
    est_a = aggregate_all_and_fit_opto(all_animals_copy[:n_a], parameter, trajectory_agg)
    est_b = aggregate_all_and_fit_opto(all_animals_copy[n_a:], parameter, trajectory_agg)
    if not np.isnan(est_a) and not np.isnan(est_b):
        return est_a - est_b
    return None


def _permutation_worker_maxt_opto(args):
    """Multi-comparison Max-T classical permutation worker."""
    perm_idx, comparisons_data, parameter, trajectory_agg = args
    perm_stats = []
    for group_a_animals, group_b_animals in comparisons_data:
        all_animals = group_a_animals + group_b_animals
        np.random.shuffle(all_animals)
        n_a = len(group_a_animals)
        est_a = aggregate_all_and_fit_opto(all_animals[:n_a], parameter, trajectory_agg)
        est_b = aggregate_all_and_fit_opto(all_animals[n_a:], parameter, trajectory_agg)
        if not np.isnan(est_a) and not np.isnan(est_b):
            perm_stats.append(abs(est_a - est_b))
    return max(perm_stats) if perm_stats else None


# =============================================================================
# Permutation test functions
# =============================================================================

def standard_permutation_test_opto(
    group_a_animals: List[pd.DataFrame],
    group_b_animals: List[pd.DataFrame],
    observed_diff: float,
    parameter: str,
    n_permutations: int = 10000,
    n_cores: Optional[int] = None,
    trajectory_agg: str = 'median',
    progress_callback=None,
) -> PermutationResult:
    """
    Classical permutation test for one opto pairwise comparison.

    Each permutation shuffles animals WITHOUT replacement, splits into groups
    of original sizes, aggregates ALL animals per group, fits ONE curve, and
    computes the parameter difference.  No bootstrap within each permutation.
    Phipson & Smyth +1 correction prevents p = 0.
    """
    all_animals = group_a_animals + group_b_animals
    n_a = len(group_a_animals)
    if n_cores is None:
        n_cores = max(1, int(cpu_count() * 0.75))
    if progress_callback:
        progress_callback(0, n_permutations)

    tasks = [(i, all_animals.copy(), n_a, parameter, trajectory_agg)
             for i in range(n_permutations)]

    null_diffs = []
    with Pool(n_cores) as pool:
        for i, result in enumerate(
                pool.imap_unordered(_permutation_worker_standard_opto, tasks, chunksize=10)):
            if result is not None:
                null_diffs.append(result)
            if progress_callback and (i % 100 == 0 or i == n_permutations - 1):
                progress_callback(i + 1, n_permutations)

    null_diffs = np.array(null_diffs)
    p_value = (np.sum(np.abs(null_diffs) >= abs(observed_diff)) + 1) / (len(null_diffs) + 1)
    return PermutationResult(
        comparison_name="Standard permutation", observed_diff=observed_diff,
        p_value=p_value, null_distribution=null_diffs,
        method='standard', n_permutations=n_permutations)


def maxt_permutation_test_opto(
    comparisons: List[Comparison],
    animals_by_group: Dict[str, List[pd.DataFrame]],
    parameter: str,
    n_permutations: int = 10000,
    n_cores: Optional[int] = None,
    trajectory_agg: str = 'median',
    progress_callback=None,
) -> List[PermutationResult]:
    """
    Classical Max-T permutation test for multiple opto comparisons with FWER control.

    Each permutation shuffles animals for EVERY comparison, tracks the maximum
    absolute difference across all comparisons.  The shared null distribution
    controls family-wise error rate (FWER) without additional post-hoc correction.
    Phipson & Smyth +1 correction prevents p = 0.
    """
    observed_diffs = [comp.observed_diff for comp in comparisons]
    if n_cores is None:
        n_cores = max(1, int(cpu_count() * 0.75))
    if progress_callback:
        progress_callback(0, n_permutations)

    comparisons_data = [(animals_by_group[comp.group_a], animals_by_group[comp.group_b])
                        for comp in comparisons]
    tasks = [(i, comparisons_data, parameter, trajectory_agg)
             for i in range(n_permutations)]

    max_null_stats = []
    with Pool(n_cores) as pool:
        for i, result in enumerate(
                pool.imap_unordered(_permutation_worker_maxt_opto, tasks, chunksize=10)):
            if result is not None:
                max_null_stats.append(result)
            if progress_callback and (i % 100 == 0 or i == n_permutations - 1):
                progress_callback(i + 1, n_permutations)

    max_null_stats = np.array(max_null_stats)
    results = []
    for i, comp in enumerate(comparisons):
        p_value = (np.sum(max_null_stats >= abs(observed_diffs[i])) + 1) / (len(max_null_stats) + 1)
        results.append(PermutationResult(
            comparison_name=comp.name, observed_diff=comp.observed_diff,
            p_value=p_value, null_distribution=max_null_stats,
            method='maxt', n_permutations=n_permutations))
    return results


# =============================================================================
# Multiple-comparison corrections
# =============================================================================

def bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    return np.minimum(p_values * len(p_values), 1.0)


def holm_bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted[sorted_idx[i]] = min(p * (n - i), 1.0)
    for i in range(1, n):
        adjusted[sorted_idx[i]] = max(adjusted[sorted_idx[i]], adjusted[sorted_idx[i - 1]])
    return adjusted


def fdr_benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.zeros(n)
    for i in range(n):
        adjusted[sorted_idx[i]] = min(sorted_p[i] * n / (i + 1), 1.0)
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]])
    return adjusted


def bootstrap_one_sample_test(
    bootstrap_scores: np.ndarray,
    group_label: str,
    parameter: str,
    mu0: float = 1.0,
) -> OneSampleResult:
    """
    One-sample test using the bootstrap distribution as the sampling distribution.

    Primary readout: does mu0 fall outside the 95% CI (2.5/97.5 percentiles)?
      ci_lo > mu0  →  group significantly ABOVE mu0
      ci_hi < mu0  →  group significantly BELOW mu0
    p-value: two-tailed empirical tail probability with +1/(n+1) floor (never exactly 0).
    No SEM, no t-test — bootstrap CI is the direct readout.
    """
    scores = bootstrap_scores[np.isfinite(bootstrap_scores)]
    n = len(scores)
    mean_score = float(np.mean(scores))
    ci_lo = float(np.percentile(scores, 2.5))
    ci_hi = float(np.percentile(scores, 97.5))
    if mean_score >= mu0:
        p = 2.0 * (np.sum(scores <= mu0) + 1) / (n + 1)
    else:
        p = 2.0 * (np.sum(scores >= mu0) + 1) / (n + 1)
    return OneSampleResult(
        group=group_label, parameter=parameter, mu0=mu0,
        mean=mean_score, ci_lo=ci_lo, ci_hi=ci_hi,
        mean_minus_mu0=mean_score - mu0,
        p_value=float(min(p, 1.0)),
        significant=not (ci_lo <= mu0 <= ci_hi),
        n_bootstrap=n,
    )


# =============================================================================
# Data loading
# =============================================================================

def load_opto_package(
    package_dir: Path,
) -> Tuple[Optional[pd.DataFrame], Dict[str, List[pd.DataFrame]]]:
    """
    Load an opto bootstrap output directory.

    Expects:
      all_bootstrap_results_long.csv  — wide format per-bootstrap-iteration params
          columns: group, treatment, sex, genotype, iter, y_min_param, tau_decay, ...
      {group}_normalized_traces.csv   — per-animal normalized traces
          columns: animal_id, time_rel, speed_norm, treatment, sex, genotype

    Returns (bootstrap_df, animals_by_group).
    """
    bootstrap_df = None
    master_path = package_dir / 'all_bootstrap_results_long.csv'
    if master_path.exists():
        bootstrap_df = pd.read_csv(master_path)

    animals_by_group: Dict[str, List[pd.DataFrame]] = {}
    for trace_csv in sorted(package_dir.glob('*_normalized_traces.csv')):
        try:
            df = pd.read_csv(trace_csv)
        except Exception:
            continue

        required = ['animal_id', 'time_rel', 'speed_norm'] + GROUP_COLUMNS
        if any(c not in df.columns for c in required):
            continue

        for group_vals, group_df in df.groupby(GROUP_COLUMNS):
            group_label = GROUP_SEP.join(str(v) for v in group_vals)
            animals = []
            for aid, adf in group_df.groupby('animal_id'):
                aframe = adf[['time_rel', 'speed_norm']].copy().reset_index(drop=True)
                aframe.attrs['animal_id'] = aid
                aframe.attrs['group_key'] = tuple(str(v) for v in group_vals)
                animals.append(aframe)
            if group_label in animals_by_group:
                animals_by_group[group_label].extend(animals)
            else:
                animals_by_group[group_label] = animals

    return bootstrap_df, animals_by_group


def get_observed_differences_opto(
    bootstrap_df: pd.DataFrame,
    comparisons: List[Tuple[str, str, str]],
    parameter: str,
) -> List[Comparison]:
    """
    Calculate observed differences from bootstrap results using median (robust to outliers).
    Group labels use '__' separator matching the bootstrap output format.
    """
    result_list = []
    for name, group_a_label, group_b_label in comparisons:
        group_a_vals = group_a_label.split(GROUP_SEP)
        group_b_vals = group_b_label.split(GROUP_SEP)

        mask_a = np.ones(len(bootstrap_df), dtype=bool)
        mask_b = np.ones(len(bootstrap_df), dtype=bool)
        for col, va, vb in zip(GROUP_COLUMNS, group_a_vals, group_b_vals):
            mask_a &= (bootstrap_df[col].astype(str) == va)
            mask_b &= (bootstrap_df[col].astype(str) == vb)

        data_a = bootstrap_df[mask_a][parameter].dropna().values
        data_b = bootstrap_df[mask_b][parameter].dropna().values
        if len(data_a) == 0 or len(data_b) == 0:
            continue

        median_a = float(np.median(data_a))
        median_b = float(np.median(data_b))
        result_list.append(Comparison(
            name=name, group_a=group_a_label, group_b=group_b_label,
            observed_diff=median_a - median_b,
            group_a_mean=median_a, group_b_mean=median_b))
    return result_list


# =============================================================================
# GUI Application
# =============================================================================

class OptoPermutationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Opto Permutation Testing")

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = min(1400, int(sw * 0.8))
        h = min(900, int(sh * 0.85))
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self.root.minsize(1200, 700)

        # Data
        self.bootstrap_df: Optional[pd.DataFrame] = None
        self.animals_by_group: Dict[str, List[pd.DataFrame]] = {}
        self.selected_comparisons: List[Tuple[str, str, str]] = []
        self.results: List[PermutationResult] = []
        self.comparisons: List[Comparison] = []
        self.one_sample_results: List[OneSampleResult] = []
        self.output_dir: Optional[Path] = None

        # Tkinter vars
        self.parameter = tk.StringVar(value='drop_depth')
        self.test_method = tk.StringVar(value='maxt')
        self.correction_method = tk.StringVar(value='none')
        self.n_permutations = tk.IntVar(value=10000)
        self.trajectory_agg = tk.StringVar(value='median')
        self.alpha = tk.DoubleVar(value=0.05)
        self.mu0_var = tk.DoubleVar(value=1.0)
        self.max_cores = cpu_count()
        self.n_cores = tk.IntVar(value=max(1, int(self.max_cores * 0.75)))

        self._build_gui()

    # ─── Layout builders ──────────────────────────────────────────────────────

    def _build_gui(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=2)
        main.rowconfigure(3, weight=1)

        ttk.Label(main, text="Opto Permutation Testing",
                  font=("Helvetica", 16, "bold")).grid(
            row=0, column=0, columnspan=2, pady=10)

        # Scrollable left column
        left_outer = ttk.Frame(main)
        left_outer.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        left_outer.columnconfigure(0, weight=1)
        left_outer.rowconfigure(0, weight=1)

        lc = tk.Canvas(left_outer, highlightthickness=0)
        lc.grid(row=0, column=0, sticky="nsew")
        ls = ttk.Scrollbar(left_outer, orient="vertical", command=lc.yview)
        ls.grid(row=0, column=1, sticky="ns")
        lc.configure(yscrollcommand=ls.set)

        lf = ttk.Frame(lc)
        lf.columnconfigure(0, weight=1)
        cw_id = lc.create_window((0, 0), window=lf, anchor="nw")
        lf.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.bind("<Configure>", lambda e: lc.itemconfig(cw_id, width=e.width))

        def _mw(event):
            lc.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mw(widget):
            widget.bind("<MouseWheel>", _mw)
            widget.bind("<Button-4>", lambda e: lc.yview_scroll(-1, "units"))
            widget.bind("<Button-5>", lambda e: lc.yview_scroll(1, "units"))
            for child in widget.winfo_children():
                _bind_mw(child)

        lf.bind("<Map>", lambda e: _bind_mw(lf))

        self._create_file_section(lf)
        self._create_parameters_section(lf)
        self._create_method_section(lf)

        # Right column
        rf = ttk.Frame(main)
        rf.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(0, weight=1)
        self._create_comparison_section(rf)

        self._create_action_section(main)
        self._create_results_section(main)

    def _create_file_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Data Files", padding="10")
        frame.pack(fill="x", pady=(0, 5))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Bootstrap output directory:",
                  font=("Helvetica", 9, "bold")).grid(row=0, column=0, sticky="w", pady=3)
        ttk.Label(frame, text="(folder created by bootstrap_opto.py)",
                  foreground="blue", font=("", 8)).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Load Directory",
                   command=self._load_package, width=15).grid(row=0, column=2, padx=3)

        ttk.Separator(frame, orient='horizontal').grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=8)

        ttk.Label(frame, text="Package status:").grid(row=2, column=0, sticky="w", pady=3)
        self._pkg_label = ttk.Label(frame, text="Not loaded", foreground="gray")
        self._pkg_label.grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(frame, text="Output dir:").grid(row=3, column=0, sticky="w", pady=3)
        self._out_label = ttk.Label(frame, text="Not selected", foreground="gray")
        self._out_label.grid(row=3, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Select",
                   command=self._select_output, width=10).grid(row=3, column=2, padx=3)

    def _create_parameters_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Test Parameters", padding="10")
        frame.pack(fill="x", pady=(0, 5))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Parameter:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Combobox(frame, textvariable=self.parameter,
                     values=OPTO_PARAMS, width=25).grid(row=0, column=1, sticky="w", padx=3)

        ttk.Label(frame, text="Permutations:").grid(row=1, column=0, sticky="w", pady=3)
        pf = ttk.Frame(frame)
        pf.grid(row=1, column=1, sticky="w", padx=3)
        ttk.Entry(pf, textvariable=self.n_permutations, width=10).pack(side="left")
        ttk.Label(pf, text="(10000 recommended, 5000 for quick test)",
                  foreground="gray", font=("", 8)).pack(side="left", padx=5)

        ttk.Label(frame, text="Trajectory agg:").grid(row=2, column=0, sticky="w", pady=3)
        af = ttk.Frame(frame)
        af.grid(row=2, column=1, sticky="w", padx=3)
        ttk.Combobox(af, textvariable=self.trajectory_agg,
                     values=["median", "mean"], width=12, state="readonly").pack(side="left")
        ttk.Label(af, text="(pointwise across animals per group)",
                  foreground="gray", font=("", 8)).pack(side="left", padx=5)

        ttk.Label(frame, text="CPU cores:").grid(row=3, column=0, sticky="w", pady=3)
        cf = ttk.Frame(frame)
        cf.grid(row=3, column=1, sticky="w", padx=3)
        ttk.Entry(cf, textvariable=self.n_cores, width=10).pack(side="left")
        ttk.Label(cf, text=f"(max {self.max_cores}, default = 75%)",
                  foreground="gray", font=("", 8)).pack(side="left", padx=5)

        ttk.Label(frame, text="Alpha:").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.alpha, width=10).grid(
            row=4, column=1, sticky="w", padx=3)

    def _create_method_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Test Method", padding="10")
        frame.pack(fill="x", pady=(0, 5))

        ttk.Label(frame, text="Method:",
                  font=("Helvetica", 10, "bold")).pack(anchor="w", pady=3)
        ttk.Radiobutton(frame, text="Max-T (built-in FWER control — RECOMMENDED)",
                        variable=self.test_method, value='maxt').pack(anchor="w", padx=20)
        ttk.Radiobutton(frame, text="Standard (requires post-hoc correction)",
                        variable=self.test_method, value='standard').pack(anchor="w", padx=20)

        ttk.Label(frame, text="Post-hoc correction:",
                  font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(10, 3))
        ttk.Label(frame,
                  text="Note: BH FDR on Max-T p-values is more conservative than\n"
                       "Standard + FDR. Use Standard + FDR for maximum power.",
                  foreground="gray", font=("", 8), justify="left").pack(anchor="w", padx=20)
        for text, val in [("None", "none"), ("Bonferroni", "bonferroni"),
                          ("Holm-Bonferroni", "holm"),
                          ("FDR (Benjamini-Hochberg)  ← more power", "fdr")]:
            ttk.Radiobutton(frame, text=text, variable=self.correction_method,
                            value=val).pack(anchor="w", padx=20)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(10, 5))

        ttk.Label(frame, text="One-Sample Test μ₀ (reference value):",
                  font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 3))
        ttk.Label(frame,
                  text="Tests whether each group's bootstrap distribution excludes μ₀.\n"
                       "Typical defaults: drop_depth=1.0 (no slowing), "
                       "recovery_completeness=1.0 (full recovery).\n"
                       "Uses bootstrap 95% CI directly — no SEM, no t-test.",
                  foreground="gray", font=("", 8), justify="left").pack(anchor="w", padx=20)
        mf = ttk.Frame(frame)
        mf.pack(anchor="w", padx=20, pady=(3, 0))
        ttk.Label(mf, text="μ₀:").pack(side="left")
        ttk.Entry(mf, textvariable=self.mu0_var, width=8).pack(side="left", padx=5)

    def _create_comparison_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Comparisons", padding="10")
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        gf = ttk.Frame(frame)
        gf.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        gf.columnconfigure(1, weight=1)
        ttk.Label(gf, text="Available groups:",
                  font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w")
        self._groups_display = ttk.Label(gf, text="Load data first", foreground="gray")
        self._groups_display.grid(row=0, column=1, sticky="w", padx=5)

        bf = ttk.Frame(frame)
        bf.grid(row=1, column=0, sticky="nsew")
        bf.columnconfigure(0, weight=1)
        bf.columnconfigure(2, weight=1)

        ttk.Label(bf, text="Group A:").grid(row=0, column=0, sticky="w", pady=3)
        self._group_a_var = tk.StringVar()
        self._group_a_combo = ttk.Combobox(bf, textvariable=self._group_a_var, width=30)
        self._group_a_combo.grid(row=1, column=0, sticky="ew", padx=3)

        ttk.Label(bf, text="VS", font=("Helvetica", 12, "bold")).grid(row=1, column=1, padx=10)

        ttk.Label(bf, text="Group B:").grid(row=0, column=2, sticky="w", pady=3)
        self._group_b_var = tk.StringVar()
        self._group_b_combo = ttk.Combobox(bf, textvariable=self._group_b_var, width=30)
        self._group_b_combo.grid(row=1, column=2, sticky="ew", padx=3)

        ttk.Label(bf, text="Comparison name (optional):").grid(
            row=2, column=0, sticky="w", pady=(10, 3))
        self._comp_name_var = tk.StringVar()
        ttk.Entry(bf, textvariable=self._comp_name_var).grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=3)

        ttk.Button(bf, text="Add Comparison",
                   command=self._add_comparison).grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(bf, text="Browse All…",
                   command=self._open_browser).grid(row=4, column=2, pady=10, sticky="ew", padx=3)

        ttk.Label(bf, text="Selected Comparisons:",
                  font=("Helvetica", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(10, 3))

        lf = ttk.Frame(bf)
        lf.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=3)
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)
        bf.rowconfigure(6, weight=1)

        sb = ttk.Scrollbar(lf)
        sb.pack(side="right", fill="y")
        self._comp_listbox = tk.Listbox(lf, yscrollcommand=sb.set, height=10)
        self._comp_listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self._comp_listbox.yview)

        ttk.Button(bf, text="Remove Selected",
                   command=self._remove_comparison).grid(row=7, column=0, columnspan=3, pady=5)

        btnf = ttk.Frame(bf)
        btnf.grid(row=8, column=0, columnspan=3, pady=(0, 5))
        ttk.Button(btnf, text="Save Comparisons",
                   command=self._save_comparisons, width=18).pack(side="left", padx=3)
        ttk.Button(btnf, text="Load Comparisons",
                   command=self._load_comparisons, width=18).pack(side="left", padx=3)
        ttk.Button(btnf, text="Clear All",
                   command=self._clear_comparisons, width=10).pack(side="left", padx=3)

    def _create_action_section(self, parent):
        frame = ttk.Frame(parent)
        frame.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Run Tests",
                   command=self._run_tests, width=20).pack(side="left", padx=5)
        ttk.Button(frame, text="Run One-Sample Tests",
                   command=self._run_one_sample_tests, width=22).pack(side="left", padx=5)
        ttk.Button(frame, text="Clear Results",
                   command=self._clear_results, width=15).pack(side="left", padx=5)
        ttk.Button(frame, text="Save Results CSV",
                   command=self._save_results, width=17).pack(side="left", padx=5)
        ttk.Button(frame, text="Save One-Sample CSV",
                   command=self._save_one_sample_results, width=20).pack(side="left", padx=5)

    def _create_results_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Results", padding="10")
        frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        sb = ttk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        self._results_text = tk.Text(
            frame, wrap=tk.WORD, yscrollcommand=sb.set, font=("Courier", 9))
        self._results_text.pack(fill="both", expand=True)
        sb.config(command=self._results_text.yview)

    # ─── Event handlers ────────────────────────────────────────────────────────

    def _load_package(self):
        dirname = filedialog.askdirectory(
            title="Select bootstrap_opto.py output directory")
        if not dirname:
            return
        pkg_dir = Path(dirname)
        try:
            self._log("\n" + "=" * 70)
            self._log("LOADING OPTO BOOTSTRAP PACKAGE")
            self._log("=" * 70)

            bootstrap_df, animals_by_group = load_opto_package(pkg_dir)

            if bootstrap_df is None:
                self._log("\n⚠ all_bootstrap_results_long.csv not found "
                          "— one-sample tests unavailable")
            else:
                self.bootstrap_df = bootstrap_df
                n_rows = len(bootstrap_df)
                n_groups = bootstrap_df['group'].nunique() if 'group' in bootstrap_df.columns else '?'
                self._log(f"\n✓ Loaded bootstrap results: {n_rows} rows, {n_groups} groups")
                avail = [p for p in OPTO_PARAMS if p in bootstrap_df.columns]
                self._log(f"  Available parameters: {', '.join(avail)}")

            if not animals_by_group:
                self._log("\n⚠ No *_normalized_traces.csv files found "
                          "— permutation testing unavailable")
            else:
                self.animals_by_group = animals_by_group
                n_grp = len(animals_by_group)
                total = sum(len(a) for a in animals_by_group.values())
                self._log(f"\n✓ Loaded normalized traces: {n_grp} groups, {total} animals")
                group_names = sorted(animals_by_group.keys())
                self._group_a_combo['values'] = group_names
                self._group_b_combo['values'] = group_names
                self._groups_display.config(
                    text=f"{n_grp} groups, {total} animals total", foreground="black")
                self._log("\nGroups:")
                for gname in group_names:
                    self._log(f"  {gname}: {len(animals_by_group[gname])} animals")

            self._pkg_label.config(text=f"Loaded: {pkg_dir.name}", foreground="black")
            self._log("\n" + "=" * 70)
            self._log("✓ PACKAGE LOADED SUCCESSFULLY")
            self._log("=" * 70)
            messagebox.showinfo("Success",
                                "Opto bootstrap package loaded!\nReady for permutation testing.")
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Failed to load package:\n{e}")
            self._log(f"\n✗ ERROR:\n{traceback.format_exc()}")

    def _select_output(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.output_dir = Path(d)
            self._out_label.config(text=str(self.output_dir), foreground="black")
            self._log(f"✓ Output directory: {self.output_dir}")

    def _add_comparison(self):
        ga = self._group_a_var.get()
        gb = self._group_b_var.get()
        name = self._comp_name_var.get().strip()
        if not ga or not gb:
            messagebox.showwarning("Warning", "Please select both groups")
            return
        if not name:
            name = f"{ga} vs {gb}"
        for comp in self.selected_comparisons:
            if comp[1] == ga and comp[2] == gb:
                messagebox.showwarning("Warning", "This comparison is already added")
                return
        self.selected_comparisons.append((name, ga, gb))
        self._comp_listbox.insert(tk.END, name)
        self._log(f"✓ Added comparison: {name}")
        self._comp_name_var.set("")

    def _remove_comparison(self):
        sel = self._comp_listbox.curselection()
        if sel:
            idx = sel[0]
            self._comp_listbox.delete(idx)
            removed = self.selected_comparisons.pop(idx)
            self._log(f"✗ Removed comparison: {removed[0]}")

    def _open_browser(self):
        all_groups = sorted(self.animals_by_group.keys())
        if not all_groups:
            messagebox.showwarning("Warning", "No groups loaded yet.\nLoad a package first.")
            return

        import itertools
        win = tk.Toplevel(self.root)
        win.title("All Possible Comparisons")
        win.geometry("720x600")
        win.resizable(True, True)

        hdr = ttk.Frame(win, padding=(10, 8))
        hdr.pack(fill="x")
        n_pairs = len(all_groups) * (len(all_groups) - 1) // 2
        ttk.Label(hdr, text="Select comparisons",
                  font=("Helvetica", 12, "bold")).pack(side="left")
        ttk.Label(hdr, text=f"  ({len(all_groups)} groups, {n_pairs} possible pairs)",
                  foreground="gray").pack(side="left")

        btn_row = ttk.Frame(win, padding=(10, 0))
        btn_row.pack(fill="x")
        check_vars = {}

        def select_all():
            for v in check_vars.values(): v.set(True)
        def clear_all():
            for v in check_vars.values(): v.set(False)
        def invert():
            for v in check_vars.values(): v.set(not v.get())

        ttk.Button(btn_row, text="Select All", command=select_all,
                   width=12).pack(side="left", padx=3, pady=4)
        ttk.Button(btn_row, text="Clear All", command=clear_all,
                   width=12).pack(side="left", padx=3)
        ttk.Button(btn_row, text="Invert", command=invert,
                   width=12).pack(side="left", padx=3)

        outer = ttk.Frame(win)
        outer.pack(fill="both", expand=True, padx=10, pady=4)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas)
        cw = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        already = {(a, b) for _, a, b in self.selected_comparisons}
        for i, (ga, gb) in enumerate(itertools.combinations(all_groups, 2)):
            var = tk.BooleanVar(value=((ga, gb) in already or (gb, ga) in already))
            check_vars[(ga, gb)] = var
            bg = "#f0f0f0" if i % 2 == 0 else "white"
            row_f = tk.Frame(inner, background=bg)
            row_f.pack(fill="x")
            tk.Checkbutton(row_f, variable=var, background=bg).pack(side="left")
            tk.Label(row_f, text=f"{ga}   vs   {gb}",
                     background=bg, font=("Courier", 9), anchor="w").pack(
                side="left", fill="x", expand=True, padx=(0, 6))

        act = ttk.Frame(win, padding=(10, 8))
        act.pack(fill="x")

        def _apply(replace: bool):
            chosen = [(a, b) for (a, b), v in check_vars.items() if v.get()]
            if not chosen:
                messagebox.showwarning("Warning", "No comparisons selected.", parent=win)
                return
            if replace:
                self.selected_comparisons = []
                self._comp_listbox.delete(0, tk.END)
            added = 0
            for ga, gb in chosen:
                name = f"{ga} vs {gb}"
                if (name, ga, gb) in self.selected_comparisons:
                    continue
                self.selected_comparisons.append((name, ga, gb))
                self._comp_listbox.insert(tk.END, name)
                added += 1
            action = "Replaced list with" if replace else "Added"
            self._log(f"✓ {action} {added} comparisons from browser")
            win.destroy()

        ttk.Button(act, text="Replace List with Selected",
                   command=lambda: _apply(replace=True), width=26).pack(side="left", padx=5)
        ttk.Button(act, text="Add Selected to List",
                   command=lambda: _apply(replace=False), width=22).pack(side="left", padx=5)
        ttk.Button(act, text="Cancel",
                   command=win.destroy, width=10).pack(side="right", padx=5)

    def _save_comparisons(self):
        if not self.selected_comparisons:
            messagebox.showwarning("Warning", "No comparisons to save.")
            return
        fp = filedialog.asksaveasfilename(
            title="Save comparison list", defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not fp:
            return
        try:
            data = [{"name": name, "group_a": a, "group_b": b}
                    for name, a, b in self.selected_comparisons]
            with open(fp, "w") as f:
                json.dump(data, f, indent=2)
            self._log(f"✓ Saved {len(data)} comparisons to {fp}")
            messagebox.showinfo("Saved", f"Saved {len(data)} comparisons to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    def _load_comparisons(self):
        fp = filedialog.askopenfilename(
            title="Load comparison list",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not fp:
            return
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            added = skipped = 0
            for entry in data:
                name = entry.get("name", "")
                ga = entry.get("group_a", "")
                gb = entry.get("group_b", "")
                if not ga or not gb:
                    skipped += 1
                    continue
                if not name:
                    name = f"{ga} vs {gb}"
                if (name, ga, gb) in self.selected_comparisons:
                    skipped += 1
                    continue
                self.selected_comparisons.append((name, ga, gb))
                self._comp_listbox.insert(tk.END, name)
                added += 1
            msg = f"Loaded {added} comparisons"
            if skipped:
                msg += f" ({skipped} skipped)"
            self._log(f"✓ {msg} from {fp}")
            messagebox.showinfo("Loaded", msg)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e}")

    def _clear_comparisons(self):
        self.selected_comparisons = []
        self._comp_listbox.delete(0, tk.END)
        self._log("✓ Comparisons cleared")

    def _run_tests(self):
        if not self.animals_by_group:
            messagebox.showwarning("Warning", "Please load a bootstrap package first")
            return
        if not self.selected_comparisons:
            messagebox.showwarning("Warning", "Please add at least one comparison")
            return

        try:
            param = self.parameter.get()
            self._log("\n" + "=" * 70)
            self._log(f"RUNNING OPTO PERMUTATION TESTS — {param}")
            self._log("=" * 70)

            if self.bootstrap_df is None or param not in self.bootstrap_df.columns:
                messagebox.showerror(
                    "Error",
                    f"Parameter '{param}' not found in bootstrap results.\n"
                    "Load a package that contains all_bootstrap_results_long.csv "
                    "with this parameter column.")
                return

            self.comparisons = get_observed_differences_opto(
                self.bootstrap_df, self.selected_comparisons, param)

            if not self.comparisons:
                messagebox.showerror(
                    "Error",
                    "No valid comparisons found.\n"
                    "Check that group labels in the comparison list match the loaded data.")
                return

            self._log(f"\n✓ {len(self.comparisons)} valid comparisons:")
            for comp in self.comparisons:
                self._log(f"  {comp.name}: observed diff = {comp.observed_diff:.4f} "
                          f"(A median={comp.group_a_mean:.4f}, B median={comp.group_b_mean:.4f})")

            # Verify all groups are present in trace data
            for comp in self.comparisons:
                for grp in (comp.group_a, comp.group_b):
                    if grp not in self.animals_by_group:
                        messagebox.showerror(
                            "Error",
                            f"Group '{grp}' not found in loaded trace data.\n"
                            "Make sure the bootstrap output directory contains the matching "
                            "*_normalized_traces.csv files.")
                        return

            def _progress(current, total):
                self._log(f"  Progress: {current}/{total} ({100 * current // total}%)")
                self.root.update()

            method = self.test_method.get()
            n_cores = self.n_cores.get()
            n_perm = self.n_permutations.get()
            agg = self.trajectory_agg.get()

            if method == 'maxt':
                self._log(f"\n✓ Running Max-T permutation test...")
                self._log(f"  {n_perm} permutations | {n_cores} cores | agg={agg}")
                self._log("  Method: classical permutation (ALL animals aggregated, ONE fit per group)")
                self.results = maxt_permutation_test_opto(
                    comparisons=self.comparisons,
                    animals_by_group=self.animals_by_group,
                    parameter=param, n_permutations=n_perm,
                    n_cores=n_cores, trajectory_agg=agg,
                    progress_callback=_progress)
            else:
                self._log(f"\n✓ Running standard permutation tests...")
                self._log(f"  {n_perm} permutations | {n_cores} cores | agg={agg}")
                self.results = []
                for i, comp in enumerate(self.comparisons):
                    self._log(f"\n  Test {i + 1}/{len(self.comparisons)}: {comp.name}")
                    result = standard_permutation_test_opto(
                        group_a_animals=self.animals_by_group[comp.group_a],
                        group_b_animals=self.animals_by_group[comp.group_b],
                        observed_diff=comp.observed_diff,
                        parameter=param, n_permutations=n_perm,
                        n_cores=n_cores, trajectory_agg=agg,
                        progress_callback=_progress)
                    self.results.append(result)

            self._display_results()
            self._log("\n" + "=" * 70)
            self._log("✓ TESTS COMPLETE")
            self._log("=" * 70)
            messagebox.showinfo("Success", "Tests completed successfully!")

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Failed to run tests:\n{e}")
            self._log(f"\n✗ ERROR:\n{traceback.format_exc()}")

    def _display_results(self):
        p_values = np.array([r.p_value for r in self.results])

        corr = None if self.test_method.get() == 'maxt' else self.correction_method.get()
        if corr and corr != 'none':
            if corr == 'bonferroni':
                p_adj, corr_name = bonferroni_correction(p_values), "Bonferroni"
            elif corr == 'holm':
                p_adj, corr_name = holm_bonferroni_correction(p_values), "Holm-Bonferroni"
            elif corr == 'fdr':
                p_adj, corr_name = fdr_benjamini_hochberg(p_values), "FDR (B-H)"
            else:
                p_adj, corr_name = p_values, "None"
        else:
            p_adj = p_values
            corr_name = "Max-T" if self.test_method.get() == 'maxt' else "None"

        alpha = self.alpha.get()
        out = io.StringIO()
        out.write("\n" + "=" * 100 + "\n")
        out.write("OPTO PERMUTATION TEST RESULTS\n")
        out.write(f"Parameter: {self.parameter.get()}\n")
        out.write(f"Method: {self.results[0].method.upper()}\n")
        out.write(f"Correction: {corr_name}\n")
        out.write(f"Significance level: α = {alpha}\n")
        out.write("=" * 100 + "\n\n")
        out.write(f"{'Comparison':<40} {'Group A':>12} {'Group B':>12} "
                  f"{'Diff':>10} {'p-value':>10} {'p-adj':>10} {'Sig':>5}\n")
        out.write("-" * 100 + "\n")
        for comp, result, p_raw, p_a in zip(self.comparisons, self.results, p_values, p_adj):
            sig = "*" if p_a < alpha else "ns"
            out.write(f"{comp.name:<40} {comp.group_a_mean:>12.4f} {comp.group_b_mean:>12.4f} "
                      f"{comp.observed_diff:>10.4f} {p_raw:>10.4f} {p_a:>10.4f} {sig:>5}\n")
        out.write("=" * 100 + "\n")
        out.write(f"\nSignificant comparisons: {sum(p_adj < alpha)} / {len(self.comparisons)}\n")

        self._results_text.delete(1.0, tk.END)
        self._results_text.insert(tk.END, out.getvalue())

    def _run_one_sample_tests(self):
        """
        One-sample bootstrap CI test for each group vs mu0.
        Tests whether each group's bootstrap distribution of the selected parameter
        excludes mu0 at the 95% CI level.  No permutation needed — the bootstrap
        distribution IS the sampling distribution.
        """
        if self.bootstrap_df is None:
            messagebox.showwarning(
                "Warning",
                "No bootstrap data loaded.\n"
                "Load a package that contains all_bootstrap_results_long.csv.")
            return

        param = self.parameter.get()
        if param not in self.bootstrap_df.columns:
            messagebox.showerror(
                "Error",
                f"Parameter '{param}' not found in bootstrap results.\n"
                "Select a valid opto parameter or reload the package.")
            return

        try:
            mu0 = float(self.mu0_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Error", "Invalid μ₀ value.")
            return

        self.one_sample_results = []
        alpha = self.alpha.get()

        self._log("\n" + "=" * 70)
        self._log(f"ONE-SAMPLE BOOTSTRAP CI TESTS  (μ₀ = {mu0}, parameter = {param})")
        self._log("=" * 70)
        self._log("Primary readout: does μ₀ fall outside the 95% CI?")
        self._log("  CI = [2.5th pct, 97.5th pct] of bootstrap distribution")
        self._log("  p-value = empirical two-tailed tail probability, floor = 1/(n+1)\n")
        self._log(f"{'Group':<50} {'Mean':>8} {'CI_lo':>8} {'CI_hi':>8} "
                  f"{'Diff':>8} {'p':>8} {'Sig':>5}")
        self._log("-" * 100)

        param_results = []
        group_col = 'group' if 'group' in self.bootstrap_df.columns else None
        if group_col is None:
            messagebox.showerror("Error", "bootstrap results CSV has no 'group' column.")
            return

        for group_label in sorted(self.bootstrap_df[group_col].unique()):
            scores = self.bootstrap_df[
                self.bootstrap_df[group_col] == group_label][param].dropna().values
            if len(scores) < 10:
                self._log(f"  {group_label:<48} SKIPPED (n={len(scores)} < 10)")
                continue

            result = bootstrap_one_sample_test(scores, group_label, param, mu0)
            self.one_sample_results.append(result)
            param_results.append(result)

            sig_str = ("***" if result.p_value < 0.001 else
                       "**"  if result.p_value < 0.01  else
                       "*"   if result.p_value < alpha  else "ns")
            direction = "↑" if result.mean > mu0 else "↓"
            sig_marker = f"{sig_str}{direction}" if result.significant else sig_str

            self._log(f"  {group_label:<48} {result.mean:>8.4f} {result.ci_lo:>8.4f} "
                      f"{result.ci_hi:>8.4f} {result.mean_minus_mu0:>+8.4f} "
                      f"{result.p_value:>8.4f} {sig_marker:>5}")

        if self.correction_method.get() == 'fdr' and len(param_results) > 1:
            raw_p = np.array([r.p_value for r in param_results])
            adj_p = fdr_benjamini_hochberg(raw_p)
            self._log(f"\nBH FDR adjusted p-values for {param}:")
            for r, ap in zip(param_results, adj_p):
                self._log(f"  {r.group:<48} raw p={r.p_value:.4f}  adj p={ap:.4f}"
                          f"  {'∗' if ap < alpha else 'ns'}")

        self._log(f"\n✓ One-sample tests complete: {len(self.one_sample_results)} groups tested.")
        self._log("  Use 'Save One-Sample CSV' to export results.\n")

    def _clear_results(self):
        self.results = []
        self.comparisons = []
        self.one_sample_results = []
        self._results_text.delete(1.0, tk.END)
        self._log("✓ Results cleared")

    def _save_results(self):
        if not self.output_dir:
            messagebox.showwarning("Warning", "Please select output directory first")
            return
        if not self.results:
            messagebox.showwarning("Warning", "No results to save")
            return
        try:
            p_values = np.array([r.p_value for r in self.results])
            corr = self.correction_method.get()
            if corr and corr != 'none':
                if corr == 'bonferroni':
                    p_adj = bonferroni_correction(p_values)
                elif corr == 'holm':
                    p_adj = holm_bonferroni_correction(p_values)
                elif corr == 'fdr':
                    p_adj = fdr_benjamini_hochberg(p_values)
                else:
                    p_adj = p_values
            else:
                p_adj = p_values

            df = pd.DataFrame({
                'comparison':     [c.name           for c in self.comparisons],
                'group_a':        [c.group_a         for c in self.comparisons],
                'group_b':        [c.group_b         for c in self.comparisons],
                'group_a_median': [c.group_a_mean    for c in self.comparisons],
                'group_b_median': [c.group_b_mean    for c in self.comparisons],
                'observed_diff':  [c.observed_diff   for c in self.comparisons],
                'p_value':        p_values,
                'p_adjusted':     p_adj,
                'significant':    p_adj < self.alpha.get(),
                'method':         [r.method          for r in self.results],
                'n_permutations': [r.n_permutations  for r in self.results],
                'parameter':      self.parameter.get(),
            })
            out_path = self.output_dir / "permutation_results_opto.csv"
            df.to_csv(out_path, index=False)
            self._log(f"\n✓ Results saved to {out_path}")
            messagebox.showinfo("Success", f"Results saved to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{e}")

    def _save_one_sample_results(self):
        if not self.one_sample_results:
            messagebox.showwarning("Warning",
                                   "No one-sample results to save.\n"
                                   "Run 'Run One-Sample Tests' first.")
            return
        if not self.output_dir:
            messagebox.showwarning("Warning", "Please select output directory first.")
            return
        try:
            rows = [{'group':          r.group,
                     'parameter':      r.parameter,
                     'mu0':            r.mu0,
                     'mean':           r.mean,
                     'ci_lo_2.5pct':   r.ci_lo,
                     'ci_hi_97.5pct':  r.ci_hi,
                     'mean_minus_mu0': r.mean_minus_mu0,
                     'p_value':        r.p_value,
                     'significant':    r.significant,
                     'n_bootstrap':    r.n_bootstrap}
                    for r in self.one_sample_results]
            df = pd.DataFrame(rows)
            out_path = self.output_dir / "one_sample_results_opto.csv"
            df.to_csv(out_path, index=False)
            self._log(f"✓ One-sample results saved to {out_path}")
            messagebox.showinfo("Saved", f"Saved to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    def _log(self, message: str):
        self._results_text.insert(tk.END, message + "\n")
        self._results_text.see(tk.END)
        self.root.update()


# =============================================================================
# Entry point
# =============================================================================

def main():
    root = tk.Tk()
    OptoPermutationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
