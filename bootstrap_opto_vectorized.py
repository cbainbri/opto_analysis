#!/usr/bin/env python3
"""
Bootstrap decay-recovery fitting with simplified recovery model.
Fits observed trajectory without forcing asymptotic plateau.
GUI version using PyQt5 to avoid tkinter/multiprocessing conflicts.

VECTORIZED: Bootstrap mean traces are pre-computed as a batch using NumPy array
indexing before the fitting pool, eliminating per-iteration pandas overhead.
All analytical logic (model, bounds, fitting, derived metrics) is unchanged.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - NO GUI conflicts
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats as scipy_stats
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QFileDialog, QSpinBox, QDoubleSpinBox, QMessageBox,
                             QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ============================================================
# MODEL CONFIGURATION
# ============================================================
# Transition sharpness controls how quickly model switches from decay to recovery
# Options:
#   'auto' - Adaptive based on tau_decay (recommended)
#   float  - Fixed multiplier (e.g., 3.0, 5.0, 7.0)
TRANSITION_MODE = 'auto'  # or set to a fixed value like 5.0

def get_transition_factor(tau_decay, tau_rec):
    """
    Calculate transition factor based on recovery speed.

    Fast recovery (tau_rec < 10s) → Sharp transition (factor = 3-4)
    Slow recovery (tau_rec > 50s) → Smooth transition (factor = 6-8)

    Logic: Animals with fast recovery switch quickly from decay to recovery.
           Animals with slow recovery have a more gradual transition at the trough.
    """
    if TRANSITION_MODE == 'auto':
        # Adaptive: faster recovery = sharper transition
        # Linear scaling: tau_rec 5s → factor 3, tau_rec 70s → factor 7
        factor = 3.0 + (tau_rec - 5.0) * (7.0 - 3.0) / (70.0 - 5.0)
        # Clamp between 3 and 8
        return np.clip(factor, 3.0, 8.0)
    else:
        # Fixed value
        return float(TRANSITION_MODE)

# ============================================================
# SIMPLIFIED TRAJECTORY MODEL
# ============================================================

def decay_recovery_trajectory_model(t, y_min, tau_decay, A_rec, tau_rec):
    """
    Simplified model that fits observed trajectory without predicting final asymptote.

    Models:
    1. Exponential decay from baseline (1.0) to minimum (y_min) with tau_decay
    2. Exponential recovery with amplitude A_rec and time constant tau_rec

    Key feature: Doesn't force a specific yinf. If data plateaus, it will capture it.
                 If data is still recovering, it describes the trajectory without over-extrapolating.

    Parameters:
    - y_min: minimum speed level reached (nadir)
    - tau_decay: time constant for initial decay
    - A_rec: amplitude/strength of recovery (how much recovery occurs)
    - tau_rec: recovery time constant (how fast recovery happens)
    """

    # Decay phase: exponential from 1.0 toward y_min
    decay = y_min + (1.0 - y_min) * np.exp(-t / tau_decay)

    # Recovery phase: exponential recovery from y_min with amplitude A_rec
    # This grows from 0 toward A_rec
    recovery_component = A_rec * (1 - np.exp(-t / tau_rec))

    # Combined trajectory with adaptive transition
    transition_factor = get_transition_factor(tau_decay, tau_rec)
    decay_weight = np.exp(-t / (transition_factor * tau_decay))

    result = decay_weight * decay + (1 - decay_weight) * (y_min + recovery_component)

    return result


# ============================================================
# BOOTSTRAP FITTING FUNCTION (VECTORIZED)
# ============================================================
def _fit_replicate_precomputed(args):
    """
    Worker for vectorized bootstrap: receives a pre-computed mean trace and runs
    curve fitting + metric extraction. Analytical logic is identical to the original
    fit_bootstrap_iteration; only the input signature changes (pre-built mean array
    instead of raw traces dict).
    """
    i, mean_trace, time_grid_arr = args
    try:
        post_mask = time_grid_arr >= 0
        t = time_grid_arr[post_mask]
        y = mean_trace[post_mask]

        if len(t) < 6 or np.any(np.isnan(y)):
            return None

        # Initial guesses
        y_min_guess = y[:5].min() if len(y) >= 5 else y.min()
        y_end = y[-10:].mean() if len(y) >= 10 else y[-1]
        A_rec_guess = y_end - y_min_guess  # How much recovery we observe

        p0 = [y_min_guess, 2.0, A_rec_guess, 20.0]

        # Bounds:
        # y_min: 0 to 1
        # tau_decay: 0.1 to 20
        # A_rec: 0 to 2 (can't recover more than 2x baseline)
        # tau_rec: 1 to 300 (recovery time constant)
        bounds = (
            [0,    0.1,  0,     1],
            [1,    20,   2,     300]
        )

        popt, pcov = curve_fit(
            decay_recovery_trajectory_model,
            t, y,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
            method='trf'
        )

        y_min, tau_decay, A_rec, tau_rec = popt

        # Evaluate trajectory on fine grid
        t_eval = np.linspace(0, t[-1], 1000)
        y_traj = decay_recovery_trajectory_model(t_eval, y_min, tau_decay, A_rec, tau_rec)

        # Find actual minimum of trajectory
        actual_min_idx = np.argmin(y_traj)
        actual_y_min = y_traj[actual_min_idx]
        t_at_min = t_eval[actual_min_idx]

        # Drop depth: express as percentage of baseline retained (not dropped)
        # E.g., if speed drops to 10% of baseline, drop_depth = 0.10 (10% baseline speed)
        drop_depth = actual_y_min  # Changed from 1.0 - actual_y_min

        # Calculate half-times based on ACTUAL TRAJECTORY
        # Decay half-time: time to reach halfway between baseline (1.0) and actual_y_min
        decay_drop = 1.0 - actual_y_min
        target_decay_half = 1.0 - (decay_drop / 2)  # Halfway down
        decay_half_idx = np.argmin(np.abs(y_traj[:actual_min_idx+1] - target_decay_half))
        t_half_decay_actual = t_eval[decay_half_idx]

        # Recovery half-time: time to reach halfway between actual_y_min and final level
        y_final = y_traj[-1]  # Value at end of window
        target_recovery_half = actual_y_min + (y_final - actual_y_min) / 2
        # Only look after the minimum
        recovery_half_idx = actual_min_idx + np.argmin(np.abs(y_traj[actual_min_idx:] - target_recovery_half))
        t_half_recovery_actual = t_eval[recovery_half_idx] - t_at_min  # Time from minimum

        # Calculate trough widths at different thresholds
        # FWHM: Full Width at Half Maximum (time below 50% of baseline)
        threshold_50 = 0.5
        below_50 = y_traj < threshold_50
        if np.any(below_50):
            # Find first and last time below threshold
            below_indices = np.where(below_50)[0]
            t_enter_50 = t_eval[below_indices[0]]
            t_exit_50 = t_eval[below_indices[-1]]
            trough_width_50 = t_exit_50 - t_enter_50
        else:
            trough_width_50 = 0.0
            t_enter_50 = np.nan
            t_exit_50 = np.nan

        # Trough width at 75% (time below 75% of baseline - mild impairment)
        threshold_75 = 0.75
        below_75 = y_traj < threshold_75
        if np.any(below_75):
            below_indices = np.where(below_75)[0]
            t_enter_75 = t_eval[below_indices[0]]
            t_exit_75 = t_eval[below_indices[-1]]
            trough_width_75 = t_exit_75 - t_enter_75
        else:
            trough_width_75 = 0.0
            t_enter_75 = np.nan
            t_exit_75 = np.nan

        # Trough width at 25% (time below 25% of baseline - severe impairment)
        threshold_25 = 0.25
        below_25 = y_traj < threshold_25
        if np.any(below_25):
            below_indices = np.where(below_25)[0]
            t_enter_25 = t_eval[below_indices[0]]
            t_exit_25 = t_eval[below_indices[-1]]
            trough_width_25 = t_exit_25 - t_enter_25
        else:
            trough_width_25 = 0.0
            t_enter_25 = np.nan
            t_exit_25 = np.nan

        # Speed at end of observation window
        y_at_end = y_traj[-1]
        recovery_at_end = y_at_end

        # Recovery completeness: how much has recovered by end / how much dropped
        # Now that drop_depth is the minimum level, calculate the actual drop for this metric
        actual_drop = 1.0 - drop_depth
        if actual_drop > 0:
            recovery_completeness = (y_at_end - drop_depth) / actual_drop
        else:
            recovery_completeness = 1.0

        # Also keep parameter-based half-times for comparison
        t_half_decay_param = tau_decay * np.log(2)
        t_half_recovery_param = tau_rec * np.log(2)

        return {
            "iter": i,
            "y_min_param": y_min,
            "y_min_actual": actual_y_min,
            "t_at_min": t_at_min,
            "tau_decay": tau_decay,
            "A_rec": A_rec,
            "tau_rec": tau_rec,
            "drop_depth": drop_depth,
            "recovery_at_end": recovery_at_end,
            "recovery_completeness": recovery_completeness,
            "t_half_decay_param": t_half_decay_param,
            "t_half_decay_actual": t_half_decay_actual,
            "t_half_recovery_param": t_half_recovery_param,
            "t_half_recovery_actual": t_half_recovery_actual,
            "trough_width_50": trough_width_50,
            "trough_width_75": trough_width_75,
            "trough_width_25": trough_width_25,
            "t_enter_50": t_enter_50,
            "t_exit_50": t_exit_50
        }

    except Exception:
        return None


# ============================================================
# DIAGNOSTIC: BASELINE SPEED vs. % MAX SLOWING (per group)
# ============================================================

def export_baseline_maxslow_diagnostic(df, group_cols, outdir, progress_fn=None):
    """
    Diagnostic: for each group (treatment × sex × strain_genotype), plot each
    individual animal's pre-stimulus baseline speed (mm/s) against the fraction of
    that baseline retained at the post-stimulus minimum (min_post_speed / baseline).

    Steps per animal:
      1. Apply a 3-frame rolling median to the raw speed trace (smoothing for
         diagnostic purposes only — does not affect any other pipeline output).
      2. baseline_speed  = mean of smoothed speed where time_rel < 0
      3. min_post_speed  = minimum of smoothed speed where time_rel >= 0
      4. pct_retained    = min_post_speed / baseline_speed
         (1.0 = no slowing, 0.0 = complete arrest)

    Outputs per group  (saved to outdir/diagnostics/):
      • <group>_baseline_vs_maxslow.png   – scatter plot
      • <group>_baseline_vs_maxslow.csv   – per-animal values

    Parameters
    ----------
    df          : filtered DataFrame with columns time_rel, speed, animal_id,
                  treatment, sex, strain_genotype
    group_cols  : list of column names used for grouping, e.g.
                  ['treatment', 'sex', 'strain_genotype']
    outdir      : output directory (diagnostics/ sub-folder is created inside)
    progress_fn : optional callable for status messages (e.g. self.progress.emit)
    """

    diag_dir = os.path.join(outdir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    def _emit(msg):
        if progress_fn is not None:
            progress_fn(msg)

    # Sex color palette
    def _sex_color(sex_str):
        s = str(sex_str).lower()
        if "herm" in s or s == "h":
            return "#E8748A"   # reddish pink for hermaphrodites
        elif "male" in s or s == "m":
            return "#6EB5D8"   # baby blue for males
        return "#999999"

    def _sex_edge(sex_str):
        s = str(sex_str).lower()
        if "herm" in s or s == "h":
            return "#A03050"
        elif "male" in s or s == "m":
            return "#2A6A9A"
        return "#555555"

    _emit("Generating baseline-vs-maxslow diagnostics...")

    # ── PASS 1: collect all per-animal records, then filter non-responders ──
    # Collecting everything first lets us compute a shared x-axis max so all
    # plots are on an identical scale for direct visual comparison.
    all_group_data = {}   # gname_str -> DataFrame

    for gname, g in df.groupby(group_cols):
        gname_str = "__".join(map(str, gname))
        sex_val = gname[group_cols.index("sex")] if "sex" in group_cols else ""

        records = []
        n_excluded = 0
        for aid, sub in g.groupby("animal_id"):
            sub = sub.sort_values("time_rel").copy()

            # 3-frame rolling median on raw speed (diagnostic only)
            sub["speed_smooth"] = (
                sub["speed"]
                .rolling(window=3, center=True, min_periods=1)
                .median()
            )

            pre  = sub[sub["time_rel"] < 0]["speed_smooth"]
            post = sub[sub["time_rel"] >= 0]["speed_smooth"]

            if pre.empty or post.empty or pre.isna().all() or post.isna().all():
                continue

            baseline_speed = pre.mean()
            if baseline_speed <= 0 or np.isnan(baseline_speed):
                continue

            min_post_speed = post.min()
            frac_retained  = min_post_speed / baseline_speed  # 0–1 scale

            # Exclude animals whose post-stim minimum is faster than baseline
            # (frac > 1.0): these are non-responders or animals with a bad
            # baseline estimate; they distort the axes and the regression.
            if frac_retained > 1.0:
                n_excluded += 1
                continue

            records.append({
                "animal_id":            aid,
                "sex":                  sex_val,
                "treatment":            gname[group_cols.index("treatment")]       if "treatment"       in group_cols else "",
                "strain_genotype":      gname[group_cols.index("strain_genotype")] if "strain_genotype" in group_cols else "",
                "baseline_speed_mm_s":  baseline_speed,
                "min_post_speed_mm_s":  min_post_speed,
                "frac_baseline_at_min": frac_retained,
                "pct_max_slow":         1.0 - frac_retained,
                "abs_drop_mm_s":        baseline_speed - min_post_speed,
            })

        if n_excluded > 0:
            _emit(f"  {gname_str}: excluded {n_excluded} non-responder(s) (frac > 1.0)")

        if records:
            all_group_data[gname_str] = pd.DataFrame(records)
        else:
            _emit(f"  WARNING: no valid animals remaining for diagnostic in {gname_str}")

    if not all_group_data:
        _emit("  WARNING: no diagnostic data collected across any group.")
        return

    # Shared x-axis limit derived from all groups after filtering
    global_x_max = max(d["baseline_speed_mm_s"].max() for d in all_group_data.values())
    x_lim = (0, global_x_max * 1.05)

    # Shared y-axis limit for the absolute drop plots
    global_drop_max = max(d["abs_drop_mm_s"].max() for d in all_group_data.values())
    y_lim_drop = (0, global_drop_max * 1.05)

    # ── PASS 2: export CSVs and draw plots ──────────────────────────────────
    for gname_str, diag_df in all_group_data.items():
        _emit(f"  Diagnostic plot: {gname_str}")

        if len(diag_df) >= 3:
            _r, _p = scipy_stats.pearsonr(
                diag_df["baseline_speed_mm_s"].values,
                diag_df["pct_max_slow"].values
            )
        else:
            _r, _p = np.nan, np.nan

        diag_df["group_pearson_r"] = _r
        diag_df["group_pearson_p"] = _p

        csv_path = os.path.join(diag_dir, f"{gname_str}_baseline_vs_maxslow.csv")
        diag_df.to_csv(csv_path, index=False)

        dot_colors  = [_sex_color(s) for s in diag_df["sex"]]
        edge_colors = [_sex_edge(s)  for s in diag_df["sex"]]

        fig, ax = plt.subplots(figsize=(7, 5))
        n = len(diag_df)

        ax.scatter(
            diag_df["baseline_speed_mm_s"],
            diag_df["pct_max_slow"],
            c=dot_colors, edgecolors=edge_colors,
            alpha=0.80, s=60, linewidths=0.8, zorder=3
        )

        if n >= 3:
            x = diag_df["baseline_speed_mm_s"].values
            y = diag_df["pct_max_slow"].values
            m, b_int = np.polyfit(x, y, 1)
            x_line = np.linspace(0, x_lim[1], 200)
            ax.plot(x_line, m * x_line + b_int, color="dimgray", lw=1.5,
                    ls="--", alpha=0.8, label=f"Linear fit (slope={m:.3f})")

            p_str = "p < 0.001" if _p < 0.001 else f"p = {_p:.3f}"
            ax.text(
                0.97, 0.03,
                f"Pearson r = {_r:.3f}\n{p_str}\nn = {n}",
                transform=ax.transAxes,
                fontsize=10, va="bottom", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            )

        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markersize=9,
                   markerfacecolor="#E8748A", markeredgecolor="#A03050",
                   label="Hermaphrodite"),
            Line2D([0], [0], marker="o", color="w", markersize=9,
                   markerfacecolor="#6EB5D8", markeredgecolor="#2A6A9A",
                   label="Male"),
        ]
        ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

        ax.set_xlim(x_lim)
        ax.set_ylim(0, 1)
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("Pre-stimulus baseline speed (mm/s)", fontsize=12)
        ax.set_ylabel("% max slowing\n(1 − trough/baseline)", fontsize=12)
        ax.set_title(
            f"Baseline speed vs. % max slowing\n{gname_str}",
            fontsize=13, fontweight="bold"
        )
        ax.grid(alpha=0.3)

        ax.text(
            0.03, 0.97,
            "Higher value = greater slowing  |  non-responders (trough > baseline) excluded\n"
            "(3-frame rolling median, post-stim minimum)",
            transform=ax.transAxes,
            fontsize=8, va="top", ha="left", color="gray", style="italic"
        )

        plt.tight_layout()
        png_path = os.path.join(diag_dir, f"{gname_str}_baseline_vs_maxslow.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()

        # ── SECOND DIAGNOSTIC: absolute speed drop vs baseline ───────────────
        # y = baseline_speed - min_post_speed  (mm/s)
        # Answers: do faster animals drop more mm/s, or is the drop proportional?
        # If this is flat while the fraction plot slopes up → floor effect.
        # If both slope up → faster animals genuinely respond more in absolute terms.

        fig2, ax2 = plt.subplots(figsize=(7, 5))

        ax2.scatter(
            diag_df["baseline_speed_mm_s"],
            diag_df["abs_drop_mm_s"],
            c=dot_colors, edgecolors=edge_colors,
            alpha=0.80, s=60, linewidths=0.8, zorder=3
        )

        if n >= 3:
            x2 = diag_df["baseline_speed_mm_s"].values
            y2 = diag_df["abs_drop_mm_s"].values
            m2, b2 = np.polyfit(x2, y2, 1)
            x2_line = np.linspace(0, x_lim[1], 200)
            ax2.plot(x2_line, m2 * x2_line + b2, color="dimgray", lw=1.5,
                     ls="--", alpha=0.8, label=f"Linear fit (slope={m2:.3f})")

            r2, p2 = scipy_stats.pearsonr(x2, y2)
            p2_str = "p < 0.001" if p2 < 0.001 else f"p = {p2:.3f}"
            ax2.text(
                0.97, 0.97,
                f"Pearson r = {r2:.3f}\n{p2_str}\nn = {n}",
                transform=ax2.transAxes,
                fontsize=10, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            )

        ax2.legend(handles=legend_handles, fontsize=9, loc="upper left")
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim_drop)
        ax2.set_xlabel("Pre-stimulus baseline speed (mm/s)", fontsize=12)
        ax2.set_ylabel("Absolute speed drop\n(baseline − trough, mm/s)", fontsize=12)
        ax2.set_title(
            f"Baseline speed vs. absolute drop\n{gname_str}",
            fontsize=13, fontweight="bold"
        )
        ax2.grid(alpha=0.3)

        ax2.text(
            0.03, 0.03,
            "Higher value = greater absolute slowing  |  non-responders (trough > baseline) excluded\n"
            "(3-frame rolling median, post-stim minimum)",
            transform=ax2.transAxes,
            fontsize=8, va="bottom", ha="left", color="gray", style="italic"
        )

        plt.tight_layout()
        png2_path = os.path.join(diag_dir, f"{gname_str}_baseline_vs_absdrop.png")
        plt.savefig(png2_path, dpi=150, bbox_inches="tight")
        plt.close()

    _emit("Baseline-vs-maxslow diagnostics complete.")

    # ── PASS 3: combined sex plots per treatment × genotype ─────────────────
    # Merge all per-group data into one frame, then re-group by treatment +
    # genotype (dropping sex) so herms and males appear together on one plot.
    _emit("Generating combined-sex diagnostic plots...")

    all_records = pd.concat(all_group_data.values(), ignore_index=True)

    combo_groups = all_records.groupby(["treatment", "strain_genotype"])

    from matplotlib.lines import Line2D
    combo_legend_handles = [
        Line2D([0], [0], marker="o", color="w", markersize=9,
               markerfacecolor="#E8748A", markeredgecolor="#A03050",
               label="Hermaphrodite"),
        Line2D([0], [0], marker="o", color="w", markersize=9,
               markerfacecolor="#6EB5D8", markeredgecolor="#2A6A9A",
               label="Male"),
    ]

    for (treatment, genotype), combo_df in combo_groups:
        combo_str = f"{treatment}__{genotype}"
        _emit(f"  Combined plot: {combo_str}")

        n_combo = len(combo_df)
        dot_colors_c  = [_sex_color(s) for s in combo_df["sex"]]
        edge_colors_c = [_sex_edge(s)  for s in combo_df["sex"]]

        # ── combined % max slowing ──────────────────────────────────────────
        fig_c, ax_c = plt.subplots(figsize=(7, 5))

        ax_c.scatter(
            combo_df["baseline_speed_mm_s"],
            combo_df["pct_max_slow"],
            c=dot_colors_c, edgecolors=edge_colors_c,
            alpha=0.80, s=60, linewidths=0.8, zorder=3
        )

        # Per-sex regression lines and stats (bottom-right box)
        stats_lines_c = []
        for sex_label, sex_data in combo_df.groupby("sex"):
            if len(sex_data) >= 3:
                xs = sex_data["baseline_speed_mm_s"].values
                ys = sex_data["pct_max_slow"].values
                ms, bs = np.polyfit(xs, ys, 1)
                x_fit = np.linspace(0, x_lim[1], 200)
                ax_c.plot(x_fit, ms * x_fit + bs,
                          color=_sex_color(sex_label), lw=1.5, ls="--", alpha=0.7)
                rs, ps = scipy_stats.pearsonr(xs, ys)
                ps_str = "p < 0.001" if ps < 0.001 else f"p = {ps:.3f}"
                sex_short = "Herm" if ("herm" in str(sex_label).lower() or str(sex_label).lower() == "h") else "Male"
                stats_lines_c.append(
                    f"{sex_short}: r = {rs:.3f}, {ps_str}, n = {len(sex_data)}"
                )

        if stats_lines_c:
            ax_c.text(
                0.97, 0.20,
                "\n".join(stats_lines_c),
                transform=ax_c.transAxes,
                fontsize=9, va="bottom", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            )

        ax_c.legend(handles=combo_legend_handles, fontsize=9,
                    loc="lower right", bbox_to_anchor=(1.0, 0.0))
        ax_c.set_xlim(x_lim)
        ax_c.set_ylim(0, 1)
        ax_c.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        ax_c.set_xlabel("Pre-stimulus baseline speed (mm/s)", fontsize=12)
        ax_c.set_ylabel("% max slowing\n(1 − trough/baseline)", fontsize=12)
        ax_c.set_title(
            f"Baseline speed vs. % max slowing\n{combo_str}  [H + M combined]",
            fontsize=13, fontweight="bold"
        )
        ax_c.grid(alpha=0.3)
        ax_c.text(
            0.03, 0.97,
            "Higher value = greater slowing  |  non-responders excluded\n"
            "Dashed lines = per-sex regression",
            transform=ax_c.transAxes,
            fontsize=8, va="top", ha="left", color="gray", style="italic"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, f"{combo_str}__combined_baseline_vs_maxslow.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # ── combined absolute drop ──────────────────────────────────────────
        fig_d, ax_d = plt.subplots(figsize=(7, 5))

        ax_d.scatter(
            combo_df["baseline_speed_mm_s"],
            combo_df["abs_drop_mm_s"],
            c=dot_colors_c, edgecolors=edge_colors_c,
            alpha=0.80, s=60, linewidths=0.8, zorder=3
        )

        # Per-sex regression lines and stats (bottom-right box)
        stats_lines_d = []
        for sex_label, sex_data in combo_df.groupby("sex"):
            if len(sex_data) >= 3:
                xs = sex_data["baseline_speed_mm_s"].values
                ys = sex_data["abs_drop_mm_s"].values
                ms, bs = np.polyfit(xs, ys, 1)
                x_fit = np.linspace(0, x_lim[1], 200)
                ax_d.plot(x_fit, ms * x_fit + bs,
                          color=_sex_color(sex_label), lw=1.5, ls="--", alpha=0.7)
                rs, ps = scipy_stats.pearsonr(xs, ys)
                ps_str = "p < 0.001" if ps < 0.001 else f"p = {ps:.3f}"
                sex_short = "Herm" if ("herm" in str(sex_label).lower() or str(sex_label).lower() == "h") else "Male"
                stats_lines_d.append(
                    f"{sex_short}: r = {rs:.3f}, {ps_str}, n = {len(sex_data)}"
                )

        if stats_lines_d:
            ax_d.text(
                0.97, 0.20,
                "\n".join(stats_lines_d),
                transform=ax_d.transAxes,
                fontsize=9, va="bottom", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            )

        ax_d.legend(handles=combo_legend_handles, fontsize=9,
                    loc="lower right", bbox_to_anchor=(1.0, 0.0))
        ax_d.set_xlim(x_lim)
        ax_d.set_ylim(y_lim_drop)
        ax_d.set_xlabel("Pre-stimulus baseline speed (mm/s)", fontsize=12)
        ax_d.set_ylabel("Absolute speed drop\n(baseline − trough, mm/s)", fontsize=12)
        ax_d.set_title(
            f"Baseline speed vs. absolute drop\n{combo_str}  [H + M combined]",
            fontsize=13, fontweight="bold"
        )
        ax_d.grid(alpha=0.3)
        ax_d.text(
            0.03, 0.97,
            "Higher value = greater absolute slowing  |  non-responders excluded\n"
            "Dashed lines = per-sex regression",
            transform=ax_d.transAxes,
            fontsize=8, va="top", ha="left", color="gray", style="italic"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, f"{combo_str}__combined_baseline_vs_absdrop.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    _emit("Combined-sex diagnostic plots complete.")


class AnalysisWorker(QThread):
    """Worker thread to run analysis without blocking GUI"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """Run the analysis in a separate thread"""
        try:
            self.progress.emit("Loading data...")
            df = pd.read_csv(self.params["file"])

            # Build animal_id
            if "assay_num" not in df.columns or "track_num" not in df.columns:
                self.finished.emit(False, "CSV must contain 'assay_num' and 'track_num'")
                return

            df["animal_id"] = df["assay_num"].astype(str) + "__" + df["track_num"].astype(str)

            # Validate columns
            required_cols = ["animal_id", "time", "x", "y", "stim", "treatment", "sex", "strain_genotype"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                self.finished.emit(False, f"Missing columns: {missing}")
                return

            # Compute speed
            self.progress.emit("Computing speeds...")
            df["dx"] = df.groupby("animal_id")["x"].diff()
            df["dy"] = df.groupby("animal_id")["y"].diff()
            df["dt"] = df.groupby("animal_id")["time"].diff()

            df["distance_pix"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
            df["distance_mm"] = df["distance_pix"] / self.params["pixels_per_mm"]
            df["speed"] = df["distance_mm"] / df["dt"]
            df["speed"] = df["speed"].replace([np.inf, -np.inf], np.nan)

            # Align to stim
            self.progress.emit("Aligning to stimulus...")
            aligned = []
            excluded = []

            for aid, sub in df.groupby("animal_id"):
                stim_times = sub.loc[sub["stim"] == 1, "time"]
                if stim_times.empty:
                    excluded.append(aid)
                    continue

                stim_t = stim_times.iloc[0]
                sub = sub.copy()
                sub["time_rel"] = sub["time"] - stim_t

                pre_vals = sub.loc[sub["time_rel"] < 0, "time_rel"]
                post_vals = sub.loc[sub["time_rel"] > 0, "time_rel"]

                pre_ok = len(pre_vals) > 0 and pre_vals.min() <= -self.params["min_pre"]
                post_ok = len(post_vals) > 0 and post_vals.max() >= self.params["min_post"]

                if not (pre_ok and post_ok):
                    excluded.append(aid)
                    continue

                aligned.append(sub)

            if len(aligned) == 0:
                self.finished.emit(False, "No animals passed filtering")
                return

            df = pd.concat(aligned)
            self.progress.emit(f"Excluded: {len(excluded)}, Included: {len(df['animal_id'].unique())}")

            # Normalize
            self.progress.emit("Normalizing to baseline...")
            baseline_map = df[df["time_rel"] < 0].groupby("animal_id")["speed"].mean()
            df["baseline"] = df["animal_id"].map(baseline_map)
            df["speed_norm"] = df["speed"] / df["baseline"]

            # Group and analyze
            group_cols = ["treatment", "sex", "strain_genotype"]
            os.makedirs(self.params["outdir"], exist_ok=True)
            group_summaries = []

            for gname, g in df.groupby(group_cols):
                gname_str = "__".join(map(str, gname))
                self.progress.emit(f"Processing: {gname_str}")

                animals = g["animal_id"].unique()
                traces = {}

                for aid in animals:
                    sub = g[g["animal_id"] == aid]
                    window = sub[(sub["time_rel"] >= -self.params["pre_window"]) &
                                 (sub["time_rel"] <= self.params["post_window"])]
                    traces[aid] = window.set_index("time_rel")["speed_norm"]

                time_grid = sorted(set(np.concatenate([t.index.values for t in traces.values()])))
                mean_trace = pd.DataFrame(traces).reindex(time_grid).mean(axis=1)

                # Export normalized traces in long format for visualization
                trace_records = []
                for aid in animals:
                    sub = g[g["animal_id"] == aid]
                    window = sub[(sub["time_rel"] >= -self.params["pre_window"]) &
                                 (sub["time_rel"] <= self.params["post_window"])]

                    for _, row in window.iterrows():
                        trace_records.append({
                            'animal_id': aid,
                            'time_rel': row['time_rel'],
                            'speed_norm': row['speed_norm'],
                            'treatment': gname[0],
                            'sex': gname[1],
                            'genotype': gname[2]
                        })

                if trace_records:
                    trace_df = pd.DataFrame(trace_records)
                    trace_df.to_csv(f"{self.params['outdir']}/{gname_str}_normalized_traces.csv", index=False)

                # Raw speed plot
                plt.figure(figsize=(10,6))
                raw_traces = {}
                for aid in animals:
                    sub = g[g["animal_id"] == aid]
                    window = sub[(sub["time_rel"] >= -self.params["pre_window"]) &
                                 (sub["time_rel"] <= self.params["post_window"])]
                    raw_traces[aid] = window.set_index("time_rel")["speed"]

                for aid in animals:
                    plt.plot(raw_traces[aid].index, raw_traces[aid].values, alpha=0.15, color="gray")

                mean_raw = pd.DataFrame(raw_traces).reindex(time_grid).mean(axis=1)
                plt.plot(mean_raw.index, mean_raw.values, color="black", lw=2.5, label="Mean", zorder=5)
                plt.axvline(0, color="red", ls="--", lw=2, label="Stimulus", zorder=6)
                plt.xlabel("Time relative to stimulus (s)", fontsize=12)
                plt.ylabel("Speed (mm/s)", fontsize=12)
                plt.title(f"RAW SPEED: {gname_str}", fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig(f"{self.params['outdir']}/{gname_str}_raw_qc.png", dpi=150, bbox_inches="tight")
                plt.close()

                # ── Vectorized bootstrap ──────────────────────────────────────────
                self.progress.emit(f"  Running {self.params['boot']} bootstrap iterations (vectorized)...")

                animal_keys = list(traces.keys())
                n_animals = len(animal_keys)
                n_boot = self.params['boot']
                time_grid_arr = np.array(time_grid)
                n_grid = len(time_grid_arr)

                # Align every animal onto the common grid once.
                # Using reindex preserves NaN for time points missing from that animal's
                # trace — identical to the original per-iteration pd.DataFrame.reindex behavior.
                animal_matrix = np.full((n_animals, n_grid), np.nan)
                for k, aid in enumerate(animal_keys):
                    animal_matrix[k] = traces[aid].reindex(pd.Index(time_grid_arr)).values

                # Draw all sample indices at once: shape (n_boot, n_animals)
                sample_idx = np.random.randint(0, n_animals, size=(n_boot, n_animals))

                # Compute all mean traces in bounded-memory chunks.
                # animal_matrix[chunk_idx] → (chunk, n_animals, n_grid); mean over axis=1 → (chunk, n_grid)
                CHUNK = 500
                all_mean_traces = np.empty((n_boot, n_grid))
                for c_start in range(0, n_boot, CHUNK):
                    c_end = min(c_start + CHUNK, n_boot)
                    chunk_idx = sample_idx[c_start:c_end]
                    all_mean_traces[c_start:c_end] = np.nanmean(animal_matrix[chunk_idx], axis=1)

                fit_args = [(i, all_mean_traces[i], time_grid_arr) for i in range(n_boot)]
                with Pool(processes=self.params['n_cores']) as pool:
                    results = pool.map(_fit_replicate_precomputed, fit_args)
                # ─────────────────────────────────────────────────────────────────

                boot_params = [r for r in results if r is not None]
                for r in boot_params:
                    r["group"] = gname_str

                self.progress.emit(f"  Successful fits: {len(boot_params)}/{self.params['boot']}")

                if len(boot_params) == 0:
                    self.progress.emit(f"  WARNING: No fits for {gname_str}!")
                    continue

                boot_df = pd.DataFrame(boot_params)
                boot_df.to_csv(f"{self.params['outdir']}/{gname_str}_bootstrap_params.csv", index=False)

                # Normalized + fit plot
                post_times = mean_trace.index[mean_trace.index >= 0]

                if len(boot_df) > 0:
                    med = boot_df.median(numeric_only=True)

                    fit_curve = decay_recovery_trajectory_model(
                        post_times,
                        med["y_min_param"],
                        med["tau_decay"],
                        med["A_rec"],
                        med["tau_rec"]
                    )

                    fit_decay = med["y_min_param"] + (1.0 - med["y_min_param"]) * np.exp(-post_times / med["tau_decay"])
                    fit_recovery = med["y_min_param"] + med["A_rec"] * (1 - np.exp(-post_times / med["tau_rec"]))

                plt.figure(figsize=(10,6))

                for aid in animals:
                    plt.plot(traces[aid].index, traces[aid].values, alpha=0.1, color="steelblue")

                plt.plot(mean_trace.index, mean_trace.values, color="black", lw=2.5, label="Mean data", zorder=5)
                plt.plot(post_times, fit_curve, color="red", lw=2.5, ls="-", label="Trajectory fit", zorder=6)
                plt.plot(post_times, fit_decay, color="orange", lw=1.5, ls=":", label="Decay component", alpha=0.7)
                plt.plot(post_times, fit_recovery, color="green", lw=1.5, ls=":", label="Recovery trajectory", alpha=0.7)

                plt.axvline(0, color="red", ls="--", lw=2, alpha=0.7)
                plt.axhline(1.0, color="gray", ls=":", alpha=0.5, label="Baseline")
                plt.xlabel("Time relative to stimulus (s)", fontsize=12)
                plt.ylabel("Normalized speed", fontsize=12)
                plt.title(f"NORMALIZED + FIT: {gname_str}", fontsize=14, fontweight='bold')
                plt.legend(fontsize=9, loc='upper left')
                plt.ylim(0, max(2.0, mean_trace.max() * 1.1))
                plt.grid(alpha=0.3)

                # Add equation text box with fitted parameters
                equation_text = (
                    f"Fitted Parameters (median):\n"
                    f"y_min = {med['y_min_param']:.4f}\n"
                    f"τ_decay = {med['tau_decay']:.2f}s\n"
                    f"A_rec = {med['A_rec']:.3f}\n"
                    f"τ_rec = {med['tau_rec']:.2f}s\n"
                    f"\nActual Trajectory:\n"
                    f"Min = {med['y_min_actual']:.3f} @ {med['t_at_min']:.1f}s\n"
                    f"Min speed = {med['drop_depth']:.1%} baseline\n"
                    f"t½_decay = {med['t_half_decay_actual']:.2f}s\n"
                    f"t½_rec = {med['t_half_recovery_actual']:.2f}s\n"
                    f"\nTrough Widths:\n"
                    f"FWHM (50%) = {med['trough_width_50']:.1f}s\n"
                    f"@ 75% = {med['trough_width_75']:.1f}s\n"
                    f"@ 25% = {med['trough_width_25']:.1f}s"
                )
                plt.text(0.98, 0.02, equation_text, transform=plt.gca().transAxes,
                        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                plt.savefig(f"{self.params['outdir']}/{gname_str}_normalized_fit.png", dpi=150, bbox_inches="tight")
                plt.close()

                # Summary stats
                summary = {
                    "group": gname_str,
                    "n_animals": len(traces),
                    "baseline_speed_mm_s": mean_raw[mean_raw.index < 0].mean()
                }

                for col in ["y_min_param", "y_min_actual", "t_at_min", "tau_decay", "A_rec", "tau_rec",
                            "drop_depth", "recovery_at_end", "recovery_completeness",
                            "t_half_decay_param", "t_half_decay_actual",
                            "t_half_recovery_param", "t_half_recovery_actual",
                            "trough_width_50", "trough_width_75", "trough_width_25",
                            "t_enter_50", "t_exit_50"]:
                    if col in boot_df.columns:
                        vals = boot_df[col].dropna()
                        if len(vals) > 0:
                            summary[f"{col}_median"] = vals.median()
                            summary[f"{col}_mean"] = vals.mean()
                            summary[f"{col}_sd"] = vals.std()
                            summary[f"{col}_sem"] = vals.sem()
                            summary[f"{col}_ci_low"] = vals.quantile(0.025)
                            summary[f"{col}_ci_high"] = vals.quantile(0.975)

                group_summaries.append(summary)

            # Save summary
            summary_df = pd.DataFrame(group_summaries)
            summary_df.to_csv(f"{self.params['outdir']}/group_param_summary.csv", index=False)

            # ============================================================
            # CREATE MASTER LONG-FORMAT FILE WITH ALL BOOTSTRAP RESULTS
            # ============================================================
            self.progress.emit("Creating master bootstrap file...")

            # Load all individual bootstrap files
            all_bootstrap = []
            for csv_file in Path(self.params['outdir']).glob("*_bootstrap_params.csv"):
                df_boot = pd.read_csv(csv_file)
                all_bootstrap.append(df_boot)

            if all_bootstrap:
                master_df = pd.concat(all_bootstrap, ignore_index=True)

                # Parse group column to extract metadata
                group_split = master_df['group'].str.split('__', expand=True)
                master_df['treatment'] = group_split[0]
                master_df['sex'] = group_split[1]
                master_df['genotype'] = group_split[2]

                # Reorder columns: metadata first, then parameters
                metadata_cols = ['group', 'treatment', 'sex', 'genotype', 'iter']
                param_cols = [col for col in master_df.columns if col not in metadata_cols]
                master_df = master_df[metadata_cols + param_cols]

                # Save master file
                master_file = f"{self.params['outdir']}/all_bootstrap_results_long.csv"
                master_df.to_csv(master_file, index=False)

                self.progress.emit(f"Master file created: {master_df.shape[0]} rows × {master_df.shape[1]} columns")

            # ============================================================
            # DIAGNOSTIC: baseline speed vs. % max slowing (per group)
            # ============================================================
            export_baseline_maxslow_diagnostic(
                df=df,
                group_cols=group_cols,
                outdir=self.params["outdir"],
                progress_fn=self.progress.emit
            )

            self.finished.emit(True,
                             f"Analysis complete!\n"
                             f"Outputs saved to: {self.params['outdir']}\n"
                             f"Master bootstrap file: all_bootstrap_results_long.csv\n"
                             f"Diagnostics: {self.params['outdir']}/diagnostics/")

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class BootstrapGUI(QMainWindow):
    """Main GUI window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bootstrap Decay-Recovery Fitting")
        self.setGeometry(100, 100, 600, 500)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # File selection
        file_group = QGroupBox("Input/Output")
        file_layout = QFormLayout()

        self.file_edit = QLineEdit()
        file_button = QPushButton("Browse...")
        file_button.clicked.connect(self.browse_file)
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_edit)
        file_row.addWidget(file_button)
        file_layout.addRow("Input CSV:", file_row)

        self.output_edit = QLineEdit("bootstrap_output")
        output_button = QPushButton("Browse...")
        output_button.clicked.connect(self.browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(output_button)
        file_layout.addRow("Output Directory:", output_row)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()

        self.pixels_spin = QDoubleSpinBox()
        self.pixels_spin.setRange(1, 1000)
        self.pixels_spin.setValue(104.0)
        self.pixels_spin.setDecimals(1)
        param_layout.addRow("Pixels per mm:", self.pixels_spin)

        self.pre_window_spin = QDoubleSpinBox()
        self.pre_window_spin.setRange(1, 100)
        self.pre_window_spin.setValue(10.0)
        param_layout.addRow("Pre-stim window (s):", self.pre_window_spin)

        self.post_window_spin = QDoubleSpinBox()
        self.post_window_spin.setRange(1, 300)
        self.post_window_spin.setValue(60.0)
        param_layout.addRow("Post-stim window (s):", self.post_window_spin)

        self.min_pre_spin = QDoubleSpinBox()
        self.min_pre_spin.setRange(1, 100)
        self.min_pre_spin.setValue(10.0)
        param_layout.addRow("Min pre-stim data (s):", self.min_pre_spin)

        self.min_post_spin = QDoubleSpinBox()
        self.min_post_spin.setRange(1, 300)
        self.min_post_spin.setValue(10.0)
        param_layout.addRow("Min post-stim data (s):", self.min_post_spin)

        self.boot_spin = QSpinBox()
        self.boot_spin.setRange(100, 10000)
        self.boot_spin.setValue(1000)
        self.boot_spin.setSingleStep(100)
        param_layout.addRow("Bootstrap iterations:", self.boot_spin)

        self.cores_spin = QSpinBox()
        self.cores_spin.setRange(1, cpu_count())
        self.cores_spin.setValue(max(1, cpu_count() - 1))
        param_layout.addRow("CPU cores:", self.cores_spin)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Progress display
        self.progress_label = QLabel("Ready")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
        layout.addWidget(self.progress_label)

        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        self.run_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_button)

        self.worker = None

    def browse_file(self):
        """Browse for input CSV file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Select Input CSV", "", "CSV Files (*.csv)")
        if filename:
            self.file_edit.setText(filename)

    def browse_output(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_edit.setText(directory)

    def run_analysis(self):
        """Start the analysis"""
        # Validate input
        if not self.file_edit.text():
            QMessageBox.warning(self, "Error", "Please select an input CSV file")
            return

        # Get parameters
        params = {
            'file': self.file_edit.text(),
            'outdir': self.output_edit.text(),
            'pixels_per_mm': self.pixels_spin.value(),
            'pre_window': self.pre_window_spin.value(),
            'post_window': self.post_window_spin.value(),
            'min_pre': self.min_pre_spin.value(),
            'min_post': self.min_post_spin.value(),
            'boot': self.boot_spin.value(),
            'n_cores': self.cores_spin.value()
        }

        # Disable button
        self.run_button.setEnabled(False)
        self.progress_label.setText("Starting analysis...")

        # Start worker thread
        self.worker = AnalysisWorker(params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()

    def update_progress(self, message):
        """Update progress display"""
        self.progress_label.setText(message)

    def analysis_finished(self, success, message):
        """Handle analysis completion"""
        self.run_button.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
            self.progress_label.setText("Analysis complete!")
        else:
            QMessageBox.critical(self, "Error", message)
            self.progress_label.setText("Analysis failed")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BootstrapGUI()
    window.show()
    sys.exit(app.exec_())
