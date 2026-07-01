"""
graphs.py — Publication Figures for the Centroid-Resolution Degradation Study

Generates seven publication-quality figures from pre-computed results in
results_resolution/. All plots can be regenerated independently of the
expensive pipeline stages — this module only reads CSVs and is safe to re-run.

Figures produced (in graphs/ subdir):
  fig1_cnn_auc.png            — CNN AUC vs scale (PSF off/on + 3 ablation lines)
  fig2_bryson_binary.png      — Bryson binary: accuracy / F1 / recall_fp
  fig3_bryson_continuous.png  — Bryson continuous: AUC PSF off/on
  fig4_cnn_auc_threshold.png  — fig1 + 0.90 threshold line + crossing annotation
  fig5_bryson_binary_threshold.png  — fig2 + accuracy CI + 0.90 threshold + annotation
  fig6_bryson_continuous_threshold.png — fig3 + 0.90 threshold + crossing annotation
  fig7_centroid_signal_comparison.png — Bryson continuous vs. centroid-only CNN AUC

Styling conventions:
  PSF off (rebinning only)    — solid lines, full opacity
  PSF on  (rebin + PSF blur)  — same hue, dashed, α=0.4 where colour overlaps
  Flux-only ablation          — green (#6acc65), solid, labelled "Flux only"
  Centroid-only ablation      — red (#d65f5f), solid, labelled "Centroid only"
  Threshold line              — dashed dark-grey (#555555)
  95% CI band                 — fill_between, α=0.15

Missing data columns → log warning + skip the affected line; never crash.

Usage:
  python graphs.py [--results-dir results_resolution] [--format png|pdf]
"""

import argparse
import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

KEPLER_SCALE     = 3.98                         # arcsec/pixel at k=1
K_VALUES         = [1, 2, 3, 4, 5]
EFFECTIVE_SCALES = [k * KEPLER_SCALE for k in K_VALUES]   # [3.98, 7.96, …, 19.9]

AUC_THRESHOLD    = 0.90                         # absolute threshold for k_break_90 plots
MODALITY_CONVERGENCE_EPSILON = 0.01             # |combined_auc - flux_only_auc| convergence tol
MIN_SAMPLES_FOR_PLOT = 10                       # matches baselines.py's hardcoded len(valid)<10

# Colour palette (Colorbrewer-safe, colour-blind tested)
COLOR_PSF_OFF          = "#4878cf"   # blue  — combined CNN / Bryson, PSF off
COLOR_PSF_ON           = "#e07b54"   # orange — PSF on variant
COLOR_FLUX_ONLY        = "#6acc65"   # green — flux-only ablation
COLOR_CENTROID_ONLY    = "#d65f5f"   # red   — centroid-only ablation
COLOR_THRESHOLD        = "#555555"   # dark grey — threshold reference line

# Matplotlib rcParams for a clean, paper-friendly style
plt.rcParams.update({
    "figure.dpi":           150,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.3,
    "grid.linestyle":       ":",
    "font.size":            11,
    "axes.labelsize":       11,
    "axes.titlesize":       12,
    "legend.fontsize":      9,
    "legend.framealpha":    0.8,
    "lines.linewidth":      1.8,
    "lines.markersize":     6,
})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _warn_missing(col: str, source: str) -> None:
    """Log a warning when a required column is absent; used to skip lines gracefully."""
    warnings.warn(
        f"Column '{col}' not found in {source}. Line will be skipped.",
        UserWarning, stacklevel=3
    )


def _save(fig: plt.Figure, path: str, fmt: str = "png") -> None:
    out = path if path.endswith(f".{fmt}") else f"{path}.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight", format=fmt)
    plt.close(fig)
    print(f"  → {out}")


def _find_first_crossing(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    direction: str = "below",
) -> float | None:
    """
    Linear interpolation to find where y crosses `threshold` from above (direction='below')
    or from below (direction='above'). Returns the x-value at the crossing, or None if
    no crossing is found within the data range.
    """
    for i in range(len(y) - 1):
        if direction == "below":
            if y[i] >= threshold > y[i + 1]:
                # Interpolate
                t = (threshold - y[i]) / (y[i + 1] - y[i])
                return float(x[i] + t * (x[i + 1] - x[i]))
        else:
            if y[i] <= threshold < y[i + 1]:
                t = (threshold - y[i]) / (y[i + 1] - y[i])
                return float(x[i] + t * (x[i + 1] - x[i]))
    return None


def _annotate_crossing(
    ax: plt.Axes,
    x_cross: float | None,
    y_threshold: float,
    label_suffix: str = "",
) -> None:
    """
    Draw a vertical dashed line at x_cross with a text annotation showing
    "↓ X.XX″ (k=N)" where N is the k-value. If x_cross is None, draw a
    text box instead: "AUC does not fall below {y_threshold}".
    """
    if x_cross is not None:
        k_approx = round(x_cross / KEPLER_SCALE)
        ax.axvline(x_cross, color=COLOR_THRESHOLD, linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate(
            f"↓ {x_cross:.2f}″ (k={k_approx}){label_suffix}",
            xy=(x_cross, y_threshold),
            xytext=(x_cross + 0.5, y_threshold + 0.03),
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=COLOR_THRESHOLD, lw=0.9),
            color=COLOR_THRESHOLD,
        )
    else:
        # No crossing found — note it as a text box at the right edge
        ax.text(
            0.97, y_threshold + 0.02,
            f"Does not fall below {y_threshold:.2f}{label_suffix}",
            ha="right", va="bottom", fontsize=8, color=COLOR_THRESHOLD,
            transform=ax.get_yaxis_transform(),
        )


def _load_ablation_aucs(results_dir: str) -> dict[str, list[float | None]]:
    """
    Scan results_resolution/ for ablation score files (scores_k*_psf0_flux_only.csv,
    scores_k*_psf0_centroid_only.csv) and compute per-k AUC from each.

    Returns dict with keys 'flux_only' and 'centroid_only', each mapping to a list
    of AUC values aligned to K_VALUES (None where the file is missing).
    """
    from sklearn.metrics import roc_auc_score

    result: dict[str, list] = {"flux_only": [], "centroid_only": []}

    for variant in ["flux_only", "centroid_only"]:
        for k in K_VALUES:
            path = os.path.join(results_dir, f"scores_k{k}_psf0_{variant}.csv")
            if not os.path.exists(path):
                result[variant].append(None)
                continue
            try:
                df = pd.read_csv(path)
                if "label" not in df.columns or "score" not in df.columns:
                    result[variant].append(None)
                    continue
                y_true  = df["label"].values
                y_score = df["score"].values
                if len(np.unique(y_true)) < 2:
                    result[variant].append(None)
                    continue
                auc = float(roc_auc_score(y_true, y_score))
                result[variant].append(auc)
            except Exception as exc:
                print(f"  Ablation AUC load error ({path}): {exc}")
                result[variant].append(None)

    return result


def find_modality_convergence_k(
    combined_auc_by_k: dict[int, float],
    flux_only_auc_by_k: dict[int, float],
    epsilon: float = MODALITY_CONVERGENCE_EPSILON,
) -> dict:
    """
    Determine where (if ever) the combined-model AUC converges onto the flux-only
    ablation AUC ceiling, meaning centroid information stops adding measurable value
    even though the centroid channel is still present in the input.

    Not built on _find_first_crossing/_annotate_crossing — those are for "y crosses
    a fixed y-threshold," this is "two series converge onto each other."

    Algorithm: scan k descending; find the smallest k such that
    |combined[k] - flux_only[k]| <= epsilon holds for that k AND every larger k.

    Returns dict:
        converged_k: int | None
        status: "converges" | "persists" | "no_contribution"
            "converges"       — centroid helps up to some k, then AUC saturates onto
                                 the flux-only ceiling for all coarser k.
            "no_contribution" — converged_k is the smallest available k: combined
                                 never meaningfully exceeds flux-only, even natively.
            "persists"        — never converges through k_max: centroid information
                                 contributes across the whole tested degradation range.
    """
    usable_ks = sorted(
        k for k in combined_auc_by_k
        if k in flux_only_auc_by_k
        and combined_auc_by_k[k] is not None
        and flux_only_auc_by_k[k] is not None
    )
    if not usable_ks:
        print("  find_modality_convergence_k: no overlapping k values — skipping annotation")
        return {"converged_k": None, "status": "persists"}

    diffs = {k: abs(combined_auc_by_k[k] - flux_only_auc_by_k[k]) for k in usable_ks}

    converged_k = None
    for k in usable_ks:
        if all(diffs[k2] <= epsilon for k2 in usable_ks if k2 >= k):
            converged_k = k
            break

    if converged_k is None:
        return {"converged_k": None, "status": "persists"}
    if converged_k == min(usable_ks):
        return {"converged_k": converged_k, "status": "no_contribution"}
    return {"converged_k": converged_k, "status": "converges"}


def _annotate_modality_convergence(
    ax: plt.Axes, df: pd.DataFrame, ablation: dict[str, list]
) -> None:
    """
    Shared annotation logic for plot_cnn_auc and plot_cnn_auc_threshold: mark the k
    at which the combined-model (PSF-off) AUC converges onto the flux-only ablation
    AUC ceiling (see find_modality_convergence_k).
    """
    psf0 = df[df["psf"] == 0].sort_values("k")
    if psf0.empty or "cnn_auc" not in psf0.columns:
        return
    combined_by_k = dict(zip(psf0["k"], psf0["cnn_auc"]))
    flux_only_by_k = {k: a for k, a in zip(K_VALUES, ablation.get("flux_only", []))}

    conv = find_modality_convergence_k(combined_by_k, flux_only_by_k)

    if conv["status"] == "converges":
        x_conv = conv["converged_k"] * KEPLER_SCALE
        ax.axvline(x_conv, color=COLOR_THRESHOLD, linestyle=":", linewidth=1, alpha=0.6)
        ax.annotate(
            f"Combined ≈ flux-only from k={conv['converged_k']}",
            xy=(x_conv, 0.5), xytext=(x_conv + 0.5, 0.55),
            fontsize=8, arrowprops=dict(arrowstyle="->", color=COLOR_THRESHOLD, lw=0.9),
            color=COLOR_THRESHOLD,
        )
    elif conv["status"] == "persists":
        ax.text(0.03, 0.06, "Centroid signal contributes across all tested k",
                transform=ax.transAxes, fontsize=8, color=COLOR_THRESHOLD)
    elif conv["status"] == "no_contribution":
        ax.text(0.03, 0.06, "No measurable centroid contribution at any k",
                transform=ax.transAxes, fontsize=8, color=COLOR_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — CNN AUC vs scale (no threshold)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cnn_auc(df: pd.DataFrame, results_dir: str, out_path: str, fmt: str) -> None:
    """
    Figure 1: CNN AUC vs effective pixel scale.
    Lines: PSF off (solid blue), PSF on (dashed orange), flux-only (green),
           centroid-only (red). No 95% CI band or threshold line — those are
           reserved for fig4 (the threshold figure).

    Ablation lines are computed from psf=0 ablation score files if present;
    missing files are silently skipped.
    """
    required = ["k", "psf", "cnn_auc", "effective_scale_arcsec"]
    for col in required:
        if col not in df.columns:
            _warn_missing(col, "breakdown.csv")
            return

    ablation = _load_ablation_aucs(results_dir)

    fig, ax = plt.subplots(figsize=(7, 5))

    for psf_val, color, ls, alpha, label in [
        (0, COLOR_PSF_OFF, "-", 1.0, "CNN (PSF off)"),
        (1, COLOR_PSF_ON,  "--", 0.9, "CNN (PSF on)"),
    ]:
        sub = df[df["psf"] == psf_val].sort_values("k")
        if sub.empty:
            continue
        x = sub["effective_scale_arcsec"].values
        y = sub["cnn_auc"].values
        valid = ~np.isnan(y)
        if valid.any():
            ax.plot(x[valid], y[valid], color=color, linestyle=ls, marker="o",
                    alpha=alpha, label=label)

    # Ablation lines (psf=0 only)
    scales = EFFECTIVE_SCALES
    for variant, color, label in [
        ("flux_only",     COLOR_FLUX_ONLY,     "Flux only"),
        ("centroid_only", COLOR_CENTROID_ONLY,  "Centroid only"),
    ]:
        aucs = ablation.get(variant, [])
        if any(a is not None for a in aucs):
            x_abl = [scales[i] for i, a in enumerate(aucs) if a is not None]
            y_abl = [a for a in aucs if a is not None]
            ax.plot(x_abl, y_abl, color=color, linestyle="-", marker="s",
                    label=label)

    _annotate_modality_convergence(ax, df, ablation)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Random (0.5)")
    ax.set_xlabel("Effective pixel scale (arcsec/pixel)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("CNN classification performance vs. spatial sampling")
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save(fig, out_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Bryson binary metrics vs scale (no threshold)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bryson_binary(df: pd.DataFrame, out_path: str, fmt: str) -> None:
    """
    Figure 2: Bryson binary classifier metrics vs effective pixel scale.
    Three metric lines per PSF setting: accuracy (solid), F1-macro (dashed),
    recall_fp (dotted). PSF off = full opacity, PSF on = α=0.4.

    This is the PRIMARY Bryson result.
    """
    metric_styles = [
        ("bryson_binary_accuracy",  "-",  "Accuracy"),
        ("bryson_binary_f1_macro",  "--", "F1-macro"),
        ("bryson_binary_recall_fp", ":",  "Recall FP"),
    ]
    psf_configs = [
        (0, COLOR_PSF_OFF, 1.0, "PSF off"),
        (1, COLOR_PSF_ON,  0.4, "PSF on"),
    ]

    # Check at least one required metric column exists
    if not any(col in df.columns for col, _, _ in metric_styles):
        print("  Figure 2 (Bryson binary): required columns missing — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for psf_val, base_color, alpha, psf_label in psf_configs:
        sub = df[df["psf"] == psf_val].sort_values("k")
        if sub.empty:
            continue
        x = sub["effective_scale_arcsec"].values

        for col, ls, metric_label in metric_styles:
            if col not in sub.columns:
                _warn_missing(col, "breakdown.csv")
                continue
            y = sub[col].values.astype(float)
            valid = ~np.isnan(y)
            if not valid.any():
                continue
            label = f"{metric_label} ({psf_label})"
            ax.plot(x[valid], y[valid], color=base_color, linestyle=ls,
                    marker="o", alpha=alpha, label=label)

    ax.set_xlabel("Effective pixel scale (arcsec/pixel)")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Bryson binary classifier (PRIMARY) — metrics vs. spatial sampling")
    ax.legend(loc="lower left", ncol=2)
    plt.tight_layout()
    _save(fig, out_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Bryson continuous AUC vs scale (no threshold)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bryson_continuous(df: pd.DataFrame, out_path: str, fmt: str) -> None:
    """
    Figure 3: Bryson continuous-score AUC (negated SNR) vs effective pixel scale.
    PSF off = solid blue, PSF on = dashed orange.
    """
    required = ["bryson_auc", "effective_scale_arcsec", "psf"]
    for col in required:
        if col not in df.columns:
            _warn_missing(col, "breakdown.csv")
            return

    fig, ax = plt.subplots(figsize=(7, 5))

    for psf_val, color, ls, label in [
        (0, COLOR_PSF_OFF, "-",  "Bryson continuous (PSF off)"),
        (1, COLOR_PSF_ON,  "--", "Bryson continuous (PSF on)"),
    ]:
        sub = df[df["psf"] == psf_val].sort_values("k")
        if sub.empty:
            continue
        x = sub["effective_scale_arcsec"].values
        y = sub["bryson_auc"].values.astype(float)
        valid = ~np.isnan(y)
        if valid.any():
            ax.plot(x[valid], y[valid], color=color, linestyle=ls, marker="s", label=label)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Random (0.5)")
    ax.set_xlabel("Effective pixel scale (arcsec/pixel)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Bryson continuous classifier (secondary) — AUC vs. spatial sampling")
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save(fig, out_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — CNN AUC with threshold line + crossing annotation
# ─────────────────────────────────────────────────────────────────────────────

def plot_cnn_auc_threshold(
    df: pd.DataFrame, results_dir: str, out_path: str, fmt: str
) -> None:
    """
    Figure 4: CNN AUC vs scale with the absolute 0.90 threshold line.
    PSF-off 95% CI band added. First-crossing annotations per PSF setting.
    """
    required = ["k", "psf", "cnn_auc", "cnn_ci_lo", "cnn_ci_hi", "effective_scale_arcsec"]
    for col in required:
        if col not in df.columns:
            _warn_missing(col, "breakdown.csv")
            return

    ablation = _load_ablation_aucs(results_dir)

    fig, ax = plt.subplots(figsize=(7, 5))

    for psf_val, color, ls, alpha, label_base in [
        (0, COLOR_PSF_OFF, "-",  1.0, "CNN (PSF off)"),
        (1, COLOR_PSF_ON,  "--", 0.9, "CNN (PSF on)"),
    ]:
        sub = df[df["psf"] == psf_val].sort_values("k")
        if sub.empty:
            continue
        x = sub["effective_scale_arcsec"].values
        y = sub["cnn_auc"].values.astype(float)
        valid = ~np.isnan(y)
        if not valid.any():
            continue

        ax.plot(x[valid], y[valid], color=color, linestyle=ls, marker="o",
                alpha=alpha, label=label_base)

        # 95% CI band on PSF-off only (PSF-on is a robustness check)
        if psf_val == 0:
            y_lo = sub["cnn_ci_lo"].values.astype(float)
            y_hi = sub["cnn_ci_hi"].values.astype(float)
            ci_valid = valid & ~np.isnan(y_lo) & ~np.isnan(y_hi)
            if ci_valid.any():
                ax.fill_between(x[ci_valid], y_lo[ci_valid], y_hi[ci_valid],
                                color=color, alpha=0.15, label="95% CI (PSF off)")

        # Crossing annotation (PSF-off only to keep the figure uncluttered)
        if psf_val == 0:
            x_cross = _find_first_crossing(x[valid], y[valid], AUC_THRESHOLD)
            _annotate_crossing(ax, x_cross, AUC_THRESHOLD)

    # Ablation lines
    scales = EFFECTIVE_SCALES
    for variant, color, label in [
        ("flux_only",     COLOR_FLUX_ONLY,    "Flux only"),
        ("centroid_only", COLOR_CENTROID_ONLY, "Centroid only"),
    ]:
        aucs = ablation.get(variant, [])
        if any(a is not None for a in aucs):
            x_abl = [scales[i] for i, a in enumerate(aucs) if a is not None]
            y_abl = [a for a in aucs if a is not None]
            ax.plot(x_abl, y_abl, color=color, linestyle="-", marker="s", label=label)

    _annotate_modality_convergence(ax, df, ablation)

    ax.axhline(AUC_THRESHOLD, color=COLOR_THRESHOLD, linestyle="--",
               linewidth=1.5, label=f"AUC = {AUC_THRESHOLD:.2f}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Effective pixel scale (arcsec/pixel)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"CNN AUC vs. spatial sampling  (threshold = {AUC_THRESHOLD:.2f})")
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save(fig, out_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Bryson binary accuracy with CI + threshold
# ─────────────────────────────────────────────────────────────────────────────

def plot_bryson_binary_threshold(df: pd.DataFrame, out_path: str, fmt: str) -> None:
    """
    Figure 5: Bryson binary accuracy vs scale with:
      - Exact binomial 95% CI (scipy.stats.binom.interval) on PSF-off accuracy
      - 0.90 threshold line
      - First-crossing annotation or "never reaches 90%" box

    PSF-on shown at reduced alpha.
    """
    if "bryson_binary_accuracy" not in df.columns:
        _warn_missing("bryson_binary_accuracy", "breakdown.csv")
        return
    if "n_test" not in df.columns:
        _warn_missing("n_test", "breakdown.csv")

    fig, ax = plt.subplots(figsize=(7, 5))

    for psf_val, color, ls, alpha, label_base in [
        (0, COLOR_PSF_OFF, "-",  1.0, "Accuracy (PSF off)"),
        (1, COLOR_PSF_ON,  "--", 0.4, "Accuracy (PSF on)"),
    ]:
        sub = df[df["psf"] == psf_val].sort_values("k")
        if sub.empty:
            continue
        x = sub["effective_scale_arcsec"].values
        acc = sub["bryson_binary_accuracy"].values.astype(float)
        valid = ~np.isnan(acc)
        if not valid.any():
            continue

        ax.plot(x[valid], acc[valid], color=color, linestyle=ls, marker="o",
                alpha=alpha, label=label_base)

        # Binomial 95% CI on PSF-off accuracy using exact Clopper-Pearson interval
        if psf_val == 0 and "n_test" in sub.columns:
            n_arr = sub["n_test"].values.astype(float)
            ci_lo_list, ci_hi_list = [], []
            for i, (a, n) in enumerate(zip(acc, n_arr)):
                if np.isfinite(a) and np.isfinite(n) and n > 0:
                    k_success = int(round(a * n))
                    # binom.interval gives the equal-tails exact (Clopper-Pearson) CI
                    lo, hi = binom.interval(0.95, n=int(n), p=a)
                    ci_lo_list.append(lo / int(n))
                    ci_hi_list.append(hi / int(n))
                else:
                    ci_lo_list.append(np.nan)
                    ci_hi_list.append(np.nan)
            ci_lo = np.array(ci_lo_list)
            ci_hi = np.array(ci_hi_list)
            ci_valid = valid & ~np.isnan(ci_lo) & ~np.isnan(ci_hi)
            if ci_valid.any():
                ax.fill_between(x[ci_valid], ci_lo[ci_valid], ci_hi[ci_valid],
                                color=color, alpha=0.15, label="95% CI (PSF off)")

            # Crossing annotation for PSF-off accuracy
            x_cross = _find_first_crossing(x[valid], acc[valid], AUC_THRESHOLD)
            _annotate_crossing(ax, x_cross, AUC_THRESHOLD)

    ax.axhline(AUC_THRESHOLD, color=COLOR_THRESHOLD, linestyle="--",
               linewidth=1.5, label=f"Acc. = {AUC_THRESHOLD:.2f}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Effective pixel scale (arcsec/pixel)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title(
        f"Bryson binary classifier accuracy vs. spatial sampling\n"
        f"(σ = {3.0:.0f} threshold; shaded = 95% binomial CI)"
    )
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save(fig, out_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Bryson continuous AUC with CI + threshold
# ─────────────────────────────────────────────────────────────────────────────

def plot_bryson_continuous_threshold(df: pd.DataFrame, out_path: str, fmt: str) -> None:
    """
    Figure 6: Bryson continuous AUC vs scale with 0.90 threshold + crossing annotation.
    PSF-off 95% CI band from bryson_ci_lo/hi.
    """
    required = ["bryson_auc", "effective_scale_arcsec", "psf"]
    for col in required:
        if col not in df.columns:
            _warn_missing(col, "breakdown.csv")
            return

    fig, ax = plt.subplots(figsize=(7, 5))

    for psf_val, color, ls, alpha, label_base in [
        (0, COLOR_PSF_OFF, "-",  1.0, "Bryson continuous (PSF off)"),
        (1, COLOR_PSF_ON,  "--", 0.4, "Bryson continuous (PSF on)"),
    ]:
        sub = df[df["psf"] == psf_val].sort_values("k")
        if sub.empty:
            continue
        x = sub["effective_scale_arcsec"].values
        y = sub["bryson_auc"].values.astype(float)
        valid = ~np.isnan(y)
        if not valid.any():
            continue

        ax.plot(x[valid], y[valid], color=color, linestyle=ls, marker="s",
                alpha=alpha, label=label_base)

        if psf_val == 0:
            # 95% CI band from bootstrap (if columns present)
            ci_lo_col = "bryson_ci_lo" if "bryson_ci_lo" in sub.columns else None
            ci_hi_col = "bryson_ci_hi" if "bryson_ci_hi" in sub.columns else None
            if ci_lo_col and ci_hi_col:
                y_lo = sub[ci_lo_col].values.astype(float)
                y_hi = sub[ci_hi_col].values.astype(float)
                ci_valid = valid & ~np.isnan(y_lo) & ~np.isnan(y_hi)
                if ci_valid.any():
                    ax.fill_between(x[ci_valid], y_lo[ci_valid], y_hi[ci_valid],
                                    color=color, alpha=0.15, label="95% CI (PSF off)")

            # Crossing annotation
            x_cross = _find_first_crossing(x[valid], y[valid], AUC_THRESHOLD)
            _annotate_crossing(ax, x_cross, AUC_THRESHOLD)

    ax.axhline(AUC_THRESHOLD, color=COLOR_THRESHOLD, linestyle="--",
               linewidth=1.5, label=f"AUC = {AUC_THRESHOLD:.2f}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Effective pixel scale (arcsec/pixel)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(
        f"Bryson continuous classifier AUC vs. spatial sampling\n"
        f"(threshold = {AUC_THRESHOLD:.2f}; shaded = bootstrap 95% CI)"
    )
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save(fig, out_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Bryson continuous vs. centroid-only CNN (direct signal comparison)
# ─────────────────────────────────────────────────────────────────────────────

def plot_centroid_signal_comparison(
    df: pd.DataFrame, results_dir: str, out_path: str, fmt: str
) -> None:
    """
    Figure 7: direct comparison of a classical statistic (Bryson continuous SNR
    test) vs. a learned feature extractor (centroid-only CNN ablation), both
    applied to the same degrading centroid signal, PSF off only.

    Bryson and the centroid-only CNN are evaluated on genuinely different
    populations by construction: Bryson's n (n_valid_centroid) is quality-gated
    (excludes both hard- and soft-degenerate targets) and drawn from a population
    with no train/test split (Bryson needs no training); the CNN ablation's n
    (centroid_only_n) is its held-out test-fold row count (~20% of data), gated
    only by hard-degeneracy (soft-degenerate targets are deliberately kept in as
    flat-channel CNN inputs). These will generally differ — each line gets its
    own sample-size annotation, never a shared one.

    Points with fewer than MIN_SAMPLES_FOR_PLOT valid samples are omitted rather
    than plotted as a misleading AUC from too small a sample.

    CI bands use the same inline fill_between pattern as the other _threshold
    plotters — no shared CI helper exists yet in this file, and creating one now
    (its second use, not a third+) would be premature (YAGNI).
    """
    required = ["k", "psf", "bryson_auc", "bryson_ci_lo", "bryson_ci_hi",
                "centroid_only_auc", "centroid_only_auc_ci_lo", "centroid_only_auc_ci_hi",
                "n_valid_centroid", "centroid_only_n", "effective_scale_arcsec"]
    for col in required:
        if col not in df.columns:
            _warn_missing(col, "breakdown.csv")
            return

    sub = df[df["psf"] == 0].sort_values("k")
    if sub.empty:
        print("  Figure 7: no PSF-off rows — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for auc_col, lo_col, hi_col, n_col, color, label in [
        ("bryson_auc", "bryson_ci_lo", "bryson_ci_hi", "n_valid_centroid",
         COLOR_PSF_OFF, "Bryson continuous"),
        ("centroid_only_auc", "centroid_only_auc_ci_lo", "centroid_only_auc_ci_hi",
         "centroid_only_n", COLOR_CENTROID_ONLY, "Centroid-only CNN"),
    ]:
        n_vals = sub[n_col].values.astype(float)
        mask = n_vals >= MIN_SAMPLES_FOR_PLOT
        plot_sub = sub[mask]
        if plot_sub.empty:
            continue
        x = plot_sub["effective_scale_arcsec"].values
        y = plot_sub[auc_col].values.astype(float)
        valid = ~np.isnan(y)
        if not valid.any():
            continue
        ax.plot(x[valid], y[valid], color=color, marker="o", label=label)

        y_lo = plot_sub[lo_col].values.astype(float)
        y_hi = plot_sub[hi_col].values.astype(float)
        ci_valid = valid & ~np.isnan(y_lo) & ~np.isnan(y_hi)
        if ci_valid.any():
            ax.fill_between(x[ci_valid], y_lo[ci_valid], y_hi[ci_valid],
                             color=color, alpha=0.15)

        n_label = "n_bryson" if n_col == "n_valid_centroid" else "n_cnn"
        n_plot = plot_sub[n_col].values[valid]
        for xi, yi, n in zip(x[valid], y[valid], n_plot):
            ax.annotate(f"{n_label}={int(n)}", xy=(xi, yi), xytext=(0, 6),
                        textcoords="offset points", fontsize=7, ha="center", color=color)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Random (0.5)")
    ax.set_xlabel("Effective pixel scale (arcsec/pixel)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Centroid signal extraction: classical statistic vs. learned feature")
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save(fig, out_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_graphs(results_dir: str, fmt: str = "png") -> None:
    """
    Load breakdown.csv and call all 6 plot functions.
    Missing columns in the DataFrame are tolerated: each plotter logs a warning
    and returns early, so one missing column never blocks the others.
    """
    graphs_dir = os.path.join(results_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    breakdown_path = os.path.join(results_dir, "breakdown.csv")
    if not os.path.exists(breakdown_path):
        print(f"ERROR: {breakdown_path} not found. "
              f"Run stats_ml.py and baselines.py first.")
        sys.exit(1)

    df = pd.read_csv(breakdown_path)
    print(f"Loaded breakdown.csv: {len(df)} rows, columns: {list(df.columns)}")

    print("\nGenerating figures ...")

    plot_cnn_auc(
        df, results_dir,
        os.path.join(graphs_dir, f"fig1_cnn_auc.{fmt}"), fmt
    )
    plot_bryson_binary(
        df,
        os.path.join(graphs_dir, f"fig2_bryson_binary.{fmt}"), fmt
    )
    plot_bryson_continuous(
        df,
        os.path.join(graphs_dir, f"fig3_bryson_continuous.{fmt}"), fmt
    )
    plot_cnn_auc_threshold(
        df, results_dir,
        os.path.join(graphs_dir, f"fig4_cnn_auc_threshold.{fmt}"), fmt
    )
    plot_bryson_binary_threshold(
        df,
        os.path.join(graphs_dir, f"fig5_bryson_binary_threshold.{fmt}"), fmt
    )
    plot_bryson_continuous_threshold(
        df,
        os.path.join(graphs_dir, f"fig6_bryson_continuous_threshold.{fmt}"), fmt
    )
    plot_centroid_signal_comparison(
        df, results_dir,
        os.path.join(graphs_dir, f"fig7_centroid_signal_comparison.{fmt}"), fmt
    )

    print(f"\nAll graphs written to {graphs_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication figures for the centroid degradation study"
    )
    parser.add_argument(
        "--results-dir", default="results_resolution",
        help="Directory containing breakdown.csv and score CSVs (default: results_resolution)"
    )
    parser.add_argument(
        "--format", choices=["png", "pdf"], default="png",
        help="Output format: png (default, 150dpi) or pdf (vector)"
    )
    args = parser.parse_args()

    generate_all_graphs(args.results_dir, fmt=args.format)


if __name__ == "__main__":
    main()
