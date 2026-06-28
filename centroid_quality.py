"""
centroid_quality.py — Centroid Quality Aggregation

Reads pre-computed centroid quality metrics from cache/centroids/ and aggregates
them per (k, PSF setting, label) to produce the intermediate analysis that connects:

  Spatial sampling (k)  →  Centroid measurement quality  →  Classification performance

Outputs:
  results_resolution/centroid_quality.csv   — per (k, psf, label) summary statistics
  results_resolution/centroid_scaling.csv   — per (k, psf) medians for scaling check
  results_resolution/figures/fig2_centroid_quality.png — 2×2 quality metrics vs. k
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


KEPLER_PIXEL_SCALE_ARCSEC = 3.98
DEFAULT_K_VALUES = [1, 2, 3, 4, 5]
RESULTS_DIR = "results_resolution"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_quality_records(
    manifest: pd.DataFrame,
    centroid_dir: str,
    k_values: list[int],
) -> pd.DataFrame:
    """
    Load centroid quality metrics from cache and join with manifest labels.

    Returns a DataFrame with columns:
      kepid, kepoi_name, label, fp_subtype, k, psf,
      centroid_rms, centroid_uncertainty, centroid_snr, offset_arcsec
    """
    records = []
    label_map = manifest.set_index("kepid")[["kepoi_name", "label", "fp_subtype"]]

    for k in k_values:
        for psf in [0, 1]:
            for kepid in manifest["kepid"].unique():
                path = os.path.join(centroid_dir, f"{kepid}_k{k}_psf{psf}.npz")
                if not os.path.exists(path):
                    continue
                try:
                    data = np.load(path)
                    row_info = label_map.loc[kepid]
                    # kepid may map to multiple KOIs; take the first (centroid is per-star)
                    if isinstance(row_info, pd.DataFrame):
                        row_info = row_info.iloc[0]

                    records.append({
                        "kepid": kepid,
                        "kepoi_name": str(row_info["kepoi_name"]),
                        "label": int(row_info["label"]),
                        "fp_subtype": str(row_info.get("fp_subtype", "")),
                        "k": k,
                        "psf": psf,
                        "centroid_rms": float(data["centroid_rms"])
                            if "centroid_rms" in data else np.nan,
                        "centroid_uncertainty": float(data["centroid_uncertainty"])
                            if "centroid_uncertainty" in data else np.nan,
                        "centroid_snr": float(data["centroid_snr"])
                            if "centroid_snr" in data else np.nan,
                        "offset_arcsec": float(data["offset_arcsec"])
                            if "offset_arcsec" in data else np.nan,
                    })
                except Exception:
                    continue

    if not records:
        print("WARNING: No centroid cache files found. Run getInputData.py first.")
        return pd.DataFrame()

    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (k, psf, label): median and IQR of each quality metric."""
    metrics = ["centroid_rms", "centroid_uncertainty", "centroid_snr", "offset_arcsec"]
    rows = []

    for (k, psf, label), group in df.groupby(["k", "psf", "label"]):
        row = {
            "k": k,
            "effective_scale_arcsec": k * KEPLER_PIXEL_SCALE_ARCSEC,
            "psf": psf,
            "label": int(label),
            "label_name": "planet" if label == 1 else "fp",
            "n_targets": len(group),
        }
        for m in metrics:
            vals = group[m].dropna()
            row[f"median_{m}"] = float(vals.median()) if len(vals) else np.nan
            q25, q75 = (float(vals.quantile(0.25)), float(vals.quantile(0.75))) \
                if len(vals) >= 4 else (np.nan, np.nan)
            row[f"iqr_{m}"] = q75 - q25
        rows.append(row)

    return pd.DataFrame(rows)


def scaling_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (k, psf): median uncertainty and SNR across all targets."""
    rows = []
    for (k, psf), group in df.groupby(["k", "psf"]):
        rows.append({
            "k": k,
            "effective_scale_arcsec": k * KEPLER_PIXEL_SCALE_ARCSEC,
            "psf": int(psf),
            "n_targets": len(group),
            "median_uncertainty": float(group["centroid_uncertainty"].median()),
            "median_rms": float(group["centroid_rms"].median()),
            "median_snr": float(group["centroid_snr"].median()),
        })
    return pd.DataFrame(rows)


def check_scaling_linearity(scaling: pd.DataFrame) -> None:
    """Print Pearson r(k, median_uncertainty) per PSF setting."""
    print("\nCentroid scaling linearity check:")
    for psf_val in sorted(scaling["psf"].unique()):
        sub = scaling[scaling["psf"] == psf_val].sort_values("k")
        if len(sub) >= 3:
            r, _ = pearsonr(sub["k"].values, sub["median_uncertainty"].fillna(0).values)
            label = "on" if psf_val == 1 else "off"
            status = "OK" if r > 0.8 else "WARNING — non-linear"
            print(f"  PSF {label}: Pearson r = {r:.3f}  [{status}]")
        print(f"  PSF {'on' if psf_val else 'off'} scaling table:")
        for _, row in sub.iterrows():
            print(f"    k={row['k']:d}  {row['effective_scale_arcsec']:.1f} arcsec  "
                  f"median_unc={row['median_uncertainty']:.4f}  "
                  f"median_snr={row['median_snr']:.3f}")


def plot_quality(agg: pd.DataFrame, out_path: str) -> None:
    """
    Figure 2: 2×2 panel of centroid quality metrics vs. effective pixel scale,
    split by label (planet vs. FP), PSF off only (PSF on as dashed variant).
    """
    metrics = [
        ("median_centroid_rms", "Centroid RMS (arcsec)"),
        ("median_centroid_uncertainty", "Centroid Uncertainty (arcsec)"),
        ("median_centroid_snr", "Centroid SNR (|offset|/uncertainty)"),
        ("median_offset_arcsec", "Diff-Image Offset (arcsec)"),
    ]
    label_styles = {
        0: {"color": "#e07b54", "linestyle": "-", "marker": "o", "name": "FP"},
        1: {"color": "#4878cf", "linestyle": "-", "marker": "s", "name": "Planet"},
    }
    psf_styles = {0: "-", 1: "--"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (col, ylabel) in zip(axes, metrics):
        for psf_val in sorted(agg["psf"].unique()):
            ls = psf_styles[psf_val]
            for label_val, style in label_styles.items():
                sub = agg[(agg["psf"] == psf_val) & (agg["label"] == label_val)].sort_values("k")
                if sub.empty:
                    continue
                x = sub["effective_scale_arcsec"].values
                y = sub[col].values
                psf_tag = " (PSF on)" if psf_val == 1 else ""
                ax.plot(x, y, color=style["color"], linestyle=ls,
                        marker=style["marker"], label=f"{style['name']}{psf_tag}")

        ax.set_xlabel("Effective pixel scale (arcsec/pixel)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("(")[0].strip())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Centroid measurement quality vs. spatial sampling\n"
        "(Solid = PSF off, Dashed = PSF on)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 written to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate centroid quality metrics from cache"
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--manifest", default="koi_manifest.csv")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Directory for output files (default: results_resolution)")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"ERROR: {args.manifest} not found.")
        sys.exit(1)

    manifest = pd.read_csv(args.manifest)
    centroid_dir = os.path.join(args.cache_dir, "centroids")
    results_dir = args.results_dir
    figures_dir = os.path.join(results_dir, "figures")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading centroid quality records from {centroid_dir} ...")
    df = load_quality_records(manifest, centroid_dir, args.k_values)

    if df.empty:
        sys.exit(0)

    print(f"Loaded {len(df):,} records ({df['kepid'].nunique():,} unique targets)")

    agg = aggregate(df)

    # Fix 16: Assert required columns are present before writing.
    # These columns are consumed by stats_ml.py (k_break_snr), graphs.py, and compare.py.
    required_agg_cols = [
        "median_centroid_snr",
        "median_centroid_rms",
        "median_centroid_uncertainty",
        "median_offset_arcsec",
    ]
    missing_cols = [c for c in required_agg_cols if c not in agg.columns]
    assert not missing_cols, (
        f"aggregate() output is missing expected columns: {missing_cols}. "
        f"Check that cache/centroids/*.npz files contain the required keys."
    )

    quality_path = os.path.join(results_dir, "centroid_quality.csv")
    agg.to_csv(quality_path, index=False)
    print(f"centroid_quality.csv written to {quality_path}")

    # Fix 16: Also write the raw per-target records so stats_ml.py can filter
    # to test kepids when computing k_break_snr (avoids leakage from train targets).
    per_target_path = os.path.join(results_dir, "centroid_quality_per_target.csv")
    df.to_csv(per_target_path, index=False)
    print(f"centroid_quality_per_target.csv written to {per_target_path} "
          f"({len(df):,} records)")

    scaling = scaling_summary(df)
    scaling_path = os.path.join(results_dir, "centroid_scaling.csv")
    scaling.to_csv(scaling_path, index=False)
    print(f"centroid_scaling.csv written to {scaling_path}")

    check_scaling_linearity(scaling)

    fig_path = os.path.join(figures_dir, "fig2_centroid_quality.png")
    plot_quality(agg, fig_path)


if __name__ == "__main__":
    main()
