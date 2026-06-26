"""
compare.py — Assemble Final Output

Reads pre-computed results from results_resolution/ and produces:
  - results_resolution/comparison.csv  — one row per (k, psf) with all AUCs + CI
  - results_resolution/report.md       — human-readable study summary
  - results_resolution/figures/
        fig1_auc_vs_k.png              — AUC vs. pixel scale (CNN + Bryson [+ vetting])
        fig3_difference_images.png     — example diff images at k=1..5

Never re-runs expensive compute. All inputs are read from CSV/cache.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results_resolution"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
KEPLER_PIXEL_SCALE_ARCSEC = 3.98


# ============================================================================
# comparison.csv
# ============================================================================

def build_comparison_csv(results_dir: str) -> pd.DataFrame:
    """
    Read breakdown.csv and centroid_quality.csv; join into one comparison table.
    If either source is missing, return what is available.
    """
    breakdown_path = os.path.join(results_dir, "breakdown.csv")
    if not os.path.exists(breakdown_path):
        print(f"WARNING: {breakdown_path} not found. Run stats_ml.py and baselines.py first.")
        return pd.DataFrame()

    df = pd.read_csv(breakdown_path)

    cq_path = os.path.join(results_dir, "centroid_quality.csv")
    if os.path.exists(cq_path):
        cq = pd.read_csv(cq_path)
        # Merge median centroid SNR and uncertainty for all targets (label-agnostic)
        cq_all = cq.groupby(["k", "psf"], as_index=False).agg(
            median_centroid_snr=("median_centroid_snr", "mean"),
            median_centroid_uncertainty=("median_centroid_uncertainty", "mean"),
        )
        df = df.merge(cq_all, on=["k", "psf"], how="left")

    out_path = os.path.join(results_dir, "comparison.csv")
    df.sort_values(["k", "psf"]).to_csv(out_path, index=False)
    print(f"comparison.csv written to {out_path}")
    return df


# ============================================================================
# Figure 1: AUC vs. pixel scale
# ============================================================================

def plot_auc_vs_k(df: pd.DataFrame, out_path: str) -> None:
    """
    Figure 1: AUC vs. effective pixel scale for CNN and Bryson (+ vetting if present).
    Shaded 95% CI bands. PSF off = solid lines, PSF on = dashed.
    x-axis in arcsec (3.98·k).
    """
    if df.empty:
        print("WARNING: No data for Figure 1 — skipping.")
        return

    method_styles = {
        "CNN":     {"color": "#4878cf", "marker": "o", "label_base": "CNN"},
        "Bryson":  {"color": "#e07b54", "marker": "s", "label_base": "Bryson (2013)"},
        "vetting": {"color": "#6acc65", "marker": "^", "label_base": "vetting (Hedges 2021)"},
    }
    psf_linestyles = {0: "-", 1: "--"}

    fig, ax = plt.subplots(figsize=(9, 6))

    for psf_val in sorted(df["psf"].unique()):
        sub = df[df["psf"] == psf_val].sort_values("k")
        x = sub["effective_scale_arcsec"].values
        ls = psf_linestyles[psf_val]
        psf_label = " (PSF on)" if psf_val == 1 else ""

        for method, col_auc, col_lo, col_hi in [
            ("CNN",     "cnn_auc",     "cnn_ci_lo",     "cnn_ci_hi"),
            ("Bryson",  "bryson_auc",  "bryson_ci_lo",  "bryson_ci_hi"),
            ("vetting", "vetting_auc", "vetting_ci_lo", "vetting_ci_hi"),
        ]:
            if col_auc not in sub.columns or sub[col_auc].isna().all():
                continue
            y    = sub[col_auc].values
            y_lo = sub[col_lo].values if col_lo in sub.columns else np.full_like(y, np.nan)
            y_hi = sub[col_hi].values if col_hi in sub.columns else np.full_like(y, np.nan)

            valid = ~np.isnan(y)
            if not valid.any():
                continue

            sty = method_styles[method]
            label = f"{sty['label_base']}{psf_label}"
            ax.plot(x[valid], y[valid], color=sty["color"], linestyle=ls,
                    marker=sty["marker"], label=label, linewidth=1.8, markersize=6)
            if not np.isnan(y_lo[valid]).all():
                ax.fill_between(x[valid], y_lo[valid], y_hi[valid],
                                color=sty["color"], alpha=0.15)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xlabel("Effective pixel scale (arcsec/pixel)", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        "Classification performance vs. spatial sampling\n"
        "(Solid = PSF off, Dashed = PSF on; shaded = 95% CI)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 written to {out_path}")


# ============================================================================
# Figure 3: example difference images
# ============================================================================

def plot_difference_images(
    cache_dir: str,
    manifest: pd.DataFrame,
    tpf_dir: str,
    k_values: list[int],
    out_path: str,
) -> None:
    """
    Figure 3: difference images at k=1..5 for one planet and one FP example.
    Requires degrade.py and cached TPFs. Skips gracefully if unavailable.
    """
    try:
        import lightkurve as lk
        from degrade import load_cube, superpixel_rebin, difference_image_centroid
    except ImportError as e:
        print(f"  Figure 3: skipping — {e}")
        return

    # Find one planet and one FP in the manifest
    planets = manifest[manifest["label"] == 1]
    fps     = manifest[manifest["label"] == 0]
    if planets.empty or fps.empty:
        print("  Figure 3: skipping — no examples in manifest")
        return

    examples = [
        (int(planets.iloc[0]["kepid"]), "Planet"),
        (int(fps.iloc[0]["kepid"]),     "FP"),
    ]

    fig, axes = plt.subplots(
        2, len(k_values),
        figsize=(3 * len(k_values), 6),
        squeeze=False,
    )

    for row_idx, (kepid, label_name) in enumerate(examples):
        row_data = manifest[manifest["kepid"] == kepid].iloc[0]
        period   = float(row_data["koi_period"])
        t0_bkjd  = float(row_data["koi_time0bk"])
        dur_days = float(row_data["koi_duration"]) / 24.0

        try:
            search = lk.search_targetpixelfile(
                f"KIC {kepid}", mission="Kepler", cadence="long"
            )
            if len(search) == 0:
                raise ValueError("No TPFs found")
            tpf = search[0].download(quality_bitmask="default", download_dir=tpf_dir)
            cube, time_tpf, _, _, aperture = load_cube(tpf)
            if cube is None:
                raise ValueError("Could not load cube")
        except Exception as exc:
            print(f"  Figure 3: KIC {kepid} ({label_name}): {exc}")
            for col_idx in range(len(k_values)):
                axes[row_idx, col_idx].axis("off")
            continue

        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            try:
                if k > 1:
                    coarse, _ = superpixel_rebin(cube, aperture, k)
                else:
                    coarse = cube

                result = difference_image_centroid(
                    coarse, time_tpf, period, t0_bkjd, dur_days, k=k
                )
                diff_img = result.get("diff_image")
                if diff_img is None:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.axis("off")
                    continue

                vmax = np.nanpercentile(np.abs(diff_img), 99)
                ax.imshow(diff_img, origin="lower", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, aspect="equal")
                oot_col = result.get("oot_centroid_col")
                oot_row = result.get("oot_centroid_row")
                if oot_col is not None and np.isfinite(oot_col):
                    ax.plot(oot_col, oot_row, "k+", markersize=8, markeredgewidth=1.5)

                ax.set_title(f"k={k} ({k*KEPLER_PIXEL_SCALE_ARCSEC:.1f}″/px)",
                             fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception as exc:
                ax.text(0.5, 0.5, str(exc)[:30], ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)
                ax.axis("off")

        axes[row_idx, 0].set_ylabel(label_name, fontsize=11, rotation=90, labelpad=4)

    fig.suptitle(
        "Difference images (OOT − IT) at increasing k\n"
        "(+ = OOT centroid / target position; red = positive signal)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 written to {out_path}")


# ============================================================================
# report.md
# ============================================================================

def write_report(df: pd.DataFrame, manifest: pd.DataFrame, results_dir: str) -> None:
    """
    Write results_resolution/report.md — human-readable study summary.
    """
    n_planet = int((manifest["label"] == 1).sum())
    n_fp     = int((manifest["label"] == 0).sum())
    n_kic    = manifest["kepid"].nunique()
    n_fp_co  = int(manifest.get("koi_fpflag_co", pd.Series([0])).eq(1).sum())

    psf_off = df[df["psf"] == 0].sort_values("k") if not df.empty else pd.DataFrame()
    psf_on  = df[df["psf"] == 1].sort_values("k") if not df.empty else pd.DataFrame()

    def fmt_row(row, method_auc, method_lo, method_hi):
        auc = row.get(method_auc, np.nan)
        lo  = row.get(method_lo, np.nan)
        hi  = row.get(method_hi, np.nan)
        if np.isnan(auc):
            return "—"
        if np.isnan(lo):
            return f"{auc:.4f}"
        return f"{auc:.4f} [{lo:.4f}, {hi:.4f}]"

    lines = [
        "# Kepler Centroid-Resolution Degradation Study — Results",
        "",
        "## 1. Scientific Objective",
        "",
        "> **How does progressively degraded spatial sampling (effective astrometric",
        "> resolution) reduce the information content of centroid measurements for",
        "> automated exoplanet vetting?**",
        "",
        "The degradation engine is an intentional experimental control:",
        "the photometric light curve is held at native Kepler quality while",
        "the centroid pixel scale is coarsened by integer factor k.",
        "This isolates centroid information as the independent variable.",
        "This is **not** a simulation of TESS or any other telescope.",
        "",
        "## 2. Dataset",
        "",
        f"- Source: NASA Exoplanet Archive KOI cumulative table (TAP/ADQL)",
        f"- Positives (label=1): {n_planet:,} CONFIRMED planets",
        f"- Negatives (label=0): {n_fp:,} FALSE POSITIVE KOIs (all subtypes, Option A)",
        f"- Unique KIC targets: {n_kic:,}",
        f"- FP subgroup with centroid offset flag (koi_fpflag_co=1): {n_fp_co:,}",
        "",
        "**Dataset design note (Option A):** All FALSE POSITIVE subtypes are included",
        "as negatives. This avoids pre-selecting for centroid-offset FPs, which would",
        "artificially inflate centroid-method AUCs. Subgroup analysis (§8) reports",
        "performance on the centroid-offset and significant-secondary FP subsets",
        "separately, allowing the paper to discuss where centroid information helps",
        "without baking the answer into the training set.",
        "",
        "**Epoch convention:** `koi_time0bk` used directly as BKJD (BJD − 2,454,833).",
        "No 2,457,000 offset was applied (that is the TESS BTJD offset).",
        "",
        "**Group integrity:** All KOIs from one KIC target (`kepid`) remain entirely",
        "within one CV fold and one side of the train/test split.",
        "",
        "## 3. Experimental Design",
        "",
        "- Degradation factors k = [1, 2, 3, 4, 5] (effective scales: "
        + ", ".join(f"{k*KEPLER_PIXEL_SCALE_ARCSEC:.1f}″" for k in [1,2,3,4,5]) + ")",
        "- Centroid channels degraded at each k; flux channel held at native resolution",
        "- PSF broadening: two result sets (off = rebinning only; on = rebin + Gaussian)",
        "- Cache hierarchy: tpf_temp/ → cache/cubes/ → cache/centroids/ + cache/flux/",
        "",
        "## 4. k=1 Validation",
        "",
        "The k=1 validation gate asserts that on ≥80% of qualifying targets the",
        "native-resolution difference-image centroid uncertainty is < 1.0 arcsec.",
        "This verifies the centroid computation is in a physically sensible regime.",
        "(Bryson et al. 2013 report a ~0.067 arcsec systematic floor for bright",
        "Kepler targets; we use the more generous 1.0 arcsec to account for faint",
        "targets and our simplified implementation.)",
        "",
        "See `results_resolution/centroid_scaling.csv` for the k=1 median uncertainty.",
        "",
        "## 5. Centroid Quality vs. Spatial Sampling",
        "",
        "See `results_resolution/centroid_quality.csv` and",
        "`results_resolution/figures/fig2_centroid_quality.png`.",
        "",
        "Expected: centroid uncertainty increases with k (fewer photons per superpixel",
        "over the same sky area; astrometric noise ∝ pixel scale at fixed SNR).",
        "Centroid SNR should decrease with k, approaching noise-dominated values at",
        "high k.",
        "",
        "## 6. Centroid Scaling Check",
        "",
        "See `results_resolution/centroid_scaling.csv`.",
        "A Pearson correlation r(k, median_uncertainty) > 0.8 is required as a",
        "realism safeguard.",
        "",
        "## 7. AUC vs. Pixel Scale",
        "",
        "Primary result: `results_resolution/figures/fig1_auc_vs_k.png`",
        "",
    ]

    if not psf_off.empty:
        lines += [
            "### AUC table (PSF off — rebinning only)",
            "",
            "| k | Scale (arcsec) | CNN AUC [95% CI] | Bryson AUC [95% CI] |",
            "|---|---|---|---|",
        ]
        for _, row in psf_off.iterrows():
            lines.append(
                f"| {int(row['k'])} | {row['effective_scale_arcsec']:.1f} | "
                f"{fmt_row(row, 'cnn_auc', 'cnn_ci_lo', 'cnn_ci_hi')} | "
                f"{fmt_row(row, 'bryson_auc', 'bryson_ci_lo', 'bryson_ci_hi')} |"
            )
        lines.append("")

    lines += [
        "## 8. Subgroup Analysis",
        "",
        "Subgroup AUCs (CNN, centroid-offset FPs vs. planets; significant-secondary",
        "FPs vs. planets) are stored in `results_resolution/breakdown.csv` columns",
        "`cnn_auc_fp_co` and `cnn_auc_fp_ss` (where available).",
        "",
        "## 9. Breakdown Resolution",
        "",
    ]

    if not psf_off.empty:
        k90  = psf_off["k_break_90"].dropna()
        ksnr = psf_off["k_break_snr"].dropna()
        lines += [
            f"- **k_break_90** (AUC < 90% of k=1 AUC, PSF off): "
            f"{int(k90.min()) if len(k90) else 'not reached within k=[1–5]'}",
            f"- **k_break_snr** (centroid SNR < 3, PSF off): "
            f"{int(ksnr.min()) if len(ksnr) else 'not reached'}",
            "",
        ]

    lines += [
        "## 10. PSF Broadening (Second-Order Robustness)",
        "",
        "Two result sets are produced: PSF off (rebinning only) and PSF on",
        "(rebinning + Gaussian broadening with σ = √(k²−1) / (2√(2 ln 2)) native px).",
        "PSF broadening is a simplifying assumption; the paper should note that the",
        "PSF on/off comparison is a robustness check, not a physical model.",
        "",
        "Conclusions should hold with both settings; if they diverge, report and",
        "discuss the discrepancy.",
        "",
        "## 11. Difference-Image Implementation Note",
        "",
        "The Bryson baseline uses a **simplified implementation following the",
        "methodology of Bryson et al. (2013)**. It is not a reproduction of the",
        "Kepler pipeline and makes no claim of pipeline equivalence.",
        "",
        "## 12. Future Work",
        "",
        "- **TESS endpoint anchoring:** comparing the k≈5 (≈19.9 arcsec) result",
        "  against real TESS centroid precision would be the strongest external",
        "  validation of the degradation model. Deferred to future work.",
        "- **Formal pipeline equivalence:** verifying that our simplified",
        "  difference-image centroid matches the Kepler DV centroid pipeline",
        "  for bright targets at k=1.",
        "",
        "## 13. Realism Safeguards",
        "",
        "| Safeguard | Description | Result |",
        "|---|---|---|",
        "| k=1 validation | uncertainty < 1.0 arcsec on ≥80% of targets | see §4 |",
        "| Scaling linearity | Pearson r(k, median_unc) > 0.8 | see §6 |",
        "| PSF on/off | conclusions hold with and without PSF broadening | see §10 |",
        "",
        "---",
        f"*Generated by compare.py*",
    ]

    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report.md written to {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Assemble final comparison.csv, report.md, and figures"
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--cache-dir",   default="./cache")
    parser.add_argument("--tpf-dir",     default="./tpf_temp")
    parser.add_argument("--manifest",    default="koi_manifest.csv")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--no-fig3", action="store_true",
                        help="Skip figure 3 (requires cached TPFs)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load manifest
    manifest = pd.read_csv(args.manifest) if os.path.exists(args.manifest) else pd.DataFrame()

    # Build comparison.csv
    df = build_comparison_csv(args.results_dir)

    # Figure 1: AUC vs. k
    plot_auc_vs_k(df, os.path.join(FIGURES_DIR, "fig1_auc_vs_k.png"))

    # Figure 3: difference images
    if not args.no_fig3 and not manifest.empty:
        plot_difference_images(
            args.cache_dir, manifest, args.tpf_dir, args.k_values,
            os.path.join(FIGURES_DIR, "fig3_difference_images.png"),
        )

    # report.md
    write_report(df, manifest, args.results_dir)

    print(f"\nAll outputs written to {args.results_dir}/")
    print("  comparison.csv")
    print("  report.md")
    print("  figures/fig1_auc_vs_k.png")
    print("  figures/fig2_centroid_quality.png  (produced by centroid_quality.py)")
    print("  figures/fig3_difference_images.png")


if __name__ == "__main__":
    main()
