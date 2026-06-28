"""
stats_ml.py — Bootstrap Statistics and Breakdown Resolution

Reads per-target CNN scores from results_resolution/scores_k{K}_psf{P}.csv
and computes ROC-AUC with bootstrap 95% CIs for each (k, PSF setting).

Primary uncertainty quantification: bootstrap CIs by resampling targets.
DeLong pairwise significance testing has been removed (--enable-delong is
retained as a deprecated no-op for backwards-compatible pipeline invocation).

Breakdown resolution metrics (configurable, physically motivated):
  k_break_90  — first k where AUC falls below the ABSOLUTE threshold of 0.90
                (not relative to the k=1 AUC; a score of 0.90 is a meaningful
                 classifier quality marker independent of the native-k performance)
  k_break_snr — first k where median centroid_snr (from centroid_quality_per_target.csv,
                filtered to FP test targets) falls below --snr-threshold (default 3)

BH FDR correction: one-sided p-values P(bootstrap AUC ≤ 0.90) computed via
bootstrap_auc_with_p, corrected at FDR α=0.05 via scipy.stats.false_discovery_control
(scipy ≥ 1.11.0) with a manual BH fallback for older installations.

Outputs:
  results_resolution/breakdown.csv   — per (k, psf): AUC + CI + breakdown flags
"""

import argparse
import glob
import os
import re
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RESULTS_DIR         = "results_resolution"
N_BOOTSTRAP         = 2000
RANDOM_SEED         = 42
ABSOLUTE_AUC_THRESHOLD = 0.90   # k_break_90 tests against this fixed value


# ============================================================================
# Bootstrap AUC — SIGNATURE FROZEN (3-tuple); do not change.
# baselines.py unpacks (auc, lo, hi); any extension must go in a sibling function.
# ============================================================================

def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray,
                  n_iter: int = N_BOOTSTRAP, seed: int = RANDOM_SEED
                  ) -> tuple[float, float, float]:
    """
    Bootstrap ROC-AUC with 95% CI by resampling *targets* (rows).

    Signature frozen at (auc, ci_lo, ci_hi) — baselines.py unpacks exactly 3 values.
    Do not extend this function; use bootstrap_auc_with_p for the 4-tuple form.

    Args:
        y_true:  ground-truth labels
        y_score: predicted probability scores
        n_iter:  bootstrap iterations (default 2000)
        seed:    random seed

    Returns:
        (auc, ci_lo, ci_hi) — point estimate and 95% percentile CI
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_point = float(roc_auc_score(y_true, y_score))

    aucs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, s_b))

    if len(aucs) < 10:
        return auc_point, np.nan, np.nan

    ci_lo = float(np.percentile(aucs, 2.5))
    ci_hi = float(np.percentile(aucs, 97.5))
    return auc_point, ci_lo, ci_hi


def bootstrap_auc_with_p(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    threshold: float = ABSOLUTE_AUC_THRESHOLD,
) -> tuple[float, float, float, float]:
    """
    Bootstrap ROC-AUC with 95% CI and one-sided p-value for BH FDR correction.

    Sibling of bootstrap_auc — shares the resampling loop but also returns
    p = P(bootstrap AUC ≤ threshold), the one-sided p-value for testing
    H₀: true AUC ≥ threshold (default 0.90). Used for Benjamini-Hochberg FDR.

    Separated from bootstrap_auc to preserve its 3-tuple signature (baselines.py
    unpacks exactly 3 values and must not be broken by a signature change).

    Args:
        y_true:    ground-truth labels
        y_score:   predicted probability scores
        n_iter:    bootstrap iterations (default 2000)
        seed:      random seed
        threshold: AUC threshold for the one-sided test (default ABSOLUTE_AUC_THRESHOLD=0.90)

    Returns:
        (auc, ci_lo, ci_hi, p_one_sided)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_point = float(roc_auc_score(y_true, y_score))

    aucs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, s_b))

    if len(aucs) < 10:
        return auc_point, np.nan, np.nan, np.nan

    ci_lo = float(np.percentile(aucs, 2.5))
    ci_hi = float(np.percentile(aucs, 97.5))

    # One-sided p = fraction of bootstrap samples with AUC ≤ threshold.
    # Large p → bootstrap distribution is mostly below threshold (classifier is poor).
    # Small p → AUC reliably exceeds threshold (classifier is good).
    p_one_sided = float(np.mean(np.array(aucs) <= threshold))
    return auc_point, ci_lo, ci_hi, p_one_sided


def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Manual Benjamini-Hochberg FDR correction (fallback for scipy < 1.11.0).

    Controls the expected fraction of false rejections (discoveries) among all
    rejections. Appropriate here because we are making m = 10 simultaneous
    comparisons (one per (k, psf) pair) and want to know which ones are
    "significantly above AUC=0.90" after accounting for multiple testing.

    Returns a boolean array: True where H₀ is rejected at FDR ≤ alpha.
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool)
    ranked = np.argsort(p_values)
    ranks  = np.arange(1, m + 1)
    # Reject null for all i where p_(i) ≤ (i/m)*alpha, choosing the largest i
    bh_thresholds = (ranks / m) * alpha
    sorted_p      = p_values[ranked]
    # Find the largest rank where the condition holds
    significant   = sorted_p <= bh_thresholds
    if not significant.any():
        return np.zeros(m, dtype=bool)
    # All hypotheses up to and including the largest significant rank are rejected
    last_sig = np.where(significant)[0][-1]
    result = np.zeros(m, dtype=bool)
    result[ranked[:last_sig + 1]] = True
    return result


def apply_bh_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction.
    Uses scipy.stats.false_discovery_control (scipy ≥ 1.11.0) when available,
    falling back to the manual BH implementation.

    Returns boolean array of rejections at FDR ≤ alpha.
    """
    finite_mask = np.isfinite(p_values)
    result = np.zeros(len(p_values), dtype=bool)
    if not finite_mask.any():
        return result

    finite_ps = p_values[finite_mask]
    try:
        from scipy.stats import false_discovery_control
        # false_discovery_control returns adjusted p-values; reject where adj_p ≤ alpha
        adj_p = false_discovery_control(finite_ps, method="bh")
        result[finite_mask] = adj_p <= alpha
    except ImportError:
        # Fallback: scipy < 1.11.0; use manual BH
        result[finite_mask] = _benjamini_hochberg(finite_ps, alpha=alpha)
    return result


# ============================================================================
# Load scores
# ============================================================================

def parse_tag(filename: str) -> tuple[int, int]:
    """Extract k and psf from scores_k{K}_psf{P}.csv filename."""
    m = re.search(r"scores_k(\d+)_psf(\d+)\.csv", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return -1, -1


def load_all_scores(results_dir: str) -> dict[tuple[int, int], pd.DataFrame]:
    """
    Load all per-target score CSVs written by theModel.py.
    Excludes ablation files (*_flux_only.csv, *_centroid_only.csv).
    Returns dict keyed by (k, psf).
    """
    pattern = os.path.join(results_dir, "scores_k*_psf*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No score files found matching {pattern}")
        return {}

    scores = {}
    for f in files:
        basename = os.path.basename(f)
        # Exclude ablation score files
        if "flux_only" in basename or "centroid_only" in basename:
            continue
        k, psf = parse_tag(basename)
        if k == -1:
            continue
        df = pd.read_csv(f)
        scores[(k, psf)] = df
        print(f"  Loaded {f}: {len(df)} test targets")

    return scores


def load_centroid_quality_per_target(
    results_dir: str,
    test_kepids: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    """
    Load centroid_quality_per_target.csv (written by centroid_quality.py Fix 16).
    Optionally filters to test_kepids so that centroid SNR estimates are drawn
    only from held-out test targets (preventing leakage from training targets
    into the k_break_snr breakdown metric).

    Falls back to centroid_quality.csv (pre-aggregated, no kepid column) with
    a warning if the per-target file does not yet exist.
    """
    per_target_path = os.path.join(results_dir, "centroid_quality_per_target.csv")
    if os.path.exists(per_target_path):
        df = pd.read_csv(per_target_path)
        if test_kepids is not None and "kepid" in df.columns:
            df = df[df["kepid"].isin(test_kepids)]
            print(f"  centroid_quality_per_target: {len(df)} records "
                  f"after filtering to {len(test_kepids)} test kepids")
        return df

    # Fallback: pre-aggregated file (no kepid filtering possible)
    agg_path = os.path.join(results_dir, "centroid_quality.csv")
    if os.path.exists(agg_path):
        warnings.warn(
            "centroid_quality_per_target.csv not found; falling back to "
            "centroid_quality.csv (no test-kepid filtering). "
            "Re-run centroid_quality.py to generate the per-target file.",
            UserWarning, stacklevel=2
        )
        return pd.read_csv(agg_path)
    return None


# ============================================================================
# Breakdown resolution
# ============================================================================

def compute_breakdown(
    auc_table: pd.DataFrame,
    cq: Optional[pd.DataFrame],
    snr_threshold: float = 3.0,
    test_kepids: Optional[set] = None,
) -> pd.DataFrame:
    """
    Add breakdown resolution columns to the AUC table.

    k_break_90  — first k where AUC < ABSOLUTE_AUC_THRESHOLD (0.90).
                  This is an absolute criterion, not relative to the k=1 AUC.
                  A classifier achieving AUC=0.90 is a meaningful quality marker
                  regardless of native-resolution performance.
    k_break_snr — first k where median centroid_snr (among FP test targets)
                  falls below snr_threshold.

    Args:
        auc_table:     DataFrame with cnn_auc, k, psf columns
        cq:            centroid quality DataFrame (per-target or pre-aggregated)
        snr_threshold: SNR floor for k_break_snr (default 3.0)
        test_kepids:   set of kepids in the test set (for FP filtering if cq is per-target)
    """
    auc_table = auc_table.copy()
    auc_table["k_break_90"]  = np.nan
    auc_table["k_break_snr"] = np.nan

    for psf_val in auc_table["psf"].unique():
        sub = auc_table[auc_table["psf"] == psf_val].sort_values("k")
        if sub.empty:
            continue

        # k_break_90: first k where AUC < 0.90 (absolute threshold, not relative to k=1)
        for _, row in sub.iterrows():
            idx = row.name
            if row["cnn_auc"] < ABSOLUTE_AUC_THRESHOLD:
                auc_table.loc[idx, "k_break_90"] = row["k"]
                break

        # k_break_snr: first k where FP centroid SNR drops below snr_threshold.
        # Uses FP label (label == 0) because centroid information matters most for
        # discriminating centroid-offset FPs from planets.
        if cq is not None:
            # Determine if cq is per-target (has 'kepid') or pre-aggregated (has 'label')
            if "centroid_snr" in cq.columns:
                # Per-target file: compute medians ourselves
                cq_fp = cq[(cq["psf"] == psf_val) & (cq["label"] == 0)].copy()
                for k_val in sorted(sub["k"].values):
                    cq_k = cq_fp[cq_fp["k"] == k_val]
                    if cq_k.empty:
                        continue
                    med_snr = cq_k["centroid_snr"].median()
                    if np.isfinite(med_snr) and med_snr < snr_threshold:
                        mask = (auc_table["psf"] == psf_val) & (auc_table["k"] == k_val)
                        auc_table.loc[mask, "k_break_snr"] = k_val
                        break
            elif "median_centroid_snr" in cq.columns:
                # Pre-aggregated file: median_centroid_snr is already computed per group
                cq_psf = cq[(cq["psf"] == psf_val) & (cq["label"] == 0)].sort_values("k")
                for _, cq_row in cq_psf.iterrows():
                    if cq_row.get("median_centroid_snr", np.inf) < snr_threshold:
                        mask = ((auc_table["psf"] == psf_val) &
                                (auc_table["k"] == cq_row["k"]))
                        auc_table.loc[mask, "k_break_snr"] = cq_row["k"]
                        break

    return auc_table


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute bootstrap AUC statistics and breakdown resolution"
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help=f"Bootstrap iterations (default {N_BOOTSTRAP})")
    parser.add_argument("--snr-threshold", type=float, default=3.0,
                        help="centroid_snr threshold for k_break_snr (default 3.0)")
    parser.add_argument("--enable-delong", action="store_true",
                        help="[DEPRECATED] DeLong tests have been removed; "
                             "this flag is accepted but has no effect.")
    args = parser.parse_args()

    if args.enable_delong:
        warnings.warn(
            "--enable-delong is deprecated and has no effect. "
            "DeLong tests have been removed; primary uncertainty quantification "
            "is bootstrap 95% CIs. Use --n-bootstrap to adjust bootstrap iterations.",
            DeprecationWarning, stacklevel=1
        )

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading scores from {args.results_dir} ...")
    scores = load_all_scores(args.results_dir)
    if not scores:
        print("No scores found. Run python theModel.py first.")
        sys.exit(1)

    # Collect test kepids from all score files for centroid filtering
    all_test_kepids: set = set()
    for df in scores.values():
        if "kepid" in df.columns:
            all_test_kepids.update(df["kepid"].tolist())

    cq = load_centroid_quality_per_target(args.results_dir, test_kepids=all_test_kepids)
    if cq is not None:
        print(f"Loaded centroid quality data ({len(cq)} records)")

    print(f"\nComputing bootstrap AUCs ({args.n_bootstrap} iterations) ...")
    rows = []
    bh_inputs = []   # (k, psf, p_one_sided) for BH correction

    for (k, psf), df in sorted(scores.items()):
        y_true  = df["label"].values
        y_score = df["score"].values

        if len(np.unique(y_true)) < 2:
            print(f"  k={k} psf={psf}: skipping — only one class in test set")
            continue

        # Use bootstrap_auc_with_p for the one-sided p-value needed by BH correction.
        # Note: bootstrap_auc's signature is frozen (3-tuple); this is a separate function.
        auc, ci_lo, ci_hi, p_one_sided = bootstrap_auc_with_p(
            y_true, y_score,
            n_iter=args.n_bootstrap,
            threshold=ABSOLUTE_AUC_THRESHOLD,
        )

        row = {
            "k":                      k,
            "effective_scale_arcsec": k * 3.98,
            "psf":                    psf,
            "n_test":                 len(df),
            "n_planet":               int((y_true == 1).sum()),
            "n_fp":                   int((y_true == 0).sum()),
            "cnn_auc":                auc,
            "cnn_ci_lo":              ci_lo,
            "cnn_ci_hi":              ci_hi,
            "p_auc_below_90":         p_one_sided,   # one-sided; for BH below
            # Bryson and vetting columns filled in later by baselines.py
            "bryson_auc":             np.nan,
            "bryson_ci_lo":           np.nan,
            "bryson_ci_hi":           np.nan,
            "vetting_auc":            np.nan,
            "vetting_ci_lo":          np.nan,
            "vetting_ci_hi":          np.nan,
            "n_vetting_untestable":   np.nan,
        }

        # Optional subgroup AUCs
        if "fp_subtype" in df.columns:
            for name, pattern in [("co", "co"), ("ss", "ss")]:
                fp_sub_mask = (
                    df["fp_subtype"].str.contains(pattern, na=False) & (y_true == 0)
                )
                sub_mask = fp_sub_mask | (y_true == 1)
                if sub_mask.sum() > 10 and len(np.unique(y_true[sub_mask])) == 2:
                    sub_auc, sub_lo, sub_hi = bootstrap_auc(
                        y_true[sub_mask], y_score[sub_mask], n_iter=args.n_bootstrap
                    )
                    row[f"cnn_auc_fp_{name}"]       = sub_auc
                    row[f"cnn_auc_fp_{name}_ci_lo"] = sub_lo
                    row[f"cnn_auc_fp_{name}_ci_hi"] = sub_hi

        rows.append(row)
        bh_inputs.append((k, psf, p_one_sided))
        print(f"  k={k} psf={psf}: AUC={auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"p_below_90={p_one_sided:.4f}")

    if not rows:
        print("No results to write.")
        sys.exit(1)

    auc_df = pd.DataFrame(rows).sort_values(["k", "psf"])
    auc_df = compute_breakdown(
        auc_df, cq,
        snr_threshold=args.snr_threshold,
        test_kepids=all_test_kepids,
    )

    # Benjamini-Hochberg FDR correction on P(AUC ≤ 0.90).
    # bh_significant = True means "AUC is significantly above 0.90 after FDR control",
    # i.e. we reject H₀: AUC ≤ 0.90 for this (k, psf) pair at FDR α=0.05.
    p_arr = auc_df["p_auc_below_90"].values.astype(float)
    bh_sig = apply_bh_correction(p_arr, alpha=0.05)
    auc_df["bh_significant"] = bh_sig
    print(f"\nBH FDR correction (α=0.05): {bh_sig.sum()} / {len(bh_sig)} "
          f"(k,psf) pairs have AUC significantly above {ABSOLUTE_AUC_THRESHOLD}")

    out_path = os.path.join(args.results_dir, "breakdown.csv")
    auc_df.to_csv(out_path, index=False)
    print(f"\nbreakdown.csv written to {out_path}")

    # Print summary
    print("\nBreakdown resolution summary:")
    for psf_val in sorted(auc_df["psf"].unique()):
        sub = auc_df[auc_df["psf"] == psf_val].sort_values("k")
        label = "PSF on" if psf_val == 1 else "PSF off"
        print(f"\n  {label}:")
        print(f"  {'k':>3}  {'scale':>8}  {'AUC':>8}  {'CI_lo':>8}  "
              f"{'CI_hi':>8}  {'p≤0.90':>7}  {'BH_sig':>6}")
        for _, row in sub.iterrows():
            bh = "✓" if row.get("bh_significant", False) else " "
            print(f"  {row['k']:>3.0f}  {row['effective_scale_arcsec']:>7.1f}\"  "
                  f"{row['cnn_auc']:>8.4f}  {row['cnn_ci_lo']:>8.4f}  "
                  f"{row['cnn_ci_hi']:>8.4f}  {row['p_auc_below_90']:>7.4f}  {bh:>6}")

        k90  = sub["k_break_90"].dropna()
        ksnr = sub["k_break_snr"].dropna()
        print(f"  k_break_90  (AUC < {ABSOLUTE_AUC_THRESHOLD:.2f} absolute): "
              f"{int(k90.min()) if len(k90) else 'not reached within k=[1–5]'}")
        print(f"  k_break_snr (centroid SNR < {args.snr_threshold}):  "
              f"{int(ksnr.min()) if len(ksnr) else 'not reached'}")

    print("\nNext step: python baselines.py")


if __name__ == "__main__":
    main()
