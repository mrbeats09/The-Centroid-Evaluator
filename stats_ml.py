"""
stats_ml.py — Bootstrap Statistics and Breakdown Resolution

Reads per-target CNN scores from results_resolution/scores_k{K}_psf{P}.csv
and computes ROC-AUC with bootstrap 95% CIs for each (k, PSF setting).

Primary uncertainty quantification: bootstrap CIs by resampling targets.
DeLong pairwise significance testing is optional (--enable-delong, default OFF).

Breakdown resolution metrics (configurable, physically motivated):
  k_break_90  — first k where AUC falls below 90% of native-resolution (k=1) AUC
  k_break_snr — first k where median centroid_snr (from centroid_quality.csv)
                falls below --snr-threshold (default 3)

Outputs:
  results_resolution/breakdown.csv   — per (k, psf): AUC + CI + breakdown flags
"""

import argparse
import glob
import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "results_resolution"
N_BOOTSTRAP = 2000
RANDOM_SEED = 42


# ============================================================================
# Bootstrap AUC
# ============================================================================

def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray,
                  n_iter: int = N_BOOTSTRAP, seed: int = RANDOM_SEED
                  ) -> tuple[float, float, float]:
    """
    Bootstrap ROC-AUC with 95% CI by resampling *targets* (rows).

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


# ============================================================================
# DeLong (optional)
# ============================================================================

def delong_auc_variance(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast DeLong variance estimate (Sun & Xu 2014)."""
    n1 = int(y_true.sum())
    n0 = int((1 - y_true).sum())
    score1 = y_score[y_true == 1]
    score0 = y_score[y_true == 0]

    # Structural components V10 and V01
    v10 = np.array([(score1 > s).mean() + 0.5 * (score1 == s).mean() for s in score0])
    v01 = np.array([(score0 < s).mean() + 0.5 * (score0 == s).mean() for s in score1])

    var = (np.var(v10) / n0 + np.var(v01) / n1)
    return float(var)


def delong_p_value(y_true: np.ndarray,
                   y_score_a: np.ndarray,
                   y_score_b: np.ndarray) -> float:
    """
    DeLong test for difference between two correlated ROC AUCs
    (Sun & Xu 2014, fast parametric form).
    Returns two-sided p-value.
    """
    from scipy import stats as scipy_stats

    auc_a = roc_auc_score(y_true, y_score_a)
    auc_b = roc_auc_score(y_true, y_score_b)

    var_a = delong_auc_variance(y_true, y_score_a)
    var_b = delong_auc_variance(y_true, y_score_b)

    # Covariance term requires structural components (simplified: assume independent)
    # A full DeLong covariance requires paired structural components; this
    # implementation treats them as independent (conservative).
    se = np.sqrt(var_a + var_b)
    if se == 0:
        return 1.0

    z = (auc_a - auc_b) / se
    return float(2 * (1 - scipy_stats.norm.cdf(abs(z))))


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
    """Load all scores CSV files. Returns dict keyed by (k, psf)."""
    pattern = os.path.join(results_dir, "scores_k*_psf*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No score files found matching {pattern}")
        return {}

    scores = {}
    for f in files:
        k, psf = parse_tag(os.path.basename(f))
        if k == -1:
            continue
        df = pd.read_csv(f)
        scores[(k, psf)] = df
        print(f"  Loaded {f}: {len(df)} test targets")

    return scores


def load_centroid_quality(results_dir: str) -> Optional[pd.DataFrame]:
    """Load centroid_quality.csv for SNR-based breakdown resolution."""
    path = os.path.join(results_dir, "centroid_quality.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ============================================================================
# Breakdown resolution
# ============================================================================

def compute_breakdown(
    auc_table: pd.DataFrame,
    cq: Optional[pd.DataFrame],
    snr_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Add breakdown resolution columns to the AUC table.

    k_break_90  — first k where AUC < 0.9 × AUC_at_k1 (per psf)
    k_break_snr — first k where median centroid_snr < snr_threshold (per psf)
    """
    auc_table = auc_table.copy()
    auc_table["k_break_90"] = np.nan
    auc_table["k_break_snr"] = np.nan

    for psf_val in auc_table["psf"].unique():
        sub = auc_table[auc_table["psf"] == psf_val].sort_values("k")
        if sub.empty:
            continue

        native = sub[sub["k"] == 1]["cnn_auc"].values
        if len(native) == 0:
            continue
        auc_k1 = float(native[0])
        threshold_90 = 0.9 * auc_k1

        for _, row in sub.iterrows():
            idx = row.name
            if row["cnn_auc"] < threshold_90:
                auc_table.loc[idx, "k_break_90"] = row["k"]
                break

        if cq is not None:
            cq_psf = cq[(cq["psf"] == psf_val) & (cq["label"] == 0)].sort_values("k")
            for _, cq_row in cq_psf.iterrows():
                if cq_row.get("median_centroid_snr", np.inf) < snr_threshold:
                    auc_table.loc[
                        (auc_table["psf"] == psf_val) &
                        (auc_table["k"] == cq_row["k"]),
                        "k_break_snr"
                    ] = cq_row["k"]
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
                        help="Compute DeLong p-values (disabled by default)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading scores from {args.results_dir} ...")
    scores = load_all_scores(args.results_dir)
    if not scores:
        print("No scores found. Run python theModel.py first.")
        sys.exit(1)

    cq = load_centroid_quality(args.results_dir)
    if cq is not None:
        print(f"Loaded centroid_quality.csv ({len(cq)} rows)")

    print(f"\nComputing bootstrap AUCs ({args.n_bootstrap} iterations) ...")
    rows = []
    for (k, psf), df in sorted(scores.items()):
        y_true  = df["label"].values
        y_score = df["score"].values

        if len(np.unique(y_true)) < 2:
            print(f"  k={k} psf={psf}: skipping — only one class in test set")
            continue

        auc, ci_lo, ci_hi = bootstrap_auc(y_true, y_score, n_iter=args.n_bootstrap)

        row = {
            "k":                    k,
            "effective_scale_arcsec": k * 3.98,
            "psf":                  psf,
            "n_test":               len(df),
            "n_planet":             int((y_true == 1).sum()),
            "n_fp":                 int((y_true == 0).sum()),
            "cnn_auc":              auc,
            "cnn_ci_lo":            ci_lo,
            "cnn_ci_hi":            ci_hi,
            # Bryson and vetting columns filled in later by baselines.py
            "bryson_auc":           np.nan,
            "bryson_ci_lo":         np.nan,
            "bryson_ci_hi":         np.nan,
            "vetting_auc":          np.nan,
            "vetting_ci_lo":        np.nan,
            "vetting_ci_hi":        np.nan,
            "n_vetting_untestable": np.nan,
        }

        # Optional subgroup AUCs
        for subtype_col in ["fp_subtype"]:
            if subtype_col in df.columns:
                co_mask = df[subtype_col].str.contains("co", na=False) & (y_true == 0)
                ss_mask = df[subtype_col].str.contains("ss", na=False) & (y_true == 0)
                for name, fp_sub_mask in [("co", co_mask), ("ss", ss_mask)]:
                    sub_mask = fp_sub_mask | (y_true == 1)
                    if sub_mask.sum() > 10 and len(np.unique(y_true[sub_mask])) == 2:
                        sub_auc, sub_lo, sub_hi = bootstrap_auc(
                            y_true[sub_mask], y_score[sub_mask], n_iter=args.n_bootstrap
                        )
                        row[f"cnn_auc_fp_{name}"] = sub_auc
                        row[f"cnn_auc_fp_{name}_ci_lo"] = sub_lo
                        row[f"cnn_auc_fp_{name}_ci_hi"] = sub_hi

        rows.append(row)
        print(f"  k={k} psf={psf}: AUC={auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    if not rows:
        print("No results to write.")
        sys.exit(1)

    auc_df = pd.DataFrame(rows).sort_values(["k", "psf"])
    auc_df = compute_breakdown(auc_df, cq, snr_threshold=args.snr_threshold)

    # Optional DeLong — runs but does not block pipeline
    if args.enable_delong:
        print("\nRunning DeLong tests (CNN vs Bryson) ...")
        auc_df["delong_p_cnn_bryson"] = np.nan
        for (k, psf), df in sorted(scores.items()):
            bryson_path = os.path.join(args.results_dir, f"bryson_scores_k{k}_psf{psf}.csv")
            if not os.path.exists(bryson_path):
                continue
            bryson_df = pd.read_csv(bryson_path)
            merged = df.merge(bryson_df[["kepid", "bryson_score"]], on="kepid", how="inner")
            if len(merged) < 10:
                continue
            try:
                p = delong_p_value(
                    merged["label"].values,
                    merged["score"].values,
                    merged["bryson_score"].values,
                )
                mask = (auc_df["k"] == k) & (auc_df["psf"] == psf)
                auc_df.loc[mask, "delong_p_cnn_bryson"] = p
                print(f"  DeLong CNN vs Bryson k={k} psf={psf}: p={p:.4f}")
            except Exception as exc:
                print(f"  DeLong k={k} psf={psf}: {exc}")

    out_path = os.path.join(args.results_dir, "breakdown.csv")
    auc_df.to_csv(out_path, index=False)
    print(f"\nbreakdown.csv written to {out_path}")

    # Print summary
    print("\nBreakdown resolution summary:")
    for psf_val in sorted(auc_df["psf"].unique()):
        sub = auc_df[auc_df["psf"] == psf_val].sort_values("k")
        label = "PSF on" if psf_val == 1 else "PSF off"
        print(f"\n  {label}:")
        print(f"  {'k':>3}  {'scale':>8}  {'AUC':>8}  {'CI_lo':>8}  {'CI_hi':>8}")
        for _, row in sub.iterrows():
            print(f"  {row['k']:>3.0f}  {row['effective_scale_arcsec']:>7.1f}\"  "
                  f"{row['cnn_auc']:>8.4f}  {row['cnn_ci_lo']:>8.4f}  {row['cnn_ci_hi']:>8.4f}")
        k90 = sub["k_break_90"].dropna()
        ksnr = sub["k_break_snr"].dropna()
        print(f"  k_break_90  (AUC < 90% of k=1 AUC): {k90.min() if len(k90) else 'not reached'}")
        print(f"  k_break_snr (centroid SNR < {args.snr_threshold}):     "
              f"{ksnr.min() if len(ksnr) else 'not reached'}")

    print("\nNext step: python baselines.py")


if __name__ == "__main__":
    main()
