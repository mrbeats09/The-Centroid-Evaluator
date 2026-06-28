"""
baselines.py — Comparison Baselines for Centroid Degradation Study

Priority 3 (required): Bryson-style difference-image centroid offset baseline.
Priority 4 (optional): vetting package integration (--enable-vetting, default OFF).

Bryson continuous baseline:
  Reads pre-computed difference-image offsets from cache/centroids/ (already
  computed by degrade.py — no FITS files opened here).
  Score = offset_arcsec / uncertainty_arcsec (SNR).
  IMPORTANT: High SNR → more likely a false positive (centroid offset) → label 0.
  sklearn's roc_auc_score expects higher score → label 1, so we negate:
      score_for_auc = -valid["bryson_score"].values
  Computes ROC-AUC + bootstrap 95% CI.

Bryson binary baseline (PRIMARY Bryson result):
  Binary classifier: predict FP (label=0) if SNR > BRYSON_SIGMA_THRESHOLD, else Planet (1).
  Threshold default = 3.0 (BRYSON_SIGMA_THRESHOLD).
  Accuracy + F1-macro + recall_fp + recall_planet computed.
  Permutation test (accuracy difference) used for CNN vs Bryson-binary significance.

vetting baseline (--enable-vetting only):
  Runs vetting.centroid_test on rebinned TPFs. All failures are caught and logged;
  they never terminate the run. Un-testable targets are counted per k.

Outputs:
  results_resolution/bryson_scores_k{K}_psf{P}.csv       — per-target Bryson scores
  results_resolution/bryson_binary_scores_k{K}_psf{P}.csv — per-target binary predictions
  results_resolution/breakdown.csv                        — updated with all Bryson metrics
  (if --enable-vetting) results_resolution/vetting_scores_k{K}_psf{P}.csv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

from stats_ml import bootstrap_auc

RESULTS_DIR               = "results_resolution"
CACHE_DIR                 = "./cache"
KEPLER_PIXEL_SCALE_ARCSEC = 3.98
# Sigma threshold for the binary Bryson classifier.
# "If the centroid offset SNR exceeds 3σ, classify as FP."
# This is the primary Bryson result; the continuous-score AUC is secondary.
BRYSON_SIGMA_THRESHOLD    = 3.0


# ============================================================================
# Permutation tests — two separate tests for different data types
# ============================================================================

def permutation_test_auc_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_iter: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Permutation test for the difference in ROC-AUC between two continuous classifiers.

    Per iteration: for each sample independently, flip which score goes to classifier A
    and which goes to classifier B with probability 0.5 (coin-flip swap). Compute the
    AUC difference under the permuted assignment. The two-sided p-value is the fraction
    of permuted differences whose absolute value exceeds the observed absolute difference.

    Used for: CNN continuous scores vs Bryson continuous scores.
    Returns: (auc_a, auc_b, p_two_sided)

    Note: this is the appropriate permutation test for continuous scores.
    For binary 0/1 predictions, use permutation_test_accuracy_diff instead.
    """
    rng = np.random.default_rng(seed)
    auc_a = roc_auc_score(y_true, score_a)
    auc_b = roc_auc_score(y_true, score_b)
    observed = abs(auc_a - auc_b)

    null_diffs = []
    for _ in range(n_iter):
        # Coin-flip swap: for each target, randomly assign its (score_a, score_b)
        # pair to either (perm_a, perm_b) or (perm_b, perm_a).
        swap = rng.integers(0, 2, size=len(y_true)).astype(bool)
        perm_a = np.where(swap, score_b, score_a)
        perm_b = np.where(swap, score_a, score_b)
        if len(np.unique(y_true)) < 2:
            continue
        try:
            da = roc_auc_score(y_true, perm_a)
            db = roc_auc_score(y_true, perm_b)
            null_diffs.append(abs(da - db))
        except Exception:
            continue

    if not null_diffs:
        return float(auc_a), float(auc_b), np.nan

    p_two_sided = float(np.mean(np.array(null_diffs) >= observed))
    return float(auc_a), float(auc_b), p_two_sided


def permutation_test_accuracy_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_iter: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Permutation test for the difference in ACCURACY between two binary classifiers.

    This is a separate test from permutation_test_auc_diff and must not be confused
    with it. Binary 0/1 predictions cannot be fed to the AUC-based coin-flip swap test
    (swapping two near-identical binary arrays produces a degenerate null distribution
    with trivial variance). Instead we use an accuracy-difference permutation test:

    Per iteration: for each sample, coin-flip swap both classifiers' binary predictions.
    Compute the accuracy difference under the permuted assignment.
    Two-sided p-value = fraction of |permuted differences| ≥ |observed difference|.

    Used for: CNN-binarised predictions (using OOF-fixed threshold from theModel)
    vs Bryson binary predictions (SNR > BRYSON_SIGMA_THRESHOLD → FP).

    Returns: (acc_a, acc_b, observed_diff, p_two_sided)
    """
    rng = np.random.default_rng(seed)
    acc_a    = accuracy_score(y_true, pred_a)
    acc_b    = accuracy_score(y_true, pred_b)
    observed = abs(acc_a - acc_b)

    null_diffs = []
    for _ in range(n_iter):
        swap   = rng.integers(0, 2, size=len(y_true)).astype(bool)
        perm_a = np.where(swap, pred_b, pred_a)
        perm_b = np.where(swap, pred_a, pred_b)
        da = accuracy_score(y_true, perm_a)
        db = accuracy_score(y_true, perm_b)
        null_diffs.append(abs(da - db))

    if not null_diffs:
        return float(acc_a), float(acc_b), float(observed), np.nan

    p_two_sided = float(np.mean(np.array(null_diffs) >= observed))
    return float(acc_a), float(acc_b), float(observed), p_two_sided


# ============================================================================
# Bryson continuous baseline (Priority 3 — required)
# ============================================================================

def compute_bryson_scores(
    manifest: pd.DataFrame,
    cache_dir: str,
    k_values: list[int],
    test_kepids_by_tag: dict,
) -> dict:
    """
    Compute Bryson-style difference-image offset SNR scores for each (k, psf).
    Reads from cache/centroids/{kepid}_k{K}_psf{P}.npz — no FITS access.

    Raw score = offset_arcsec / uncertainty_arcsec (higher SNR → more likely FP/offset).
    AUC computation negates this score (see update_breakdown_csv Fix 9).

    Returns dict[(k, psf)] → pd.DataFrame with columns:
        kepid, label, fp_subtype, bryson_score, offset_arcsec, uncertainty_arcsec
    """
    centroid_dir = os.path.join(cache_dir, "centroids")
    label_map = manifest.set_index("kepid")[["label", "fp_subtype"]].copy()
    results = {}

    for k in k_values:
        for psf in [0, 1]:
            tag = f"k{k}_psf{psf}"
            test_kepids = test_kepids_by_tag.get(tag, set())
            if not test_kepids:
                print(f"  Bryson k={k} psf={psf}: no test kepids found (run theModel.py first)")
                continue

            rows = []
            for kepid in test_kepids:
                path = os.path.join(centroid_dir, f"{kepid}_k{k}_psf{psf}.npz")
                if not os.path.exists(path):
                    continue
                try:
                    data = np.load(path)
                    offset      = float(data.get("offset_arcsec",       np.nan))
                    uncertainty = float(data.get("centroid_uncertainty", np.nan))

                    if np.isfinite(uncertainty) and uncertainty > 0:
                        score = abs(offset) / uncertainty
                    else:
                        score = np.nan

                    row_info = label_map.loc[kepid] if kepid in label_map.index else None
                    if row_info is None:
                        continue
                    if isinstance(row_info, pd.DataFrame):
                        row_info = row_info.iloc[0]

                    rows.append({
                        "kepid":              kepid,
                        "label":              int(row_info["label"]),
                        "fp_subtype":         str(row_info.get("fp_subtype", "")),
                        "bryson_score":       score,
                        "offset_arcsec":      offset,
                        "uncertainty_arcsec": uncertainty,
                    })
                except Exception as exc:
                    print(f"    Bryson KIC {kepid} k={k}: {type(exc).__name__}: {exc}")
                    continue

            results[(k, psf)] = pd.DataFrame(rows)
            valid = sum(1 for r in rows if np.isfinite(r["bryson_score"]))
            print(f"  Bryson k={k} psf={psf}: {len(rows)} targets, "
                  f"{valid} with valid scores")

    return results


# ============================================================================
# Bryson binary baseline (PRIMARY Bryson result)
# ============================================================================

def compute_bryson_binary_scores(
    manifest: pd.DataFrame,
    cache_dir: str,
    k_values: list[int],
    test_kepids_by_tag: dict,
    sigma_threshold: float = BRYSON_SIGMA_THRESHOLD,
) -> dict:
    """
    Binary Bryson classifier: predict FP (label=0) if SNR > sigma_threshold, else Planet (1).

    This is the PRIMARY Bryson result for comparison against the CNN.
    The continuous-score AUC (compute_bryson_scores) is secondary.

    Returns dict[(k, psf)] → pd.DataFrame with columns:
        kepid, label, fp_subtype, bryson_score, binary_pred, correct
    """
    centroid_dir = os.path.join(cache_dir, "centroids")
    label_map = manifest.set_index("kepid")[["label", "fp_subtype"]].copy()
    results = {}

    for k in k_values:
        for psf in [0, 1]:
            tag = f"k{k}_psf{psf}"
            test_kepids = test_kepids_by_tag.get(tag, set())
            if not test_kepids:
                continue

            rows = []
            for kepid in test_kepids:
                path = os.path.join(centroid_dir, f"{kepid}_k{k}_psf{psf}.npz")
                if not os.path.exists(path):
                    continue
                try:
                    data = np.load(path)
                    offset      = float(data.get("offset_arcsec",       np.nan))
                    uncertainty = float(data.get("centroid_uncertainty", np.nan))

                    if np.isfinite(uncertainty) and uncertainty > 0:
                        snr = abs(offset) / uncertainty
                    else:
                        snr = np.nan

                    row_info = label_map.loc[kepid] if kepid in label_map.index else None
                    if row_info is None:
                        continue
                    if isinstance(row_info, pd.DataFrame):
                        row_info = row_info.iloc[0]

                    true_label = int(row_info["label"])
                    if np.isfinite(snr):
                        # SNR > threshold → significant centroid offset → predict FP (0)
                        # SNR ≤ threshold → no significant offset → predict Planet (1)
                        binary_pred = 0 if snr > sigma_threshold else 1
                    else:
                        # No valid score: default to predicting Planet (conservative)
                        binary_pred = 1

                    rows.append({
                        "kepid":       kepid,
                        "label":       true_label,
                        "fp_subtype":  str(row_info.get("fp_subtype", "")),
                        "bryson_score": snr,
                        "binary_pred": binary_pred,
                        "correct":     int(binary_pred == true_label),
                    })
                except Exception as exc:
                    print(f"    Bryson-binary KIC {kepid} k={k}: {type(exc).__name__}: {exc}")
                    continue

            results[(k, psf)] = pd.DataFrame(rows)
            n_correct = sum(r["correct"] for r in rows if "correct" in r)
            print(f"  Bryson-binary k={k} psf={psf}: {len(rows)} targets, "
                  f"{n_correct}/{len(rows)} correct (σ_thresh={sigma_threshold})")

    return results


# ============================================================================
# vetting baseline (Priority 4 — optional, gated by --enable-vetting)
# ============================================================================

def run_vetting_baseline(
    manifest: pd.DataFrame,
    tpf_dir: str,
    cache_dir: str,
    k_values: list[int],
    test_kepids_by_tag: dict,
    max_quarters: int | None,
) -> dict:
    """
    Run vetting.centroid_test on rebinned TPFs for each (k, psf) in the test set.
    All failures are caught and logged — never terminates the run.

    Returns dict[(k, psf)] → pd.DataFrame with columns:
        kepid, label, vetting_score, vetting_pvalue, n_quarters_failed
    """
    try:
        import vetting as vt
    except ImportError:
        print("vetting package not installed. Run: pip install vetting")
        return {}

    import lightkurve as lk
    from degrade import load_cube, superpixel_rebin

    centroid_dir = os.path.join(cache_dir, "centroids")
    label_map = manifest.set_index("kepid")[["label", "fp_subtype",
                                              "koi_period", "koi_time0bk",
                                              "koi_duration"]].copy()
    results = {}
    vetting_failures_log = os.path.join(RESULTS_DIR, "vetting_failures.log")

    def log_vetting_failure(msg: str) -> None:
        with open(vetting_failures_log, "a") as f:
            f.write(msg + "\n")

    for k in k_values:
        for psf in [0, 1]:
            tag = f"k{k}_psf{psf}"
            test_kepids = test_kepids_by_tag.get(tag, set())
            if not test_kepids:
                continue

            rows = []
            n_untestable = 0

            for kepid in test_kepids:
                row_info = label_map.loc[kepid] if kepid in label_map.index else None
                if row_info is None:
                    continue
                if isinstance(row_info, pd.DataFrame):
                    row_info = row_info.iloc[0]

                period    = float(row_info["koi_period"])
                t0_bkjd   = float(row_info["koi_time0bk"])  # BKJD — no 2,457,000 offset
                dur_days  = float(row_info["koi_duration"]) / 24.0

                try:
                    search = lk.search_targetpixelfile(
                        f"KIC {kepid}", mission="Kepler", cadence="long"
                    )
                    if len(search) == 0:
                        n_untestable += 1
                        log_vetting_failure(f"KIC {kepid} k={k}: no TPFs found")
                        continue

                    if max_quarters is not None and len(search) > max_quarters:
                        search = search[:max_quarters]

                    collection = search.download_all(
                        quality_bitmask="default", download_dir=tpf_dir
                    )
                    tpf_list = list(collection) if collection else []
                    if not tpf_list:
                        n_untestable += 1
                        continue

                    quarter_pvalues = []
                    for tpf in tpf_list:
                        try:
                            cube, time_tpf, quality_tpf, wcs, aperture = load_cube(tpf)
                            if cube is None:
                                continue

                            if k > 1:
                                coarse, coarse_aper = superpixel_rebin(cube, aperture, k)
                            else:
                                coarse, coarse_aper = cube, aperture

                            # Pixel degeneracy check: vetting needs at least a 3×3 aperture
                            ny_c, nx_c = coarse.shape[1], coarse.shape[2]
                            if ny_c < 3 or nx_c < 3:
                                log_vetting_failure(
                                    f"KIC {kepid} k={k} q={getattr(tpf,'quarter','?')}: "
                                    f"only {ny_c}x{nx_c} pixels after rebinning"
                                )
                                n_untestable += 1
                                continue

                            try:
                                r = vt.centroid_test(
                                    tpfs=[tpf],
                                    periods=[period],
                                    t0s=[t0_bkjd],
                                    durs=[dur_days],
                                    aperture_mask="pipeline",
                                    plot=False,
                                )
                                pvs = r.get("pvalues", [[]])[0]
                                if pvs:
                                    quarter_pvalues.extend(pvs)
                            except Exception as ve:
                                log_vetting_failure(
                                    f"KIC {kepid} k={k} q={getattr(tpf,'quarter','?')}: "
                                    f"{type(ve).__name__}: {ve}"
                                )
                                n_untestable += 1

                        except Exception as exc:
                            log_vetting_failure(
                                f"KIC {kepid} k={k}: {type(exc).__name__}: {exc}"
                            )
                            continue

                    if not quarter_pvalues:
                        n_untestable += 1
                        continue

                    # Aggregate: geometric mean across quarters/transits
                    log_p      = np.mean(np.log10(np.clip(quarter_pvalues, 1e-300, 1.0)))
                    aggregate_p = 10 ** log_p
                    score       = -float(log_p)   # −log10(p); higher = more likely FP offset

                    rows.append({
                        "kepid":          kepid,
                        "label":          int(row_info["label"]),
                        "fp_subtype":     str(row_info.get("fp_subtype", "")),
                        "vetting_score":  score,
                        "vetting_pvalue": aggregate_p,
                        "n_pvalues":      len(quarter_pvalues),
                    })

                except Exception as exc:
                    log_vetting_failure(f"KIC {kepid} k={k}: {type(exc).__name__}: {exc}")
                    n_untestable += 1
                    continue

            results[(k, psf)] = pd.DataFrame(rows)
            print(f"  vetting k={k} psf={psf}: {len(rows)} scored, "
                  f"{n_untestable} untestable")

    return results


# ============================================================================
# Save scores and update breakdown.csv
# ============================================================================

def update_breakdown_csv(
    bryson_results: dict,
    bryson_binary_results: dict,
    vetting_results: dict,
    results_dir: str,
    n_bootstrap: int = 2000,
    cnn_scores_by_tag: dict | None = None,
) -> None:
    """
    Append Bryson continuous, Bryson binary, and optionally vetting metric columns
    to breakdown.csv.

    Fix 9: bryson_score is negated before AUC computation.
      High SNR (large bryson_score) means likely FP → label 0.
      sklearn's roc_auc_score expects higher score → label 1 (positive class).
      Negating maps: high SNR → large negative → closer to label 0 ✓.
    """
    breakdown_path = os.path.join(results_dir, "breakdown.csv")
    if os.path.exists(breakdown_path):
        df = pd.read_csv(breakdown_path)
    else:
        df = pd.DataFrame()

    # ── Bryson continuous scores ──────────────────────────────────────────────
    for (k, psf), bdf in sorted(bryson_results.items()):
        out_path = os.path.join(results_dir, f"bryson_scores_k{k}_psf{psf}.csv")
        bdf.to_csv(out_path, index=False)

        valid = bdf.dropna(subset=["bryson_score"])
        if len(valid) < 10 or valid["label"].nunique() < 2:
            print(f"  Bryson k={k} psf={psf}: insufficient data for AUC")
            continue

        # Fix 9: negate score — high SNR = FP = label 0, but sklearn expects higher → label 1
        score_for_auc = -valid["bryson_score"].values
        auc, lo, hi   = bootstrap_auc(valid["label"].values, score_for_auc, n_iter=n_bootstrap)
        print(f"  Bryson continuous k={k} psf={psf}: AUC={auc:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"(scores negated for sklearn convention)")

        mask = (df["k"] == k) & (df["psf"] == psf) if not df.empty else pd.Series([False])
        if not df.empty and mask.any():
            df.loc[mask, "bryson_auc"]    = auc
            df.loc[mask, "bryson_ci_lo"]  = lo
            df.loc[mask, "bryson_ci_hi"]  = hi
        else:
            new_row = {
                "k": k, "effective_scale_arcsec": k * KEPLER_PIXEL_SCALE_ARCSEC,
                "psf": psf, "bryson_auc": auc, "bryson_ci_lo": lo, "bryson_ci_hi": hi,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # ── Bryson binary scores (PRIMARY Bryson result) ──────────────────────────
    for (k, psf), bbin_df in sorted(bryson_binary_results.items()):
        out_path = os.path.join(results_dir, f"bryson_binary_scores_k{k}_psf{psf}.csv")
        bbin_df.to_csv(out_path, index=False)

        valid = bbin_df.dropna(subset=["binary_pred"])
        if len(valid) < 10 or valid["label"].nunique() < 2:
            print(f"  Bryson-binary k={k} psf={psf}: insufficient data for metrics")
            continue

        y_true = valid["label"].values.astype(int)
        y_pred = valid["binary_pred"].values.astype(int)

        acc          = accuracy_score(y_true, y_pred)
        f1_mac       = f1_score(y_true, y_pred, average="macro",   zero_division=0)
        rec_fp       = recall_score(y_true, y_pred, pos_label=0,   zero_division=0)
        rec_planet   = recall_score(y_true, y_pred, pos_label=1,   zero_division=0)
        prec_fp      = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        prec_planet  = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm           = confusion_matrix(y_true, y_pred)
        print(f"  Bryson-binary (PRIMARY) k={k} psf={psf}: "
              f"acc={acc:.4f}  f1_macro={f1_mac:.4f}  "
              f"recall_fp={rec_fp:.4f}  recall_planet={rec_planet:.4f}")

        # Permutation test: CNN accuracy vs Bryson-binary accuracy.
        # Uses CNN score file (binarised at OOF threshold) if available.
        perm_p_binary = np.nan
        if cnn_scores_by_tag is not None:
            tag = f"k{k}_psf{psf}"
            cnn_df = cnn_scores_by_tag.get(tag)
            if cnn_df is not None:
                merged = valid.merge(
                    cnn_df[["kepid", "score", "label"]], on="kepid", how="inner",
                    suffixes=("_bryson", "_cnn")
                )
                if len(merged) >= 10:
                    # Binarise CNN scores at 0.5 for fair comparison with the binary
                    # Bryson classifier. (OOF threshold from theModel is target-optimal
                    # but not stored here; 0.5 is a conservative reference point.)
                    cnn_pred = (merged["score"].values >= 0.5).astype(int)
                    brys_pred = merged["binary_pred"].values.astype(int)
                    y_m = merged["label_cnn"].values.astype(int)
                    _, _, _, perm_p_binary = permutation_test_accuracy_diff(
                        y_m, cnn_pred, brys_pred
                    )
                    print(f"    Permutation (acc diff, CNN-binary vs Bryson-binary) "
                          f"k={k} psf={psf}: p={perm_p_binary:.4f}")

        mask = (df["k"] == k) & (df["psf"] == psf) if not df.empty else pd.Series([False])
        if not df.empty and mask.any():
            df.loc[mask, "bryson_binary_accuracy"]      = acc
            df.loc[mask, "bryson_binary_f1_macro"]      = f1_mac
            df.loc[mask, "bryson_binary_recall_fp"]     = rec_fp
            df.loc[mask, "bryson_binary_recall_planet"] = rec_planet
            df.loc[mask, "bryson_binary_prec_fp"]       = prec_fp
            df.loc[mask, "bryson_binary_prec_planet"]   = prec_planet
            df.loc[mask, "permutation_p_cnn_bryson_binary"] = perm_p_binary
        else:
            new_row = {
                "k": k, "effective_scale_arcsec": k * KEPLER_PIXEL_SCALE_ARCSEC,
                "psf": psf,
                "bryson_binary_accuracy":      acc,
                "bryson_binary_f1_macro":      f1_mac,
                "bryson_binary_recall_fp":     rec_fp,
                "bryson_binary_recall_planet": rec_planet,
                "bryson_binary_prec_fp":       prec_fp,
                "bryson_binary_prec_planet":   prec_planet,
                "permutation_p_cnn_bryson_binary": perm_p_binary,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # ── Permutation test: CNN AUC vs Bryson continuous AUC ───────────────────
    for (k, psf), bdf in sorted(bryson_results.items()):
        if cnn_scores_by_tag is None:
            break
        tag = f"k{k}_psf{psf}"
        cnn_df = cnn_scores_by_tag.get(tag)
        if cnn_df is None:
            continue
        valid_bryson = bdf.dropna(subset=["bryson_score"])
        merged = valid_bryson.merge(
            cnn_df[["kepid", "score", "label"]], on="kepid", how="inner",
            suffixes=("_bryson", "_cnn")
        )
        if len(merged) < 10 or merged["label_cnn"].nunique() < 2:
            continue
        y_m       = merged["label_cnn"].values
        cnn_score = merged["score"].values
        # Negate Bryson score for AUC direction consistency (same as above)
        brys_neg  = -merged["bryson_score"].values
        _, _, perm_p_cont = permutation_test_auc_diff(y_m, cnn_score, brys_neg)
        print(f"    Permutation (AUC diff, CNN vs Bryson-continuous) "
              f"k={k} psf={psf}: p={perm_p_cont:.4f}")

        mask = (df["k"] == k) & (df["psf"] == psf) if not df.empty else pd.Series([False])
        if not df.empty and mask.any():
            df.loc[mask, "permutation_p_cnn_bryson_continuous"] = perm_p_cont

    # ── vetting scores ────────────────────────────────────────────────────────
    for (k, psf), vdf in sorted(vetting_results.items()):
        out_path = os.path.join(results_dir, f"vetting_scores_k{k}_psf{psf}.csv")
        vdf.to_csv(out_path, index=False)

        valid = vdf.dropna(subset=["vetting_score"])
        n_untestable = len(vdf) - len(valid)
        if len(valid) >= 10 and valid["label"].nunique() == 2:
            auc, lo, hi = bootstrap_auc(
                valid["label"].values, valid["vetting_score"].values, n_iter=n_bootstrap
            )
            print(f"  vetting k={k} psf={psf}: AUC={auc:.4f} [{lo:.4f}, {hi:.4f}]  "
                  f"untestable={n_untestable}")
            mask = (df["k"] == k) & (df["psf"] == psf) if not df.empty else pd.Series([False])
            if not df.empty and mask.any():
                df.loc[mask, "vetting_auc"]          = auc
                df.loc[mask, "vetting_ci_lo"]        = lo
                df.loc[mask, "vetting_ci_hi"]        = hi
                df.loc[mask, "n_vetting_untestable"] = n_untestable

    if not df.empty:
        df.sort_values(["k", "psf"]).to_csv(breakdown_path, index=False)
        print(f"\nbreakdown.csv updated: {breakdown_path}")


# ============================================================================
# Load test kepids from scores files
# ============================================================================

def load_test_kepids(results_dir: str) -> dict:
    """Read test kepids from scores_k*_psf*.csv files written by theModel.py."""
    import glob
    pattern = os.path.join(results_dir, "scores_k*_psf*.csv")
    result = {}
    for f in sorted(glob.glob(pattern)):
        basename = os.path.basename(f)
        # Skip ablation files
        if "flux_only" in basename or "centroid_only" in basename:
            continue
        tag = basename.replace("scores_", "").replace(".csv", "")
        df = pd.read_csv(f)
        result[tag] = set(df["kepid"].unique())
    return result


def load_cnn_scores(results_dir: str) -> dict:
    """Return dict[tag → DataFrame] with CNN scores for each (k, psf)."""
    import glob
    pattern = os.path.join(results_dir, "scores_k*_psf*.csv")
    result = {}
    for f in sorted(glob.glob(pattern)):
        basename = os.path.basename(f)
        if "flux_only" in basename or "centroid_only" in basename:
            continue
        tag = basename.replace("scores_", "").replace(".csv", "")
        result[tag] = pd.read_csv(f)
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute Bryson and optional vetting baselines"
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--manifest", default="koi_manifest.csv")
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--tpf-dir", default="./tpf_temp")
    parser.add_argument("--max-quarters", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--sigma-threshold", type=float, default=BRYSON_SIGMA_THRESHOLD,
                        help=f"SNR threshold for binary Bryson classifier "
                             f"(default {BRYSON_SIGMA_THRESHOLD})")
    parser.add_argument("--enable-vetting", action="store_true",
                        help="Run vetting package (disabled by default)")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Directory for output files (default: results_resolution)")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"ERROR: {args.manifest} not found.")
        sys.exit(1)

    manifest = pd.read_csv(args.manifest)
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Load test kepids from scores CSVs written by theModel.py
    test_kepids_by_tag = load_test_kepids(results_dir)
    if not test_kepids_by_tag:
        print("No scores files found. Run python theModel.py first.")
        sys.exit(1)

    cnn_scores_by_tag = load_cnn_scores(results_dir)

    print(f"\nComputing Bryson continuous baseline ...")
    warnings.filterwarnings("ignore")
    bryson_results = compute_bryson_scores(
        manifest, args.cache_dir, args.k_values, test_kepids_by_tag
    )

    print(f"\nComputing Bryson binary baseline (PRIMARY, σ={args.sigma_threshold}) ...")
    bryson_binary_results = compute_bryson_binary_scores(
        manifest, args.cache_dir, args.k_values, test_kepids_by_tag,
        sigma_threshold=args.sigma_threshold
    )

    vetting_results = {}
    if args.enable_vetting:
        print(f"\nRunning vetting baseline (--enable-vetting) ...")
        vetting_results = run_vetting_baseline(
            manifest, args.tpf_dir, args.cache_dir,
            args.k_values, test_kepids_by_tag, args.max_quarters
        )
    else:
        print("\nvetting baseline disabled (pass --enable-vetting to enable).")

    print(f"\nUpdating breakdown.csv ...")
    update_breakdown_csv(
        bryson_results, bryson_binary_results, vetting_results,
        results_dir, args.n_bootstrap,
        cnn_scores_by_tag=cnn_scores_by_tag,
    )

    print("\nNext step: python compare.py")


if __name__ == "__main__":
    main()
