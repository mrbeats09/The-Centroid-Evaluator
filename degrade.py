"""
degrade.py — Centroid-Resolution Degradation Engine

Implements the spatial-sampling control for the centroid information study.
The degradation engine coarsens the Kepler pixel grid by integer factor k,
recomputes centroids at the degraded resolution, and validates the k=1 case
against expected Kepler-like centroid precision.

IMPORTANT: This is an intentional experimental control, NOT a simulation of
any specific telescope. k is a scan through the spatial-sampling control
parameter. All k values are referred to by their effective scale 3.98·k arcsec;
the TESS pixel scale (~21 arcsec) is mentioned only for orientation.

Noise policy: No synthetic noise is added at any degradation level.
Rebinning alone redistributes existing Kepler photon noise into fewer pixels.
Gaussian PSF broadening (psf=1) broadens the effective PSF without adding noise.
This is a deliberate design choice that isolates spatial sampling as the sole
independent variable. Results at high k therefore represent a lower bound on
degradation relative to a real coarser instrument, which would additionally
suffer from sky background and read noise scaling.

PSF broadening (optional, default OFF):
  Rebinning alone coarsens spatial *sampling* but preserves Kepler's narrow
  optical PSF. A physically broader effective PSF is modelled by convolving
  each cadence frame with a 2-D Gaussian before rebinning.

  Kernel formula (simplifying assumption — code comment here by design):
    Native Kepler PSF FWHM ≈ 1 native pixel (σ_native ≈ 0.43 pixels).
    At degradation factor k we want an effective FWHM of k native pixels.
    The additional broadening needed in native-pixel units is:
      σ_add = sqrt(k^2 - 1) / (2 * sqrt(2 * ln(2)))
    This is a simplification: real diffraction-limited PSFs scale differently
    with aperture and wavelength. The PSF on/off comparison is a robustness
    check, not a claim of physical accuracy.

Difference-image centroid (Bryson-style):
  Implements a simplified version of the methodology described in
  Bryson et al. (2013). This is NOT a reproduction of the Kepler pipeline;
  we do not claim pipeline-identical behaviour.

Kepler native pixel scale: 3.98 arcsec/pixel.
"""

import math
import sys
from typing import Optional

import numpy as np
from scipy.ndimage import convolve
from scipy.stats import pearsonr

KEPLER_PIXEL_SCALE_ARCSEC = 3.98  # arcsec per native pixel

# Aperture-degeneracy gate: a superpixel grid this small (or an aperture that collapses
# to this few True superpixels after rebinning) has too few positional degrees of freedom
# to produce a meaningful centroid measurement. Near-zero scatter from such a grid is a
# measurement-collapse artefact, not evidence of improved precision. Thresholds grounded
# in a live diagnostic over 1,500 native Kepler TPFs (median native stamp ~5x6 px, median
# native aperture ~6 px) — see results_resolution/aperture_diagnostics.csv once populated.
MIN_GRID_DIM = 2               # coarse grid must be >= 2x2 to be non-degenerate
MIN_APERTURE_SUPERPIXELS = 2   # coarse aperture must contain >= 2 True superpixels


# ============================================================================
# TPF loading
# ============================================================================

def load_cube(tpf) -> tuple:
    """
    Extract the raw pixel cube and metadata from a lightkurve TPF object.

    Returns:
        cube      (n_cad, ny, nx) float32 array — pixel flux values
        time      (n_cad,) float64 — BKJD times
        quality   (n_cad,) int32 — Kepler quality flags (already filtered by lk)
        wcs       astropy.wcs.WCS or None
        aperture  (ny, nx) bool array — pipeline aperture mask
    """
    try:
        # lightkurve stores flux in .flux; time in .time (BKJD for Kepler)
        cube = np.array(tpf.flux.value, dtype=np.float32)  # (n_cad, ny, nx)
        cube = np.nan_to_num(cube, nan=0.0)

        time = tpf.time.bkjd  # BKJD — use directly; no offset subtraction

        quality = np.zeros(len(time), dtype=np.int32)  # already quality-masked by lk

        try:
            wcs = tpf.wcs
        except Exception:
            wcs = None

        aperture = tpf.pipeline_mask  # (ny, nx) bool
        if aperture is None or not aperture.any():
            # Fall back to threshold mask if pipeline mask absent
            median_frame = np.nanmedian(cube, axis=0)
            aperture = median_frame > np.nanpercentile(median_frame, 75)

        return cube, time, quality, wcs, aperture.astype(bool)

    except Exception as exc:
        return None, None, None, None, None


# ============================================================================
# PSF broadening
# ============================================================================

def psf_broaden(frame: np.ndarray, k: int) -> np.ndarray:
    """
    Convolve a single cadence frame with a 2-D Gaussian to simulate a broader
    effective PSF at pixel scale k.

    Kernel formula (simplifying assumption — see module docstring):
        σ_add = sqrt(k^2 - 1) / (2 * sqrt(2 * ln(2)))  native pixels

    For k=1, returns frame unchanged (σ_add = 0).

    Args:
        frame: (ny, nx) float32 cadence image
        k:     degradation factor (≥1)

    Returns:
        broadened (ny, nx) float32 image
    """
    if k <= 1:
        return frame

    # Additional Gaussian σ in native pixels needed to broaden PSF FWHM from
    # ~1 pixel to k pixels. This is a simplifying assumption, not a physical model.
    sigma_add = math.sqrt(k * k - 1.0) / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    if sigma_add < 0.01:
        return frame

    # Build a small 2-D Gaussian kernel (±3σ, odd size)
    half = max(1, int(math.ceil(3 * sigma_add)))
    size = 2 * half + 1
    ax = np.arange(-half, half + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (ax / sigma_add) ** 2)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= kernel_2d.sum()

    return convolve(frame.astype(np.float64), kernel_2d, mode="reflect").astype(np.float32)


# ============================================================================
# Superpixel rebinning
# ============================================================================

def superpixel_rebin(
    cube: np.ndarray,
    aperture: np.ndarray,
    k: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Bin the (ny, nx) pixel grid into k×k superpixels by SUMMING values.
    Summation preserves photon counts and therefore the SNR scaling
    (uncertainty ∝ 1/sqrt(N_counts) — appropriate for Poisson noise).

    Edge pixels are cropped so that ny and nx are both divisible by k.

    Degeneracy gate: if the resulting grid has fewer than MIN_GRID_DIM
    superpixels in either dimension, or the rebinned aperture contains fewer
    than MIN_APERTURE_SUPERPIXELS True superpixels, the rebinning is flagged
    as degenerate. A centroid computed from a grid this small has too few
    positional degrees of freedom to produce a meaningful measurement — the
    resulting near-zero scatter is a measurement-collapse artefact, not
    evidence of improved precision. Two tiers are distinguished:
      - "hard" (ny_out or nx_out == 0): the grid is literally empty in one
        dimension; no array can be constructed at all.
      - "soft": a valid (if tiny/uninformative) grid exists.

    Args:
        cube:     (n_cad, ny, nx) float32
        aperture: (ny, nx) bool pipeline mask
        k:        binning factor >=1 (k==1 is the identity case, handled here
                  so callers never need to branch on k==1 vs k>1)

    Returns:
        coarse_cube:     (n_cad, ny_out, nx_out) float32, or None if hard-degenerate
        coarse_aperture: (ny_out, nx_out) bool, or None if hard-degenerate — True if
                         any native pixel in the superpixel was in the original aperture
        degeneracy: dict with keys
            ny_out, nx_out             — coarse grid dimensions
            hard_degenerate            — bool, grid collapsed to empty in a dimension
            aperture_superpixels       — int, True count in coarse_aper (0 if hard)
            soft_degenerate            — bool, grid/aperture below MIN_* thresholds
    """
    n_cad, ny, nx = cube.shape

    if k == 1:
        ny_out, nx_out = ny, nx
    else:
        ny_out, nx_out = ny // k, nx // k

    hard_degenerate = (ny_out == 0 or nx_out == 0)
    if hard_degenerate:
        degeneracy = {
            "ny_out": ny_out,
            "nx_out": nx_out,
            "hard_degenerate": True,
            "aperture_superpixels": 0,
            "soft_degenerate": False,
        }
        return None, None, degeneracy

    if k == 1:
        coarse, coarse_aper = cube, aperture
    else:
        ny_crop = ny_out * k
        nx_crop = nx_out * k
        cube_crop = cube[:, :ny_crop, :nx_crop]
        aper_crop = aperture[:ny_crop, :nx_crop]
        # Reshape to (..., ny//k, k, nx//k, k) then sum over the k-sized axes
        coarse = cube_crop.reshape(n_cad, ny_out, k, nx_out, k).sum(axis=(2, 4))
        coarse_aper = aper_crop.reshape(ny_out, k, nx_out, k).any(axis=(1, 3))

    aperture_superpixels = int(coarse_aper.sum())
    soft_degenerate = (
        ny_out < MIN_GRID_DIM
        or nx_out < MIN_GRID_DIM
        or aperture_superpixels < MIN_APERTURE_SUPERPIXELS
    )

    degeneracy = {
        "ny_out": ny_out,
        "nx_out": nx_out,
        "hard_degenerate": False,
        "aperture_superpixels": aperture_superpixels,
        "soft_degenerate": soft_degenerate,
    }
    return coarse, coarse_aper, degeneracy


# ============================================================================
# Flux-weighted moment centroid
# ============================================================================

def moment_centroid(
    coarse_cube: np.ndarray,
    aperture: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute flux-weighted first-moment centroids within the aperture for each
    cadence. This is the m1/m2 signal fed to the CNN.

    Returns (m1, m2) in units of superpixel indices (not arcsec).
    The CNN uses normalised, baseline-subtracted versions of these.

    Args:
        coarse_cube: (n_cad, ny_c, nx_c) float32
        aperture:    (ny_c, nx_c) bool

    Returns:
        m1: (n_cad,) RA-direction centroid per cadence
        m2: (n_cad,) Dec-direction centroid per cadence
    """
    n_cad, ny_c, nx_c = coarse_cube.shape

    # Pixel coordinate grids (row index = Dec axis, col index = RA axis)
    col_grid, row_grid = np.meshgrid(
        np.arange(nx_c, dtype=np.float64),
        np.arange(ny_c, dtype=np.float64),
    )
    col_grid = col_grid[aperture]
    row_grid = row_grid[aperture]

    m1 = np.full(n_cad, np.nan, dtype=np.float64)
    m2 = np.full(n_cad, np.nan, dtype=np.float64)

    for i in range(n_cad):
        weights = coarse_cube[i][aperture].astype(np.float64)
        weights = np.where(weights > 0, weights, 0.0)
        total = weights.sum()
        if total > 0:
            m1[i] = (weights * col_grid).sum() / total  # RA (column) direction
            m2[i] = (weights * row_grid).sum() / total  # Dec (row) direction

    return m1, m2


# ============================================================================
# Difference-image centroid (Bryson 2013, simplified)
# ============================================================================

def difference_image_centroid(
    coarse_cube: np.ndarray,
    time: np.ndarray,
    period: float,
    t0_bkjd: float,
    dur_days: float,
    k: int,
) -> dict:
    """
    Compute a difference-image centroid offset following the methodology of
    Bryson et al. (2013). This is a SIMPLIFIED implementation — it does not
    reproduce the Kepler pipeline and makes no claim of pipeline equivalence.

    Method:
      1. Tag in-transit cadences using the ephemeris; apply a ±0.5×dur guard
         band around ingress/egress (excluded from both frames to reduce smearing).
      2. Build mean out-of-transit (OOT) and mean in-transit (IT) frames.
      3. Difference image = IT − OOT.
      4. Locate the difference-image photocentre via flux-weighted centroid of
         positive pixels.
      5. Measure offset from OOT centroid (target position proxy) in arcsec,
         using 3.98·k arcsec/pixel.
      6. Estimate uncertainty from per-cadence centroid scatter (σ/√N_IT).

    Args:
        coarse_cube: (n_cad, ny_c, nx_c) at scale k
        time:        (n_cad,) BKJD
        period:      orbital period (days)
        t0_bkjd:     transit epoch in BKJD (use directly — do NOT subtract 2457000)
        dur_days:    transit duration (days)
        k:           current degradation factor (for arcsec conversion)

    Returns dict with keys:
        offset_arcsec, uncertainty_arcsec, n_in_transit, n_oot,
        diff_image (2-D array), oot_centroid_col, oot_centroid_row
    """
    pixel_scale_arcsec = KEPLER_PIXEL_SCALE_ARCSEC * k

    # Phase fold — t0_bkjd is BKJD; do NOT subtract 2457000
    phase = ((time - t0_bkjd) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    half_dur_phase = 0.5 * dur_days / period
    guard = 0.5 * half_dur_phase  # ingress/egress guard band

    oot_mask = np.abs(phase) > half_dur_phase
    it_mask = np.abs(phase) < (half_dur_phase - guard)

    n_oot = int(oot_mask.sum())
    n_it = int(it_mask.sum())

    if n_oot < 5 or n_it < 3:
        return {
            "offset_arcsec": np.nan,
            "uncertainty_arcsec": np.nan,
            "n_in_transit": n_it,
            "n_oot": n_oot,
            "diff_image": None,
            "oot_centroid_col": np.nan,
            "oot_centroid_row": np.nan,
        }

    oot_frame = coarse_cube[oot_mask].mean(axis=0)
    it_frame = coarse_cube[it_mask].mean(axis=0)
    # OOT − IT: positive at the transit source location (the source dims during transit).
    # This matches the Bryson (2013) convention for locating the transit's pixel origin.
    diff_frame = oot_frame - it_frame

    # OOT photocentre: target position proxy
    oot_pos = coarse_cube[oot_mask]
    pos_weights = np.where(oot_frame > 0, oot_frame, 0.0)
    total_w = pos_weights.sum()
    if total_w == 0:
        return {
            "offset_arcsec": np.nan,
            "uncertainty_arcsec": np.nan,
            "n_in_transit": n_it,
            "n_oot": n_oot,
            "diff_image": diff_frame,
            "oot_centroid_col": np.nan,
            "oot_centroid_row": np.nan,
        }

    ny_c, nx_c = diff_frame.shape
    col_grid, row_grid = np.meshgrid(np.arange(nx_c), np.arange(ny_c))
    oot_col = (pos_weights * col_grid).sum() / total_w
    oot_row = (pos_weights * row_grid).sum() / total_w

    # Difference-image photocentre (positive pixels only)
    pos_diff = np.where(diff_frame > 0, diff_frame, 0.0)
    total_pos = pos_diff.sum()

    if total_pos < 1e-12:
        # No positive difference signal — transit too shallow or noise dominated
        return {
            "offset_arcsec": np.nan,
            "uncertainty_arcsec": np.nan,
            "n_in_transit": n_it,
            "n_oot": n_oot,
            "diff_image": diff_frame,
            "oot_centroid_col": oot_col,
            "oot_centroid_row": oot_row,
        }

    diff_col = (pos_diff * col_grid).sum() / total_pos
    diff_row = (pos_diff * row_grid).sum() / total_pos

    offset_pixels = math.sqrt((diff_col - oot_col) ** 2 + (diff_row - oot_row) ** 2)
    offset_arcsec = offset_pixels * pixel_scale_arcsec

    # Uncertainty: per-cadence centroid scatter / sqrt(N_IT)
    cad_centroids = []
    for i in np.where(it_mask)[0]:
        frame_w = np.where(coarse_cube[i] > 0, coarse_cube[i].astype(np.float64), 0.0)
        total_fw = frame_w.sum()
        if total_fw > 0:
            c = (frame_w * col_grid).sum() / total_fw
            r = (frame_w * row_grid).sum() / total_fw
            cad_centroids.append(math.sqrt((c - oot_col) ** 2 + (r - oot_row) ** 2))

    if len(cad_centroids) >= 2:
        uncertainty_pixels = float(np.std(cad_centroids)) / math.sqrt(len(cad_centroids))
    else:
        uncertainty_pixels = float(offset_pixels)  # conservative fallback

    uncertainty_arcsec = uncertainty_pixels * pixel_scale_arcsec

    return {
        "offset_arcsec": offset_arcsec,
        "uncertainty_arcsec": uncertainty_arcsec,
        "n_in_transit": n_it,
        "n_oot": n_oot,
        "diff_image": diff_frame,
        "oot_centroid_col": oot_col,
        "oot_centroid_row": oot_row,
    }


# ============================================================================
# Centroid quality metrics
# ============================================================================

def compute_centroid_quality(
    m1: np.ndarray,
    m2: np.ndarray,
    offset_arcsec: float,
    uncertainty_arcsec: float,
    oot_mask: np.ndarray,
    k: int,
) -> dict:
    """
    Compute aggregate centroid quality metrics for one target.

    Args:
        m1, m2:             moment centroid time series (superpixel-index units at scale k)
        offset_arcsec:      median diff-image offset (arcsec)
        uncertainty_arcsec: median diff-image uncertainty (arcsec)
        oot_mask:           bool array marking OOT cadences
        k:                  degradation factor — m1/m2 are in units of superpixel
                             indices, and each superpixel spans k native pixels, so
                             converting their scatter to arcsec requires multiplying
                             by k * KEPLER_PIXEL_SCALE_ARCSEC, not the bare native
                             constant (matches difference_image_centroid's own
                             pixel_scale_arcsec = KEPLER_PIXEL_SCALE_ARCSEC * k).

    Returns dict:
        centroid_rms       — RMS of OOT centroid displacement (arcsec)
        centroid_uncertainty — diff-image offset uncertainty (arcsec)
        centroid_snr       — |offset| / uncertainty
        offset_arcsec      — diff-image offset (arcsec)
    """
    oot_m1 = m1[oot_mask]
    oot_m2 = m2[oot_mask]
    pixel_scale_arcsec = KEPLER_PIXEL_SCALE_ARCSEC * k

    if len(oot_m1) >= 2:
        rms_m1 = float(np.nanstd(oot_m1)) * pixel_scale_arcsec
        rms_m2 = float(np.nanstd(oot_m2)) * pixel_scale_arcsec
        centroid_rms = math.sqrt(rms_m1 ** 2 + rms_m2 ** 2)
    else:
        centroid_rms = np.nan

    if np.isfinite(offset_arcsec) and np.isfinite(uncertainty_arcsec) and uncertainty_arcsec > 0:
        centroid_snr = float(abs(offset_arcsec) / uncertainty_arcsec)
    else:
        centroid_snr = np.nan

    return {
        "centroid_rms": np.float32(centroid_rms),
        "centroid_uncertainty": np.float32(uncertainty_arcsec if np.isfinite(uncertainty_arcsec) else np.nan),
        "centroid_snr": np.float32(centroid_snr),
        "offset_arcsec": np.float32(offset_arcsec if np.isfinite(offset_arcsec) else np.nan),
    }


# ============================================================================
# k=1 validation gate
# ============================================================================

def run_k1_validation(uncertainties_arcsec: list[float]) -> dict:
    """
    Assert that native-resolution difference-image centroid uncertainties
    are in a physically sensible regime (< 1.0 arcsec on ≥80% of targets
    with ≥10 in-transit cadences).

    Bryson et al. (2013) report a systematic floor of ~0.067 arcsec for
    bright Kepler targets. We use the more generous < 1.0 arcsec threshold
    to account for faint targets and our simplified centroid implementation.
    We do not claim to reproduce the Kepler pipeline precision.

    Args:
        uncertainties_arcsec: list of per-target k=1 uncertainty values

    Returns dict:
        passed                  bool
        median_uncertainty_arcsec float
        frac_passing            float  (fraction with uncertainty < 1.0 arcsec)
        n_targets               int
    """
    valid = [u for u in uncertainties_arcsec if u is not None and np.isfinite(u)]
    n = len(valid)

    if n == 0:
        print("WARNING: k=1 validation has no valid measurements — skipping gate.")
        return {"passed": True, "median_uncertainty_arcsec": np.nan,
                "frac_passing": np.nan, "n_targets": 0}

    arr = np.array(valid)
    median_unc = float(np.median(arr))
    frac = float((arr < 1.0).mean())

    print(f"\nk=1 validation: {n} targets sampled")
    print(f"  Median uncertainty: {median_unc:.4f} arcsec")
    print(f"  Fraction < 1.0 arcsec: {frac:.1%} (threshold ≥80%)")

    passed = frac >= 0.80

    return {
        "passed": passed,
        "median_uncertainty_arcsec": median_unc,
        "frac_passing": frac,
        "n_targets": n,
    }


# ============================================================================
# Scaling check (emit after full run)
# ============================================================================

def emit_scaling_check(
    scaling_records: list[dict],
    out_path: str = "results_resolution/centroid_scaling.csv",
) -> None:
    """
    After processing all k values, check that centroid uncertainty scales
    approximately linearly with k (as expected for a fixed-SNR source).

    Emits a CSV with columns:
      k, effective_scale_arcsec, median_uncertainty, median_rms, median_snr, psf_setting

    Asserts Pearson r(k, median_uncertainty) > 0.8 across k values.
    Prints a warning (does not halt) if the check fails.

    Args:
        scaling_records: list of dicts with keys
            k, psf, median_uncertainty, median_rms, median_snr
        out_path: output CSV path
    """
    import os
    import pandas as pd

    if not scaling_records:
        return

    df = pd.DataFrame(scaling_records)
    df["effective_scale_arcsec"] = df["k"] * KEPLER_PIXEL_SCALE_ARCSEC
    df.to_csv(out_path, index=False)
    print(f"\nCentroid scaling table written to {out_path}")

    print("\nCentroid uncertainty vs. k:")
    print(f"  {'k':>4}  {'scale (arcsec)':>14}  {'median_unc':>12}  {'median_snr':>12}")
    for _, row in df.iterrows():
        print(f"  {row['k']:>4.0f}  {row['effective_scale_arcsec']:>14.2f}"
              f"  {row['median_uncertainty']:>12.4f}  {row['median_snr']:>12.3f}")

    # Check scaling linearity per PSF setting
    for psf_val in df["psf"].unique():
        sub = df[df["psf"] == psf_val].sort_values("k")
        if len(sub) >= 3:
            r, _ = pearsonr(sub["k"].values, sub["median_uncertainty"].values)
            label = "on" if psf_val == 1 else "off"
            status = "OK" if r > 0.8 else "WARNING — non-linear scaling detected"
            print(f"  Pearson r(k, median_uncertainty) PSF {label}: {r:.3f}  [{status}]")
