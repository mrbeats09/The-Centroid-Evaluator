"""
getInputData.py — Kepler TPF Loading and Preprocessing Pipeline

Reads koi_manifest.csv, loads Kepler long-cadence Target Pixel Files (TPFs) from
the local filesystem (no MAST re-query during cache build), and orchestrates the
degradation + preprocessing pipeline to produce per-k training CSVs.

The pipeline has four modes:
  (default)         Download TPFs → build degraded cache → write training CSVs
  --phase download  Download TPFs only (skip cache build and CSV writing)
  --phase cache     Build degraded cache from existing local TPFs (skip download + CSV)
  --phase csv-only  Write training CSVs from existing cache (no FITS access)

Cache hierarchy (all gitignored):
  tpf_temp/                                  raw FITS (lightkurve managed)
  cache/flux/{kepoi_name}.npz                native-res folded flux per KOI
  cache/cubes/{kepid}_q{Q}_k{K}_psf{P}.npz  degraded pixel cubes
  cache/centroids/{kepid}_k{K}_psf{P}.npz   centroid time series + quality metrics

Note on cache-key design:
  Flux cache is keyed by kepoi_name (e.g. K00001.01) because each KOI has its own
  ephemeris (period, t0, duration) that determines how the light curve is phase-folded.
  Two planets around the same star require two separate folded flux series.

  Centroid cache is keyed by kepid (KIC ID) because moment centroids are a property
  of the pixel aperture and the star's position — they are the same for all KOIs on
  that star. The difference-image centroid uses the first KOI's ephemeris as a proxy
  (acknowledged limitation; centroid is per-star not per-planet).

Output: kepler_training_data_k{K}_psf{0|1}.csv  (one per degradation level + PSF setting)

BKJD epoch note: koi_time0bk is already in BKJD (BJD − 2454833.0). Used directly.
Do NOT subtract 2457000 (that is the TESS BTJD offset).
"""

import argparse
import os
import pickle
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from tqdm import tqdm

import lightkurve as lk

# Training CSV parameters
NUM_BINS = 1000
SPARSITY_THRESHOLD = 0.30  # discard if > 30% empty bins (relaxed for long cadence)

# Default degradation levels; matches plan k=[1,2,3,4,5]
DEFAULT_K_VALUES = [1, 2, 3, 4, 5]


# ============================================================================
# Helpers
# ============================================================================

def interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in a binned array by linear interpolation."""
    nans = np.isnan(arr)
    if nans.any() and (~nans).sum() >= 2:
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def log_failure(path: str, msg: str) -> None:
    """Append a failure message to a log file."""
    with open(path, "a") as f:
        f.write(msg + "\n")


# ============================================================================
# Local TPF loader (Fix 1) — replaces the MAST re-query in _load_tpfs
# ============================================================================

def load_local_tpfs(kepid: int, tpf_dir: str, max_quarters: int | None) -> list:
    """
    Load pre-downloaded Kepler long-cadence TPFs for a kepid from the local filesystem.

    Scans tpf_dir recursively for FITS files whose filename contains the zero-padded
    9-digit KIC ID (e.g. 'kplr009757613'), excluding macOS AppleDouble sidecar files
    (filenames prefixed with '._').

    Files are sorted by filename. The timestamp portion of the Kepler filename
    (e.g. kplr010872983-2009259160929_lpd-targ.fits.gz) encodes the observation
    start date, so lexicographic sort gives chronological quarter order.
    Files are capped at max_quarters if set.

    This function does NOT contact MAST or any network resource. If no files are
    found for a target, a warning is logged to failed_targets.log and [] is returned.

    Args:
        kepid:        KIC target identifier (integer)
        tpf_dir:      root directory to scan recursively (may be a deep path such as
                      /Volumes/Stuff/Research Work/TPFs — nested mastDownload/Kepler
                      subdirectories are traversed automatically)
        max_quarters: maximum number of TPF FITS files to load; None = all found

    Returns:
        List of lightkurve TPF objects (possibly empty).
    """
    kic_str = f"kplr{kepid:09d}"
    fits_files = []

    for root, _dirs, files in os.walk(tpf_dir):
        for fname in files:
            # Exclude macOS AppleDouble sidecar files (always prefixed with '._')
            if fname.startswith("._"):
                continue
            # Match Kepler long-cadence TPF pattern for this target:
            # e.g. kplr010872983-2009259160929_lpd-targ.fits.gz
            if kic_str in fname and "_lpd-targ.fits" in fname:
                fits_files.append(os.path.join(root, fname))

    if not fits_files:
        log_failure(
            "failed_targets.log",
            f"KIC {kepid}: no local TPF FITS files found under {tpf_dir}",
        )
        return []

    # Sort by filename — timestamp in name gives chronological (quarter) order
    fits_files.sort()

    if max_quarters is not None:
        fits_files = fits_files[:max_quarters]

    tpf_list = []
    for path in fits_files:
        try:
            tpf = lk.read(path)
            tpf_list.append(tpf)
        except Exception as exc:
            log_failure(
                "failed_targets.log",
                f"KIC {kepid}: could not read {os.path.basename(path)}: "
                f"{type(exc).__name__}: {exc}",
            )

    return tpf_list


# ============================================================================
# Phase 1: TPF download
# ============================================================================

def enable_cloud_storage() -> bool:
    """
    Enable anonymous S3 access to the MAST public data bucket (s3://stpubdata).
    When active, lightkurve routes downloads through AWS rather than MAST HTTPS,
    giving ~3x higher throughput per connection at no cost (bucket is public).
    Returns True if S3 was successfully enabled, False if falling back to MAST HTTPS.
    """
    try:
        from astroquery.mast import Observations
        Observations.enable_cloud_dataset(provider="AWS", profile="anon")
        print("  S3 cloud storage enabled (s3://stpubdata) — downloads will prefer AWS.")
        return True
    except Exception as exc:
        print(f"  S3 unavailable ({exc}); falling back to MAST HTTPS.")
        return False


# ============================================================================
# Search result cache (phase 1a resumability)
# ============================================================================

def _search_cache_path(cache_dir: str, kepid: int) -> str:
    return os.path.join(cache_dir, "search", f"{kepid}.pkl")


def _load_cached_search(cache_dir: str, kepid: int):
    """
    Return a cached SearchResult (or None = confirmed no data) if this kepid
    has been queried before. Returns the sentinel _NOT_CACHED if not found.
    """
    path = _search_cache_path(cache_dir, kepid)
    if not os.path.exists(path):
        return _NOT_CACHED
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return _NOT_CACHED  # corrupt cache entry: re-query


def _save_cached_search(cache_dir: str, kepid: int, result) -> None:
    """Persist a SearchResult (or None) for this kepid immediately after querying."""
    os.makedirs(os.path.join(cache_dir, "search"), exist_ok=True)
    path = _search_cache_path(cache_dir, kepid)
    try:
        with open(path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # non-fatal: worst case we re-query on next run


_NOT_CACHED = object()  # sentinel distinct from None ("confirmed no data")


class _MastRateLimiter:
    """
    Token-bucket rate limiter for MAST search API calls.

    MAST starts returning 429s or stalling connections at roughly 3-5 concurrent
    search queries. This limiter enforces a ceiling of `calls_per_second` search
    requests globally across all threads, while leaving S3 data downloads (which
    hit AWS, not MAST) completely unrestricted.
    """

    def __init__(self, calls_per_second: float = 3.0):
        self._min_interval = 1.0 / calls_per_second
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def __enter__(self):
        with self._lock:
            now = time.monotonic()
            gap = self._min_interval - (now - self._last_call)
            if gap > 0:
                time.sleep(gap)
            self._last_call = time.monotonic()
        return self

    def __exit__(self, *_):
        pass


def _search_with_retry(
    kepid: int,
    max_quarters: int | None,
    rate_limiter: _MastRateLimiter,
    max_retries: int = 5,
) -> tuple:
    """
    Query the MAST search API for one kepid, respecting the rate limiter and
    retrying on transient failures (429, connection errors) with exponential backoff.
    Returns a lightkurve SearchResult, or None if the target has no data.
    Raises on permanent failure after max_retries attempts.
    """
    for attempt in range(max_retries):
        try:
            with rate_limiter:
                result = lk.search_targetpixelfile(
                    f"KIC {kepid}", mission="Kepler", cadence="long"
                )
            if len(result) == 0:
                return None
            if max_quarters is not None and len(result) > max_quarters:
                result = result[:max_quarters]
            return result
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff: 2s, 4s, 8s, 16s
            backoff = 2 ** (attempt + 1)
            time.sleep(backoff)
    return None


def _download_with_retry(
    kepid: int,
    search_result,
    tpf_dir: str,
    max_retries: int = 4,
) -> tuple[int, bool, str]:
    """
    Download TPFs for one kepid given a pre-fetched SearchResult.
    Retries S3/network failures with exponential backoff.
    The MAST search is NOT repeated here — that already happened in the search phase.
    Returns (kepid, success, error_message).
    """
    for attempt in range(max_retries):
        try:
            search_result.download_all(
                quality_bitmask="default",
                download_dir=tpf_dir,
            )
            return kepid, True, ""
        except Exception as exc:
            if attempt == max_retries - 1:
                return kepid, False, f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** (attempt + 1))
    return kepid, False, "exceeded max retries"


def download_tpfs(
    manifest: pd.DataFrame,
    tpf_dir: str,
    cache_dir: str,
    max_quarters: int | None,
    n_workers: int = 8,
    search_rate: float = 3.0,
) -> None:
    """
    Download Kepler long-cadence TPFs for each unique kepid in the manifest.

    Two-phase design that separates the MAST-rate-limited search from the
    S3 downloads (which have no meaningful rate limit):

      Phase 1a — MAST search queries (serial, rate-limited to search_rate/sec).
                 Each result is cached to cache/search/{kepid}.pkl immediately,
                 so the process can be interrupted and resumed without re-querying
                 targets that have already been searched.
      Phase 1b — S3 data downloads (parallel, n_workers threads, retry on error).
                 lightkurve's file cache means existing FITS files are skipped.

    This prevents 429 / connection-stall errors from MAST while still saturating
    your S3 download bandwidth.
    """
    os.makedirs(tpf_dir, exist_ok=True)
    unique_kepids = manifest["kepid"].unique()
    quarters_note = f", capped at {max_quarters} quarters" if max_quarters else ""

    # ── Phase 1a: MAST search (with per-kepid disk cache) ───────────────────

    # Separate kepids into already-cached vs. need-to-query
    cached_results: dict[int, object] = {}
    to_query: list[int] = []
    for kepid in unique_kepids:
        hit = _load_cached_search(cache_dir, kepid)
        if hit is _NOT_CACHED:
            to_query.append(kepid)
        elif hit is not None:
            cached_results[kepid] = hit
        # hit is None → confirmed no data; skip silently

    n_cached = len(cached_results) + (len(unique_kepids) - len(to_query) - len(cached_results))
    print(f"\nPhase 1a: Querying MAST for {len(unique_kepids):,} targets "
          f"(rate-limited to {search_rate:.0f} req/s{quarters_note}) ...")
    if n_cached:
        print(f"  ↳ {len(unique_kepids) - len(to_query):,} already cached — "
              f"querying {len(to_query):,} new targets.")

    rate_limiter = _MastRateLimiter(calls_per_second=search_rate)
    search_results: dict[int, object] = dict(cached_results)
    search_failed: list[str] = []

    for kepid in tqdm(to_query, desc="MAST search", unit="target"):
        try:
            result = _search_with_retry(kepid, max_quarters, rate_limiter)
            _save_cached_search(cache_dir, kepid, result)  # None = confirmed no data
            if result is None:
                search_failed.append(f"KIC {kepid}: No Kepler long-cadence TPFs found")
            else:
                search_results[kepid] = result
        except Exception as exc:
            # Do not cache failures — allow retry on next run
            search_failed.append(f"KIC {kepid}: search failed: {type(exc).__name__}: {exc}")

    for msg in search_failed:
        log_failure("failed_targets.log", msg)

    print(f"Phase 1a complete: {len(search_results):,} targets found, "
          f"{len(search_failed):,} not found / failed.")

    if not search_results:
        print("No targets to download.")
        return

    print(f"\nPhase 1b: Downloading TPFs ({n_workers} concurrent S3 workers) ...")
    success = failed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_download_with_retry, kepid, sr, tpf_dir): kepid
            for kepid, sr in search_results.items()
        }
        with tqdm(total=len(search_results), desc="S3 downloads", unit="target") as pbar:
            for future in as_completed(futures):
                kepid, ok, err = future.result()
                with lock:
                    if ok:
                        success += 1
                    else:
                        failed += 1
                        log_failure("failed_targets.log", f"KIC {kepid}: {err}")
                pbar.update(1)

    print(f"Phase 1 complete: {success:,} downloaded, {failed:,} failed.")


# ============================================================================
# Phase 2: Build degraded cache
# ============================================================================

def build_cache(
    manifest: pd.DataFrame,
    tpf_dir: str,
    cache_dir: str,
    k_values: list[int],
    max_quarters: int | None,
) -> None:
    """
    Load TPFs for each kepid from the local filesystem (no MAST re-query),
    run the degradation engine at each k, and write:
      cache/flux/{kepoi_name}.npz            — per-KOI phase-folded flux (one per KOI)
      cache/centroids/{kepid}_k{K}_psf{P}.npz — centroid time series + quality metrics

    Flux is per-KOI (each KOI has its own ephemeris → own folded light curve).
    Centroids are per-KIC star (moment centroids are a property of the aperture,
    not the individual transit; difference-image centroid uses the first KOI's
    ephemeris as a representative proxy).
    """
    try:
        from degrade import (
            load_cube,
            superpixel_rebin,
            psf_broaden,
            moment_centroid,
            difference_image_centroid,
            compute_centroid_quality,
            run_k1_validation,
        )
    except ImportError:
        print("ERROR: degrade.py not found. Run Stage 4 first.")
        sys.exit(1)

    flux_dir = os.path.join(cache_dir, "flux")
    centroid_dir = os.path.join(cache_dir, "centroids")
    os.makedirs(flux_dir, exist_ok=True)
    os.makedirs(centroid_dir, exist_ok=True)

    unique_kepids = manifest["kepid"].unique()
    koi_by_kepid = manifest.groupby("kepid")

    # First pass: check k=1 validation gate across a sample of targets
    print("\nPhase 2: Running k=1 validation gate before full cache build...")
    k1_uncertainties = []
    sample_kepids = unique_kepids[:min(50, len(unique_kepids))]
    for kepid in tqdm(sample_kepids, desc="k=1 validation sample", unit="target"):
        rows = koi_by_kepid.get_group(kepid)
        row = rows.iloc[0]
        try:
            tpf_list = load_local_tpfs(kepid, tpf_dir, max_quarters)
            if not tpf_list:
                continue
            for tpf in tpf_list:
                cube, time, quality, wcs, aperture = load_cube(tpf)
                if cube is None:
                    continue
                # k=1: no rebinning; use first KOI's ephemeris for validation check
                unc = _get_k1_uncertainty(
                    cube, time, quality, aperture,
                    float(row["koi_period"]),
                    float(row["koi_time0bk"]),
                    float(row["koi_duration"]) / 24.0,
                    difference_image_centroid,
                )
                if unc is not None:
                    k1_uncertainties.append(unc)
                break  # one quarter is enough for validation
        except Exception as exc:
            log_failure("failed_targets.log",
                        f"KIC {kepid}: k=1 validation: {type(exc).__name__}: {exc}")
            continue

    gate_result = run_k1_validation(k1_uncertainties)
    if not gate_result["passed"]:
        os.makedirs("results_resolution", exist_ok=True)
        with open("results_resolution/k1_validation_FAILED.txt", "w") as f:
            f.write(
                f"k=1 validation FAILED\n"
                f"Median uncertainty: {gate_result['median_uncertainty_arcsec']:.4f} arcsec\n"
                f"Fraction < 1.0 arcsec: {gate_result['frac_passing']:.2%}\n"
                f"Required: >= 80% of targets with uncertainty < 1.0 arcsec\n\n"
                f"Diagnosis: the difference_image_centroid function may have a bug.\n"
                f"Inspect the centroid computation in degrade.py before proceeding.\n"
            )
        print(f"\nERROR: k=1 validation FAILED. See results_resolution/k1_validation_FAILED.txt")
        sys.exit(1)

    print(f"k=1 validation PASSED. Median uncertainty: "
          f"{gate_result['median_uncertainty_arcsec']:.4f} arcsec")

    # Full cache build
    print(f"\nBuilding cache for k = {k_values} × PSF {{off, on}}...")
    for kepid in tqdm(unique_kepids, desc="Building cache", unit="target"):
        rows = koi_by_kepid.get_group(kepid)
        try:
            tpf_list = load_local_tpfs(kepid, tpf_dir, max_quarters)
            if not tpf_list:
                continue

            # Flux cache: one file per KOI, using that KOI's own ephemeris.
            # A star with two planets gets two separate folded flux series.
            for _, row in rows.iterrows():
                flux_path = os.path.join(flux_dir, f"{row['kepoi_name']}.npz")
                if not os.path.exists(flux_path):
                    _compute_flux_cache(row, tpf_list, flux_path)

            # Centroid cache: one file per (kepid, k, psf). Centroids are per-star.
            for k in k_values:
                for psf in [0, 1]:
                    out_path = os.path.join(centroid_dir,
                                            f"{kepid}_k{k}_psf{psf}.npz")
                    if os.path.exists(out_path):
                        continue
                    _compute_centroid_cache(
                        kepid, tpf_list, rows, k, psf, out_path,
                        load_cube, superpixel_rebin, psf_broaden,
                        moment_centroid, difference_image_centroid,
                        compute_centroid_quality,
                    )

        except Exception as exc:
            log_failure("failed_targets.log",
                        f"KIC {kepid}: cache build: {type(exc).__name__}: {exc}")

    print("Phase 2 complete.")


def _load_tpfs(kepid: int, tpf_dir: str, max_quarters: int | None) -> list:
    """
    Load cached TPFs for a kepid from tpf_dir.

    Delegates to load_local_tpfs — no MAST re-query. This function is kept
    for backward compatibility with any callers; load_local_tpfs is the
    canonical exported entry point.
    """
    return load_local_tpfs(kepid, tpf_dir, max_quarters)


def _compute_flux_cache(row: pd.Series, tpf_list: list, out_path: str) -> None:
    """
    Compute native-resolution phase-folded, binned flux for one KOI.

    Uses this KOI's own ephemeris (period, t0, duration) to fold the light curve.
    This is correct: two KOIs on the same star have different transit times and
    periods, so each needs its own phase-folded representation.

    Writes cache/flux/{kepoi_name}.npz with arrays: flux_binned, bin_centres.

    koi_time0bk is in BKJD (BJD − 2454833) — use directly, no offset subtraction.
    Do NOT subtract 2457000 (that is the TESS BTJD offset).

    Args:
        row:      single KOI row from the manifest (period, t0, duration, kepoid_name)
        tpf_list: list of TPF objects for the parent KIC star
        out_path: destination path for the .npz output
    """
    kepid = int(row["kepid"])
    period = float(row["koi_period"])
    # koi_time0bk is BKJD — use directly; do NOT subtract 2457000 (TESS offset)
    t0_bkjd = float(row["koi_time0bk"])
    dur_days = float(row["koi_duration"]) / 24.0

    time_parts, flux_parts = [], []
    for tpf in tpf_list:
        try:
            lc = tpf.to_lightcurve(aperture_mask="pipeline")
            t = lc.time.bkjd
            f = np.array(lc.flux.value, dtype=np.float64)
            # Per-quarter median normalisation (replicate stitch behaviour)
            med = np.nanmedian(f)
            if med == 0 or not np.isfinite(med):
                continue
            time_parts.append(t)
            flux_parts.append(f / med)
        except Exception as exc:
            log_failure("failed_targets.log",
                        f"KIC {kepid} {row['kepoi_name']}: flux lc extraction: "
                        f"{type(exc).__name__}: {exc}")
            continue

    if not time_parts:
        return

    time_arr = np.concatenate(time_parts)
    flux_arr = np.concatenate(flux_parts)
    idx = np.argsort(time_arr)
    time_arr, flux_arr = time_arr[idx], flux_arr[idx]

    # 3-sigma clip to remove instrumental glitches and cosmic rays
    med, std = np.nanmedian(flux_arr), np.nanstd(flux_arr)
    good = np.abs(flux_arr - med) <= 3 * std
    time_arr, flux_arr = time_arr[good], flux_arr[good]

    # Phase fold — t0_bkjd already in BKJD
    phase = ((time_arr - t0_bkjd) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    half_window = 1.5 * (dur_days / period)
    oot_mask = np.abs(phase) > half_window
    oot_flux = flux_arr[oot_mask]
    if len(oot_flux) < 10:
        return
    flux_arr = flux_arr / np.nanmedian(oot_flux)

    idx = np.argsort(phase)
    phase, flux_arr = phase[idx], flux_arr[idx]

    flux_binned, _, _ = binned_statistic(
        phase, flux_arr, statistic="median",
        bins=NUM_BINS, range=(-0.5, 0.5)
    )
    bin_edges = np.linspace(-0.5, 0.5, NUM_BINS + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    empty_frac = np.isnan(flux_binned).mean()
    if empty_frac > SPARSITY_THRESHOLD:
        log_failure("sparse_targets.log",
                    f"KIC {kepid} {row['kepoi_name']}: {empty_frac:.1%} empty bins "
                    f"— discarded (flux cache)")
        return

    flux_binned = interpolate_nans(flux_binned)

    oot_bins = flux_binned[np.abs(bin_centres) > 2 * (dur_days / period)]
    if len(oot_bins) < 10:
        return
    oot_std = np.nanstd(oot_bins)
    if oot_std < 1e-10:
        return
    flux_binned = (flux_binned - np.nanmedian(oot_bins)) / oot_std

    np.savez_compressed(out_path, flux_binned=flux_binned, bin_centres=bin_centres)


def _get_k1_uncertainty(cube, time, quality, aperture, period, t0_bkjd, dur_days,
                        difference_image_centroid):
    """Return k=1 diff-image uncertainty in arcsec, or None on failure."""
    try:
        result = difference_image_centroid(
            cube, time, period, t0_bkjd, dur_days, k=1
        )
        return result.get("uncertainty_arcsec")
    except Exception:
        return None


def _compute_centroid_cache(
    kepid, tpf_list, koi_rows, k, psf,
    out_path,
    load_cube, superpixel_rebin, psf_broaden,
    moment_centroid, difference_image_centroid,
    compute_centroid_quality,
) -> None:
    """
    Compute centroid time series and quality metrics at scale k for this kepid.
    Writes cache/centroids/{kepid}_k{k}_psf{psf}.npz.

    The difference-image centroid uses koi_rows.iloc[0]'s ephemeris as a proxy for
    the star's transit timing. For multi-KOI systems this is an acknowledged
    approximation — centroids are per-star, not per-planet.
    koi_time0bk is in BKJD — use directly; do NOT subtract 2457000 (TESS offset).
    """
    row = koi_rows.iloc[0]
    period = float(row["koi_period"])
    t0_bkjd = float(row["koi_time0bk"])  # BKJD — use directly
    dur_days = float(row["koi_duration"]) / 24.0

    m1_parts, m2_parts, time_parts = [], [], []
    diff_offsets, diff_uncertainties = [], []

    for tpf in tpf_list:
        try:
            cube, time, quality, wcs, aperture = load_cube(tpf)
            if cube is None or len(time) == 0:
                continue

            # PSF broadening (optional; k=1 is identity since sqrt(1-1)=0)
            if psf == 1 and k > 1:
                cube = np.stack([psf_broaden(frame, k) for frame in cube])

            # Superpixel rebinning (k=1 is identity)
            if k > 1:
                coarse_cube, coarse_aperture = superpixel_rebin(cube, aperture, k)
            else:
                coarse_cube, coarse_aperture = cube, aperture

            # Moment centroid time series
            m1, m2 = moment_centroid(coarse_cube, coarse_aperture)

            # Difference-image centroid offset
            diff_result = difference_image_centroid(
                coarse_cube, time, period, t0_bkjd, dur_days, k=k
            )

            time_parts.append(time)
            m1_parts.append(m1)
            m2_parts.append(m2)
            if diff_result.get("offset_arcsec") is not None:
                diff_offsets.append(diff_result["offset_arcsec"])
            if diff_result.get("uncertainty_arcsec") is not None:
                diff_uncertainties.append(diff_result["uncertainty_arcsec"])

        except Exception as exc:
            log_failure("failed_targets.log",
                        f"KIC {kepid} k={k} psf={psf}: {type(exc).__name__}: {exc}")
            continue

    if not time_parts:
        return

    time_arr = np.concatenate(time_parts)
    m1_arr = np.concatenate(m1_parts)
    m2_arr = np.concatenate(m2_parts)

    idx = np.argsort(time_arr)
    time_arr = time_arr[idx]
    m1_arr = m1_arr[idx]
    m2_arr = m2_arr[idx]

    # Phase fold — t0_bkjd is BKJD
    phase = ((time_arr - t0_bkjd) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    half_window = 1.5 * (dur_days / period)
    oot_mask = np.abs(phase) > half_window

    # Compute quality metrics
    offset_median = float(np.median(diff_offsets)) if diff_offsets else np.nan
    uncertainty_median = float(np.median(diff_uncertainties)) if diff_uncertainties else np.nan
    quality_metrics = compute_centroid_quality(
        m1_arr, m2_arr, offset_median, uncertainty_median, oot_mask
    )

    np.savez_compressed(
        out_path,
        time=time_arr,
        phase=phase,
        m1=m1_arr,
        m2=m2_arr,
        oot_mask=oot_mask,
        diff_offset_arcsec=np.array(diff_offsets) if diff_offsets else np.array([np.nan]),
        diff_uncertainty_arcsec=np.array(diff_uncertainties) if diff_uncertainties else np.array([np.nan]),
        **quality_metrics,
    )


# ============================================================================
# Phase 3: Write training CSVs from cache
# ============================================================================

def write_training_csvs(
    manifest: pd.DataFrame,
    cache_dir: str,
    k_values: list[int],
) -> dict:
    """
    Read from cache/flux/ (per-KOI) and cache/centroids/ (per-KIC) and write one
    training CSV per (k, psf). No FITS files are opened here.

    Flux cache is keyed by kepoi_name; centroid cache is keyed by kepid.
    A star with two planets writes two CSV rows (one per KOI) using the shared
    centroid data but each KOI's own flux series.

    Output schema (3009 columns):
      kepid, kepoi_name, label, fp_subtype, koi_period, koi_time0bk, koi_duration,
      f_0..f_999, m1_0..m1_999, m2_0..m2_999

    Returns:
        csv_counts dict[(k, psf)] → int rows written, for the attrition report.
    """
    flux_dir = os.path.join(cache_dir, "flux")
    centroid_dir = os.path.join(cache_dir, "centroids")

    koi_by_kepid = manifest.groupby("kepid")
    unique_kepids = manifest["kepid"].unique()

    csv_counts = {}

    for k in k_values:
        for psf in [0, 1]:
            csv_path = f"kepler_training_data_k{k}_psf{psf}.csv"
            print(f"\nWriting {csv_path} ...")

            header_written = False
            success, skipped = 0, 0

            for kepid in tqdm(unique_kepids, desc=f"k={k} psf={psf}", unit="target"):
                # Centroid data is per-star; load once for all KOIs of this kepid
                centroid_path = os.path.join(centroid_dir,
                                             f"{kepid}_k{k}_psf{psf}.npz")
                if not os.path.exists(centroid_path):
                    skipped += len(koi_by_kepid.get_group(kepid))
                    continue

                try:
                    centroid_data = np.load(centroid_path)
                    phase = centroid_data["phase"]
                    m1_arr = centroid_data["m1"]
                    m2_arr = centroid_data["m2"]
                    oot_mask = centroid_data["oot_mask"]

                    # Normalise centroid channels: OOT baseline subtract + standardise
                    m1_oot = m1_arr[oot_mask]
                    m2_oot = m2_arr[oot_mask]

                    m1_arr = m1_arr - np.nanmedian(m1_oot)
                    m2_arr = m2_arr - np.nanmedian(m2_oot)

                    m1_std = np.nanstd(m1_arr[oot_mask])
                    m2_std = np.nanstd(m2_arr[oot_mask])
                    if m1_std > 1e-10:
                        m1_arr = m1_arr / m1_std
                    if m2_std > 1e-10:
                        m2_arr = m2_arr / m2_std

                    # Bin centroid time series to NUM_BINS phase bins
                    m1_binned, _, _ = binned_statistic(
                        phase, m1_arr, statistic="median",
                        bins=NUM_BINS, range=(-0.5, 0.5)
                    )
                    m2_binned, _, _ = binned_statistic(
                        phase, m2_arr, statistic="median",
                        bins=NUM_BINS, range=(-0.5, 0.5)
                    )

                    empty_frac = np.isnan(m1_binned).mean()
                    if empty_frac > SPARSITY_THRESHOLD:
                        log_failure("sparse_targets.log",
                                    f"KIC {kepid} k={k}: {empty_frac:.1%} empty bins (centroid)")
                        skipped += len(koi_by_kepid.get_group(kepid))
                        continue

                    m1_binned = interpolate_nans(m1_binned)
                    m2_binned = interpolate_nans(m2_binned)

                except Exception as exc:
                    log_failure("failed_targets.log",
                                f"KIC {kepid} k={k} centroid load: "
                                f"{type(exc).__name__}: {exc}")
                    skipped += len(koi_by_kepid.get_group(kepid))
                    continue

                # Write one row per KOI, each using its own flux cache
                rows = koi_by_kepid.get_group(kepid)
                for _, row in rows.iterrows():
                    kepoi_name = row["kepoi_name"]
                    flux_path = os.path.join(flux_dir, f"{kepoi_name}.npz")

                    if not os.path.exists(flux_path):
                        skipped += 1
                        continue

                    try:
                        flux_data = np.load(flux_path)
                        flux_binned = flux_data["flux_binned"]

                        entry = {
                            "kepid": kepid,
                            "kepoi_name": kepoi_name,
                            "label": int(row["label"]),
                            "fp_subtype": row.get("fp_subtype", ""),
                            "koi_period": row["koi_period"],
                            "koi_time0bk": row["koi_time0bk"],
                            "koi_duration": row["koi_duration"],
                        }
                        for i in range(NUM_BINS):
                            entry[f"f_{i}"] = float(flux_binned[i])
                        for i in range(NUM_BINS):
                            entry[f"m1_{i}"] = float(m1_binned[i])
                        for i in range(NUM_BINS):
                            entry[f"m2_{i}"] = float(m2_binned[i])

                        df_row = pd.DataFrame([entry])
                        df_row.to_csv(csv_path, mode="a",
                                      header=not header_written, index=False)
                        header_written = True
                        success += 1

                    except Exception as exc:
                        log_failure("failed_targets.log",
                                    f"KIC {kepid} {kepoi_name} k={k} csv: "
                                    f"{type(exc).__name__}: {exc}")
                        skipped += 1

            csv_counts[(k, psf)] = success
            print(f"  Done: {success:,} KOI rows written, {skipped:,} skipped/failed")

    return csv_counts


def write_data_attrition_report(
    manifest: pd.DataFrame,
    cache_dir: str,
    k_values: list[int],
    csv_counts: dict,
    results_dir: str,
) -> None:
    """
    Write results_resolution/data_attrition.txt summarising how many KOIs were
    dropped at each pipeline stage.

    Sources:
      - manifest: total KOI and confirmed/FP counts
      - failed_targets.log: zero-local-TPF skips
      - sparse_targets.log: sparsity filter skips
      - csv_counts: final row counts per (k, psf) from write_training_csvs

    Args:
        manifest:    full KOI manifest DataFrame
        cache_dir:   root cache directory (unused currently; reserved for future counts)
        k_values:    degradation factors to report
        csv_counts:  dict[(k, psf)] → int rows written (from write_training_csvs)
        results_dir: directory to write data_attrition.txt into
    """
    os.makedirs(results_dir, exist_ok=True)

    n_total = len(manifest)
    n_confirmed = int((manifest["label"] == 1).sum())
    n_fp = int((manifest["label"] == 0).sum())

    # Count unique KIC IDs with zero local TPF files (distinct from download failures)
    no_tpf_kics: set[int] = set()
    if os.path.exists("failed_targets.log"):
        with open("failed_targets.log") as fh:
            for line in fh:
                if "no local TPF FITS files found" in line:
                    # Log format: "KIC {kepid}: no local TPF FITS files found ..."
                    try:
                        kic_part = line.split(":")[0]
                        no_tpf_kics.add(int(kic_part.replace("KIC", "").strip()))
                    except (ValueError, IndexError):
                        pass

    # Count unique KIC IDs that tripped the sparsity filter
    sparse_kics: set[int] = set()
    if os.path.exists("sparse_targets.log"):
        with open("sparse_targets.log") as fh:
            for line in fh:
                try:
                    kic_part = line.split(":")[0]
                    sparse_kics.add(int(kic_part.replace("KIC", "").strip()))
                except (ValueError, IndexError):
                    pass

    lines = [
        "DATA ATTRITION REPORT",
        "=" * 60,
        f"Total KOIs in manifest:          {n_total:>7,}",
        f"  Confirmed planets (label=1):   {n_confirmed:>7,}",
        f"  False positives  (label=0):    {n_fp:>7,}",
        f"  Unique KIC targets:            {manifest['kepid'].nunique():>7,}",
        "",
        "KIC targets with zero local TPF files found:",
        f"  (logged to failed_targets.log)   {len(no_tpf_kics):>7,} unique KIC targets",
        "",
        "KIC targets that tripped the sparsity filter (>30% empty bins):",
        f"  (logged to sparse_targets.log)   {len(sparse_kics):>7,} unique KIC targets",
        "  Note: one KIC may appear multiple times if tripped at several k values.",
        "",
        "Final row counts per training CSV:",
        f"  {'CSV file':<40}  {'rows written':>12}",
        "  " + "-" * 53,
    ]

    for k in k_values:
        for psf in [0, 1]:
            fname = f"kepler_training_data_k{k}_psf{psf}.csv"
            count = csv_counts.get((k, psf), "not written")
            count_str = f"{count:>12,}" if isinstance(count, int) else f"  {count:>10}"
            lines.append(f"  {fname:<40}  {count_str}")

    lines += [
        "",
        "=" * 60,
    ]

    out_path = os.path.join(results_dir, "data_attrition.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nData attrition report → {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Load local Kepler TPFs and build per-k training CSVs"
    )
    parser.add_argument(
        "--max-quarters", type=int, default=None,
        help="Cap the number of quarters per target (default: all available)"
    )
    parser.add_argument(
        "--phase",
        choices=["download", "cache", "csv-only", "all"],
        default="all",
        help=(
            "Pipeline phase to run: "
            "'download' = TPF download only; "
            "'cache' = build degraded cache from existing local TPFs; "
            "'csv-only' = write training CSVs from existing cache; "
            "'all' (default) = run all phases in order"
        ),
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
        help=f"Pixel-scale factors to process (default: {DEFAULT_K_VALUES})"
    )
    parser.add_argument(
        "--manifest", default="koi_manifest.csv",
        help="Path to KOI manifest CSV (default: koi_manifest.csv)"
    )
    parser.add_argument(
        "--tpf-dir", default="./tpf_temp",
        help=(
            "Directory for Kepler TPF FITS files "
            "(default: ./tpf_temp, which is gitignored). "
            "Scanned recursively — a path like "
            "'/Volumes/Stuff/Research Work/TPFs' works directly. "
            "If you specify a custom path, add it to .gitignore manually — "
            "TPF archives can be tens of GB."
        ),
    )
    parser.add_argument(
        "--cache-dir", default="./cache",
        help="Directory for computed intermediate cache files (default: ./cache)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help=(
            "Number of concurrent S3 download threads (default: 8). "
            "Has no effect with --phase csv-only."
        ),
    )
    parser.add_argument(
        "--search-rate", type=float, default=3.0,
        help=(
            "Max MAST search API calls per second (default: 3.0). "
            "Raise cautiously — MAST soft-throttles at ~5 req/s."
        ),
    )
    parser.add_argument(
        "--results-dir", default="results_resolution",
        help="Directory for attrition report (default: results_resolution)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"ERROR: {args.manifest} not found. Run getExamples.py first.")
        sys.exit(1)

    manifest = pd.read_csv(args.manifest)
    print(f"Loaded manifest: {len(manifest):,} KOIs, "
          f"{manifest['kepid'].nunique():,} unique KIC targets")

    tpf_dir = args.tpf_dir
    cache_dir = args.cache_dir

    # Clear log files at the start of a fresh run (download phase only)
    if args.phase in ("download", "all"):
        for log in ["failed_targets.log", "sparse_targets.log"]:
            open(log, "w").close()

    warnings.filterwarnings("ignore")

    run_download = args.phase in ("download", "all")
    run_cache = args.phase in ("cache", "all")
    run_csv = args.phase in ("csv-only", "all")

    if run_download:
        enable_cloud_storage()
        download_tpfs(
            manifest, tpf_dir, cache_dir, args.max_quarters,
            n_workers=args.workers,
            search_rate=args.search_rate,
        )

    if run_cache:
        build_cache(manifest, tpf_dir, cache_dir, args.k_values, args.max_quarters)

    csv_counts = {}
    if run_csv:
        csv_counts = write_training_csvs(manifest, cache_dir, args.k_values)
        write_data_attrition_report(
            manifest, cache_dir, args.k_values, csv_counts, args.results_dir
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
