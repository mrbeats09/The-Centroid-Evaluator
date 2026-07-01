"""
getInputData.py — Kepler TPF Loading and Preprocessing Pipeline

Reads koi_manifest.csv, loads Kepler long-cadence Target Pixel Files (TPFs) and
light-curve (LC) files from the local filesystem (no MAST re-query during cache
build), and orchestrates the degradation + preprocessing pipeline to produce per-k
training CSVs.

The pipeline has four modes:
  (default)         Download TPFs + LCs → build degraded cache → write training CSVs
  --phase download  Download TPFs + LCs only (skip cache build and CSV writing)
  --phase cache     Build degraded cache from existing local files (skip download + CSV)
  --phase csv-only  Write training CSVs from existing cache (no FITS access)

Preprocessing improvements over the prior version:
  - PDC-SAP flux from LC files (replaces SAP from TPFs) to remove instrumental
    systematics (thermal drifts, attitude tweaks, focus changes).
  - Per-quarter Savitzky-Golay detrending to remove stellar variability and
    residual quarter-level trends that survive PDC.
  - Multi-planet masking: cadences contaminated by other KOIs on the same star
    are excluded before phase-folding and OOT normalisation.
  - Global bins reduced from 1000 to 301 (AstroNet convention).
  - Adaptive local view (61 bins, ±2×transit_duration in phase) stored per KOI.

Cache hierarchy (all gitignored):
  tpf_temp/                                  raw FITS (lightkurve managed)
  cache/flux/{kepoi_name}.npz                native-res folded flux per KOI
  cache/cubes/{kepid}_q{Q}_k{K}_psf{P}.npz  degraded pixel cubes
  cache/centroids/{kepid}_k{K}_psf{P}.npz   centroid time series + quality metrics
  cache/search/{kepid}.pkl                   TPF MAST search result cache
  cache/search/lc_{kepid}.pkl                LC MAST search result cache

Note on cache-key design:
  Flux cache is keyed by kepoi_name (e.g. K00001.01) because each KOI has its own
  ephemeris (period, t0, duration) that determines how the light curve is phase-folded.
  Two planets around the same star require two separate folded flux series.

  Centroid cache is keyed by kepid (KIC ID) because moment centroids are a property
  of the pixel aperture and the star's position — they are the same for all KOIs on
  that star. The difference-image centroid uses the first KOI's ephemeris as a proxy
  (acknowledged limitation; centroid is per-star not per-planet).

Output: kepler_training_data_k{K}_psf{0|1}.csv  (one per degradation level + PSF setting)
Output schema (971 columns per row):
  7 metadata: kepid, kepoi_name, label, fp_subtype, koi_period, koi_time0bk, koi_duration
  f_0..f_300   (301 global flux bins — OOT-normalised, OOT-standardised)
  m1_0..m1_300 (301 global RA centroid bins at scale k)
  m2_0..m2_300 (301 global Dec centroid bins at scale k)
  l_0..l_60    (61 adaptive local flux bins, ±2×transit_duration in phase)

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
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
from tqdm import tqdm

import lightkurve as lk

# Training CSV parameters
NUM_BINS = 301   # was 1000; matches AstroNet global view bin count
LOCAL_BINS = 61  # adaptive local view bins (per _extract_local_bins)
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


def _log_aperture_diagnostic(path: str, row: dict) -> None:
    """Append one row to aperture_diagnostics.csv; write header only if the file is new."""
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=header)


# ============================================================================
# Local TPF loader — no MAST re-query during cache build
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
# Local LC loader — PDC-SAP source; sibling of load_local_tpfs
# ============================================================================

def load_local_lcs(kepid: int, tpf_dir: str, max_quarters: int | None) -> list:
    """
    Scan tpf_dir recursively for Kepler long-cadence light-curve FITS files
    matching this kepid. Returns a list of KeplerLightCurve objects loaded via
    lk.read(), sorted by filename (chronological quarter order), capped at
    max_quarters if set. macOS ._-prefixed sidecars are excluded.

    LC files follow the naming pattern: kplr{kepid:09d}*_llc.fits*
    (as opposed to TPF files which match *_lpd-targ.fits*).

    This function does NOT contact MAST or any network resource. If no files are
    found, a warning is logged to failed_targets.log (prefix LC_LOAD:) and []
    is returned.

    PDC-SAP flux is the default when loading Kepler LC files via lk.read().
    The flux column accessed via lc.flux.value returns PDC-SAP.

    Args:
        kepid:        KIC target identifier (integer)
        tpf_dir:      root directory to scan recursively
        max_quarters: cap on number of LC files to load; None = all found

    Returns:
        List of lightkurve LightCurve objects (possibly empty).
    """
    kic_str = f"kplr{kepid:09d}"
    fits_files = []

    for root, _dirs, files in os.walk(tpf_dir):
        for fname in files:
            # Exclude macOS AppleDouble sidecar files
            if fname.startswith("._"):
                continue
            # Match Kepler long-cadence LC pattern:
            # e.g. kplr009757613-2009259160929_llc.fits.gz
            if kic_str in fname and "_llc.fits" in fname:
                fits_files.append(os.path.join(root, fname))

    if not fits_files:
        log_failure(
            "failed_targets.log",
            f"LC_LOAD: KIC {kepid}: no local LC FITS files found under {tpf_dir}",
        )
        return []

    # Sort by filename — timestamp in name gives chronological (quarter) order
    fits_files.sort()

    if max_quarters is not None:
        fits_files = fits_files[:max_quarters]

    lc_list = []
    for path in fits_files:
        try:
            lc = lk.read(path)
            lc_list.append(lc)
        except Exception as exc:
            log_failure(
                "failed_targets.log",
                f"LC_LOAD: KIC {kepid}: could not read {os.path.basename(path)}: "
                f"{type(exc).__name__}: {exc}",
            )

    return lc_list


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


def _lc_search_cache_path(cache_dir: str, kepid: int) -> str:
    """LC search cache path — uses lc_ prefix to distinguish from TPF cache."""
    return os.path.join(cache_dir, "search", f"lc_{kepid}.pkl")


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
    Query the MAST search API for one kepid (TPF), respecting the rate limiter
    and retrying on transient failures with exponential backoff.
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


def _search_lc_with_retry(
    kepid: int,
    rate_limiter: _MastRateLimiter,
    max_retries: int = 5,
):
    """
    Query MAST for Kepler long-cadence light-curve files for one kepid.
    Sibling of _search_with_retry; calls search_lightcurve instead of
    search_targetpixelfile. All available quarters are returned — no
    max_quarters cap on LC downloads (LC and TPF quarter counts may differ;
    using all LC quarters maximises the PDC-SAP baseline coverage).
    Retries transient failures with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            with rate_limiter:
                result = lk.search_lightcurve(
                    f"KIC {kepid}", mission="Kepler", cadence="long"
                )
            if len(result) == 0:
                return None
            return result  # All quarters — no cap applied
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
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


def _download_lc_with_retry(
    kepid: int,
    search_result,
    tpf_dir: str,
    max_retries: int = 4,
) -> tuple[int, bool, str]:
    """
    Download LC files for one kepid given a pre-fetched SearchResult.
    Sibling of _download_with_retry; downloads into the same tpf_dir tree.
    Lightkurve places LC files under {tpf_dir}/mastDownload/Kepler/kplr.../
    alongside any existing TPF files.
    Returns (kepid, success, error_message).
    """
    for attempt in range(max_retries):
        try:
            search_result.download_all(download_dir=tpf_dir)
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

    tqdm.write(f"Phase 1 complete: {success:,} downloaded, {failed:,} failed.")


def download_lcs(
    manifest: pd.DataFrame,
    tpf_dir: str,
    cache_dir: str,
    n_workers: int = 8,
    search_rate: float = 3.0,
) -> None:
    """
    Download Kepler long-cadence light-curve (LLC) FITS files for each unique
    kepid in the manifest. Mirrors download_tpfs with two differences:

      - Uses lk.search_lightcurve (not search_targetpixelfile).
      - Downloads ALL available quarters regardless of --max-quarters, so that
        PDC-SAP flux covers the full mission baseline for each target (TPF and LC
        quarter counts may differ; capping LC quarters could produce a shorter flux
        baseline than the centroid time series built from TPFs).
      - LC search results are cached to cache/search/lc_{kepid}.pkl (prefix
        distinguishes them from the TPF search cache {kepid}.pkl).
      - Download failures are logged to failed_targets.log with prefix LC:.

    Two-phase design (same as download_tpfs):
      Phase 1c-a — MAST search (rate-limited, resumable via disk cache)
      Phase 1c-b — Parallel downloads (retry on error)

    Lightkurve places LC files under {tpf_dir}/mastDownload/Kepler/kplr{KIC:09d}_lc_Q*/
    interleaved with TPF files in the same directory tree.
    """
    os.makedirs(tpf_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "search"), exist_ok=True)
    unique_kepids = manifest["kepid"].unique()

    # ── Phase 1c-a: MAST search for LC files ────────────────────────────────

    cached_results: dict[int, object] = {}
    to_query: list[int] = []

    for kepid in unique_kepids:
        path = _lc_search_cache_path(cache_dir, kepid)
        if not os.path.exists(path):
            to_query.append(kepid)
        else:
            try:
                with open(path, "rb") as fh:
                    hit = pickle.load(fh)
                if hit is not None:
                    cached_results[kepid] = hit
                # hit is None → confirmed no LC data; skip silently
            except Exception:
                to_query.append(kepid)  # corrupt cache: re-query

    print(f"\nPhase 1c-a: Querying MAST for {len(unique_kepids):,} LC targets "
          f"(rate-limited to {search_rate:.0f} req/s, all quarters) ...")
    if cached_results:
        print(f"  ↳ {len(unique_kepids) - len(to_query):,} already cached — "
              f"querying {len(to_query):,} new targets.")

    rate_limiter = _MastRateLimiter(calls_per_second=search_rate)
    search_results: dict[int, object] = dict(cached_results)
    search_failed: list[str] = []

    for kepid in tqdm(to_query, desc="MAST LC search", unit="target"):
        try:
            result = _search_lc_with_retry(kepid, rate_limiter)
            # Cache immediately (None = confirmed no LC data for this target)
            path = _lc_search_cache_path(cache_dir, kepid)
            try:
                with open(path, "wb") as fh:
                    pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass  # non-fatal: re-query on next run
            if result is None:
                search_failed.append(f"LC: KIC {kepid}: No Kepler long-cadence LCs found")
            else:
                search_results[kepid] = result
        except Exception as exc:
            # Do not cache failures — allow retry on next run
            search_failed.append(
                f"LC: KIC {kepid}: search failed: {type(exc).__name__}: {exc}"
            )

    for msg in search_failed:
        log_failure("failed_targets.log", msg)

    print(f"Phase 1c-a complete: {len(search_results):,} targets found, "
          f"{len(search_failed):,} not found / failed.")

    if not search_results:
        print("No LC targets to download.")
        return

    # ── Phase 1c-b: Parallel LC downloads ────────────────────────────────────
    print(f"\nPhase 1c-b: Downloading LC files ({n_workers} concurrent workers) ...")
    success = failed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_download_lc_with_retry, kepid, sr, tpf_dir): kepid
            for kepid, sr in search_results.items()
        }
        with tqdm(total=len(search_results), desc="LC downloads", unit="target") as pbar:
            for future in as_completed(futures):
                kepid, ok, err = future.result()
                with lock:
                    if ok:
                        success += 1
                    else:
                        failed += 1
                        log_failure("failed_targets.log", f"LC: KIC {kepid}: {err}")
                pbar.update(1)

    tqdm.write(f"Phase 1c complete: {success:,} LC targets downloaded, {failed:,} failed.")


# ============================================================================
# Preprocessing helpers: SG detrending, transit masking, local bin extraction
# ============================================================================

def _sg_detrend(flux: np.ndarray, transit_duration_cadences: float,
                poly_order: int = 3) -> np.ndarray:
    """
    Remove stellar variability and instrumental trends from a single-quarter
    flux array using a Savitzky-Golay filter.

    The filter estimates a smooth, slowly-varying baseline (stellar rotation,
    spot modulation, quarter-level instrumental drifts). Dividing by this
    baseline flattens the light curve while preserving sharp transit signals,
    because transits are much shorter than the filter window and are therefore
    not absorbed into the polynomial fit. SG filtering is applied even when
    PDC-SAP flux is available, because quarter-level residual trends survive
    PDC processing for magnetically active stars.

    Window length = max(31, next_odd(3 * transit_duration_cadences)),
    clamped to at most len(flux) // 4 (to prevent over-smoothing on short
    quarters). Window is always rounded to an odd integer.

    Args:
        flux: 1-D float array, one Kepler quarter, already median-normalised.
        transit_duration_cadences: transit duration in units of long cadences
            (29.4 min each). Derived as (koi_duration_hours / 24) / (29.4/1440).
        poly_order: SG polynomial order (default 3).

    Returns:
        flux_detrended: flux divided by the SG baseline. NaN-safe: NaNs in the
        input are linearly interpolated before filtering, then restored after
        division. If the detrended result contains non-finite values, or if the
        SG filter raises ValueError, returns the original flux unchanged with a
        logged warning (prefix SG_WARN:).
    """
    if len(flux) < max(31, poly_order + 2):
        # Quarter too short to detrend meaningfully — return unchanged
        return flux

    # Compute window length (always odd)
    raw_window = int(round(3.0 * transit_duration_cadences))
    # next_odd: return n if already odd, else n+1
    window = raw_window if raw_window % 2 == 1 else raw_window + 1
    window = max(31, window)
    # Clamp to at most len(flux)//4; bitwise OR 1 ensures result stays odd
    max_window = (len(flux) // 4) | 1
    window = min(window, max_window)

    # SG requirement: window must exceed poly_order
    if window <= poly_order:
        return flux

    # NaN-safe: interpolate gaps before filtering, restore NaN positions afterwards
    nan_pos = np.isnan(flux)
    flux_work = flux.copy()
    if nan_pos.any() and (~nan_pos).sum() >= 2:
        x = np.arange(len(flux_work))
        flux_work[nan_pos] = np.interp(x[nan_pos], x[~nan_pos], flux_work[~nan_pos])

    try:
        baseline = savgol_filter(flux_work, window_length=window, polyorder=poly_order)
    except ValueError as exc:
        log_failure(
            "failed_targets.log",
            f"SG_WARN: savgol_filter failed (window={window}, n={len(flux)}): "
            f"{type(exc).__name__}: {exc}",
        )
        return flux

    # Guard against near-zero baseline (prevents division blowup on flat/zeroed cadences)
    baseline = np.where(np.abs(baseline) < 1e-10, 1.0, baseline)
    detrended = flux_work / baseline

    # Restore original NaN positions
    detrended[nan_pos] = np.nan

    # Sanity check: non-finite output means SG produced a bad baseline
    if not np.all(np.isfinite(detrended[~nan_pos])):
        log_failure(
            "failed_targets.log",
            f"SG_WARN: non-finite values after SG detrend (window={window}, "
            f"n={len(flux)}); using original quarter unchanged",
        )
        return flux

    return detrended


def _compute_transit_mask(time_arr: np.ndarray,
                          other_koi_rows: pd.DataFrame) -> np.ndarray:
    """
    Compute a boolean mask (True = cadence is free of other planets' transits)
    over time_arr, given a DataFrame of other KOIs sharing this star.

    Multi-planet masking ensures that when phase-folding on planet X's period,
    the OOT baseline and centroid OOT statistics are not contaminated by transits
    of planets Y, Z, ... on the same star.

    For each other KOI, all transit mid-times within [time_arr.min(), time_arr.max()]
    are computed from koi_period and koi_time0bk (BKJD — used directly; do NOT
    subtract 2457000, that is the TESS BTJD offset):
        t_transit = koi_time0bk + n * koi_period  for integer n

    Any cadence within ±1 × (koi_duration / 24) days of a transit mid-time is
    masked (set False). Using one full transit-duration as the guard band on each
    side protects the OOT baseline from ingress/egress wing contamination.

    Cadences already NaN in time_arr are set False regardless.

    Args:
        time_arr:       BKJD timestamps for this target's full time series.
        other_koi_rows: rows from koi_manifest for OTHER KOIs on the same star.

    Returns:
        clean_mask: bool array, same length as time_arr.
            True  = cadence is not contaminated by any other planet's transit.
            False = cadence falls within ±1×dur of another planet's transit, or is NaN.
    """
    # NaN timestamps are always excluded from the clean set
    clean_mask = ~np.isnan(time_arr)

    if len(other_koi_rows) == 0:
        return clean_mask

    t_min = float(np.nanmin(time_arr))
    t_max = float(np.nanmax(time_arr))

    for _, koi_row in other_koi_rows.iterrows():
        try:
            period = float(koi_row["koi_period"])
            t0 = float(koi_row["koi_time0bk"])   # BKJD — use directly
            dur_h = float(koi_row["koi_duration"])
        except (KeyError, TypeError, ValueError):
            continue  # skip rows with missing/uncastable values

        if not (np.isfinite(period) and period > 0
                and np.isfinite(t0) and np.isfinite(dur_h) and dur_h > 0):
            continue

        # Guard band = ±1 × transit_duration each side (in days)
        half_mask = dur_h / 24.0

        # Enumerate integer transit indices spanning the observation window
        n_start = int(np.ceil((t_min - t0) / period))
        n_end = int(np.floor((t_max - t0) / period))

        for n in range(n_start, n_end + 1):
            t_transit = t0 + n * period
            # Mask cadences within ±half_mask of this transit mid-time
            clean_mask &= ~(np.abs(time_arr - t_transit) <= half_mask)

    return clean_mask


def _extract_local_bins(flux_binned: np.ndarray,
                        bin_centres: np.ndarray,
                        dur_days: float,
                        period: float,
                        local_bins: int = 61) -> np.ndarray:
    """
    Extract and resample an adaptive transit-centred local view from the global
    phase-folded flux array.

    Window in phase space: [-2*(dur_days/period), +2*(dur_days/period)],
    clamped to [-0.5, 0.5]. This ensures the transit always fills the local view
    proportionally regardless of period or duration, matching AstroNet's approach
    of sizing the local view to 2× the transit duration on each side.

    The global phase-folded array (NUM_BINS bins) is resampled onto a uniform
    grid of `local_bins` points within the window using np.interp (linear
    interpolation). This gives a fixed-size (local_bins,) array regardless of the
    target's orbital parameters.

    Global bins (NUM_BINS=301) feed the global CNN branch. Local bins (61) are an
    adaptive per-target resample centred on the transit, stored pre-computed in
    l_0..l_60 CSV columns to avoid runtime re-interpolation during training.

    Args:
        flux_binned:  (NUM_BINS,) phase-folded, normalised flux.
        bin_centres:  (NUM_BINS,) phase values corresponding to flux_binned.
        dur_days:     transit duration in days (koi_duration_hours / 24).
        period:       orbital period in days.
        local_bins:   number of output bins (default 61, matching AstroNet local).

    Returns:
        local_flux: (local_bins,) float array of the resampled local view.
    """
    half_window = min(0.5, 2.0 * dur_days / period)
    local_phase = np.linspace(-half_window, half_window, local_bins)
    local_flux = np.interp(local_phase, bin_centres, flux_binned)
    return local_flux


# ============================================================================
# Phase 2: Build degraded cache
# ============================================================================

def build_cache(
    manifest: pd.DataFrame,
    tpf_dir: str,
    cache_dir: str,
    k_values: list[int],
    max_quarters: int | None,
    results_dir: str = "results_resolution",
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

    Stale-cache detection: if existing flux cache files were built with the old
    NUM_BINS (1000-bin schema), or existing centroid cache files predate the
    aperture-degeneracy gate (missing the 'quality_degenerate' key), this function
    emits a clear error and exits. The user must manually delete cache/flux/ and/or
    cache/centroids/ before re-running.
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

    # ── Stale-cache guard ────────────────────────────────────────────────────
    # Detect if existing flux cache was built with a different NUM_BINS.
    # A schema mismatch causes silently wrong tensor shapes in theModel.py.
    npz_candidates = [f for f in os.listdir(flux_dir) if f.endswith(".npz")]
    if npz_candidates:
        try:
            probe = np.load(os.path.join(flux_dir, npz_candidates[0]))
            stored_bins = probe["flux_binned"].shape[0]
            if stored_bins != NUM_BINS:
                print(
                    f"\nWARNING: Stale flux cache detected "
                    f"(built with {stored_bins} bins; current NUM_BINS={NUM_BINS}).\n"
                    f"Delete cache/flux/ and cache/centroids/ before proceeding, "
                    f"then re-run --phase cache."
                )
                sys.exit(1)
        except Exception:
            pass  # Unreadable probe: proceed; corrupt cache will surface later

    # ── Stale centroid-cache guard ───────────────────────────────────────────
    # Detect centroid cache files built before the aperture-degeneracy gate
    # (superpixel_rebin previously had no size check at all, so any existing
    # cache/centroids/*.npz predating this fix is known-corrupted at high k —
    # e.g. centroid_scaling.csv showing median_uncertainty/median_rms == 0.0
    # exactly at k=4/k=5). Detect via the presence of the new 'quality_degenerate'
    # key, which every post-fix write includes.
    centroid_npz_candidates = [f for f in os.listdir(centroid_dir) if f.endswith(".npz")]
    if centroid_npz_candidates:
        try:
            probe = np.load(os.path.join(centroid_dir, centroid_npz_candidates[0]))
            if "quality_degenerate" not in probe.files:
                print(
                    "\nERROR: Stale centroid cache detected — missing 'quality_degenerate' key.\n"
                    "Existing cache/centroids/*.npz files were built before the aperture-degeneracy "
                    "fix and are known-corrupted at high k (see centroid_scaling.csv medians of "
                    "exactly 0.0 at k=4/k=5).\n"
                    "Action required: delete cache/centroids/ (NOT cache/flux/, which is unaffected "
                    "by this bug) and re-run '--phase cache'."
                )
                sys.exit(1)
        except Exception as exc:
            print(f"WARNING: could not probe centroid cache ({type(exc).__name__}: {exc}); "
                  f"proceeding — corrupt cache will surface later.")

    os.makedirs(results_dir, exist_ok=True)
    aperture_diag_path = os.path.join(results_dir, "aperture_diagnostics.csv")
    if os.path.exists(aperture_diag_path):
        os.remove(aperture_diag_path)

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
    build_stats: dict = {}
    for kepid in tqdm(unique_kepids, desc="Building cache", unit="target"):
        rows = koi_by_kepid.get_group(kepid)
        try:
            tpf_list = load_local_tpfs(kepid, tpf_dir, max_quarters)
            if not tpf_list:
                continue

            # Flux cache: one file per KOI, using that KOI's own ephemeris.
            # A star with two planets gets two separate folded flux series.
            # koi_rows (= rows) is passed for multi-planet masking.
            for _, row in rows.iterrows():
                flux_path = os.path.join(flux_dir, f"{row['kepoi_name']}.npz")
                if not os.path.exists(flux_path):
                    _compute_flux_cache(row, tpf_list, flux_path, tpf_dir, rows)

            # Centroid cache: one file per (kepid, k, psf). Centroids are per-star.
            for k in k_values:
                for psf in [0, 1]:
                    out_path = os.path.join(centroid_dir,
                                            f"{kepid}_k{k}_psf{psf}.npz")
                    stats = build_stats.setdefault(
                        (k, psf), {"n_targets": 0, "n_hard_excluded": 0, "n_quality_degenerate": 0}
                    )
                    if os.path.exists(out_path):
                        continue
                    status = _compute_centroid_cache(
                        kepid, tpf_list, rows, k, psf, out_path, results_dir,
                        load_cube, superpixel_rebin, psf_broaden,
                        moment_centroid, difference_image_centroid,
                        compute_centroid_quality,
                    )
                    stats["n_targets"] += 1
                    if not status["written"]:
                        stats["n_hard_excluded"] += 1
                    elif status["quality_degenerate"]:
                        stats["n_quality_degenerate"] += 1

        except Exception as exc:
            log_failure("failed_targets.log",
                        f"KIC {kepid}: cache build: {type(exc).__name__}: {exc}")

    if build_stats:
        print("\nAperture-degeneracy exclusion summary (per k, psf):")
        print(f"  {'k':>4} {'psf':>4} {'n_targets':>10} {'hard_excl':>16} {'quality_degenerate':>20}")
        for (k, psf), s in sorted(build_stats.items()):
            n = s["n_targets"]
            if n == 0:
                continue
            hard_str = f"{s['n_hard_excluded']} ({s['n_hard_excluded']/n:.1%})"
            deg_str = f"{s['n_quality_degenerate']} ({s['n_quality_degenerate']/n:.1%})"
            print(f"  {k:>4} {psf:>4} {n:>10} {hard_str:>16} {deg_str:>20}")

    print("Phase 2 complete.")


def _load_tpfs(kepid: int, tpf_dir: str, max_quarters: int | None) -> list:
    """
    Load cached TPFs for a kepid from tpf_dir.

    Delegates to load_local_tpfs — no MAST re-query. This function is kept
    for backward compatibility with any callers; load_local_tpfs is the
    canonical exported entry point.
    """
    return load_local_tpfs(kepid, tpf_dir, max_quarters)


def _compute_flux_cache(row: pd.Series, tpf_list: list, out_path: str,
                        tpf_dir: str, koi_rows: pd.DataFrame) -> None:
    """
    Compute native-resolution phase-folded, binned flux for one KOI.

    Uses this KOI's own ephemeris (period, t0, duration) to fold the light curve.
    Two KOIs on the same star have different transit times and periods, so each
    needs its own phase-folded representation.

    PDC-SAP flux is used when available (from LC FITS files) in preference to SAP
    flux from TPFs, because PDC removes instrumental systematics (thermal drifts,
    attitude tweaks, focus changes) that would otherwise alias into the phase-folded
    light curve as structured noise.

    Fallback chain:
      1. PDC-SAP from local LC FITS files (kplr*_llc.fits*) — lc.flux.value is
         PDC-SAP by default for Kepler LC files loaded via lightkurve.
      2. If pdcsap_flux is all-NaN for a quarter → sap_flux from that LC file.
      3. If no LC files found → SAP from tpf.to_lightcurve("pipeline").

    Savitzky-Golay detrending (per quarter, after median normalisation) removes
    stellar variability and quarter-level instrumental trends that survive PDC
    processing for magnetically active stars.

    Multi-planet masking is applied after concatenating all quarters (before
    phase-folding) to exclude cadences contaminated by other KOIs' transits.
    This prevents OOT-baseline contamination for multi-planet systems.

    Writes cache/flux/{kepoi_name}.npz with arrays: flux_binned, bin_centres.

    koi_time0bk is in BKJD (BJD − 2454833) — use directly, no offset subtraction.
    Do NOT subtract 2457000 (that is the TESS BTJD offset).

    Args:
        row:      single KOI row from the manifest (period, t0, duration, kepoi_name)
        tpf_list: list of TPF objects for the parent KIC star
        out_path: destination path for the .npz output
        tpf_dir:  root TPF/LC directory, passed to load_local_lcs
        koi_rows: all KOI rows for this star (used for multi-planet masking)
    """
    kepid = int(row["kepid"])
    period = float(row["koi_period"])
    # koi_time0bk is BKJD — use directly; do NOT subtract 2457000 (TESS offset)
    t0_bkjd = float(row["koi_time0bk"])
    dur_days = float(row["koi_duration"]) / 24.0
    # Transit duration in Kepler long-cadence cadences (29.4 min per cadence)
    transit_dur_cadences = (dur_days * 1440.0) / 29.4

    time_parts, flux_parts = [], []

    # ── Primary source: PDC-SAP from local LC files ──────────────────────────
    lc_list = load_local_lcs(kepid, tpf_dir, None)  # all quarters; no cap on LC

    if lc_list:
        for lc in lc_list:
            try:
                t = lc.time.bkjd
                # PDC-SAP is the default flux column for Kepler LC files in lightkurve
                f = np.array(lc.flux.value, dtype=np.float64)

                if np.all(np.isnan(f)):
                    # Fall back to SAP flux from the same LC file for this quarter
                    log_failure(
                        "failed_targets.log",
                        f"KIC {kepid} {row['kepoi_name']}: pdcsap_flux all-NaN "
                        f"— falling back to sap_flux",
                    )
                    try:
                        f = np.array(lc["sap_flux"].value, dtype=np.float64)
                    except Exception as sap_exc:
                        log_failure(
                            "failed_targets.log",
                            f"KIC {kepid} {row['kepoi_name']}: sap_flux fallback "
                            f"also failed: {type(sap_exc).__name__}: {sap_exc}",
                        )
                        continue

                # Per-quarter median normalisation
                med = np.nanmedian(f)
                if med == 0 or not np.isfinite(med):
                    continue
                f = f / med

                # Savitzky-Golay detrending: removes stellar variability and
                # quarter-level trends even when PDC flux is available
                f = _sg_detrend(f, transit_dur_cadences)

                # Re-normalise to median 1.0 after SG detrending (safeguard against
                # residual baseline offset from SG division on short quarters)
                med2 = np.nanmedian(f)
                if np.isfinite(med2) and med2 != 0:
                    f = f / med2

                time_parts.append(t)
                flux_parts.append(f)

            except Exception as exc:
                log_failure(
                    "failed_targets.log",
                    f"KIC {kepid} {row['kepoi_name']}: LC flux extraction: "
                    f"{type(exc).__name__}: {exc}",
                )
                continue

    else:
        # ── Fallback: SAP from TPF (no LC files available) ──────────────────
        log_failure(
            "failed_targets.log",
            f"KIC {kepid}: no LC files — using SAP fallback from TPF",
        )
        for tpf in tpf_list:
            try:
                lc = tpf.to_lightcurve(aperture_mask="pipeline")
                t = lc.time.bkjd
                f = np.array(lc.flux.value, dtype=np.float64)

                med = np.nanmedian(f)
                if med == 0 or not np.isfinite(med):
                    continue
                f = f / med

                # SG detrending still applied for SAP fallback
                f = _sg_detrend(f, transit_dur_cadences)

                med2 = np.nanmedian(f)
                if np.isfinite(med2) and med2 != 0:
                    f = f / med2

                time_parts.append(t)
                flux_parts.append(f)

            except Exception as exc:
                log_failure(
                    "failed_targets.log",
                    f"KIC {kepid} {row['kepoi_name']}: SAP flux extraction: "
                    f"{type(exc).__name__}: {exc}",
                )
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

    # Multi-planet masking: exclude cadences contaminated by other KOIs' transits.
    # Applied before phase-folding so the OOT baseline is not contaminated by
    # other planets' transits contributing spurious depth to the folded flux.
    other_kois = koi_rows[koi_rows["kepoi_name"] != row["kepoi_name"]]
    if len(other_kois) > 0:
        mp_mask = _compute_transit_mask(time_arr, other_kois)
        time_arr = time_arr[mp_mask]
        flux_arr = flux_arr[mp_mask]

    # Phase fold — t0_bkjd already in BKJD
    phase = ((time_arr - t0_bkjd) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    half_window = 1.5 * (dur_days / period)
    oot_mask = np.abs(phase) > half_window
    oot_flux = flux_arr[oot_mask]

    # Require at least 50 OOT cadences after multi-planet masking; fewer cadences
    # means the OOT baseline would be unreliable for normalisation
    if oot_mask.sum() < 50:
        log_failure(
            "failed_targets.log",
            f"MASK: KIC {kepid} {row['kepoi_name']}: insufficient OOT cadences "
            f"({oot_mask.sum()}) after multi-planet mask — skipped",
        )
        return

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
    out_path, results_dir,
    load_cube, superpixel_rebin, psf_broaden,
    moment_centroid, difference_image_centroid,
    compute_centroid_quality,
) -> dict:
    """
    Compute centroid time series and quality metrics at scale k for this kepid.
    Writes cache/centroids/{kepid}_k{k}_psf{psf}.npz.

    The difference-image centroid uses koi_rows.iloc[0]'s ephemeris as a proxy for
    the star's transit timing. For multi-KOI systems this is an acknowledged
    approximation — centroids are per-star, not per-planet.

    Multi-planet masking is applied after concatenating the per-quarter time/m1/m2
    arrays (before phase-folding) to exclude cadences contaminated by other KOIs'
    transits. This ensures OOT centroid statistics are not contaminated by transits
    of other planets on the same star.

    koi_time0bk is in BKJD — use directly; do NOT subtract 2457000 (TESS offset).

    Two-tier aperture-degeneracy handling (see superpixel_rebin in degrade.py):
      - hard-degenerate quarter (grid collapsed to empty in a dimension): the
        quarter contributes nothing at all — no array exists to build a centroid
        from. If every quarter for this target is hard-degenerate, no cache file
        is written (matches how a missing cache file already excludes a kepid
        from write_training_csvs).
      - soft-degenerate quarter (grid/aperture exists but is too small to trust):
        m1/m2 ARE concatenated (a real, if flat/uninformative, signal — legitimate
        input for the CNN), but difference_image_centroid is NOT called for that
        quarter, since its offset/uncertainty would come from a near-zero-variance
        grid and would corrupt the aggregate Bryson/quality statistics.
      - quality_degenerate (whole-target flag, written to the npz) is True if ANY
        contributing quarter was soft-degenerate, because compute_centroid_quality's
        centroid_rms is computed over the ENTIRE concatenated m1/m2 array across all
        quarters, not per-quarter — a single soft-degenerate quarter's near-zero
        samples would contaminate that whole-target statistic even if other quarters
        were clean. This is a conservative choice: it may occasionally discard an
        otherwise-good target's Bryson/quality data because of one bad quarter, but
        never lets a contaminated statistic through as trusted.

    Returns:
        dict with keys "written" (bool, whether a cache file was written) and
        "quality_degenerate" (bool | None, None if no file was written).
    """
    row = koi_rows.iloc[0]
    period = float(row["koi_period"])
    t0_bkjd = float(row["koi_time0bk"])  # BKJD — use directly
    dur_days = float(row["koi_duration"]) / 24.0

    m1_parts, m2_parts, time_parts = [], [], []
    diff_offsets, diff_uncertainties = [], []
    any_quarter_soft = False
    aperture_diag_path = os.path.join(results_dir, "aperture_diagnostics.csv")

    for q_idx, tpf in enumerate(tpf_list):
        try:
            cube, time, quality, wcs, aperture = load_cube(tpf)
            if cube is None or len(time) == 0:
                continue

            # PSF broadening (optional; k=1 is identity since sqrt(1-1)=0)
            if psf == 1 and k > 1:
                cube = np.stack([psf_broaden(frame, k) for frame in cube])

            ny_native, nx_native = cube.shape[1], cube.shape[2]
            aper_native = int(aperture.sum())

            coarse_cube, coarse_aperture, degeneracy = superpixel_rebin(cube, aperture, k)

            _log_aperture_diagnostic(aperture_diag_path, {
                "kepid": kepid, "k": k, "psf": psf,
                "quarter_index": q_idx, "quarter": getattr(tpf, "quarter", "?"),
                "ny_native": ny_native, "nx_native": nx_native,
                "aperture_superpixels_native": aper_native,
                "ny_out": degeneracy["ny_out"], "nx_out": degeneracy["nx_out"],
                "aperture_superpixels_coarse": degeneracy["aperture_superpixels"],
                "degeneracy_tier": (
                    "hard" if degeneracy["hard_degenerate"]
                    else "soft" if degeneracy["soft_degenerate"]
                    else "none"
                ),
            })

            if degeneracy["hard_degenerate"]:
                log_failure(
                    "degenerate_aperture.log",
                    f"KIC {kepid} k={k} psf={psf} quarter={getattr(tpf, 'quarter', '?')}: "
                    f"HARD degenerate ({degeneracy['ny_out']}x{degeneracy['nx_out']}) "
                    f"— quarter excluded entirely"
                )
                continue

            quarter_soft = degeneracy["soft_degenerate"]
            if quarter_soft:
                any_quarter_soft = True
                log_failure(
                    "degenerate_aperture.log",
                    f"KIC {kepid} k={k} psf={psf} quarter={getattr(tpf, 'quarter', '?')}: "
                    f"SOFT degenerate (grid {degeneracy['ny_out']}x{degeneracy['nx_out']}, "
                    f"aperture_superpixels={degeneracy['aperture_superpixels']}) "
                    f"— quarter's diff-image stats excluded, m1/m2 retained"
                )

            # Moment centroid time series — real (if flat) signal even when soft-degenerate
            m1, m2 = moment_centroid(coarse_cube, coarse_aperture)

            if not quarter_soft:
                # Difference-image centroid offset — skipped for soft-degenerate quarters,
                # since a near-zero-variance grid would produce a misleadingly precise
                # offset/uncertainty that contaminates the aggregate Bryson statistic.
                diff_result = difference_image_centroid(
                    coarse_cube, time, period, t0_bkjd, dur_days, k=k
                )
                if diff_result.get("offset_arcsec") is not None:
                    diff_offsets.append(diff_result["offset_arcsec"])
                if diff_result.get("uncertainty_arcsec") is not None:
                    diff_uncertainties.append(diff_result["uncertainty_arcsec"])

            time_parts.append(time)
            m1_parts.append(m1)
            m2_parts.append(m2)

        except Exception as exc:
            log_failure("failed_targets.log",
                        f"KIC {kepid} k={k} psf={psf}: {type(exc).__name__}: {exc}")
            continue

    if not time_parts:
        return {"written": False, "quality_degenerate": None}

    time_arr = np.concatenate(time_parts)
    m1_arr = np.concatenate(m1_parts)
    m2_arr = np.concatenate(m2_parts)

    idx = np.argsort(time_arr)
    time_arr = time_arr[idx]
    m1_arr = m1_arr[idx]
    m2_arr = m2_arr[idx]

    # Multi-planet masking: exclude cadences contaminated by other KOIs' transits.
    # The centroid cache uses koi_rows.iloc[0] as the reference KOI; all other
    # KOIs on this star are treated as contaminants.
    other_kois = koi_rows[koi_rows["kepoi_name"] != row["kepoi_name"]]
    if len(other_kois) > 0:
        mp_mask = _compute_transit_mask(time_arr, other_kois)
        time_arr = time_arr[mp_mask]
        m1_arr = m1_arr[mp_mask]
        m2_arr = m2_arr[mp_mask]

    # Phase fold — t0_bkjd is BKJD
    phase = ((time_arr - t0_bkjd) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    half_window = 1.5 * (dur_days / period)
    oot_mask = np.abs(phase) > half_window

    # Require at least 50 OOT cadences for reliable centroid statistics
    if oot_mask.sum() < 50:
        log_failure(
            "failed_targets.log",
            f"MASK: KIC {kepid} k={k} psf={psf}: insufficient OOT cadences "
            f"({oot_mask.sum()}) after multi-planet mask — skipped",
        )
        return {"written": False, "quality_degenerate": None}

    # Compute quality metrics
    offset_median = float(np.median(diff_offsets)) if diff_offsets else np.nan
    uncertainty_median = float(np.median(diff_uncertainties)) if diff_uncertainties else np.nan
    quality_metrics = compute_centroid_quality(
        m1_arr, m2_arr, offset_median, uncertainty_median, oot_mask, k=k
    )

    quality_degenerate = any_quarter_soft
    if quality_degenerate:
        # Belt: force NaN at write time so any downstream code path that forgets to
        # check the quality_degenerate flag still sees NaN, not a misleadingly-precise
        # number (e.g. an exact 0.0 that would otherwise pass an isfinite() check).
        quality_metrics = {key: np.float32(np.nan) for key in quality_metrics}

    np.savez_compressed(
        out_path,
        time=time_arr,
        phase=phase,
        m1=m1_arr,
        m2=m2_arr,
        oot_mask=oot_mask,
        diff_offset_arcsec=np.array(diff_offsets) if diff_offsets else np.array([np.nan]),
        diff_uncertainty_arcsec=np.array(diff_uncertainties) if diff_uncertainties else np.array([np.nan]),
        quality_degenerate=np.bool_(quality_degenerate),
        **quality_metrics,
    )
    return {"written": True, "quality_degenerate": quality_degenerate}


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

    Output schema (971 columns per row):
      7 metadata: kepid, kepoi_name, label, fp_subtype, koi_period, koi_time0bk,
                  koi_duration
      f_0..f_300   (301 global flux bins — OOT-normalised, OOT-standardised)
      m1_0..m1_300 (301 global RA centroid bins at scale k; OOT-baseline removed)
      m2_0..m2_300 (301 global Dec centroid bins; same normalisation)
      l_0..l_60    (61 adaptive local flux bins, pre-computed via _extract_local_bins;
                   window = ±2×transit_duration in phase, resampled onto 61 points)

    Total feature columns: 301×3 + 61 = 964. Total columns: 964 + 7 = 971.

    Global bins (301) feed the global CNN branch. Local bins (61) are a per-target
    adaptive resample centred on the transit, stored pre-computed to avoid runtime
    re-interpolation during theModel.py training.

    Returns:
        csv_counts dict[(k, psf)] → int rows written, for the attrition report.
    """
    flux_dir = os.path.join(cache_dir, "flux")
    centroid_dir = os.path.join(cache_dir, "centroids")

    if not os.path.isdir(centroid_dir):
        print(
            f"\nERROR: centroid cache directory not found: {centroid_dir}\n"
            f"Run '--phase cache --tpf-dir <path>' first to build it.\n"
            f"If your cache is on an external volume, pass '--cache-dir <path>' too."
        )
        return {}
    if not os.path.isdir(flux_dir):
        print(
            f"\nERROR: flux cache directory not found: {flux_dir}\n"
            f"Run '--phase cache --tpf-dir <path>' first to build it."
        )
        return {}

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
                    log_failure(
                        "failed_targets.log",
                        f"KIC {kepid} k={k} psf={psf}: centroid cache missing — "
                        f"run '--phase cache' to build it",
                    )
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
                        log_failure(
                            "failed_targets.log",
                            f"KIC {kepid} {kepoi_name} k={k} psf={psf}: "
                            f"flux cache missing — run '--phase cache' to build it",
                        )
                        skipped += 1
                        continue

                    try:
                        flux_data = np.load(flux_path)
                        flux_binned = flux_data["flux_binned"]
                        bin_centres = flux_data["bin_centres"]

                        # Adaptive local view: per-target 61-bin resample centred on
                        # the transit (±2×transit_duration in phase). Pre-computed here
                        # and stored as l_0..l_60 to avoid re-interpolation in theModel.py.
                        dur_days = float(row["koi_duration"]) / 24.0
                        local_binned = _extract_local_bins(
                            flux_binned, bin_centres,
                            dur_days=dur_days,
                            period=float(row["koi_period"]),
                        )

                        entry = {
                            "kepid": kepid,
                            "kepoi_name": kepoi_name,
                            "label": int(row["label"]),
                            "fp_subtype": row.get("fp_subtype", ""),
                            "koi_period": row["koi_period"],
                            "koi_time0bk": row["koi_time0bk"],
                            "koi_duration": row["koi_duration"],
                        }
                        # Global flux bins: f_0..f_{NUM_BINS-1}
                        for i in range(NUM_BINS):
                            entry[f"f_{i}"] = float(flux_binned[i])
                        # Global RA centroid bins
                        for i in range(NUM_BINS):
                            entry[f"m1_{i}"] = float(m1_binned[i])
                        # Global Dec centroid bins
                        for i in range(NUM_BINS):
                            entry[f"m2_{i}"] = float(m2_binned[i])
                        # Adaptive local flux bins: l_0..l_{LOCAL_BINS-1}
                        for i in range(LOCAL_BINS):
                            entry[f"l_{i}"] = float(local_binned[i])

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
      - failed_targets.log: zero-local-TPF skips, LC failures, masking skips
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
        "Degenerate-aperture exclusions per (k, psf):",
    ]
    aperture_diag_path = os.path.join(results_dir, "aperture_diagnostics.csv")
    if os.path.exists(aperture_diag_path):
        diag = pd.read_csv(aperture_diag_path)
        for k in k_values:
            for psf in [0, 1]:
                sub = diag[(diag["k"] == k) & (diag["psf"] == psf)]
                if sub.empty:
                    continue
                n_hard = int((sub["degeneracy_tier"] == "hard").sum())
                n_soft = int((sub["degeneracy_tier"] == "soft").sum())
                n_q = len(sub)
                lines.append(
                    f"  k={k} psf={psf}: {n_hard:>6,}/{n_q:,} quarters hard-excluded, "
                    f"{n_soft:>6,}/{n_q:,} quarters soft-flagged (quality metrics only)"
                )
        lines.append("  Targets losing ALL centroid data (no cache file written) per (k, psf):")
        centroid_dir_ = os.path.join(cache_dir, "centroids")
        n_kic_total = manifest["kepid"].nunique()
        for k in k_values:
            for psf in [0, 1]:
                n_missing = sum(
                    1 for kepid in manifest["kepid"].unique()
                    if not os.path.exists(os.path.join(centroid_dir_, f"{kepid}_k{k}_psf{psf}.npz"))
                )
                lines.append(f"    k={k} psf={psf}: {n_missing:,} / {n_kic_total:,} targets")
    else:
        lines.append("  (aperture_diagnostics.csv not found — run --phase cache to generate)")

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
        description="Load local Kepler TPFs/LCs and build per-k training CSVs"
    )
    parser.add_argument(
        "--max-quarters", type=int, default=None,
        help="Cap the number of TPF quarters per target (default: all available). "
             "LC downloads always use all available quarters regardless of this flag."
    )
    parser.add_argument(
        "--phase",
        choices=["download", "cache", "csv-only", "all"],
        default="all",
        help=(
            "Pipeline phase to run: "
            "'download' = TPF + LC download only; "
            "'cache' = build degraded cache from existing local files; "
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
            "Directory for Kepler TPF and LC FITS files "
            "(default: ./tpf_temp, which is gitignored). "
            "Scanned recursively — a path like "
            "'/Volumes/Stuff/Research Work/TPFs' works directly. "
            "LC files are downloaded into the same directory tree as TPFs."
        ),
    )
    parser.add_argument(
        "--cache-dir", default="./cache",
        help="Directory for computed intermediate cache files (default: ./cache)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help=(
            "Number of concurrent download threads (default: 8). "
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
        for log in ["failed_targets.log", "sparse_targets.log", "degenerate_aperture.log"]:
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
        # Phase 1c: also download LC files (PDC-SAP source; all quarters)
        download_lcs(
            manifest, tpf_dir, cache_dir,
            n_workers=args.workers,
            search_rate=args.search_rate,
        )

    if run_cache:
        build_cache(manifest, tpf_dir, cache_dir, args.k_values, args.max_quarters, args.results_dir)

    csv_counts = {}
    if run_csv:
        csv_counts = write_training_csvs(manifest, cache_dir, args.k_values)
        write_data_attrition_report(
            manifest, cache_dir, args.k_values, csv_counts, args.results_dir
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
