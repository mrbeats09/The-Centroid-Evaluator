"""
run_pipeline.py — End-to-End Pipeline Orchestrator

Runs every pipeline stage in order. Each stage is gated on the previous one:
if a step exits non-zero the pipeline aborts immediately and reports which step
failed, so partial results are never silently passed downstream.

Stages (in order):
  1. getExamples.py       — build KOI manifest
  2. getInputData.py      — download TPFs and build degraded cache
  3. getInputData.py      — write training CSVs from cache (csv-only phase)
  4. centroid_quality.py  — aggregate centroid quality metrics
  5. theModel.py          — train CNN at each k, produce per-target scores
  6. stats_ml.py          — bootstrap AUC CIs + breakdown resolution
  7. baselines.py         — Bryson baseline (+ optional vetting)
  8. compare.py           — assemble comparison.csv, report.md, figures

Usage:
  python run_pipeline.py [options]

Options mirror the underlying scripts; run python run_pipeline.py --help.
"""

import argparse
import os
import subprocess
import sys
import time

PYTHON = sys.executable  # same interpreter that launched this script


# ============================================================================
# Step runner
# ============================================================================

def run_step(
    step_num: int,
    label: str,
    cmd: list[str],
    skip_if_exists: str | None = None,
) -> None:
    """
    Run one pipeline step as a subprocess.

    Args:
        step_num:        1-based step index (for display)
        label:           human-readable step name
        cmd:             command + arguments list
        skip_if_exists:  if this path already exists, skip the step and print a note
    """
    header = f"[{step_num}/8] {label}"
    separator = "─" * 60

    if skip_if_exists and os.path.exists(skip_if_exists):
        print(f"\n{separator}")
        print(f"{header}")
        print(f"  ↳ SKIP — output already exists: {skip_if_exists}")
        return

    print(f"\n{separator}")
    print(f"{header}")
    print(f"  $ {' '.join(cmd)}")
    print(separator)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n{'═' * 60}")
        print(f"PIPELINE ABORTED at step {step_num}: {label}")
        print(f"Exit code: {exc.returncode}  (elapsed: {elapsed:.1f}s)")
        print(f"Fix the error above and re-run. Steps 1–{step_num - 1} are cached.")
        print(f"{'═' * 60}")
        sys.exit(exc.returncode)

    elapsed = time.perf_counter() - t0
    print(f"  ✓ done in {elapsed:.1f}s")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the full Kepler centroid-degradation pipeline end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── getExamples.py ──────────────────────────────────────────────────────
    parser.add_argument(
        "--fp-subset", choices=["all", "centroid_offset"], default="all",
        help="FP subset: 'all' = all FALSE POSITIVE KOIs; 'centroid_offset' = koi_fpflag_co=1 only",
    )
    parser.add_argument(
        "--manifest", default="koi_manifest.csv",
        help="Path to KOI manifest CSV",
    )

    # ── getInputData.py ─────────────────────────────────────────────────────
    parser.add_argument(
        "--tpf-dir", default="./tpf_temp",
        help=(
            "Directory for Kepler TPF FITS files. The default (./tpf_temp) is "
            "gitignored. If you use a custom path, add it to .gitignore manually "
            "— TPF archives can be tens of GB."
        ),
    )
    parser.add_argument(
        "--cache-dir", default="./cache",
        help="Directory for intermediate cache files (flux, centroids, cubes)",
    )
    parser.add_argument(
        "--max-quarters", type=int, default=None,
        help="Cap the number of Kepler quarters downloaded per target",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Concurrent S3 download threads for Stage 2 (default: 8)",
    )
    parser.add_argument(
        "--search-rate", type=float, default=3.0,
        help="MAST search API rate limit in requests/sec (default: 3.0; max safe ~5.0)",
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Pixel-scale factors k to process",
    )

    # ── stats_ml.py ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--n-bootstrap", type=int, default=2000,
        help="Bootstrap iterations for AUC confidence intervals",
    )
    parser.add_argument(
        "--snr-threshold", type=float, default=3.0,
        help="Centroid SNR threshold for k_break_snr breakdown metric",
    )
    parser.add_argument(
        "--enable-delong", action="store_true",
        help="Compute DeLong pairwise AUC significance tests (default: off)",
    )

    # ── baselines.py ────────────────────────────────────────────────────────
    parser.add_argument(
        "--enable-vetting", action="store_true",
        help="Run the vetting.centroid_test baseline (requires: pip install vetting; default: off)",
    )

    # ── flow control ────────────────────────────────────────────────────────
    parser.add_argument(
        "--start-from", type=int, default=1, metavar="STEP",
        help=(
            "Skip steps before this number (1–8). Useful for resuming after a failure. "
            "Caching means earlier outputs are still on disk."
        ),
    )
    parser.add_argument(
        "--results-dir", default="results_resolution",
        help="Directory where per-level metrics and figures are written",
    )

    args = parser.parse_args()

    k_str = [str(k) for k in args.k_values]

    # Shared flags forwarded to multiple scripts
    k_flags    = ["--k-values"] + k_str
    manifest_f = ["--manifest", args.manifest]

    # ── build step definitions ───────────────────────────────────────────────

    steps = [
        # Step 1 — KOI manifest
        (
            1, "Build KOI manifest (getExamples.py)",
            [PYTHON, "getExamples.py", "--fp-subset", args.fp_subset] + manifest_f,
            args.manifest,           # skip if manifest already exists
        ),

        # Step 2 — TPF download + cache build
        (
            2, "Download TPFs and build degraded cache (getInputData.py)",
            [PYTHON, "getInputData.py",
             "--phase", "all",
             "--tpf-dir", args.tpf_dir,
             "--cache-dir", args.cache_dir,
             "--workers", str(args.workers),
             "--search-rate", str(args.search_rate),
             *(["--max-quarters", str(args.max_quarters)] if args.max_quarters else []),
             ] + k_flags + manifest_f,
            None,   # always run; internal caching handles skip-if-present per file
        ),

        # Step 3 — Training CSVs from cache (fast; no FITS access)
        (
            3, "Write training CSVs from cache (getInputData.py --phase csv-only)",
            [PYTHON, "getInputData.py",
             "--phase", "csv-only",
             "--cache-dir", args.cache_dir,
             ] + k_flags + manifest_f,
            # Skip if at least one CSV already exists (proxy for completion)
            f"kepler_training_data_k{args.k_values[0]}_psf0.csv",
        ),

        # Step 4 — Centroid quality analysis
        (
            4, "Centroid quality analysis (centroid_quality.py)",
            [PYTHON, "centroid_quality.py",
             "--cache-dir", args.cache_dir,
             "--results-dir", args.results_dir,
             ] + k_flags + manifest_f,
            os.path.join(args.results_dir, "centroid_quality.csv"),
        ),

        # Step 5 — CNN training
        (
            5, "Train CNN at each degradation level (theModel.py)",
            [PYTHON, "theModel.py",
             "--results-dir", args.results_dir,
             ],
            os.path.join(args.results_dir, f"scores_k{args.k_values[0]}_psf0.csv"),
        ),

        # Step 6 — Bootstrap statistics + breakdown resolution
        (
            6, "Compute bootstrap AUC statistics (stats_ml.py)",
            [PYTHON, "stats_ml.py",
             "--results-dir", args.results_dir,
             "--n-bootstrap", str(args.n_bootstrap),
             "--snr-threshold", str(args.snr_threshold),
             *(["--enable-delong"] if args.enable_delong else []),
             ],
            os.path.join(args.results_dir, "breakdown.csv"),
        ),

        # Step 7 — Bryson baseline (+ optional vetting)
        (
            7, "Bryson baseline (baselines.py)",
            [PYTHON, "baselines.py",
             "--results-dir", args.results_dir,
             "--cache-dir", args.cache_dir,
             *(["--enable-vetting"] if args.enable_vetting else []),
             ] + k_flags,
            None,   # always run; idempotent (appends to breakdown.csv)
        ),

        # Step 8 — Final output assembly
        (
            8, "Assemble final output (compare.py)",
            [PYTHON, "compare.py",
             "--results-dir", args.results_dir,
             "--cache-dir", args.cache_dir,
             "--tpf-dir", args.tpf_dir,
             "--k-values"] + k_str + manifest_f,
            None,   # always run; cheap read-only assembly
        ),
    ]

    # ── print run plan ───────────────────────────────────────────────────────

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      Kepler Centroid-Resolution Degradation Pipeline         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  k values     : {args.k_values}")
    print(f"  Workers      : {args.workers} concurrent S3 download threads")
    print(f"  Search rate  : {args.search_rate} MAST API req/s (rate-limited)")
    print(f"  TPF dir      : {args.tpf_dir}")
    print(f"  Cache dir    : {args.cache_dir}")
    print(f"  Manifest     : {args.manifest}")
    print(f"  FP subset    : {args.fp_subset}")
    print(f"  DeLong       : {'ON' if args.enable_delong else 'off'}")
    print(f"  Vetting      : {'ON' if args.enable_vetting else 'off'}")
    if args.start_from > 1:
        print(f"  Starting from: step {args.start_from}")

    wall_t0 = time.perf_counter()

    for step_num, label, cmd, skip_sentinel in steps:
        if step_num < args.start_from:
            print(f"\n[{step_num}/8] {label}  →  skipped (--start-from {args.start_from})")
            continue
        run_step(step_num, label, cmd, skip_sentinel)

    wall_elapsed = time.perf_counter() - wall_t0
    print(f"\n{'═' * 60}")
    print(f"Pipeline complete in {wall_elapsed / 60:.1f} min")
    print(f"Results: {args.results_dir}/")
    print(f"  comparison.csv  · report.md  · figures/")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
