"""
archive_results.py — Archive results_resolution/ contents before a fresh pipeline run.

run_pipeline.py's steps use file-existence checks to decide whether to skip a stage
(e.g. Step 3 skips if kepler_training_data_k{K}_psf0.csv already exists, Step 4 skips
if results_resolution/centroid_quality.csv exists, Step 6 skips if
results_resolution/breakdown.csv exists). After rebuilding cache/centroids/ with the
aperture-degeneracy fix, those stale marker files must be moved out of the way so the
pipeline actually regenerates them from the fixed cache, rather than silently reusing
pre-fix results.

This script does nothing else: it does not touch cache/, does not re-run any pipeline
stage, and does not modify the *contents* of anything in results_resolution/ — it only
relocates the current top-level entries into a new results_resolution/old_v{N}/ folder,
matching the old_v1/old_v2 archive convention already used in this repo.

Usage:
  python archive_results.py
      Archive results_resolution/* into results_resolution/old_v{N}/ (next available N).

  python archive_results.py --delete-training-csvs
      Also delete kepler_training_data_k{1..5}_psf{0,1}.csv at the repo root, so
      run_pipeline.py Step 3 (csv-only) regenerates them instead of skipping. These
      files live at the repo root (not inside results_resolution/), are gitignored,
      and are fully regenerable from cache/ — so they are deleted outright rather
      than archived.
"""

import argparse
import os
import re
import shutil

RESULTS_DIR = "results_resolution"
ARCHIVE_PREFIX = "old_v"
K_VALUES = [1, 2, 3, 4, 5]


def _next_archive_name(results_dir: str) -> str:
    """Find the next unused old_v{N} name by scanning existing archive folders."""
    existing = []
    if os.path.isdir(results_dir):
        for name in os.listdir(results_dir):
            m = re.fullmatch(rf"{ARCHIVE_PREFIX}(\d+)", name)
            if m and os.path.isdir(os.path.join(results_dir, name)):
                existing.append(int(m.group(1)))
    next_n = max(existing, default=0) + 1
    return f"{ARCHIVE_PREFIX}{next_n}"


def archive_results(results_dir: str = RESULTS_DIR) -> str:
    """
    Move every current top-level entry of results_dir into a new
    results_dir/old_v{N}/ subfolder. Previously-archived old_v* folders and
    dotfiles (e.g. .DS_Store) are left in place. Never touches anything outside
    results_dir — in particular, never touches cache/.

    Returns the path to the new archive folder (empty string if results_dir
    doesn't exist).
    """
    if not os.path.isdir(results_dir):
        print(f"{results_dir} does not exist — nothing to archive.")
        return ""

    archive_name = _next_archive_name(results_dir)
    archive_path = os.path.join(results_dir, archive_name)
    os.makedirs(archive_path, exist_ok=False)

    moved = []
    for name in sorted(os.listdir(results_dir)):
        if name == archive_name:
            continue
        if re.fullmatch(rf"{ARCHIVE_PREFIX}\d+", name):
            continue  # leave previously-archived folders where they are
        if name.startswith("."):
            continue  # skip OS artefacts (.DS_Store etc.)
        src = os.path.join(results_dir, name)
        dst = os.path.join(archive_path, name)
        shutil.move(src, dst)
        moved.append(name)

    print(f"Archived {len(moved)} item(s) from {results_dir}/ into {archive_path}/:")
    for name in moved:
        print(f"  {name}")
    if not moved:
        print("  (nothing to archive — results_resolution was already empty)")

    return archive_path


def delete_training_csvs(k_values: list[int] = K_VALUES) -> None:
    """
    Delete kepler_training_data_k{K}_psf{P}.csv at the repo root for all K in
    k_values, psf in [0, 1]. These live at the repo root (not inside
    results_resolution/, so archive_results() never touches them) and are fully
    regenerable via 'getInputData.py --phase csv-only' from cache/ — deleted
    outright, not archived, since they carry no historical value once stale.
    """
    deleted = []
    for k in k_values:
        for psf in [0, 1]:
            path = f"kepler_training_data_k{k}_psf{psf}.csv"
            if os.path.exists(path):
                os.remove(path)
                deleted.append(path)

    print(f"\nDeleted {len(deleted)} training CSV(s):")
    for path in deleted:
        print(f"  {path}")
    if not deleted:
        print("  (none found)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive results_resolution/ before a fresh pipeline run, so "
                     "run_pipeline.py's file-existence skip checks don't reuse stale results."
    )
    parser.add_argument(
        "--results-dir", default=RESULTS_DIR,
        help=f"Directory to archive (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--delete-training-csvs", action="store_true",
        help="Also delete kepler_training_data_k{1..5}_psf{0,1}.csv at the repo root, "
             "so run_pipeline.py Step 3 (csv-only) regenerates them instead of skipping.",
    )
    args = parser.parse_args()

    archive_results(args.results_dir)

    if args.delete_training_csvs:
        delete_training_csvs()


if __name__ == "__main__":
    main()
