# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git remotes — IMPORTANT

This local repository has two remotes:

| Remote | URL | Purpose |
|---|---|---|
| `centroid-evaluator` | `https://github.com/mrbeats09/The-Centroid-Evaluator.git` | **Active development — push here** |
| `origin` | `https://github.com/mrbeats09/Yet-Another-Exoplanet-Classifier.git` | Legacy/archived — do NOT push here |

**Always push to `centroid-evaluator`**, never to `origin`:
```bash
git push centroid-evaluator main
```
`origin` hosts an older version of the codebase. All current and future work lives on `The-Centroid-Evaluator`.

## Project Summary

A study of how progressively degraded spatial sampling reduces centroid information content
for automated Kepler exoplanet vetting. A dual-branch 1D CNN and a simplified
Bryson (2013)-style difference-image baseline are evaluated at k = [1,2,3,4,5] effective
pixel scales (3.98·k arcsec/pixel) to produce an AUC-vs-pixel-scale degradation curve.

**This is NOT a TESS simulation.** The k parameter is a spatial-sampling control variable,
not an approximation of any particular telescope. All code comments, docstrings, and output
files must use this framing.

## Pipeline (run in order)

```bash
# Step 1: Build KOI manifest from NASA Exoplanet Archive
python getExamples.py [--fp-subset {all,centroid_offset}]   # → koi_manifest.csv

# Step 2a: Download Kepler TPFs and build degraded cache
python getInputData.py --tpf-dir "/Volumes/Stuff/Research Work/TPFs" [--max-quarters N]
# ↳ scans local TPFs → cache/cubes/ → cache/centroids/ + cache/flux/

# Step 2b: Write per-k training CSVs from cache (fast; no FITS access)
python getInputData.py --phase csv-only

# Step 3: Centroid quality analysis
python centroid_quality.py       # reads cache/centroids/ → results_resolution/

# Step 4: Train CNN per degradation level
python theModel.py               # reads training CSVs → results_resolution/scores_k*.csv

# Step 5: Bootstrap statistics
python stats_ml.py

# Step 6: Bryson baseline
python baselines.py [--enable-vetting]

# Step 7: Assemble final output
python compare.py                # → results_resolution/comparison.csv, report.md

# Step 8: Publication figures
python graphs.py                 # → results_resolution/graphs/*.png
```

Or run everything at once:
```bash
python run_pipeline.py --tpf-dir "/Volumes/Stuff/Research Work/TPFs" [--start-from N]
```

## Install dependencies

```bash
pip install -r requirements.txt
# vetting is optional; only needed with --enable-vetting:
pip install vetting
```

## Architecture

Dual-branch 1D CNN — **architecture is frozen, do not change it**:

- **Global branch** — full 1,000-bin sequence, 3 channels (flux=0, RA centroid=1, Dec
  centroid=2). Three Conv1D blocks (32→64→128 filters, kernel 5) + BatchNorm + MaxPooling
  → GlobalAveragePooling → 128-dim vector.
- **Local branch** — central 200 bins (`LOCAL_START=400` to `LOCAL_END=600`) of **channel 0
  (flux) only** (`X[:, 400:600, 0:1]`). Two Conv1D blocks → 64-dim vector.
- **Head** — Concatenate(192) → Dense(128) → Dropout(0.5) → Dense(32) → Dropout(0.5) →
  Dense(1, sigmoid).

Training: focal loss (γ=2.0, α=0.5, smoothing=0.1), Adam 1e-3, linear LR warmup epochs
0–4 (1e-4→1e-3), ReduceLROnPlateau(val_auc, patience=7), EarlyStopping(patience=15),
class weights {FP:2.0, Planet:1.0}, augmentations: phase roll ±20 bins, Gaussian noise
(σ=0.02) and scaling (×0.95–1.05) on flux channel only.

Ablation variants (`theModel.py`, psf=0 only): `build_model_flux_only()` and
`build_model_centroid_only()` train in a self-contained loop that must never be merged
into the primary fold loop. Their scores land in `scores_{tag}_flux_only.csv` and
`scores_{tag}_centroid_only.csv`.

## Data format

`kepler_training_data_k{K}_psf{0|1}.csv` — 3,009 columns per row:
- 7 metadata fields: `kepid, kepoi_name, label, fp_subtype, koi_period, koi_time0bk, koi_duration`
- `f_0`…`f_999` — native-res flux, OOT-normalised, OOT-standardised (clipped ±3σ then ±10 before training)
- `m1_0`…`m1_999` — RA flux-weighted moment centroid at scale k; OOT-subtracted + standardised
- `m2_0`…`m2_999` — Dec centroid, same normalisation

Labels: 0 = False Positive, 1 = Confirmed planet.
`fp_subtype` encodes which FP flags are set (e.g. `co`, `ss`, `nt`, `ec`, `other`).
Flux columns are byte-identical across all k values; only m1/m2 differ by k and PSF setting.
Flux cache is **per-KOI** (keyed by `kepoi_name`, each planet's own ephemeris); centroid
cache is **per-KIC star** (keyed by `kepid`, aperture property not transit-specific).

## Key design decisions

- **BKJD epoch** — `koi_time0bk` is already in BKJD (BJD − 2,454,833). Use it directly.
  **Never subtract 2,457,000** (that is the TESS BTJD offset). Every file that uses the
  epoch must carry a comment to this effect.
- **No target leakage** — group identity is `kepid`. All KOIs from one KIC star must stay
  within one CV fold and entirely on one side of the train/test split. `theModel.py` uses
  `StratifiedGroupKFold` with `groups=kepid_array`. Runtime asserts check for leakage.
- **OOF threshold** — the ensemble test-set threshold in `theModel.py` is derived from
  out-of-fold CV predictions only, never from the test set. The test-set oracle threshold
  is computed as an informational line but never used for reported metrics.
- **Bryson score direction** — high SNR (offset/uncertainty) means likely FP (label 0).
  sklearn's `roc_auc_score` expects higher score → label 1, so `baselines.py` negates:
  `score_for_auc = -valid["bryson_score"].values`. The binary Bryson classifier is the
  **primary** Bryson result; continuous-score AUC is secondary.
- **k_break_90** — uses an **absolute** AUC threshold of 0.90, not 90% of the k=1 AUC.
  Defined in `stats_ml.py` as `ABSOLUTE_AUC_THRESHOLD = 0.90`.
- **k=1 validation gate** — `degrade.py` asserts that on ≥80% of qualifying targets the
  native-resolution difference-image centroid uncertainty is < 1.0″ before producing any
  higher-k outputs.
- **Local TPF scan** — `getInputData.py` never re-queries MAST during Phase 2. It scans
  the local `--tpf-dir` recursively via `load_local_tpfs()`, matching
  `kplr{kepid:09d}*_lpd-targ.fits*` and excluding macOS `._` AppleDouble sidecar files.
  TPFs live on an external volume: `/Volumes/Stuff/Research Work/TPFs`.
- **Cache hierarchy** — expensive operations run once and are skipped on re-run:
  local FITS → `cache/cubes/` → `cache/centroids/` + `cache/flux/`. CNN training reads
  training CSVs only; never opens FITS or recomputes degradation.
- **Additive pattern** — when adding capability near shared or critical code, add a
  sibling function rather than changing an existing function's signature or refactoring
  the primary CNN fold loop. Example: `bootstrap_auc_with_p` is a separate 4-tuple
  sibling of `bootstrap_auc` (which stays frozen at 3-tuple because `baselines.py`
  unpacks exactly 3 values).
- **Sparsity threshold** — targets with >30% empty phase bins are discarded to
  `sparse_targets.log`. Failed TPF loads go to `failed_targets.log`.
- **PSF broadening** — two result sets per k: PSF off (rebinning only) and PSF on
  (rebinning + Gaussian convolution). Both are second-order robustness checks, not
  physical models. Code must say so in comments.
- **DeLong removed** — `--enable-delong` is accepted by `stats_ml.py` and
  `run_pipeline.py` as a deprecated no-op (prints `DeprecationWarning`). Do not
  re-add DeLong computation; primary significance is bootstrap permutation tests.
- **Difference-image centroids** — described as "a simplified implementation following
  the methodology of Bryson et al. (2013)" throughout. No claim of pipeline equivalence.
- **vetting integration** — optional, gated by `--enable-vetting`. Disabled by default.
  All `vetting.centroid_test` calls are wrapped in `try/except`.

## Output locations

| File | Location |
|---|---|
| Target manifest | `koi_manifest.csv` |
| Native flux cache | `cache/flux/{kepoi_name}.npz` |
| Centroid / quality cache | `cache/centroids/{kepid}_k{K}_psf{P}.npz` |
| Degraded cube cache | `cache/cubes/{kepid}_q{Q}_k{K}_psf{P}.npz` |
| Training CSVs | `kepler_training_data_k{K}_psf{0|1}.csv` |
| CNN scores | `results_resolution/scores_k{K}_psf{P}.csv` |
| Ablation scores | `results_resolution/scores_k{K}_psf0_{flux_only\|centroid_only}.csv` |
| CNN per-level metrics | `results_resolution/metrics_k{K}_psf{P}.txt` |
| Bryson binary scores | `results_resolution/bryson_binary_scores_k{K}_psf{P}.csv` |
| Centroid quality (agg) | `results_resolution/centroid_quality.csv` |
| Centroid quality (raw) | `results_resolution/centroid_quality_per_target.csv` |
| Centroid scaling table | `results_resolution/centroid_scaling.csv` |
| Data attrition report | `results_resolution/data_attrition.txt` |
| Breakdown resolution | `results_resolution/breakdown.csv` |
| Full comparison | `results_resolution/comparison.csv` |
| Human-readable report | `results_resolution/report.md` |
| Centroid quality figure | `results_resolution/figures/fig2_centroid_quality.png` |
| Difference image figure | `results_resolution/figures/fig3_difference_images.png` |
| Publication figures | `results_resolution/graphs/fig{1–6}_*.png` |

## Gitignored paths

`.venv`, `__pycache__`, `tpf_temp`, `cache`, `kepler_training_data_*.csv`,
`failed_targets.log`, `sparse_targets.log`, `.DS_Store`, `.claude`, `CLAUDE.md`
