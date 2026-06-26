# Yet-Another-Exoplanet-Classifier

A study of how progressively degraded spatial sampling reduces the information content of
centroid measurements for automated exoplanet vetting, using a dual-branch 1D CNN and a
simplified Bryson (2013)-style difference-image baseline evaluated on Kepler KOI data.

---

## Scientific Objective

This repository answers a single question:

> **How does progressively degraded spatial sampling (effective astrometric resolution)
> reduce the information content of centroid measurements for automated exoplanet vetting?**

The degradation engine is an intentional experimental control: the photometric light curve
is held at native Kepler quality while the centroid pixel scale is systematically coarsened
by integer factors k = [1, 2, 3, 4, 5] (effective scales 3.98″ → 19.9″ arcsec/pixel).
This isolates centroid information content as the independent variable. This is **not** a
simulation of TESS or any other telescope — k is a scan through a control parameter.

The expected chain of reasoning is:

```
Spatial sampling (k)  →  Centroid measurement quality  →  Classification performance (AUC)
```

---

## Pipeline

### Step 1 — Build the target manifest
```bash
python getExamples.py [--fp-subset {all,centroid_offset}]
```
Queries the NASA Exoplanet Archive KOI cumulative table via TAP/ADQL and writes
`koi_manifest.csv`. Default (`--fp-subset all`): positives = CONFIRMED, negatives = all
FALSE POSITIVE subtypes. Pass `--fp-subset centroid_offset` to restrict negatives to
centroid-offset FPs only (`koi_fpflag_co=1`).

### Step 2 — Download TPFs and build cache
```bash
python getInputData.py [--max-quarters N]
```
Downloads Kepler long-cadence Target Pixel Files (TPFs) for each KIC target, then runs the
degradation engine to produce cached intermediate files. `--max-quarters N` caps how many
quarters are used per target (no cap if omitted). Resumable: existing cache files are
skipped. Outputs:
- `tpf_temp/` — raw FITS cache (lightkurve managed)
- `cache/flux/{kepid}.npz` — native-resolution phase-folded flux (shared across all k)
- `cache/centroids/{kepid}_k{K}_psf{0|1}.npz` — centroid time series per (k, PSF setting)

### Step 3 — Write training CSVs
```bash
python getInputData.py --phase csv-only
```
Reads from `cache/` (no FITS access) and writes one CSV per (k, PSF setting):
`kepler_training_data_k{K}_psf{0|1}.csv`. Flux columns are byte-identical across k; only
centroid columns (m1, m2) differ.

### Step 4 — Centroid quality analysis
```bash
python centroid_quality.py
```
Aggregates centroid RMS, uncertainty, SNR, and offset across targets per (k, PSF setting).
Writes `results_resolution/centroid_quality.csv` and `figures/fig2_centroid_quality.png`.

### Step 5 — Train CNN per degradation level
```bash
python theModel.py
```
Trains the dual-branch 1D CNN with KIC-grouped 5-fold CV plus a held-out test set for each
`kepler_training_data_k{K}_psf{*}.csv`. Saves per-target scores to
`results_resolution/scores_k{K}_psf{P}.csv`.

### Step 6 — Compute statistics
```bash
python stats_ml.py [--enable-delong]
```
Bootstrap 95% CIs on ROC-AUC and PR-AUC (resampling targets, 2000 iterations). Computes
breakdown resolution k_break_90 and k_break_snr. DeLong pairwise testing is optional
(disabled by default).

### Step 7 — Bryson baseline
```bash
python baselines.py [--enable-vetting]
```
Reads pre-computed difference-image offsets from `cache/centroids/` and computes ROC-AUC
+ bootstrap CIs per k. Optionally runs the `vetting` package (Hedges 2021) if
`--enable-vetting` is passed; failures are logged and never terminate the run.

### Step 8 — Assemble outputs
```bash
python compare.py
```
Writes `results_resolution/comparison.csv`, `results_resolution/report.md`, and figures.

---

## Model Architecture

A dual-branch 1D CNN (preserved from prior work, architecture untouched):

- **Global branch** — full 1,000-bin sequence, all 3 channels (flux, RA centroid, Dec
  centroid). Three Conv1D blocks (32→64→128 filters, kernel 5) + BatchNorm + MaxPooling →
  GlobalAveragePooling → 128-dim vector.
- **Local branch** — central 200 bins of the flux channel only (bins 400–600, the transit
  window). Two Conv1D blocks (32→64 filters) + GlobalAveragePooling → 64-dim vector.
- **Head** — Concatenate(192) → Dense(128) → Dropout(0.5) → Dense(32) → Dropout(0.5) →
  Dense(1, sigmoid).

Training: focal loss (γ=2.0, label smoothing=0.1), Adam 1e-3, linear LR warmup (epochs
0–4), ReduceLROnPlateau on val AUC, EarlyStopping (patience=15), class weights
{FP: 2.0, Planet: 1.0}, phase jitter + Gaussian noise + flux scaling augmentations.

---

## Data

- **Source:** NASA Exoplanet Archive KOI cumulative table; Kepler long-cadence TPFs via MAST
- **Positives (label 1):** `koi_disposition = CONFIRMED`
- **Negatives (label 0):** `koi_disposition = FALSE POSITIVE` (all subtypes, Option A default)
- **Epoch convention:** `koi_time0bk` is already in BKJD (BJD − 2,454,833). Used directly.
  Do NOT subtract 2,457,000 (that is the TESS BTJD offset).
- **Group identity:** `kepid` — all KOIs from one KIC star stay in one CV fold and on one
  side of the train/test split.

---

## Training CSV schema

`kepler_training_data_k{K}_psf{0|1}.csv` — 3,009 columns per row:
- 7 metadata fields: `kepid, kepoi_name, label, fp_subtype, koi_period, koi_time0bk, koi_duration`
- `f_0`…`f_999` — native-res flux, OOT-normalised, standardised (clipped ±10 before training)
- `m1_0`…`m1_999` — RA flux-weighted moment centroid at scale k, OOT-subtracted and standardised
- `m2_0`…`m2_999` — Dec centroid, same normalisation

Flux columns are identical across all k; only m1/m2 change with k and PSF setting.

---

## Cache hierarchy

```
tpf_temp/                          ← raw FITS (lightkurve cache; gitignored)
cache/
    flux/{kepid}.npz               ← native-res folded flux (computed once)
    centroids/{kepid}_k{K}_psf{P}.npz   ← centroid series + quality metrics per (k, PSF)
    cubes/{kepid}_q{Q}_k{K}_psf{P}.npz  ← degraded pixel cubes (intermediate)
results_resolution/
    centroid_quality.csv
    centroid_scaling.csv
    breakdown.csv
    comparison.csv
    report.md
    scores_k{K}_psf{P}.csv
    metrics_k{K}_psf{P}.txt
    figures/
        fig1_auc_vs_k.png
        fig2_centroid_quality.png
        fig3_difference_images.png
```

---

## Install

```bash
pip install lightkurve tensorflow scikit-learn numpy pandas scipy astropy tqdm matplotlib
# optional (only for --enable-vetting):
pip install vetting
```

---

## Key constraints

- **BKJD epoch** — `koi_time0bk` is used directly; never subtract 2,457,000.
- **No target leakage** — `kepid` groups are never split across CV folds or train/test boundary.
- **k=1 validation gate** — the degradation engine asserts that native-resolution
  difference-image centroid uncertainty is < 1.0″ on ≥80% of targets before proceeding.
- **CNN architecture is preserved** — `theModel.py` changes only the CV grouping, the
  held-out test split, and the outer per-k loop. All layers, loss, and training config
  are untouched.
- **vetting is optional** — `baselines.py --enable-vetting`; defaults OFF. All vetting
  failures are caught and logged; they never terminate the pipeline.
