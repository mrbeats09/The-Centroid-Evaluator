# The Centroid Evaluator

To what extent does spatial resolution make a difference when using centroid movement to vet out exoplanets? 

This project answers exactly this by taking real Kepler data and systematically coarsening the pixel scale of the centroid measurements while holding everything else constant. At native resolution a centroid shift can tell you whether a transit is coming from the target star or a nearby background eclipsing binary. As the pixels get coarser that signal degrades. This project measures exactly how fast it degrades and at what point it stops being useful for vetting exoplanets.

The result is a degradation curve: ROC-AUC vs. effective pixel scale, produced by a 1-dimensional convolutional neural network classifier as well as a simpler Bryson-style baseline [(Bryson et al.)](http://arxiv.org/abs/1303.0052), with bootstrap confidence intervals at each level.

---

## How it works (Independent Variable) 

As you may know, Kepler's native pixel scale is 3.98 arcseconds per pixel. We apply a rebinning factor `k` from 1 to 5, which gives effective scales ranging from 3.98 to 19.9 arcsec/pixel. The flux light curve is kept at native resolution throughout. Only the centroid channels are degraded. This isolates spatial sampling as the independent variable.

```
Pixel scale (k)  →  Centroid quality  →  Classification performance
```

This is not a simulation of TESS or any other telescope, given that flux remains as-is. However, future work could potentially simulate a reduced flux imaging resolution for greater accuracy. It would allow for existing projects such as PLATO or RST to be directly compared in order to view the efficacy of centroid motion on their observations, or even influence future projects when deciding on an ideal pixel scale for observation. 

---

## Stages

The pipeline runs in eight stages. You can run them individually or all at once with `run_pipeline.py`.

**Stage 1 — Build the target list** (`getExamples.py`)  
Queries the NASA Exoplanet Archive for Kepler Objects of Interest and writes a manifest of confirmed planets and false positives to use as training data.

**Stage 2 — Download and process data** (`getInputData.py`)  
Downloads Kepler Target Pixel Files from MAST and runs the degradation engine to build a local cache of flux light curves and centroid time series at each k value. This is the slow step. It is fully resumable since completed targets are skipped.

**Stage 3 — Write training CSVs** (`getInputData.py --phase csv-only`)  
Reads from the cache and writes one CSV per (k, PSF setting) combination. No internet access or FITS files needed at this point.

**Stage 4 — Centroid quality analysis** (`centroid_quality.py`)  
Aggregates centroid RMS, uncertainty and SNR across all targets at each k level. This makes the middle link in the chain explicit before any model training happens.

**Stage 5 — Train the classifier** (`theModel.py`)  
Trains a dual-branch 1D CNN on each training CSV using 5-fold cross-validation grouped by KIC target ID, with a separate held-out test set. Produces per-target predicted scores for evaluation.

**Stage 6 — Statistics** (`stats_ml.py`)  
Computes ROC-AUC with bootstrap 95% confidence intervals by resampling targets (not cadences). Also identifies the breakdown point where performance falls below 90% of native-resolution AUC.

**Stage 7 — Bryson baseline** (`baselines.py`)  
A simpler classifier using the difference-image centroid offset divided by its uncertainty as a detection score. Provides a direct comparison against the neural network and requires no training.

**Stage 8 — Assemble results** (`compare.py`)  
Collects all scores and metrics into a single comparison table, generates figures and writes a human-readable report.

---

## Usage

### Run everything at once

```bash
python run_pipeline.py
```

This runs all eight stages in order. Each stage is gated on the previous one: if a step fails the pipeline stops immediately and tells you which step broke and that earlier outputs are still cached.

### Common options

```bash
# Limit download to 4 quarters per target (much faster for testing)
python run_pipeline.py --max-quarters 4

# Store TPFs somewhere other than the default ./tpf_temp
# (if you change this, add the path to .gitignore yourself since TPFs can be tens of GB)
python run_pipeline.py --tpf-dir /data/kepler_tpfs

# Resume from a specific step after fixing a failure
python run_pipeline.py --start-from 5

# Enable optional extras
python run_pipeline.py --enable-delong --enable-vetting
```

### Run stages individually

```bash
python getExamples.py
python getInputData.py --max-quarters 4
python getInputData.py --phase csv-only
python centroid_quality.py
python theModel.py
python stats_ml.py
python baselines.py
python compare.py
```

### Output

Results land in `results_resolution/`:

```
results_resolution/
    comparison.csv          one row per (k, PSF setting) with all AUCs and CIs
    breakdown.csv           where performance degrades below key thresholds
    report.md               narrative summary of findings
    figures/
        fig1_auc_vs_k.png
        fig2_centroid_quality.png
        fig3_difference_images.png
```

---

## Dependencies

Install core dependencies with:

```bash
pip install lightkurve tensorflow scikit-learn numpy pandas scipy astropy tqdm matplotlib
```

The `vetting` package is optional and only needed if you pass `--enable-vetting`:

```bash
pip install vetting
```

Or install everything from the requirements file:

```bash
pip install -r requirements.txt
```

---

## A few things worth knowing

**Data source:** Positives are KOIs with `koi_disposition = CONFIRMED`, ignoring the candidate ones. Negatives are all `FALSE POSITIVE` subtypes (the default). You can restrict to centroid-offset false positives only with `--fp-subset centroid_offset`.

**No leakage:** All KOIs from the same KIC star are always on the same side of any train/test split. Cross-validation folds are grouped by `kepid`.

**Caching:** The two most 'expensive' operations (downloading TPFs + building the degraded cache) each write their outputs to disk and skip targets that are already done. Re-running after an interruption isn't a hassle. 

**PSF broadening:** Each k level is processed twice, once with rebinning only and once with an additional Gaussian blur to simulate PSF growth. The comparison between these two settings is a robustness check rather than a physical claim.

**Vetting and DeLong are off by default:** Added in for extra comparison points between the ML and statistical results, they are not present in the main evaluation which utilises the pipeline established by Bryson et al. (2013). 
