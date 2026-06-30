"""
theModel.py — Dual-Branch 1D CNN for Kepler Exoplanet Centroid Degradation Study

Architecture: Two parallel convolutional branches whose outputs are concatenated
before a regularised classification head.
  - Global branch: all 3 channels (flux=ch0, RA centroid=ch1, Dec centroid=ch2)
    × 301 phase bins. Learns coarse features: centroid trends, OOT variability.
  - Local branch: flux channel only (ch0), 61-bin adaptive transit-centred window.
    Learns fine transit morphology: flat-bottom vs V-shape, ingress sharpness.
    The local view is an adaptive per-target resample stored as l_0..l_60 in the
    training CSV (±2×transit_duration in phase, resampled to 61 bins via np.interp).
    It is NOT sliced from the global tensor at training time.

Architecture and training configuration are FROZEN. Do not modify them.
The channel layout (flux=0, m1=1, m2=2) is preserved.

# ARCHITECTURE NOTE: Input dimensions changed from (1000, 3) global + (200, 1) local
# to (301, 3) global + (61, 1) local after preprocessing overhaul.
# Local input is now a pre-computed adaptive window stored in l_0..l_60 CSV columns.
# The local branch no longer slices the global tensor — LOCAL_START / LOCAL_END removed.
# Do not revert these dimensions without also rebuilding the training CSVs.

Changes from the prior TESS version:
  - CV uses StratifiedGroupKFold keyed on kepid (no KIC target leakage)
  - 20% held-out test set, split at kepid level before any CV
  - Ensemble threshold fixed from OOF CV predictions (not test-set optimised)
  - Outer loop over per-k training CSVs (kepler_training_data_k{K}_psf{P}.csv)
  - Per-k test scores saved to results_resolution/scores_k{K}_psf{P}.csv
  - Ablation study (flux-only, centroid-only) for psf=0 CSVs
  - OUTPUT_DIR → results_resolution/

All other code (focal loss, optimiser, LR schedule, augmentations, architecture)
is untouched from the working TESS version except dimension updates.

Outputs (per k, psf):
  - results_resolution/scores_k{K}_psf{P}.csv  — per-target test-set scores
  - results_resolution/metrics_k{K}_psf{P}.txt — per-fold + ensemble metrics
  - results_resolution/confusion_matrix_k{K}_psf{P}.png
  - results_resolution/scores_k{K}_psf0_flux_only.csv    (psf=0 only)
  - results_resolution/scores_k{K}_psf0_centroid_only.csv (psf=0 only)
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import Sequence

from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — architecture/training constants frozen; do not change
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "results_resolution"
NUM_BINS    = 301   # global phase bins (was 1000; matches AstroNet global view)
LOCAL_BINS  = 61    # adaptive local view bins (was 200 slice-based; now pre-computed)
# LOCAL_START / LOCAL_END removed: local input is now read from l_0..l_60 CSV columns,
# not sliced from the global tensor at training time.
N_FOLDS     = 5
EPOCHS      = 120
BATCH_SIZE  = 32
RANDOM_SEED = 42
TEST_FRAC   = 0.20   # 20% of unique kepids held out as test set

# Class weights: down-weight the planet class so FP misclassifications
# contribute proportionally more to the loss and gradient updates.
CLASS_WEIGHT = {0: 2.0, 1: 1.0}

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path):
    """
    Load the per-k training CSV and reshape flat columns into:
      X_global: (N, 301, 3) — global phase-folded view, channels-last
      X_local:  (N,  61, 1) — adaptive transit-centred local view, channels-last

    TensorFlow's Conv1D expects channels-last: (batch, length, channels).

    Channel layout for X_global (FROZEN — do not change):
      channel 0 = flux  (clipped ±10)
      channel 1 = m1    (RA flux-weighted moment centroid at scale k)
      channel 2 = m2    (Dec centroid)

    X_local is the pre-computed adaptive local window stored as l_0..l_60 in
    the CSV (flux channel only; computed by _extract_local_bins in getInputData.py).
    Phase jitter augmentation is NOT applied to X_local at training time — rolling
    a transit-centred resample defeats the purpose of centring it.

    Also returns kepid_arr and fp_subtype_arr for grouping and subgroup analysis.

    Returns:
        X_global:      (N, 301, 3) float32 array
        X_local:       (N,  61, 1) float32 array
        y:             (N,) int32 labels
        kepid_arr:     (N,) int array of KIC IDs
        fp_subtype:    (N,) str array of FP subtype codes
        n:             total example count
    """
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    n  = len(df)
    print(f"  {n} examples  |  "
          f"Label 0 (FP): {(df['label']==0).sum()}  |  "
          f"Label 1 (Planet): {(df['label']==1).sum()}  |  "
          f"{df['kepid'].nunique()} unique KIC targets")

    # Global view: 301 bins × 3 channels
    flux   = df[[f"f_{i}"  for i in range(NUM_BINS)]].values.astype(np.float32)
    centr1 = df[[f"m1_{i}" for i in range(NUM_BINS)]].values.astype(np.float32)
    centr2 = df[[f"m2_{i}" for i in range(NUM_BINS)]].values.astype(np.float32)

    flux = np.clip(flux, -10.0, 10.0)

    # Stack to (N, 301, 3) — channels last; ch0=flux, ch1=m1, ch2=m2
    X_global = np.stack([flux, centr1, centr2], axis=2)
    X_global = np.nan_to_num(X_global, nan=0.0)

    # Local view: 61 adaptive bins, flux channel only
    local = df[[f"l_{i}" for i in range(LOCAL_BINS)]].values.astype(np.float32)
    local = np.clip(local, -10.0, 10.0)
    X_local = local[:, :, np.newaxis]       # (N, 61, 1)
    X_local = np.nan_to_num(X_local, nan=0.0)

    y           = df["label"].values.astype(np.int32)
    kepid_arr   = df["kepid"].values
    fp_subtype  = df.get("fp_subtype", pd.Series([""] * n)).values

    print(f"  Global tensor shape: {X_global.shape}  dtype: {X_global.dtype}")
    print(f"  Local  tensor shape: {X_local.shape}  dtype: {X_local.dtype}")
    return X_global, X_local, y, kepid_arr, fp_subtype, n


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — AUGMENTATION SEQUENCE
# ═════════════════════════════════════════════════════════════════════════════

class AugmentedSequence(Sequence):
    """
    A Keras Sequence that applies three physically motivated augmentations
    to the global branch input on-the-fly per training batch. The local branch
    input (X_local) is kept fixed — rolling or scaling an adaptive transit-centred
    resample would destroy the transit alignment that is the local view's purpose.

    Augmentations applied to X_global only:
      1. Phase jitter: roll the full 301-bin light curve by ±20 bins.
         Prevents the model learning "the dip is always at bin 150" and makes
         it robust to small errors in the reported transit epoch.
      2. Gaussian noise on flux channel (ch0) only.
         Simulates photon noise variability between observing epochs.
      3. Flux scaling: multiply flux channel by a factor drawn from [0.95, 1.05].
         Simulates transit depth uncertainty due to dilution or stellar variability.

    Using a Sequence rather than a manual loop means we can safely pass this into
    model.fit(), getting all the stability benefits of Keras's training loop.
    Augmentations are ONLY applied during training; validation data is never augmented.
    """
    def __init__(self, X_global, X_local, y, batch_size, shuffle=True):
        self.X_global   = X_global
        self.X_local    = X_local
        self.y          = y
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.indices    = np.arange(len(X_global))
        self.on_epoch_end()

    def __len__(self):
        return len(self.X_global) // self.batch_size

    def on_epoch_end(self):
        # Reshuffle training order each epoch so the model cannot
        # memorise the sequence in which examples are presented.
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_global_batch = self.X_global[batch_idx].copy()
        X_local_batch  = self.X_local[batch_idx]          # no copy — not augmented
        y_batch        = self.y[batch_idx]

        for i in range(len(X_global_batch)):
            # 1. Phase jitter: roll the entire global light curve by ±20 bins.
            shift = np.random.randint(-20, 21)
            X_global_batch[i] = np.roll(X_global_batch[i], shift, axis=0)

            # 2. Gaussian noise on the flux channel only (channels-last: [:, 0]).
            X_global_batch[i, :, 0] += np.random.normal(
                0, 0.02, NUM_BINS).astype(np.float32)

            # 3. Flux scaling: multiply flux by a factor drawn from [0.95, 1.05].
            X_global_batch[i, :, 0] *= np.random.uniform(0.95, 1.05)

        return (X_global_batch, X_local_batch), y_batch


def split_inputs(X_global, X_local):
    """
    Return the two-branch model inputs as a tuple, without augmentation.
    Used for validation and inference.
    """
    return (X_global, X_local)


def split_inputs_flux_only(X_global, X_local):
    """
    Split for flux-only ablation: global branch receives flux channel only
    (shape (N, 301, 1)); local branch receives the pre-computed adaptive local
    window (already flux-only, shape (N, 61, 1)).
    """
    return (X_global[:, :, 0:1], X_local)


def split_inputs_centroid_only(X_global, X_local):
    """
    Split for centroid-only ablation: both branches receive the full centroid
    sequence (ch1=RA, ch2=Dec from X_global), shape (N, 301, 2). The local
    branch is not used — slicing the central bins of a centroid series has no
    physical transit-morphology interpretation — so both input slots receive the
    same centroid array.
    """
    centroid = X_global[:, :, 1:3]
    return (centroid, centroid)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FOCAL LOSS WITH LABEL SMOOTHING
# ═════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma=2.0, alpha=0.5, smoothing=0.1):
    """
    Focal loss is a modification of binary cross-entropy that down-weights
    the loss contribution of easy, well-classified examples and focuses
    learning on the hard, misclassified boundary cases.

    For classification with small datasets where "obviously planet" examples
    dominate, focal loss prevents those easy examples from drowning out
    gradient signal to the ambiguous cases where the model most needs to learn.

    Label smoothing replaces hard 0/1 labels with 0.05/0.95, preventing the
    model from becoming overconfident and driving its sigmoid outputs to
    exactly 0 or 1, which causes gradients to vanish.

    Parameters:
      gamma    — focusing parameter (default 2.0). Higher γ focuses more on
                 hard examples. γ=0 recovers vanilla BCE.
      alpha    — class weighting (default 0.5). Controls the contribution
                 of the foreground class relative to the background.
      smoothing — label smoothing strength (default 0.1). Replaces labels
                  0 → 0.05 and 1 → 0.95 to prevent overconfidence.
    """
    def loss_fn(y_true, y_pred):
        # Cast labels to float32 for loss computation
        y_true = tf.cast(y_true, tf.float32)

        # Apply label smoothing: 0 → 0.05, 1 → 0.95
        y_true = y_true * (1.0 - smoothing) + smoothing * 0.5

        # Clip predictions to a safe range to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Vanilla binary cross-entropy
        bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)

        # Focal term: (1 - p_t)^gamma, where p_t is the model confidence
        # in the true class.
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # Focal loss: weight hard examples (low p_t) more than easy ones (high p_t)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        # Combine: alpha balances foreground/background, focal_weight emphasises hard examples
        return tf.reduce_mean(alpha * focal_weight * bce)

    return loss_fn


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LEARNING RATE WARM-UP SCHEDULE
# ═════════════════════════════════════════════════════════════════════════════

def warmup_schedule(epoch, lr):
    """
    Linear learning rate warm-up over the first 5 epochs, ramping from
    1e-4 to 1e-3. After epoch 5, the learning rate is controlled entirely
    by ReduceLROnPlateau, which responds to validation AUC plateau.

    Rationale: Adam's moving averages are very noisy at the start of training.
    Starting with a cold learning rate of 1e-3 causes wild gradient swings
    before the exponential moving averages have stabilised. Warming up from
    a much lower starting point gives Adam time to accumulate meaningful
    history before making large parameter updates.
    """
    if epoch < 5:
        return 1e-4 + (1e-3 - 1e-4) * (epoch / 5.0)
    return lr  # After warmup, let ReduceLROnPlateau take over


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

def build_model():
    """
    Dual-branch shallow 1D CNN with batch normalisation after every Conv1D.
    Deliberately shallow to match the ~1,600 training examples available per
    fold. Deeper architectures have more parameters than the data can
    reliably constrain and will overfit.

    Input shapes use channels-last convention: (length, channels).
    GlobalAveragePooling collapses the time dimension and acts as a mild
    regulariser — averaging rather than selecting the max makes it less
    prone to latching onto single-timestep outliers.

    Global branch pooling: MaxPooling1D(4) → 301→75, then MaxPooling1D(5) → 75→15.
    Local branch pooling:  MaxPooling1D(3) → 61→20, then GlobalAveragePooling.
    """

    # ── Global branch: full orbit, all 3 channels ────────────────────────────
    global_input = keras.Input(shape=(301, 3), name="global_input")

    x = layers.Conv1D(32, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(global_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)                    # 301 → 75

    x = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=5)(x)                    # 75  → 15

    x = layers.Conv1D(128, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    global_out = layers.GlobalAveragePooling1D()(x)            # → (128,)

    # ── Local branch: adaptive transit window, flux channel only ─────────────
    # Input is pre-computed l_0..l_60 (61 bins), not sliced from the global tensor.
    local_input = keras.Input(shape=(61, 1), name="local_input")

    y = layers.Conv1D(32, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(local_input)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.MaxPooling1D(pool_size=3)(y)                    # 61  → 20

    y = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    local_out = layers.GlobalAveragePooling1D()(y)             # → (64,)

    # ── Classification head ───────────────────────────────────────────────────
    combined = layers.Concatenate()([global_out, local_out])   # → (192,)

    z = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(combined)
    z = layers.Dropout(0.5)(z)

    z = layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(z)
    z = layers.Dropout(0.5)(z)

    output = layers.Dense(1, activation="sigmoid")(z)

    model = keras.Model(inputs=[global_input, local_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.5, smoothing=0.1),
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


def build_model_flux_only():
    """
    Flux-only ablation variant of the dual-branch CNN.

    Removes the centroid channels entirely — both branches operate on the flux
    channel only (ch0). Used to establish the baseline performance achievable
    from transit morphology alone, with no centroid information.

    Architecture mirrors the combined model exactly:
      Global: (301, 1), same pooling (4 then 5)
      Local:  (61,  1), same pooling (3)
    """
    # ── Global branch: full orbit, flux only ────────────────────────────────
    global_input = keras.Input(shape=(301, 1), name="global_input")

    x = layers.Conv1D(32, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(global_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)                    # 301 → 75

    x = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=5)(x)                    # 75  → 15

    x = layers.Conv1D(128, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    global_out = layers.GlobalAveragePooling1D()(x)            # → (128,)

    # ── Local branch: adaptive transit window, flux only ─────────────────────
    local_input = keras.Input(shape=(61, 1), name="local_input")

    y = layers.Conv1D(32, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(local_input)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.MaxPooling1D(pool_size=3)(y)                    # 61  → 20

    y = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    local_out = layers.GlobalAveragePooling1D()(y)             # → (64,)

    # ── Classification head ───────────────────────────────────────────────────
    combined = layers.Concatenate()([global_out, local_out])   # → (192,)

    z = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(combined)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(z)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(1, activation="sigmoid")(z)

    model = keras.Model(inputs=[global_input, local_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.5, smoothing=0.1),
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


def build_model_centroid_only():
    """
    Centroid-only ablation variant of the dual-branch CNN.

    Removes the flux channel entirely — both branches operate on centroid channels
    (ch1=RA, ch2=Dec from X_global), shape (N, 301, 2). The local transit-window
    view is not used — slicing the central bins of a centroid series has no
    physical transit-morphology interpretation — so both input slots receive the
    same full centroid sequence.

    Branch 1 (3 Conv1D blocks, 32→64→128 filters, pools 4 then 5) mirrors the
      combined global branch.
    Branch 2 (2 Conv1D blocks, 32→64 filters, pool 3) mirrors the combined local-
      branch architecture but on the full centroid sequence.

    Used to isolate the contribution of centroid information alone, without transit
    morphology features. Performance degrades with k as centroid SNR decreases.
    """
    # ── Branch 1: full centroid sequence, 3-block architecture ──────────────
    global_input_1 = keras.Input(shape=(301, 2), name="global_input_1")

    x = layers.Conv1D(32, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(global_input_1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)                    # 301 → 75

    x = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=5)(x)                    # 75  → 15

    x = layers.Conv1D(128, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    branch1_out = layers.GlobalAveragePooling1D()(x)           # → (128,)

    # ── Branch 2: same centroid sequence, 2-block architecture ──────────────
    global_input_2 = keras.Input(shape=(301, 2), name="global_input_2")

    y = layers.Conv1D(32, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(global_input_2)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.MaxPooling1D(pool_size=3)(y)                    # 301 → 100

    y = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    branch2_out = layers.GlobalAveragePooling1D()(y)           # → (64,)

    # ── Classification head ───────────────────────────────────────────────────
    combined = layers.Concatenate()([branch1_out, branch2_out])  # → (192,)

    z = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(combined)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(z)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(1, activation="sigmoid")(z)

    model = keras.Model(inputs=[global_input_1, global_input_2], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.5, smoothing=0.1),
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_fold(X_global_train, y_train, X_global_val, y_val,
               X_local_train, X_local_val):
    """
    Train one fold using model.fit() with proper Keras callbacks.

    Using model.fit() rather than a manual loop is critical: Keras's built-in
    training loop handles gradient accumulation, callback sequencing, and
    learning rate scheduling correctly. The previous manual loop had a broken
    plateau-detection condition that halved the LR nearly every epoch,
    causing the optimiser to stall before the model could properly learn.

    class_weight tells the loss function to treat each FP misclassification
    as twice as costly as a planet misclassification, directly counteracting
    the model's natural bias toward the morphologically cleaner planet class.

    The callback sequence is:
      1. LearningRateScheduler (warmup) — linear ramp from 1e-4 to 1e-3 over
         first 5 epochs, then defers to ReduceLROnPlateau
      2. ReduceLROnPlateau — halve LR when val AUC plateaus for 7 epochs,
         enabling fine-grained convergence without overshooting
      3. EarlyStopping — stop and restore best weights when val AUC fails
         to improve for 15 epochs, preventing overtraining

    Args:
        X_global_train: (N_train, 301, 3) training global input
        y_train:        (N_train,) labels
        X_global_val:   (N_val, 301, 3) validation global input
        y_val:          (N_val,) labels
        X_local_train:  (N_train, 61, 1) training local input
        X_local_val:    (N_val, 61, 1) validation local input
    """
    model     = build_model()
    train_gen = AugmentedSequence(
        X_global_train, X_local_train, y_train, BATCH_SIZE, shuffle=True
    )
    val_data  = (split_inputs(X_global_val, X_local_val), y_val)

    callbacks = [
        # Warm up learning rate from 1e-4 to 1e-3 over first 5 epochs
        LearningRateScheduler(warmup_schedule, verbose=0),
        # Halve LR when val AUC plateaus for 7 epochs, enabling fine convergence.
        ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=7,
                          mode="max", min_lr=1e-6, verbose=0),
        # Stop when val AUC fails to improve for 15 epochs; restore best weights.
        EarlyStopping(monitor="val_auc", patience=15, mode="max",
                      restore_best_weights=True, verbose=0),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=CLASS_WEIGHT,
        verbose=0
    )

    best_epoch = int(np.argmax(history.history["val_auc"])) + 1
    best_auc   = max(history.history["val_auc"])
    print(f"    Best epoch: {best_epoch}  |  Best val AUC: {best_auc:.4f}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — THRESHOLD OPTIMISATION + EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def find_best_threshold(y_true, y_prob):
    """
    Search for the sigmoid threshold that maximises F1-macro on the given set.
    The default of 0.5 assumes output probabilities are centred at 0.5,
    which is rarely true — especially when class weights shift the effective
    decision boundary. We search 99 candidates between 0.01 and 0.99 and pick
    the one that maximises F1-macro, treating both classes equally.

    IMPORTANT: For reported ensemble test metrics, this function is called on
    out-of-fold (OOF) CV predictions, NOT on test-set predictions. The returned
    threshold is then applied fixed to the test set, preventing evaluation leakage.
    """
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_true, (y_prob >= t).astype(int),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def evaluate_fold(model, X_global_val, X_local_val, y_val):
    """
    Compute all metrics for one validation fold with optimised threshold.

    Args:
        model:         trained Keras model
        X_global_val:  (N_val, 301, 3) validation global input
        X_local_val:   (N_val,  61, 1) validation local input
        y_val:         (N_val,) true labels

    Returns:
        metrics: dict of per-fold metrics
        y_prob:  raw sigmoid probabilities (for OOF threshold optimisation)
    """
    y_prob = model.predict(
        split_inputs(X_global_val, X_local_val), verbose=0, batch_size=64
    ).flatten()
    best_thresh, _ = find_best_threshold(y_val, y_prob)
    y_pred = (y_prob >= best_thresh).astype(int)
    print(f"    Optimal threshold: {best_thresh:.3f}")

    metrics = {
        "threshold":        best_thresh,
        "accuracy":         accuracy_score(y_val, y_pred),
        "f1_planet":        f1_score(y_val, y_pred, pos_label=1, zero_division=0),
        "f1_fp":            f1_score(y_val, y_pred, pos_label=0, zero_division=0),
        "f1_macro":         f1_score(y_val, y_pred, average="macro", zero_division=0),
        "f1_weighted":      f1_score(y_val, y_pred, average="weighted", zero_division=0),
        "precision_planet": precision_score(y_val, y_pred, pos_label=1, zero_division=0),
        "precision_fp":     precision_score(y_val, y_pred, pos_label=0, zero_division=0),
        "recall_planet":    recall_score(y_val, y_pred, pos_label=1, zero_division=0),
        "recall_fp":        recall_score(y_val, y_pred, pos_label=0, zero_division=0),
        "roc_auc":          roc_auc_score(y_val, y_prob),
        "confusion_matrix": confusion_matrix(y_val, y_pred),
    }
    # Return y_prob alongside metrics so callers can collect OOF predictions
    return metrics, y_prob


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(cm_avg, output_path):
    """
    Save a clean sklearn-style confusion matrix using matplotlib's standard
    'Purples' colormap on a plain white background. Each cell shows the
    normalised proportion (rows sum to 1 within each true class) and the
    approximate raw count averaged across folds.
    """
    cm_norm = cm_avg / cm_avg.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Purples", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    class_names = ["False Positive\n(EB/NEB)", "Planet\n(CP/KP/PC)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i,
                    f"{cm_norm[i,j]:.2%}\n(n≈{int(cm_avg[i,j])})",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(
        "Yet Another Exoplanet Classifier\n"
        f"Confusion Matrix | {N_FOLDS}-Fold CV Average",
        fontsize=11, pad=12
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {output_path}")


def save_scores(kepid_arr, kepoi_arr, fp_subtype_arr, y_true, y_score, out_path):
    """Save per-target held-out test scores to CSV for downstream analysis."""
    pd.DataFrame({
        "kepid":      kepid_arr,
        "kepoi_name": kepoi_arr,
        "label":      y_true,
        "fp_subtype": fp_subtype_arr,
        "score":      y_score,
    }).to_csv(out_path, index=False)
    print(f"  Test scores → {out_path}")


def save_metrics_report(fold_metrics, ensemble_metrics, n_total, output_path,
                        tag="", cv_threshold=None, test_oracle_threshold=None):
    """
    Write a structured text report with per-fold, CV-averaged, and test-set metrics.

    Args:
        fold_metrics:           list of per-fold metric dicts
        ensemble_metrics:       dict of ensemble test-set metrics
        n_total:                total training examples
        output_path:            output .txt file path
        tag:                    CSV tag (e.g. 'k3_psf0')
        cv_threshold:           OOF-derived threshold used for test metrics
        test_oracle_threshold:  threshold from test-set search (informational only)
    """
    metric_keys = [
        "threshold", "accuracy", "f1_planet", "f1_fp", "f1_macro",
        "f1_weighted", "precision_planet", "precision_fp",
        "recall_planet", "recall_fp", "roc_auc"
    ]
    sep = "=" * 67
    lines = [
        sep,
        f"KEPLER CENTROID RESOLUTION DEGRADATION STUDY — CNN RESULTS {tag}",
        f"Architecture  : Dual-Branch 1D CNN (channels-last)",
        f"Dataset       : {n_total} examples  |  {N_FOLDS}-fold KIC-grouped CV",
        f"Class weights : FP × {CLASS_WEIGHT[0]}  |  Planet × {CLASS_WEIGHT[1]}",
        f"Classes       : 0 = False Positive  |  1 = Confirmed Planet",
        sep, "",
        "PER-FOLD RESULTS", "-" * 67,
        f"{'Metric':<22}" + "".join(f"  Fold {k+1}" for k in range(N_FOLDS)),
        "-" * 67,
    ]
    for key in metric_keys:
        row = f"{key:<22}"
        for m in fold_metrics:
            row += f"  {m[key]:.4f}"
        lines.append(row)

    lines += ["", sep, "CROSS-VALIDATION SUMMARY  (mean ± std)", sep]
    for key in metric_keys:
        vals = [m[key] for m in fold_metrics]
        mean, std = np.mean(vals), np.std(vals)
        flag = "  ✓ ≥ 90%" if mean >= 0.90 else ""
        lines.append(f"  {key:<22}  {mean:.4f} ± {std:.4f}{flag}")

    cm_avg = np.mean([m["confusion_matrix"] for m in fold_metrics], axis=0)
    lines += [
        "", sep, "AVERAGED CONFUSION MATRIX (5-Fold CV)", sep,
        f"                  Predicted FP    Predicted Planet",
        f"  True FP           {cm_avg[0,0]:>8.1f}        {cm_avg[0,1]:>8.1f}",
        f"  True Planet       {cm_avg[1,0]:>8.1f}        {cm_avg[1,1]:>8.1f}",
    ]

    # Ensemble results with clear threshold labelling to prevent misinterpretation
    oof_thresh_str = (f"{cv_threshold:.3f}" if cv_threshold is not None
                      else f"{ensemble_metrics.get('threshold', 0.5):.3f}")
    oracle_thresh_str = (f"{test_oracle_threshold:.3f}"
                         if test_oracle_threshold is not None else "N/A")
    lines += [
        "", sep,
        f"ENSEMBLE TEST METRICS (threshold fixed from OOF CV: {oof_thresh_str})",
        sep,
        "Predictions: Average sigmoid probability from all 5 fold models",
        f"Test-set oracle threshold (informational only, not used): {oracle_thresh_str}",
        "-" * 67,
    ]
    for key in metric_keys:
        lines.append(f"  {key:<22}  {ensemble_metrics[key]:.4f}")

    cm_ensemble = ensemble_metrics["confusion_matrix"]
    lines += [
        "", "ENSEMBLE CONFUSION MATRIX", "-" * 67,
        f"                  Predicted FP    Predicted Planet",
        f"  True FP           {cm_ensemble[0,0]:>8.1f}        {cm_ensemble[0,1]:>8.1f}",
        f"  True Planet       {cm_ensemble[1,0]:>8.1f}        {cm_ensemble[1,1]:>8.1f}",
    ]

    lines += [
        "", sep, "INTERPRETATION GUIDE", sep,
        "  recall_fp     = fraction of true EBs correctly identified",
        "                  (critical: missing an EB = wasted follow-up resources)",
        "  recall_planet = fraction of true planets correctly identified",
        "                  (critical: missing a planet = lost discovery)",
        "  roc_auc       = discrimination ability independent of threshold",
        "                  (0.5 = random chance, 1.0 = perfect)",
        "  f1_macro      = unweighted mean of per-class F1",
        "                  (best single summary for balanced classes)",
        "  threshold     = OOF-optimised sigmoid cut-point (not test-set optimised)",
        "  ensemble      = average probability across all 5 fold models",
    ]

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  Metrics report  → {output_path}")
    return report


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ABLATION STUDY (flux-only and centroid-only variants)
# ═════════════════════════════════════════════════════════════════════════════

def _run_ablation_study(
    X_global_cv: np.ndarray,
    X_local_cv: np.ndarray,
    y_cv: np.ndarray,
    groups_cv: np.ndarray,
    X_global_test: np.ndarray,
    X_local_test: np.ndarray,
    y_test: np.ndarray,
    kepid_test: np.ndarray,
    kepoi_test: np.ndarray,
    fp_subtype_test: np.ndarray,
    tag: str,
) -> None:
    """
    Train flux-only and centroid-only CNN ablation variants using the same
    StratifiedGroupKFold splits (same seed) as the primary combined model.

    This is a self-contained, simplified training loop — no OOF threshold
    optimisation, no full metrics report. It exists solely to produce per-target
    test scores for the ablation panel of the AUC-vs-resolution figure.

    Code duplication relative to the main combined-model loop is intentional:
    refactoring the combined loop to share code would risk touching the primary
    study path, potentially introducing subtle behavioural drift in the most
    important model. Ablation models are secondary.

    Only runs for psf=0 CSVs (indicated by the caller checking tag.endswith('psf0')).

    Outputs:
        results_resolution/scores_{tag}_flux_only.csv
        results_resolution/scores_{tag}_centroid_only.csv
    """
    variants = [
        ("flux_only",     build_model_flux_only,    split_inputs_flux_only,    "[flux-only]"),
        ("centroid_only", build_model_centroid_only, split_inputs_centroid_only, "[centroid-only]"),
    ]

    # Use the same fold structure as the combined model (same seed, same data)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for variant_key, model_builder, input_splitter, print_label in variants:
        print(f"\n  Ablation {print_label} [{tag}] — training {N_FOLDS} folds ...")

        fold_models_ab = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            sgkf.split(X_global_cv, y_cv, groups=groups_cv)
        ):
            # Leakage check — same guarantee as the combined loop
            train_kps_ab = set(groups_cv[train_idx])
            val_kps_ab   = set(groups_cv[val_idx])
            assert train_kps_ab.isdisjoint(val_kps_ab), \
                f"Ablation {print_label} fold {fold_idx+1}: kepid leakage"

            model_ab = model_builder()

            # Build augmented sequence with overridden input splitter.
            # X_global is augmented; X_local is passed through unchanged.
            class _AblationSeq(Sequence):
                """Augmented sequence for ablation with overridden input split."""
                def __init__(self, Xg, Xl, y, batch_size, shuffle, splitter):
                    self.Xg = Xg; self.Xl = Xl; self.y = y
                    self.batch_size = batch_size
                    self.shuffle = shuffle; self.splitter = splitter
                    self.indices = np.arange(len(Xg)); self.on_epoch_end()

                def __len__(self):
                    return len(self.Xg) // self.batch_size

                def on_epoch_end(self):
                    if self.shuffle:
                        np.random.shuffle(self.indices)

                def __getitem__(self, idx):
                    bidx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                    Xg_b = self.Xg[bidx].copy()   # augment global branch
                    Xl_b = self.Xl[bidx]           # local branch unchanged
                    yb   = self.y[bidx]
                    for i in range(len(Xg_b)):
                        shift = np.random.randint(-20, 21)
                        Xg_b[i] = np.roll(Xg_b[i], shift, axis=0)
                        Xg_b[i, :, 0] += np.random.normal(
                            0, 0.02, NUM_BINS).astype(np.float32)
                        Xg_b[i, :, 0] *= np.random.uniform(0.95, 1.05)
                    return self.splitter(Xg_b, Xl_b), yb

            train_gen_ab = _AblationSeq(
                X_global_cv[train_idx], X_local_cv[train_idx],
                y_cv[train_idx], BATCH_SIZE, shuffle=True,
                splitter=input_splitter,
            )
            val_data_ab = (
                input_splitter(X_global_cv[val_idx], X_local_cv[val_idx]),
                y_cv[val_idx],
            )

            callbacks_ab = [
                LearningRateScheduler(warmup_schedule, verbose=0),
                ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=7,
                                  mode="max", min_lr=1e-6, verbose=0),
                EarlyStopping(monitor="val_auc", patience=15, mode="max",
                              restore_best_weights=True, verbose=0),
            ]

            history_ab = model_ab.fit(
                train_gen_ab,
                validation_data=val_data_ab,
                epochs=EPOCHS,
                callbacks=callbacks_ab,
                class_weight=CLASS_WEIGHT,
                verbose=0,
            )
            best_auc_ab = max(history_ab.history["val_auc"])
            print(f"    {print_label} Fold {fold_idx+1}: best val AUC = {best_auc_ab:.4f}")
            fold_models_ab.append(model_ab)

        # Ensemble test scores
        test_probs_ab = np.zeros(len(X_global_test))
        for m_ab in fold_models_ab:
            test_probs_ab += m_ab.predict(
                input_splitter(X_global_test, X_local_test), verbose=0, batch_size=64
            ).flatten()
            tf.keras.backend.clear_session()
        test_probs_ab /= N_FOLDS

        test_auc_ab = roc_auc_score(y_test, test_probs_ab)
        print(f"  {print_label} [{tag}] Test AUC: {test_auc_ab:.4f}")

        scores_path_ab = os.path.join(OUTPUT_DIR, f"scores_{tag}_{variant_key}.csv")
        save_scores(kepid_test, kepoi_test, fp_subtype_test,
                    y_test, test_probs_ab, scores_path_ab)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def _train_and_eval_one_csv(csv_path: str) -> None:
    """
    Train the combined dual-branch CNN for one per-k training CSV and save
    scores + metrics. Also triggers the ablation study for psf=0 CSVs.

    Ensemble threshold is derived from out-of-fold (OOF) CV predictions to
    prevent evaluation leakage — the test set is never used for threshold
    selection.
    """
    # Parse k and psf from filename, e.g. kepler_training_data_k3_psf1.csv
    basename = os.path.basename(csv_path)
    tag = basename.replace("kepler_training_data_", "").replace(".csv", "")

    X_global, X_local, y, kepid_arr, fp_subtype_arr, n_total = load_and_preprocess(csv_path)

    # ── Held-out test set: 20% of unique kepids, stratified by label ──────────
    unique_kepids = np.unique(kepid_arr)
    # Build per-kepid label (majority vote to handle multi-KOI kepids)
    kepid_label = {}
    for kp in unique_kepids:
        mask = kepid_arr == kp
        kepid_label[kp] = int(np.bincount(y[mask]).argmax())
    kepid_labels_arr = np.array([kepid_label[kp] for kp in unique_kepids])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=RANDOM_SEED)
    cv_kepids_idx, test_kepids_idx = next(sss.split(unique_kepids, kepid_labels_arr))
    cv_kepids   = set(unique_kepids[cv_kepids_idx])
    test_kepids = set(unique_kepids[test_kepids_idx])

    # Assert no leakage — a kepid must not appear on both sides
    assert cv_kepids.isdisjoint(test_kepids), \
        f"kepid leakage: {cv_kepids & test_kepids}"

    cv_mask   = np.array([kp in cv_kepids   for kp in kepid_arr])
    test_mask = np.array([kp in test_kepids for kp in kepid_arr])

    X_global_cv   = X_global[cv_mask]
    X_local_cv    = X_local[cv_mask]
    y_cv          = y[cv_mask]
    groups_cv     = kepid_arr[cv_mask]

    X_global_test = X_global[test_mask]
    X_local_test  = X_local[test_mask]
    y_test        = y[test_mask]
    kepid_test    = kepid_arr[test_mask]
    fp_subtype_test = fp_subtype_arr[test_mask]

    print(f"\nCV set:   {cv_mask.sum()} examples, {len(cv_kepids)} unique KIC")
    print(f"Test set: {test_mask.sum()} examples, {len(test_kepids)} unique KIC")

    # ── KIC-grouped 5-fold CV ─────────────────────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []
    fold_models  = []
    all_cms      = []

    # Collect OOF predictions for threshold optimisation (prevent leakage)
    oof_true  = []
    oof_probs = []

    print(f"\nBeginning {N_FOLDS}-fold KIC-grouped CV  [{tag}]")
    print(f"Class weights: {CLASS_WEIGHT}")
    print("-" * 67)

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(X_global_cv, y_cv, groups=groups_cv)
    ):
        # Runtime leakage check: no kepid in both train and val
        train_kepids = set(groups_cv[train_idx])
        val_kepids   = set(groups_cv[val_idx])
        assert train_kepids.isdisjoint(val_kepids), \
            f"Fold {fold_idx+1}: kepid leakage into validation set"

        print(f"\nFold {fold_idx + 1}/{N_FOLDS}  "
              f"(train: {len(train_idx)}  val: {len(val_idx)}  "
              f"val_kic: {len(val_kepids)})")

        model = train_fold(
            X_global_cv[train_idx], y_cv[train_idx],
            X_global_cv[val_idx],   y_cv[val_idx],
            X_local_cv[train_idx],  X_local_cv[val_idx],
        )
        # evaluate_fold returns (metrics_dict, y_prob) — y_prob for OOF collection
        metrics, y_prob_val = evaluate_fold(
            model, X_global_cv[val_idx], X_local_cv[val_idx], y_cv[val_idx]
        )
        fold_metrics.append(metrics)
        fold_models.append(model)
        all_cms.append(metrics["confusion_matrix"])

        # Accumulate OOF predictions for post-CV threshold optimisation
        oof_true.append(y_cv[val_idx])
        oof_probs.append(y_prob_val)

        print(f"    Accuracy: {metrics['accuracy']:.4f}  |  "
              f"AUC-ROC: {metrics['roc_auc']:.4f}  |  "
              f"F1-macro: {metrics['f1_macro']:.4f}")

    # ── OOF threshold (no test-set leakage) ──────────────────────────────────
    oof_y = np.concatenate(oof_true)
    oof_p = np.concatenate(oof_probs)
    # Threshold derived entirely from CV out-of-fold predictions — test set never seen here
    cv_threshold, _ = find_best_threshold(oof_y, oof_p)
    print(f"\nOOF threshold (from {len(oof_y)} OOF predictions): {cv_threshold:.3f}")

    # ── Ensemble inference on held-out test set ───────────────────────────────
    print(f"\nComputing ensemble scores on held-out test set ({len(X_global_test)} examples)...")
    test_probs = np.zeros(len(X_global_test))
    for fold_model in fold_models:
        test_probs += fold_model.predict(
            split_inputs(X_global_test, X_local_test), verbose=0, batch_size=64
        ).flatten()
        tf.keras.backend.clear_session()
    test_probs /= N_FOLDS

    # Read kepoi_name from CSV for the test-set rows
    df_full = pd.read_csv(csv_path, usecols=["kepid", "kepoi_name"])
    kepoi_map = df_full.set_index("kepid")["kepoi_name"].to_dict()
    kepoi_test = np.array([str(kepoi_map.get(kp, "")) for kp in kepid_test])

    # Save raw per-target scores for downstream baselines + stats_ml.py
    scores_path = os.path.join(OUTPUT_DIR, f"scores_{tag}.csv")
    save_scores(kepid_test, kepoi_test, fp_subtype_test, y_test, test_probs, scores_path)

    # Apply the OOF-derived threshold to the test set for reported metrics.
    # This threshold was never optimised on the test set — no leakage.
    test_pred = (test_probs >= cv_threshold).astype(int)

    # Test-set oracle threshold: found by searching the test set itself.
    # This is informational ONLY — it shows the ceiling for a "cheating" threshold.
    # It is NOT used for any reported metric.
    test_thresh_oracle, _ = find_best_threshold(y_test, test_probs)

    ensemble_metrics = {
        "threshold":        cv_threshold,           # OOF threshold (reported)
        "accuracy":         accuracy_score(y_test, test_pred),
        "f1_planet":        f1_score(y_test, test_pred, pos_label=1, zero_division=0),
        "f1_fp":            f1_score(y_test, test_pred, pos_label=0, zero_division=0),
        "f1_macro":         f1_score(y_test, test_pred, average="macro", zero_division=0),
        "f1_weighted":      f1_score(y_test, test_pred, average="weighted", zero_division=0),
        "precision_planet": precision_score(y_test, test_pred, pos_label=1, zero_division=0),
        "precision_fp":     precision_score(y_test, test_pred, pos_label=0, zero_division=0),
        "recall_planet":    recall_score(y_test, test_pred, pos_label=1, zero_division=0),
        "recall_fp":        recall_score(y_test, test_pred, pos_label=0, zero_division=0),
        "roc_auc":          roc_auc_score(y_test, test_probs),
        "confusion_matrix": confusion_matrix(y_test, test_pred),
    }
    print(f"  Test AUC-ROC: {ensemble_metrics['roc_auc']:.4f}  |  "
          f"F1-macro: {ensemble_metrics['f1_macro']:.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_confusion_matrix(
        np.mean(all_cms, axis=0),
        os.path.join(OUTPUT_DIR, f"confusion_matrix_{tag}.png"),
    )
    save_metrics_report(
        fold_metrics, ensemble_metrics, n_total,
        os.path.join(OUTPUT_DIR, f"metrics_{tag}.txt"),
        tag=tag,
        cv_threshold=cv_threshold,
        test_oracle_threshold=test_thresh_oracle,
    )

    print(f"\nCV summary [{tag}]:")
    for key in ["accuracy", "roc_auc", "f1_macro", "recall_fp", "recall_planet"]:
        vals = [m[key] for m in fold_metrics]
        print(f"  {key:<22}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ── Ablation study (psf=0 CSVs only) ─────────────────────────────────────
    # The ablation is a secondary analysis; only run for the rebinning-only
    # (psf=0) result set to avoid 2x the compute with minimal additional insight.
    if tag.endswith("psf0"):
        print(f"\n  Running ablation study for {tag} ...")
        _run_ablation_study(
            X_global_cv, X_local_cv, y_cv, groups_cv,
            X_global_test, X_local_test, y_test,
            kepid_test, kepoi_test, fp_subtype_test,
            tag,
        )


def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Train dual-branch CNN for each per-k training CSV"
    )
    parser.add_argument(
        "csv_files", nargs="*",
        help="Training CSV paths. If omitted, scans for kepler_training_data_k*.csv"
    )
    parser.add_argument(
        "--results-dir", default=OUTPUT_DIR,
        help="Directory for scores, metrics, and figures (default: results_resolution)"
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.results_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.csv_files:
        csv_list = args.csv_files
    else:
        csv_list = sorted(glob.glob("kepler_training_data_k*_psf*.csv"))
        if not csv_list:
            print("No training CSVs found. Run: python getInputData.py --phase csv-only")
            return

    print("\n" + "=" * 67)
    print("  Kepler Centroid Degradation Study — CNN Training")
    print("=" * 67)
    print(f"  Training CSV files: {csv_list}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Architecture: Dual-Branch 1D CNN (FROZEN)")
    build_model().summary(line_length=67)

    for csv_path in csv_list:
        print("\n" + "=" * 67)
        print(f"  Processing: {csv_path}")
        print("=" * 67)
        _train_and_eval_one_csv(csv_path)

    print("\n" + "=" * 67)
    print(f"All CSVs processed. Outputs in {OUTPUT_DIR}/")
    print("Next step: python stats_ml.py")


if __name__ == "__main__":
    main()
