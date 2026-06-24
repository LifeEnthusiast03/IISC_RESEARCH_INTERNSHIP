"""
build_hybrid_dataset.py — Stage 1B Hybrid Classifier: Dataset Builder
======================================================================

PURPOSE:
  Constructs a balanced, stratified train/val/test dataset for the Stage 1B
  hybrid multi-class classifier.  The hybrid classifier catches attack types
  that the Stage 1 autoencoder cannot reliably separate:

  Original 5 weak classes (low reconstruction error, AE-evading):
      FTP-Patator  |  Botnet_ARES  |  SSH-Patator
      Web_Brute_Force  |  Web_XSS

  v2 extension — DoS_Hulk as 7th class (Step 1B added):
      Diagnosis (diagnose_dos_hulk.py) showed 143K / 297K DoS_Hulk flows
      fall FAR BELOW the autoencoder threshold with a clear feature
      separation from detected flows.  These missed flows are added here
      as positive training examples for a new 'DoS_Hulk' class so the
      hybrid classifier can catch what the AE misses.
      Only the MISSED rows (AE error ≤ threshold) are included — detected
      rows are already handled by the autoencoder and must not be
      double-counted.

INPUTS (data/processed/):
  - X_attacks.npy        — MinMax-scaled attack features, 115 columns
  - y_attacks_str.npy    — string attack-type labels, aligned row-wise
  - errors_attack.npy    — per-sample AE reconstruction error (aligned)
  - X_train_benign.npy   — MinMax-scaled benign features (same scaler)
  models/threshold.json  — current autoencoder anomaly threshold

OUTPUTS (data/processed/):
  - X_train_hybrid.npy, y_train_hybrid.npy   (70 %)
  - X_val_hybrid.npy,   y_val_hybrid.npy     (15 %)
  - X_test_hybrid.npy,  y_test_hybrid.npy    (15 %)
  - hybrid_label_map.json  — {class_name: int_index} for the 7 classes

SPLIT STRATEGY:
  70 / 15 / 15  with stratify=y  so the minority class (Web_XSS, ~1,357
  samples total) is proportionally represented in every split.
  Class imbalance (DoS_Hulk ~143K vs Web_XSS ~1,357) is handled downstream
  by compute_sample_weight('balanced') in train_hybrid_classifier.py.

USAGE (from project root):
    python training/build_hybrid_dataset.py
"""

import os
import sys
import json
import logging

import numpy as np
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Path anchoring — same pattern as train_autoencoder.py
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../training/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
# The 5 attack types the autoencoder structurally cannot detect
WEAK_ATTACK_CLASSES = [
    "FTP-Patator",
    "Botnet_ARES",
    "SSH-Patator",
    "Web_Brute_Force",
    "Web_XSS",
]

# ── DoS_Hulk extension (v2) ───────────────────────────────────────────
# Diagnosis showed ~143K DoS_Hulk flows evade the AE (error ≤ threshold).
# Include ONLY those missed flows as positive training examples for the
# hybrid classifier.  Set to False to revert to the original 6-class model.
INCLUDE_DOS_HULK_MISSED = True
DOS_HULK_CLASS          = "DoS_Hulk"
# Cap: None = use all missed rows (~143K); set an int to sub-sample.
# sample_weight='balanced' in the trainer handles the resulting imbalance.
DOS_HULK_MAX_SAMPLES    = None

BENIGN_LABEL    = "Benign"
BENIGN_SAMPLE_N = 25_000      # target benign rows ≈ combined 5-class attack count
RANDOM_STATE    = 42
VAL_FRAC        = 0.15        # fraction of full dataset reserved for val
TEST_FRAC       = 0.15        # fraction of full dataset reserved for test
# Train frac = 1 - VAL_FRAC - TEST_FRAC = 0.70  (implicit)

# ─────────────────────────────────────────────
# Logging  (console + UTF-8 safe, mirrors train_autoencoder.py style)
# ─────────────────────────────────────────────
_stream_handler = logging.StreamHandler()
_stream_handler.stream = open(
    _stream_handler.stream.fileno(),
    mode="w", encoding="utf-8", closefd=False, buffering=1,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_stream_handler],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load attack features and filter to 5 weak classes
# ═══════════════════════════════════════════════════════════
def step1_load_attack_subset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load X_attacks.npy + y_attacks_str.npy + errors_attack.npy.
    Returns the full arrays (needed by Step 1B) AND the 5-class subset.

    Returns
    -------
    X_all     : np.ndarray  shape (N_all, 115)   — full attack features
    y_all     : np.ndarray  shape (N_all,)        — full string labels
    X_attack5 : np.ndarray  shape (N_5class, 115) — 5 weak classes only
    y_attack5 : np.ndarray  shape (N_5class,)     — 5 weak class labels
    errors_all: np.ndarray  shape (N_all,)         — AE reconstruction errors
    """
    x_path = os.path.join(DATA_DIR, "X_attacks.npy")
    y_path = os.path.join(DATA_DIR, "y_attacks_str.npy")
    e_path = os.path.join(DATA_DIR, "errors_attack.npy")

    log.info(f"  Loading X_attacks from      : {x_path}")
    X_all = np.load(x_path).astype(np.float32)
    log.info(f"  Loading y_attacks_str from  : {y_path}")
    y_all = np.load(y_path, allow_pickle=True)
    log.info(f"  Loading errors_attack from  : {e_path}")
    errors_all = np.load(e_path)

    log.info(f"  Full attack set  shape: {X_all.shape}")
    log.info(f"  Unique attack types ({len(np.unique(y_all))}): {sorted(np.unique(y_all).tolist())}")

    # Filter to 5 weak classes
    mask = np.isin(y_all, WEAK_ATTACK_CLASSES)
    X_attack5 = X_all[mask]
    y_attack5  = y_all[mask]

    log.info(f"\n  After filtering to 5 weak classes: {X_attack5.shape[0]:,} rows")
    for cls in WEAK_ATTACK_CLASSES:
        n = int((y_attack5 == cls).sum())
        log.info(f"    {cls:<22s}: {n:>6,} rows")

    return X_all, y_all, errors_all, X_attack5, y_attack5


# ═══════════════════════════════════════════════════════════
# STEP 1B — Load DoS_Hulk missed flows (AE error ≤ threshold)
# ═══════════════════════════════════════════════════════════
def step1b_load_dos_hulk_missed(
    X_all: np.ndarray,
    y_all: np.ndarray,
    errors_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract DoS_Hulk rows whose AE reconstruction error is AT OR BELOW
    the current anomaly threshold — i.e., the flows the autoencoder
    already MISSES and that the hybrid classifier must now catch.

    Optionally sub-samples to DOS_HULK_MAX_SAMPLES (default: use all).

    Returns
    -------
    X_hulk_missed : np.ndarray  shape (N_missed, 115)
    y_hulk_missed : np.ndarray  shape (N_missed,)  — all 'DoS_Hulk'
    """
    # Load threshold
    thresh_path = os.path.join(MODEL_DIR, "threshold.json")
    with open(thresh_path, "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    log.info(f"  AE threshold (models/threshold.json) : {threshold:.10f}")

    # Filter: DoS_Hulk AND missed by AE
    mask_hulk   = (y_all == DOS_HULK_CLASS)
    mask_missed = (errors_all <= threshold)
    mask        = mask_hulk & mask_missed

    n_hulk_total  = int(mask_hulk.sum())
    n_hulk_missed = int(mask.sum())
    n_hulk_det    = n_hulk_total - n_hulk_missed

    log.info(f"  DoS_Hulk total                       : {n_hulk_total:>8,}")
    log.info(f"  DoS_Hulk detected by AE (excluded)   : {n_hulk_det:>8,}  "
             f"({100.*n_hulk_det/n_hulk_total:.1f}%)")
    log.info(f"  DoS_Hulk missed by AE  (included)    : {n_hulk_missed:>8,}  "
             f"({100.*n_hulk_missed/n_hulk_total:.1f}%)")

    X_hulk_missed = X_all[mask]
    y_hulk_missed = y_all[mask]   # all == 'DoS_Hulk'

    # Optional cap
    if DOS_HULK_MAX_SAMPLES is not None and n_hulk_missed > DOS_HULK_MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n_hulk_missed, size=DOS_HULK_MAX_SAMPLES, replace=False)
        X_hulk_missed = X_hulk_missed[idx]
        y_hulk_missed = y_hulk_missed[idx]
        log.info(f"  Sub-sampled to DOS_HULK_MAX_SAMPLES : {DOS_HULK_MAX_SAMPLES:,}")

    return X_hulk_missed, y_hulk_missed


# ═══════════════════════════════════════════════════════════
# STEP 2 — Sample benign rows
# ═══════════════════════════════════════════════════════════
def step2_sample_benign(n_attack: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load X_train_benign.npy and randomly sample min(BENIGN_SAMPLE_N, n_attack)
    rows ≈ the combined attack count so the binary benign/attack balance is
    roughly 1:1 before stratified splitting.

    Returns
    -------
    X_benign : np.ndarray  shape (N_benign, 115)
    y_benign : np.ndarray  shape (N_benign,)  — all "Benign"
    """
    benign_path = os.path.join(DATA_DIR, "X_train_benign.npy")
    log.info(f"\n  Loading X_train_benign from : {benign_path}")
    X_train_benign = np.load(benign_path).astype(np.float32)
    log.info(f"  Full benign train shape     : {X_train_benign.shape}")

    # Sample roughly the same number of rows as the 5 weak attack classes combined
    n_sample = min(BENIGN_SAMPLE_N, len(X_train_benign), n_attack)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_train_benign), size=n_sample, replace=False)
    X_benign = X_train_benign[idx]
    y_benign  = np.full(n_sample, BENIGN_LABEL, dtype=object)

    log.info(f"  Sampled {n_sample:,} benign rows  (random_state={RANDOM_STATE})")
    return X_benign, y_benign


# ═══════════════════════════════════════════════════════════
# STEP 3 — Combine, encode labels, stratified split
# ═══════════════════════════════════════════════════════════
def step3_combine_and_split(
    X_attack5: np.ndarray,
    y_attack5: np.ndarray,
    X_benign: np.ndarray,
    y_benign: np.ndarray,
    X_hulk_missed: np.ndarray | None = None,
    y_hulk_missed: np.ndarray | None = None,
) -> dict:
    """
    Concatenate all arrays (5 weak attack classes + Benign + optional
    DoS_Hulk missed rows), build an integer label encoder, and perform
    a stratified 70/15/15 train/val/test split.

    Returns
    -------
    splits : dict with keys:
        X_train, X_val, X_test, y_train, y_val, y_test  (string labels)
        label_map  : {class_name: int_index}
    """
    arrays_X = [X_attack5, X_benign]
    arrays_y = [y_attack5, y_benign]
    if X_hulk_missed is not None and len(X_hulk_missed) > 0:
        arrays_X.append(X_hulk_missed)
        arrays_y.append(y_hulk_missed)
        log.info(f"  DoS_Hulk missed rows appended: {len(X_hulk_missed):,}")

    X = np.concatenate(arrays_X, axis=0)
    y = np.concatenate(arrays_y, axis=0)

    log.info(f"\n  Combined dataset  X: {X.shape}   y: {y.shape}")
    log.info(f"  Class distribution (pre-split):")
    for cls in sorted(np.unique(y).tolist()):
        n = int((y == cls).sum())
        pct = 100.0 * n / len(y)
        log.info(f"    {cls:<22s}: {n:>7,}  ({pct:.2f}%)")

    # Build label map: sorted classes → 0-indexed integers
    classes   = sorted(np.unique(y).tolist())
    label_map = {cls: i for i, cls in enumerate(classes)}
    log.info(f"\n  Label map: {label_map}")

    # ── First split: train vs (val+test) ──────────────────────────────
    val_test_frac = VAL_FRAC + TEST_FRAC      # e.g. 0.30
    X_train, X_valtest, y_train, y_valtest = train_test_split(
        X, y,
        test_size=val_test_frac,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ── Second split: val vs test (equal halves of the held-out 30%) ──
    val_share_of_valtest = VAL_FRAC / val_test_frac   # 0.15 / 0.30 = 0.50
    X_val, X_test, y_val, y_test = train_test_split(
        X_valtest, y_valtest,
        test_size=1.0 - val_share_of_valtest,   # 0.50 → test gets half
        stratify=y_valtest,
        random_state=RANDOM_STATE,
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "label_map": label_map,
        "classes":   classes,
    }


# ═══════════════════════════════════════════════════════════
# STEP 4 — Save arrays and label map
# ═══════════════════════════════════════════════════════════
def step4_save(splits: dict):
    """
    Save the six .npy arrays (string labels kept as-is) and the label
    map JSON to data/processed/.
    """
    save_pairs = [
        ("X_train_hybrid.npy", splits["X_train"]),
        ("X_val_hybrid.npy",   splits["X_val"]),
        ("X_test_hybrid.npy",  splits["X_test"]),
        ("y_train_hybrid.npy", splits["y_train"]),
        ("y_val_hybrid.npy",   splits["y_val"]),
        ("y_test_hybrid.npy",  splits["y_test"]),
    ]
    for fname, arr in save_pairs:
        path = os.path.join(DATA_DIR, fname)
        np.save(path, arr)
        log.info(f"  Saved → {path}  shape={arr.shape}")

    label_map_path = os.path.join(DATA_DIR, "hybrid_label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(splits["label_map"], f, indent=2)
    log.info(f"  Saved → {label_map_path}")


# ═══════════════════════════════════════════════════════════
# STEP 5 — Log per-split class distribution
# ═══════════════════════════════════════════════════════════
def step5_log_distributions(splits: dict):
    """Print a formatted class distribution table for each split."""
    W = 70
    for split_name, y_arr in [
        ("TRAIN",       splits["y_train"]),
        ("VAL",         splits["y_val"]),
        ("TEST",        splits["y_test"]),
    ]:
        print(f"\n{'─'*W}")
        print(f"  Split: {split_name}  ({len(y_arr):,} rows)")
        print(f"{'─'*W}")
        print(f"  {'Class':22s}  {'Count':>8}  {'Pct':>7}")
        print(f"  {'─'*22}  {'─'*8}  {'─'*7}")

        for cls in splits["classes"]:
            n   = int((y_arr == cls).sum())
            pct = 100.0 * n / len(y_arr)
            print(f"  {cls:<22s}  {n:>8,}  {pct:>6.2f}%")

        print(f"  {'─'*22}  {'─'*8}  {'─'*7}")
        print(f"  {'TOTAL':22s}  {len(y_arr):>8,}  {100.0:>6.2f}%")
        print(f"{'─'*W}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    n_classes = 6 + (1 if INCLUDE_DOS_HULK_MISSED else 0)
    print("\n" + "=" * 70)
    print("  BUILD HYBRID DATASET — build_hybrid_dataset.py")
    print(f"  Stage 1B: {n_classes}-class dataset  (70/15/15 stratified split)")
    if INCLUDE_DOS_HULK_MISSED:
        print("  DoS_Hulk v2 extension: ENABLED  (AE-missed flows as 7th class)")
    print("=" * 70)
    log.info(f"  Data dir  : {DATA_DIR}")
    log.info(f"  Model dir : {MODEL_DIR}")
    log.info(f"  Weak classes targeted        : {WEAK_ATTACK_CLASSES}")
    log.info(f"  DoS_Hulk missed rows included: {INCLUDE_DOS_HULK_MISSED}")

    # ── STEP 1 ────────────────────────────────────────────────────────
    log.info("\n[STEP 1] Loading attack data and filtering to 5 weak classes...")
    X_all, y_all, errors_all, X_attack5, y_attack5 = step1_load_attack_subset()
    n_attack5 = len(X_attack5)

    # ── STEP 1B ───────────────────────────────────────────────────────
    X_hulk_missed, y_hulk_missed = None, None
    if INCLUDE_DOS_HULK_MISSED:
        log.info("\n[STEP 1B] Extracting DoS_Hulk missed flows (AE error ≤ threshold)...")
        X_hulk_missed, y_hulk_missed = step1b_load_dos_hulk_missed(
            X_all, y_all, errors_all
        )
        log.info(f"  DoS_Hulk missed rows to include: {len(X_hulk_missed):,}")
    else:
        log.info("\n[STEP 1B] SKIPPED (INCLUDE_DOS_HULK_MISSED = False)")

    # Free the full attack arrays — no longer needed
    del X_all, errors_all

    # ── STEP 2 ────────────────────────────────────────────────────────
    log.info("\n[STEP 2] Sampling benign rows from X_train_benign.npy...")
    X_benign, y_benign = step2_sample_benign(n_attack5)

    # ── STEP 3 ────────────────────────────────────────────────────────
    log.info("\n[STEP 3] Combining datasets and performing stratified split...")
    splits = step3_combine_and_split(
        X_attack5, y_attack5,
        X_benign,  y_benign,
        X_hulk_missed, y_hulk_missed,
    )

    # ── STEP 4 ────────────────────────────────────────────────────────
    log.info("\n[STEP 4] Saving arrays and label map to data/processed/...")
    step4_save(splits)

    # ── STEP 5 ────────────────────────────────────────────────────────
    log.info("\n[STEP 5] Per-split class distribution:")
    step5_log_distributions(splits)

    # ── Summary ───────────────────────────────────────────────────────
    total = len(splits['y_train']) + len(splits['y_val']) + len(splits['y_test'])
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total samples    : {total:,}")
    print(f"  Train / Val / Test: "
          f"{len(splits['y_train']):,}  /  "
          f"{len(splits['y_val']):,}  /  "
          f"{len(splits['y_test']):,}")
    print(f"  Classes ({len(splits['classes'])}): {splits['classes']}")
    if INCLUDE_DOS_HULK_MISSED:
        hulk_n = int((np.concatenate([splits['y_train'],
                                       splits['y_val'],
                                       splits['y_test']]) == DOS_HULK_CLASS).sum())
        print(f"  DoS_Hulk (missed) rows       : {hulk_n:,}  "
              f"(AE error ≤ threshold — these are the new positive examples)")
    print(f"\n  Outputs saved to : {DATA_DIR}")
    print(f"    X_train_hybrid.npy, y_train_hybrid.npy")
    print(f"    X_val_hybrid.npy,   y_val_hybrid.npy")
    print(f"    X_test_hybrid.npy,  y_test_hybrid.npy")
    print(f"    hybrid_label_map.json")
    print(f"\n  Next step: python training/train_hybrid_classifier.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
