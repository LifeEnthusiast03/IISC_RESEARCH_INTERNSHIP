"""
build_attack_type_dataset.py — Stage 2 Data Preparation (Attack-Type NN)
=========================================================================

PURPOSE:
  Prepares a stratified, class-weighted dataset for training the Attack-Type
  Neural Network (Stage 2). This NN classifies WHICH of the CICIDS2017 attack
  types a flow belongs to, given that Stage 1 (Autoencoder + Hybrid Classifier)
  has already flagged it as anomalous.

  This is deliberately BROADER in scope than the hybrid classifier (Stage 1B):
  it covers ALL attack types present in X_attacks.npy, not just the 5 AE-weak
  classes.  A broad probability vector over attack types gives the downstream
  DQN agent (Stage 3) much richer state information for choosing the right
  remediation action.

EXCLUSIONS (documented, intentional):
  - Heartbleed       (n=12) — statistically meaningless for supervised training
  - Web_SQL_Injection (n=24) — too few samples to learn a reliable decision boundary
  These two classes will always be marked "unclassified" by the Attack-Type NN.
  They remain detectable via the autoencoder's reconstruction error signal.

INPUTS (data/processed/):
  - X_attacks.npy       (600,141 × 115) — MinMax-scaled attack flow features
  - y_attacks_str.npy   (600,141,)      — string attack-type labels

OUTPUTS (data/processed/):
  - attack_type_label_map.json              # int → class-name mapping (saved to both
  - (also written to models/)              #   data/processed/ AND models/ for runtime)
  - X_train_attacktype.npy  (70% stratified)
  - X_val_attacktype.npy    (15% stratified)
  - X_test_attacktype.npy   (15% stratified)
  - y_train_attacktype.npy
  - y_val_attacktype.npy
  - y_test_attacktype.npy
  - sample_weights_train_attacktype.npy    # compute_sample_weight('balanced')

USAGE (from project root):
    python data_preparation/build_attack_type_dataset.py
"""

import os
import json
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────────────────────────────────────
# Path anchoring — same pattern as existing scripts
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../data_preparation/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Exclusion list — classes with too few samples
# ─────────────────────────────────────────────
EXCLUDE_CLASSES = {
    "Heartbleed",         # n=12  — statistically meaningless
    "Web_SQL_Injection",  # n=24  — too few to learn a decision boundary
}

# ─────────────────────────────────────────────
# Split configuration
# ─────────────────────────────────────────────
VAL_FRAC  = 0.15   # 15% of total → validation
TEST_FRAC = 0.15   # 15% of total → test  (70% remainder → train)
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# Logging setup — console + file, UTF-8 safe on Windows
# ─────────────────────────────────────────────
LOG_PATH = os.path.join(DATA_DIR, "build_attack_type_dataset.log")

_stream_handler = logging.StreamHandler()
_stream_handler.stream = open(
    _stream_handler.stream.fileno(),
    mode="w", encoding="utf-8", closefd=False, buffering=1
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load raw attack arrays
# ═══════════════════════════════════════════════════════════
def step1_load_raw(data_dir: str):
    """Load X_attacks.npy and y_attacks_str.npy from data/processed/."""
    log.info("[STEP 1] Loading X_attacks.npy and y_attacks_str.npy ...")
    X = np.load(os.path.join(data_dir, "X_attacks.npy"))
    y = np.load(os.path.join(data_dir, "y_attacks_str.npy"), allow_pickle=True).astype(str)

    log.info(f"  X_attacks shape : {X.shape}  dtype={X.dtype}")
    log.info(f"  y_attacks shape : {y.shape}  dtype={y.dtype}")

    unique, counts = np.unique(y, return_counts=True)
    log.info("  Full class distribution (all 13 classes):")
    for cls, cnt in sorted(zip(unique, counts), key=lambda t: -t[1]):
        log.info(f"    {cls:<30s}  {cnt:>8,}")

    return X, y


# ═══════════════════════════════════════════════════════════
# STEP 2 — Filter out excluded classes
# ═══════════════════════════════════════════════════════════
def step2_filter(X: np.ndarray, y: np.ndarray):
    """
    Remove rows whose label is in EXCLUDE_CLASSES.

    Heartbleed (n=12) and Web_SQL_Injection (n=24) are excluded because
    supervised learning requires a minimum viable sample size to learn a
    class boundary.  These two will always be 'unclassified' by the NN.
    They remain detectable via the autoencoder reconstruction error.
    """
    log.info("[STEP 2] Filtering excluded classes ...")
    for cls in sorted(EXCLUDE_CLASSES):
        n = int(np.sum(y == cls))
        log.info(f"  EXCLUDED: {cls}  (n={n}) — too few samples for reliable training")

    mask = ~np.isin(y, list(EXCLUDE_CLASSES))
    X_f = X[mask]
    y_f = y[mask]
    removed = X.shape[0] - X_f.shape[0]
    log.info(f"  Rows after filtering: {X_f.shape[0]:,}  (removed {removed:,})")
    return X_f, y_f


# ═══════════════════════════════════════════════════════════
# STEP 3 — Encode string labels to integers
# ═══════════════════════════════════════════════════════════
def step3_encode(y_str: np.ndarray, data_dir: str, model_dir: str):
    """
    Fit a LabelEncoder on the remaining ~11 attack-type strings and
    save the int→class-name mapping to both data/processed/ and models/.

    Returns
    -------
    y_int        : np.ndarray  int64
    le           : LabelEncoder  (for inverse-transform in downstream scripts)
    label_map    : dict  {int: str}
    """
    log.info("[STEP 3] Encoding string labels to integers ...")
    le = LabelEncoder()
    y_int = le.fit_transform(y_str).astype(np.int64)

    classes = le.classes_.tolist()
    label_map = {i: c for i, c in enumerate(classes)}
    n_classes = len(classes)

    log.info(f"  Number of classes (after exclusions): {n_classes}")
    log.info("  Label encoding:")
    for idx, cls in enumerate(classes):
        log.info(f"    {idx:>2}  →  {cls}")

    # Save to data/processed/ and models/ for runtime use
    for out_dir in (data_dir, model_dir):
        path = os.path.join(out_dir, "attack_type_label_map.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)
        log.info(f"  Saved label map → {path}")

    return y_int, le, label_map


# ═══════════════════════════════════════════════════════════
# STEP 4 — Stratified 70 / 15 / 15 split
# ═══════════════════════════════════════════════════════════
def step4_split(X: np.ndarray, y: np.ndarray):
    """
    Stratified train / val / test split (70 / 15 / 15).

    Stratification is critical here because class sizes range from ~1,357
    (Web_XSS) to ~297,642 (DoS_Hulk) — a 219× imbalance ratio.  Without
    stratification, small classes would be under-represented or absent in
    splits, making val/test metrics unreliable.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
    """
    log.info("[STEP 4] Stratified 70 / 15 / 15 split ...")

    # First cut: 85% train+val, 15% test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=TEST_FRAC,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Second cut: split the 85% into 70% train and 15% val
    # val_frac of the total = val / (1 - test) relative to the 85% slice
    val_of_tv = VAL_FRAC / (1.0 - TEST_FRAC)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_of_tv,
        random_state=RANDOM_STATE,
        stratify=y_tv,
    )

    log.info(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
    log.info(f"  X_val   : {X_val.shape}  y_val   : {y_val.shape}")
    log.info(f"  X_test  : {X_test.shape}  y_test  : {y_test.shape}")

    # Per-class breakdown in each split
    for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        log.info(f"  Class distribution in {split_name}:")
        for cls_int, cnt in zip(unique, counts):
            log.info(f"    class {cls_int:>2}  →  {cnt:>8,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════
# STEP 5 — Compute balanced sample weights on the training split
# ═══════════════════════════════════════════════════════════
def step5_compute_weights(y_train: np.ndarray):
    """
    Compute per-sample class weights using sklearn's 'balanced' strategy:
        weight[i] = n_samples / (n_classes * count[class_of_i])

    This counteracts the severe imbalance between DoS_Hulk (~208K training
    rows) and Web_XSS (~950 training rows), preventing the NN from ignoring
    minority classes during training.

    Note: these are SAMPLE weights (one per training row), not class weights.
    They are saved separately because the NN training script passes them to
    the weighted CrossEntropyLoss via a sample-level weight tensor.

    Returns
    -------
    weights : np.ndarray  float32, shape (n_train,)
    """
    log.info("[STEP 5] Computing balanced sample weights for training set ...")
    weights = compute_sample_weight("balanced", y=y_train).astype(np.float32)
    unique, counts = np.unique(y_train, return_counts=True)
    log.info("  Per-class statistics (class_int → count | mean_weight):")
    for cls_int, cnt in zip(unique, counts):
        mask = y_train == cls_int
        log.info(
            f"    class {cls_int:>2}  count={cnt:>8,}  "
            f"mean_weight={weights[mask].mean():.4f}"
        )
    log.info(f"  Sample weight range: [{weights.min():.6f}, {weights.max():.4f}]")
    return weights


# ═══════════════════════════════════════════════════════════
# STEP 6 — Save all arrays to data/processed/
# ═══════════════════════════════════════════════════════════
def step6_save(
    data_dir: str,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    sample_weights_train,
):
    """Save all split arrays and sample weights to data/processed/."""
    log.info("[STEP 6] Saving arrays to data/processed/ ...")

    arrays = {
        "X_train_attacktype.npy":              X_train,
        "X_val_attacktype.npy":                X_val,
        "X_test_attacktype.npy":               X_test,
        "y_train_attacktype.npy":              y_train,
        "y_val_attacktype.npy":                y_val,
        "y_test_attacktype.npy":               y_test,
        "sample_weights_train_attacktype.npy": sample_weights_train,
    }

    for fname, arr in arrays.items():
        path = os.path.join(data_dir, fname)
        np.save(path, arr)
        log.info(f"  Saved {fname:45s} shape={arr.shape}  dtype={arr.dtype}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("build_attack_type_dataset.py — Stage 2 Data Preparation")
    log.info("=" * 70)

    X, y_str = step1_load_raw(DATA_DIR)
    X_f, y_str_f = step2_filter(X, y_str)
    y_int, le, label_map = step3_encode(y_str_f, DATA_DIR, MODEL_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test = step4_split(X_f, y_int)
    sample_weights = step5_compute_weights(y_train)
    step6_save(DATA_DIR, X_train, X_val, X_test, y_train, y_val, y_test, sample_weights)

    log.info("")
    log.info("✓  build_attack_type_dataset.py complete.")
    log.info(f"   Classes         : {len(le.classes_)}")
    log.info(f"   Training rows   : {X_train.shape[0]:,}")
    log.info(f"   Validation rows : {X_val.shape[0]:,}")
    log.info(f"   Test rows       : {X_test.shape[0]:,}")
    log.info("   Next step: python training/train_attack_type_nn.py")


if __name__ == "__main__":
    main()
