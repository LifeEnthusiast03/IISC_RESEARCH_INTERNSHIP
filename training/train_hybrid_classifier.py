"""
train_hybrid_classifier.py — Stage 1B Hybrid Multi-Class Classifier
====================================================================

PURPOSE:
  Trains an XGBoost multi-class classifier (Stage 1B) on the stratified
  hybrid dataset built by build_hybrid_dataset.py.

  This classifier operates DOWNSTREAM of the autoencoder: it only sees
  flows that the autoencoder cleared as "normal" (reconstruction error
  <= threshold). It then tries to catch the 5 attack types that produce
  benign-like flow statistics and evade the autoencoder:

      FTP-Patator  |  Botnet_ARES  |  SSH-Patator
      Web_Brute_Force  |  Web_XSS

INPUTS (data/processed/):
  - X_train_hybrid.npy, y_train_hybrid.npy
  - X_val_hybrid.npy,   y_val_hybrid.npy
  - X_test_hybrid.npy,  y_test_hybrid.npy
  - hybrid_label_map.json

TRAINING STRATEGY:
  - Objective       : multi:softprob  (multi-class probability output)
  - Sample weights  : sklearn compute_sample_weight('balanced') on y_train
                      so Web_XSS (~1,357 rows) is NOT dwarfed by FTP-Patator
                      (~9,531 rows) during gradient updates
  - Hyperparameters : max_depth=6, n_estimators=200, learning_rate=0.1
                      (starting point; tunable)
  - Early stopping  : 20 rounds on val logloss to prevent overfitting

OUTPUTS:
  - models/hybrid_classifier.pkl        — serialised XGBClassifier
  - models/hybrid_label_encoder.pkl     — class-name list (index = XGB class)
  - data/processed/hybrid_feature_importance.png  — top-20 feature bar chart

USAGE (from project root):
    python training/train_hybrid_classifier.py
"""

import os
import sys
import json
import logging
import pickle

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

# ─────────────────────────────────────────────
# Path anchoring — same pattern as train_autoencoder.py
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../training/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Hyperparameters  (tunable — not final)
# ─────────────────────────────────────────────
XGB_PARAMS = dict(
    objective            = "multi:softprob",
    max_depth            = 6,
    n_estimators         = 200,
    learning_rate        = 0.1,
    subsample            = 0.9,          # row subsampling per tree
    colsample_bytree     = 0.9,          # feature subsampling per tree
    tree_method          = "hist",       # fast histogram-based approx
    random_state         = 42,
    n_jobs               = -1,
    eval_metric          = "mlogloss",   # multi-class log-loss for early stopping
    early_stopping_rounds= 20,           # XGBoost 3.x: must be in constructor, not fit()
)
EARLY_STOPPING_ROUNDS = 20  # kept for log display
TOP_N_FEATURES        = 20          # how many features to show in importance plot

# ─────────────────────────────────────────────
# Logging  (console, UTF-8 safe — mirrors existing scripts)
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
# STEP 1 — Load hybrid dataset and label map
# ═══════════════════════════════════════════════════════════
def step1_load_data() -> dict:
    """
    Load the six .npy arrays and hybrid_label_map.json from data/processed/.
    Convert string labels → integer indices for XGBoost.

    Returns a dict with X/y splits (float32 / int32) plus metadata.
    """
    log.info("  Loading hybrid dataset arrays...")

    X_train = np.load(os.path.join(DATA_DIR, "X_train_hybrid.npy")).astype(np.float32)
    X_val   = np.load(os.path.join(DATA_DIR, "X_val_hybrid.npy"  )).astype(np.float32)
    X_test  = np.load(os.path.join(DATA_DIR, "X_test_hybrid.npy" )).astype(np.float32)

    y_train_str = np.load(os.path.join(DATA_DIR, "y_train_hybrid.npy"), allow_pickle=True)
    y_val_str   = np.load(os.path.join(DATA_DIR, "y_val_hybrid.npy"  ), allow_pickle=True)
    y_test_str  = np.load(os.path.join(DATA_DIR, "y_test_hybrid.npy" ), allow_pickle=True)

    label_map_path = os.path.join(DATA_DIR, "hybrid_label_map.json")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # Sorted class list (index = XGB class integer)
    classes = sorted(label_map.keys(), key=lambda k: label_map[k])

    log.info(f"  Classes ({len(classes)}): {classes}")
    log.info(f"  X_train: {X_train.shape}   X_val: {X_val.shape}   X_test: {X_test.shape}")

    # Encode string labels → integers
    y_train = np.array([label_map[s] for s in y_train_str], dtype=np.int32)
    y_val   = np.array([label_map[s] for s in y_val_str  ], dtype=np.int32)
    y_test  = np.array([label_map[s] for s in y_test_str ], dtype=np.int32)

    return {
        "X_train": X_train, "y_train": y_train, "y_train_str": y_train_str,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,  "y_test_str":  y_test_str,
        "label_map": label_map,
        "classes":   classes,
    }


# ═══════════════════════════════════════════════════════════
# STEP 2 — Compute sample weights for imbalanced classes
# ═══════════════════════════════════════════════════════════
def step2_compute_weights(y_train: np.ndarray, classes: list) -> np.ndarray:
    """
    Use sklearn's compute_sample_weight('balanced') to up-weight minority
    classes (especially Web_XSS with ~950 train rows) relative to majority
    classes (FTP-Patator with ~6,672 train rows).

    Returns per-sample weights aligned with y_train.
    """
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    log.info("  Per-class effective weight (mean over samples in that class):")
    for i, cls in enumerate(classes):
        mask = (y_train == i)
        if mask.any():
            w_mean = sample_weights[mask].mean()
            log.info(f"    {cls:<22s}: {w_mean:.4f}  (n={mask.sum():,})")

    return sample_weights


# ═══════════════════════════════════════════════════════════
# STEP 3 — Train XGBoost with early stopping on val set
# ═══════════════════════════════════════════════════════════
def step3_train(data: dict, sample_weights: np.ndarray) -> xgb.XGBClassifier:
    """
    Instantiate and fit XGBClassifier.
    Early stopping is driven by validation mlogloss.

    Returns the fitted classifier.
    """
    n_classes = len(data["classes"])
    params    = dict(XGB_PARAMS, num_class=n_classes)

    log.info(f"  Hyperparameters:")
    for k, v in params.items():
        log.info(f"    {k:<25s}: {v}")
    log.info(f"  Early stopping rounds : {EARLY_STOPPING_ROUNDS}  (set in constructor — XGBoost 3.x)")

    clf = xgb.XGBClassifier(**params)

    log.info("\n  Fitting XGBClassifier  (this may take a moment)...")
    clf.fit(
        data["X_train"],
        data["y_train"],
        sample_weight = sample_weights,
        eval_set      = [(data["X_val"], data["y_val"])],
        verbose       = 20,   # print eval every 20 rounds
    )

    best_round = clf.best_iteration
    log.info(f"\n  Training complete — best iteration: {best_round}")
    return clf


# ═══════════════════════════════════════════════════════════
# STEP 4 — Evaluate on held-out test set
# ═══════════════════════════════════════════════════════════
def step4_evaluate(
    clf: xgb.XGBClassifier,
    data: dict,
):
    """
    Compute and print:
      - Per-class precision, recall, F1 (classification_report)
      - Overall accuracy and macro-F1
    """
    y_pred = clf.predict(data["X_test"])
    y_true = data["y_test"]
    classes = data["classes"]

    overall_acc    = accuracy_score(y_true, y_pred)
    overall_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    W = 72
    print(f"\n{'='*W}")
    print(f"  HYBRID CLASSIFIER — Test Set Evaluation")
    print(f"{'='*W}")
    print(f"\n  Overall Accuracy  : {overall_acc:.4f}")
    print(f"  Macro-F1          : {overall_macro_f1:.4f}")
    print()

    # classification_report with string target names
    y_pred_str = np.array([classes[i] for i in y_pred])
    y_true_str = data["y_test_str"]
    print(classification_report(y_true_str, y_pred_str, zero_division=0))
    print("=" * W)

    return y_pred


# ═══════════════════════════════════════════════════════════
# STEP 5 — Save model and label encoder
# ═══════════════════════════════════════════════════════════
def step5_save_model(clf: xgb.XGBClassifier, classes: list):
    """
    Persist:
      - models/hybrid_classifier.pkl     (joblib dump of XGBClassifier)
      - models/hybrid_label_encoder.pkl  (joblib dump of sorted class list)
    """
    clf_path = os.path.join(MODEL_DIR, "hybrid_classifier.pkl")
    enc_path = os.path.join(MODEL_DIR, "hybrid_label_encoder.pkl")

    joblib.dump(clf, clf_path)
    log.info(f"  Saved classifier    → {clf_path}")

    joblib.dump(classes, enc_path)
    log.info(f"  Saved label encoder → {enc_path}")


# ═══════════════════════════════════════════════════════════
# STEP 6 — Feature importance plot (top-N)
# ═══════════════════════════════════════════════════════════
def step6_feature_importance_plot(clf: xgb.XGBClassifier):
    """
    Extract XGBoost feature importance (weight = number of splits),
    load feature_names.json for column labels, and save a horizontal
    bar chart of the top-N most important features.
    """
    # Load feature names
    feat_path = os.path.join(DATA_DIR, "feature_names.json")
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)   # list of 115 strings

    importance_scores = clf.feature_importances_   # shape (115,)
    n_total = len(importance_scores)

    # Pair (name, score) and sort descending
    pairs = sorted(
        zip(feature_names[:n_total], importance_scores),
        key=lambda t: t[1],
        reverse=True,
    )[:TOP_N_FEATURES]

    feat_labels, feat_scores = zip(*pairs)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    # Colour gradient from teal (most important) to lighter
    colours = plt.cm.GnBu(np.linspace(0.9, 0.4, len(feat_labels)))[::-1]
    bars = ax.barh(range(len(feat_labels)), feat_scores, color=colours, edgecolor="white")

    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels[::-1] if False else feat_labels, fontsize=9)
    ax.invert_yaxis()   # highest at top

    ax.set_xlabel("XGBoost Feature Importance (weight: number of splits)", fontsize=10)
    ax.set_title(
        f"Top {TOP_N_FEATURES} Features — Stage 1B Hybrid Classifier\n"
        f"(6-class: Benign + 5 autoencoder-weak attack types)",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Value labels on bars
    for bar, score in zip(bars, feat_scores):
        ax.text(
            bar.get_width() + max(feat_scores) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.0f}",
            va="center", ha="left", fontsize=8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(DATA_DIR, "hybrid_feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Feature importance plot → {out_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 72)
    print("  TRAIN HYBRID CLASSIFIER — train_hybrid_classifier.py")
    print("  Stage 1B: XGBoost multi-class  (Benign + 5 weak attack types)")
    print("=" * 72)
    log.info(f"  Data dir  : {DATA_DIR}")
    log.info(f"  Model dir : {MODEL_DIR}")
    log.info(f"  XGBoost   : {xgb.__version__}")

    # ── STEP 1 ────────────────────────────────────────────────────────
    log.info("\n[STEP 1] Loading hybrid dataset...")
    data = step1_load_data()

    # ── STEP 2 ────────────────────────────────────────────────────────
    log.info("\n[STEP 2] Computing sample weights (balanced)...")
    sample_weights = step2_compute_weights(data["y_train"], data["classes"])

    # ── STEP 3 ────────────────────────────────────────────────────────
    log.info("\n[STEP 3] Training XGBoost classifier...")
    clf = step3_train(data, sample_weights)

    # ── STEP 4 ────────────────────────────────────────────────────────
    log.info("\n[STEP 4] Evaluating on test set...")
    step4_evaluate(clf, data)

    # ── STEP 5 ────────────────────────────────────────────────────────
    log.info("\n[STEP 5] Saving model and label encoder...")
    step5_save_model(clf, data["classes"])

    # ── STEP 6 ────────────────────────────────────────────────────────
    log.info("\n[STEP 6] Generating feature importance plot...")
    step6_feature_importance_plot(clf)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  OUTPUTS SAVED")
    print("=" * 72)
    print(f"  models/hybrid_classifier.pkl")
    print(f"  models/hybrid_label_encoder.pkl")
    print(f"  data/processed/hybrid_feature_importance.png")
    print(f"\n  Next step: python training/evaluate_combined_pipeline.py")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
