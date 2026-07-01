"""
eval_threshold.py — Threshold Analysis for Trained Autoencoder
===============================================================

PURPOSE:
  Without retraining, finds the optimal anomaly detection threshold by:
    1. Computing reconstruction errors on the FULL val benign set and ALL attacks
    2. Saving error arrays to data/processed/ for later use
    3. Using sklearn precision_recall_curve to sweep all possible thresholds and
       find: (a) the threshold maximising F1, and (b) the highest-precision
       threshold that still achieves recall >= 0.90
    4. Printing a per-attack-type breakdown at the best-F1 threshold to show
       which attack classes are still being missed

  Does NOT modify models/threshold.json — all results are printed for review.

USAGE (from project root):
    python training/eval_threshold.py

REQUIREMENTS:
    pip install torch numpy scikit-learn
"""

import os
import sys
import json
import logging

import numpy as np
from sklearn.metrics import precision_recall_curve, auc  # noqa: E402
import torch

# ─────────────────────────────────────────────
# Path anchoring — same pattern as train_autoencoder.py
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../evaluation/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

# Add training/ to sys.path so we can import from train_autoencoder.py
_TRAINING_DIR = os.path.join(_PROJECT_ROOT, "training")
sys.path.insert(0, _TRAINING_DIR)

from train_autoencoder import Autoencoder, compute_reconstruction_errors  # noqa: E402

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

RECALL_TARGET = 0.90   # minimum recall we want the "high-precision" threshold to meet
BATCH_SIZE    = 512

# ─────────────────────────────────────────────
# Logging  (console only — UTF-8 safe)
# ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_handler],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
def load_model(model_dir: str, device: torch.device) -> Autoencoder:
    """Load trained autoencoder from models/autoencoder.pt."""
    path = os.path.join(model_dir, "autoencoder.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    log.info(f"  Loaded model from : {path}")
    return model


def load_current_threshold(model_dir: str) -> float:
    """Read the current threshold from models/threshold.json."""
    path = os.path.join(model_dir, "threshold.json")
    with open(path, "r", encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load data and compute reconstruction errors
# ═══════════════════════════════════════════════════════════
def step1_compute_errors(model: Autoencoder, device: torch.device):
    """
    Compute reconstruction errors on the FULL benign val set and ALL attacks.
    Save both arrays to data/processed/ and return them.

    Returns
    -------
    errors_benign : np.ndarray  shape (N_val,)
    errors_attack : np.ndarray  shape (N_attack,)
    y_attacks_str : np.ndarray  string labels, same order as errors_attack
    """
    # ── Benign val set ────────────────────────────────────────────────
    benign_path = os.path.join(DATA_DIR, "X_val_benign.npy")
    log.info(f"  Loading X_val_benign from: {benign_path}")
    X_val = np.load(benign_path).astype(np.float32)
    log.info(f"  Shape: {X_val.shape}")

    log.info("  Computing benign reconstruction errors (full val set)...")
    errors_benign = compute_reconstruction_errors(model, X_val, device, BATCH_SIZE)

    out_benign = os.path.join(DATA_DIR, "errors_benign.npy")
    np.save(out_benign, errors_benign)
    log.info(f"  Saved → {out_benign}")

    # ── Full attack set ───────────────────────────────────────────────
    attack_path = os.path.join(DATA_DIR, "X_attacks.npy")
    label_path  = os.path.join(DATA_DIR, "y_attacks_str.npy")
    log.info(f"  Loading X_attacks from : {attack_path}")
    X_attacks = np.load(attack_path).astype(np.float32)
    log.info(f"  Shape: {X_attacks.shape}")

    log.info("  Loading string labels from : {label_path}")
    y_attacks_str = np.load(label_path, allow_pickle=True)
    log.info(f"  Labels shape: {y_attacks_str.shape}")

    log.info("  Computing attack reconstruction errors (full attack set)...")
    errors_attack = compute_reconstruction_errors(model, X_attacks, device, BATCH_SIZE)

    out_attack = os.path.join(DATA_DIR, "errors_attack.npy")
    np.save(out_attack, errors_attack)
    log.info(f"  Saved → {out_attack}")

    return errors_benign, errors_attack, y_attacks_str


# ═══════════════════════════════════════════════════════════
# STEP 2 — Threshold sweep via precision_recall_curve
# ═══════════════════════════════════════════════════════════
def step2_threshold_sweep(
    errors_benign: np.ndarray,
    errors_attack: np.ndarray,
    current_threshold: float,
):
    """
    Build y_true / y_score, run precision_recall_curve, and find:
      (a) threshold that maximises F1
      (b) highest-precision threshold where recall >= RECALL_TARGET

    Also computes PR-AUC and prints comparison against the current threshold.

    Returns
    -------
    best_f1_threshold       : float
    best_recall90_threshold : float or None
    """
    # Build labels and scores
    y_true  = np.concatenate([
        np.zeros(len(errors_benign), dtype=np.int32),
        np.ones(len(errors_attack),  dtype=np.int32),
    ])
    y_score = np.concatenate([errors_benign, errors_attack])

    # precision_recall_curve returns arrays for all unique thresholds
    # Note: len(thresholds) == len(precision) - 1 == len(recall) - 1
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_true, y_score)

    # ── PR-AUC ────────────────────────────────────────────────────────
    pr_auc = auc(recall_arr, precision_arr)

    # ── (a) Best F1 threshold ─────────────────────────────────────────
    f1_arr     = np.array([compute_f1(p, r) for p, r in zip(precision_arr[:-1], recall_arr[:-1])])
    best_idx   = int(np.argmax(f1_arr))
    best_f1_threshold = float(thresholds_arr[best_idx])
    best_f1_precision = float(precision_arr[best_idx])
    best_f1_recall    = float(recall_arr[best_idx])
    best_f1_value     = float(f1_arr[best_idx])

    # FPR at best-F1 threshold
    fp_best = int((errors_benign > best_f1_threshold).sum())
    fpr_best = fp_best / len(errors_benign)

    # ── (b) Highest-precision threshold with recall >= RECALL_TARGET ──
    recall90_mask = recall_arr[:-1] >= RECALL_TARGET
    recall90_threshold = None
    recall90_precision = None
    recall90_recall    = None
    recall90_f1        = None
    recall90_fpr       = None

    if recall90_mask.any():
        r90_idx = int(np.argmax(precision_arr[:-1][recall90_mask]))
        # map back to original indices
        all_r90_indices = np.where(recall90_mask)[0]
        orig_idx = all_r90_indices[r90_idx]
        recall90_threshold = float(thresholds_arr[orig_idx])
        recall90_precision = float(precision_arr[orig_idx])
        recall90_recall    = float(recall_arr[orig_idx])
        recall90_f1        = float(f1_arr[orig_idx])
        fp_r90             = int((errors_benign > recall90_threshold).sum())
        recall90_fpr       = fp_r90 / len(errors_benign)

    # ── Stats at current threshold ────────────────────────────────────
    tp_cur     = int((errors_attack > current_threshold).sum())
    fp_cur     = int((errors_benign > current_threshold).sum())
    cur_recall    = tp_cur / len(errors_attack)
    cur_precision = tp_cur / (tp_cur + fp_cur) if (tp_cur + fp_cur) > 0 else 0.0
    cur_f1        = compute_f1(cur_precision, cur_recall)
    cur_fpr       = fp_cur / len(errors_benign)

    # ── Print results ─────────────────────────────────────────────────
    W = 70
    print(f"\n{'='*W}")
    print(f"  THRESHOLD ANALYSIS  (full val benign + full attack set)")
    print(f"{'='*W}")
    print(f"  Benign samples : {len(errors_benign):>10,}  (full X_val_benign.npy)")
    print(f"  Attack samples : {len(errors_attack):>10,}  (full X_attacks.npy)")
    print(f"  PR-AUC         : {pr_auc:.4f}")
    print()

    hdr = f"  {'Label':35s}  {'Threshold':>12}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}  {'FPR':>8}"
    sep = f"  {'─'*35}  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}"

    print(f"  Candidate Thresholds:")
    print(hdr)
    print(sep)

    print(f"  {'[CURRENT]  95th-pct baseline':35s}  "
          f"{current_threshold:>12.8f}  "
          f"{cur_recall:>8.4f}  {cur_precision:>10.4f}  {cur_f1:>8.4f}  {cur_fpr:>8.4f}")

    print(f"  {'[A]  Best F1':35s}  "
          f"{best_f1_threshold:>12.8f}  "
          f"{best_f1_recall:>8.4f}  {best_f1_precision:>10.4f}  {best_f1_value:>8.4f}  {fpr_best:>8.4f}")

    if recall90_threshold is not None:
        print(f"  {'[B]  Best Precision @ Recall≥0.90':35s}  "
              f"{recall90_threshold:>12.8f}  "
              f"{recall90_recall:>8.4f}  {recall90_precision:>10.4f}  {recall90_f1:>8.4f}  {recall90_fpr:>8.4f}")
    else:
        print(f"\n  ⚠️   No threshold achieves recall >= {RECALL_TARGET:.0%} with this model.")
        print(f"       Maximum achievable recall: {recall_arr.max():.4f}")
        print(f"       Consider retraining with a deeper architecture or hybrid loss.")

    print(sep)
    print()

    # ── Improvement summary ───────────────────────────────────────────
    print(f"  Switching CURRENT → [A] (Best F1):")
    f1_delta     = best_f1_value - cur_f1
    recall_delta = best_f1_recall - cur_recall
    fpr_delta    = fpr_best - cur_fpr
    print(f"    F1    : {cur_f1:.4f} → {best_f1_value:.4f}  ({f1_delta:+.4f})")
    print(f"    Recall: {cur_recall:.4f} → {best_f1_recall:.4f}  ({recall_delta:+.4f})")
    print(f"    FPR   : {cur_fpr:.4f} → {fpr_best:.4f}  ({fpr_delta:+.4f})")

    if recall90_threshold is not None:
        print(f"\n  Switching CURRENT → [B] (Recall≥{RECALL_TARGET:.0%} + best precision):")
        f1b_delta     = recall90_f1    - cur_f1
        recb_delta    = recall90_recall - cur_recall
        fprb_delta    = recall90_fpr   - cur_fpr
        print(f"    F1    : {cur_f1:.4f} → {recall90_f1:.4f}  ({f1b_delta:+.4f})")
        print(f"    Recall: {cur_recall:.4f} → {recall90_recall:.4f}  ({recb_delta:+.4f})")
        print(f"    FPR   : {cur_fpr:.4f} → {recall90_fpr:.4f}  ({fprb_delta:+.4f})")

    print(f"\n  NOTE: models/threshold.json has NOT been modified.")
    print(f"{'='*W}")

    return best_f1_threshold, recall90_threshold


# ═══════════════════════════════════════════════════════════
# STEP 3 — Per-attack-type breakdown at best-F1 threshold
# ═══════════════════════════════════════════════════════════
def step3_attack_breakdown(
    errors_attack: np.ndarray,
    y_attacks_str: np.ndarray,
    best_f1_threshold: float,
):
    """
    For each unique attack type in y_attacks_str, print:
      - Total row count in full attack set
      - Mean reconstruction error
      - Detection rate (% whose error exceeds best_f1_threshold)

    Rows are sorted by detection rate (ascending) so problem classes are visible
    at a glance.
    """
    W = 70
    print(f"\n{'─'*W}")
    print(f"  Per-Attack-Type Breakdown  (threshold = {best_f1_threshold:.8f}  [A] Best F1)")
    print(f"{'─'*W}")
    print(f"  {'Attack Type':30s}  {'Count':>7}  {'Mean Err':>10}  {'Detected':>9}  {'TPR':>7}")
    print(f"  {'─'*30}  {'─'*7}  {'─'*10}  {'─'*9}  {'─'*7}")

    unique_types = np.unique(y_attacks_str)

    rows = []
    for attack_type in unique_types:
        mask      = (y_attacks_str == attack_type)
        errs      = errors_attack[mask]
        count     = len(errs)
        mean_err  = float(errs.mean())
        detected  = int((errs > best_f1_threshold).sum())
        tpr       = detected / count if count > 0 else 0.0
        rows.append((tpr, attack_type, count, mean_err, detected))

    # Sort ascending by TPR so the hardest classes appear first
    rows.sort(key=lambda r: r[0])

    for tpr, attack_type, count, mean_err, detected in rows:
        flag = "✅" if tpr >= 0.50 else "⚠️ "
        print(f"  {flag} {attack_type:28s}  {count:>7,}  {mean_err:>10.6f}  {detected:>9,}  {tpr:>7.4f}")

    total_attacks = len(errors_attack)
    total_detected = int((errors_attack > best_f1_threshold).sum())
    overall_tpr = total_detected / total_attacks if total_attacks > 0 else 0.0
    print(f"  {'─'*30}  {'─'*7}  {'─'*10}  {'─'*9}  {'─'*7}")
    print(f"  {'TOTAL':30s}  {total_attacks:>7,}  {'':>10}  {total_detected:>9,}  {overall_tpr:>7.4f}")
    print(f"{'─'*W}")
    print()
    print("  ⚠️  = detection rate < 50%  (attack class still poorly detected)")
    print("  ✅  = detection rate >= 50%")
    print()
    print("  Attack types with very low mean reconstruction error are hard to detect")
    print("  because they produce traffic patterns similar to benign flows.")
    print("  Improving these would require retraining with a more expressive model.")
    print(f"{'─'*W}\n")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("  THRESHOLD ANALYSIS — eval_threshold.py")
    print("  (No retraining — threshold search on full error distributions)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device : {device}")

    # Load current threshold for comparison
    current_threshold = load_current_threshold(MODEL_DIR)
    log.info(f"  Current threshold (models/threshold.json): {current_threshold:.8f}")

    # ── STEP 1 ────────────────────────────────────────────────────────
    log.info("\n[STEP 1] Loading model and computing reconstruction errors...")
    model = load_model(MODEL_DIR, device)
    errors_benign, errors_attack, y_attacks_str = step1_compute_errors(model, device)

    # ── STEP 2 ────────────────────────────────────────────────────────
    log.info("\n[STEP 2] Running precision-recall threshold sweep...")
    best_f1_threshold, recall90_threshold = step2_threshold_sweep(
        errors_benign, errors_attack, current_threshold
    )

    # ── STEP 3 ────────────────────────────────────────────────────────
    log.info("\n[STEP 3] Per-attack-type breakdown at best-F1 threshold...")
    step3_attack_breakdown(errors_attack, y_attacks_str, best_f1_threshold)

    # ── Final decision prompt ──────────────────────────────────────────
    print("=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print(f"  Current threshold : {current_threshold:.8f}")
    print(f"  [A] Best F1       : {best_f1_threshold:.8f}  ← recommended for best overall F1")
    if recall90_threshold is not None:
        print(f"  [B] Recall ≥ 0.90 : {recall90_threshold:.8f}  ← if high recall is the priority")
    print()
    print("  To commit a new threshold, update models/threshold.json manually:")
    print('    { "threshold": <chosen_value> }')
    print()
    print("  To re-run the full test against the new threshold:")
    print("    python tests/test_autoencoder.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
