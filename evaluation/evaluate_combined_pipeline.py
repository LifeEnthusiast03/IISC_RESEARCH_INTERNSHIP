"""
evaluate_combined_pipeline.py — Full Two-Stage Pipeline End-to-End Evaluation
==============================================================================

PURPOSE:
  Simulates the COMPLETE two-stage anomaly detection pipeline on the ORIGINAL
  full-scale test data (NOT the sub-sampled hybrid dataset):

    Stage 1  — Autoencoder (models/autoencoder.pt + models/threshold.json)
               For each flow, compute MSE reconstruction error.
               error > threshold → flag as ANOMALY (autoencoder caught it)
               error ≤ threshold → pass to Stage 1B

    Stage 1B — Hybrid XGBoost classifier (models/hybrid_classifier.pkl)
               Applied only to flows the autoencoder cleared as "normal".
               prediction ≠ "Benign" → flag as ANOMALY (hybrid caught it)
               prediction == "Benign" → classify as NORMAL

  Test data used:
    - X_test_benign.npy   (full set, no sub-sampling)
    - X_attacks.npy       (full set — ALL 13 attack classes)
    - y_attacks_str.npy   (string labels aligned row-wise with X_attacks)

  METRICS REPORTED:
    Combined pipeline  vs  Autoencoder-alone — side-by-side table of:
      TPR (overall attack recall), FPR, Precision, F1

    Per-attack-class detection rate for ALL 13 classes under the combined
    pipeline, so you can confirm:
      - Previously-weak 5 classes improved (FTP-Patator, Botnet_ARES,
        SSH-Patator, Web_Brute_Force, Web_XSS)
      - Previously-strong classes are unaffected (DoS_Hulk, Port_Scan, etc.)

  STATISTICAL CAVEATS:
    Heartbleed (n=12) and Web_SQL_Injection (n=24) results are explicitly
    flagged as UNRELIABLE due to sample sizes too small to draw conclusions.

USAGE (from project root):
    python training/evaluate_combined_pipeline.py
"""

import os
import sys
import json
import logging

import numpy as np
import sklearn
import torch
import joblib

# ─────────────────────────────────────────────
# Path anchoring — same pattern as train_autoencoder.py
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../evaluation/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

# Make train_autoencoder importable (for Autoencoder class +
# compute_reconstruction_errors helper)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "training"))
from train_autoencoder import Autoencoder, compute_reconstruction_errors  # noqa: E402

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

BATCH_SIZE = 512

# Attack classes with tiny sample counts — flag these in output
UNRELIABLE_CLASSES = {"Heartbleed": 12, "Web_SQL_Injection": 24}
UNRELIABLE_N_THRESHOLD = 50   # n < this threshold → unreliable flag

# ─────────────────────────────────────────────
# Logging  (console, UTF-8 safe)
# ─────────────────────────────────────────────
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
_stream_handler = logging.StreamHandler(sys.stderr)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_stream_handler],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load models and threshold
# ═══════════════════════════════════════════════════════════
def step1_load_models(device: torch.device):
    """
    Load:
      - Autoencoder weights from models/autoencoder.pt
      - Anomaly threshold from models/threshold.json
      - Hybrid classifier from models/hybrid_classifier.pkl
      - Label encoder (class list) from models/hybrid_label_encoder.pkl

    Returns autoencoder, threshold, hybrid_clf, hybrid_classes
    """
    # ── Autoencoder ───────────────────────────────────────────────────
    ae_path = os.path.join(MODEL_DIR, "autoencoder.pt")
    if not os.path.exists(ae_path):
        raise FileNotFoundError(f"Autoencoder not found: {ae_path}")
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
    autoencoder.eval()
    log.info(f"  Loaded autoencoder       : {ae_path}")

    # ── Threshold ─────────────────────────────────────────────────────
    thresh_path = os.path.join(MODEL_DIR, "threshold.json")
    with open(thresh_path, "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])
    log.info(f"  Anomaly threshold        : {threshold:.10f}")

    # ── Hybrid classifier ─────────────────────────────────────────────
    clf_path = os.path.join(MODEL_DIR, "hybrid_classifier.pkl")
    if not os.path.exists(clf_path):
        raise FileNotFoundError(
            f"Hybrid classifier not found: {clf_path}\n"
            "Run: python training/train_hybrid_classifier.py"
        )
    hybrid_clf = joblib.load(clf_path)
    log.info(f"  Loaded hybrid classifier : {clf_path}")

    # ── Label encoder ─────────────────────────────────────────────────
    enc_path = os.path.join(MODEL_DIR, "hybrid_label_encoder.pkl")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Label encoder not found: {enc_path}")
    hybrid_classes = joblib.load(enc_path)   # sorted list: index → class name
    log.info(f"  Hybrid classes           : {hybrid_classes}")

    return autoencoder, threshold, hybrid_clf, hybrid_classes


# ═══════════════════════════════════════════════════════════
# STEP 2 — Load test data (full scale)
# ═══════════════════════════════════════════════════════════
def step2_load_test_data():
    """
    Load the FULL (no sub-sampling):
      - X_test_benign.npy   → true negative ground truth
      - X_attacks.npy + y_attacks_str.npy  → all 13 attack classes

    Returns X_benign, X_attacks, y_attacks_str
    """
    benign_path = os.path.join(DATA_DIR, "X_test_benign.npy")
    attack_path = os.path.join(DATA_DIR, "X_attacks.npy")
    label_path  = os.path.join(DATA_DIR, "y_attacks_str.npy")

    log.info(f"  Loading X_test_benign   : {benign_path}")
    X_benign = np.load(benign_path).astype(np.float32)
    log.info(f"  Shape: {X_benign.shape}")

    log.info(f"  Loading X_attacks       : {attack_path}")
    X_attacks = np.load(attack_path).astype(np.float32)
    log.info(f"  Shape: {X_attacks.shape}")

    log.info(f"  Loading y_attacks_str   : {label_path}")
    y_attacks_str = np.load(label_path, allow_pickle=True)
    log.info(f"  Labels shape: {y_attacks_str.shape}")
    log.info(f"  Unique attack types ({len(np.unique(y_attacks_str))}): "
             f"{sorted(np.unique(y_attacks_str).tolist())}")

    return X_benign, X_attacks, y_attacks_str


# ═══════════════════════════════════════════════════════════
# STEP 3 — Run autoencoder on all flows, get reconstruction errors
# ═══════════════════════════════════════════════════════════
def step3_compute_ae_errors(
    autoencoder: Autoencoder,
    X_benign: np.ndarray,
    X_attacks: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample MSE reconstruction errors.

    Returns
    -------
    ae_errors_benign  : np.ndarray  shape (N_benign,)
    ae_errors_attacks : np.ndarray  shape (N_attacks,)
    """
    log.info("  Computing benign reconstruction errors...")
    ae_errors_benign = compute_reconstruction_errors(
        autoencoder, X_benign, device, BATCH_SIZE
    )
    log.info(f"    Shape: {ae_errors_benign.shape}  "
             f"mean={ae_errors_benign.mean():.6f}  max={ae_errors_benign.max():.6f}")

    log.info("  Computing attack reconstruction errors...")
    ae_errors_attacks = compute_reconstruction_errors(
        autoencoder, X_attacks, device, BATCH_SIZE
    )
    log.info(f"    Shape: {ae_errors_attacks.shape}  "
             f"mean={ae_errors_attacks.mean():.6f}  max={ae_errors_attacks.max():.6f}")

    return ae_errors_benign, ae_errors_attacks


# ═══════════════════════════════════════════════════════════
# STEP 4 — Run two-stage pipeline, collect decisions
# ═══════════════════════════════════════════════════════════
def step4_run_pipeline(
    ae_errors_benign:  np.ndarray,
    ae_errors_attacks: np.ndarray,
    X_benign:          np.ndarray,
    X_attacks:         np.ndarray,
    threshold:         float,
    hybrid_clf,
    hybrid_classes:    list,
) -> dict:
    """
    For each flow, implement the decision tree:

      if ae_error > threshold:
          decision = ANOMALY  (autoencoder caught it)
      else:
          hybrid_pred = hybrid_clf.predict(flow)
          if hybrid_pred != 'Benign':
              decision = ANOMALY  (hybrid caught it)
          else:
              decision = NORMAL

    Returns a dict with binary decisions for benign and attack flows,
    plus per-attack-class hybrid predictions for breakdown analysis.
    """
    # ── Benign flows ──────────────────────────────────────────────────
    log.info("  Running pipeline on benign flows...")
    ae_flags_benign    = (ae_errors_benign > threshold)       # True = AE flagged
    ae_passed_benign   = ~ae_flags_benign                     # flows AE cleared
    n_ae_passed_benign = int(ae_passed_benign.sum())

    log.info(f"    Benign: {len(ae_errors_benign):,} flows")
    log.info(f"      AE flagged (FP) : {int(ae_flags_benign.sum()):,}")
    log.info(f"      AE passed to 1B : {n_ae_passed_benign:,}")

    # Run hybrid on benign flows that AE cleared
    hybrid_flagged_benign = np.zeros(len(X_benign), dtype=bool)
    if n_ae_passed_benign > 0:
        X_b_passed  = X_benign[ae_passed_benign]
        hyb_preds_b = hybrid_clf.predict(X_b_passed)           # integer labels
        hyb_classes_b = np.array([hybrid_classes[i] for i in hyb_preds_b])
        hyb_flags_b   = (hyb_classes_b != "Benign")
        # Map back to full-length benign index
        hybrid_flagged_benign[ae_passed_benign] = hyb_flags_b
        log.info(f"      Hybrid flagged  : {int(hyb_flags_b.sum()):,}")

    # Final combined decision for benign flows (True = any stage flagged = FP)
    combined_flagged_benign = ae_flags_benign | hybrid_flagged_benign

    # ── Attack flows ──────────────────────────────────────────────────
    log.info("  Running pipeline on attack flows...")
    ae_flags_attacks    = (ae_errors_attacks > threshold)     # True = AE detected
    ae_passed_attacks   = ~ae_flags_attacks
    n_ae_passed_attacks = int(ae_passed_attacks.sum())

    log.info(f"    Attacks: {len(ae_errors_attacks):,} flows")
    log.info(f"      AE detected (TP) : {int(ae_flags_attacks.sum()):,}")
    log.info(f"      AE passed to 1B  : {n_ae_passed_attacks:,}")

    # Per-sample hybrid decisions for attacks
    hybrid_flagged_attacks = np.zeros(len(X_attacks), dtype=bool)
    hybrid_preds_attacks   = np.full(len(X_attacks), "AE_detected", dtype=object)

    if n_ae_passed_attacks > 0:
        X_a_passed   = X_attacks[ae_passed_attacks]
        hyb_preds_a  = hybrid_clf.predict(X_a_passed)
        hyb_classes_a = np.array([hybrid_classes[i] for i in hyb_preds_a])
        hyb_flags_a   = (hyb_classes_a != "Benign")

        hybrid_flagged_attacks[ae_passed_attacks] = hyb_flags_a
        hybrid_preds_attacks[ae_passed_attacks]   = hyb_classes_a
        log.info(f"      Hybrid detected (TP): {int(hyb_flags_a.sum()):,}")

    # Combined: attack flagged by EITHER stage = TP
    combined_flagged_attacks = ae_flags_attacks | hybrid_flagged_attacks

    return {
        # Benign flags (True = incorrectly flagged = FP)
        "ae_flagged_benign":       ae_flags_benign,
        "combined_flagged_benign": combined_flagged_benign,
        # Attack flags (True = correctly detected = TP)
        "ae_flagged_attacks":       ae_flags_attacks,
        "combined_flagged_attacks": combined_flagged_attacks,
        # Per-flow stage decisions for attack breakdown
        "ae_passed_attacks":        ae_passed_attacks,
        "hybrid_flagged_attacks":   hybrid_flagged_attacks,
    }


# ═══════════════════════════════════════════════════════════
# STEP 5 — Compute and print comparative metrics table
# ═══════════════════════════════════════════════════════════
def _compute_metrics(flagged_benign: np.ndarray, flagged_attacks: np.ndarray) -> dict:
    """Compute TPR, FPR, Precision, F1 from boolean flag arrays."""
    n_benign  = len(flagged_benign)
    n_attacks = len(flagged_attacks)

    tp = int(flagged_attacks.sum())   # attacks correctly flagged
    fp = int(flagged_benign.sum())    # benign wrongly flagged
    fn = n_attacks - tp
    tn = n_benign  - fp

    tpr = tp / n_attacks if n_attacks > 0 else 0.0
    fpr = fp / n_benign  if n_benign  > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tpr, "fpr": fpr, "precision": precision, "f1": f1,
        "n_attacks": n_attacks, "n_benign": n_benign,
    }


def step5_print_comparison(pipeline_results: dict):
    """
    Print a side-by-side comparison table of autoencoder-alone vs
    combined-pipeline metrics.
    """
    ae_m  = _compute_metrics(
        pipeline_results["ae_flagged_benign"],
        pipeline_results["ae_flagged_attacks"],
    )
    comb_m = _compute_metrics(
        pipeline_results["combined_flagged_benign"],
        pipeline_results["combined_flagged_attacks"],
    )

    W = 78
    print(f"\n{'='*W}")
    print(f"  COMBINED PIPELINE vs AUTOENCODER-ALONE — FULL-SCALE TEST DATA")
    print(f"{'='*W}")
    print(f"  Benign flows  (X_test_benign.npy, full) : {ae_m['n_benign']:>10,}")
    print(f"  Attack flows  (X_attacks.npy,    full)  : {ae_m['n_attacks']:>10,}")
    print()
    print(f"  {'Metric':35s}  {'Autoencoder alone':>18}  {'Combined pipeline':>18}")
    print(f"  {'─'*35}  {'─'*18}  {'─'*18}")

    rows = [
        ("True Positive Rate (TPR / Recall)", ae_m["tpr"],       comb_m["tpr"]),
        ("False Positive Rate (FPR)",          ae_m["fpr"],       comb_m["fpr"]),
        ("Precision",                           ae_m["precision"], comb_m["precision"]),
        ("F1 Score",                            ae_m["f1"],        comb_m["f1"]),
    ]
    for label, ae_val, comb_val in rows:
        delta = comb_val - ae_val
        delta_str = f"({delta:+.4f})"
        print(f"  {label:35s}  {ae_val:>18.4f}  {comb_val:>18.4f}  {delta_str}")

    print()
    print(f"  {'Confusion Matrix':35s}  {'Autoencoder alone':>18}  {'Combined pipeline':>18}")
    print(f"  {'─'*35}  {'─'*18}  {'─'*18}")
    print(f"  {'True Positives (TP)':35s}  {ae_m['tp']:>18,}  {comb_m['tp']:>18,}")
    print(f"  {'False Positives (FP)':35s}  {ae_m['fp']:>18,}  {comb_m['fp']:>18,}")
    print(f"  {'False Negatives (FN)':35s}  {ae_m['fn']:>18,}  {comb_m['fn']:>18,}")
    print(f"  {'True Negatives (TN)':35s}  {ae_m['tn']:>18,}  {comb_m['tn']:>18,}")
    print(f"{'='*W}")

    return ae_m, comb_m


# ═══════════════════════════════════════════════════════════
# STEP 6 — Per-attack-class detection breakdown (13 classes)
# ═══════════════════════════════════════════════════════════
def step6_per_class_breakdown(
    pipeline_results: dict,
    y_attacks_str: np.ndarray,
    ae_m: dict,
    comb_m: dict,
):
    """
    For each of the 13 attack classes, print:
      - n (sample count)
      - AE-alone detection rate (TPR)
      - Combined pipeline detection rate (TPR)
      - Delta
      - Stage that caught each flow (AE / Hybrid / Missed)
      - Reliability flag for tiny-sample classes
    """
    ae_flagged  = pipeline_results["ae_flagged_attacks"]
    hyb_flagged = pipeline_results["hybrid_flagged_attacks"]
    comb_flagged= pipeline_results["combined_flagged_attacks"]

    unique_classes = sorted(np.unique(y_attacks_str).tolist())

    W = 85
    print(f"\n{'─'*W}")
    print(f"  Per-Attack-Class Detection Rate — Combined Pipeline  (ALL 13 classes)")
    print(f"{'─'*W}")
    print(f"  {'Attack Type':22s}  {'N':>6}  {'AE TPR':>8}  {'Comb TPR':>9}  "
          f"{'Delta':>7}  {'AE':>7}  {'Hybrid':>7}  {'Missed':>7}  Note")
    print(f"  {'─'*22}  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*12}")

    # Group classes: weak 5 (targeted by hybrid) vs strong (AE should handle)
    WEAK_CLASSES = {
        "FTP-Patator", "Botnet_ARES", "SSH-Patator",
        "Web_Brute_Force", "Web_XSS",
    }

    rows = []
    for cls in unique_classes:
        mask    = (y_attacks_str == cls)
        n       = int(mask.sum())
        ae_det  = int(ae_flagged[mask].sum())
        hyb_det = int(hyb_flagged[mask].sum())
        # Hybrid-only detections (not already caught by AE)
        hyb_only = int((hyb_flagged[mask] & ~ae_flagged[mask]).sum())
        comb_det = int(comb_flagged[mask].sum())
        missed   = n - comb_det

        ae_tpr   = ae_det   / n if n > 0 else 0.0
        comb_tpr = comb_det / n if n > 0 else 0.0
        delta    = comb_tpr - ae_tpr

        unreliable = (n < UNRELIABLE_N_THRESHOLD)
        weak_class = (cls in WEAK_CLASSES)

        rows.append((cls, n, ae_det, ae_tpr, hyb_only, comb_det, comb_tpr, delta, missed, unreliable, weak_class))

    # Sort: weak classes first (alphabetical), then strong classes (by combined TPR desc)
    weak_rows   = sorted([r for r in rows if r[10]],  key=lambda r: r[0])
    strong_rows = sorted([r for r in rows if not r[10]], key=lambda r: -r[6])
    sorted_rows = weak_rows + strong_rows

    for (cls, n, ae_det, ae_tpr, hyb_only, comb_det, comb_tpr, delta, missed,
         unreliable, weak_class) in sorted_rows:

        if unreliable:
            note = "⚠️  UNRELIABLE (n<50)"
        elif weak_class:
            note = "★  hybrid target"
        else:
            note = ""

        tpr_flag  = "✅" if comb_tpr >= 0.50 else "❌"
        delta_str = f"{delta:+.4f}"

        print(f"  {tpr_flag} {cls:<20s}  {n:>6,}  {ae_tpr:>8.4f}  {comb_tpr:>9.4f}  "
              f"{delta_str:>7}  {ae_det:>7,}  {hyb_only:>7,}  {missed:>7,}  {note}")

    print(f"  {'─'*22}  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

    # Totals
    n_total   = int(ae_m["n_attacks"])
    ae_total  = int(ae_m["tp"])
    comb_total= int(comb_m["tp"])
    print(f"  {'TOTAL':22s}  {n_total:>6,}  {ae_m['tpr']:>8.4f}  {comb_m['tpr']:>9.4f}  "
          f"{comb_m['tpr']-ae_m['tpr']:>+7.4f}  {ae_total:>7,}  "
          f"{comb_total-ae_total:>7,}  {n_total-comb_total:>7,}")
    print(f"{'─'*W}")
    print()
    print("  Legend:")
    print("    ★  hybrid target — one of the 5 classes the hybrid classifier is trained to catch")
    print("    ⚠️  UNRELIABLE   — n < 50; TPR estimate has very high variance, not conclusive")
    print("    AE              — detected by autoencoder alone")
    print("    Hybrid          — detected by hybrid (additional TPs beyond AE)")
    print("    Missed          — neither stage detected (FN)")
    print()
    print("  NOTE: Heartbleed (n=12) and Web_SQL_Injection (n=24) results are")
    print("        STATISTICALLY UNRELIABLE. With single-digit or low-double-digit")
    print("        samples any TPR figure has very wide confidence intervals and")
    print("        should NOT be used to draw conclusions about model performance.")
    print(f"{'─'*W}\n")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 78)
    print("  EVALUATE COMBINED PIPELINE — evaluate_combined_pipeline.py")
    print("  Full two-stage simulation: Autoencoder → Hybrid XGBoost")
    print("  Test data: X_test_benign.npy (full) + X_attacks.npy (full, 13 classes)")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device : {device}")

    # ── STEP 1 ────────────────────────────────────────────────────────
    log.info("\n[STEP 1] Loading models and threshold...")
    autoencoder, threshold, hybrid_clf, hybrid_classes = step1_load_models(device)

    # ── STEP 2 ────────────────────────────────────────────────────────
    log.info("\n[STEP 2] Loading full-scale test data...")
    X_benign, X_attacks, y_attacks_str = step2_load_test_data()

    # ── STEP 3 ────────────────────────────────────────────────────────
    log.info("\n[STEP 3] Computing autoencoder reconstruction errors...")
    ae_errors_benign, ae_errors_attacks = step3_compute_ae_errors(
        autoencoder, X_benign, X_attacks, device
    )

    # ── STEP 4 ────────────────────────────────────────────────────────
    log.info("\n[STEP 4] Running two-stage pipeline decision logic...")
    pipeline_results = step4_run_pipeline(
        ae_errors_benign,
        ae_errors_attacks,
        X_benign,
        X_attacks,
        threshold,
        hybrid_clf,
        hybrid_classes,
    )

    # ── STEP 5 ────────────────────────────────────────────────────────
    log.info("\n[STEP 5] Computing and printing comparative metrics table...")
    ae_m, comb_m = step5_print_comparison(pipeline_results)

    # ── STEP 6 ────────────────────────────────────────────────────────
    log.info("\n[STEP 6] Per-attack-class breakdown (all 13 classes)...")
    step6_per_class_breakdown(pipeline_results, y_attacks_str, ae_m, comb_m)

    # ── Final summary ──────────────────────────────────────────────────
    print("=" * 78)
    print("  EVALUATION COMPLETE")
    print("=" * 78)
    print(f"  Autoencoder-alone    TPR={ae_m['tpr']:.4f}  FPR={ae_m['fpr']:.4f}  "
          f"Precision={ae_m['precision']:.4f}  F1={ae_m['f1']:.4f}")
    print(f"  Combined pipeline    TPR={comb_m['tpr']:.4f}  FPR={comb_m['fpr']:.4f}  "
          f"Precision={comb_m['precision']:.4f}  F1={comb_m['f1']:.4f}")
    delta_tpr  = comb_m["tpr"] - ae_m["tpr"]
    delta_fpr  = comb_m["fpr"] - ae_m["fpr"]
    delta_f1   = comb_m["f1"]  - ae_m["f1"]
    print(f"\n  Delta (combined - AE) TPR={delta_tpr:+.4f}  FPR={delta_fpr:+.4f}  "
          f"F1={delta_f1:+.4f}")
    if delta_tpr > 0 and delta_f1 > 0:
        print("\n  ✅ Combined pipeline improves on autoencoder-alone.")
    elif delta_tpr > 0 and delta_fpr > 0:
        print("\n  ⚠️  Combined pipeline raises TPR but also FPR — check the FPR impact.")
    else:
        print("\n  ℹ️  Review per-class breakdown for nuanced interpretation.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
