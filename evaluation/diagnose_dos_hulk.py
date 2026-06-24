"""
diagnose_dos_hulk.py — DoS_Hulk False-Negative Root-Cause Analysis
===================================================================

PURPOSE:
  DoS_Hulk is the dominant source of false negatives in the combined
  two-stage pipeline (TPR ≈ 0.52, ~143K missed flows out of 297K total).
  This script diagnoses WHY before prescribing a fix.

  It runs four steps in order:

    STEP 1  — Error distribution: are missed flows near the threshold (fixable
              by lowering it) or structurally far below it (need a model/feature
              change)?

    STEP 2  — Threshold sensitivity sweep: what TPR/FPR tradeoff do we get at
              several lower threshold candidates?

    STEP 3  — Sub-population feature analysis (only if Step 1 says "far below"):
              do "detected" and "missed" DoS_Hulk flows differ systematically
              across the 115 features?

    STEP 4  — Recommendation: one clear next action based on Steps 1–3.

DATA USED:
  data/processed/errors_attack.npy    — per-sample reconstruction MSE,
                                        aligned with X_attacks.npy row-wise
  data/processed/errors_benign.npy    — per-sample reconstruction MSE on
                                        X_val_benign.npy   ← NOTE: produced by
                                        training/eval_threshold.py which loads
                                        X_val_benign.npy, NOT X_test_benign.npy
  data/processed/y_attacks_str.npy    — string attack-type labels
  data/processed/X_attacks.npy        — 115-col MinMax-scaled attack features
  data/processed/feature_names.json   — ordered list of 115 feature names
  models/threshold.json               — current anomaly threshold

USAGE (from project root):
    python training/diagnose_dos_hulk.py
"""

import os
import sys
import json
import logging

import numpy as np

# ─────────────────────────────────────────────
# Path anchoring — same pattern as train_autoencoder.py
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../training/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TARGET_CLASS       = "DoS_Hulk"
FPR_UPPER_BOUND    = 0.10   # show tradeoff up to this FPR (charter target is 0.05)
FPR_CHARTER_TARGET = 0.05

# NOTE: the current threshold is NOT hardcoded here — it is prepended at
# runtime inside step2_threshold_sweep() using the value loaded from
# threshold.json so the dict key is always the exact same float object.
THRESHOLD_EXTRA_CANDIDATES = [
    0.0007,
    0.0005,
    0.0003,
    0.0002,
    0.0001,
]

TOP_N_FEATURES = 15   # features to show in Step 3

# ─────────────────────────────────────────────
# Logging  (console, UTF-8 safe — mirrors train_autoencoder.py)
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
# Data loader
# ═══════════════════════════════════════════════════════════
def load_data():
    """
    Load all arrays needed for the four diagnostic steps.
    Also reads and returns the current threshold from threshold.json.
    """
    log.info("  Loading reconstruction error arrays...")

    errors_attack = np.load(os.path.join(DATA_DIR, "errors_attack.npy"))
    log.info(f"    errors_attack  : {errors_attack.shape}  "
             f"(aligned with X_attacks.npy / y_attacks_str.npy)")

    errors_benign = np.load(os.path.join(DATA_DIR, "errors_benign.npy"))
    log.info(f"    errors_benign  : {errors_benign.shape}  "
             f"(computed on X_val_benign.npy by training/eval_threshold.py)")

    y_attacks_str = np.load(
        os.path.join(DATA_DIR, "y_attacks_str.npy"), allow_pickle=True
    )
    log.info(f"    y_attacks_str  : {y_attacks_str.shape}")

    log.info("  Loading X_attacks.npy  (may take a moment — 600K × 115)...")
    X_attacks = np.load(os.path.join(DATA_DIR, "X_attacks.npy")).astype(np.float32)
    log.info(f"    X_attacks      : {X_attacks.shape}")

    with open(os.path.join(DATA_DIR, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    log.info(f"    feature_names  : {len(feature_names)} features")

    with open(os.path.join(MODEL_DIR, "threshold.json"), "r", encoding="utf-8") as f:
        current_threshold = float(json.load(f)["threshold"])
    log.info(f"    Current threshold (threshold.json): {current_threshold:.10f}")

    return errors_attack, errors_benign, y_attacks_str, X_attacks, feature_names, current_threshold


# ═══════════════════════════════════════════════════════════
# STEP 1 — Error distribution & near/far bucket analysis
# ═══════════════════════════════════════════════════════════
def step1_error_distribution(
    errors_attack: np.ndarray,
    y_attacks_str: np.ndarray,
    threshold: float,
) -> str:
    """
    Analyse the distribution of DoS_Hulk reconstruction errors and
    classify missed flows as "near threshold" or "far below threshold".

    Returns the verdict string: either "THRESHOLD-FIXABLE" or "FEATURE/MODEL FIX".
    """
    W = 72
    print(f"\n{'='*W}")
    print(f"  STEP 1 — DoS_Hulk Error Distribution")
    print(f"{'='*W}")

    # ── Filter to DoS_Hulk ────────────────────────────────────────────
    mask_hulk  = (y_attacks_str == TARGET_CLASS)
    errs_hulk  = errors_attack[mask_hulk]
    n_hulk     = len(errs_hulk)

    print(f"\n  {TARGET_CLASS} total samples : {n_hulk:,}")
    print(f"  Current threshold           : {threshold:.10f}")
    print()

    # ── Percentile table ──────────────────────────────────────────────
    pcts   = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    labels = ["min", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"]
    values = np.percentile(errs_hulk, pcts)

    print(f"  {'Statistic':10s}  {'Value':>16}")
    print(f"  {'─'*10}  {'─'*16}")
    for lbl, val in zip(labels, values):
        marker = " ← threshold is here" if val >= threshold and (lbl == "p10" or labels[labels.index(lbl)-1] == "min" or
                 np.percentile(errs_hulk, pcts[labels.index(lbl)-1]) < threshold <= val) else ""
        # simpler: mark if this percentile just crossed the threshold
        print(f"  {lbl:10s}  {val:>16.10f}")

    # ── Mark where threshold sits among percentiles ───────────────────
    pct_below = float(np.mean(errs_hulk < threshold)) * 100
    pct_above = 100.0 - pct_below
    n_missed  = int((errs_hulk <= threshold).sum())
    n_detected= int((errs_hulk  > threshold).sum())

    # Find which percentile bracket the threshold falls in
    all_pcts = np.arange(1, 101)
    pct_vals = np.percentile(errs_hulk, all_pcts)
    bracket  = int(np.searchsorted(pct_vals, threshold, side="right"))

    print()
    print(f"  Threshold sits at approximately the {bracket}th percentile of DoS_Hulk errors")
    print(f"  Detected (error > threshold) : {n_detected:>8,}  ({pct_above:6.2f}%)")
    print(f"  Missed   (error ≤ threshold) : {n_missed:>8,}  ({pct_below:6.2f}%)")

    # ── Near / far bucket split ───────────────────────────────────────
    errs_missed = errs_hulk[errs_hulk <= threshold]
    near_lo     = threshold * 0.5
    near_mask   = errs_missed >= near_lo     # 0.5×threshold ≤ error ≤ threshold
    far_mask    = errs_missed <  near_lo     # error < 0.5×threshold

    n_near = int(near_mask.sum())
    n_far  = int(far_mask.sum())
    pct_near = 100.0 * n_near / n_missed if n_missed > 0 else 0.0
    pct_far  = 100.0 * n_far  / n_missed if n_missed > 0 else 0.0

    print()
    print(f"  Missed-flow bucket analysis  (threshold = {threshold:.10f}):")
    print(f"  {'─'*65}")
    print(f"  Bucket                        Count     Pct of missed")
    print(f"  {'─'*65}")
    print(f"  Near threshold  [0.5×T, T]  : {n_near:>8,}   ({pct_near:6.2f}%)")
    print(f"    (error in [{near_lo:.10f}, {threshold:.10f}])")
    print(f"  Far below       [0, 0.5×T)  : {n_far:>8,}   ({pct_far:6.2f}%)")
    print(f"    (error in [0, {near_lo:.10f}))")
    print(f"  {'─'*65}")

    # ── Verdict ───────────────────────────────────────────────────────
    print()
    if n_near >= n_far:
        verdict = "THRESHOLD-FIXABLE"
        print(f"  ★ VERDICT: LIKELY THRESHOLD-FIXABLE")
        print(f"    The majority of missed DoS_Hulk flows ({pct_near:.1f}%) have")
        print(f"    reconstruction errors in the near-threshold band [0.5×T, T].")
        print(f"    A modest reduction in the detection threshold should recover")
        print(f"    a large fraction of these without necessarily exploding FPR.")
    else:
        verdict = "FEATURE/MODEL FIX"
        print(f"  ★ VERDICT: LIKELY REQUIRES FEATURE/MODEL FIX, NOT THRESHOLD")
        print(f"    The majority of missed DoS_Hulk flows ({pct_far:.1f}%) have")
        print(f"    reconstruction errors WELL BELOW 0.5×threshold — they look")
        print(f"    almost indistinguishable from benign traffic to the autoencoder.")
        print(f"    Lowering the threshold enough to catch them would likely")
        print(f"    flood the pipeline with false positives. Step 3 will examine")
        print(f"    whether these missed flows form a distinguishable sub-population.")

    print(f"{'='*W}")
    return verdict


# ═══════════════════════════════════════════════════════════
# STEP 2 — Threshold sensitivity sweep
# ═══════════════════════════════════════════════════════════
def step2_threshold_sweep(
    errors_attack: np.ndarray,
    errors_benign: np.ndarray,
    y_attacks_str: np.ndarray,
    current_threshold: float,
) -> dict:
    """
    Sweep threshold candidates. For each, compute:
      - DoS_Hulk TPR
      - Overall benign FPR
      - Overall attack TPR (all 13 classes)

    Returns a dict keyed by threshold value with the computed metrics.
    """
    W = 72
    print(f"\n{'='*W}")
    print(f"  STEP 2 — Threshold Sensitivity Sweep")
    print(f"{'='*W}")
    print(f"\n  Benign errors source : X_val_benign.npy  (n={len(errors_benign):,})")
    print(f"  Attack errors source : X_attacks.npy      (n={len(errors_attack):,}, all 13 classes)")
    print(f"  FPR upper bound shown: {FPR_UPPER_BOUND:.2f}  "
          f"(charter target is FPR < {FPR_CHARTER_TARGET:.2f})")
    print()

    mask_hulk       = (y_attacks_str == TARGET_CLASS)
    errs_hulk       = errors_attack[mask_hulk]
    n_hulk          = len(errs_hulk)
    n_benign        = len(errors_benign)
    n_attacks_total = len(errors_attack)

    hdr = (f"  {'Threshold':>14}  {'Hulk TPR':>9}  {'Overall TPR':>11}  "
           f"{'Benign FPR':>10}  Note")
    sep = f"  {'─'*14}  {'─'*9}  {'─'*11}  {'─'*10}  {'─'*20}"
    print(hdr)
    print(sep)

    results = {}
    best_hulk_tpr_under_fpr_bound   = None
    best_thresh_under_fpr_bound      = None
    best_thresh_under_charter        = None
    best_hulk_tpr_under_charter      = None

    # Build the sweep list at runtime so current_threshold (full precision
    # from JSON) is the exact same object used as both dict key and lookup.
    candidates = [current_threshold] + THRESHOLD_EXTRA_CANDIDATES

    for thresh in candidates:
        hulk_detected  = int((errs_hulk        > thresh).sum())
        all_detected   = int((errors_attack     > thresh).sum())
        fp_benign      = int((errors_benign     > thresh).sum())

        hulk_tpr  = hulk_detected / n_hulk          if n_hulk          > 0 else 0.0
        all_tpr   = all_detected  / n_attacks_total if n_attacks_total  > 0 else 0.0
        fpr       = fp_benign     / n_benign         if n_benign         > 0 else 0.0

        results[thresh] = {"hulk_tpr": hulk_tpr, "all_tpr": all_tpr, "fpr": fpr}

        notes = []
        if thresh == current_threshold:
            notes.append("[CURRENT]")
        if fpr <= FPR_UPPER_BOUND:
            notes.append(f"FPR≤{FPR_UPPER_BOUND}")
            if best_hulk_tpr_under_fpr_bound is None or hulk_tpr > best_hulk_tpr_under_fpr_bound:
                best_hulk_tpr_under_fpr_bound = hulk_tpr
                best_thresh_under_fpr_bound   = thresh
        else:
            notes.append(f"FPR>{FPR_UPPER_BOUND} ⚠️")
        if fpr <= FPR_CHARTER_TARGET:
            if best_hulk_tpr_under_charter is None or hulk_tpr > best_hulk_tpr_under_charter:
                best_hulk_tpr_under_charter = hulk_tpr
                best_thresh_under_charter   = thresh

        note_str = "  ".join(notes)
        cur_marker = " ←" if thresh == current_threshold else ""
        print(f"  {thresh:>14.10f}  {hulk_tpr:>9.4f}  {all_tpr:>11.4f}  "
              f"{fpr:>10.4f}  {note_str}{cur_marker}")

    print(sep)

    # ── Highlight best threshold ──────────────────────────────────────
    print()
    if best_thresh_under_fpr_bound is not None and best_thresh_under_fpr_bound != current_threshold:
        m = results[best_thresh_under_fpr_bound]
        hulk_n_at_best = int(m["hulk_tpr"] * n_hulk)
        hulk_n_current = int(results[current_threshold]["hulk_tpr"] * n_hulk)
        extra_tp = hulk_n_at_best - hulk_n_current
        fp_at_best = int(m["fpr"] * n_benign)
        fp_current = int(results[current_threshold]["fpr"] * n_benign)
        extra_fp   = fp_at_best - fp_current

        print(f"  Best threshold within FPR ≤ {FPR_UPPER_BOUND}: "
              f"{best_thresh_under_fpr_bound:.10f}")
        print(f"    DoS_Hulk TPR: "
              f"{results[current_threshold]['hulk_tpr']:.4f} → {m['hulk_tpr']:.4f}  "
              f"(+{extra_tp:,} additional DoS_Hulk TPs)")
        print(f"    Benign FPR  : "
              f"{results[current_threshold]['fpr']:.4f} → {m['fpr']:.4f}  "
              f"(+{extra_fp:,} additional FPs on val benign set)")
    else:
        print(f"  No threshold candidate within FPR ≤ {FPR_UPPER_BOUND} "
              f"substantially improves DoS_Hulk TPR beyond current.")

    if best_thresh_under_charter is not None and best_thresh_under_charter != current_threshold:
        m = results[best_thresh_under_charter]
        print(f"\n  Best threshold within charter FPR ≤ {FPR_CHARTER_TARGET}: "
              f"{best_thresh_under_charter:.10f}")
        print(f"    DoS_Hulk TPR: "
              f"{results[current_threshold]['hulk_tpr']:.4f} → {m['hulk_tpr']:.4f}")
        print(f"    Benign FPR  : "
              f"{results[current_threshold]['fpr']:.4f} → {m['fpr']:.4f}")

    print(f"{'='*W}")
    return results, best_thresh_under_fpr_bound, best_thresh_under_charter


# ═══════════════════════════════════════════════════════════
# STEP 3 — Sub-population feature analysis
# ═══════════════════════════════════════════════════════════
def step3_subpopulation_features(
    errors_attack: np.ndarray,
    y_attacks_str: np.ndarray,
    X_attacks: np.ndarray,
    feature_names: list,
    threshold: float,
):
    """
    Within DoS_Hulk flows only, compare mean feature values between
    "detected" (error > threshold) and "missed" (error ≤ threshold)
    sub-populations. Large differences indicate a structurally distinct
    sub-variant that could be targeted by the hybrid classifier.
    """
    W = 72
    print(f"\n{'='*W}")
    print(f"  STEP 3 — DoS_Hulk Sub-Population Feature Analysis")
    print(f"{'='*W}")

    mask_hulk     = (y_attacks_str == TARGET_CLASS)
    errs_hulk     = errors_attack[mask_hulk]
    X_hulk        = X_attacks[mask_hulk]

    detected_mask = (errs_hulk  > threshold)
    missed_mask   = (errs_hulk <= threshold)

    n_detected = int(detected_mask.sum())
    n_missed   = int(missed_mask.sum())
    n_total    = n_detected + n_missed

    print(f"\n  DoS_Hulk split at threshold = {threshold:.10f}")
    print(f"  Detected sub-population : {n_detected:>8,}  ({100.*n_detected/n_total:.2f}%)")
    print(f"  Missed   sub-population : {n_missed:>8,}  ({100.*n_missed/n_total:.2f}%)  ← target for extension")
    print()

    if n_detected == 0 or n_missed == 0:
        print("  ⚠️  One sub-population is empty — cannot compute feature differences.")
        return

    X_detected = X_hulk[detected_mask]
    X_missed   = X_hulk[missed_mask]

    mean_detected = X_detected.mean(axis=0)   # shape (115,)
    mean_missed   = X_missed.mean(axis=0)     # shape (115,)
    abs_diff      = np.abs(mean_detected - mean_missed)

    # Sort by abs_diff descending, take top N
    top_idx = np.argsort(abs_diff)[::-1][:TOP_N_FEATURES]

    print(f"  Top {TOP_N_FEATURES} features with largest mean difference "
          f"(detected vs missed DoS_Hulk):")
    print()
    print(f"  {'Rank':>4}  {'Feature':35s}  {'Mean (detected)':>16}  "
          f"{'Mean (missed)':>14}  {'|Δ|':>10}")
    print(f"  {'─'*4}  {'─'*35}  {'─'*16}  {'─'*14}  {'─'*10}")

    for rank, idx in enumerate(top_idx, 1):
        fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        m_det = float(mean_detected[idx])
        m_mis = float(mean_missed[idx])
        diff  = float(abs_diff[idx])
        direction = "↑ det" if m_det > m_mis else "↑ mis"
        print(f"  {rank:>4}  {fname:35s}  {m_det:>16.6f}  {m_mis:>14.6f}  "
              f"{diff:>10.6f}  ({direction})")

    # ── Interpretive summary ──────────────────────────────────────────
    print()
    avg_top_diff = float(abs_diff[top_idx].mean())
    overall_mean = float(X_hulk.mean(axis=0).mean())

    print(f"  Average |Δ| across top-{TOP_N_FEATURES} features : {avg_top_diff:.6f}")
    print(f"  Average feature mean across all DoS_Hulk         : {overall_mean:.6f}")

    ratio = avg_top_diff / overall_mean if overall_mean > 0 else 0.0
    print(f"  Separation ratio (avg top-Δ / avg feature mean)  : {ratio:.3f}x")
    print()

    if ratio >= 0.20:
        print(f"  ★ INTERPRETATION: Clear feature separation (ratio ≥ 0.20).")
        print(f"    Missed DoS_Hulk flows show systematically different feature values.")
        print(f"    They appear to represent a DISTINGUISHABLE SUB-VARIANT of DoS_Hulk")
        print(f"    that produces benign-like reconstruction patterns.")
        print(f"    The features listed above are candidate discriminating signals")
        print(f"    for a hybrid classifier extension.")
    elif ratio >= 0.10:
        print(f"  ★ INTERPRETATION: Moderate feature separation (ratio 0.10–0.20).")
        print(f"    Some signal exists but sub-populations partially overlap.")
        print(f"    A hybrid classifier may help but improvement is not guaranteed.")
    else:
        print(f"  ★ INTERPRETATION: Weak feature separation (ratio < 0.10).")
        print(f"    Detected and missed DoS_Hulk flows are nearly indistinguishable")
        print(f"    in the feature space. This suggests label noise or random variation")
        print(f"    rather than a genuine sub-variant — a classifier is unlikely to help.")

    print(f"{'='*W}")
    return ratio, top_idx, feature_names


# ═══════════════════════════════════════════════════════════
# STEP 4 — Summary and recommendation
# ═══════════════════════════════════════════════════════════
def step4_recommendation(
    verdict: str,
    sweep_results: dict,
    best_thresh_under_fpr_bound: float,
    best_thresh_under_charter: float,
    errors_benign: np.ndarray,
    errors_attack: np.ndarray,
    y_attacks_str: np.ndarray,
    step3_ratio: float,
    current_threshold: float,
):
    W = 72
    print(f"\n{'='*W}")
    print(f"  STEP 4 — Summary and Recommendation")
    print(f"{'='*W}")

    mask_hulk = (y_attacks_str == TARGET_CLASS)
    n_hulk    = int(mask_hulk.sum())
    n_benign  = len(errors_benign)
    errs_hulk = errors_attack[mask_hulk]

    cur_hulk_tpr = sweep_results[current_threshold]["hulk_tpr"]
    cur_fpr      = sweep_results[current_threshold]["fpr"]

    print(f"\n  Current status:")
    print(f"    DoS_Hulk TPR : {cur_hulk_tpr:.4f}  ({int(cur_hulk_tpr*n_hulk):,} / {n_hulk:,} detected)")
    print(f"    Benign   FPR : {cur_fpr:.4f}  ({int(cur_fpr*n_benign):,} / {n_benign:,} false positives)")
    print(f"    Step 1 verdict : {verdict}")
    if step3_ratio is not None:
        print(f"    Step 3 feature separation ratio : {step3_ratio:.3f}x")

    # ── Decision logic ────────────────────────────────────────────────
    print()

    # Condition A: threshold adjustment is recommended
    #   - Step 1 says near-threshold (fixable), OR
    #   - Step 2 shows a substantially better threshold within FPR budget
    threshold_improvement_exists = (
        best_thresh_under_fpr_bound is not None
        and best_thresh_under_fpr_bound != current_threshold
    )
    meaningful_tpr_gain = False
    extra_tp = 0
    extra_fp = 0
    if threshold_improvement_exists:
        best_m   = sweep_results[best_thresh_under_fpr_bound]
        extra_tp = int((best_m["hulk_tpr"] - cur_hulk_tpr) * n_hulk)
        extra_fp = int((best_m["fpr"]      - cur_fpr      ) * n_benign)
        meaningful_tpr_gain = (best_m["hulk_tpr"] - cur_hulk_tpr) >= 0.05

    # Condition B: hybrid extension is recommended
    #   - Step 1 says "far below", step 3 shows meaningful separation
    hybrid_recommended = (
        verdict == "FEATURE/MODEL FIX"
        and step3_ratio is not None
        and step3_ratio >= 0.10
    )

    if threshold_improvement_exists and meaningful_tpr_gain:
        best_m = sweep_results[best_thresh_under_fpr_bound]
        print(f"  ┌─ RECOMMENDATION A ─────────────────────────────────────────────┐")
        print(f"  │  THRESHOLD ADJUSTMENT RECOMMENDED                              │")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        print()
        print(f"  Lower threshold from {current_threshold:.10f}")
        print(f"  to                   {best_thresh_under_fpr_bound:.10f}")
        print()
        print(f"  Expected effect:")
        print(f"    DoS_Hulk TPR : {cur_hulk_tpr:.4f} → {best_m['hulk_tpr']:.4f}  "
              f"(≈+{extra_tp:,} additional DoS_Hulk detections)")
        print(f"    Benign   FPR : {cur_fpr:.4f} → {best_m['fpr']:.4f}  "
              f"(≈+{extra_fp:,} additional false positives on val benign set)")
        print(f"    Overall TPR  : {sweep_results[current_threshold]['all_tpr']:.4f} → "
              f"{best_m['all_tpr']:.4f}")
        print()
        if best_m["fpr"] <= FPR_CHARTER_TARGET:
            print(f"  ✅ New threshold keeps FPR within charter target "
                  f"(FPR < {FPR_CHARTER_TARGET}).")
        else:
            print(f"  ⚠️  New threshold exceeds charter FPR target "
                  f"({FPR_CHARTER_TARGET}).")
            if best_thresh_under_charter is not None:
                cm = sweep_results[best_thresh_under_charter]
                ctp = int(cm["hulk_tpr"] * n_hulk)
                cfp = int(cm["fpr"] * n_benign)
                print(f"     Charter-compliant best: {best_thresh_under_charter:.10f}")
                print(f"     (TPR {cm['hulk_tpr']:.4f}, FPR {cm['fpr']:.4f}, "
                      f"+{ctp - int(cur_hulk_tpr*n_hulk):,} DoS_Hulk TPs, "
                      f"+{cfp - int(cur_fpr*n_benign):,} FPs)")
        print()
        print(f"  To apply: update models/threshold.json:")
        print(f'    {{ "threshold": {best_thresh_under_fpr_bound} }}')
        print(f"  Then re-run: python training/evaluate_combined_pipeline.py")

    elif hybrid_recommended:
        print(f"  ┌─ RECOMMENDATION B ─────────────────────────────────────────────┐")
        print(f"  │  HYBRID CLASSIFIER EXTENSION RECOMMENDED                       │")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        print()
        n_missed_hulk = int((errs_hulk <= current_threshold).sum())
        print(f"  Missed DoS_Hulk forms a DISTINGUISHABLE SUB-POPULATION")
        print(f"  (Step 3 feature separation ratio = {step3_ratio:.3f}x ≥ 0.10).")
        print(f"  These {n_missed_hulk:,} missed flows ({100.*n_missed_hulk/n_hulk:.1f}% of DoS_Hulk)")
        print(f"  exhibit systematically different feature values from detected flows,")
        print(f"  suggesting a real traffic sub-variant the autoencoder cannot reconstruct")
        print(f"  distinctively.")
        print()
        print(f"  NEXT ACTION: Add 'DoS_Hulk' as a 7th class to the hybrid XGBoost")
        print(f"  classifier (currently: Benign + 5 weak classes).")
        print(f"  Training approach:")
        print(f"    Positive examples : the {n_missed_hulk:,} currently-missed DoS_Hulk rows")
        print(f"      (those with autoencoder error ≤ {current_threshold:.10f})")
        print(f"    Negative examples : existing Benign sample from hybrid dataset")
        print(f"      (already balanced ~25K rows)")
        print(f"  Then rebuild the hybrid dataset:")
        print(f"    python training/build_hybrid_dataset.py  [after adding DoS_Hulk class]")
        print(f"    python training/train_hybrid_classifier.py")
        print(f"    python training/evaluate_combined_pipeline.py")

    else:
        # Neither clear threshold fix nor model fix is strongly indicated
        print(f"  ┌─ RECOMMENDATION: INVESTIGATE FURTHER ──────────────────────────┐")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        print()
        print(f"  Step 1 verdict       : {verdict}")
        print(f"  Threshold fixable?   : {'Yes (modest gain)' if threshold_improvement_exists else 'No'}")
        print(f"  Feature separation   : {f'{step3_ratio:.3f}x' if step3_ratio else 'N/A'} (threshold for action: 0.10)")
        print()
        print(f"  Neither a large threshold gain nor a clear feature sub-variant")
        print(f"  was identified. Options to consider:")
        print(f"    1. Retrain the autoencoder with DoS_Hulk included in the")
        print(f"       training set as a separate reconstruction target (semi-supervised).")
        print(f"    2. Add flow-timing or inter-arrival-time features that DoS_Hulk")
        print(f"       may distinguish more clearly at the raw packet level.")
        print(f"    3. Accept the current ~52% DoS_Hulk TPR and focus engineering")
        print(f"       effort on classes with near-zero detection rates instead.")

    print(f"\n{'='*W}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 72)
    print(f"  DIAGNOSE DoS_Hulk — diagnose_dos_hulk.py")
    print(f"  False-negative root-cause analysis for the combined pipeline")
    print("=" * 72)
    log.info(f"  Data dir  : {DATA_DIR}")
    log.info(f"  Model dir : {MODEL_DIR}")
    log.info(f"  Target    : {TARGET_CLASS}")

    # ── Load data ─────────────────────────────────────────────────────
    log.info("\n[LOAD] Loading data arrays...")
    (errors_attack, errors_benign, y_attacks_str,
     X_attacks, feature_names, current_threshold) = load_data()

    # ── STEP 1 ────────────────────────────────────────────────────────
    log.info("\n[STEP 1] Error distribution analysis...")
    verdict = step1_error_distribution(errors_attack, y_attacks_str, current_threshold)

    # ── STEP 2 ────────────────────────────────────────────────────────
    log.info("\n[STEP 2] Threshold sensitivity sweep...")
    sweep_results, best_thresh_fpr, best_thresh_charter = step2_threshold_sweep(
        errors_attack, errors_benign, y_attacks_str, current_threshold
    )

    # ── STEP 3 ────────────────────────────────────────────────────────
    step3_ratio = None
    if verdict == "FEATURE/MODEL FIX":
        log.info("\n[STEP 3] Sub-population feature analysis (verdict = FEATURE/MODEL FIX)...")
        result3 = step3_subpopulation_features(
            errors_attack, y_attacks_str, X_attacks, feature_names, current_threshold
        )
        if result3 is not None:
            step3_ratio = result3[0]
    else:
        print(f"\n{'='*72}")
        print(f"  STEP 3 — Sub-Population Feature Analysis  [SKIPPED]")
        print(f"  Verdict from Step 1 is '{verdict}' — threshold adjustment")
        print(f"  is the primary lever. Feature analysis not needed.")
        print(f"{'='*72}")

    # ── STEP 4 ────────────────────────────────────────────────────────
    log.info("\n[STEP 4] Generating recommendation...")
    step4_recommendation(
        verdict,
        sweep_results,
        best_thresh_fpr,
        best_thresh_charter,
        errors_benign,
        errors_attack,
        y_attacks_str,
        step3_ratio,
        current_threshold,
    )

    print()
    log.info("diagnose_dos_hulk.py complete.")


if __name__ == "__main__":
    main()
