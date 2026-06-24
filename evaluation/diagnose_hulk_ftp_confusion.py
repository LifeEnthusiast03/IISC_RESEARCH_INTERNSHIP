"""
diagnose_hulk_ftp_confusion.py — Full 115-feature DoS_Hulk/FTP-Patator Diagnosis
==================================================================================

PURPOSE:
  Investigates the ~3,363 test-set rows where the Attack-Type NN (correctly
  true-labeled DoS_Hulk) is misclassified as FTP-Patator, using two
  complementary lenses:

  PART A — Full 115-feature separation analysis
    Computes normalized separation scores between the misclassified Hulk rows
    and true FTP-Patator rows across all features, identifying the actual
    driving features (rather than the 8 quiet-subvariant candidates ruled
    out by the previous diagnostic).

  PART B — DQN behavior on the misclassified rows
    Constructs the EXACT state vectors the DQN would receive at real
    inference time (using the NN's ACTUAL wrong softmax output, not ground
    truth) and checks whether the DQN's raw-feature signal overrides the
    NN's bad classification to still choose Block IP (action 0, correct for
    DoS_Hulk), or inherits the error and chooses Revoke Credentials (action
    1, correct for FTP-Patator but wrong here).

  PART C — Combined recommendation
    Combines Part A and Part B into a single priority/fix-path recommendation.

STATE VECTOR CONTRACT (must match build_dqn_environment.py exactly):
  state = [x (115) | ae_error (1) | softmax_probs (N) | max_confidence (1)]
  state_dim = 115 + 1 + N + 1   (N = number of attack-type classes, ~11)

OPTIMAL ACTION TABLE (from train_dqn.py):
  DoS_Hulk    → 0  Block IP
  FTP-Patator → 1  Revoke Credentials

INPUTS (data/processed/):
  X_test_attacktype.npy, y_test_attacktype.npy
  attack_type_label_map.json, feature_names.json

INPUTS (models/):
  attack_type_nn.pt, autoencoder.pt, dqn_agent.pt

USAGE (from project root):
    python evaluation/diagnose_hulk_ftp_confusion.py
"""

import os
import sys
import json
import logging

import numpy as np
import torch

# ─────────────────────────────────────────────
# Path anchoring — add training/ to path for imports
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "training"))

# Import model classes and the canonical state-construction function
from train_autoencoder    import Autoencoder,   INPUT_DIM as AE_INPUT_DIM
from train_attack_type_nn import AttackTypeNN
from train_dqn            import DQNNetwork, N_ACTIONS, ACTION_NAMES
from build_dqn_environment import build_state_vectors   # canonical — do NOT reimplement

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

# ─────────────────────────────────────────────
# Label integers (from attack_type_label_map.json)
# ─────────────────────────────────────────────
LABEL_DOS_HULK    = 3
LABEL_FTP_PATATOR = 6

# ─────────────────────────────────────────────
# Verdict thresholds (same logic as diagnose_nn_confusions.py)
# ─────────────────────────────────────────────
STRONG_SEP_THRESH   = 1.0
MODERATE_SEP_THRESH = 0.5
DQN_COMPENSATE_THRESH = 0.70   # >70% Block IP → DQN compensates well

# ─────────────────────────────────────────────
# Logging — console + file, UTF-8 safe on Windows
# ─────────────────────────────────────────────
LOG_PATH = os.path.join(DATA_DIR, "diagnose_hulk_ftp_confusion.log")
_sh = logging.StreamHandler()
_sh.stream = open(_sh.stream.fileno(), mode="w", encoding="utf-8",
                  closefd=False, buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_sh, logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════

def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled standard deviation of two 1-D arrays."""
    na, nb = len(a), len(b)
    if na < 2 and nb < 2:
        return 1e-9
    va = float(np.var(a, ddof=1)) if na >= 2 else 0.0
    vb = float(np.var(b, ddof=1)) if nb >= 2 else 0.0
    denom = na + nb - 2
    return float(np.sqrt(max((na - 1) * va + (nb - 1) * vb, 0.0) / max(denom, 1)))


def _load_resources():
    """Load all required arrays and models."""
    log.info("[LOAD] Loading test arrays, models, and metadata ...")

    X_test = np.load(os.path.join(DATA_DIR, "X_test_attacktype.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test_attacktype.npy"))

    with open(os.path.join(DATA_DIR, "attack_type_label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(label_map)

    # ── Autoencoder ────────────────────────────────────────
    ae = Autoencoder(input_dim=AE_INPUT_DIM)
    ae.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder.pt"),
                                  map_location=device))
    ae = ae.to(device).eval()

    # ── Attack-Type NN ─────────────────────────────────────
    atnn = AttackTypeNN(input_dim=AE_INPUT_DIM, n_classes=n_classes)
    atnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "attack_type_nn.pt"),
                                    map_location=device))
    atnn = atnn.to(device).eval()

    # ── DQN policy network ─────────────────────────────────
    state_dim = AE_INPUT_DIM + 1 + n_classes + 1   # 115 + 1 + 11 + 1 = 128
    dqn = DQNNetwork(state_dim=state_dim, n_actions=N_ACTIONS)
    dqn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dqn_agent.pt"),
                                   map_location=device))
    dqn = dqn.to(device).eval()

    log.info(f"  X_test  : {X_test.shape}   device: {device}")
    log.info(f"  n_classes (Attack-Type NN) : {n_classes}")
    log.info(f"  DQN state_dim              : {state_dim}")
    log.info(f"  Label map  : { {v: k for k, v in label_map.items()} }")

    return X_test, y_test, ae, atnn, dqn, label_map, feature_names, device, n_classes


def _run_atnn_inference(atnn, X, device, batch_size=2048):
    """Run Attack-Type NN; return int predictions AND softmax probability matrix."""
    preds_list = []
    probs_list = []
    sfn = torch.nn.Softmax(dim=1)
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            batch  = X_t[start:start + batch_size].to(device)
            logits = atnn(batch)
            probs  = sfn(logits)
            preds_list.append(probs.argmax(dim=1).cpu().numpy())
            probs_list.append(probs.cpu().numpy())
    return np.concatenate(preds_list), np.concatenate(probs_list, axis=0)


# ═══════════════════════════════════════════════════════════
# PART A — Full 115-feature separation analysis
# ═══════════════════════════════════════════════════════════
def part_a_feature_separation(X_test, y_test, y_pred, feature_names):
    """
    Compare the 3 sub-populations across ALL 115 features:
      Group 1: true DoS_Hulk AND predicted FTP-Patator (the confused rows)
      Group 2: true FTP-Patator AND predicted FTP-Patator (correct FTP)
      Group 3: true DoS_Hulk AND predicted DoS_Hulk (correct Hulk, reference)

    Prints top-20 by normalized separation (Group 1 vs Group 2), plus
    the Group 3 mean for context.
    """
    log.info("")
    log.info("═" * 72)
    log.info("PART A — Full 115-Feature Separation: Misclassified Hulk vs True FTP")
    log.info("═" * 72)

    # ── Identify sub-populations ─────────────────────────
    mask_mf  = (y_test == LABEL_DOS_HULK)    & (y_pred == LABEL_FTP_PATATOR)  # confused
    mask_ftp = (y_test == LABEL_FTP_PATATOR) & (y_pred == LABEL_FTP_PATATOR)  # correct FTP
    mask_hok = (y_test == LABEL_DOS_HULK)    & (y_pred == LABEL_DOS_HULK)     # correct Hulk

    n_mf  = int(mask_mf.sum())
    n_ftp = int(mask_ftp.sum())
    n_hok = int(mask_hok.sum())

    log.info(f"  Misclassified Hulk (true=Hulk, pred=FTP)   : {n_mf:>7,} rows  ← subject")
    log.info(f"  Correctly classified FTP-Patator            : {n_ftp:>7,} rows  ← comparison")
    log.info(f"  Correctly classified DoS_Hulk               : {n_hok:>7,} rows  ← reference")

    if n_mf == 0:
        log.warning("  No Hulk→FTP misclassifications found — nothing to diagnose.")
        return None

    X_mf  = X_test[mask_mf]    # (n_mf,  115)
    X_ftp = X_test[mask_ftp]   # (n_ftp, 115)
    X_hok = X_test[mask_hok]   # (n_hok, 115)

    # ── Per-feature stats ─────────────────────────────────
    n_feat   = X_test.shape[1]
    mean_mf  = X_mf.mean(axis=0)
    mean_ftp = X_ftp.mean(axis=0)
    mean_hok = X_hok.mean(axis=0)
    abs_diff  = np.abs(mean_mf - mean_ftp)

    pooled_stds = np.array([
        _pooled_std(X_mf[:, i], X_ftp[:, i]) for i in range(n_feat)
    ])

    # ── Exclude constant-feature columns (pooled_std ≈ 0) ────────────
    # Features where both groups are constant (e.g. urg_flag_counts = 0
    # everywhere) produce pooled_std < 1e-8.  Division by such a value
    # gives inf or an astronomically large number that dominates the sort.
    # These features carry no discriminating information and must be excluded
    # from the ranking and threshold counts.
    PSTD_MIN    = 1e-8
    valid_mask  = pooled_stds >= PSTD_MIN          # True for non-constant features
    invalid_mask = ~valid_mask
    n_excluded  = int(invalid_mask.sum())
    n_valid     = int(valid_mask.sum())

    # Compute norm_sep only for valid features; set invalid to -1 so they
    # sort below every real score and never appear in top-N.
    # np.where evaluates BOTH branches before selecting — suppress the
    # inevitable divide-by-near-zero warning for the invalid (excluded) branch.
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_sep = np.where(valid_mask, abs_diff / pooled_stds, -1.0)

    # ── Count summary (valid features only) ──────────────────────────
    n_strong   = int(((norm_sep > STRONG_SEP_THRESH)   & valid_mask).sum())
    n_moderate = int(((norm_sep > MODERATE_SEP_THRESH) & valid_mask).sum())
    max_ns     = float(norm_sep[valid_mask].max()) if n_valid > 0 else 0.0

    log.info(f"  Features excluded (pooled_std < {PSTD_MIN:.0e}, constant in both groups): "
             f"{n_excluded}")
    if n_excluded > 0:
        excl_names = [feature_names[i] for i in range(n_feat) if invalid_mask[i]]
        for en in excl_names:
            log.info(f"    excluded: {en}")
    log.info(f"  Features used in ranking                                          : "
             f"{n_valid}")

    # ── Top-20 table (from valid features only) ───────────────────────
    valid_indices = np.where(valid_mask)[0]
    top20_idx = valid_indices[np.argsort(norm_sep[valid_indices])[::-1][:20]]

    log.info("")
    log.info("  TOP 20 FEATURES by norm_sep (misclassified-Hulk vs true-FTP-Patator)")
    log.info("  All values are means of MinMax-scaled features [0,1]")
    log.info("  norm_sep = |misclf_mean − ftp_mean| / pooled_std")
    log.info("")

    hdr = (
        f"  {'Rank':>4}  {'Feature':<36s}"
        f"{'Misclf-Hulk':>13s}"
        f"{'True-FTP':>11s}"
        f"{'Correct-Hulk':>14s}"
        f"{'|diff|':>9s}"
        f"{'norm_sep':>10s}"
        f"  {'signal':>8s}"
    )
    log.info(hdr)
    log.info("  " + "─" * 110)

    top20_data = []
    for rank, fidx in enumerate(top20_idx, start=1):
        fname  = feature_names[fidx]
        m_mf   = float(mean_mf[fidx])
        m_ftp  = float(mean_ftp[fidx])
        m_hok  = float(mean_hok[fidx])
        diff   = float(abs_diff[fidx])
        ns     = float(norm_sep[fidx])
        signal = "STRONG" if ns > STRONG_SEP_THRESH else (
                 "MODERATE" if ns > MODERATE_SEP_THRESH else "weak")
        log.info(
            f"  {rank:>4}  {fname:<36s}"
            f"{m_mf:>13.6f}"
            f"{m_ftp:>11.6f}"
            f"{m_hok:>14.6f}"
            f"{diff:>9.6f}"
            f"{ns:>10.4f}"
            f"  {signal:>8s}"
        )
        top20_data.append({
            "rank": rank, "feature": fname,
            "mean_mf": m_mf, "mean_ftp": m_ftp, "mean_hok": m_hok,
            "abs_diff": diff, "norm_sep": ns,
        })

    log.info("  " + "─" * 110)
    log.info(f"  Features with norm_sep > {STRONG_SEP_THRESH:.1f}  (strong separation)   : {n_strong}")
    log.info(f"  Features with norm_sep > {MODERATE_SEP_THRESH:.1f}  (moderate+ separation): {n_moderate}")
    log.info(f"  Max norm_sep across all 115 features               : {max_ns:.4f}")

    # ── Additional: is misclassified-Hulk CLOSER to FTP or to correct-Hulk? ──
    log.info("")
    log.info("  DIRECTIONAL CHECK (top 20 features): is misclf-Hulk closer to FTP or correct-Hulk?")
    closer_to_ftp = 0
    for d in top20_data:
        d2ftp  = abs(d["mean_mf"] - d["mean_ftp"])
        d2hok  = abs(d["mean_mf"] - d["mean_hok"])
        if d2ftp < d2hok:
            closer_to_ftp += 1
    closer_to_hok = 20 - closer_to_ftp
    log.info(f"  → Closer to true FTP-Patator : {closer_to_ftp}/20 features")
    log.info(f"  → Closer to correct Hulk     : {closer_to_hok}/20 features")

    # ── Driving features list ─────────────────────────────
    driving = [d["feature"] for d in top20_data if d["norm_sep"] > MODERATE_SEP_THRESH]
    log.info("")
    if driving:
        log.info(f"  Driving features (norm_sep > {MODERATE_SEP_THRESH}): {len(driving)} feature(s)")
        for f in driving:
            log.info(f"    • {f}")
    else:
        log.info("  No features exceed the moderate separation threshold.")

    # ── Verdict ───────────────────────────────────────────
    log.info("")
    log.info("  VERDICT — Part A:")
    log.info("  " + "─" * 68)

    if n_strong >= 3:
        verdict_a = (
            f"SEPARATING SIGNAL EXISTS (strong): {n_strong} features with norm_sep > 1.0.\n"
            "  The misclassified Hulk rows ARE distinguishable from true FTP-Patator\n"
            "  on these specific features.  The NN confusion is a model/training issue\n"
            "  (insufficient capacity or weight balance on the FTP-Patator class pulling\n"
            "  the Hulk decision boundary), NOT a feature-representation ceiling.\n"
            "  → Driving features listed above are the candidates for targeted\n"
            "    class-weight tuning, feature attention, or threshold adjustment."
        )
        verdict_type_a = "SIGNAL_STRONG"
    elif n_moderate >= 3:
        verdict_a = (
            f"SEPARATING SIGNAL EXISTS (moderate): {n_moderate} features with norm_sep > 0.5.\n"
            "  There is real but modest separation between misclassified Hulk and\n"
            "  true FTP-Patator.  Within-class variance is high relative to the\n"
            "  between-group difference.  The NN could potentially learn to separate\n"
            "  these rows with improved training (higher FTP class weight, more\n"
            "  epochs for these specific flows, or feature engineering on the\n"
            "  driving features above)."
        )
        verdict_type_a = "SIGNAL_MODERATE"
    elif max_ns < MODERATE_SEP_THRESH:
        verdict_a = (
            f"LIKELY FEATURE CEILING: no feature exceeds norm_sep={MODERATE_SEP_THRESH}.\n"
            "  The misclassified Hulk rows and true FTP-Patator rows appear\n"
            "  statistically near-identical at the 115-feature flow level.\n"
            "  The NN cannot reliably separate them with the current feature set.\n"
            "  → These rows are genuinely ambiguous at the feature level.  Proceed\n"
            "    to Part B to determine whether the DQN can compensate."
        )
        verdict_type_a = "CEILING"
    else:
        verdict_a = (
            f"BORDERLINE: max norm_sep = {max_ns:.3f}.  Some signal exists but it is\n"
            "  marginal.  Combination of class imbalance and feature overlap.\n"
            "  → Focal loss or moderate weight increase for FTP-Patator may help."
        )
        verdict_type_a = "BORDERLINE"

    for line in verdict_a.splitlines():
        log.info(f"  {line}")

    return {
        "verdict_type": verdict_type_a,
        "n_strong": n_strong,
        "n_moderate": n_moderate,
        "max_norm_sep": max_ns,
        "n_misclassified": n_mf,
        "closer_to_ftp_count": closer_to_ftp,
        "driving_features": driving,
        "top20": top20_data,
        "mask_mf": mask_mf,
    }


# ═══════════════════════════════════════════════════════════
# PART B — DQN behavior on the actual misclassified rows
# ═══════════════════════════════════════════════════════════
def part_b_dqn_behavior(X_test, mask_mf, atnn_probs_full,
                         ae, atnn, dqn, n_classes, device):
    """
    For the misclassified-Hulk rows:
      1. Build their DQN state vectors using the NN's ACTUAL (wrong) softmax
         output — exactly as would happen at real inference time.
         State construction uses build_state_vectors() imported from
         build_dqn_environment.py; this guarantees identical composition.
      2. Run the DQN to get its chosen action for each row.
      3. Evaluate how often the DQN still picks Block IP (action 0, correct
         for DoS_Hulk) despite the wrong NN classification.
    """
    log.info("")
    log.info("═" * 72)
    log.info("PART B — DQN Behavior on the ~3,363 Misclassified DoS_Hulk Rows")
    log.info("═" * 72)

    X_mf = X_test[mask_mf].astype(np.float32)   # (n_mf, 115)
    n_mf = len(X_mf)
    log.info(f"  Processing {n_mf:,} misclassified rows ...")

    # ── Build DQN state vectors ──────────────────────────
    # We use build_state_vectors() from build_dqn_environment.py.
    # That function calls the AE and the ATNN internally — so it produces
    # ACTUAL NN softmax probabilities for these rows (the wrong FTP-heavy
    # distribution), which is exactly what the DQN would see at inference time.
    log.info("  Building DQN state vectors (AE + Attack-Type NN forward passes) ...")
    states_mf = build_state_vectors(X_mf, ae, atnn, n_classes, device,
                                    batch_size=1024)
    # Silence the internal log line — already confirmed state_dim above
    log.info(f"  State vectors shape: {states_mf.shape}  "
             f"(115 features + 1 AE err + {n_classes} softmax probs + 1 conf)")

    # ── Sanity: verify that the NN softmax in the state correctly reflects
    #    the wrong prediction (high mass on FTP-Patator class = 6)
    ftp_class_col = 115 + 1 + LABEL_FTP_PATATOR   # index into state vector
    hulk_class_col = 115 + 1 + LABEL_DOS_HULK
    mean_ftp_prob  = float(states_mf[:, ftp_class_col].mean())
    mean_hulk_prob = float(states_mf[:, hulk_class_col].mean())
    log.info(f"  Sanity — mean softmax prob for FTP-Patator class in these states : "
             f"{mean_ftp_prob:.4f}  (should be high — these rows were predicted FTP)")
    log.info(f"  Sanity — mean softmax prob for DoS_Hulk class in these states    : "
             f"{mean_hulk_prob:.4f}  (should be low)")

    # ── Run DQN ─────────────────────────────────────────
    log.info("  Running DQN policy network ...")
    dqn_preds = []
    states_t  = torch.tensor(states_mf, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(states_t), 2048):
            batch  = states_t[start:start + 2048].to(device)
            q_vals = dqn(batch)
            preds  = q_vals.argmax(dim=1).cpu().numpy()
            dqn_preds.append(preds)
    dqn_preds = np.concatenate(dqn_preds)

    # ── Action distribution ───────────────────────────────
    log.info("")
    log.info("  DQN ACTION DISTRIBUTION for the misclassified DoS_Hulk rows:")
    log.info(f"  Ground-truth optimal for DoS_Hulk: Action 0 (Block IP)")
    log.info(f"  Ground-truth optimal for FTP-Patator: Action 1 (Revoke Credentials)")
    log.info("")
    log.info(f"  {'Action':<4}  {'Name':<24s}  {'Count':>7}  {'Fraction':>9}  {'Correct?':>9}")
    log.info("  " + "─" * 60)

    n_correct = 0
    n_wrong   = 0
    action_counts = {}
    for a in range(N_ACTIONS):
        cnt = int((dqn_preds == a).sum())
        frac = cnt / n_mf
        is_correct = (a == 0)   # Block IP is optimal for DoS_Hulk
        if is_correct:
            n_correct += cnt
        else:
            n_wrong += cnt
        action_counts[a] = cnt
        correct_str = "← CORRECT" if is_correct else ("← optimal for FTP" if a == 1 else "")
        log.info(f"  {a:<4}  {ACTION_NAMES[a]:<24s}  {cnt:>7,}  {frac:>8.2%}  {correct_str}")

    log.info("  " + "─" * 60)
    frac_correct = n_correct / n_mf
    frac_wrong   = n_wrong   / n_mf
    log.info(f"  CORRECT (Block IP, action 0)  : {n_correct:>7,}  ({frac_correct:.2%})")
    log.info(f"  WRONG   (any other action)    : {n_wrong:>7,}  ({frac_wrong:.2%})")

    # ── Wrong-action breakdown ─────────────────────────────
    if n_wrong > 0:
        log.info("")
        log.info("  Wrong-action breakdown:")
        for a in range(N_ACTIONS):
            if a == 0:
                continue
            cnt = action_counts[a]
            if cnt > 0:
                log.info(f"    Action {a} ({ACTION_NAMES[a]:<22s}): {cnt:>6,}  ({cnt/n_mf:.2%})")

    # ── Verdict ───────────────────────────────────────────
    log.info("")
    log.info("  VERDICT — Part B:")
    log.info("  " + "─" * 68)

    if frac_correct >= DQN_COMPENSATE_THRESH:
        verdict_b = (
            f"DQN COMPENSATES WELL: {frac_correct:.1%} of misclassified Hulk rows\n"
            f"  still receive the correct action (Block IP) despite the upstream\n"
            f"  NN predicting FTP-Patator.  The raw flow features in the state vector\n"
            f"  (the 115 MinMax-scaled features + AE reconstruction error) provide\n"
            f"  enough signal for the DQN to override the NN's wrong softmax.\n"
            f"  → End-to-end pipeline is robust to this specific NN confusion.\n"
            f"    Fixing the NN is LOW PRIORITY from an operational standpoint."
        )
        verdict_type_b = "DQN_COMPENSATES"
    elif frac_correct >= 0.40:
        verdict_b = (
            f"DQN PARTIALLY INHERITS ERROR: {frac_correct:.1%} correct, {frac_wrong:.1%} wrong.\n"
            f"  The DQN is partially trusting the NN's wrong softmax signal but not\n"
            f"  completely.  The raw features provide some corrective signal but\n"
            f"  not enough to fully override FTP-Patator's high softmax probability.\n"
            f"  → Moderate priority: fixing the NN would improve DQN accuracy,\n"
            f"    but the pipeline is not critically broken for these rows."
        )
        verdict_type_b = "DQN_PARTIAL"
    else:
        verdict_b = (
            f"DQN INHERITS NN ERROR: only {frac_correct:.1%} of misclassified Hulk\n"
            f"  rows get the correct action (Block IP).  The DQN is heavily relying\n"
            f"  on the NN's softmax signal and the wrong FTP-Patator prediction\n"
            f"  propagates directly into wrong remediation actions.\n"
            f"  → HIGH PRIORITY: fix the upstream NN classification for these rows.\n"
            f"    Use the driving features from Part A for targeted retraining."
        )
        verdict_type_b = "DQN_INHERITS"

    for line in verdict_b.splitlines():
        log.info(f"  {line}")

    return {
        "verdict_type": verdict_type_b,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "frac_correct": frac_correct,
        "action_counts": action_counts,
    }


# ═══════════════════════════════════════════════════════════
# PART C — Combined summary and final recommendation
# ═══════════════════════════════════════════════════════════
def part_c_summary(result_a, result_b):
    log.info("")
    log.info("═" * 72)
    log.info("PART C — Combined Summary & Final Recommendation")
    log.info("═" * 72)
    log.info("")

    va = result_a["verdict_type"]
    vb = result_b["verdict_type"]
    n_mf = result_a["n_misclassified"]
    frac_correct = result_b["frac_correct"]

    log.info(f"  Part A verdict : {va}  "
             f"(n_strong={result_a['n_strong']}, n_moderate={result_a['n_moderate']}, "
             f"max_norm_sep={result_a['max_norm_sep']:.4f})")
    log.info(f"  Part B verdict : {vb}  "
             f"(DQN correct rate = {frac_correct:.2%} on {n_mf:,} rows)")
    log.info("")

    # ── Combine verdicts into a single priority + fix path ──
    signal_exists = va in ("SIGNAL_STRONG", "SIGNAL_MODERATE", "BORDERLINE")
    dqn_ok        = (vb == "DQN_COMPENSATES")
    dqn_partial   = (vb == "DQN_PARTIAL")
    dqn_bad       = (vb == "DQN_INHERITS")

    log.info("  ┌────────────────────────────────────────────────────────────────────┐")
    log.info("  │ FINAL RECOMMENDATION                                              │")
    log.info("  ├────────────────────────────────────────────────────────────────────┤")

    if signal_exists and dqn_bad:
        priority = "HIGH"
        rec = (
            "Both conditions met: Part A found real separating features AND\n"
            "  Part B shows the DQN inherits the NN's error for these rows.\n"
            "  → Retrain the Attack-Type NN with focus on the driving features\n"
            "    identified in Part A.  Recommended interventions (in order):\n"
            "    1. Increase DoS_Hulk class weight to prevent FTP-Patator from\n"
            "       pulling these borderline rows across the decision boundary.\n"
            "    2. Feature engineering: add explicit features that distinguish\n"
            "       FTP-Patator (port 21, protocol TCP with small fixed payload)\n"
            "       from DoS_Hulk short connections.\n"
            "    3. If interventions 1-2 fail, consider a dedicated binary\n"
            "       Hulk-vs-FTP post-hoc classifier on the driving features."
        )
    elif signal_exists and dqn_partial:
        priority = "MEDIUM"
        rec = (
            "Part A found separating features but Part B shows the DQN only\n"
            "  partially inherits the error.\n"
            "  → NN improvement is worthwhile but not urgent.  Try increasing\n"
            "    FTP-Patator vs DoS_Hulk class weight ratio.  If Macro-F1 improves\n"
            "    for both classes, retrain the DQN with the improved NN."
        )
    elif signal_exists and dqn_ok:
        priority = "LOW"
        rec = (
            "Part A found separating signal but Part B shows the DQN compensates\n"
            f"  well ({frac_correct:.1%} of rows still get Block IP despite wrong NN label).\n"
            "  → The end-to-end pipeline result is what matters operationally, not\n"
            "    the intermediate NN classification report in isolation.\n"
            "  → Mark as LOW PRIORITY / acceptable.  The DQN's raw-feature signal\n"
            "    is strong enough to override the NN's FTP-Patator softmax.\n"
            "  → Document this in the evaluation report as an example of the DQN\n"
            "    providing robustness against upstream classifier errors."
        )
    elif not signal_exists and dqn_bad:
        priority = "HIGH — different path"
        rec = (
            "Part A found no separating features (feature ceiling) but Part B\n"
            "  shows the DQN inherits the error regardless.\n"
            "  → The flow features alone cannot distinguish these rows from\n"
            "    true FTP-Patator.  Adding metadata (port, protocol) or external\n"
            "    context is needed.  In the short term, consider merging DoS_Hulk\n"
            "    and FTP-Patator into the same DQN action (both are high-impact;\n"
            "    Block IP is more conservative and safer than Revoke Credentials)."
        )
    else:  # ceiling + DQN ok or partial
        priority = "LOW / ACCEPTABLE"
        rec = (
            "Feature ceiling (Part A) AND DQN performs acceptably (Part B).\n"
            f"  DQN correct rate = {frac_correct:.1%}.  No actionable improvement path\n"
            "  from the NN side.  Accept this confusion and proceed to integration."
        )

    log.info(f"  │ Priority : {priority:<58s} │")
    log.info("  │                                                                    │")
    for line in rec.splitlines():
        log.info(f"  │ {line:<68s} │")
    log.info("  └────────────────────────────────────────────────────────────────────┘")
    log.info("")
    log.info("═" * 72)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("=" * 72)
    log.info("diagnose_hulk_ftp_confusion.py — Full Hulk/FTP Confusion Analysis")
    log.info("=" * 72)

    (X_test, y_test, ae, atnn, dqn,
     label_map, feature_names, device, n_classes) = _load_resources()

    log.info("[INFERENCE] Running Attack-Type NN on full test set ...")
    y_pred, atnn_probs = _run_atnn_inference(atnn, X_test, device)
    overall_acc = float(np.mean(y_pred == y_test))
    log.info(f"  Overall test accuracy (sanity check): {overall_acc:.4f}")

    result_a = part_a_feature_separation(X_test, y_test, y_pred, feature_names)

    if result_a is not None:
        result_b = part_b_dqn_behavior(
            X_test, result_a["mask_mf"], atnn_probs,
            ae, atnn, dqn, n_classes, device,
        )
        part_c_summary(result_a, result_b)

    log.info(f"✓  diagnose_hulk_ftp_confusion.py complete.  Log → {LOG_PATH}")


if __name__ == "__main__":
    main()
