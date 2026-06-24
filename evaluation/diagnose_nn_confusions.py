"""
diagnose_nn_confusions.py — Attack-Type NN Confusion Pattern Diagnosis
=======================================================================

PURPOSE:
  Diagnoses two specific confusion patterns observed in the Attack-Type NN
  test-set evaluation:

    1. DoS_Hulk → FTP-Patator misclassifications:
       Are these the same "quiet sub-variant" of DoS_Hulk previously
       identified in diagnose_dos_hulk.py, or a different phenomenon?

    2. Web_Brute_Force ↔ Web_XSS mutual confusion:
       Is there any feature signal that meaningfully separates these two
       classes, or have we hit a feature-representation ceiling?

INPUTS (data/processed/):
  - X_test_attacktype.npy       (90,016 × 115) test features
  - y_test_attacktype.npy       (90,016,)      integer test labels
  - attack_type_label_map.json  int → class-name
  - feature_names.json          ordered 115 feature names

INPUTS (models/):
  - attack_type_nn.pt           trained AttackTypeNN weights

USAGE (from project root):
    python evaluation/diagnose_nn_confusions.py
"""

import os
import sys
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Path anchoring — add training/ to path for imports
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "training"))

from train_attack_type_nn import AttackTypeNN

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

# ─────────────────────────────────────────────
# Logging — console + file, UTF-8 safe on Windows
# ─────────────────────────────────────────────
LOG_PATH = os.path.join(DATA_DIR, "diagnose_nn_confusions.log")
_sh = logging.StreamHandler()
_sh.stream = open(_sh.stream.fileno(), mode="w", encoding="utf-8",
                  closefd=False, buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_sh, logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# "Quiet sub-variant" features from diagnose_dos_hulk.py
# These are the 8 features that were near-zero in the missed/quiet
# DoS_Hulk sub-population but populated in normally-detected Hulk flows.
# ─────────────────────────────────────────────
QUIET_HULK_FEATURES = [
    "packets_IAT_mean",
    "packet_IAT_max",
    "packet_IAT_min",
    "packet_IAT_total",
    "fwd_packets_IAT_mean",
    "bwd_max_header_bytes",
    "bwd_init_win_bytes",
    "bwd_payload_bytes_variance",
]

# Label integers (from attack_type_label_map.json)
LABEL_DOS_HULK       = 3
LABEL_FTP_PATATOR    = 6
LABEL_WEB_BRUTE      = 9
LABEL_WEB_XSS        = 10


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
def _load_resources():
    """Load test arrays, model, label map, and feature names."""
    log.info("[LOAD] Loading test arrays, model, label map, feature names ...")

    X_test = np.load(os.path.join(DATA_DIR, "X_test_attacktype.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test_attacktype.npy"))

    with open(os.path.join(DATA_DIR, "attack_type_label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(label_map)
    model = AttackTypeNN(input_dim=115, n_classes=n_classes)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "attack_type_nn.pt"), map_location=device)
    )
    model = model.to(device).eval()

    log.info(f"  X_test shape : {X_test.shape}")
    log.info(f"  y_test shape : {y_test.shape}  unique classes : {np.unique(y_test).tolist()}")
    log.info(f"  Device       : {device}")

    return X_test, y_test, model, label_map, feature_names, device


def _run_inference(model, X_test, device, batch_size=2048):
    """Run the trained model on X_test, return integer predictions."""
    preds = []
    X_t   = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            batch  = X_t[start:start + batch_size].to(device)
            logits = model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def _feat_idx(feature_names, name):
    """Return the column index for a named feature (case-insensitive fallback)."""
    if name in feature_names:
        return feature_names.index(name)
    # Case-insensitive fallback
    lower = [f.lower() for f in feature_names]
    if name.lower() in lower:
        return lower.index(name.lower())
    raise KeyError(f"Feature '{name}' not found in feature_names.json")


def _fmt_row(label, vals, width=14):
    """Format a table row with label + numeric values."""
    return f"  {label:<36s}" + "".join(f"{v:{width}.6f}" for v in vals)


def _pooled_std(vals_a, vals_b):
    """
    Compute the pooled standard deviation for two 1-D arrays.
    pooled_std = sqrt( ((n_a-1)*var_a + (n_b-1)*var_b) / (n_a + n_b - 2) )
    Returns 1e-9 if both arrays have < 2 elements (degenerate case).
    """
    n_a, n_b = len(vals_a), len(vals_b)
    if n_a < 2 and n_b < 2:
        return 1e-9
    var_a = float(np.var(vals_a, ddof=1)) if n_a >= 2 else 0.0
    var_b = float(np.var(vals_b, ddof=1)) if n_b >= 2 else 0.0
    denom = n_a + n_b - 2
    if denom <= 0:
        return 1e-9
    pstd = ((n_a - 1) * var_a + (n_b - 1) * var_b) / denom
    return float(np.sqrt(max(pstd, 1e-18)))


# ═══════════════════════════════════════════════════════════
# STEP 1 — DoS_Hulk → FTP-Patator confusion vs quiet-Hulk overlap
# ═══════════════════════════════════════════════════════════
def step1_hulk_ftp_diagnosis(X_test, y_test, y_pred, feature_names, label_map):
    log.info("")
    log.info("═" * 70)
    log.info("STEP 1 — DoS_Hulk → FTP-Patator confusion vs quiet-Hulk overlap")
    log.info("═" * 70)

    # ── Identify the three sub-populations ──────────────────
    hulk_ftp_mask  = (y_test == LABEL_DOS_HULK)    & (y_pred == LABEL_FTP_PATATOR)
    hulk_ok_mask   = (y_test == LABEL_DOS_HULK)    & (y_pred == LABEL_DOS_HULK)
    ftp_ok_mask    = (y_test == LABEL_FTP_PATATOR)  & (y_pred == LABEL_FTP_PATATOR)

    n_hulk_ftp = int(hulk_ftp_mask.sum())
    n_hulk_ok  = int(hulk_ok_mask.sum())
    n_ftp_ok   = int(ftp_ok_mask.sum())

    log.info(f"  True DoS_Hulk in test set          : {int((y_test == LABEL_DOS_HULK).sum()):>7,}")
    log.info(f"  → Correctly classified as DoS_Hulk : {n_hulk_ok:>7,}")
    log.info(f"  → Misclassified as FTP-Patator      : {n_hulk_ftp:>7,}  ← subject of this analysis")
    log.info(f"  True FTP-Patator (correctly classed): {n_ftp_ok:>7,}")

    if n_hulk_ftp == 0:
        log.warning("  No DoS_Hulk→FTP-Patator misclassifications found — nothing to diagnose.")
        return

    X_hulk_ftp = X_test[hulk_ftp_mask]
    X_hulk_ok  = X_test[hulk_ok_mask]
    X_ftp_ok   = X_test[ftp_ok_mask]

    # ── Resolve quiet-Hulk feature indices ──────────────────
    feat_indices = []
    resolved_names = []
    for fname in QUIET_HULK_FEATURES:
        try:
            idx = _feat_idx(feature_names, fname)
            feat_indices.append(idx)
            resolved_names.append(feature_names[idx])
        except KeyError:
            log.warning(f"  Feature '{fname}' not found in feature_names.json — skipped.")

    log.info("")
    log.info("  3-WAY FEATURE COMPARISON: quiet-Hulk signature features")
    log.info("  (all values are means of MinMax-scaled features [0,1])")
    log.info("")
    hdr = (
        f"  {'Feature':<36s}"
        f"{'Misclassified':>14s}"
        f"{'Correct Hulk':>14s}"
        f"{'True FTP':>14s}"
        f"{'Misclf≈FTP?':>13s}"
        f"{'Misclf≈Quiet?':>14s}"
    )
    log.info(hdr)
    log.info("  " + "─" * 105)

    close_to_ftp_count   = 0
    close_to_quiet_count = 0
    # "quiet sub-variant" means near-zero → threshold: misclf mean < 0.5 × correct-Hulk mean
    # "close to FTP" means |misclf mean - FTP mean| < 0.5 × |correct-Hulk mean - FTP mean|

    for fname, fidx in zip(resolved_names, feat_indices):
        m_mf  = float(X_hulk_ftp[:, fidx].mean())
        m_ok  = float(X_hulk_ok[:, fidx].mean())
        m_ftp = float(X_ftp_ok[:, fidx].mean())

        # Is misclassified closer to FTP than to correct Hulk?
        dist_to_ftp  = abs(m_mf - m_ftp)
        dist_to_hulk = abs(m_mf - m_ok)
        closer_to_ftp  = dist_to_ftp < dist_to_hulk
        closer_to_ftp_str = "YES" if closer_to_ftp else "no"

        # Is misclassified in the "quiet" regime (near zero, much less than correct Hulk)?
        # Use absolute threshold: if m_ok > 0.01 and m_mf < 0.5 * m_ok → quiet
        if m_ok > 0.01:
            is_quiet = m_mf < 0.5 * m_ok
        else:
            is_quiet = False
        quiet_str = "YES" if is_quiet else "no"

        if closer_to_ftp:
            close_to_ftp_count += 1
        if is_quiet:
            close_to_quiet_count += 1

        log.info(
            f"  {fname:<36s}"
            f"{m_mf:>14.6f}"
            f"{m_ok:>14.6f}"
            f"{m_ftp:>14.6f}"
            f"{closer_to_ftp_str:>13s}"
            f"{quiet_str:>14s}"
        )

    n_feats = len(resolved_names)
    log.info("  " + "─" * 105)
    log.info(f"  Features where misclassified is closer to FTP than to correct Hulk : "
             f"{close_to_ftp_count}/{n_feats}")
    log.info(f"  Features in 'quiet' regime (misclf mean < 50% of correct-Hulk mean): "
             f"{close_to_quiet_count}/{n_feats}")

    # ── Verdict ─────────────────────────────────────────────
    log.info("")
    log.info("  VERDICT — DoS_Hulk → FTP-Patator:")
    log.info("  " + "─" * 68)
    ftp_frac   = close_to_ftp_count   / n_feats
    quiet_frac = close_to_quiet_count / n_feats

    if ftp_frac >= 0.5 and quiet_frac >= 0.4:
        verdict = (
            "CONFIRMED: misclassified DoS_Hulk rows ARE the quiet sub-variant.\n"
            "  These flows have near-zero IAT and backward-traffic features\n"
            "  (short/single-packet connections), which structurally resembles\n"
            "  FTP-Patator's short brute-force connection pattern at the feature\n"
            "  level. The NN is correctly sensing the structural similarity but\n"
            "  assigning the wrong class label.\n"
            "  → FIX PATH: the quiet sub-variant needs a distinguishing feature\n"
            "    (e.g., FTP port 21 indicator, fwd/bwd packet ratio, protocol)\n"
            "    or the hybrid classifier's DoS_Hulk 7th-class signal should be\n"
            "    incorporated into the NN's training data."
        )
    elif ftp_frac >= 0.5 and quiet_frac < 0.4:
        verdict = (
            "PARTIAL: misclassified Hulk rows resemble FTP-Patator on these\n"
            "  features, but are NOT clearly in the quiet/near-zero regime.\n"
            "  This suggests a DIFFERENT mechanism from the previously identified\n"
            "  quiet sub-variant — possibly FTP-Patator's class weight is\n"
            "  over-pulling the Hulk decision boundary.\n"
            "  → FIX PATH: reduce FTP-Patator class weight or apply label\n"
            "    smoothing to prevent the NN over-correcting for small classes."
        )
    elif quiet_frac >= 0.5 and ftp_frac < 0.5:
        verdict = (
            "PARTIAL: misclassified Hulk rows ARE in the quiet regime (near-zero\n"
            "  backward/IAT features) but do NOT closely resemble true FTP-Patator.\n"
            "  The NN is confused by the quiet sub-variant but routing them to\n"
            "  FTP-Patator for a different reason (possibly protocol or port features).\n"
            "  → FIX PATH: investigate protocol/port features in the quiet sub-variant."
        )
    else:
        verdict = (
            "NOT CONFIRMED: misclassification appears UNRELATED to the previously\n"
            "  identified quiet sub-variant on these 8 features. The confusion\n"
            "  pattern requires separate investigation of other feature dimensions.\n"
            "  → FIX PATH: run a full 115-feature separation analysis between\n"
            "    misclassified-Hulk and true FTP-Patator to find the driving features."
        )

    for line in verdict.splitlines():
        log.info(f"  {line}")

    return {
        "close_to_ftp_fraction":   ftp_frac,
        "close_to_quiet_fraction": quiet_frac,
        "n_misclassified":         n_hulk_ftp,
        "verdict_type": (
            "CONFIRMED" if ftp_frac >= 0.5 and quiet_frac >= 0.4
            else "PARTIAL_FTP" if ftp_frac >= 0.5
            else "PARTIAL_QUIET" if quiet_frac >= 0.5
            else "NOT_CONFIRMED"
        ),
    }


# ═══════════════════════════════════════════════════════════
# STEP 2 — Web_Brute_Force vs Web_XSS feature separation check
# ═══════════════════════════════════════════════════════════
def step2_web_separation(X_test, y_test, feature_names):
    log.info("")
    log.info("═" * 70)
    log.info("STEP 2 — Web_Brute_Force vs Web_XSS feature separation check")
    log.info("═" * 70)

    wb_mask  = (y_test == LABEL_WEB_BRUTE)
    xss_mask = (y_test == LABEL_WEB_XSS)

    X_wb  = X_test[wb_mask]
    X_xss = X_test[xss_mask]

    log.info(f"  True Web_Brute_Force rows in test set : {len(X_wb):>5,}")
    log.info(f"  True Web_XSS rows in test set         : {len(X_xss):>5,}")
    log.info("")

    n_features = X_test.shape[1]
    mean_wb  = X_wb.mean(axis=0)   # (115,)
    mean_xss = X_xss.mean(axis=0)  # (115,)
    abs_diff  = np.abs(mean_wb - mean_xss)

    # Compute pooled std and normalized separation for every feature
    pooled_stds = np.empty(n_features)
    for i in range(n_features):
        pooled_stds[i] = _pooled_std(X_wb[:, i], X_xss[:, i])
    norm_sep = abs_diff / pooled_stds   # element-wise

    # Top 15 by absolute difference
    top15_idx = np.argsort(abs_diff)[::-1][:15]

    log.info("  TOP 15 FEATURES by absolute mean difference (Web_Brute_Force vs Web_XSS):")
    log.info("  (MinMax-scaled values; norm_sep = |diff| / pooled_std)")
    log.info("")
    hdr = (
        f"  {'#':>3}  {'Feature':<36s}"
        f"{'WBrute mean':>13s}"
        f"{'XSS mean':>11s}"
        f"{'|diff|':>10s}"
        f"{'norm_sep':>11s}"
        f"  {'signal?':>8s}"
    )
    log.info(hdr)
    log.info("  " + "─" * 97)

    n_above_1   = 0
    n_above_0_5 = 0
    max_norm_sep = 0.0

    top15_data = []
    for rank, fidx in enumerate(top15_idx, start=1):
        fname  = feature_names[fidx]
        m_wb   = float(mean_wb[fidx])
        m_xss  = float(mean_xss[fidx])
        diff   = float(abs_diff[fidx])
        nsep   = float(norm_sep[fidx])
        signal = "STRONG" if nsep > 1.0 else ("MODERATE" if nsep > 0.5 else "weak")

        max_norm_sep = max(max_norm_sep, nsep)
        if nsep > 1.0:
            n_above_1 += 1
        if nsep > 0.5:
            n_above_0_5 += 1

        log.info(
            f"  {rank:>3}  {fname:<36s}"
            f"{m_wb:>13.6f}"
            f"{m_xss:>11.6f}"
            f"{diff:>10.6f}"
            f"{nsep:>11.4f}"
            f"  {signal:>8s}"
        )
        top15_data.append({
            "rank": rank, "feature": fname,
            "mean_wb": m_wb, "mean_xss": m_xss,
            "abs_diff": diff, "norm_sep": nsep,
        })

    log.info("  " + "─" * 97)
    log.info(f"  Features with norm_sep > 1.0 (STRONG separation) : {n_above_1}")
    log.info(f"  Features with norm_sep > 0.5 (MODERATE+)          : {n_above_0_5}")
    log.info(f"  Max normalized separation score (any feature)     : {max_norm_sep:.4f}")

    # ── All-features distribution summary ───────────────────
    log.info("")
    log.info("  FULL 115-FEATURE SEPARATION SUMMARY:")
    bins = [(2.0, "norm_sep > 2.0 (very strong)"),
            (1.0, "norm_sep > 1.0 (strong)"),
            (0.5, "norm_sep > 0.5 (moderate)"),
            (0.0, "norm_sep > 0.0 (any signal)")]
    for thresh, label in bins:
        cnt = int((norm_sep > thresh).sum())
        log.info(f"    {label:<35s}: {cnt:>3} features")

    # ── Verdict ─────────────────────────────────────────────
    log.info("")
    log.info("  VERDICT — Web_Brute_Force vs Web_XSS:")
    log.info("  " + "─" * 68)

    if n_above_1 >= 3:
        verdict = (
            "SEPARATING SIGNAL EXISTS (strong): "
            f"{n_above_1} features have norm_sep > 1.0.\n"
            "  The confusion is a model/training issue, NOT a feature ceiling.\n"
            "  The NN has sufficient signal to distinguish these classes but is\n"
            "  failing due to severe class imbalance (Web_XSS has only ~204\n"
            "  test rows, ~949 training rows vs ~1,913 for Web_Brute_Force).\n"
            "  → FIX PATH: class-weight tuning (increase Web_XSS weight further),\n"
            "    focal loss (γ=2), or oversampling (SMOTE on Web_XSS training rows).\n"
            "    Do NOT merge these classes — the NN can learn to separate them."
        )
        verdict_type = "SIGNAL_EXISTS_STRONG"
    elif n_above_0_5 >= 3:
        verdict = (
            "SEPARATING SIGNAL EXISTS (moderate): "
            f"{n_above_0_5} features have norm_sep > 0.5.\n"
            "  There is moderate separation available, but within-class variance\n"
            "  is high relative to the between-class differences. Confusion is\n"
            "  likely driven by both class imbalance AND partially overlapping\n"
            "  feature distributions.\n"
            "  → FIX PATH: attempt focal loss + moderate class-weight increase\n"
            "    for Web_XSS. Also consider inspecting the specific features with\n"
            "    norm_sep > 0.5 as candidate engineered features.\n"
            "    Merging is premature — try re-weighting first."
        )
        verdict_type = "SIGNAL_EXISTS_MODERATE"
    elif max_norm_sep < 0.5:
        verdict = (
            "LIKELY FEATURE CEILING: no feature achieves norm_sep > 0.5.\n"
            "  Web_Brute_Force and Web_XSS are statistically very similar at the\n"
            "  115-feature flow level. The difference between classes is smaller\n"
            "  than within-class noise for every feature.\n"
            "  → FIX PATH: consider treating them as a merged 'ambiguous web\n"
            "    attack' action category in the DQN. Both map to similar remediation\n"
            "    actions (Kill Process / Revoke Credentials) so merging into a\n"
            "    single 'web_application_attack' class has low operational cost.\n"
            "    Do NOT spend further model capacity trying to separate them at\n"
            "    this feature granularity."
        )
        verdict_type = "FEATURE_CEILING"
    else:
        verdict = (
            "BORDERLINE: some features have moderate separation (max norm_sep = "
            f"{max_norm_sep:.3f})\n"
            "  but the overall picture is weak. The NN is likely failing due to\n"
            "  both class imbalance and marginal feature separation.\n"
            "  → FIX PATH: try focal loss first (low-cost intervention). If macro\n"
            "    F1 for these two classes does not improve above 0.5 after\n"
            "    re-training, merge them into one web-attack category."
        )
        verdict_type = "BORDERLINE"

    for line in verdict.splitlines():
        log.info(f"  {line}")

    return {
        "n_above_1":    n_above_1,
        "n_above_0_5":  n_above_0_5,
        "max_norm_sep": max_norm_sep,
        "verdict_type": verdict_type,
        "top15":        top15_data,
    }


# ═══════════════════════════════════════════════════════════
# STEP 3 — Combined summary and recommendations
# ═══════════════════════════════════════════════════════════
def step3_summary(step1_result, step2_result, label_map):
    log.info("")
    log.info("═" * 70)
    log.info("STEP 3 — Combined Summary & Recommendations")
    log.info("═" * 70)
    log.info("")

    # ── Confusion 1: DoS_Hulk → FTP-Patator ─────────────────
    log.info("  ┌─────────────────────────────────────────────────────────────────┐")
    log.info("  │ Confusion 1: DoS_Hulk → FTP-Patator                           │")
    log.info("  ├─────────────────────────────────────────────────────────────────┤")

    v1 = step1_result["verdict_type"] if step1_result else "NOT_RUN"
    n1 = step1_result["n_misclassified"] if step1_result else 0

    if v1 == "CONFIRMED":
        log.info(f"  │ Verdict    : CONFIRMED — quiet sub-variant / FTP structural    │")
        log.info(f"  │              similarity  ({n1:,} rows affected)               │")
        log.info(f"  │ Root cause : Short/single-packet Hulk flows share feature      │")
        log.info(f"  │              profile with FTP brute-force short connections    │")
        log.info(f"  │ Fix path   : Add protocol/port feature engineering; or         │")
        log.info(f"  │              incorporate hybrid-classifier DoS_Hulk signal     │")
        log.info(f"  │              into Attack-Type NN state                         │")
        log.info(f"  │ Priority   : MEDIUM — DQN maps both to Block IP (action 0),   │")
        log.info(f"  │              so operational impact is limited to misreporting  │")
    elif v1 in ("PARTIAL_FTP", "PARTIAL_QUIET"):
        log.info(f"  │ Verdict    : PARTIAL — overlap with quiet sub-variant or FTP   │")
        log.info(f"  │              profile detected but not both ({n1:,} rows)       │")
        log.info(f"  │ Fix path   : Reduce FTP-Patator class weight / label smooth    │")
        log.info(f"  │ Priority   : MEDIUM                                            │")
    else:
        log.info(f"  │ Verdict    : NOT CONFIRMED — separate investigation needed     │")
        log.info(f"  │ Fix path   : Full 115-feature separation analysis required     │")
        log.info(f"  │ Priority   : HIGH                                              │")
    log.info("  └─────────────────────────────────────────────────────────────────┘")
    log.info("")

    # ── Confusion 2: Web_Brute_Force ↔ Web_XSS ───────────────
    log.info("  ┌─────────────────────────────────────────────────────────────────┐")
    log.info("  │ Confusion 2: Web_Brute_Force ↔ Web_XSS                        │")
    log.info("  ├─────────────────────────────────────────────────────────────────┤")

    v2       = step2_result["verdict_type"]
    above1   = step2_result["n_above_1"]
    above05  = step2_result["n_above_0_5"]
    max_ns   = step2_result["max_norm_sep"]

    if v2 == "SIGNAL_EXISTS_STRONG":
        log.info(f"  │ Verdict    : SEPARATING SIGNAL EXISTS (strong)                 │")
        log.info(f"  │              {above1} features with norm_sep > 1.0              │")
        log.info(f"  │ Root cause : Severe class imbalance (XSS has ~949 train rows)  │")
        log.info(f"  │ Fix path   : Focal loss (γ=2) + higher Web_XSS class weight    │")
        log.info(f"  │              Or SMOTE oversampling of Web_XSS training rows    │")
        log.info(f"  │ Priority   : HIGH — XSS→Kill Process vs Brute→Revoke differ   │")
    elif v2 == "SIGNAL_EXISTS_MODERATE":
        log.info(f"  │ Verdict    : SEPARATING SIGNAL EXISTS (moderate)               │")
        log.info(f"  │              {above05} features with norm_sep > 0.5             │")
        log.info(f"  │ Root cause : Class imbalance + partially overlapping features  │")
        log.info(f"  │ Fix path   : Focal loss first; consider SMOTE if F1 < 0.5     │")
        log.info(f"  │ Priority   : HIGH (same reason: different optimal DQN actions) │")
    elif v2 == "FEATURE_CEILING":
        log.info(f"  │ Verdict    : LIKELY FEATURE CEILING (max norm_sep={max_ns:.3f}) │")
        log.info(f"  │ Root cause : Classes are statistically near-identical at the   │")
        log.info(f"  │              flow-feature level                                │")
        log.info(f"  │ Fix path   : Merge into a single 'web_application_attack'      │")
        log.info(f"  │              class; assign unified DQN action (Kill Process,   │")
        log.info(f"  │              action 3, is conservative and correct for both)   │")
        log.info(f"  │ Priority   : LOW (merging reduces operational risk, not raises)│")
    else:  # BORDERLINE
        log.info(f"  │ Verdict    : BORDERLINE (max norm_sep={max_ns:.3f})              │")
        log.info(f"  │ Root cause : Marginal feature separation + class imbalance     │")
        log.info(f"  │ Fix path   : Try focal loss; merge if F1 < 0.5 after retrain  │")
        log.info(f"  │ Priority   : MEDIUM                                            │")
    log.info("  └─────────────────────────────────────────────────────────────────┘")
    log.info("")

    # ── DQN impact note ──────────────────────────────────────
    log.info("  NOTE ON DQN IMPACT:")
    log.info("  ─────────────────────────────────────────────────────────────────")
    log.info("  Confusion 1 (Hulk→FTP): both DoS_Hulk and FTP-Patator map to")
    log.info("  different optimal DQN actions (Block IP=0 vs Revoke Creds=1).")
    log.info("  The ~3,363 misclassified rows will cause the DQN to receive a")
    log.info("  wrong attack-type probability vector, potentially biasing it")
    log.info("  toward action 1 for those flows. However, since the NN softmax")
    log.info("  is part of the DQN state (not a hard decision), the DQN can")
    log.info("  learn to partially compensate via the raw flow features also")
    log.info("  present in its state.")
    log.info("")
    log.info("  Confusion 2 (WBrute↔XSS): DQN optimal actions differ (Revoke=1")
    log.info("  vs Kill Process=3). This confusion directly degrades DQN quality")
    log.info("  for these two classes and should be prioritised for fixing.")
    log.info("═" * 70)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("diagnose_nn_confusions.py — Attack-Type NN Confusion Diagnosis")
    log.info("=" * 70)

    X_test, y_test, model, label_map, feature_names, device = _load_resources()

    log.info("[INFERENCE] Running model on test set ...")
    y_pred = _run_inference(model, X_test, device)
    overall_acc = float(np.mean(y_pred == y_test))
    log.info(f"  Overall test accuracy (sanity check): {overall_acc:.4f}")

    step1_result = step1_hulk_ftp_diagnosis(X_test, y_test, y_pred, feature_names, label_map)
    step2_result = step2_web_separation(X_test, y_test, feature_names)
    step3_summary(step1_result, step2_result, label_map)

    log.info("")
    log.info(f"✓  diagnose_nn_confusions.py complete.  Log → {LOG_PATH}")


if __name__ == "__main__":
    main()
