"""
analyze_feature_errors.py — Per-Feature Reconstruction Error Analysis
======================================================================

PURPOSE:
  Diagnoses WHY certain attack classes are poorly detected by the autoencoder
  by examining per-feature squared reconstruction error (NOT averaged across
  features). For each weak attack class, computes:
    - Mean per-feature error for benign traffic (the "baseline profile")
    - Mean per-feature error for each attack class
    - Difference = attack_error - benign_error for each of the 115 features
  Then ranks the TOP 10 features with the largest positive deviation, which
  shows where the attack class has genuine signal even if the overall
  sample-level MSE stays low.

  Also prints a summary table showing, for each class:
    - Max single-feature error difference
    - How many features have difference > 2*std(benign_feature_error_profile)
      (call this "significant features count")

CLASSES ANALYSED:
  Weak  : SSH-Patator, Web_Brute_Force, Web_XSS, Web_SQL_Injection,
           FTP-Patator, Botnet_ARES
  Strong: DoS_Hulk  (used as a reference / sanity check)

USAGE (from project root):
    python training/analyze_feature_errors.py

OUTPUTS:
    - Console: feature ranking tables + summary
    - data/processed/benign_feature_error_profile.npy  (shape 115)
    - data/processed/attack_feature_error_{class}.npy  (one per class)
"""

import os
import sys
import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# Path anchoring (same pattern as other scripts)
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../training/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

# Ensure training/ is importable so we can reuse Autoencoder
sys.path.insert(0, _SCRIPT_DIR)
from train_autoencoder import Autoencoder   # noqa: E402

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

BATCH_SIZE   = 512
INPUT_DIM    = 115

# Attack classes to analyse
WEAK_CLASSES    = [
    "SSH-Patator",
    "Web_Brute_Force",
    "Web_XSS",
    "Web_SQL_Injection",
    "FTP-Patator",
    "Botnet_ARES",
]
STRONG_CLASS    = "DoS_Hulk"
ALL_CLASSES     = WEAK_CLASSES + [STRONG_CLASS]

# Threshold for "significant" feature deviation
SIG_SIGMA_MULT  = 2.0   # feature diff > 2 * std(benign_feature_error_profile)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
_handler = logging.StreamHandler(sys.stdout)
_handler.stream = open(
    _handler.stream.fileno(),
    mode="w", encoding="utf-8", closefd=False, buffering=1
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_handler],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Per-feature squared error computation
# ═══════════════════════════════════════════════════════════
def compute_per_feature_sq_error(
    model: Autoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Run X through the autoencoder in eval mode.
    Returns per-sample, per-feature squared error — shape (N, 115).
    Does NOT average across the feature dimension, giving us the full error
    matrix needed to identify which specific features drive the deviation.

    Parameters
    ----------
    model     : Autoencoder  (eval mode)
    X         : np.ndarray   shape (N, 115), float32
    device    : torch.device
    batch_size: int

    Returns
    -------
    sq_errors : np.ndarray  shape (N, 115)
    """
    model.eval()
    X_t    = torch.from_numpy(X.astype(np.float32))
    ds     = TensorDataset(X_t, X_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    chunks = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            x_recon = model(x_batch)
            sq_err  = (x_recon - x_batch) ** 2   # (batch, 115) — NOT averaged
            chunks.append(sq_err.cpu().numpy())

    return np.concatenate(chunks, axis=0)   # (N, 115)


# ═══════════════════════════════════════════════════════════
# Printing helpers
# ═══════════════════════════════════════════════════════════
def print_top10_features(
    class_name: str,
    feature_names: list,
    benign_profile: np.ndarray,
    attack_profile: np.ndarray,
    n_samples: int,
):
    """
    Print the top-10 features (by attack - benign difference) for one class.
    """
    diff = attack_profile - benign_profile
    top10_idx = np.argsort(diff)[::-1][:10]

    W = 78
    label = f"  {'STRONG':7s}: {class_name}" if class_name == STRONG_CLASS \
            else f"  {'WEAK':7s}: {class_name}"
    print(f"\n{'─'*W}")
    print(label + f"  (n={n_samples:,}  |  mean sample-MSE = {attack_profile.mean():.6f})")
    print(f"{'─'*W}")
    print(f"  {'#':>3}  {'Feature':35s}  {'Benign Err':>11}  {'Attack Err':>11}  {'Diff':>11}")
    print(f"  {'─'*3}  {'─'*35}  {'─'*11}  {'─'*11}  {'─'*11}")

    for rank, idx in enumerate(top10_idx, 1):
        fname   = feature_names[idx]
        b_err   = benign_profile[idx]
        a_err   = attack_profile[idx]
        d_err   = diff[idx]
        arrow   = "▲" if d_err > 0 else "▼"
        print(f"  {rank:>3}  {fname:35s}  {b_err:>11.6f}  {a_err:>11.6f}  {arrow}{abs(d_err):>10.6f}")

    # Also show overall mean error ratio for context
    overall_diff = attack_profile.mean() - benign_profile.mean()
    max_diff     = diff.max()
    pos_features = int((diff > 0).sum())
    print(f"\n  Overall mean-error diff (attack − benign): {overall_diff:+.6f}")
    print(f"  Max single-feature diff:                   {max_diff:+.6f}")
    print(f"  Features where attack error > benign:      {pos_features}/115")


def print_summary_table(
    feature_names: list,
    benign_profile: np.ndarray,
    class_profiles: dict,   # class_name → mean per-feature error (115,)
    class_sizes: dict,      # class_name → int
):
    """
    Summary table: max single-feature diff and count of "significant" features
    (diff > SIG_SIGMA_MULT * std_of_benign_feature_error_profile).
    """
    benign_std   = benign_profile.std()
    sig_threshold = SIG_SIGMA_MULT * benign_std

    W = 78
    print(f"\n{'='*W}")
    print(f"  SUMMARY TABLE  —  Significant Feature Analysis")
    print(f"  (Significant = feature diff > {SIG_SIGMA_MULT:.0f}×std(benign_profile)  "
          f"i.e. > {sig_threshold:.6f})")
    print(f"{'='*W}")
    print(f"  {'Attack Class':30s}  {'N':>7}  {'Max Diff':>11}  "
          f"{'Sig Features':>13}  {'Detected?':>10}")
    print(f"  {'─'*30}  {'─'*7}  {'─'*11}  {'─'*13}  {'─'*10}")

    print(f"  {'[baseline] BENIGN':30s}  "
          f"{'-':>7}  {'─':>11}  {'─':>13}  {'N/A':>10}")

    for cls in ALL_CLASSES:
        profile  = class_profiles[cls]
        diff     = profile - benign_profile
        max_diff = float(diff.max())
        sig_cnt  = int((diff > sig_threshold).sum())
        n        = class_sizes[cls]
        quality  = "WEAK  ⚠️ " if cls in WEAK_CLASSES else "STRONG ✅"

        # Name of the most-deviating feature
        top_feat = feature_names[int(np.argmax(diff))]
        print(f"  {cls:30s}  {n:>7,}  {max_diff:>11.6f}  "
              f"{sig_cnt:>7d} / 115  {quality:>10s}")

    print(f"  {'─'*30}  {'─'*7}  {'─'*11}  {'─'*13}  {'─'*10}")
    print(f"\n  Interpretation:")
    print(f"  ─────────────")
    print(f"  • High max-diff + many sig features → class IS distinguishable by the")
    print(f"    autoencoder in principle, but threshold may need adjustment.")
    print(f"  • Low max-diff + few/no sig features → class produces features that")
    print(f"    look almost identical to benign traffic. No threshold trick will")
    print(f"    fix this; the model needs richer architecture or a hybrid classifier.")
    print(f"{'='*W}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 78)
    print("  PER-FEATURE RECONSTRUCTION ERROR ANALYSIS")
    print("  (Diagnosing weak attack-class detection without retraining)")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device : {device}")

    # ── Load feature names ────────────────────────────────────────────
    feat_path = os.path.join(DATA_DIR, "feature_names.json")
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    assert len(feature_names) == INPUT_DIM, \
        f"Expected {INPUT_DIM} features, got {len(feature_names)}"
    log.info(f"  Loaded {len(feature_names)} feature names from {feat_path}")

    # ── Load model ────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "autoencoder.pt")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    log.info(f"  Loaded model from : {model_path}")

    # ── Load attacks + labels ─────────────────────────────────────────
    log.info("\n[STEP 1]  Loading X_attacks.npy and y_attacks_str.npy...")
    X_attacks     = np.load(os.path.join(DATA_DIR, "X_attacks.npy")).astype(np.float32)
    y_attacks_str = np.load(os.path.join(DATA_DIR, "y_attacks_str.npy"), allow_pickle=True)
    log.info(f"  X_attacks shape     : {X_attacks.shape}")
    log.info(f"  y_attacks_str shape : {y_attacks_str.shape}")
    log.info(f"  Unique classes      : {sorted(np.unique(y_attacks_str))}")

    # ── Compute BENIGN feature error profile (full val set) ──────────
    log.info("\n[STEP 2]  Computing per-feature squared error on X_val_benign (full val set)...")
    X_val = np.load(os.path.join(DATA_DIR, "X_val_benign.npy")).astype(np.float32)
    log.info(f"  X_val_benign shape : {X_val.shape}")

    benign_sq_err_matrix = compute_per_feature_sq_error(model, X_val, device)  # (N_val, 115)
    benign_profile       = benign_sq_err_matrix.mean(axis=0)                   # (115,)

    out_benign = os.path.join(DATA_DIR, "benign_feature_error_profile.npy")
    np.save(out_benign, benign_profile)
    log.info(f"  Saved benign feature error profile → {out_benign}")
    log.info(f"  Benign profile — min={benign_profile.min():.6f}  "
             f"mean={benign_profile.mean():.6f}  max={benign_profile.max():.6f}  "
             f"std={benign_profile.std():.6f}")

    # ── Compute ATTACK feature error profiles for each class ──────────
    log.info(f"\n[STEP 3]  Computing per-feature error profiles for {len(ALL_CLASSES)} attack classes...")

    class_profiles: dict[str, np.ndarray] = {}
    class_sizes:    dict[str, int]         = {}

    for cls in ALL_CLASSES:
        mask  = (y_attacks_str == cls)
        n     = int(mask.sum())
        class_sizes[cls] = n

        if n == 0:
            log.warning(f"  {cls}: 0 samples found — skipping")
            class_profiles[cls] = benign_profile.copy()
            continue

        X_cls = X_attacks[mask]
        log.info(f"  {cls}: {n:,} samples — computing per-feature error matrix...")

        sq_err_matrix       = compute_per_feature_sq_error(model, X_cls, device)  # (n, 115)
        attack_profile      = sq_err_matrix.mean(axis=0)                           # (115,)
        class_profiles[cls] = attack_profile

        out_cls = os.path.join(DATA_DIR, f"attack_feature_error_{cls.replace('-','_')}.npy")
        np.save(out_cls, attack_profile)
        log.info(f"  {cls}: mean sample-MSE={attack_profile.mean():.6f}  → saved {out_cls}")

    # ── STEP 4+5: Top-10 feature tables ──────────────────────────────
    print(f"\n{'='*78}")
    print(f"  TOP-10 FEATURES BY (attack − benign) ERROR DIFFERENCE")
    print(f"  (↑ = where this attack class deviates most from benign baseline)")
    print(f"{'='*78}")

    # Weak classes first
    for cls in WEAK_CLASSES:
        print_top10_features(
            cls,
            feature_names,
            benign_profile,
            class_profiles[cls],
            class_sizes[cls],
        )

    # Strong reference class
    print_top10_features(
        STRONG_CLASS,
        feature_names,
        benign_profile,
        class_profiles[STRONG_CLASS],
        class_sizes[STRONG_CLASS],
    )

    # ── STEP 6: Summary table ─────────────────────────────────────────
    log.info("\n[STEP 6]  Building summary table...")
    print_summary_table(feature_names, benign_profile, class_profiles, class_sizes)

    print("\n  Done. Next steps based on results:")
    print("  ─────────────────────────────────")
    print("  If weak classes have sig_features ≥ 5:")
    print("    → Widen bottleneck (try 32 dims) and retrain — signal exists but")
    print("      the current 16-dim bottleneck compresses it away.")
    print("  If weak classes have sig_features < 5 and max_diff < 0.001:")
    print("    → Build a hybrid: autoencoder for DoS/DDoS/PortScan,")
    print("      + a shallow supervised classifier (RF or XGBoost) for the weak")
    print("      classes using raw feature differences as inputs.")
    print()


if __name__ == "__main__":
    main()
