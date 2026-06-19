"""
compare_models.py — Side-by-Side Comparison: Autoencoder v1 vs v2
==================================================================

PURPOSE:
  Loads both trained autoencoders and their thresholds, runs inference on
  the full X_val_benign.npy + X_attacks.npy, and prints:

    1. Aggregate metrics (TPR, FPR, F1) for both models
    2. Per-attack-class detection rate comparison table (v1 vs v2, delta)
    3. Highlighted comparison for FTP-Patator and Botnet_ARES specifically
       (the classes whose per-feature signal analysis motivated v2)

  Does NOT modify any model files or thresholds.

MODELS:
  v1 : models/autoencoder.pt      + models/threshold.json
       Architecture: 115 → 64 → 32 → 16 → 32 → 64 → 115
  v2 : models/autoencoder_v2.pt   + models/threshold_v2.json
       Architecture: 115 → 80 → 48 → 24 → 48 → 80 → 115

USAGE (from project root):
    python training/compare_models.py
"""

import os
import sys
import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# Path anchoring
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

BATCH_SIZE  = 512
INPUT_DIM   = 115
DROPOUT_P   = 0.2

FOCUS_CLASSES = ["FTP-Patator", "Botnet_ARES"]   # classes that motivated v2

# ─────────────────────────────────────────────
# Logging  (console only — UTF-8 safe)
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
# Model definitions — must match their respective train scripts exactly
# ═══════════════════════════════════════════════════════════
class AutoencoderV1(nn.Module):
    """v1: 115 → 64 → 32 → 16 (bottleneck) → 32 → 64 → 115"""
    def __init__(self, input_dim: int = INPUT_DIM, dropout_p: float = DROPOUT_P):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(32, 16),        nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderV2(nn.Module):
    """v2: 115 → 80 → 48 → 24 (bottleneck) → 48 → 80 → 115"""
    def __init__(self, input_dim: int = INPUT_DIM, dropout_p: float = DROPOUT_P):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 80), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(80, 48),        nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(48, 24),        nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(24, 48), nn.ReLU(),
            nn.Linear(48, 80), nn.ReLU(),
            nn.Linear(80, input_dim), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
def load_threshold(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def compute_reconstruction_errors(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Per-sample MSE reconstruction error. Returns shape (N,)."""
    model.eval()
    X_t    = torch.from_numpy(X.astype(np.float32))
    ds     = TensorDataset(X_t, X_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    chunks = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            x_recon = model(x_batch)
            mse     = ((x_recon - x_batch) ** 2).mean(dim=1)
            chunks.append(mse.cpu().numpy())
    return np.concatenate(chunks)


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def aggregate_metrics(errors: np.ndarray, threshold: float,
                      n_benign: int, n_attack: int,
                      errors_benign: np.ndarray) -> dict:
    """Compute TPR, FPR, Precision, F1 for a given error array and threshold."""
    tp        = int((errors        > threshold).sum())
    fp        = int((errors_benign > threshold).sum())
    tpr       = tp / n_attack if n_attack > 0 else 0.0
    fpr       = fp / n_benign if n_benign > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = compute_f1(precision, tpr)
    return {"tp": tp, "fp": fp, "tpr": tpr, "fpr": fpr,
            "precision": precision, "f1": f1}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 75)
    print("  MODEL COMPARISON — Autoencoder v1 (16-dim) vs v2 (24-dim)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device : {device}")

    # ── Load both models ──────────────────────────────────────────────
    log.info("\n[1] Loading models and thresholds...")

    v1_weights = os.path.join(MODEL_DIR, "autoencoder.pt")
    v2_weights = os.path.join(MODEL_DIR, "autoencoder_v2.pt")
    v1_thresh_path = os.path.join(MODEL_DIR, "threshold.json")
    v2_thresh_path = os.path.join(MODEL_DIR, "threshold_v2.json")

    for path in [v1_weights, v2_weights, v1_thresh_path, v2_thresh_path]:
        if not os.path.exists(path):
            log.error(f"  MISSING: {path}")
            log.error("  Run train_autoencoder.py and train_autoencoder_v2.py first.")
            raise SystemExit(1)

    model_v1 = AutoencoderV1().to(device)
    model_v1.load_state_dict(torch.load(v1_weights, map_location=device))
    model_v1.eval()

    model_v2 = AutoencoderV2().to(device)
    model_v2.load_state_dict(torch.load(v2_weights, map_location=device))
    model_v2.eval()

    thresh_v1 = load_threshold(v1_thresh_path)
    thresh_v2 = load_threshold(v2_thresh_path)

    log.info(f"  v1 model : {v1_weights}  |  threshold = {thresh_v1:.8f}")
    log.info(f"  v2 model : {v2_weights}  |  threshold = {thresh_v2:.8f}")

    # ── Load data ─────────────────────────────────────────────────────
    log.info("\n[2] Loading data arrays...")

    X_val    = np.load(os.path.join(DATA_DIR, "X_val_benign.npy")).astype(np.float32)
    X_attacks = np.load(os.path.join(DATA_DIR, "X_attacks.npy")).astype(np.float32)
    y_str     = np.load(os.path.join(DATA_DIR, "y_attacks_str.npy"), allow_pickle=True)

    log.info(f"  X_val_benign : {X_val.shape}")
    log.info(f"  X_attacks    : {X_attacks.shape}")
    log.info(f"  y_attacks_str: {y_str.shape}  |  classes: {sorted(np.unique(y_str))}")

    # ── Compute errors for BOTH models ───────────────────────────────
    log.info("\n[3] Computing reconstruction errors (v1)...")
    err_benign_v1 = compute_reconstruction_errors(model_v1, X_val, device)
    err_attack_v1 = compute_reconstruction_errors(model_v1, X_attacks, device)

    log.info("[4] Computing reconstruction errors (v2)...")
    err_benign_v2 = compute_reconstruction_errors(model_v2, X_val, device)
    err_attack_v2 = compute_reconstruction_errors(model_v2, X_attacks, device)

    n_benign = len(X_val)
    n_attack = len(X_attacks)

    # ── Aggregate metrics ─────────────────────────────────────────────
    m_v1 = aggregate_metrics(err_attack_v1, thresh_v1, n_benign, n_attack, err_benign_v1)
    m_v2 = aggregate_metrics(err_attack_v2, thresh_v2, n_benign, n_attack, err_benign_v2)

    W = 75
    print(f"\n{'='*W}")
    print(f"  AGGREGATE METRICS  (full val benign + full attack set)")
    print(f"{'='*W}")
    print(f"  {'Metric':30s}  {'v1 (16-dim)':>13}  {'v2 (24-dim)':>13}  {'Delta':>10}")
    print(f"  {'─'*30}  {'─'*13}  {'─'*13}  {'─'*10}")

    rows = [
        ("Threshold",          f"{thresh_v1:.8f}",            f"{thresh_v2:.8f}",          ""),
        ("Benign samples",     f"{n_benign:,}",                f"{n_benign:,}",              ""),
        ("Attack samples",     f"{n_attack:,}",                f"{n_attack:,}",              ""),
        ("TPR (Recall)",       f"{m_v1['tpr']:.4f}",           f"{m_v2['tpr']:.4f}",
         f"{m_v2['tpr']-m_v1['tpr']:+.4f}"),
        ("FPR",                f"{m_v1['fpr']:.4f}",           f"{m_v2['fpr']:.4f}",
         f"{m_v2['fpr']-m_v1['fpr']:+.4f}"),
        ("Precision",          f"{m_v1['precision']:.4f}",     f"{m_v2['precision']:.4f}",
         f"{m_v2['precision']-m_v1['precision']:+.4f}"),
        ("F1 Score",           f"{m_v1['f1']:.4f}",            f"{m_v2['f1']:.4f}",
         f"{m_v2['f1']-m_v1['f1']:+.4f}"),
    ]
    for label, val1, val2, delta in rows:
        print(f"  {label:30s}  {val1:>13}  {val2:>13}  {delta:>10}")

    print(f"  {'─'*30}  {'─'*13}  {'─'*13}  {'─'*10}")

    # ── Per-attack-class comparison ───────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  PER-ATTACK-CLASS DETECTION RATES")
    print(f"{'─'*W}")
    print(f"  {'Attack Class':30s}  {'N':>7}  {'v1 TPR':>8}  {'v2 TPR':>8}  "
          f"{'Delta':>8}  {'Winner':>7}")
    print(f"  {'─'*30}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    unique_classes = sorted(np.unique(y_str))
    class_results = {}   # class → (n, tpr_v1, tpr_v2, delta)

    for cls in unique_classes:
        mask     = (y_str == cls)
        n        = int(mask.sum())
        e_v1     = err_attack_v1[mask]
        e_v2     = err_attack_v2[mask]
        tpr_v1   = float((e_v1 > thresh_v1).sum()) / n
        tpr_v2   = float((e_v2 > thresh_v2).sum()) / n
        delta    = tpr_v2 - tpr_v1
        class_results[cls] = (n, tpr_v1, tpr_v2, delta)

        flag_v1  = "✅" if tpr_v1 >= 0.50 else "⚠️ "
        flag_v2  = "✅" if tpr_v2 >= 0.50 else "⚠️ "
        if   delta >  0.02: winner = "v2 ↑"
        elif delta < -0.02: winner = "v1 ↑"
        else:               winner = "≈tie"

        # Highlight focus classes
        highlight = "  ◄◄◄" if cls in FOCUS_CLASSES else ""
        print(f"  {cls:30s}  {n:>7,}  "
              f"{flag_v1}{tpr_v1:>6.4f}  "
              f"{flag_v2}{tpr_v2:>6.4f}  "
              f"{delta:>+8.4f}  {winner:>7}{highlight}")

    print(f"  {'─'*30}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    # Overall row
    print(f"  {'OVERALL (all attacks)':30s}  {n_attack:>7,}  "
          f"  {m_v1['tpr']:>6.4f}    {m_v2['tpr']:>6.4f}  "
          f"{m_v2['tpr']-m_v1['tpr']:>+8.4f}  {'':>7}")

    # ── Focus class spotlight ─────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  SPOTLIGHT: FTP-Patator and Botnet_ARES")
    print(f"  (The classes whose per-feature signal analysis motivated v2)")
    print(f"{'='*W}")

    for cls in FOCUS_CLASSES:
        if cls not in class_results:
            print(f"  {cls}: not found in attack data.")
            continue
        n, tpr_v1, tpr_v2, delta = class_results[cls]
        print(f"\n  {cls}  (n={n:,})")
        print(f"    v1 detection rate : {tpr_v1:.4f}  ({int(tpr_v1*n):,}/{n:,} detected)")
        print(f"    v2 detection rate : {tpr_v2:.4f}  ({int(tpr_v2*n):,}/{n:,} detected)")
        if delta > 0:
            print(f"    Improvement       : +{delta:.4f}  ← v2 detects "
                  f"{int(delta*n):,} more samples of this class")
        elif delta < 0:
            print(f"    Regression        : {delta:.4f}  ← v2 detects "
                  f"{int(abs(delta)*n):,} fewer samples (v1 was better here)")
        else:
            print(f"    No change.")

    # ── Recommendation ────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  RECOMMENDATION")
    print(f"{'='*W}")
    f1_delta = m_v2["f1"] - m_v1["f1"]
    ftp_delta   = class_results.get("FTP-Patator",   (0,0,0,0))[3]
    botnet_delta = class_results.get("Botnet_ARES",  (0,0,0,0))[3]

    if f1_delta > 0.01:
        print(f"  Overall F1 improved by {f1_delta:+.4f} → v2 is the better general model.")
    elif f1_delta < -0.01:
        print(f"  Overall F1 regressed by {f1_delta:+.4f} → v1 is the better general model.")
    else:
        print(f"  Overall F1 difference is negligible ({f1_delta:+.4f}).")

    if ftp_delta > 0.05 or botnet_delta > 0.05:
        print(f"  FTP-Patator delta={ftp_delta:+.4f}, Botnet_ARES delta={botnet_delta:+.4f}")
        print(f"  → v2 shows meaningful gains on the targeted weak classes.")
        print(f"  → Consider a HYBRID: use v2 as the primary model, or use v2 specifically")
        print(f"    for FTP-Patator/Botnet_ARES in a class-specific ensemble.")
    else:
        print(f"  FTP-Patator delta={ftp_delta:+.4f}, Botnet_ARES delta={botnet_delta:+.4f}")
        print(f"  → Widening the bottleneck alone did not recover the weak classes.")
        print(f"  → Recommend building a supervised hybrid classifier (e.g. XGBoost)")
        print(f"    trained on the raw 115 features for these specific attack types.")

    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
