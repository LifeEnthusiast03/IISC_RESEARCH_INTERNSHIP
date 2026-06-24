"""
build_dqn_environment.py — Stage 3 Data Preparation (DQN State Vectors)
========================================================================

PURPOSE:
  Constructs the full DQN state vectors by running the trained Autoencoder
  (Stage 1A) and Attack-Type NN (Stage 2) forward passes over the attack
  dataset and a sample of benign flows.

  State vector per flow:
    [115 MinMax-scaled features]              -- raw flow representation
    [1 autoencoder reconstruction error]      -- MSE(x, x̂)
    [N attack-type NN softmax probabilities]  -- calibrated distribution over N classes
    [1 max-probability confidence score]      -- max(softmax output)
    ─────────────────────────────────────────
    Total dim = 115 + 1 + N + 1  (where N = number of attack-type classes, ~11)

  NO models are retrained here.  This script only runs forward passes with
  the already-trained weights saved in models/.

GROUND-TRUTH ACTION TABLE (used only to compute rewards — never given to agent):
  DoS_Hulk          → 0 (Block IP)
  DDoS_LOIT         → 0 (Block IP)
  Port_Scan         → 0 (Block IP)
  DoS_GoldenEye     → 0 (Block IP)
  DoS_Slowloris     → 0 (Block IP)
  DoS_Slowhttptest  → 0 (Block IP)
  FTP-Patator       → 1 (Revoke Credentials)
  SSH-Patator       → 1 (Revoke Credentials)
  Web_Brute_Force   → 1 (Revoke Credentials)
  Botnet_ARES       → 2 (Isolate Server)
  Web_XSS           → 3 (Kill Process)
  Benign            → 4 (Monitor)
  (Heartbleed / Web_SQL_Injection: not in attack_type_label_map → defaulted to Monitor)

INPUTS:
  data/processed/X_attacks.npy          (600,141 × 115, MinMax-scaled attack flows)
  data/processed/y_attacks_str.npy      (600,141,) string attack-type labels
  data/processed/X_train_benign.npy     (all benign training flows — sampled here)
  data/processed/attack_type_label_map.json
  models/autoencoder.pt
  models/attack_type_nn.pt

OUTPUTS (data/processed/):
  dqn_states.npy              — full state vectors  (M × state_dim)
  dqn_optimal_actions.npy     — ground-truth action integer per row
  dqn_labels_str.npy          — string label per row (for per-class eval in train_dqn.py)
  dqn_train_states.npy        — 70% stratified split
  dqn_train_actions.npy
  dqn_train_labels.npy
  dqn_val_states.npy          — 15% stratified split
  dqn_val_actions.npy
  dqn_val_labels.npy
  dqn_test_states.npy         — 15% stratified split
  dqn_test_actions.npy
  dqn_test_labels.npy

USAGE (from project root):
    python training/build_dqn_environment.py
"""

import os
import sys
import json
import logging

import numpy as np
import torch

# Add training/ to path so we can import from sibling scripts
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from train_autoencoder   import Autoencoder, INPUT_DIM as AE_INPUT_DIM
from train_attack_type_nn import AttackTypeNN

from sklearn.model_selection import train_test_split

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

# ─────────────────────────────────────────────
# Ground-truth optimal action mapping
# ─────────────────────────────────────────────
ACTION_MAP = {
    "DoS_Hulk":         0,   # Block IP
    "DDoS_LOIT":        0,   # Block IP
    "Port_Scan":        0,   # Block IP
    "DoS_GoldenEye":    0,   # Block IP
    "DoS_Slowloris":    0,   # Block IP
    "DoS_Slowhttptest": 0,   # Block IP
    "FTP-Patator":      1,   # Revoke Credentials
    "SSH-Patator":      1,   # Revoke Credentials
    "Web_Brute_Force":  1,   # Revoke Credentials
    "Botnet_ARES":      2,   # Isolate Server
    # Web_Brute_Force and Web_XSS are merged to a single action due to a confirmed
    # feature-level ceiling (diagnose_nn_confusions.py, max norm_sep=0.198 across all
    # 115 features) — both map to Revoke Credentials as the safer unified response for
    # ambiguous web attacks.  Web_Brute_Force is fundamentally credential-based, and
    # Revoke Credentials is more conservative than Kill Process when the classification
    # is uncertain between the two types.  (Previously: Web_XSS → 3 Kill Process.)
    "Web_XSS":          1,   # Revoke Credentials (merged with Web_Brute_Force — see above)
    "Benign":           4,   # Monitor
    # Excluded from Attack-Type NN — default to Monitor (conservative)
    "Heartbleed":          4,
    "Web_SQL_Injection":   4,
}
ACTION_NAMES = {
    0: "Block IP",
    1: "Revoke Credentials",
    2: "Isolate Server",
    3: "Kill Process",
    4: "Monitor",
}

# Benign sample size: use a count roughly equal to the smallest
# reasonably-sized attack class (DoS_Slowloris ≈ 5,122 rows)
BENIGN_SAMPLE_SIZE = 5000
RANDOM_STATE = 42

VAL_FRAC  = 0.15
TEST_FRAC = 0.15

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_PATH = os.path.join(DATA_DIR, "build_dqn_environment.log")
_sh = logging.StreamHandler()
_sh.stream = open(_sh.stream.fileno(), mode="w", encoding="utf-8", closefd=False, buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[_sh, logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load models
# ═══════════════════════════════════════════════════════════
def load_models(model_dir: str, data_dir: str, device):
    """
    Load the trained Autoencoder and Attack-Type NN from their state_dicts.

    Returns
    -------
    ae       : Autoencoder (eval mode)
    atnn     : AttackTypeNN (eval mode)
    n_classes: int
    label_map: dict {int: str}
    """
    log.info("[STEP 1] Loading trained models ...")

    # ── Label map ─────────────────────────────────────────
    lm_path = os.path.join(data_dir, "attack_type_label_map.json")
    with open(lm_path, "r") as f:
        label_map_raw = json.load(f)
    label_map = {int(k): v for k, v in label_map_raw.items()}
    n_classes = len(label_map)
    log.info(f"  Attack-Type NN classes: {n_classes}")

    # ── Autoencoder ────────────────────────────────────────
    ae = Autoencoder(input_dim=AE_INPUT_DIM)
    ae_path = os.path.join(model_dir, "autoencoder.pt")
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae = ae.to(device).eval()
    log.info(f"  Autoencoder loaded from {ae_path}")

    # ── Attack-Type NN ─────────────────────────────────────
    atnn = AttackTypeNN(input_dim=AE_INPUT_DIM, n_classes=n_classes)
    atnn_path = os.path.join(model_dir, "attack_type_nn.pt")
    atnn.load_state_dict(torch.load(atnn_path, map_location=device))
    atnn = atnn.to(device).eval()
    log.info(f"  AttackTypeNN loaded from {atnn_path}")

    return ae, atnn, n_classes, label_map


# ═══════════════════════════════════════════════════════════
# STEP 2 — Load raw flow arrays and sample benign rows
# ═══════════════════════════════════════════════════════════
def load_flow_data(data_dir: str):
    """
    Load attack flows (X_attacks + y_attacks_str) and a stratified sample
    of benign flows (X_train_benign).

    We sample BENIGN_SAMPLE_SIZE benign rows to avoid extreme benign/attack
    imbalance in the DQN simulation environment.

    Returns
    -------
    X_all : np.ndarray  float32  (M × 115)
    y_all : np.ndarray  str      (M,)      — string labels ("Benign" for benign rows)
    """
    log.info("[STEP 2] Loading flow arrays ...")

    X_attacks = np.load(os.path.join(data_dir, "X_attacks.npy"))
    y_attacks  = np.load(os.path.join(data_dir, "y_attacks_str.npy"),
                         allow_pickle=True).astype(str)

    X_benign_full = np.load(os.path.join(data_dir, "X_train_benign.npy"))
    rng = np.random.default_rng(seed=RANDOM_STATE)
    benign_idx = rng.choice(len(X_benign_full), size=BENIGN_SAMPLE_SIZE, replace=False)
    X_benign_sampled = X_benign_full[benign_idx].astype(np.float32)
    y_benign = np.full(BENIGN_SAMPLE_SIZE, "Benign", dtype=object)

    X_all = np.concatenate([X_attacks, X_benign_sampled], axis=0)
    y_all = np.concatenate([y_attacks,  y_benign],         axis=0).astype(str)

    log.info(f"  Attack rows : {X_attacks.shape[0]:,}")
    log.info(f"  Benign rows : {BENIGN_SAMPLE_SIZE:,}  (sampled from X_train_benign)")
    log.info(f"  Total rows  : {X_all.shape[0]:,}")

    # Per-class breakdown
    unique, counts = np.unique(y_all, return_counts=True)
    log.info("  Class distribution:")
    for cls, cnt in sorted(zip(unique, counts), key=lambda t: -t[1]):
        log.info(f"    {cls:<30s}  {cnt:>8,}")

    return X_all, y_all


# ═══════════════════════════════════════════════════════════
# STEP 3 — Build state vectors via forward passes
# ═══════════════════════════════════════════════════════════
def build_state_vectors(X_all: np.ndarray, ae, atnn, n_classes: int,
                        device, batch_size: int = 1024) -> np.ndarray:
    """
    Compute the full DQN state vector for every row in X_all.

    State = [x (115) | ae_error (1) | softmax_probs (N) | confidence (1)]
    Total dim = 117 + N

    We process X_all in mini-batches to avoid OOM on large arrays.

    Parameters
    ----------
    X_all      : np.ndarray  float32  (M × 115)
    ae         : Autoencoder  (eval, on device)
    atnn       : AttackTypeNN (eval, on device)
    n_classes  : int
    device     : torch.device
    batch_size : int

    Returns
    -------
    states : np.ndarray  float32  (M × state_dim)
    """
    state_dim = X_all.shape[1] + 1 + n_classes + 1  # 115 + 1 + N + 1
    log.info(f"[STEP 3] Building state vectors ... state_dim={state_dim}")

    M = X_all.shape[0]
    states = np.empty((M, state_dim), dtype=np.float32)

    n_batches = (M + batch_size - 1) // batch_size
    log.info(f"  Processing {M:,} rows in {n_batches:,} batches of {batch_size} ...")

    softmax_fn = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end   = min(start + batch_size, M)
            x_np  = X_all[start:end]
            x_t   = torch.tensor(x_np, dtype=torch.float32, device=device)

            # ── Autoencoder reconstruction error ──────────
            x_hat   = ae(x_t)
            ae_err  = ((x_t - x_hat) ** 2).mean(dim=1, keepdim=True)  # (B, 1)

            # ── Attack-Type NN softmax probs ───────────────
            logits  = atnn(x_t)
            probs   = softmax_fn(logits)                                # (B, N)
            conf    = probs.max(dim=1, keepdim=True).values             # (B, 1)

            # ── Concatenate ────────────────────────────────
            state = torch.cat([x_t, ae_err, probs, conf], dim=1)       # (B, state_dim)

            states[start:end] = state.cpu().numpy()

            if (b_idx + 1) % max(1, n_batches // 10) == 0 or b_idx == n_batches - 1:
                log.info(f"  Batch {b_idx + 1:>5}/{n_batches}  rows {end:>7,}/{M:,}")

    log.info(f"  State matrix shape: {states.shape}  dtype={states.dtype}")
    return states


# ═══════════════════════════════════════════════════════════
# STEP 4 — Map string labels to optimal action integers
# ═══════════════════════════════════════════════════════════
def build_action_labels(y_str: np.ndarray) -> np.ndarray:
    """
    For each row, look up the ground-truth optimal action from ACTION_MAP.

    Any label not in ACTION_MAP defaults to Monitor (4) with a warning.
    The optimal actions are used ONLY for computing rewards during DQN
    training — they are never passed directly to the agent as input.

    Returns
    -------
    actions : np.ndarray  int64  (M,)
    """
    log.info("[STEP 4] Mapping string labels to optimal action integers ...")
    actions = np.empty(len(y_str), dtype=np.int64)
    unknown = set()
    for i, label in enumerate(y_str):
        if label in ACTION_MAP:
            actions[i] = ACTION_MAP[label]
        else:
            actions[i] = 4   # default: Monitor
            unknown.add(label)

    if unknown:
        log.warning(f"  Unknown labels defaulted to Monitor: {unknown}")

    for action_id, action_name in ACTION_NAMES.items():
        cnt = int(np.sum(actions == action_id))
        log.info(f"  Action {action_id} ({action_name:<22s}): {cnt:>8,} rows")

    return actions


# ═══════════════════════════════════════════════════════════
# STEP 5 — Stratified 70 / 15 / 15 split
# ═══════════════════════════════════════════════════════════
def split_data(states, actions, labels):
    """
    Stratify by string label (not by action — labels carry more granularity).

    Returns three (states, actions, labels) tuples for train / val / test.
    """
    log.info("[STEP 5] Stratified 70 / 15 / 15 split ...")

    # First pass: split off test
    idx = np.arange(len(states))
    idx_tv, idx_test = train_test_split(
        idx, test_size=TEST_FRAC, random_state=RANDOM_STATE, stratify=labels
    )

    # Second pass: split train+val
    val_of_tv = VAL_FRAC / (1.0 - TEST_FRAC)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val_of_tv, random_state=RANDOM_STATE, stratify=labels[idx_tv]
    )

    for split_name, split_idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        log.info(f"  {split_name}: {len(split_idx):>7,} rows")

    return (
        (states[idx_train], actions[idx_train], labels[idx_train]),
        (states[idx_val],   actions[idx_val],   labels[idx_val]),
        (states[idx_test],  actions[idx_test],  labels[idx_test]),
    )


# ═══════════════════════════════════════════════════════════
# STEP 6 — Save all arrays
# ═══════════════════════════════════════════════════════════
def save_arrays(data_dir, states, actions, labels, train_t, val_t, test_t):
    log.info("[STEP 6] Saving arrays to data/processed/ ...")

    np.save(os.path.join(data_dir, "dqn_states.npy"),          states)
    np.save(os.path.join(data_dir, "dqn_optimal_actions.npy"), actions)
    np.save(os.path.join(data_dir, "dqn_labels_str.npy"),      labels)

    for prefix, (s, a, l) in [("train", train_t), ("val", val_t), ("test", test_t)]:
        np.save(os.path.join(data_dir, f"dqn_{prefix}_states.npy"),  s)
        np.save(os.path.join(data_dir, f"dqn_{prefix}_actions.npy"), a)
        np.save(os.path.join(data_dir, f"dqn_{prefix}_labels.npy"),  l)
        log.info(f"  {prefix:<5}: states={s.shape}  actions={a.shape}  labels={l.shape}")

    log.info(f"  dqn_states          : {states.shape}")
    log.info(f"  dqn_optimal_actions : {actions.shape}")
    log.info(f"  dqn_labels_str      : {labels.shape}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("build_dqn_environment.py — Stage 3 Data Preparation")
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")

    ae, atnn, n_classes, label_map = load_models(MODEL_DIR, DATA_DIR, device)
    X_all, y_all = load_flow_data(DATA_DIR)
    states  = build_state_vectors(X_all, ae, atnn, n_classes, device)
    actions = build_action_labels(y_all)
    labels  = y_all

    train_t, val_t, test_t = split_data(states, actions, labels)
    save_arrays(DATA_DIR, states, actions, labels, train_t, val_t, test_t)

    state_dim = states.shape[1]
    log.info("")
    log.info("✓  build_dqn_environment.py complete.")
    log.info(f"   State dimension : {state_dim}  (115 + 1 + {n_classes} + 1)")
    log.info(f"   Total rows      : {len(states):,}")
    log.info(f"   DQN splits      : {len(train_t[0]):,} train | "
             f"{len(val_t[0]):,} val | {len(test_t[0]):,} test")
    log.info("   Next step: python training/train_dqn.py")


if __name__ == "__main__":
    main()
