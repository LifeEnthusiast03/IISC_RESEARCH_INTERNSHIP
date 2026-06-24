"""
train_attack_type_nn.py — Stage 2: Attack-Type Neural Network
==============================================================

PURPOSE:
  Trains a PyTorch feedforward multi-class classifier that identifies WHICH
  of the 11 CICIDS2017 attack types (post-exclusion) a flow belongs to.

  This NN operates DOWNSTREAM of Stage 1 (Autoencoder + Hybrid Classifier),
  so at inference time it only sees flows already confirmed as anomalous.
  Its softmax output vector becomes part of the DQN agent's state in Stage 3,
  giving the agent a calibrated distribution over attack types rather than a
  single hard prediction.

  EXCLUDED CLASSES (handled by upstream AE reconstruction error):
    - Heartbleed       (n=12)
    - Web_SQL_Injection (n=24)

MODEL ARCHITECTURE:
  Input   : 115 features (MinMax-scaled network flow statistics)
  Hidden 1: 115 → 128  (ReLU + Dropout(0.2))
  Hidden 2: 128 → 64   (ReLU + Dropout(0.2))
  Output  : 64  → N    (N = number of classes from attack_type_label_map.json)
  Loss    : CrossEntropyLoss with per-class weights (handles severe imbalance;
            DoS_Hulk has 208K+ training rows vs Web_XSS ~950 rows)

TRAINING:
  - Optimizer  : Adam (lr=1e-3, weight_decay=1e-5)
  - Batch size : 256
  - Max epochs : 50
  - Early stop : patience=5 on val_loss (restores best weights)
  - Device     : CUDA if available, else CPU

INPUTS (data/processed/):
  - X_train_attacktype.npy, y_train_attacktype.npy
  - X_val_attacktype.npy,   y_val_attacktype.npy
  - X_test_attacktype.npy,  y_test_attacktype.npy
  - sample_weights_train_attacktype.npy
  - attack_type_label_map.json

OUTPUTS:
  - models/attack_type_nn.pt           — best model state_dict
  - models/attack_type_nn_history.json — train/val loss per epoch

USAGE (from project root):
    python training/train_attack_type_nn.py
"""

import os
import json
import copy
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ─────────────────────────────────────────────
# Path anchoring — same pattern as train_autoencoder.py
# ─────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
INPUT_DIM   = 115
HIDDEN_1    = 128
HIDDEN_2    = 64
DROPOUT_P   = 0.2
BATCH_SIZE  = 256
LR          = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS  = 50
PATIENCE    = 5

# Below this per-class recall, we flag the class for attention before Phase 3
RECALL_FLAG_THRESHOLD = 0.70

# ─────────────────────────────────────────────
# Logging setup — console + file, UTF-8 safe on Windows
# ─────────────────────────────────────────────
LOG_PATH = os.path.join(DATA_DIR, "train_attack_type_nn.log")

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
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════
class AttackTypeNN(nn.Module):
    """
    Feedforward multi-class classifier for CICIDS2017 attack-type identification.

    Architecture
    ------------
    Linear(115 → 128)  → ReLU → Dropout(0.2)
    Linear(128 → 64)   → ReLU → Dropout(0.2)
    Linear(64  → N)    — raw logits (CrossEntropyLoss applies softmax internally)

    At inference time, call softmax on the output to obtain a calibrated
    probability vector over N attack types.  This probability vector becomes
    part of the DQN agent's state in Stage 3.

    Parameters
    ----------
    input_dim  : int   number of input features (115)
    n_classes  : int   number of attack type classes (from label map)
    dropout_p  : float dropout probability (0.2)
    """

    def __init__(self, input_dim: int = INPUT_DIM, n_classes: int = 11,
                 dropout_p: float = DROPOUT_P):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(HIDDEN_2, n_classes),  # raw logits — CE handles softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, input_dim)

        Returns
        -------
        logits : torch.Tensor  shape (batch, n_classes)
            Raw logits.  Apply F.softmax(dim=1) to obtain probabilities.
        """
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load arrays and label map
# ═══════════════════════════════════════════════════════════
def load_data(data_dir: str):
    """
    Load the 70/15/15 split arrays produced by build_attack_type_dataset.py,
    plus the sample weights and the label map.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray  float32
    y_train, y_val, y_test : np.ndarray  int64
    sample_weights         : np.ndarray  float32  (train only)
    label_map              : dict {int: str}
    """
    log.info("[STEP 1] Loading split arrays ...")

    X_train = np.load(os.path.join(data_dir, "X_train_attacktype.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val_attacktype.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test_attacktype.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train_attacktype.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val_attacktype.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test_attacktype.npy"))
    sample_weights = np.load(os.path.join(data_dir, "sample_weights_train_attacktype.npy"))

    with open(os.path.join(data_dir, "attack_type_label_map.json"), "r") as f:
        label_map_raw = json.load(f)
    # JSON keys are strings; convert to int
    label_map = {int(k): v for k, v in label_map_raw.items()}
    n_classes = len(label_map)

    log.info(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
    log.info(f"  X_val   : {X_val.shape}  y_val   : {y_val.shape}")
    log.info(f"  X_test  : {X_test.shape}  y_test  : {y_test.shape}")
    log.info(f"  n_classes = {n_classes}")
    log.info(f"  sample_weights shape = {sample_weights.shape}  "
             f"range=[{sample_weights.min():.6f}, {sample_weights.max():.4f}]")

    return X_train, X_val, X_test, y_train, y_val, y_test, sample_weights, label_map


# ═══════════════════════════════════════════════════════════
# STEP 2 — Build DataLoaders
# ═══════════════════════════════════════════════════════════
def build_loaders(X_train, y_train, X_val, y_val, device):
    """
    Convert numpy arrays to PyTorch TensorDatasets and wrap in DataLoaders.

    The training loader includes sample weights (used by WeightedCrossEntropyLoss
    below); val loader does NOT need weights (evaluation only).

    Returns
    -------
    train_loader : DataLoader  (X, y, weight)
    val_loader   : DataLoader  (X, y)
    """
    log.info("[STEP 2] Building DataLoaders ...")

    # NOTE: we do NOT move weights to device here; they're used differently
    # (aggregated into per-class class_weights for the criterion — see step 3).
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.long)

    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset   = TensorDataset(X_v, y_v)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=(device.type == "cuda"))

    log.info(f"  train batches: {len(train_loader):,}  "
             f"val batches: {len(val_loader):,}")
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════
# STEP 3 — Compute per-class weights for CrossEntropyLoss
# ═══════════════════════════════════════════════════════════
def compute_class_weights(y_train: np.ndarray, n_classes: int, device) -> torch.Tensor:
    """
    Derive per-class scalar weights from the balanced sample weights.

    Strategy: for each class c, its weight = mean of sample_weights for all
    training rows belonging to class c.  This is equivalent to the 'balanced'
    strategy: n_samples / (n_classes * count_c).

    These weights are passed to nn.CrossEntropyLoss(weight=...) so that
    rare classes (Web_XSS, ~950 rows) are not overwhelmed by dominant classes
    (DoS_Hulk, ~208K rows) during backpropagation.

    Parameters
    ----------
    y_train   : np.ndarray  int64
    n_classes : int
    device    : torch.device

    Returns
    -------
    class_weights : torch.Tensor  float32, shape (n_classes,)
    """
    log.info("[STEP 3] Computing per-class weights for CrossEntropyLoss ...")
    n_samples = len(y_train)
    counts = np.bincount(y_train, minlength=n_classes)
    # Balanced weight: n_samples / (n_classes * count_c)
    class_weights = np.where(
        counts > 0,
        n_samples / (n_classes * counts.astype(float)),
        0.0,
    ).astype(np.float32)

    log.info("  Per-class weights (class_int → count | weight):")
    for c in range(n_classes):
        log.info(f"    class {c:>2}  count={counts[c]:>8,}  weight={class_weights[c]:.4f}")

    return torch.tensor(class_weights, dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════
# STEP 4 — Training loop with early stopping
# ═══════════════════════════════════════════════════════════
def train(model, train_loader, val_loader, class_weights, device, model_dir):
    """
    Train AttackTypeNN with early stopping on val_loss.

    Early stopping: if val_loss does not improve for PATIENCE epochs, restore
    the best weights and terminate.  Consistent with train_autoencoder.py.

    Returns
    -------
    history : dict  {"train_loss": [...], "val_loss": [...]}  one entry per epoch
    """
    log.info("[STEP 4] Starting training ...")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss   = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Training phase ───────────────────────────────
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ── Validation phase ─────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                logits  = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        log.info(
            f"  Epoch {epoch:>3}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"elapsed={elapsed:.1f}s"
        )

        # ── Early stopping check ──────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state_dict  = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(
                    f"  Early stopping triggered at epoch {epoch} "
                    f"(no val_loss improvement for {PATIENCE} epochs)"
                )
                break

    # Restore best weights
    model.load_state_dict(best_state_dict)
    log.info(f"  Best val_loss = {best_val_loss:.6f}  (weights restored)")

    # Save model
    model_path = os.path.join(model_dir, "attack_type_nn.pt")
    torch.save(model.state_dict(), model_path)
    log.info(f"  Model saved → {model_path}")

    # Save training history
    history_path = os.path.join(model_dir, "attack_type_nn_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"  Training history saved → {history_path}")

    return history


# ═══════════════════════════════════════════════════════════
# STEP 5 — Evaluate on held-out test set
# ═══════════════════════════════════════════════════════════
def evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
             label_map: dict, device):
    """
    Run inference on the held-out test set and print:
      - Per-class classification report (precision / recall / F1)
      - Confusion matrix (abridged to class indices)
      - Overall accuracy and macro-F1
      - ⚠️  Warning for any class with recall < RECALL_FLAG_THRESHOLD

    This evaluation is run once — the test set is never used during training
    or early stopping to avoid data leakage.
    """
    log.info("[STEP 5] Evaluating on held-out test set ...")
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32)
    all_preds = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            batch = X_t[start:start + batch_size].to(device)
            logits = model(batch)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    y_pred = np.concatenate(all_preds)

    class_names = [label_map[i] for i in sorted(label_map.keys())]
    present_classes = sorted(np.unique(y_test).tolist())

    acc     = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    log.info("")
    log.info("─" * 70)
    log.info("CLASSIFICATION REPORT (test set):")
    log.info("─" * 70)
    report = classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=[class_names[i] for i in present_classes],
        zero_division=0,
        digits=4,
    )
    for line in report.splitlines():
        log.info(line)

    log.info("")
    log.info(f"  Overall Accuracy : {acc:.4f}")
    log.info(f"  Macro F1-Score   : {macro_f1:.4f}")

    log.info("")
    log.info("CONFUSION MATRIX (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    header = "  " + "  ".join(f"{i:>6}" for i in present_classes)
    log.info(header)
    for row_cls, row in zip(present_classes, cm):
        log.info("  " + f"{row_cls:>2}" + "  " + "  ".join(f"{v:>6}" for v in row))

    # ── Flag poor-recall classes for Phase 3 attention ──
    log.info("")
    log.info("─" * 70)
    log.info("RECALL FLAGS (classes with recall < {:.0%}):".format(RECALL_FLAG_THRESHOLD))
    log.info("─" * 70)
    from sklearn.metrics import recall_score
    per_class_recall = recall_score(
        y_test, y_pred, labels=present_classes,
        average=None, zero_division=0
    )
    flagged = []
    for cls_idx, recall_val in zip(present_classes, per_class_recall):
        cls_name = class_names[cls_idx]
        if recall_val < RECALL_FLAG_THRESHOLD:
            flagged.append((cls_name, recall_val))
            log.warning(
                f"  ⚠️  {cls_name:<30s}  recall={recall_val:.4f}  "
                f"(<{RECALL_FLAG_THRESHOLD:.0%} — flag for Phase 3 review)"
            )

    if not flagged:
        log.info("  ✓  All classes have recall ≥ {:.0%}".format(RECALL_FLAG_THRESHOLD))
    else:
        log.warning(
            f"  {len(flagged)} class(es) flagged.  "
            "Review these before proceeding to Phase 3 (DQN training)."
        )

    return acc, macro_f1


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("train_attack_type_nn.py — Stage 2: Attack-Type Neural Network")
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU   : {torch.cuda.get_device_name(0)}")

    # Load
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     sample_weights, label_map) = load_data(DATA_DIR)

    n_classes = len(label_map)

    # Build loaders
    train_loader, val_loader = build_loaders(X_train, y_train, X_val, y_val, device)

    # Class weights for loss
    class_weights = compute_class_weights(y_train, n_classes, device)

    # Model
    log.info("[MODEL] Instantiating AttackTypeNN ...")
    model = AttackTypeNN(input_dim=INPUT_DIM, n_classes=n_classes, dropout_p=DROPOUT_P)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Architecture : {INPUT_DIM} → {HIDDEN_1} → {HIDDEN_2} → {n_classes}")
    log.info(f"  Total params : {total_params:,}")

    # Train
    history = train(model, train_loader, val_loader, class_weights, device, MODEL_DIR)

    # Evaluate
    acc, macro_f1 = evaluate(model, X_test, y_test, label_map, device)

    log.info("")
    log.info("=" * 70)
    log.info("✓  train_attack_type_nn.py complete.")
    log.info(f"   Test Accuracy  : {acc:.4f}")
    log.info(f"   Test Macro-F1  : {macro_f1:.4f}")
    log.info(f"   Epochs trained : {len(history['train_loss'])}")
    log.info("   Saved → models/attack_type_nn.pt")
    log.info("   Next step: python training/build_dqn_environment.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
