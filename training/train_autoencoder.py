"""
train_autoencoder.py — Autoencoder Training Pipeline (Stage 1: Anomaly Detection)
===================================================================================

PURPOSE:
  Trains a fully-connected autoencoder EXCLUSIVELY on benign (normal) network
  traffic so that the model learns to reconstruct normal flow patterns. When the
  trained autoencoder is later presented with attack traffic, it produces a high
  reconstruction error (MSE), which becomes the anomaly signal fed to the DQN
  agent in Stage 2.

INPUTS (data/processed/):
  - X_train_benign.npy   shape (~1,188,543 x 115), float32, MinMax-scaled to [0, 1]
  - X_val_benign.npy     shape (~254,688  x 115), float32
                         NOTE: val range may slightly exceed 1.0 on some features
                         (expected — scaler was fit on train data only; NOT a bug)

MODEL ARCHITECTURE:
  Encoder  : 115 → 64 → 32 → 16   (ReLU + Dropout(0.2))
  Bottleneck: 16 dimensions
  Decoder  : 16 → 32 → 64 → 115   (ReLU hidden; Sigmoid output)
  Output sigmoid is required because inputs are MinMax-scaled to [0, 1].

TRAINING:
  - Loss      : MSELoss
  - Optimizer : Adam (lr=1e-3, weight_decay=1e-5)
  - Batch size: 256  |  Early stopping on val_loss (patience=5)
  - Device    : CUDA if available, else CPU

OUTPUTS (models/):
  - autoencoder.pt          — best model state_dict
  - threshold.json          — 95th-percentile reconstruction error on val set
  - training_history.json   — train/val loss per epoch (for later plotting)

NEXT STEP:
  Run train_dqn.py to train the DQN agent using reconstruction errors as the
  anomaly signal.
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

# ─────────────────────────────────────────────
# Path configuration
# ─────────────────────────────────────────────
# Anchor paths to the project root regardless of where the script is invoked from:
#   python training/train_autoencoder.py    (from project root)
#   cd training && python train_autoencoder.py
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../training/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../IISC_RESEARCH_INTERNSHIP/

DATA_DIR  = os.path.join(_PROJECT_ROOT, "data",   "processed")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
LOG_DIR   = os.path.join(_PROJECT_ROOT, "data",   "processed")   # logs beside data files

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
INPUT_DIM   = 115       # number of network flow features
BATCH_SIZE  = 256
LR          = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS  = 50
PATIENCE    = 5         # early-stopping patience (val_loss epochs without improvement)
THRESHOLD_PERCENTILE = 95   # percentile of val reconstruction errors → anomaly threshold
DROPOUT_P   = 0.2

# ─────────────────────────────────────────────
# Logging setup  (console + file, UTF-8 safe on Windows)
# ─────────────────────────────────────────────
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
        logging.FileHandler(
            os.path.join(LOG_DIR, "train_autoencoder.log"),
            mode="w", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════
class Autoencoder(nn.Module):
    """
    Fully-connected autoencoder for unsupervised anomaly detection on network flows.

    Architecture
    ------------
    Encoder  : Linear(115→64) → ReLU → Dropout(0.2)
               Linear(64→32)  → ReLU → Dropout(0.2)
               Linear(32→16)  → ReLU          (bottleneck)

    Decoder  : Linear(16→32)  → ReLU
               Linear(32→64)  → ReLU
               Linear(64→115) → Sigmoid       (output in [0, 1])

    The Sigmoid output is mandatory because inputs are MinMax-scaled to [0, 1].
    MSE between sigmoid output and the original input is a well-defined,
    bounded loss function in this regime.

    Dropout is applied only in the encoder during training to regularise the
    bottleneck representation.  It is automatically disabled in eval() mode.

    Parameters
    ----------
    input_dim : int
        Number of input / output features (default 115).
    dropout_p : float
        Dropout probability applied after encoder hidden layers (default 0.2).
    """

    def __init__(self, input_dim: int = INPUT_DIM, dropout_p: float = DROPOUT_P):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # ── Decoder ──────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),    # clamps output to [0, 1] — matches MinMax-scaled inputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode x to a 16-dim bottleneck, then decode back to
        input_dim dimensions.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, input_dim)

        Returns
        -------
        torch.Tensor  shape (batch, input_dim)  — reconstructed input
        """
        return self.decoder(self.encoder(x))


# ═══════════════════════════════════════════════════════════
# STEP 1 — Load preprocessed numpy arrays
# ═══════════════════════════════════════════════════════════
def load_data(data_dir: str):
    """
    Load X_train_benign.npy and X_val_benign.npy from data/processed/.

    Both arrays are float32 and MinMax-scaled (scaler fit on train only).
    Val values may slightly exceed 1.0 on some features — this is expected
    and harmless for MSE training.

    Returns
    -------
    X_train : np.ndarray  shape (N_train, 115)
    X_val   : np.ndarray  shape (N_val,   115)
    """
    train_path = os.path.join(data_dir, "X_train_benign.npy")
    val_path   = os.path.join(data_dir, "X_val_benign.npy")

    log.info(f"  Loading X_train_benign  from: {train_path}")
    X_train = np.load(train_path)

    log.info(f"  Loading X_val_benign    from: {val_path}")
    X_val = np.load(val_path)

    log.info(f"  X_train shape: {X_train.shape}  dtype: {X_train.dtype}")
    log.info(f"  X_val   shape: {X_val.shape}  dtype: {X_val.dtype}")
    log.info(f"  X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    log.info(f"  X_val   range: [{X_val.min():.4f}, {X_val.max():.4f}]  "
             f"(may exceed 1.0 — expected)")

    return X_train, X_val


# ═══════════════════════════════════════════════════════════
# STEP 2 — Build DataLoaders
# ═══════════════════════════════════════════════════════════
def build_dataloaders(X_train: np.ndarray, X_val: np.ndarray, batch_size: int):
    """
    Convert numpy arrays to PyTorch TensorDatasets and wrap in DataLoaders.

    The autoencoder is a reconstruction task: input == target.
    Both dataset arguments to TensorDataset are the SAME tensor; there are
    no separate label arrays.

    Parameters
    ----------
    X_train, X_val : np.ndarray  float32
    batch_size : int

    Returns
    -------
    train_loader : DataLoader  (shuffle=True)
    val_loader   : DataLoader  (shuffle=False)
    """
    X_train_t = torch.from_numpy(X_train)   # already float32
    X_val_t   = torch.from_numpy(X_val)

    # TensorDataset(X, X) — input and target are the SAME tensor
    train_ds = TensorDataset(X_train_t, X_train_t)
    val_ds   = TensorDataset(X_val_t,   X_val_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,   # 0 avoids multiprocessing issues on Windows
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    log.info(f"  Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════
# STEP 3 — Training loop (with early stopping)
# ═══════════════════════════════════════════════════════════
def train(
    model: Autoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
):
    """
    Train the autoencoder using MSE loss and Adam optimiser.

    Training protocol
    -----------------
    - Standard 4-step loop per batch:
        forward → compute loss → loss.backward() → optimizer.step() → optimizer.zero_grad()
      (zero_grad is called AFTER step, not before, as specified in the project spec.)
    - Val loss is computed after each epoch in eval() mode with torch.no_grad().
    - Early stopping monitors val_loss; best weights are deep-copied and restored.

    Parameters
    ----------
    model       : Autoencoder
    train_loader: DataLoader  (benign train set)
    val_loader  : DataLoader  (benign val set)
    device      : torch.device
    num_epochs  : int   maximum training epochs
    patience    : int   early-stopping patience (epochs without val improvement)

    Returns
    -------
    model   : Autoencoder  with best-epoch weights restored
    history : dict  {"train_loss": [...], "val_loss": [...]}
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_val_loss  = float("inf")
    best_weights   = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": []}

    log.info(f"\n  Max epochs : {num_epochs}  |  Early-stop patience : {patience}")
    log.info(f"  Loss       : MSELoss")
    log.info(f"  Optimizer  : Adam  lr={LR}  weight_decay={WEIGHT_DECAY}")
    log.info(f"  Device     : {device}\n")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ── Training phase ────────────────────────────────
        model.train()
        running_train_loss = 0.0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)

            # 1. Forward pass
            x_recon = model(x_batch)

            # 2. Compute loss
            loss = criterion(x_recon, x_batch)

            # 3. Backpropagate
            loss.backward()

            # 4. Update weights
            optimizer.step()

            # 5. Clear gradients (AFTER step, per project spec)
            optimizer.zero_grad()

            running_train_loss += loss.item() * x_batch.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)

        # ── Validation phase ─────────────────────────────
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                x_recon = model(x_batch)
                loss    = criterion(x_recon, x_batch)
                running_val_loss += loss.item() * x_batch.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - epoch_start
        log.info(
            f"  Epoch [{epoch:>3}/{num_epochs}]  "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  "
            f"({elapsed:.1f}s)"
        )

        # ── Early stopping ────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            log.info(f"           ↳ New best val_loss={best_val_loss:.6f} — weights saved")
        else:
            epochs_no_improve += 1
            log.info(
                f"           ↳ No improvement ({epochs_no_improve}/{patience})"
            )
            if epochs_no_improve >= patience:
                log.info(
                    f"\n  ⏹  Early stopping triggered at epoch {epoch}. "
                    f"Best val_loss={best_val_loss:.6f}"
                )
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    log.info(f"\n  ✅  Best weights restored (val_loss={best_val_loss:.6f})")
    return model, history


# ═══════════════════════════════════════════════════════════
# STEP 4 — Compute per-sample reconstruction errors on val set
# ═══════════════════════════════════════════════════════════
def compute_reconstruction_errors(
    model: Autoencoder,
    X_val: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Run the trained autoencoder over X_val_benign in eval mode and return
    per-sample reconstruction errors (MSE per row, NOT averaged across samples).

    This produces the empirical distribution of reconstruction errors for
    BENIGN traffic, from which the anomaly detection threshold is derived.

    Parameters
    ----------
    model     : Autoencoder  (best weights already restored)
    X_val     : np.ndarray  shape (N_val, 115)
    device    : torch.device
    batch_size: int

    Returns
    -------
    errors : np.ndarray  shape (N_val,)  — MSE per sample
    """
    model.eval()
    X_val_t    = torch.from_numpy(X_val)
    val_ds     = TensorDataset(X_val_t, X_val_t)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_errors = []
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            x_recon = model(x_batch)
            # Per-sample MSE: mean over feature dimension, one scalar per row
            per_sample_mse = ((x_recon - x_batch) ** 2).mean(dim=1)
            all_errors.append(per_sample_mse.cpu().numpy())

    errors = np.concatenate(all_errors)   # shape (N_val,)
    log.info(f"  Reconstruction errors  shape : {errors.shape}")
    log.info(f"  Reconstruction errors  min   : {errors.min():.6f}")
    log.info(f"  Reconstruction errors  mean  : {errors.mean():.6f}")
    log.info(f"  Reconstruction errors  max   : {errors.max():.6f}")
    return errors


# ═══════════════════════════════════════════════════════════
# STEP 5 — Save model, threshold, and training history
# ═══════════════════════════════════════════════════════════
def save_outputs(
    model: Autoencoder,
    errors: np.ndarray,
    history: dict,
    model_dir: str,
):
    """
    Persist all training outputs required by downstream stages.

    Files saved
    -----------
    models/autoencoder.pt          — state_dict only (not the full model object)
    models/threshold.json          — {"threshold": <95th-percentile MSE value>}
    models/training_history.json   — {"train_loss": [...], "val_loss": [...]}

    The threshold is the 95th percentile of per-sample reconstruction errors
    computed on X_val_benign. At inference time, a sample whose reconstruction
    error exceeds this threshold is flagged as an anomaly (potential attack).

    Parameters
    ----------
    model     : Autoencoder
    errors    : np.ndarray  shape (N_val,) — per-sample MSE on val set
    history   : dict  {"train_loss": [...], "val_loss": [...]}
    model_dir : str   absolute path to models/
    """
    # 5a — Anomaly detection threshold (95th percentile)
    threshold = float(np.percentile(errors, THRESHOLD_PERCENTILE))
    threshold_path = os.path.join(model_dir, "threshold.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    log.info(
        f"  Threshold ({THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}  "
        f"→  {threshold_path}"
    )

    # 5b — Model weights (state_dict only — NOT the full model object)
    weights_path = os.path.join(model_dir, "autoencoder.pt")
    torch.save(model.state_dict(), weights_path)
    log.info(f"  Model state_dict saved             →  {weights_path}")

    # 5c — Training history (for later plotting / analysis)
    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    log.info(f"  Training history saved             →  {history_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("=" * 65)
    log.info("AUTOENCODER TRAINING PIPELINE")
    log.info("(Stage 1: Benign-only reconstruction → anomaly threshold)")
    log.info("=" * 65)
    log.info(f"  Data   : {DATA_DIR}")
    log.info(f"  Models : {MODEL_DIR}")

    # ── Device setup ──────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device : {device}")
    if device.type == "cuda":
        log.info(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── STEP 1: Load data ─────────────────────────────────
    log.info("\n[STEP 1] Loading preprocessed benign arrays...")
    X_train, X_val = load_data(DATA_DIR)

    # ── STEP 2: Build DataLoaders ─────────────────────────
    log.info("\n[STEP 2] Building DataLoaders...")
    train_loader, val_loader = build_dataloaders(X_train, X_val, BATCH_SIZE)

    # ── STEP 3: Initialise model ──────────────────────────
    log.info("\n[STEP 3] Initialising Autoencoder...")
    model = Autoencoder(input_dim=INPUT_DIM, dropout_p=DROPOUT_P).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Architecture : 115 → 64 → 32 → 16 → 32 → 64 → 115")
    log.info(f"  Trainable parameters : {total_params:,}")
    log.info(f"  Dropout (encoder)    : p={DROPOUT_P}")

    # ── STEP 4: Train ────────────────────────────────────
    log.info("\n[STEP 4] Training autoencoder...")
    model, history = train(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
    )

    # ── STEP 5: Compute reconstruction errors on val set ─
    log.info("\n[STEP 5] Computing per-sample reconstruction errors on val set...")
    errors = compute_reconstruction_errors(model, X_val, device, BATCH_SIZE)

    # ── STEP 6: Save all outputs ──────────────────────────
    log.info("\n[STEP 6] Saving model, threshold, and training history...")
    save_outputs(model, errors, history, MODEL_DIR)

    # ── Summary ───────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("✅  Autoencoder training complete!")
    log.info(f"    Epochs trained         : {len(history['train_loss'])}")
    log.info(f"    Final train_loss       : {history['train_loss'][-1]:.6f}")
    log.info(f"    Final val_loss         : {history['val_loss'][-1]:.6f}")
    log.info(f"    Anomaly threshold (p{THRESHOLD_PERCENTILE}) : "
             f"{float(np.percentile(errors, THRESHOLD_PERCENTILE)):.6f}")
    log.info(f"\n    → Now run: python training/train_dqn.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
