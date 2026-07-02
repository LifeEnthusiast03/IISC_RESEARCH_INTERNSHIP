"""
backend/models/init_models.py
------------------------------
Loads all ML model artefacts into memory once at application startup.

Models are stored as module-level variables so every part of the backend
can import and use them directly without reloading from disk.

Artefact paths (relative to project root  models/):
    autoencoder.pt              <- PyTorch autoencoder (anomaly detection)
    hybrid_classifier.pkl       <- scikit-learn hybrid classifier (joblib/pickle)
    attack_type_nn.pt           <- PyTorch attack type neural network
    dqn_agent.pt                <- PyTorch DQN agent (remediation)

Supporting files also loaded:
    scaler.pkl                  <- scikit-learn StandardScaler for feature normalisation
    hybrid_label_encoder.pkl    <- LabelEncoder for hybrid classifier output classes
    attack_type_label_map.json  <- int -> attack label mapping for attack_type_nn
    threshold.json              <- reconstruction error threshold for anomaly decision
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Artefact paths ────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parents[2]   # project root
_MODELS_DIR = _ROOT / "models"

_AUTOENCODER_PATH            = _MODELS_DIR / "autoencoder.pt"
_HYBRID_CLASSIFIER_PATH      = _MODELS_DIR / "hybrid_classifier.pkl"
_ATTACK_TYPE_NN_PATH         = _MODELS_DIR / "attack_type_nn.pt"
_DQN_AGENT_PATH              = _MODELS_DIR / "dqn_agent.pt"

_SCALER_PATH                 = _MODELS_DIR / "scaler.pkl"
_HYBRID_LABEL_ENCODER_PATH   = _MODELS_DIR / "hybrid_label_encoder.pkl"
_ATTACK_TYPE_LABEL_MAP_PATH  = _MODELS_DIR / "attack_type_label_map.json"
_THRESHOLD_PATH              = _MODELS_DIR / "threshold.json"

# ── Module-level model holders ────────────────────────────────────────────────
autoencoder:           "torch.nn.Module | None" = None   # PyTorch
hybrid_classifier:     object                   = None   # sklearn
attack_type_nn:        "torch.nn.Module | None" = None   # PyTorch
dqn_agent:             "torch.nn.Module | None" = None   # PyTorch

# Supporting artefacts
scaler:                object       = None   # sklearn StandardScaler
hybrid_label_encoder:  object       = None   # sklearn LabelEncoder
attack_type_label_map: dict         = {}     # {int: str}
threshold:             float        = 0.0    # anomaly reconstruction threshold

# ── Readiness flags ───────────────────────────────────────────────────────────
autoencoder_ready:        bool = False
hybrid_classifier_ready:  bool = False
attack_type_nn_ready:     bool = False
dqn_agent_ready:          bool = False


# ── Loader ────────────────────────────────────────────────────────────────────
def load_models() -> None:
    """
    Load all ML model artefacts from disk into module-level variables.

    Called once during FastAPI lifespan startup (main.py).
    Each artefact is loaded independently — a failure on one does not
    prevent the others from loading.
    """
    global autoencoder, hybrid_classifier, attack_type_nn, dqn_agent
    global autoencoder_ready, hybrid_classifier_ready, attack_type_nn_ready, dqn_agent_ready
    global scaler, hybrid_label_encoder, attack_type_label_map, threshold

    import joblib  # noqa: PLC0415
    import torch   # noqa: PLC0415
    
    # Imports for model architectures
    import sys
    # Ensure project root is in sys.path
    if str(_ROOT) not in sys.path:
        sys.path.append(str(_ROOT))
        
    try:
        from training.train_autoencoder import Autoencoder
        from training.train_attack_type_nn import AttackTypeNN
        from training.train_dqn import DQNNetwork
    except ImportError as exc:
        logger.error("[ML] Could not import model architectures from training module: %s", exc)
        return

    # ── Supporting artefacts must be loaded FIRST for dimensions ──────────────

    # Attack type label map (determines n_classes)
    try:
        with open(_ATTACK_TYPE_LABEL_MAP_PATH) as f:
            attack_type_label_map = json.load(f)
        logger.info("[ML] Attack type label map loaded (%d classes)", len(attack_type_label_map))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ML] Attack type label map not loaded: %s", exc)
        
    n_classes = len(attack_type_label_map) if attack_type_label_map else 11 # Safe default

    # Scaler
    try:
        scaler = joblib.load(_SCALER_PATH)
        logger.info("[ML] Scaler loaded from %s", _SCALER_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ML] Scaler not loaded: %s", exc)

    # Hybrid label encoder
    try:
        hybrid_label_encoder = joblib.load(_HYBRID_LABEL_ENCODER_PATH)
        logger.info("[ML] Hybrid label encoder loaded from %s", _HYBRID_LABEL_ENCODER_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ML] Hybrid label encoder not loaded: %s", exc)

    # Anomaly threshold
    try:
        with open(_THRESHOLD_PATH) as f:
            threshold = json.load(f)["threshold"]
        logger.info("[ML] Anomaly threshold loaded: %.6f", threshold)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ML] Threshold not loaded: %s", exc)


    # ── 1. Autoencoder (PyTorch) ──────────────────────────────────────────────
    try:
        state_dict = torch.load(_AUTOENCODER_PATH, map_location="cpu", weights_only=True)
        autoencoder = Autoencoder(input_dim=115, dropout_p=0.2)
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()
        autoencoder_ready = True
        logger.info("[ML] Autoencoder loaded from %s", _AUTOENCODER_PATH)
    except FileNotFoundError:
        logger.warning("[ML] Autoencoder NOT found at %s", _AUTOENCODER_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("[ML] Failed to load autoencoder: %s", exc)

    # ── 2. Hybrid Classifier (sklearn / joblib pickle) ────────────────────────
    try:
        hybrid_classifier = joblib.load(_HYBRID_CLASSIFIER_PATH)
        hybrid_classifier_ready = True
        logger.info("[ML] Hybrid classifier loaded from %s", _HYBRID_CLASSIFIER_PATH)
    except FileNotFoundError:
        logger.warning("[ML] Hybrid classifier NOT found at %s", _HYBRID_CLASSIFIER_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("[ML] Failed to load hybrid classifier: %s", exc)

    # ── 3. Attack Type NN (PyTorch) ───────────────────────────────────────────
    try:
        state_dict = torch.load(_ATTACK_TYPE_NN_PATH, map_location="cpu", weights_only=True)
        attack_type_nn = AttackTypeNN(input_dim=115, n_classes=n_classes, dropout_p=0.2)
        attack_type_nn.load_state_dict(state_dict)
        attack_type_nn.eval()
        attack_type_nn_ready = True
        logger.info("[ML] Attack type NN loaded from %s", _ATTACK_TYPE_NN_PATH)
    except FileNotFoundError:
        logger.warning("[ML] Attack type NN NOT found at %s", _ATTACK_TYPE_NN_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("[ML] Failed to load attack type NN: %s", exc)

    # ── 4. DQN Agent (PyTorch) ────────────────────────────────────────────────
    try:
        state_dim = 115 + 1 + n_classes + 1
        state_dict = torch.load(_DQN_AGENT_PATH, map_location="cpu", weights_only=True)
        dqn_agent = DQNNetwork(state_dim=state_dim, n_actions=5)
        dqn_agent.load_state_dict(state_dict)
        dqn_agent.eval()
        dqn_agent_ready = True
        logger.info("[ML] DQN agent loaded from %s", _DQN_AGENT_PATH)
    except FileNotFoundError:
        logger.warning("[ML] DQN agent NOT found at %s", _DQN_AGENT_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("[ML] Failed to load DQN agent: %s", exc)


def all_models_ready() -> bool:
    """Return True only when all 4 core models are loaded."""
    return autoencoder_ready and hybrid_classifier_ready and attack_type_nn_ready and dqn_agent_ready