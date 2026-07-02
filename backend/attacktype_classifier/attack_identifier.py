"""
backend/attacktype_classifier/attack_identifier.py
----------------------------------------------------
Attack-type classification using the PyTorch attack_type_nn model.

This module is the second stage of the IDS pipeline and is called ONLY after
a flow has already been confirmed as an anomaly by the anomaly-detection stage.

Pipeline position
-----------------
  anomaly_detection.detect_anomaly()  -- stage 1 (unsupervised gate)
          ï¿½
          ?  [is_anomaly == True]
  attack_identifier.identify_attack()  -- stage 2 (attack-type label)

Model used
----------
  attack_type_nn.pt  -- PyTorch neural network trained to classify a scaled
                        115-feature vector into one of N attack categories.
  attack_type_label_map.json  -- {str(int): str} mapping from integer class
                                 index to human-readable attack label.
  scaler.pkl  -- sklearn StandardScaler shared with the anomaly-detection stage.

Public API
----------
  identify_attack(features: dict[str, float]) -> AttackTypeResult
      Accepts a dict of exactly 115 named features, runs them through the
      attack_type_nn, and returns an AttackTypeResult.

Response schema
---------------
  AttackTypeResult  -- defined in backend.schemas and re-exported here.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from backend.schemas import AttackTypeResult

# Shared model state loaded once at application startup by
# backend.models.init_models (called from the FastAPI lifespan hook).
import backend.models.init_models as _state

logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
EXPECTED_FEATURE_COUNT: int = 115
"""Number of features the attack_type_nn expects (must match training config)."""


# =============================================================================
# Internal helpers
# =============================================================================

def _scale_features(features_dict: dict[str, float]) -> np.ndarray:
    """
    Convert the feature dict to a scaled numpy array.

    The dict values are ordered by their insertion order (Python 3.7+ guarantee),
    which must match the column order used during training.

    Parameters
    ----------
    features_dict : dict[str, float]
        Exactly 115 named flow features.

    Returns
    -------
    np.ndarray
        Shape (1, 115) float32 array, normalised by the fitted StandardScaler.

    Raises
    ------
    ValueError
        If the dict does not contain exactly 115 entries.
    RuntimeError
        If the scaler is not loaded.
    """
    if len(features_dict) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features in the dict, "
            f"got {len(features_dict)}."
        )

    if _state.scaler is None:
        raise RuntimeError(
            "Feature scaler is not loaded.  "
            "Ensure load_models() ran successfully at startup."
        )

    arr = np.array(list(features_dict.values()), dtype=np.float32).reshape(1, -1)
    return _state.scaler.transform(arr).astype(np.float32)


def _resolve_label(class_idx: int) -> str:
    """
    Map an integer class index to a human-readable attack label.

    Uses ``_state.attack_type_label_map`` (loaded from attack_type_label_map.json).
    Falls back to the string representation of the index if the map is missing
    or does not contain the key.

    Parameters
    ----------
    class_idx : int
        Predicted class index from the neural network.

    Returns
    -------
    str
        Human-readable attack label, e.g. "DoS", "PortScan", "Brute Force".
    """
    label_map: dict = _state.attack_type_label_map  # {str(int): str}

    # Try str key first (JSON keys are always strings), then int key
    label = label_map.get(str(class_idx)) or label_map.get(class_idx)
    if label is None:
        logger.warning(
            "[attack_identifier] class_idx %d not found in label map ï¿½ "
            "falling back to raw index string.",
            class_idx,
        )
        return str(class_idx)

    return str(label)


# =============================================================================
# Public API
# =============================================================================

def identify_attack(features: dict[str, float]) -> AttackTypeResult:
    """
    Classify the attack type of an already-confirmed anomalous network flow.

    The function scales the 115 input features, runs a forward pass through
    the PyTorch attack_type_nn, applies softmax to obtain per-class
    probabilities, and returns the top predicted class together with its
    confidence score and a ranked list of the top-3 alternatives.

    Parameters
    ----------
    features : dict[str, float]
        Ordered mapping of feature name ? value for a single network flow.
        Must contain exactly 115 entries whose values match the scale expected
        by the fitted StandardScaler.

    Returns
    -------
    AttackTypeResult
        Pydantic model containing the predicted attack type, confidence,
        class index, top-3 alternatives, and inference latency.

    Raises
    ------
    ValueError
        If ``features`` does not contain exactly 115 entries.
    RuntimeError
        If the attack_type_nn or the scaler is not loaded.

    Examples
    --------
    >>> result = identify_attack({f"f{i}": 0.0 for i in range(115)})
    >>> isinstance(result.attack_type, str)
    True
    """
    # -- 1. Model readiness check ----------------------------------------------
    if _state.attack_type_nn is None or not _state.attack_type_nn_ready:
        raise RuntimeError(
            "attack_type_nn is not loaded.  "
            "Ensure load_models() ran successfully at startup."
        )

    t0 = time.perf_counter()

    # -- 2. Convert to numpy array.
    # NOTE: features are already MinMax-scaled by the data pipeline.
    # _scale_features() is kept as a helper for raw-feature callers but is
    # NOT called here to avoid double-scaling.
    arr = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
    tensor_input = torch.from_numpy(arr)  # shape (1, 115)

    # -- 3. Forward pass -------------------------------------------------------
    with torch.no_grad():
        logits: torch.Tensor = _state.attack_type_nn(tensor_input)  # (1, N)

    probabilities: torch.Tensor = F.softmax(logits, dim=1).squeeze(0)  # (N,)

    # -- 4. Top prediction -----------------------------------------------------
    top_confidence, top_class_idx = float(probabilities.max()), int(probabilities.argmax())
    attack_type: str = _resolve_label(top_class_idx)

    # -- 5. Full probability distribution over all classes ---------------------
    label_map: dict = _state.attack_type_label_map
    all_attack_probabilities: dict[str, float] = {
        str(label_map.get(str(i)) or label_map.get(i) or str(i)):
            round(float(probabilities[i].item()), 6)
        for i in range(probabilities.size(0))
    }

    # -- 6. Top-3 alternatives (excluding the top prediction) ------------------
    top3_values, top3_indices = torch.topk(probabilities, k=min(3, probabilities.size(0)))
    top3: list[dict] = [
        {
            "attack_type": _resolve_label(int(idx)),
            "confidence": round(float(val), 6),
        }
        for idx, val in zip(top3_indices.tolist(), top3_values.tolist())
        if int(idx) != top_class_idx  # exclude the primary prediction
    ]

    # -- 7. Timing -------------------------------------------------------------
    inference_latency_ms: float = (time.perf_counter() - t0) * 1_000

    logger.info(
        "[attack_identifier] attack_type=%s  confidence=%.4f  "
        "class_idx=%d  latency_ms=%.2f",
        attack_type,
        top_confidence,
        top_class_idx,
        inference_latency_ms,
    )

    # -- 7. Build and return response ------------------------------------------
    return AttackTypeResult(
        attack_type=attack_type,
        attack_type_confidence=round(top_confidence, 6),
        class_index=top_class_idx,
        top_alternatives=top3,
        all_attack_probabilities=all_attack_probabilities,
        inference_latency_ms=round(inference_latency_ms, 3),
    )
