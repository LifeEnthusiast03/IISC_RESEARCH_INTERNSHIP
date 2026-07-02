"""
backend/dqn_agent/dqn_suggestion.py
-------------------------------------
DQN-based remediation-action suggestion for confirmed anomalous network flows.

This module is Stage 3 of the IDS pipeline and is called ONLY after:
  Stage 1 � anomaly_detection.detect_anomaly()   confirms is_anomaly=True
  Stage 2 � attack_identifier.identify_attack()  produces attack_type + probs

Pipeline position
-----------------
  anomaly_detection  ?  attack_identifier  ?  suggest_action()  (this module)

DQN state vector (must match training in train_dqn.py)
-------------------------------------------------------
  [115 raw flow features | 1 AE reconstruction error | N attack-type softmax
   probs | 1 max-prob confidence]

  state_dim = 115 + 1 + N + 1
  N = number of attack classes (len(attack_type_label_map))

Action space (5 discrete actions � matches ACTION_NAMES in train_dqn.py)
--------------------------------------------------------------------------
  0 = Block IP
  1 = Revoke Credentials
  2 = Isolate Server
  3 = Kill Process
  4 = Monitor

Public API
----------
  suggest_action(request: DQNSuggestionRequest) -> DQNSuggestionResult
      Builds the DQN state vector, runs a forward pass, and returns the
      recommended remediation action with Q-value details.

Schemas (defined in backend.schemas and re-exported here)
---------------------------------------------------------
  DQNSuggestionRequest  � input schema
  DQNSuggestionResult   � output schema
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from backend.schemas import DQNSuggestionRequest, DQNSuggestionResult

# Shared model state loaded once at application startup by
# backend.models.init_models (called from the FastAPI lifespan hook).
import backend.models.init_models as _state

logger = logging.getLogger(__name__)

# -- Action registry (mirrors ACTION_NAMES in train_dqn.py) -------------------
ACTION_NAMES: dict[int, str] = {
    0: "Block IP",
    1: "Revoke Credentials",
    2: "Isolate Server",
    3: "Kill Process",
    4: "Monitor",
}
N_ACTIONS: int = len(ACTION_NAMES)


# =============================================================================
# Internal helpers
# =============================================================================

def _build_state_vector(
    raw_features: list[float],
    ae_reconstruction_error: float,
    attack_type_probs: list[float],
    attack_type_confidence: float,
) -> np.ndarray:
    """
    Assemble the DQN state vector in the exact order used during training.

    Layout: [115 flow features | 1 AE error | N softmax probs | 1 confidence]

    Parameters
    ----------
    raw_features : list[float]
        115 raw (unscaled) flow features � the DQN was trained on the raw
        feature values as part of its state, not the scaler-normalised ones.
    ae_reconstruction_error : float
        MSE reconstruction error from the autoencoder stage.
    attack_type_probs : list[float]
        Full softmax probability vector over all N attack classes from
        attack_type_nn.  Must have the same length as attack_type_label_map.
    attack_type_confidence : float
        Max probability from attack_type_probs (i.e. top-1 confidence).

    Returns
    -------
    np.ndarray
        Shape (1, 115 + 1 + N + 1) float32 array ready for the DQN.

    Raises
    ------
    ValueError
        If raw_features does not have exactly 115 elements.
    """
    if len(raw_features) != 115:
        raise ValueError(
            f"raw_features must have 115 elements, got {len(raw_features)}."
        )

    state = np.concatenate([
        np.array(raw_features,           dtype=np.float32),   # 115
        np.array([ae_reconstruction_error], dtype=np.float32), #   1
        np.array(attack_type_probs,       dtype=np.float32),   #   N
        np.array([attack_type_confidence], dtype=np.float32),  #   1
    ])
    return state.reshape(1, -1)


# =============================================================================
# Public API
# =============================================================================

def suggest_action(request: DQNSuggestionRequest) -> DQNSuggestionResult:
    """
    Run the DQN agent forward pass and recommend a remediation action.

    The function builds the full DQN state vector from the request fields,
    passes it through the loaded dqn_agent.pt, and returns the greedy
    action (argmax of Q-values) together with all per-action Q-values
    and a ranked alternative-actions list.

    Parameters
    ----------
    request : DQNSuggestionRequest
        Input model carrying attack_type, attack_confidence,
        ae_reconstruction_error, attack_type_probs, and raw_features.

    Returns
    -------
    DQNSuggestionResult
        Recommended action name, action index, confidence derived from
        softmax of Q-values, full Q-value table, and inference latency.

    Raises
    ------
    RuntimeError
        If the DQN agent is not loaded.
    ValueError
        If raw_features does not have exactly 115 elements.

    Examples
    --------
    >>> req = DQNSuggestionRequest(
    ...     attack_type="DoS",
    ...     attack_confidence=0.97,
    ...     ae_reconstruction_error=0.084,
    ...     attack_type_probs=[0.0] * 11,
    ...     raw_features=[0.0] * 115,
    ... )
    >>> result = suggest_action(req)
    >>> isinstance(result.recommended_action, str)
    True
    """
    # -- 1. Model readiness check ----------------------------------------------
    if _state.dqn_agent is None or not _state.dqn_agent_ready:
        raise RuntimeError(
            "DQN agent is not loaded.  "
            "Ensure load_models() ran successfully at startup."
        )

    t0 = time.perf_counter()

    # -- 2. Build state vector -------------------------------------------------
    state_arr = _build_state_vector(
        raw_features=request.raw_features,
        ae_reconstruction_error=request.ae_reconstruction_error,
        attack_type_probs=request.attack_type_probs,
        attack_type_confidence=request.attack_confidence,
    )
    state_tensor = torch.from_numpy(state_arr)  # shape (1, state_dim)

    # -- 3. DQN forward pass � raw Q-values -----------------------------------
    with torch.no_grad():
        q_values: torch.Tensor = _state.dqn_agent(state_tensor).squeeze(0)  # (5,)

    # -- 4. Greedy action selection --------------------------------------------
    action_idx: int = int(q_values.argmax().item())
    recommended_action: str = ACTION_NAMES[action_idx]

    # -- 5. Action probabilities via softmax of Q-values ----------------------
    # Softmax of Q-values gives a probability-like ranking useful for
    # confidence reporting, even though the DQN is trained on raw Q-values.
    action_probs: torch.Tensor = F.softmax(q_values, dim=0)
    action_confidence: float = round(float(action_probs[action_idx].item()), 6)

    # -- 6. Per-action Q-value table -------------------------------------------
    q_value_table: dict[str, float] = {
        ACTION_NAMES[i]: round(float(q_values[i].item()), 6)
        for i in range(N_ACTIONS)
    }

    # -- 7. Ranked alternative actions (excluding recommended) -----------------
    sorted_actions = sorted(
        [
            {
                "action": ACTION_NAMES[i],
                "action_index": i,
                "q_value": round(float(q_values[i].item()), 6),
                "probability": round(float(action_probs[i].item()), 6),
            }
            for i in range(N_ACTIONS)
            if i != action_idx
        ],
        key=lambda x: x["q_value"],
        reverse=True,
    )

    # -- 8. Timing -------------------------------------------------------------
    inference_latency_ms: float = round((time.perf_counter() - t0) * 1_000, 3)

    logger.info(
        "[dqn_suggestion] attack=%s  action=%s (idx=%d)  "
        "confidence=%.4f  latency_ms=%.2f",
        request.attack_type,
        recommended_action,
        action_idx,
        action_confidence,
        inference_latency_ms,
    )

    # -- 9. Build and return result --------------------------------------------
    return DQNSuggestionResult(
        recommended_action=recommended_action,
        action_index=action_idx,
        action_confidence=action_confidence,
        q_value_table=q_value_table,
        alternative_actions=sorted_actions,
        inference_latency_ms=inference_latency_ms,
    )
