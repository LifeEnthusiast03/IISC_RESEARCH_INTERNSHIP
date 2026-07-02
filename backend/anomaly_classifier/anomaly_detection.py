"""
backend/anomaly_classifier/anomaly_detection.py
-------------------------------------------------
Combined anomaly-detection pipeline using:
  1. PyTorch Autoencoder  ï¿½ computes mean-squared reconstruction error (MSE).
  2. Hybrid Classifier     ï¿½ sklearn ensemble that maps the 115-feature vector
                             directly to an attack-type label.

Decision logic (two-gate approach)
-----------------------------------
A flow is marked as an anomaly when EITHER condition holds:
  ï¿½ autoencoder MSE > learned threshold  (unsupervised gate)
  ï¿½ hybrid_classifier predicts a non-benign class (supervised gate)

When both agree the flow is anomalous, the hybrid classifier label is used
as the canonical attack type.  When only the autoencoder fires we fall back to
the label "Unknown Attack".

Public API
----------
  detect_anomaly(features: list[float]) -> AnomalyDetectionResult
      Main entry-point.  Accepts a 115-element feature vector and returns
      a populated AnomalyDetectionResult Pydantic model.

Response schema
---------------
  AnomalyDetectionResult  (see below)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from backend.schemas import AnomalyDetectionResult

# Import shared model state loaded at application startup.
# backend.models.init_models loads every artefact once during the FastAPI
# lifespan startup hook and exposes them as module-level variables.
import backend.models.init_models as _state

logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
EXPECTED_FEATURE_COUNT: int = 115
"""Number of features expected in the input vector (post-preprocessing)."""

_BENIGN_LABELS: frozenset[str] = frozenset(
    {"benign", "normal", "BENIGN", "NORMAL", "0", "Benign"}
)
"""All label strings that the hybrid classifier may return for benign traffic."""


# =============================================================================
# Internal helpers
# =============================================================================

def _scale_features(raw: list[float]) -> np.ndarray:
    """
    Normalise the raw feature vector using the fitted StandardScaler.

    Parameters
    ----------
    raw : list[float]
        115 raw numeric features.

    Returns
    -------
    np.ndarray
        Shape (1, 115) scaled array ready for model inference.

    Raises
    ------
    RuntimeError
        If the scaler has not been loaded (_state.scaler is None).
    """
    if _state.scaler is None:
        raise RuntimeError(
            "Feature scaler is not loaded.  "
            "Ensure load_models() ran successfully at startup."
        )
    arr = np.array(raw, dtype=np.float32).reshape(1, -1)
    return _state.scaler.transform(arr).astype(np.float32)


def _run_autoencoder(scaled: np.ndarray) -> float:
    """
    Forward-pass through the autoencoder and compute MSE reconstruction error.

    Parameters
    ----------
    scaled : np.ndarray
        Shape (1, 115) scaled feature array.

    Returns
    -------
    float
        Mean-squared reconstruction error.

    Raises
    ------
    RuntimeError
        If the autoencoder has not been loaded.
    """
    if _state.autoencoder is None or not _state.autoencoder_ready:
        raise RuntimeError(
            "Autoencoder is not loaded.  "
            "Ensure load_models() ran successfully at startup."
        )

    tensor_input = torch.from_numpy(scaled)  # shape (1, 115)

    with torch.no_grad():
        reconstructed: torch.Tensor = _state.autoencoder(tensor_input)

    mse: float = float(
        torch.mean((tensor_input - reconstructed) ** 2).item()
    )
    return mse


def _run_hybrid_classifier(
    scaled: np.ndarray,
) -> tuple[str, float | None]:
    """
    Classify the scaled feature vector with the hybrid sklearn classifier.

    Parameters
    ----------
    scaled : np.ndarray
        Shape (1, 115) scaled feature array.

    Returns
    -------
    label : str
        Predicted class label (raw string from the encoder or model).
    confidence : float | None
        Max class probability if predict_proba is available, else None.

    Raises
    ------
    RuntimeError
        If the hybrid classifier has not been loaded.
    """
    if _state.hybrid_classifier is None or not _state.hybrid_classifier_ready:
        raise RuntimeError(
            "Hybrid classifier is not loaded.  "
            "Ensure load_models() ran successfully at startup."
        )

    raw_label_arr: Any = _state.hybrid_classifier.predict(scaled)
    raw_label: str = str(raw_label_arr[0])

    # Attempt to decode integer label through the label encoder
    if _state.hybrid_label_encoder is not None:
        try:
            pred_idx = int(float(raw_label))
            if isinstance(_state.hybrid_label_encoder, list):
                raw_label = str(_state.hybrid_label_encoder[pred_idx])
            else:
                decoded = _state.hybrid_label_encoder.inverse_transform([pred_idx])
                raw_label = str(decoded[0])
        except (ValueError, IndexError, AttributeError):
            pass  # raw_label is already a string class name or decoder failed

    # Confidence -- only available for classifiers that expose predict_proba
    confidence: float | None = None
    if hasattr(_state.hybrid_classifier, "predict_proba"):
        try:
            proba: np.ndarray = _state.hybrid_classifier.predict_proba(scaled)
            confidence = float(np.max(proba))
        except Exception as exc:  # noqa: BLE001
            logger.debug("predict_proba failed: %s", exc)

    return raw_label, confidence


# =============================================================================
# Public API
# =============================================================================

def detect_anomaly(features: list[float]) -> AnomalyDetectionResult:
    """
    Run a 115-feature network-flow vector through the cascade
    autoencoder ? hybrid-classifier anomaly-detection pipeline.

    Decision logic (cascade / short-circuit)
    ----------------------------------------
    Stage 1 ï¿½ Autoencoder (unsupervised gate):
        The scaled feature vector is passed through the autoencoder and the
        mean-squared reconstruction error (MSE) is computed.

        * If MSE > anomaly_threshold  ?  the flow is IMMEDIATELY classified
          as an anomaly.  The hybrid classifier is NOT called.
          attack_type is set to "Unknown Attack" because no supervised label
          is available at this stage.

    Stage 2 ï¿½ Hybrid Classifier (supervised gate, only if Stage 1 passed):
        Reached only when the autoencoder decides the flow is benign
        (MSE <= threshold).  The hybrid classifier then makes the final call:

        * If the predicted label is NOT in _BENIGN_LABELS  ?  anomaly, and
          attack_type is set to the classifier label.
        * Otherwise  ?  benign flow, attack_type remains None.

    Parameters
    ----------
    features : list[float]
        Ordered list of exactly 115 numeric flow features produced by the
        CIC-IDS or equivalent feature-extraction pipeline.

    Returns
    -------
    AnomalyDetectionResult
        Fully populated Pydantic response model.

    Raises
    ------
    ValueError
        When features does not have exactly 115 elements.
    RuntimeError
        When the autoencoder or hybrid classifier is not loaded.

    Examples
    --------
    >>> result = detect_anomaly([0.0] * 115)
    >>> result.is_anomaly
    False
    """
    # 1. Input validation
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"got {len(features)}."
        )

    t0 = time.perf_counter()

    # 2. Convert to numpy array.
    # NOTE: features are already MinMax-scaled by the data pipeline (scaler.pkl
    # was applied during preprocessing).  _scale_features() is kept as a helper
    # for callers that pass raw unscaled features, but is NOT called here.
    scaled = np.array(features, dtype=np.float32).reshape(1, -1)

    # -------------------------------------------------------------------------
    # Stage 1: Autoencoder gate
    # -------------------------------------------------------------------------
    reconstruction_error: float = _run_autoencoder(scaled)
    anomaly_threshold: float = float(_state.threshold)
    autoencoder_triggered: bool = reconstruction_error > anomaly_threshold

    logger.debug(
        "[AE] recon_error=%.6f  threshold=%.6f  triggered=%s",
        reconstruction_error,
        anomaly_threshold,
        autoencoder_triggered,
    )

    if autoencoder_triggered:
        # Short-circuit: autoencoder alone is sufficient to flag the flow.
        # The hybrid classifier is skipped to save compute.
        inference_latency_ms: float = (time.perf_counter() - t0) * 1_000

        logger.info(
            "[anomaly_detection] STAGE-1 HIT  recon_error=%.6f  "
            "threshold=%.6f  latency_ms=%.2f",
            reconstruction_error,
            anomaly_threshold,
            inference_latency_ms,
        )

        return AnomalyDetectionResult(
            is_anomaly=True,
            reconstruction_error=reconstruction_error,
            anomaly_threshold=anomaly_threshold,
            autoencoder_triggered=True,
            classifier_triggered=False,   # classifier was not run
            attack_type="Unknown Attack", # no supervised label available
            classifier_confidence=None,
            inference_latency_ms=inference_latency_ms,
        )

    # -------------------------------------------------------------------------
    # Stage 2: Hybrid classifier gate (autoencoder said benign)
    # -------------------------------------------------------------------------
    raw_label, confidence = _run_hybrid_classifier(scaled)
    classifier_triggered: bool = raw_label not in _BENIGN_LABELS

    logger.debug(
        "[HC] label=%s  confidence=%s  triggered=%s",
        raw_label,
        confidence,
        classifier_triggered,
    )

    is_anomaly: bool = classifier_triggered
    attack_type: str | None = raw_label if classifier_triggered else None

    inference_latency_ms = (time.perf_counter() - t0) * 1_000

    logger.info(
        "[anomaly_detection] STAGE-2  is_anomaly=%s  attack_type=%s  "
        "recon_error=%.6f  latency_ms=%.2f",
        is_anomaly,
        attack_type,
        reconstruction_error,
        inference_latency_ms,
    )

    return AnomalyDetectionResult(
        is_anomaly=is_anomaly,
        reconstruction_error=reconstruction_error,
        anomaly_threshold=anomaly_threshold,
        autoencoder_triggered=False,
        classifier_triggered=classifier_triggered,
        attack_type=attack_type,
        classifier_confidence=confidence,
        inference_latency_ms=inference_latency_ms,
    )
