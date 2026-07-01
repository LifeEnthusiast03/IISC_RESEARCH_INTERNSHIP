"""
backend/routers/predict.py
───────────────────────────
POST /predict — run a network flow through the two-stage ML pipeline.

Pipeline
--------
1. Validate + normalise the incoming feature vector.
2. Feed through the autoencoder → get reconstruction_error.
3. Compare error to the learned threshold → set is_anomaly flag.
4. If anomaly: run through the DQN agent → get dqn_action + attack_type.
5. Persist an ``Incident`` row (always — benign rows are useful for baseline).
6. If anomaly: broadcast the incident as JSON over all open WebSocket connections.

Graceful degradation
--------------------
The ML artefacts (autoencoder.pt, dqn_agent.pt, scaler.pkl, threshold.json)
will not exist until training completes.  ``backend.inference`` exposes two
module-level booleans — ``autoencoder_ready`` and ``dqn_ready`` — that are
set during startup.  If either is False, this endpoint returns HTTP 503 with
an informative message rather than crashing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.init_db import get_db
from backend.db.database_models import Incident
from backend.websocket.connection import manager
from backend.schemas import FlowFeatures, IncidentOut, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])


def _check_models_ready() -> None:
    """
    Raise HTTP 503 if the ML artefacts are not yet loaded.

    This check is intentionally defensive: it never crashes if
    ``backend.inference`` fails to import (e.g. PyTorch not installed).
    """
    autoencoder_ready = False
    dqn_ready = False
    try:
        import backend.inference as _inf

        autoencoder_ready = getattr(_inf, "autoencoder_ready", False)
        dqn_ready = getattr(_inf, "dqn_ready", False)
    except Exception:  # noqa: BLE001
        pass

    if not autoencoder_ready or not dqn_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "ML models are not yet loaded.  "
                "Ensure that models/autoencoder.pt, models/dqn_agent.pt, "
                "models/scaler.pkl, and models/threshold.json exist and that "
                "the application was restarted after training completed."
            ),
        )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run inference on a network flow",
    description=(
        "Accepts a raw feature vector for one network flow, runs it through "
        "the autoencoder → DQN pipeline, persists the result, and (if anomalous) "
        "broadcasts an alert over WebSocket /ws/alerts."
    ),
    responses={
        503: {
            "description": "ML models not yet loaded — training must complete first."
        }
    },
)
async def predict(
    payload: FlowFeatures,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """
    Two-stage inference endpoint.

    **Stage 1 — Autoencoder (anomaly detection)**
    Normalises the feature vector with the fitted scaler, reconstructs it
    through the autoencoder, and computes the per-sample MSE.  If the error
    exceeds the threshold stored in threshold.json the flow is flagged as an
    anomaly.

    **Stage 2 — DQN (attack classification + remediation)**
    Anomalous flows are passed to the DQN agent which returns:
    • ``attack_type_predicted`` — e.g. "DoS", "PortScan", "Brute Force"
    • ``dqn_action``            — one of block_ip | revoke_credentials |
                                    isolate_server | kill_process | monitor

    **Persistence**
    An ``Incident`` row is inserted for **every** call (benign included) so
    the dashboard can display full traffic baselines alongside threat events.
    Anomalous incidents are also broadcast to all WebSocket subscribers.
    """
    # ── Guard: models must be loaded ─────────────────────────────────────────
    _check_models_ready()

    # ── Stage 1: Autoencoder inference ───────────────────────────────────────
    # TODO: replace with real inference.py call once models are trained
    # Example of how this will look when inference.py is complete:
    #
    #   import backend.inference as _inf
    #   reconstruction_error = _inf.compute_reconstruction_error(payload.features)
    #   threshold            = _inf.get_threshold()
    #
    reconstruction_error: float = 0.0   # placeholder
    threshold: float = 0.5              # placeholder

    is_anomaly: bool = reconstruction_error > threshold

    # ── Stage 2: DQN agent (only for anomalous flows) ────────────────────────
    # TODO: replace with real inference.py call once models are trained
    # Example of how this will look when inference.py is complete:
    #
    #   attack_type, action = _inf.run_dqn(payload.features, reconstruction_error)
    #
    attack_type_predicted: str | None = None
    dqn_action: str | None = None

    if is_anomaly:
        attack_type_predicted = "Unknown"   # placeholder
        dqn_action = "monitor"              # placeholder (safest default)

    # ── Persist Incident ──────────────────────────────────────────────────────
    incident = Incident(
        timestamp=datetime.now(timezone.utc),
        source_ip=payload.source_ip or "0.0.0.0",
        dest_ip=payload.dest_ip or "0.0.0.0",
        src_port=payload.src_port,
        dst_port=payload.dst_port,
        reconstruction_error=reconstruction_error,
        is_anomaly=is_anomaly,
        attack_type_predicted=attack_type_predicted,
        dqn_action=dqn_action,
        action_status="simulated",  # all actions are simulated until an executor is wired
        raw_features={"features": payload.features},
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)

    logger.info(
        "Incident %d persisted  is_anomaly=%s  action=%s",
        incident.id,
        is_anomaly,
        dqn_action,
    )

    # ── Broadcast anomaly alert over WebSocket ────────────────────────────────
    if is_anomaly and manager.active_count > 0:
        alert_payload = IncidentOut.model_validate(incident).model_dump(mode="json")
        await manager.broadcast(alert_payload)
        logger.debug("Alert broadcast to %d WebSocket client(s).", manager.active_count)

    # ── Return response ───────────────────────────────────────────────────────
    return PredictResponse(
        reconstruction_error=reconstruction_error,
        is_anomaly=is_anomaly,
        attack_type_predicted=attack_type_predicted,
        dqn_action=dqn_action,
        incident_id=incident.id,
    )
