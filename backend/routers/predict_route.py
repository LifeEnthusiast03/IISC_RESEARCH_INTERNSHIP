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
from backend.anomaly_classifier.anomaly_detection import detect_anomaly
from backend.attacktype_classifier.attack_identifier import identify_attack
from backend.dqn_agent.dqn_suggestion import suggest_action
from backend.schemas import AnomalyDetectionResult, AttackTypeResult, DQNSuggestionRequest, DQNSuggestionResult

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])


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
    Three-stage IDS inference pipeline:
      Stage 1 -- Autoencoder anomaly gate
      Stage 2 -- Attack-type classifier    (only if anomaly detected)
      Stage 3 -- DQN remediation agent     (only if anomaly detected)
    Always persists an Incident row and broadcasts via WebSocket if anomalous.
    """
    try:
        # -- Stage 1: Anomaly detection ----------------------------------------
        # detect_anomaly() expects list[float] -- extract from the FlowFeatures model
        anomaly_result: AnomalyDetectionResult = detect_anomaly(payload.features)

        is_anomaly        = anomaly_result.is_anomaly
        attack_type_label: str | None  = None
        dqn_action_label:  str | None  = None

        if is_anomaly:
            # -- Stage 2: Attack-type classification ---------------------------
            # identify_attack() expects dict[str, float] (named features).
            # We generate synthetic feature names f0..f114 preserving order.
            print(anomaly_result.attack_type)
            named_features: dict[str, float] = {
                f"f{i}": v for i, v in enumerate(payload.features)
            }
            attack_type_result: AttackTypeResult = identify_attack(named_features)
            attack_type_label = attack_type_result.attack_type
            print(attack_type_result.attack_type)
            # -- Stage 3: DQN remediation suggestion ---------------------------
            # attack_type_probs is list[float]; extract values from the dict in order
            dqn_req = DQNSuggestionRequest(
                attack_type=attack_type_result.attack_type,
                attack_confidence=attack_type_result.attack_type_confidence,
                ae_reconstruction_error=anomaly_result.reconstruction_error,
                attack_type_probs=list(attack_type_result.all_attack_probabilities.values()),
                raw_features=payload.features,   # list[float], not payload.dict()
            )
            dqn_result: DQNSuggestionResult = suggest_action(dqn_req)
            dqn_action_label = dqn_result.recommended_action

        # -- Persist Incident row (always -- benign baseline is useful) ---------
        incident = Incident(
            timestamp=datetime.now(timezone.utc),
            source_ip=payload.source_ip or "0.0.0.0",
            dest_ip=payload.dest_ip or "0.0.0.0",
            src_port=payload.src_port,
            dst_port=payload.dst_port,
            reconstruction_error=anomaly_result.reconstruction_error,
            is_anomaly=is_anomaly,
            attack_type_predicted=attack_type_label,
            dqn_action=dqn_action_label,
            action_status="simulated",
            raw_features=payload.features,
        )
        db.add(incident)
        db.commit()
        db.refresh(incident)

        # -- WebSocket broadcast (anomalies only) -------------------------------
        if is_anomaly:
            await manager.broadcast({
                "event":          "anomaly",
                "incident_id":    incident.id,
                "source_ip":      incident.source_ip,
                "dest_ip":        incident.dest_ip,
                "attack_type":    attack_type_label,
                "dqn_action":     dqn_action_label,
                "recon_error":    round(anomaly_result.reconstruction_error, 6),
                "timestamp":      incident.timestamp.isoformat(),
            })

        # -- Return response ----------------------------------------------------
        return PredictResponse(
            reconstruction_error=anomaly_result.reconstruction_error,
            is_anomaly=is_anomaly,
            attack_type_predicted=attack_type_label,
            dqn_action=dqn_action_label,
            incident_id=incident.id,
        )

    except Exception as exc:
        logger.exception("Prediction pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

