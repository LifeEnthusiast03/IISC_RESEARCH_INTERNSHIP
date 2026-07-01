"""
backend/schemas.py
──────────────────
Pydantic v2 request / response schemas for the IDS API.

All schemas use ``model_config = ConfigDict(from_attributes=True)`` where they
mirror an ORM model, allowing ``model.from_orm(obj)`` / ``model_validate(obj)``
to work seamlessly with SQLAlchemy rows.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ── Request schemas ───────────────────────────────────────────────────────────

class FlowFeatures(BaseModel):
    """
    Request body for POST /predict.

    ``features`` is the raw per-flow feature vector that will be fed into
    the autoencoder.  The CICIDS2017 dataset produces 78 features; UNSW-NB15
    produces 49 (after preprocessing).  Both are accepted — the inference
    layer is responsible for validating dimensionality against the loaded
    scaler / model.

    The optional network-metadata fields are used to populate the Incident
    row and are not fed into the ML pipeline.
    """

    features: list[float] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of numeric flow features (e.g. packet lengths, "
            "inter-arrival times, flag counts).  Must match the dimensionality "
            "expected by the loaded autoencoder / scaler."
        ),
    )

    # Optional metadata — enriches the stored Incident record
    source_ip: str | None = Field(
        default=None,
        description="Source IP address of the flow (IPv4 or IPv6).",
        examples=["192.168.1.10"],
    )
    dest_ip: str | None = Field(
        default=None,
        description="Destination IP address of the flow.",
        examples=["10.0.0.1"],
    )
    src_port: int | None = Field(
        default=None,
        ge=0,
        le=65535,
        description="Source TCP/UDP port.",
    )
    dst_port: int | None = Field(
        default=None,
        ge=0,
        le=65535,
        description="Destination TCP/UDP port.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [0.0] * 78,
                "source_ip": "192.168.1.10",
                "dest_ip": "10.0.0.1",
                "src_port": 54321,
                "dst_port": 80,
            }
        }
    )


# ── Response schemas ──────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """
    Response body for POST /predict.

    Includes the autoencoder's reconstruction error, the anomaly decision,
    the predicted attack category (if anomalous), the DQN remediation action,
    and the database ID of the persisted Incident row.
    """

    reconstruction_error: float = Field(
        ...,
        description="Mean-squared reconstruction error from the autoencoder.",
    )
    is_anomaly: bool = Field(
        ...,
        description="True when reconstruction_error exceeds the learned threshold.",
    )
    attack_type_predicted: str | None = Field(
        default=None,
        description=(
            "Attack category assigned by the secondary classifier / DQN "
            "(e.g. 'DoS', 'PortScan', 'Brute Force').  None for benign flows."
        ),
    )
    dqn_action: str | None = Field(
        default=None,
        description=(
            "Remediation action chosen by the DQN agent. "
            "One of: block_ip | revoke_credentials | isolate_server | "
            "kill_process | monitor.  None for benign flows."
        ),
    )
    incident_id: int | None = Field(
        default=None,
        description=(
            "Primary key of the persisted Incident row.  "
            "None when the flow is benign and not stored."
        ),
    )

    model_config = ConfigDict(from_attributes=True)


class IncidentOut(BaseModel):
    """
    Full read schema mirroring the ``Incident`` ORM model.

    Used by GET /incidents and GET /incidents/{incident_id}.
    ``from_attributes=True`` enables direct construction from SQLAlchemy rows.
    """

    id: int
    timestamp: datetime
    source_ip: str
    dest_ip: str
    src_port: int | None
    dst_port: int | None
    reconstruction_error: float
    is_anomaly: bool
    attack_type_predicted: str | None
    dqn_action: str | None
    action_status: str
    raw_features: dict | list | None  # JSON column — can be dict or list
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IncidentListResponse(BaseModel):
    """
    Paginated wrapper returned by GET /incidents.
    """

    items: list[IncidentOut] = Field(..., description="Incidents on the current page.")
    total: int = Field(..., description="Total number of incidents matching the query.")
    page: int = Field(..., description="Current page number (1-indexed).")
    page_size: int = Field(..., description="Number of items per page.")

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    """
    Response body for GET /health.

    Provides a quick operational snapshot of the API, ML models, and the
    database connection without exposing sensitive configuration details.
    """

    status: str = Field(
        ...,
        description="Overall service status: 'ok' or 'degraded'.",
        examples=["ok"],
    )
    autoencoder_loaded: bool = Field(
        ...,
        description="True when the autoencoder .pt artefact was loaded successfully.",
    )
    dqn_loaded: bool = Field(
        ...,
        description="True when the DQN agent .pt artefact was loaded successfully.",
    )
    db_connected: bool = Field(
        ...,
        description="True when a round-trip SELECT 1 to Postgres succeeded.",
    )
    uptime_seconds: float = Field(
        ...,
        description="Number of seconds since the API process started.",
    )

    model_config = ConfigDict(from_attributes=True)
