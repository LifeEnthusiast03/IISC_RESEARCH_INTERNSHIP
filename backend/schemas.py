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

# -- Anomaly detection result schema -------------------------------------------

class AnomalyDetectionResult(BaseModel):
    """
    Return schema for anomaly_classifier.anomaly_detection.detect_anomaly().

    Fields
    ------
    is_anomaly : bool
        True when the combined autoencoder + hybrid-classifier pipeline
        determines that the flow is not benign.

    reconstruction_error : float
        Mean-squared reconstruction error produced by the autoencoder forward
        pass on the (scaled) input vector.  Higher values indicate that the
        sample deviates more from the distribution of normal traffic seen
        during training.

    anomaly_threshold : float
        The decision boundary loaded from models/threshold.json.  Any
        reconstruction error above this value triggers the autoencoder gate.

    autoencoder_triggered : bool
        True when reconstruction_error > anomaly_threshold.

    classifier_triggered : bool
        True when the hybrid classifier predicts a non-benign class.

    attack_type : str | None
        Human-readable attack category (e.g. "DoS", "PortScan",
        "Brute Force").  None for benign flows.

    classifier_confidence : float | None
        Probability score (0-1) returned by the hybrid classifier for the
        predicted class.  None when the classifier did not produce a
        probability estimate.

    inference_latency_ms : float
        Wall-clock time in milliseconds taken to execute both model forward
        passes.  Useful for SLA monitoring.
    """

    is_anomaly: bool = Field(
        ...,
        description=(
            "True when the combined autoencoder + hybrid-classifier pipeline "
            "determines that the flow is not benign."
        ),
    )

    reconstruction_error: float = Field(
        ...,
        ge=0.0,
        description=(
            "Mean-squared reconstruction error from the autoencoder. "
            "Higher values indicate greater deviation from normal traffic."
        ),
    )

    anomaly_threshold: float = Field(
        ...,
        ge=0.0,
        description=(
            "Decision boundary loaded from threshold.json. "
            "reconstruction_error > anomaly_threshold triggers the autoencoder gate."
        ),
    )

    autoencoder_triggered: bool = Field(
        ...,
        description="True when reconstruction_error exceeds anomaly_threshold.",
    )

    classifier_triggered: bool = Field(
        ...,
        description="True when the hybrid classifier predicts a non-benign class.",
    )

    attack_type: str | None = Field(
        default=None,
        description=(
            "Attack category assigned by the hybrid classifier "
            "(e.g. DoS, PortScan, Brute Force). "
            "None for benign flows."
        ),
        examples=["DoS", "PortScan", "Brute Force", None],
    )

    classifier_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Probability score (0-1) for the predicted class from the hybrid "
            "classifier. None when the model does not expose predict_proba."
        ),
    )

    inference_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Total wall-clock inference time in milliseconds.",
    )

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "is_anomaly": True,
                "reconstruction_error": 0.0842,
                "anomaly_threshold": 0.0500,
                "autoencoder_triggered": True,
                "classifier_triggered": True,
                "attack_type": "DoS",
                "classifier_confidence": 0.923,
                "inference_latency_ms": 12.4,
            }
        },
    )

# -- Attack-type classification result schema ----------------------------------

class AttackTypeResult(BaseModel):
    """
    Return schema for attacktype_classifier.attack_identifier.identify_attack().

    Fields
    ------
    attack_type : str
        Human-readable label for the predicted attack category, resolved from
        the attack_type_label_map.json file (e.g. "DoS", "PortScan",
        "Brute Force", "Bot", "Infiltration").

    attack_type_confidence : float
        Softmax probability (0-1) assigned to the top predicted class.
        Values close to 1.0 indicate high model certainty.

    class_index : int
        Raw integer class index output by the neural network before label
        resolution.  Useful for debugging or cross-referencing the label map.

    top_alternatives : list[dict]
        Ranked list of up to 2 runner-up predictions (excluding the top class),
        each containing "attack_type" (str) and "confidence" (float) keys.
        Empty list when the model has only one output class.

    all_attack_probabilities : dict[str, float]
        Full softmax probability distribution over every attack class known
        to the model, keyed by human-readable label.  All values sum to 1.0.
        Enables downstream consumers (e.g. the DQN) to see the complete
        probability mass rather than just the top prediction.

    inference_latency_ms : float
        Wall-clock time in milliseconds taken by the attack_type_nn forward
        pass (excludes feature scaling time).
    """

    attack_type: str = Field(
        ...,
        description=(
            "Human-readable attack category predicted by the attack_type_nn "
            "(e.g. 'DoS', 'PortScan', 'Brute Force')."
        ),
        examples=["DoS", "PortScan", "Brute Force", "Bot"],
    )

    attack_type_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Softmax probability (0-1) for the predicted attack class. "
            "Higher values indicate greater model certainty."
        ),
    )

    class_index: int = Field(
        ...,
        ge=0,
        description=(
            "Raw integer class index from the neural network output layer, "
            "before label resolution via attack_type_label_map.json."
        ),
    )

    top_alternatives: list[dict] = Field(
        default_factory=list,
        description=(
            "Up to 2 runner-up predictions, each with 'attack_type' (str) and "
            "'confidence' (float) keys. Empty when the model has one class."
        ),
    )

    all_attack_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Full softmax probability distribution over all attack classes, "
            "keyed by human-readable label (e.g. {\"DoS\": 0.973, \"Bot\": 0.018}). "
            "All values sum to 1.0."
        ),
    )

    inference_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Neural network forward-pass time in milliseconds.",
    )

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "attack_type": "DoS",
                "attack_type_confidence": 0.9731,
                "class_index": 2,
                "top_alternatives": [
                    {"attack_type": "Bot", "confidence": 0.0184},
                    {"attack_type": "PortScan", "confidence": 0.0061},
                ],
                "all_attack_probabilities": {
                    "DoS": 0.9731,
                    "Bot": 0.0184,
                    "PortScan": 0.0061,
                    "Brute Force": 0.0012,
                    "Infiltration": 0.0006,
                    "Web Attack": 0.0004,
                    "Heartbleed": 0.0002,
                },
                "inference_latency_ms": 4.7,
            }
        },
    )

# -- DQN remediation suggestion schemas ----------------------------------------

class DQNSuggestionRequest(BaseModel):
    """
    Input schema for dqn_agent.dqn_suggestion.suggest_action().

    This model collects everything the DQN needs to build its full state
    vector: the raw flow features, the anomaly signal from Stage 1, and the
    attack-type classification output from Stage 2.

    Fields
    ------
    attack_type : str
        Human-readable attack label predicted by the attack_type_nn
        (e.g. "DoS", "PortScan", "Brute Force").

    attack_confidence : float
        Softmax probability (0-1) for the predicted attack_type class.

    ae_reconstruction_error : float
        Mean-squared reconstruction error produced by the autoencoder in
        Stage 1.  Included in the DQN state vector as a severity signal.

    attack_type_probs : list[float]
        Full softmax probability vector over all N attack classes from
        attack_type_nn.  Length must equal len(attack_type_label_map).
        This gives the DQN richer context than the single argmax label.

    raw_features : list[float]
        The original 115 unscaled flow features.  The DQN was trained on
        these directly as part of its state representation.
    """

    attack_type: str = Field(
        ...,
        description="Attack label from Stage 2 classifier (e.g. 'DoS', 'PortScan').",
        examples=["DoS", "PortScan", "Brute Force"],
    )

    attack_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Top-1 softmax confidence from attack_type_nn (0-1).",
    )

    ae_reconstruction_error: float = Field(
        ...,
        ge=0.0,
        description="Autoencoder MSE reconstruction error from Stage 1.",
    )

    attack_type_probs: list[float] = Field(
        ...,
        min_length=1,
        description=(
            "Full softmax probability vector over all N attack classes "
            "from attack_type_nn. Must match len(attack_type_label_map)."
        ),
    )

    raw_features: list[float] = Field(
        ...,
        min_length=115,
        max_length=115,
        description="Ordered list of exactly 115 raw (unscaled) flow features.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "attack_type": "DoS",
                "attack_confidence": 0.973,
                "ae_reconstruction_error": 0.0842,
                "attack_type_probs": [0.003, 0.007, 0.973, 0.01, 0.002,
                                      0.001, 0.001, 0.001, 0.001, 0.001, 0.0],
                "raw_features": [0.0] * 115,
            }
        }
    )


class DQNSuggestionResult(BaseModel):
    """
    Output schema for dqn_agent.dqn_suggestion.suggest_action().

    Fields
    ------
    recommended_action : str
        The greedy action chosen by the DQN agent � one of:
        "Block IP" | "Revoke Credentials" | "Isolate Server" |
        "Kill Process" | "Monitor".

    action_index : int
        Integer index of the recommended action (0-4), matching the
        action space defined in train_dqn.py.

    action_confidence : float
        Softmax probability of the recommended action over all 5 Q-values.
        Indicates how decisively the agent prefers this action.

    q_value_table : dict[str, float]
        Mapping of action name -> raw Q-value for all 5 actions.
        Useful for debugging or transparency dashboards.

    alternative_actions : list[dict]
        Ranked list of the 4 non-chosen actions, each containing
        "action" (str), "action_index" (int), "q_value" (float),
        and "probability" (float) keys, sorted by Q-value descending.

    inference_latency_ms : float
        DQN forward-pass wall-clock time in milliseconds.
    """

    recommended_action: str = Field(
        ...,
        description=(
            "Greedy remediation action from the DQN. "
            "One of: Block IP | Revoke Credentials | Isolate Server | "
            "Kill Process | Monitor."
        ),
        examples=["Block IP", "Isolate Server", "Monitor"],
    )

    action_index: int = Field(
        ...,
        ge=0,
        le=4,
        description="Integer action index (0=Block IP, 1=Revoke, 2=Isolate, 3=Kill, 4=Monitor).",
    )

    action_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability of the recommended action over all 5 Q-values.",
    )

    q_value_table: dict[str, float] = Field(
        ...,
        description="Raw Q-values for all 5 actions keyed by action name.",
    )

    alternative_actions: list[dict] = Field(
        default_factory=list,
        description=(
            "Ranked list of the 4 non-chosen actions with action, "
            "action_index, q_value, and probability keys."
        ),
    )

    inference_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="DQN forward-pass time in milliseconds.",
    )

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "recommended_action": "Block IP",
                "action_index": 0,
                "action_confidence": 0.8812,
                "q_value_table": {
                    "Block IP": 4.21,
                    "Revoke Credentials": 1.05,
                    "Isolate Server": 2.34,
                    "Kill Process": 0.87,
                    "Monitor": -1.23,
                },
                "alternative_actions": [
                    {"action": "Isolate Server", "action_index": 2,
                     "q_value": 2.34, "probability": 0.0821},
                    {"action": "Revoke Credentials", "action_index": 1,
                     "q_value": 1.05, "probability": 0.0261},
                    {"action": "Kill Process", "action_index": 3,
                     "q_value": 0.87, "probability": 0.0219},
                    {"action": "Monitor", "action_index": 4,
                     "q_value": -1.23, "probability": 0.0027},
                ],
                "inference_latency_ms": 3.2,
            }
        },
    )


# ── Agent orchestration schemas ───────────────────────────────────────────────

class IncidentContext(BaseModel):
    """
    Unified incident payload produced by the ML pipeline and consumed by every
    agent in the orchestration system.

    This object is constructed from the pipeline outputs
    (anomaly classifier → attack type NN → DQN) and passed as the user
    message string to the Manager Agent.
    """

    # ── Network metadata ──────────────────────────────────────────────────────
    source_ip: str = Field(..., description="Source IP address of the suspicious flow.")
    dest_ip: str = Field(..., description="Destination IP address of the suspicious flow.")
    src_port: int | None = Field(None, ge=0, le=65535, description="Source TCP/UDP port.")
    dst_port: int | None = Field(None, ge=0, le=65535, description="Destination TCP/UDP port.")

    # ── Anomaly detection output (Stage 1 — Autoencoder) ─────────────────────
    recon_error: float = Field(
        ...,
        ge=0.0,
        description=(
            "Autoencoder mean-squared reconstruction error. "
            "Higher = more anomalous. Compared against anomaly_threshold."
        ),
    )
    anomaly_threshold: float = Field(
        0.05,
        ge=0.0,
        description="Decision boundary for the autoencoder gate.",
    )
    is_anomaly: bool = Field(
        ...,
        description="True when recon_error > anomaly_threshold.",
    )

    # ── Attack type classification (Stage 2 — Attack Type NN) ─────────────────
    attack_type: str | None = Field(
        None,
        description=(
            "Attack category predicted by the attack_type_nn. "
            "One of: DoS_Hulk | Port_Scan | DDoS_LOIT | FTP-Patator | "
            "DoS_GoldenEye | DoS_Slowhttptest | SSH-Patator | Botnet_ARES | "
            "DoS_Slowloris | Web_Brute_Force | Web_XSS | Web_SQL_Injection | "
            "Heartbleed. None for benign traffic."
        ),
    )
    attack_confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Softmax probability (0-1) for the predicted attack_type.",
    )

    # ── DQN remediation suggestion (Stage 3 — DQN Agent) ────────────────────
    dqn_action: str | None = Field(
        None,
        description=(
            "Remediation action recommended by the DQN agent. "
            "One of: Block IP | Revoke Credentials | Isolate Server | "
            "Kill Process | Monitor."
        ),
    )
    dqn_confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Softmax probability of the DQN's top action.",
    )

    # ── Optional context ──────────────────────────────────────────────────────
    incident_id: int | None = Field(
        None,
        description="Database primary key of the persisted Incident row (if stored).",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the incident was detected.",
    )
    raw_features: list[float] | None = Field(
        None,
        description="Optional: the raw 115-dim flow feature vector (for debugging).",
    )

    model_config = ConfigDict(from_attributes=True)

    # ── Serialisation helper ──────────────────────────────────────────────────
    def to_agent_message(self) -> str:
        """Return a human-readable summary string suitable as an LLM user message."""
        return (
            f"INCIDENT DETECTED\n"
            f"=================\n"
            f"Timestamp       : {self.timestamp.isoformat()}\n"
            f"Source IP       : {self.source_ip}  Port: {self.src_port}\n"
            f"Destination IP  : {self.dest_ip}  Port: {self.dst_port}\n"
            f"Recon Error     : {self.recon_error:.6f}  (threshold={self.anomaly_threshold})\n"
            f"Is Anomaly      : {self.is_anomaly}\n"
            f"Attack Type     : {self.attack_type or 'Unknown'}  "
            f"(confidence={self.attack_confidence or 0:.2%})\n"
            f"DQN Action      : {self.dqn_action or 'None'}  "
            f"(confidence={self.dqn_confidence or 0:.2%})\n"
            f"Incident ID     : {self.incident_id or 'N/A'}\n"
        )


class AgentResult(BaseModel):
    """
    Standardised response returned by the agent orchestrator after the full
    agent pipeline completes.
    """

    incident_id: int | None = None
    attack_type: str | None = None
    handling_agent: str = Field(..., description="Name of the specialist agent that handled the incident.")
    actions_taken: list[str] = Field(default_factory=list, description="List of tool actions executed.")
    final_response: str = Field(..., description="Natural-language summary from the specialist agent.")
    email_sent: bool = False
    broadcast_sent: bool = False
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)
