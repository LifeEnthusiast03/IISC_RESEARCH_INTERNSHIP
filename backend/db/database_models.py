"""
backend/db/models.py
--------------------
SQLAlchemy ORM models for the Incident Tracking system.

``Base`` (the declarative base) is defined here so this module is the single
source of truth for the database schema. ``init_db()`` in database.py imports
``Base`` from here to call ``Base.metadata.create_all()``.

Table: incidents
────────────────
Stores every flow that passes through the two-stage ML pipeline
(autoencoder → DQN).  Both normal and anomalous flows are persisted so
the dashboard can display full traffic history alongside detected threats.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

# Base is defined here — the single source of truth for the ORM schema.
# init_db() in database.py imports Base from this module to create tables.
Base = declarative_base()


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


class Incident(Base):
    """
    Represents a single network flow processed by the IDS pipeline.

    Columns
    -------
    id                   : Auto-incrementing primary key.
    timestamp            : When the flow was observed / submitted (UTC).
    source_ip            : Source IP address extracted from the flow record.
    dest_ip              : Destination IP address.
    src_port             : Source port (nullable — may not be present in all datasets).
    dst_port             : Destination port (nullable).
    reconstruction_error : Autoencoder MSE loss for this flow.
    is_anomaly           : True when reconstruction_error exceeds the threshold.
    attack_type_predicted: Human-readable attack label from the DQN / classifier
                           (e.g. "DoS", "PortScan", "Brute Force"). NULL for benign.
    dqn_action           : Remediation action chosen by the DQN agent.
                           One of: block_ip | revoke_credentials |
                                   isolate_server | kill_process | monitor
    action_status        : Lifecycle of the chosen action.
                           One of: pending | executed | simulated
    raw_features         : The original 78- or 115-feature vector stored as JSON
                           so we can replay or audit any prediction.
    created_at           : Row insertion timestamp (UTC).
    """

    __tablename__ = "incidents"

    # ── Primary key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Temporal ──────────────────────────────────────────────────────────────
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )

    # ── Network identifiers ───────────────────────────────────────────────────
    source_ip: Mapped[str] = mapped_column(String(45), nullable=False)  # IPv6 max 45 chars
    dest_ip: Mapped[str] = mapped_column(String(45), nullable=False)
    src_port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dst_port: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Autoencoder output ────────────────────────────────────────────────────
    reconstruction_error: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # ── DQN / classifier output ───────────────────────────────────────────────
    attack_type_predicted: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dqn_action: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="block_ip | revoke_credentials | isolate_server | kill_process | monitor",
    )

    # ── Action lifecycle ──────────────────────────────────────────────────────
    action_status: Mapped[str] = mapped_column(
        String(16),
        default="pending",
        nullable=False,
        comment="pending | executed | simulated",
    )

    # ── Raw feature vector ────────────────────────────────────────────────────
    raw_features: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ── Audit timestamp ───────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Incident id={self.id} is_anomaly={self.is_anomaly} "
            f"action={self.dqn_action} status={self.action_status}>"
        )
