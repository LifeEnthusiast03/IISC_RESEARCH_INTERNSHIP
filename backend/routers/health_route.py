"""
backend/routers/health.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GET /health â€” operational health-check endpoint.

Returns a snapshot of:
  â€¢ Overall service status  ("ok" | "degraded")
  â€¢ Whether the autoencoder / DQN artefacts are loaded in memory
  â€¢ Whether a Postgres round-trip succeeds
  â€¢ How long the process has been running (uptime)

The inference-readiness flags are read from ``backend.inference`` module-level
state variables (``autoencoder_ready`` and ``dqn_ready``) that are set during
startup model loading.  They default to False so the health check degrades
gracefully even before models are trained.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter

from backend.db.database import check_db_connection
from backend.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Process start time â€” used to compute uptime_seconds
_START_TIME: float = time.monotonic()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the operational status of the API, the ML model artefacts, "
        "and the PostgreSQL database connection."
    ),
)
def health_check() -> HealthResponse:
    """
    Lightweight health-check.

    The database ping uses a single ``SELECT 1`` wrapped in try/except so the
    endpoint always responds (with ``status='degraded'``) even when Postgres
    is down â€” it never raises a 500.

    ML model readiness is derived from module-level flags in
    ``backend.inference`` (set to True once the .pt artefacts are loaded).
    """
    # â”€â”€ DB connectivity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    db_ok = check_db_connection()

    # â”€â”€ ML model readiness â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    autoencoder_loaded = False
    dqn_loaded = False
    try:
        import backend.models.init_models as _inf  # lazy import avoids circular deps
        autoencoder_loaded = getattr(_inf, "autoencoder_ready", False)
        dqn_loaded = getattr(_inf, "dqn_agent_ready", False)
    except Exception:  # noqa: BLE001
        pass  # inference module may not be initialised yet

    # ── Overall status ───────────────────────────────────────────────────────
    all_ok = db_ok and autoencoder_loaded and dqn_loaded
    status = "ok" if all_ok else "degraded"

    if not all_ok:
        logger.warning(
            "Health check degraded — db=%s, autoencoder=%s, dqn=%s",
            db_ok,
            autoencoder_loaded,
            dqn_loaded,
        )

    return HealthResponse(
        status=status,
        autoencoder_loaded=autoencoder_loaded,
        dqn_loaded=dqn_loaded,
        db_connected=db_ok,
        uptime_seconds=time.monotonic() - _START_TIME,
    )
