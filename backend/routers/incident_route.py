"""
backend/routers/incidents.py
─────────────────────────────
CRUD-read endpoints for persisted Incident records.

Routes
------
GET /incidents                 — paginated list, newest-first
GET /incidents/{incident_id}   — single incident by PK or 404
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.init_db import get_db
from backend.db.database_models import Incident
from backend.schemas import IncidentListResponse, IncidentOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/incidents", tags=["incidents"])

# Hard cap on page_size to prevent abuse
_MAX_PAGE_SIZE = 100


@router.get(
    "",
    response_model=IncidentListResponse,
    summary="List incidents (paginated)",
    description=(
        "Returns a paginated, newest-first list of all persisted incidents.  "
        "Use ``page`` and ``page_size`` query parameters to navigate."
    ),
)
def list_incidents(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(
        default=20,
        ge=1,
        le=_MAX_PAGE_SIZE,
        description=f"Items per page (max {_MAX_PAGE_SIZE}).",
    ),
    db: Session = Depends(get_db),
) -> IncidentListResponse:
    """
    Return all incidents ordered by ``created_at`` descending (newest first).

    The total count is fetched in a separate scalar query so the offset/limit
    pagination is accurate even when rows are being inserted concurrently.
    """
    total: int = db.query(Incident).count()

    offset = (page - 1) * page_size
    rows = (
        db.query(Incident)
        .order_by(Incident.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    logger.debug(
        "list_incidents: page=%d size=%d  returned=%d  total=%d",
        page,
        page_size,
        len(rows),
        total,
    )

    return IncidentListResponse(
        items=[IncidentOut.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{incident_id}",
    response_model=IncidentOut,
    summary="Get a single incident",
    description="Retrieve a single Incident record by its integer primary key.",
    responses={404: {"description": "Incident not found"}},
)
def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
) -> IncidentOut:
    """
    Fetch one incident by primary key.

    Raises HTTP 404 if the ID does not exist in the database.
    """
    row = db.query(Incident).filter(Incident.id == incident_id).first()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Incident with id={incident_id} not found.",
        )
    return IncidentOut.model_validate(row)
