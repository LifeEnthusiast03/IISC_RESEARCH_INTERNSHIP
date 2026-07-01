"""
backend/db/init_db.py
----------------------
Database helper functions for FastAPI:

  get_db()   -- SQLAlchemy session dependency (used by routers via Depends)
  init_db()  -- startup helper that creates all ORM tables

Separated from database.py so that connection setup (engine, SessionLocal)
stays isolated from application-level helpers.
"""

import logging

from backend.db.database import SessionLocal, engine

logger = logging.getLogger(__name__)


# ── FastAPI session dependency ────────────────────────────────────────────────
def get_db():
    """
    Yields a SQLAlchemy session and guarantees it is closed after the request,
    even if an exception occurs.

    Usage inside a router:
        def endpoint(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Table creation (called once at startup) ───────────────────────────────────
def init_db() -> None:
    """
    Create all tables registered on ``Base.metadata``.

    Imports ``Base`` and all model classes from ``backend.db.database_models`` so
    SQLAlchemy discovers every table definition before calling create_all().
    The import is done lazily inside this function to avoid circular imports
    (models.py defines Base; database.py provides the engine).
    """
    from backend.db.database_models import Base  # noqa: PLC0415 — lazy to avoid circular import

    logger.info("Running init_db(): creating tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("init_db() complete.")
