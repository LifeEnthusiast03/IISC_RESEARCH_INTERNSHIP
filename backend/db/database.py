"""
backend/db/database.py
----------------------
Database engine, session factory, and helper utilities.

Reads DATABASE_URL from the environment (via .env loaded by python-dotenv).
Falls back to a local Postgres instance with default credentials if the
variable is not set.

Note: ``Base`` (the SQLAlchemy declarative base) lives in ``backend.db.database_models``
and is imported from there by ``init_db()``.
"""

import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)

# ── Connection string ─────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/ids_db",
)

# ── Engine ────────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # recycle stale connections automatically
    pool_size=10,
    max_overflow=20,
)

# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



def check_db_connection() -> bool:
    """
    Lightweight connectivity check.  Returns True if a round-trip SELECT 1
    succeeds, False otherwise.  Used by the /health endpoint.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB connectivity check failed: %s", exc)
        return False
