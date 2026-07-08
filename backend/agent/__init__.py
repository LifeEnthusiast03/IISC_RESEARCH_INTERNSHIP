"""
backend/agent/__init__.py
──────────────────────────
Agent orchestration package.

Public API:
  run_agent_pipeline      — async entry point (use in FastAPI)
  run_agent_pipeline_sync — sync wrapper (use in scripts / Streamlit)
  IncidentContext         — input model
  AgentResult             — output model
  manager_agent           — the root Manager Agent object
"""

from backend.schemas import AgentResult, IncidentContext
from backend.agent.orchestrator import run_agent_pipeline, run_agent_pipeline_sync
from backend.agent.manager_agent import manager_agent

__all__ = [
    "run_agent_pipeline",
    "run_agent_pipeline_sync",
    "IncidentContext",
    "AgentResult",
    "manager_agent",
]
