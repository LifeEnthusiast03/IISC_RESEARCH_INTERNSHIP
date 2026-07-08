"""
backend/agent/orchestrator.py
───────────────────────────────
Public API for the agent orchestration system.

This module provides the single callable function `run_agent_pipeline()` that:
1. Accepts an IncidentContext (or a dict that can be coerced into one)
2. Constructs the user message for the Manager Agent
3. Runs the full agent pipeline via the OpenAI Agents SDK
4. Returns a structured AgentResult

Usage (from FastAPI or CLI):
    from backend.agent.orchestrator import run_agent_pipeline
    from backend.schemas import IncidentContext

    context = IncidentContext(
        source_ip="192.168.1.105",
        dest_ip="10.0.0.1",
        src_port=54321,
        dst_port=80,
        recon_error=0.142,
        is_anomaly=True,
        attack_type="DoS_Hulk",
        attack_confidence=0.97,
        dqn_action="Block IP",
        dqn_confidence=0.88,
        incident_id=42,
    )

    result = await run_agent_pipeline(context)
    print(result.final_response)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agents import Runner

from backend.agent.manager_agent import manager_agent
from backend.schemas import AgentResult, IncidentContext

logger = logging.getLogger(__name__)


# ── Main orchestration function ───────────────────────────────────────────────

async def run_agent_pipeline(
    context: IncidentContext | dict[str, Any],
    openai_api_key: str | None = None,
) -> AgentResult:
    """
    Run the full multi-agent orchestration pipeline for a security incident.

    This is the primary public API of the agent module. It takes an
    IncidentContext, passes it to the Manager Agent, which then hands off
    to the appropriate specialist agent. The specialist executes its tools
    and returns a final response.

    Parameters
    ----------
    context        : The incident data from the ML pipeline. Can be an
                     IncidentContext object or a plain dict (will be coerced).
    openai_api_key : Optional OpenAI API key override. If not provided,
                     reads from OPENAI_API_KEY environment variable.

    Returns
    -------
    AgentResult    : Structured result with actions taken, final response,
                     and metadata.

    Raises
    ------
    ValueError     : If OPENAI_API_KEY is not set and no override is provided.
    Exception      : Re-raised from the OpenAI Agents SDK on API errors.
    """

    # ── Coerce dict → IncidentContext ─────────────────────────────────────────
    if isinstance(context, dict):
        context = IncidentContext(**context)

    # ── API key check ─────────────────────────────────────────────────────────
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Set it in your .env file or pass openai_api_key= parameter."
        )
    os.environ["OPENAI_API_KEY"] = api_key

    # ── Build the user message for the Manager Agent ──────────────────────────
    user_message = context.to_agent_message()

    logger.info(
        "[ORCHESTRATOR] Running agent pipeline | attack_type=%s | src=%s | is_anomaly=%s",
        context.attack_type,
        context.source_ip,
        context.is_anomaly,
    )

    # ── Run the agent pipeline ─────────────────────────────────────────────────
    try:
        result = await Runner.run(
            starting_agent=manager_agent,
            input=user_message,
        )

        # Extract the final response text
        final_response = result.final_output or "Agent pipeline completed with no text output."

        # Determine which agent handled the incident
        handling_agent = "ManagerAgent"
        if hasattr(result, "last_agent") and result.last_agent:
            handling_agent = result.last_agent.name

        logger.info(
            "[ORCHESTRATOR] Pipeline complete | handling_agent=%s | attack_type=%s",
            handling_agent,
            context.attack_type,
        )

        return AgentResult(
            incident_id=context.incident_id,
            attack_type=context.attack_type,
            handling_agent=handling_agent,
            actions_taken=[],   # Could parse from result.new_items in a real implementation
            final_response=final_response,
            broadcast_sent=True,
            metadata={
                "source_ip": context.source_ip,
                "dest_ip": context.dest_ip,
                "recon_error": context.recon_error,
                "dqn_action": context.dqn_action,
                "attack_confidence": context.attack_confidence,
            },
        )

    except Exception as exc:
        logger.error("[ORCHESTRATOR] Agent pipeline failed: %s", exc, exc_info=True)
        return AgentResult(
            incident_id=context.incident_id,
            attack_type=context.attack_type,
            handling_agent="ManagerAgent (FAILED)",
            final_response=f"Agent pipeline error: {exc}",
            metadata={"error": str(exc)},
        )


# ── Synchronous wrapper ───────────────────────────────────────────────────────

def run_agent_pipeline_sync(
    context: IncidentContext | dict[str, Any],
    openai_api_key: str | None = None,
) -> AgentResult:
    """
    Synchronous wrapper around run_agent_pipeline() for use in
    non-async contexts (e.g., scripts, Jupyter notebooks, Streamlit).
    """
    import asyncio
    return asyncio.run(run_agent_pipeline(context, openai_api_key))


# ── CLI test entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke test — run with:
        python -m backend.agent.orchestrator

    Reads OPENAI_API_KEY from the project .env file automatically.
    """
    import asyncio
    from datetime import datetime
    from pathlib import Path

    # ── Auto-load .env from project root ──────────────────────────────────────
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
    print(f"[ENV] Loaded .env from: {_env_path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    # ── Test payload — simulates a DoS_Hulk detection ─────────────────────────
    test_context = IncidentContext(
        source_ip="203.0.113.42",
        dest_ip="10.0.0.5",
        src_port=49512,
        dst_port=80,
        recon_error=0.1423,
        anomaly_threshold=0.05,
        is_anomaly=True,
        attack_type="DoS_Hulk",
        attack_confidence=0.973,
        dqn_action="Block IP",
        dqn_confidence=0.881,
        incident_id=1001,
        timestamp=datetime.utcnow(),
    )

    print("\n" + "=" * 60)
    print("  IDS Agent Orchestration — Smoke Test")
    print("=" * 60)
    print(test_context.to_agent_message())
    print("=" * 60 + "\n")

    result = asyncio.run(run_agent_pipeline(test_context))

    print("\n" + "=" * 60)
    print("  AGENT RESULT")
    print("=" * 60)
    print(f"Handling Agent : {result.handling_agent}")
    print(f"Attack Type    : {result.attack_type}")
    print(f"Incident ID    : {result.incident_id}")
    print(f"\nFinal Response :\n{result.final_response}")
    print("=" * 60 + "\n")
