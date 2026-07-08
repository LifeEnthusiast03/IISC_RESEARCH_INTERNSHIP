"""
backend/agent/pipeline_runner.py
─────────────────────────────────
Helper that runs the multi-agent pipeline for a detected incident and
broadcasts the final agent response back to all connected WebSocket clients.

This module is the bridge between the ML inference route and the agent
orchestration system — it is imported by predict_route.py and called
as a background asyncio task.
"""

from __future__ import annotations

import logging

from backend.agent.orchestrator import run_agent_pipeline
from backend.schemas import IncidentContext
from backend.websocket.connection import manager

logger = logging.getLogger(__name__)


async def run_and_broadcast(ctx: IncidentContext) -> None:
    """
    Await the multi-agent pipeline and push its final output to the dashboard.

    Intended to be scheduled as a background task via asyncio.create_task():
        asyncio.create_task(run_and_broadcast(agent_context))

    On success, broadcasts an ``agent_response`` event to all WebSocket clients.
    On failure, broadcasts an ``agent_error`` event so the dashboard can surface
    the failure without silently swallowing it.

    Parameters
    ----------
    ctx : IncidentContext
        The fully-populated incident context produced by the ML pipeline.
    """
    try:
        result = await run_agent_pipeline(ctx)
        await manager.broadcast({
            "event":          "agent_response",
            "incident_id":    result.incident_id,
            "attack_type":    result.attack_type,
            "handling_agent": result.handling_agent,
            "actions_taken":  result.actions_taken,
            "final_response": result.final_response,
            "broadcast_sent": result.broadcast_sent,
        })
        logger.info(
            "[PIPELINE RUNNER] Broadcast complete | agent=%s | incident=%s",
            result.handling_agent,
            result.incident_id,
        )
    except Exception as exc:
        logger.error(
            "[PIPELINE RUNNER] Agent pipeline failed for incident %s: %s",
            ctx.incident_id,
            exc,
        )
        await manager.broadcast({
            "event":       "agent_error",
            "incident_id": ctx.incident_id,
            "error":       str(exc),
        })
