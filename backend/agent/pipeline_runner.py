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


async def run_normal_traffic_agent(ctx: IncidentContext) -> None:
    """
    Run the standalone NormalTrafficBaselineAgent for benign flows and
    broadcast its summary back to the dashboard.

    Called as a fire-and-forget background task from predict_route.py
    whenever is_anomaly=False.  Uses the NormalTrafficBaselineAgent which
    is NOT wired into the Manager Agent — it is a fully independent agent.

    On success, broadcasts a ``normal_traffic_response`` event.
    On failure, broadcasts a ``normal_traffic_error`` event.

    Parameters
    ----------
    ctx : IncidentContext
        The incident context for the benign flow.
    """
    from backend.agent.normal_traffic_agent import normal_traffic_agent  # local import avoids circular deps
    from agents import Runner
    import os

    try:
        # Build a minimal user message describing the benign flow
        user_message = (
            f"Normal (benign) network flow detected.\n\n"
            f"Source IP     : {ctx.source_ip}:{ctx.src_port}\n"
            f"Destination IP: {ctx.dest_ip}:{ctx.dst_port}\n"
            f"Recon Error   : {ctx.recon_error:.6f}\n"
            f"Is Anomaly    : {ctx.is_anomaly}\n"
            f"Incident ID   : {ctx.incident_id}\n\n"
            f"Please log this flow as baseline traffic and confirm no action is needed."
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        os.environ["OPENAI_API_KEY"] = api_key

        result = await Runner.run(
            starting_agent=normal_traffic_agent,
            input=user_message,
        )

        final_response = result.final_output or "Baseline flow logged — no anomaly detected."

        await manager.broadcast({
            "event":          "normal_traffic_response",
            "incident_id":    ctx.incident_id,
            "source_ip":      ctx.source_ip,
            "dest_ip":        ctx.dest_ip,
            "recon_error":    round(ctx.recon_error, 6),
            "handling_agent": "NormalTrafficBaselineAgent",
            "final_response": final_response,
        })

        logger.info(
            "[PIPELINE RUNNER] Normal traffic agent complete | incident=%s | src=%s",
            ctx.incident_id,
            ctx.source_ip,
        )

    except Exception as exc:
        logger.error(
            "[PIPELINE RUNNER] Normal traffic agent failed for incident %s: %s",
            ctx.incident_id,
            exc,
        )
        await manager.broadcast({
            "event":       "normal_traffic_error",
            "incident_id": ctx.incident_id,
            "error":       str(exc),
        })
