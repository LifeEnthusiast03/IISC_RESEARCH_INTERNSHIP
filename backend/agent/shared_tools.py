"""
backend/agent/shared_tools.py
──────────────────────────────
Shared tool functions available to ALL agents in the orchestration system.

Tools defined here:
  • broadcast_alert_to_client  — sends a JSON alert over the WebSocket channel
  • log_incident_action        — writes an action record to the Python logger

Each function is decorated with @function_tool so the OpenAI Agents SDK can
expose it as a callable tool to any agent that imports it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from agents import function_tool

logger = logging.getLogger(__name__)


# ── WebSocket broadcast ────────────────────────────────────────────────────────

def broadcast_alert_to_client(
    source_ip: str,
    dest_ip: str,
    attack_type: str,
    action_taken: str,
    severity: str,
    message: str,
    incident_id: int = -1,
) -> str:
    """
    Broadcast a real-time security alert to all connected WebSocket clients
    on the dashboard (ws://host/ws/connect).

    Parameters
    ----------
    source_ip    : The attacker / anomalous source IP address.
    dest_ip      : The target destination IP address.
    attack_type  : Detected attack category (e.g. 'DoS_Hulk', 'Port_Scan').
    action_taken : The remediation action being executed (e.g. 'Block IP').
    severity     : Alert severity level — one of: LOW | MEDIUM | HIGH | CRITICAL.
    message      : Human-readable description of what happened and what action was taken.
    incident_id  : Database incident record ID (-1 if not persisted yet).

    Returns
    -------
    JSON string confirming the broadcast was dispatched.
    """
    payload = {
        "event": "security_alert",
        "timestamp": datetime.utcnow().isoformat(),
        "incident_id": incident_id,
        "source_ip": source_ip,
        "dest_ip": dest_ip,
        "attack_type": attack_type,
        "action_taken": action_taken,
        "severity": severity,
        "message": message,
    }

    # ── PRODUCTION HOOK ──────────────────────────────────────────────────────
    from backend.websocket.connection import manager
    import asyncio
    
    # Depending on how the agent is executed (sync or async context),
    # we dispatch the WebSocket broadcast safely.
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(manager.broadcast(payload))
    except RuntimeError:
        # Fallback if called from a synchronous thread without a running loop
        asyncio.run(manager.broadcast(payload))
    # ────────────────────────────────────────────────────────────────────────
    logger.info("[BROADCAST] WebSocket alert dispatched → %s", json.dumps(payload))

    return json.dumps({
        "status": "broadcast_sent",
        "channel": "ws://host/ws/connect",
        "payload": payload,
    })




# ── Action logger ─────────────────────────────────────────────────────────────

@function_tool
def log_incident_action(
    incident_id: int,
    agent_name: str,
    action: str,
    outcome: str,
    notes: str = "",
) -> str:
    """
    Write a structured action log entry for an incident.

    This is always called by specialist agents after executing a remediation
    action so there is a complete audit trail regardless of whether the action
    was a real API call or a simulation.

    Parameters
    ----------
    incident_id : Database incident record ID.
    agent_name  : Name of the specialist agent taking the action.
    action      : The action that was executed (e.g. 'block_ip', 'isolate_server').
    outcome     : Result description — 'success', 'failed', 'pending', etc.
    notes       : Optional free-text notes or error details.

    Returns
    -------
    JSON confirmation string.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "incident_id": incident_id,
        "agent": agent_name,
        "action": action,
        "outcome": outcome,
        "notes": notes,
    }
    logger.info("[ACTION LOG] %s", json.dumps(entry))
    return json.dumps({"status": "logged", "entry": entry})
