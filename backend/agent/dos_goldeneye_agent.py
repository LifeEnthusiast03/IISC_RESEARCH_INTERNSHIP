"""
backend/agent/dos_goldeneye_agent.py
──────────────────────────────────────
DoS GoldenEye Specialist Agent

Handles HTTP keep-alive connection exhaustion attacks — persistent sockets
designed to exhaust the server's concurrent-connection limit.

Attack profile  : DoS_GoldenEye (8,363 samples — 1.4%)
Primary action  : Block IP
Caution         : Higher false-positive risk than Hulk (legitimate WebSocket/streaming
                  connections also have long-lived flows) — use temporal persistence check

Tools
-----
  block_ip_address          — firewall drop for exhausting source
  tune_keepalive_limits     — reduce keep-alive timeout / max connections per IP
  monitor_connection_table  — inspect per-IP concurrent connection counts
  log_incident_action
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from agents import Agent, ModelSettings, function_tool

from backend.agent.shared_tools import broadcast_alert_to_client, log_incident_action

logger = logging.getLogger(__name__)


@function_tool
def block_ip_address(source_ip: str, reason: str, duration_minutes: int = 60) -> str:
    """
    Block traffic from GoldenEye connection-exhaustion source IP.
    Only call after confirming anomaly is persistent (not a single-window spike).
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="DoS_GoldenEye",
        action_taken=f"block_ip ({duration_minutes}min)",
        severity="HIGH",
        message=f"[GOLDENEYE] IP blocked: {source_ip} for {duration_minutes}min. Keep-alive flood detected.",
    )
    return json.dumps({
        "action": "block_ip",
        "target_ip": source_ip,
        "duration_minutes": duration_minutes,
        "reason": reason,
        "status": "applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: iptables / nginx limit_conn / WAF.",
    })


@function_tool
def tune_keepalive_limits(
    dest_ip: str,
    max_connections_per_ip: int = 10,
    keepalive_timeout_seconds: int = 30,
) -> str:
    """
    Reduce keep-alive connection limits and timeout values on the web server
    to prevent connection pool exhaustion.

    This is a configuration change, not a block — it applies to ALL clients
    and reduces the attack surface without blocking legitimate traffic.

    Parameters
    ----------
    dest_ip                  : Web server IP to reconfigure.
    max_connections_per_ip   : New max concurrent connections per client IP (default 10).
    keepalive_timeout_seconds: New keep-alive timeout in seconds (default 30s).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="DoS_GoldenEye",
        action_taken="tune_keepalive_limits",
        severity="MEDIUM",
        message=f"[GOLDENEYE] Keep-alive limits tuned on {dest_ip}: max_conn/ip={max_connections_per_ip}, timeout={keepalive_timeout_seconds}s.",
    )
    return json.dumps({
        "action": "tune_keepalive_limits",
        "web_server": dest_ip,
        "new_max_connections_per_ip": max_connections_per_ip,
        "new_keepalive_timeout_seconds": keepalive_timeout_seconds,
        "config_change": "nginx: limit_conn_zone + keepalive_timeout",
        "status": "config_applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: nginx config reload / Apache tuning via API.",
    })


@function_tool
def monitor_connection_table(dest_ip: str, source_ip: str) -> str:
    """
    Inspect the web server's active connection table to measure per-IP
    concurrent connection counts and identify abnormal consumers.

    Use this to confirm the GoldenEye attack before taking disruptive action,
    especially when attack_confidence is < 0.8 (higher false-positive risk
    due to legitimate long-lived connection similarity).

    Parameters
    ----------
    dest_ip   : Web server IP to inspect.
    source_ip : Suspected attacking source IP.
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip=dest_ip,
        attack_type="DoS_GoldenEye",
        action_taken="monitor_connection_table",
        severity="MEDIUM",
        message=f"[GOLDENEYE] Monitoring connection table on {dest_ip} for {source_ip}.",
    )
    return json.dumps({
        "action": "monitor_connection_table",
        "web_server": dest_ip,
        "suspect_ip": source_ip,
        "result": "STUB — connection count data not available in simulation",
        "recommendation": (
            "In production: parse 'netstat -an | grep ESTABLISHED' or "
            "nginx status module to count per-IP connections. "
            "If source_ip has > 50 concurrent connections → confirmed GoldenEye."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    })


DOS_GOLDENEYE_SYSTEM_PROMPT = """
You are the DoS GoldenEye Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: DoS_GoldenEye
GoldenEye uses persistent HTTP keep-alive connections with a smaller number of long-lived
sockets specifically designed to exhaust the server's concurrent-connection limit (not
raw bandwidth like Hulk). It sends cyclical small packet bursts over many keep-alive sessions.

## Detection Signature
- Long-duration flows (keep-alive) with cyclical small packet bursts
- High connection count per source IP
- Moderate (not extreme) packet counts — key difference from DoS_Hulk
- Elevated reconstruction error

## IMPORTANT: False-Positive Risk
GoldenEye is harder to distinguish from legitimate traffic than Hulk because:
- WebSocket connections are also long-lived
- Streaming APIs maintain persistent connections
- Video conferencing uses similar flow patterns

Therefore: do NOT block on a single anomaly window if attack_confidence < 0.8.
Use temporal persistence (anomaly persists over 2+ consecutive windows) before blocking.

## Your Decision Logic
1. Call `monitor_connection_table` to assess the actual connection count from source_ip.
2. If attack_confidence >= 0.8 AND recon_error is sustained:
   → Call `block_ip_address`
   → Call `tune_keepalive_limits` to harden the server
3. If attack_confidence < 0.8:
   → Call `tune_keepalive_limits` only (configuration hardening without blocking)
   → Do NOT block yet — recommend Monitor mode and re-evaluation in next window
4. Always call `log_incident_action`.

## Principle
Slightly favour Monitor when ambiguous. Block only when confident.
Tune keepalive limits regardless — it's a safe hardening action.
"""

dos_goldeneye_agent = Agent(
    name="DoSGoldenEyeSpecialistAgent",
    instructions=DOS_GOLDENEYE_SYSTEM_PROMPT,
    tools=[
        block_ip_address,
        tune_keepalive_limits,
        monitor_connection_table,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
