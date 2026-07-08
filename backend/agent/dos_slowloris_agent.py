"""
backend/agent/dos_slowloris_agent.py
──────────────────────────────────────
DoS Slowloris Specialist Agent

Handles Slowloris attacks — sends partial HTTP headers very slowly to hold open
many connections without ever completing a request, exhausting max-connections limit.

Attack profile  : DoS_Slowloris (5,122 samples — 0.85%)
Primary action  : Block IP
Note            : Nearly identical to Slowhttptest at the flow-feature level.
                  Same "slow connection exhaustion" policy applies to both.

Tools
-----
  block_ip_address         — drop slowloris source connections
  enforce_request_timeouts — evict slow connections via server timeout tuning
  check_connection_pool    — assess how full the server's connection pool is
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
def block_ip_address(source_ip: str, reason: str, duration_minutes: int = 90) -> str:
    """Block Slowloris attack source IP."""
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="DoS_Slowloris",
        action_taken=f"block_ip ({duration_minutes}min)",
        severity="HIGH",
        message=f"[SLOWLORIS] IP blocked: {source_ip} for {duration_minutes}min. Slowloris attack detected.",
    )
    return json.dumps({
        "action": "block_ip",
        "target_ip": source_ip,
        "duration_minutes": duration_minutes,
        "reason": reason,
        "status": "applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: iptables / nginx deny / mod_antiloris.",
    })


@function_tool
def enforce_request_timeouts(
    dest_ip: str,
    client_header_timeout_seconds: int = 10,
    keepalive_requests_max: int = 100,
) -> str:
    """
    Enforce request completion timeouts and limit max keep-alive requests
    to evict Slowloris connections before they fill the pool.

    Slowloris specifically targets the max-connections limit by sending
    keep-alive headers just often enough to avoid timing out.

    Parameters
    ----------
    dest_ip                    : Web server to reconfigure.
    client_header_timeout_seconds: Max time to receive full headers (default 10s).
    keepalive_requests_max     : Max requests per keep-alive connection (default 100).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="DoS_Slowloris",
        action_taken="enforce_request_timeouts",
        severity="MEDIUM",
        message=f"[SLOWLORIS] Timeouts enforced on {dest_ip}: header_timeout={client_header_timeout_seconds}s, keepalive_reqs={keepalive_requests_max}.",
    )
    return json.dumps({
        "action": "enforce_request_timeouts",
        "web_server": dest_ip,
        "client_header_timeout": client_header_timeout_seconds,
        "keepalive_requests_max": keepalive_requests_max,
        "apache_module": "mod_antiloris / mod_reqtimeout",
        "nginx_config": f"client_header_timeout {client_header_timeout_seconds}s; keepalive_requests {keepalive_requests_max};",
        "status": "config_applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: nginx/Apache config reload.",
    })


@function_tool
def check_connection_pool_status(dest_ip: str) -> str:
    """
    Check how full the server's connection pool is to assess urgency.
    If pool is > 80% full, escalate to CRITICAL immediately.

    Parameters
    ----------
    dest_ip : Web server IP to check.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="DoS_Slowloris",
        action_taken="check_connection_pool_status",
        severity="MEDIUM",
        message=f"[SLOWLORIS] Checking connection pool status on {dest_ip}.",
    )
    return json.dumps({
        "action": "check_connection_pool_status",
        "web_server": dest_ip,
        "result": "STUB — pool status not available in simulation",
        "thresholds": {
            "MEDIUM": "< 50% pool utilisation",
            "HIGH": "50-80% pool utilisation",
            "CRITICAL": "> 80% pool utilisation (imminent service failure)",
        },
        "recommendation": (
            "In production: check nginx active_connections / Apache MaxClients "
            "vs. ServerLimit. If > 80%: consider temporary emergency connection cap."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    })


DOS_SLOWLORIS_SYSTEM_PROMPT = """
You are the DoS Slowloris Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: DoS_Slowloris
Slowloris sends partial HTTP headers very slowly — one connection at a time — to hold open
many connections without ever completing a request, exhausting the server's max-connections
limit with minimal bandwidth.

## Detection Signature
- Nearly identical to Slowhttptest: long duration, minimal bytes, slow periodic keep-alive packets
- Many concurrent flows from one source
- The DQN generalizes a single "slow connection exhaustion" policy for both Slowloris and Slowhttptest

## Your Decision Logic
1. Call `check_connection_pool_status` to assess urgency and set severity.
2. Call `enforce_request_timeouts` — safe, server-wide mitigation. Always apply this.
3. Call `block_ip_address` to stop the ongoing slow connection attack.
   - Pool < 50%: HIGH
   - Pool 50-80%: HIGH → CRITICAL
   - Pool > 80%: CRITICAL (immediate action required)
5. Call `log_incident_action`.

## Relationship to Slowhttptest
Slowloris and Slowhttptest are nearly indistinguishable at the flow-feature level.
The DQN uses the same policy for both. Your remediation strategy is identical.
The key difference (Slowloris specifically holds headers open, Slowhttptest holds bodies)
does not change the optimal response.

Apply the same "timeout enforcement + block" response as for Slowhttptest.
Early action (before pool exhaustion) is significantly rewarded.
"""

dos_slowloris_agent = Agent(
    name="DoSSlowlorisSpecialistAgent",
    instructions=DOS_SLOWLORIS_SYSTEM_PROMPT,
    tools=[
        block_ip_address,
        enforce_request_timeouts,
        check_connection_pool_status,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
