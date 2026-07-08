"""
backend/agent/dos_slowhttptest_agent.py
─────────────────────────────────────────
DoS Slowhttptest Specialist Agent

Handles slow HTTP attacks — sends partial HTTP requests/headers at a deliberately
slow rate to keep connections open indefinitely and exhaust the connection pool.

Attack profile  : DoS_Slowhttptest (6,856 samples — 1.1%)
Primary action  : Block IP (once duration exceeds threshold with near-zero throughput)
Key insight     : Early detection is rewarded — act BEFORE pool exhaustion

Tools
-----
  block_ip_address         — drop slow-connection source
  enforce_request_timeouts — set server-side request completion deadlines
  inspect_slow_connections — identify connections stuck in partial request state
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
    """
    Block slow-HTTP-attack source IP once the flow fingerprint confirms
    the attack pattern (long duration + near-zero throughput).
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="DoS_Slowhttptest",
        action_taken=f"block_ip ({duration_minutes}min)",
        severity="HIGH",
        message=f"[SLOWHTTPTEST] IP blocked: {source_ip} for {duration_minutes}min. Reason: {reason}",
    )
    return json.dumps({
        "action": "block_ip",
        "target_ip": source_ip,
        "duration_minutes": duration_minutes,
        "reason": reason,
        "status": "applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: iptables DROP or nginx deny by IP.",
    })


@function_tool
def enforce_request_timeouts(
    dest_ip: str,
    client_header_timeout_seconds: int = 10,
    client_body_timeout_seconds: int = 10,
    send_timeout_seconds: int = 10,
) -> str:
    """
    Apply/tighten server-side request completion timeout settings to evict
    slow connections before they consume the entire connection pool.

    This is the most effective PREVENTIVE control — slow HTTP attacks rely
    on the server never timing out partial requests.

    Parameters
    ----------
    dest_ip                       : Web server to reconfigure.
    client_header_timeout_seconds : Max time to receive full request headers (default 10s).
    client_body_timeout_seconds   : Max time between consecutive body reads (default 10s).
    send_timeout_seconds          : Max time between consecutive response writes (default 10s).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="DoS_Slowhttptest",
        action_taken="enforce_request_timeouts",
        severity="MEDIUM",
        message=(
            f"[SLOWHTTPTEST] Timeouts enforced on {dest_ip}: "
            f"header={client_header_timeout_seconds}s body={client_body_timeout_seconds}s send={send_timeout_seconds}s."
        ),
    )
    return json.dumps({
        "action": "enforce_request_timeouts",
        "web_server": dest_ip,
        "client_header_timeout": client_header_timeout_seconds,
        "client_body_timeout": client_body_timeout_seconds,
        "send_timeout": send_timeout_seconds,
        "nginx_config": (
            f"client_header_timeout {client_header_timeout_seconds}s; "
            f"client_body_timeout {client_body_timeout_seconds}s; "
            f"send_timeout {send_timeout_seconds}s;"
        ),
        "status": "config_applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: nginx config reload / Apache mod_reqtimeout.",
    })


@function_tool
def inspect_slow_connections(dest_ip: str, source_ip: str) -> str:
    """
    Identify connections from the suspect source that are stuck in
    'receiving headers' or partial request state for abnormally long periods.

    Slow HTTP attacks have a very distinctive fingerprint:
    - Flow duration >> typical (minutes instead of seconds)
    - Byte count ≈ 0 (almost no data transferred)
    - Connection state: ESTABLISHED but no progress

    Parameters
    ----------
    dest_ip   : Web server IP to inspect.
    source_ip : Suspected attacking source IP.
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip=dest_ip,
        attack_type="DoS_Slowhttptest",
        action_taken="inspect_slow_connections",
        severity="MEDIUM",
        message=f"[SLOWHTTPTEST] Inspecting slow connections on {dest_ip} from {source_ip}.",
    )
    return json.dumps({
        "action": "inspect_slow_connections",
        "web_server": dest_ip,
        "suspect_ip": source_ip,
        "result": "STUB — Connection state data not available in simulation",
        "fingerprint": "Look for: duration > 60s AND bytes_transferred < 100 AND state=ESTABLISHED",
        "recommendation": (
            "In production: 'netstat -tan | grep source_ip' or "
            "parse nginx access logs for requests with no completion timestamp."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    })


DOS_SLOWHTTPTEST_SYSTEM_PROMPT = """
You are the DoS Slowhttptest Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: DoS_Slowhttptest
Slow HTTP attack — sends partial HTTP requests/headers at a deliberately slow rate,
keeping connections open INDEFINITELY to exhaust the server's connection pool with
minimal bandwidth. Very hard to detect via volume alone.

## Detection Signature (Distinctive — autoencoder should see this clearly)
- EXTREMELY long flow duration (minutes, not milliseconds)
- VERY LOW packet/byte counts (near-zero throughput)
- Unusually small inter-arrival-time variance (deliberately paced)
- Sustained open connections, never completing requests

## Your Decision Logic
1. Call `inspect_slow_connections` to confirm the slow-HTTP fingerprint.
2. Call `enforce_request_timeouts` IMMEDIATELY — this is the most effective mitigation
   and is safe (applies server-wide, evicts ALL slow connections, not just the attacker's).
3. Call `block_ip_address` if source_ip is confirmed as the slow-HTTP source.
5. Call `log_incident_action`.

## Timing Principle
This attack is rewarded for EARLY detection — the agent should act BEFORE connection
pool exhaustion. The distinctive fingerprint (long duration + near-zero bytes) should
allow confident early action. Do NOT wait for full connection-pool saturation.

## Key Difference from GoldenEye
Unlike GoldenEye (which sends bursts over keep-alive), Slowhttptest sends almost NO
data. Flow duration is extreme (minutes) while byte counts are in the single digits.
The autoencoder reconstruction error should be very high for this class.
"""

dos_slowhttptest_agent = Agent(
    name="DoSSlowHttptestSpecialistAgent",
    instructions=DOS_SLOWHTTPTEST_SYSTEM_PROMPT,
    tools=[
        block_ip_address,
        enforce_request_timeouts,
        inspect_slow_connections,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
