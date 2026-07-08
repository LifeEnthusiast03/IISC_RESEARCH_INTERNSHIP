"""
backend/agent/web_bruteforce_agent.py
───────────────────────────────────────
Web Brute Force Specialist Agent

Handles automated login-guessing against a web application's authentication endpoint.
Application-layer brute force (login form / API), NOT a network-layer service.

Attack profile  : Web_Brute_Force (2,733 samples — 0.46%)
Primary action  : Block IP
Escalation      : Revoke Credentials if successful login after burst

Tools
-----
  block_ip_address          — WAF / application-layer IP block on login endpoint
  revoke_web_credentials    — force password reset / session invalidation
  enable_captcha_lockout    — activate CAPTCHA and account lockout policy
  check_web_auth_logs       — check application logs for successful login
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
def block_ip_address(source_ip: str, reason: str, duration_minutes: int = 120) -> str:
    """
    Block the brute-forcing source IP at the WAF or application layer,
    specifically targeting the login endpoint rate-limiting rule.
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="Web_Brute_Force",
        action_taken=f"block_ip ({duration_minutes}min)",
        severity="HIGH",
        message=f"[WEB_BRUTEFORCE] IP blocked: {source_ip} for {duration_minutes}min. Web brute-force detected.",
    )
    return json.dumps({
        "action": "block_ip",
        "target_ip": source_ip,
        "layer": "WAF / application (login endpoint)",
        "duration_minutes": duration_minutes,
        "reason": reason,
        "status": "applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: WAF IP block / nginx geo deny / Cloudflare IP rule.",
    })


@function_tool
def revoke_web_credentials(
    username: str,
    dest_ip: str,
    invalidate_active_sessions: bool = True,
) -> str:
    """
    Force password reset for a web application account that may have been
    compromised during a brute-force attack, and invalidate all active sessions.

    Parameters
    ----------
    username                  : Application account username to reset.
    dest_ip                   : Web application server IP.
    invalidate_active_sessions: Whether to kill all existing sessions (default True).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="Web_Brute_Force",
        action_taken=f"revoke_web_credentials (user={username})",
        severity="CRITICAL",
        message=f"[WEB_BRUTEFORCE] CRITICAL: Web credentials revoked for '{username}' on {dest_ip}.",
    )
    return json.dumps({
        "action": "revoke_web_credentials",
        "username": username,
        "web_server": dest_ip,
        "password_reset_required": True,
        "sessions_invalidated": invalidate_active_sessions,
        "severity": "CRITICAL",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: application API / user DB update / session store flush.",
    })


@function_tool
def enable_captcha_and_lockout(
    dest_ip: str,
    lockout_threshold: int = 5,
    lockout_duration_minutes: int = 30,
) -> str:
    """
    Activate CAPTCHA challenge on the login endpoint and enable account
    lockout policy after N failed attempts.

    This is a preventive hardening action applied to the web server/application,
    not a block on a specific IP — it improves defense for all future attempts.

    Parameters
    ----------
    dest_ip               : Web server IP where the login endpoint lives.
    lockout_threshold     : Number of failed attempts before lockout (default 5).
    lockout_duration_minutes: How long the lockout lasts (default 30 min).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="Web_Brute_Force",
        action_taken=f"enable_captcha_and_lockout (threshold={lockout_threshold})",
        severity="MEDIUM",
        message=f"[WEB_BRUTEFORCE] CAPTCHA and lockout enabled on {dest_ip}. Threshold: {lockout_threshold}.",
    )
    return json.dumps({
        "action": "enable_captcha_and_lockout",
        "web_server": dest_ip,
        "captcha_enabled": True,
        "lockout_threshold_attempts": lockout_threshold,
        "lockout_duration_minutes": lockout_duration_minutes,
        "status": "config_applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: WAF rule update / application config change.",
    })


@function_tool
def check_web_auth_logs(source_ip: str, dest_ip: str, window_minutes: int = 10) -> str:
    """
    Check web application authentication logs for successful logins from
    the brute-forcing source IP within the attack window.

    Parameters
    ----------
    source_ip      : Attacking source IP to check in auth logs.
    dest_ip        : Web application server to query.
    window_minutes : How far back to look in logs (default 10 min).
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip=dest_ip,
        attack_type="Web_Brute_Force",
        action_taken=f"check_web_auth_logs (window={window_minutes}m)",
        severity="MEDIUM",
        message=f"[WEB_BRUTEFORCE] Checking web auth logs for {source_ip} on {dest_ip} (window: {window_minutes}m).",
    )
    return json.dumps({
        "action": "check_web_auth_logs",
        "source_ip": source_ip,
        "web_server": dest_ip,
        "window_minutes": window_minutes,
        "result": "NO_SUCCESS_DETECTED_STUB",
        "note": (
            "STUB — In production: query application logs / SIEM for HTTP 200 "
            "responses to POST /login from source_ip within window_minutes."
        ),
        "recommendation": "If SUCCESS: call revoke_web_credentials immediately.",
        "timestamp": datetime.utcnow().isoformat(),
    })


WEB_BRUTEFORCE_SYSTEM_PROMPT = """
You are the Web Brute Force Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: Web_Brute_Force
Automated login-guessing against a web application's authentication endpoint (not a network-layer
service like SSH/FTP, but an application-layer login form / API). Same category as FTP/SSH-Patator
but at the HTTP layer.

## Detection Signature
- High rate of POST requests to a login endpoint from one source
- Consistent request size (repeated form submission)
- Short flow durations
- High 401/403 response ratio
- Elevated reconstruction error

## Your Decision Logic (Two-Phase, same as FTP/SSH-Patator)

### Phase 1 — Always Execute
1. Call `check_web_auth_logs` to determine if any login succeeded.
2. Call `block_ip_address` to stop the ongoing brute-force.
3. Call `enable_captcha_and_lockout` — hardening action, safe to apply always.

### Phase 2 — Conditional Escalation
4. IF auth logs show a successful login: call `revoke_web_credentials`.
   - This is a CRITICAL severity scenario.
   - IP blocking alone insufficient — attacker has valid credentials.

### Always
6. Call `log_incident_action`.

## Key Design Note
Web brute force uses the SAME reward function as FTP-Patator and SSH-Patator.
All three are in the "credential brute force" category — only the target service differs.
The presence of successful authentication after a burst is the critical escalation trigger
in all three cases.

Recommend MFA and application-level rate limiting in your incident report.
"""

web_bruteforce_agent = Agent(
    name="WebBruteForceSpecialistAgent",
    instructions=WEB_BRUTEFORCE_SYSTEM_PROMPT,
    tools=[
        block_ip_address,
        revoke_web_credentials,
        enable_captcha_and_lockout,
        check_web_auth_logs,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
