"""
backend/agent/ftp_patator_agent.py
────────────────────────────────────
FTP-Patator Specialist Agent

Handles brute-force credential guessing against FTP (port 21).

Attack profile  : FTP-Patator (9,531 samples — 1.6%)
Primary action  : Block IP (attempt volume alone)
Escalation      : Revoke Credentials IF a successful login is detected after burst

Tools
-----
  block_ip_address       — drop inbound FTP connections from attacker
  revoke_ftp_credentials — force-disable compromised FTP account
  check_auth_success     — check if any FTP login succeeded during burst
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
    Block all inbound FTP connection attempts from the attacking source IP.

    Parameters
    ----------
    source_ip        : Brute-forcing source IP to block.
    reason           : Reason (FTP brute-force detected).
    duration_minutes : Block duration (default 2h — longer than DoS since credential
                       attacks are often sustained/retried from same IP).
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="FTP-Patator",
        action_taken=f"block_ip ({duration_minutes}min)",
        severity="HIGH",
        message=f"[FTP_PATATOR] IP blocked: {source_ip} for {duration_minutes}min. FTP brute-force detected.",
    )
    return json.dumps({
        "action": "block_ip",
        "target_ip": source_ip,
        "protocol": "FTP (TCP 21)",
        "duration_minutes": duration_minutes,
        "reason": reason,
        "status": "applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: iptables / fail2ban / cloud WAF.",
    })


@function_tool
def revoke_ftp_credentials(
    username: str,
    dest_ip: str,
    reason: str,
) -> str:
    """
    Immediately disable the FTP account that was compromised or targeted.

    CRITICAL: Call this ONLY if a successful FTP login was detected after
    the brute-force burst. IP blocking alone is insufficient once the attacker
    has valid credentials — they can pivot to a different source IP.

    Parameters
    ----------
    username : FTP username to disable.
    dest_ip  : FTP server IP where the account should be revoked.
    reason   : Reason for revocation (credential compromise).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="FTP-Patator",
        action_taken=f"revoke_ftp_credentials (user={username})",
        severity="CRITICAL",
        message=f"[FTP_PATATOR] CRITICAL: FTP credentials revoked for '{username}' on {dest_ip}. Possible compromise.",
    )
    return json.dumps({
        "action": "revoke_ftp_credentials",
        "username": username,
        "ftp_server": dest_ip,
        "account_status": "disabled_stub",
        "reason": reason,
        "severity": "CRITICAL",
        "timestamp": datetime.utcnow().isoformat(),
        "follow_up": "Force password reset, review file access logs for data exfiltration.",
        "note": "STUB — In production: PAM / vsftpd user DB / Active Directory API call.",
    })


@function_tool
def check_auth_success_in_burst(
    source_ip: str,
    dest_ip: str,
    burst_window_seconds: int = 300,
) -> str:
    """
    Check FTP authentication logs to determine whether any login attempt
    succeeded during the brute-force burst window.

    This is the critical escalation trigger: if auth succeeded, credential
    revocation is mandatory regardless of whether IP blocking was applied.

    Parameters
    ----------
    source_ip             : The attacking source IP.
    dest_ip               : The FTP server IP.
    burst_window_seconds  : Time window to check for successful logins (default 5 min).
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip=dest_ip,
        attack_type="FTP-Patator",
        action_taken="check_auth_success_in_burst",
        severity="MEDIUM",
        message=f"[FTP_PATATOR] Checking auth logs: {source_ip} → {dest_ip} window={burst_window_seconds}s.",
    )
    # STUB: In production, query auth logs / SIEM
    return json.dumps({
        "action": "check_auth_success",
        "source_ip": source_ip,
        "dest_ip": dest_ip,
        "window_seconds": burst_window_seconds,
        "result": "NO_SUCCESS_DETECTED_STUB",
        "note": (
            "STUB — In production: query /var/log/vsftpd.log or SIEM for "
            "'Login successful' events from source_ip within burst_window."
        ),
        "recommendation": "If result were SUCCESS: immediately call revoke_ftp_credentials.",
        "timestamp": datetime.utcnow().isoformat(),
    })


FTP_PATATOR_SYSTEM_PROMPT = """
You are the FTP-Patator Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: FTP-Patator
FTP-Patator is an automated brute-force credential guessing tool targeting the FTP service
(port 21) — rapid-fire username/password attempts. It is a precursor to data theft or
server compromise if any login succeeds.

## Detection Signature
- Many short-lived flows to port 21 from one source
- High connection rate with repeated failed-auth patterns
- Low bytes-per-flow (just auth handshake, no data transfer)
- Elevated reconstruction error

## Your Decision Logic (Two-Phase)
### Phase 1 — Always Execute
1. Call `check_auth_success_in_burst` to determine if any login succeeded.
2. Call `block_ip_address` to stop the ongoing brute-force attempts.

### Phase 2 — Conditional Escalation (MOST IMPORTANT)
3. IF check_auth_success returns SUCCESS: IMMEDIATELY call `revoke_ftp_credentials`.
   - IP blocking alone is INSUFFICIENT when credentials are compromised.
   - The attacker has valid credentials and can use them from any IP.
   - This is a CRITICAL severity situation.
4. IF check_auth_success returns NO_SUCCESS: block is sufficient for now.

### Always
   - NO_SUCCESS → severity=HIGH
   - SUCCESS → severity=CRITICAL
6. Call `log_incident_action` for audit trail.

## Key Principle
Missing a credential compromise (false negative here) is one of the worst outcomes
for the entire IDS system. When in doubt about auth success, escalate to CRITICAL
and recommend human review of the FTP server logs.

FTP is inherently insecure — also recommend in your response that FTP be replaced
with SFTP/FTPS.
"""

ftp_patator_agent = Agent(
    name="FTPPatatorSpecialistAgent",
    instructions=FTP_PATATOR_SYSTEM_PROMPT,
    tools=[
        block_ip_address,
        revoke_ftp_credentials,
        check_auth_success_in_burst,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
