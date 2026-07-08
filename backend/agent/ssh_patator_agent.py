"""
backend/agent/ssh_patator_agent.py
────────────────────────────────────
SSH-Patator Specialist Agent

Handles brute-force credential guessing against SSH (port 22).
HIGHEST SEVERITY brute-force attack — SSH compromise = full shell access.

Attack profile  : SSH-Patator (5,949 samples — 0.99%)
Primary action  : Block IP
Escalation      : BOTH Revoke Credentials AND Isolate Server if login succeeded
                  (compound response for confirmed SSH compromise)

Tools
-----
  block_ip_address       — firewall drop for SSH brute-force source
  revoke_ssh_credentials — disable compromised SSH account / revoke keys
  isolate_compromised_host — network-isolate the server if SSH breach confirmed
  audit_authorized_keys  — check for unauthorized SSH key additions
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
def block_ip_address(source_ip: str, reason: str, duration_minutes: int = 1440) -> str:
    """
    Block SSH brute-force source IP.
    Duration default is 24h (much longer than HTTP attacks — SSH brute-forcers retry aggressively).
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="SSH-Patator",
        action_taken=f"block_ip ({duration_minutes}min)",
        severity="HIGH",
        message=f"[SSH_PATATOR] IP blocked: {source_ip} for {duration_minutes}min. Reason: {reason}",
    )
    return json.dumps({
        "action": "block_ip",
        "target_ip": source_ip,
        "protocol": "SSH (TCP 22)",
        "duration_minutes": duration_minutes,
        "reason": reason,
        "status": "applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: iptables / fail2ban / sshguard.",
    })


@function_tool
def revoke_ssh_credentials(
    username: str,
    dest_ip: str,
    also_revoke_keys: bool = True,
) -> str:
    """
    Disable SSH account and optionally revoke authorized keys for a user
    whose credentials may have been compromised during a brute-force attack.

    SSH compromise implies SHELL ACCESS — this is one of the highest-severity
    credential revocation actions in the entire system.

    Parameters
    ----------
    username         : SSH username to disable.
    dest_ip          : SSH server IP.
    also_revoke_keys : Whether to also clear ~/.ssh/authorized_keys (default True).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="SSH-Patator",
        action_taken=f"revoke_ssh_credentials (user={username})",
        severity="CRITICAL",
        message=f"[SSH_PATATOR] CRITICAL: SSH credentials revoked for '{username}' on {dest_ip}. Keys cleared: {also_revoke_keys}.",
    )
    return json.dumps({
        "action": "revoke_ssh_credentials",
        "username": username,
        "ssh_server": dest_ip,
        "account_disabled": True,
        "authorized_keys_cleared": also_revoke_keys,
        "severity": "CRITICAL",
        "timestamp": datetime.utcnow().isoformat(),
        "follow_up": [
            "Force password rotation for all system accounts",
            "Audit /var/log/auth.log for post-compromise activity",
            "Check for cron jobs or backdoors added during breach window",
            "Forensically image if pivot/lateral movement suspected",
        ],
        "note": "STUB — In production: usermod -L {username} + key revocation via PAM/AD.",
    })


@function_tool
def isolate_compromised_host(
    dest_ip: str,
    reason: str,
) -> str:
    """
    Network-isolate the SSH server if a successful login breach is confirmed.

    A compromised SSH session can pivot laterally to other systems. Isolation
    prevents the attacker from using the compromised host as a staging ground.

    COMPOUND RESPONSE: For confirmed SSH breach, both revoke_ssh_credentials
    AND this function should be called.

    Parameters
    ----------
    dest_ip : IP of the SSH server to isolate.
    reason  : Confirmation of SSH breach justification.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="SSH-Patator",
        action_taken="isolate_compromised_host",
        severity="CRITICAL",
        message=f"[SSH_PATATOR] CRITICAL: {dest_ip} isolated from network. Reason: {reason}",
    )
    return json.dumps({
        "action": "isolate_compromised_host",
        "target_ip": dest_ip,
        "network_rule": f"DENY all traffic to/from {dest_ip}",
        "reason": reason,
        "severity": "CRITICAL",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: remove from security group / VLAN reassignment.",
    })


@function_tool
def audit_authorized_keys(dest_ip: str, username: str) -> str:
    """
    Check ~/.ssh/authorized_keys and /etc/ssh/authorized_keys for unauthorized
    key additions that may have been made during a breach window.

    Parameters
    ----------
    dest_ip  : SSH server IP to audit.
    username : Account to check (or 'all' for all users).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="SSH-Patator",
        action_taken="audit_authorized_keys",
        severity="MEDIUM",
        message=f"[SSH_PATATOR] Auditing authorized_keys for user '{username}' on {dest_ip}.",
    )
    return json.dumps({
        "action": "audit_authorized_keys",
        "ssh_server": dest_ip,
        "username": username,
        "result": "STUB — Key audit not available in simulation",
        "recommendation": (
            "In production: read /home/{username}/.ssh/authorized_keys and "
            "compare against known-good baseline. Any new key is a backdoor indicator."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    })


SSH_PATATOR_SYSTEM_PROMPT = """
You are the SSH-Patator Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: SSH-Patator
SSH brute-force credential guessing against port 22. This is the HIGHEST SEVERITY
brute-force class in the system — SSH compromise grants FULL SHELL ACCESS to the server,
enabling command execution, data theft, and lateral movement to other systems.

## Detection Signature
- High connection rate to port 22 from one source
- Short flows (auth handshake only, no interactive session data)
- Repeated rapid connection attempts
- Elevated reconstruction error

## Your Decision Logic (COMPOUND RESPONSE for confirmed breach)

### Phase 1 — Always Execute
1. Call `block_ip_address` (duration=1440min / 24h for SSH attacks).
2. Call `audit_authorized_keys` to check for backdoors.

### Phase 2 — Critical Escalation (MANDATORY if breach confirmed)
3. IF any login succeeded (check dqn_action signal or attack context):
   a. Call `revoke_ssh_credentials` — IMMEDIATELY disable the account.
   b. Call `isolate_compromised_host` — prevent lateral movement.
   This COMPOUND RESPONSE (revoke + isolate) is mandatory for confirmed SSH compromise.

### Always
5. Call `log_incident_action`.

## Severity Note
Missing a successful SSH breach (false negative) has the HARSHEST PENALTY in the system.
When in doubt, escalate to CRITICAL and recommend human forensic investigation.

A compromised SSH session can:
- Add persistence (cron jobs, backdoor SSH keys)
- Exfiltrate data
- Pivot laterally to internal systems
- Deploy ransomware or botnet malware

ALWAYS recommend human review of /var/log/auth.log in your response.
"""

ssh_patator_agent = Agent(
    name="SSHPatatorSpecialistAgent",
    instructions=SSH_PATATOR_SYSTEM_PROMPT,
    tools=[
        block_ip_address,
        revoke_ssh_credentials,
        isolate_compromised_host,
        audit_authorized_keys,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
