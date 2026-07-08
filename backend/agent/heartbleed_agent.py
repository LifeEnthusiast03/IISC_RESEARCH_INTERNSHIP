"""
backend/agent/heartbleed_agent.py
───────────────────────────────────
Heartbleed Specialist Agent

Handles Heartbleed OpenSSL exploit (CVE-2014-0160) — malformed heartbeat requests
leak up to 64KB of process memory, exposing private keys and session tokens.

Attack profile  : Heartbleed (12 samples — 0.002%)
STATISTICAL CAVEAT: Only 12 samples. DQN cannot be meaningfully evaluated on this class.
HYBRID APPROACH: Use hardcoded detection rule + agent reasoning (not purely DQN-learned).

Primary action  : Isolate Server (active key/memory leakage — severe despite tiny sample size)
Hard-coded rule : port 443 + response/request size ratio > threshold → isolate + alert

Tools
-----
  isolate_vulnerable_server  — network-isolate OpenSSL server immediately
  rotate_tls_certificates    — invalidate and rotate TLS certs + session tokens
  check_openssl_version      — verify OpenSSL version (is patch applied?)
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
def isolate_vulnerable_server(
    dest_ip: str,
    reason: str,
) -> str:
    """
    Immediately isolate the server running a potentially vulnerable OpenSSL version.

    Heartbleed causes active memory leakage — each malformed heartbeat request
    can leak up to 64KB of server memory including private TLS keys and session tokens.
    Isolation is the correct response despite the tiny training sample size,
    because the severity (key compromise) justifies the disruption cost.

    This action is HARDCODED as the default for Heartbleed — it does NOT rely on
    the DQN having learned a reliable policy from only 12 samples.

    Parameters
    ----------
    dest_ip : IP of the vulnerable OpenSSL server to isolate.
    reason  : Heartbleed detection evidence.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="Heartbleed",
        action_taken="isolate_vulnerable_server",
        severity="CRITICAL",
        message=f"[HEARTBLEED] CRITICAL: Vulnerable server {dest_ip} isolated. Reason: {reason}",
    )
    return json.dumps({
        "action": "isolate_vulnerable_server",
        "target_ip": dest_ip,
        "network_rule": f"DENY all traffic to/from {dest_ip}",
        "reason": reason,
        "severity": "CRITICAL",
        "cve": "CVE-2014-0160 (Heartbleed)",
        "timestamp": datetime.utcnow().isoformat(),
        "immediate_next_steps": [
            "Patch OpenSSL to non-vulnerable version (>= 1.0.1g)",
            "Rotate ALL TLS certificates — private keys may be compromised",
            "Invalidate ALL active sessions — session tokens may have been leaked",
            "Determine exposure window: how long was the vulnerable server accessible?",
        ],
        "note": "STUB — In production: SDN/VLAN/security group isolation.",
    })


@function_tool
def rotate_tls_certificates(
    dest_ip: str,
    domain_names: list[str],
) -> str:
    """
    Trigger rotation of TLS certificates and invalidation of active sessions
    for the affected server. Assumes private keys may be compromised.

    Heartbleed can expose the server's private TLS key — all certificates
    signed with that key must be revoked and replaced, even if not directly
    confirmed as stolen.

    Parameters
    ----------
    dest_ip      : Affected server IP.
    domain_names : List of domain names whose certs need rotation.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="Heartbleed",
        action_taken=f"rotate_tls_certificates ({len(domain_names)} domains)",
        severity="CRITICAL",
        message=f"[HEARTBLEED] CRITICAL: Initiating TLS cert rotation for {len(domain_names)} domains on {dest_ip}.",
    )
    return json.dumps({
        "action": "rotate_tls_certificates",
        "server": dest_ip,
        "domains": domain_names,
        "reason": "Heartbleed: private key may be compromised",
        "steps": [
            "1. Generate new private key pair",
            "2. Submit new CSR to CA",
            "3. Install new certificate",
            "4. Revoke old certificate (CRL / OCSP)",
            "5. Rotate session secrets / invalidate all active sessions",
        ],
        "severity": "CRITICAL",
        "status": "rotation_initiated_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: ACME/certbot automation or manual CA workflow.",
    })


@function_tool
def check_openssl_version(dest_ip: str) -> str:
    """
    Check the OpenSSL version on the targeted server to confirm whether
    it is running a vulnerable version (1.0.1 through 1.0.1f).

    Heartbleed was patched in OpenSSL 1.0.1g (April 7, 2014).
    Any version before this is vulnerable. This check helps confirm whether
    the detection is a true positive or a false alarm.

    Parameters
    ----------
    dest_ip : Server IP to check OpenSSL version on.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="Heartbleed",
        action_taken="check_openssl_version",
        severity="MEDIUM",
        message=f"[HEARTBLEED] Checking OpenSSL version on {dest_ip}.",
    )
    return json.dumps({
        "action": "check_openssl_version",
        "server": dest_ip,
        "result": "STUB — OpenSSL version check not available in simulation",
        "vulnerable_versions": "OpenSSL 1.0.1 through 1.0.1f (inclusive)",
        "patched_versions": "OpenSSL >= 1.0.1g, or any 1.0.2+ / 1.1.x+ version",
        "recommendation": (
            "In production: run 'openssl version' on target host via EDR/SSH. "
            "If vulnerable: patch IMMEDIATELY, then rotate certs and sessions."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    })


HEARTBLEED_SYSTEM_PROMPT = """
You are the Heartbleed Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: Heartbleed (CVE-2014-0160)
Exploits the OpenSSL Heartbeat extension vulnerability — a malformed heartbeat request
tricks a vulnerable OpenSSL version into leaking up to 64KB of server process memory,
potentially exposing private TLS keys, session tokens, and credentials.

## CRITICAL STATISTICAL CAVEAT
Heartbleed has only 12 training samples. The DQN CANNOT be meaningfully evaluated
on this class. This agent uses a HYBRID APPROACH:
- HARDCODED rule: port 443 + inverted response/request size ratio → ISOLATE + ALERT
- Agent reasoning for contextual enrichment and response guidance

## Detection Signature
- Small request to port 443 (TLS)
- Unusually LARGE response relative to the request size (inverted ratio)
- This size inversion is the distinctive Heartbleed fingerprint

## Your Decision Logic (HARDCODED — Does not depend on DQN confidence)
1. Call `check_openssl_version` to confirm vulnerability.
2. Call `isolate_vulnerable_server` — Heartbleed severity justifies isolation
   even with 12 training samples. DO NOT wait for high DQN confidence here.
   This action is HARDCODED as the default for this attack type.
3. Call `rotate_tls_certificates` — assume private keys may be compromised.
5. Call `log_incident_action`.

## Why Hardcoded?
For extremely rare, high-severity classes like Heartbleed:
- A hybrid approach is preferred over purely DQN-learned policy.
- The hardcoded rule (inverted size ratio on port 443) doesn't depend on
  having sufficient training samples.
- The consequence of missing Heartbleed (key compromise, session token theft)
  far outweighs the disruption cost of isolation.

Always explicitly state in your response:
1. The CVE number and that this vulnerability has been patched since 2014.
2. That real-world prevention is simply staying current on OpenSSL patching.
3. The 12-sample limitation and why this uses a hardcoded rather than learned policy.
"""

heartbleed_agent = Agent(
    name="HeartbleedSpecialistAgent",
    instructions=HEARTBLEED_SYSTEM_PROMPT,
    tools=[
        isolate_vulnerable_server,
        rotate_tls_certificates,
        check_openssl_version,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
