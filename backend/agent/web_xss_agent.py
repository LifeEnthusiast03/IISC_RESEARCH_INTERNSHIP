"""
backend/agent/web_xss_agent.py
─────────────────────────────────
Web XSS Specialist Agent

Handles Cross-Site Scripting attacks — malicious script injection into web pages.
IMPORTANT: Flow-level features have LIMITED precision for this class.
Prefer Monitor + human escalation over aggressive autonomous blocking.

Attack profile  : Web_XSS (1,357 samples — 0.23%)
Primary action  : Monitor and alert (NOT autonomous block — high false-positive risk)
Supplement      : Deploy CSP headers, recommend WAF rule tuning

Tools
-----
  monitor_and_alert_xss   — log high-priority alert for human SOC review
  deploy_csp_headers       — push Content-Security-Policy headers to web server
  invalidate_user_sessions — invalidate potentially stolen session tokens
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
def monitor_and_alert_xss(
    source_ip: str,
    dest_ip: str,
    dst_port: int,
    confidence: float,
    recon_error: float,
) -> str:
    """
    Create a high-priority monitoring alert for suspected XSS activity and
    route to human SOC analyst for application-layer investigation.

    XSS is a content-based attack — the real signal lives in HTTP payload/body,
    which network flow features cannot capture reliably. Do NOT take aggressive
    autonomous action (blocking) based on flow features alone for this class.

    Parameters
    ----------
    source_ip    : Suspected XSS source IP.
    dest_ip      : Targeted web application server IP.
    dst_port     : Destination port (typically 80 or 443).
    confidence   : Attack type classifier confidence (0-1). Report this to human.
    recon_error  : Autoencoder reconstruction error. Report this to human.
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip=dest_ip,
        attack_type="Web_XSS",
        action_taken="monitor_and_alert — XSS suspected, human review required",
        severity="HIGH",
        message=(
            f"[WEB_XSS] Suspected XSS from {source_ip} → {dest_ip}:{dst_port} | "
            f"confidence={confidence:.2%} recon_err={recon_error:.4f}. "
            "Flow-level detection has limited precision — WAF/app log review required."
        ),
    )
    return json.dumps({
        "action": "monitor_and_alert_xss",
        "source_ip": source_ip,
        "dest_ip": dest_ip,
        "dst_port": dst_port,
        "classifier_confidence": confidence,
        "recon_error": recon_error,
        "alert_type": "XSS_SUSPECTED_HUMAN_REVIEW_REQUIRED",
        "severity": "HIGH",
        "limitation": (
            "IMPORTANT: Network flow features have LIMITED precision for XSS. "
            "The real attack signal (script tags, encoded payloads) lives in the "
            "HTTP payload/body, which flow-level features cannot capture reliably. "
            "Human review of WAF/application logs is required."
        ),
        "human_action_needed": [
            "Review WAF logs for script injection patterns",
            "Inspect application access logs for suspicious POST/GET parameters",
            "Check for script tag presence in application input fields",
        ],
        "timestamp": datetime.utcnow().isoformat(),
    })


@function_tool
def deploy_csp_headers(
    dest_ip: str,
    policy: str = "default-src 'self'; script-src 'self'; object-src 'none';",
) -> str:
    """
    Deploy Content-Security-Policy (CSP) headers to the web server as a
    mitigating control against XSS — restricts script execution sources
    so even if XSS succeeds, the injected script cannot load external resources.

    This is a SAFE, non-disruptive action that can be applied autonomously
    while human review of the suspected XSS is underway.

    Parameters
    ----------
    dest_ip : Web server IP to configure.
    policy  : CSP policy string (default: restrictive allow-self-only policy).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="Web_XSS",
        action_taken="deploy_csp_headers",
        severity="MEDIUM",
        message=(
            f"[WEB_XSS] CSP headers deployed on {dest_ip}. "
            f"Policy: '{policy}'. Script execution restricted to 'self' origin."
        ),
    )
    return json.dumps({
        "action": "deploy_csp_headers",
        "web_server": dest_ip,
        "csp_policy": policy,
        "effect": "Restricts script execution to 'self' origin only — limits XSS payload capabilities.",
        "status": "deployed_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: nginx add_header / Apache Header directive / WAF response rule.",
    })


@function_tool
def invalidate_user_sessions(dest_ip: str, reason: str) -> str:
    """
    Invalidate all active user sessions on the web application as a precaution
    against session cookie theft — the primary goal of XSS attacks.

    DISRUPTIVE: Will log all users out. Call this only if XSS is confirmed
    or if session theft is strongly suspected.

    Parameters
    ----------
    dest_ip : Web application server where sessions should be invalidated.
    reason  : Justification (XSS confirmed / session theft risk).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="Web_XSS",
        action_taken="invalidate_user_sessions",
        severity="HIGH",
        message=(
            f"[WEB_XSS] All user sessions invalidated on {dest_ip}. "
            f"Reason: {reason}. All users must re-authenticate."
        ),
    )
    return json.dumps({
        "action": "invalidate_user_sessions",
        "web_server": dest_ip,
        "sessions_cleared": True,
        "reason": reason,
        "impact": "All users will be logged out and must re-authenticate.",
        "status": "applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: flush session store (Redis/DB) / rotate session secret.",
    })


WEB_XSS_SYSTEM_PROMPT = """
You are the Web XSS Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: Web_XSS
Cross-Site Scripting — attacker injects malicious script into web pages to steal session
cookies or perform actions on behalf of other users. This is an APPLICATION-LAYER content
attack, not a volumetric or connection-based attack.

## CRITICAL LIMITATION: Flow Features Are Weak for XSS
XSS payloads appear in HTTP payload/body (script tags, encoded payloads). Network flow
features like packet counts, byte counts, and durations CANNOT reliably detect XSS.
The autoencoder has LIMITED precision for this class.

## Your Decision Logic (CONSERVATIVE — prefer Monitor over autonomous block)
1. ALWAYS call `monitor_and_alert_xss` — this is your primary action.
   Report the confidence, recon_error, and the limitation explicitly.
2. Call `deploy_csp_headers` — safe, non-disruptive hardening action.
   Apply regardless of confidence level.
3. ONLY call `invalidate_user_sessions` if attack is CONFIRMED via WAF logs.
   Do NOT call this autonomously based on flow features alone.
4. Do NOT call `block_ip_address` for this class autonomously — false-positive
   risk is too high. The human SOC analyst should make that call after review.
6. Call `log_incident_action`.

## Why Conservative Action?
- False alarm penalty for blocking a legitimate user IP is significant (-3).
- XSS cannot be fully neutralised from the network layer — the fix is in the application code.
- Overconfident autonomous action on this class would undermine trust in the IDS.

Explicitly state the flow-level limitation in your incident report.
Recommend WAF log review and application code audit as the real remediation path.
"""

web_xss_agent = Agent(
    name="WebXSSSpecialistAgent",
    instructions=WEB_XSS_SYSTEM_PROMPT,
    tools=[
        monitor_and_alert_xss,
        deploy_csp_headers,
        invalidate_user_sessions,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
