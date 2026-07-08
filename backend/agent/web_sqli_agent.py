"""
backend/agent/web_sqli_agent.py
─────────────────────────────────
Web SQL Injection Specialist Agent

Handles SQL injection attacks — malicious SQL injected through application inputs
to manipulate backend database queries.

Attack profile  : Web_SQL_Injection (24 samples — 0.004%)
STATISTICAL CAVEAT: With only 24 training samples, the DQN CANNOT reliably learn
this class. Default to Monitor + human escalation. Document in evaluation report.

Primary action  : Monitor and escalate (NOT autonomous disruptive action)

Tools
-----
  monitor_and_flag_sqli   — log and escalate to human for application-layer review
  flag_db_account_for_audit — mark DB accounts for credential rotation
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
def monitor_and_flag_sqli(
    source_ip: str,
    dest_ip: str,
    dst_port: int,
    confidence: float,
    recon_error: float,
) -> str:
    """
    Flag suspected SQL injection for immediate human SOC and development team review.

    With only 24 training samples this class cannot be reliably learned by the DQN.
    The agent should NOT take disruptive autonomous action based on a class it has
    essentially no statistical basis to recognise confidently.

    SQL injection requires a CODE FIX (parameterized queries) — not a network control.

    Parameters
    ----------
    source_ip    : Suspected SQLi source IP.
    dest_ip      : Web application server IP.
    dst_port     : Destination port.
    confidence   : Attack classifier confidence (REPORT THIS — likely low).
    recon_error  : Autoencoder reconstruction error.
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip=dest_ip,
        attack_type="Web_SQL_Injection",
        action_taken="monitor_and_flag_sqli — human review required",
        severity="HIGH",
        message=(
            f"[WEB_SQLI] Suspected SQLi from {source_ip} → {dest_ip}:{dst_port} | "
            f"confidence={confidence:.2%}. Flow-level detection limits apply; WAF review required."
        ),
    )
    return json.dumps({
        "action": "monitor_and_flag_sqli",
        "source_ip": source_ip,
        "dest_ip": dest_ip,
        "dst_port": dst_port,
        "classifier_confidence": confidence,
        "recon_error": recon_error,
        "alert_type": "SQL_INJECTION_SUSPECTED",
        "severity": "HIGH",
        "statistical_caveat": (
            "CRITICAL LIMITATION: Web_SQL_Injection class has only 24 training samples. "
            "Any model metric on this class is not statistically meaningful. "
            "The DQN cannot reliably recognize this attack. Treat as LOW-CONFIDENCE detection."
        ),
        "flow_level_limitation": (
            "SQL injection payload is in HTTP body/parameters — flow-level features "
            "(packet counts, byte sizes) cannot directly detect SQL payload content."
        ),
        "human_action_required": [
            "Review WAF logs for SQL metacharacters (\", ', --, ;, UNION, SELECT, DROP)",
            "Check application/database query logs for anomalous queries",
            "Audit database tables for unauthorized data reads or modifications",
            "Identify and patch the vulnerable application input/query",
        ],
        "remediation_note": (
            "Real fix is a CODE CHANGE: parameterized queries / prepared statements. "
            "WAF SQLi signatures are defense-in-depth, NOT a substitute for code fix."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    })


@function_tool
def flag_db_accounts_for_rotation(
    dest_ip: str,
    database_type: str = "unknown",
    reason: str = "Suspected SQL injection — precautionary credential rotation",
) -> str:
    """
    Flag database accounts associated with the targeted web application for
    credential rotation as a precautionary measure.

    If SQL injection succeeded, the database credentials used by the application
    may need to be rotated (especially if stored procedures or auth tables were accessed).

    Parameters
    ----------
    dest_ip       : Web/DB server IP.
    database_type : DB type hint (MySQL, PostgreSQL, MSSQL, etc.).
    reason        : Justification for flagging.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="Web_SQL_Injection",
        action_taken=f"flag_db_accounts_for_rotation ({database_type})",
        severity="HIGH",
        message=f"[WEB_SQLI] Flagged DB accounts on {dest_ip} ({database_type}) for rotation. Reason: {reason}",
    )
    return json.dumps({
        "action": "flag_db_accounts_for_rotation",
        "server": dest_ip,
        "database_type": database_type,
        "reason": reason,
        "accounts_flagged": ["app_user", "readonly_user"],
        "status": "flagged_for_human_rotation_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": (
            "STUB — In production: send alert to DBA team to rotate app DB credentials. "
            "Ensure new credentials use least-privilege (no DROP/ALTER for app user)."
        ),
    })


WEB_SQLI_SYSTEM_PROMPT = """
You are the Web SQL Injection Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: Web_SQL_Injection
SQL injection — attacker injects malicious SQL through application input fields to manipulate
backend database queries. Can result in data theft, authentication bypass, or data destruction.

## CRITICAL STATISTICAL LIMITATION
Web_SQL_Injection has only 24 training samples (0.004% of total data).
ANY precision/recall metric on this class is NOT statistically meaningful.
The DQN CANNOT reliably recognise this class.

THIS IS A DOCUMENTED DESIGN DECISION in the evaluation report:
- Default to Monitor + human escalation for this class
- Do NOT let the agent take disruptive autonomous action (Isolate Server, Block IP)
  based on a class with no statistical basis
- Document this as a known data limitation

## Your Decision Logic (VERY CONSERVATIVE)
1. ALWAYS call `monitor_and_flag_sqli` — report the detection with full context
   including the statistical caveat and flow-level limitation.
2. Call `flag_db_accounts_for_rotation` as a precautionary measure.
4. Call `log_incident_action`.
5. DO NOT call block_ip, isolate_server, or any disruptive action autonomously.

## What You MUST Say in Your Response
Explicitly state:
1. The 24-sample limitation and why model confidence is unreliable for this class.
2. That the real fix requires a CODE CHANGE (parameterized queries).
3. WAF signatures are defense-in-depth, not a complete solution.
4. Human review of application and database logs is mandatory.
5. That this class is excluded from quantitative DQN evaluation.

Alternatively, if attack is clearly confirmed by context (human-provided evidence),
escalate severity accordingly but still recommend human-led remediation.
"""

web_sqli_agent = Agent(
    name="WebSQLInjectionSpecialistAgent",
    instructions=WEB_SQLI_SYSTEM_PROMPT,
    tools=[
        monitor_and_flag_sqli,
        flag_db_accounts_for_rotation,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
