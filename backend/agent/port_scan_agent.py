"""
backend/agent/port_scan_agent.py
──────────────────────────────────
Port Scan Specialist Agent

Handles reconnaissance port scanning — systematic probing of destination
ports/hosts to discover open services before a real attack.

Attack profile  : Port_Scan (161,315 samples — 26.9%)
Primary action  : Block IP (clean, unambiguous case)
Secondary action: Monitor with elevated logging (for low-rate stealth scans)

Tools
-----
  block_ip_address        — firewall drop rule for the scanning source
  flag_for_recon_followup — mark source IP for elevated monitoring (post-scan)
  check_scanned_services  — enumerate which services were probed
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
def block_ip_address(
    source_ip: str,
    reason: str,
    duration_minutes: int = 60,
) -> str:
    """
    Block inbound traffic from a scanning source IP at the perimeter firewall.

    Port scans have no legitimate service dependency — blocking is low-risk
    and high-value. However, consider source reputation (internal scanners
    like vulnerability assessment tools should be allowlisted).

    Parameters
    ----------
    source_ip        : The scanning source IP to block.
    reason           : Reason for block (port scan detection).
    duration_minutes : Block duration before auto-expiry.
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="Port_Scan",
        action_taken=f"block_ip ({duration_minutes}min)",
        severity="HIGH",
        message=f"[PORT_SCAN] IP blocked: {source_ip} for {duration_minutes}min. Reason: {reason}",
    )
    return json.dumps({
        "action": "block_ip",
        "target_ip": source_ip,
        "rule": f"DROP all inbound from {source_ip}",
        "duration_minutes": duration_minutes,
        "reason": reason,
        "status": "applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: iptables / cloud firewall API call.",
    })


@function_tool
def flag_for_recon_followup(
    source_ip: str,
    ports_probed_estimate: str,
    follow_up_window_hours: int = 24,
) -> str:
    """
    Flag the scanning source IP for elevated monitoring in the follow-up window.

    Port scans are reconnaissance — the actual attack (brute force, exploit)
    often follows within minutes to hours. This tool marks the source IP in
    the threat intelligence store for heightened alerting.

    Parameters
    ----------
    source_ip              : The confirmed scanning source IP.
    ports_probed_estimate  : Description of port range observed (e.g. '1-1024', 'random wide scan').
    follow_up_window_hours : How long to maintain elevated monitoring (default 24h).
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="Port_Scan",
        action_taken="flag_for_recon_followup",
        severity="MEDIUM",
        message=f"[PORT_SCAN] {source_ip} flagged for recon follow-up. Est. ports probed: {ports_probed_estimate}. Window: {follow_up_window_hours}h.",
    )
    return json.dumps({
        "action": "flag_for_recon_followup",
        "source_ip": source_ip,
        "ports_probed": ports_probed_estimate,
        "monitoring_window_hours": follow_up_window_hours,
        "status": "flagged_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: update threat-intel DB / SIEM watchlist.",
    })


@function_tool
def check_scanned_services(
    dest_ip: str,
    dst_port: int,
) -> str:
    """
    Enumerate which services are actually exposed on the scanned destination
    and assess whether any critical services were probed.

    This helps prioritise: a scan touching port 22 (SSH) or 3389 (RDP) is
    more urgent than a scan of unused ports.

    Parameters
    ----------
    dest_ip  : The destination IP that was scanned.
    dst_port : The specific destination port that triggered this alert.
    """
    critical_ports = {22: "SSH", 23: "Telnet", 25: "SMTP", 80: "HTTP", 443: "HTTPS",
                      3306: "MySQL", 5432: "PostgreSQL", 3389: "RDP", 6379: "Redis",
                      27017: "MongoDB", 8080: "HTTP-Alt", 8443: "HTTPS-Alt"}
    service = critical_ports.get(dst_port, "Unknown/Non-standard")
    is_critical = dst_port in critical_ports

    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="Port_Scan",
        action_taken="check_scanned_services",
        severity="HIGH" if is_critical else "MEDIUM",
        message=f"[PORT_SCAN] Service check on {dest_ip}:{dst_port} ({service}). Critical: {is_critical}.",
    )
    return json.dumps({
        "action": "check_scanned_services",
        "dest_ip": dest_ip,
        "probed_port": dst_port,
        "service_name": service,
        "is_critical_service": is_critical,
        "recommendation": "Verify service is hardened and patched." if is_critical else "Low-risk port, continue monitoring.",
        "timestamp": datetime.utcnow().isoformat(),
    })


PORT_SCAN_SYSTEM_PROMPT = """
You are the Port Scan Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: Port_Scan
Port scanning is a reconnaissance technique where a single source systematically probes
many destination ports or hosts to discover open services, versions, and potential
vulnerabilities. It is usually the FIRST stage of a multi-stage intrusion.
Port scans are NOT damaging by themselves — but they PREDICT an imminent follow-up attack.

## Detection Signature
- One source IP contacting many distinct destination ports in a short window
- Abnormal SYN/RST flag ratios (many SYN with no completed handshake)
- Very small packet sizes and short flow durations
- High reconstruction error from the autoencoder

## Your Decision Logic
1. Call `check_scanned_services` first to assess which services were probed.
2. Call `block_ip_address` — scanning has no legitimate dependency, blocking is low-risk.
   EXCEPTION: if attack_confidence < 0.7, prefer `flag_for_recon_followup` + Monitor
   instead of a hard block, to avoid false-positives on internal security scanners.
3. ALWAYS call `flag_for_recon_followup` — the real threat is the follow-up attack.
5. Call `log_incident_action` for the audit trail.

## Important Context
- Internal vulnerability scanners and CI/CD security scans are legitimate and should
  NOT be blocked. If source_ip appears to be internal (RFC1918 space), flag for review
  rather than blocking automatically.
- The follow-up attack (brute force, exploit) may arrive within minutes — set a short
  follow_up_window_hours = 4 for active scan detections.

Always explain your confidence level and whether you believe this is a stealth scan
(low-rate, needs monitoring) or an aggressive scan (immediate block warranted).
"""

port_scan_agent = Agent(
    name="PortScanSpecialistAgent",
    instructions=PORT_SCAN_SYSTEM_PROMPT,
    tools=[
        block_ip_address,
        flag_for_recon_followup,
        check_scanned_services,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
