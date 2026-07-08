"""
backend/agent/dos_hulk_agent.py
────────────────────────────────
DoS Hulk Specialist Agent

Handles HTTP flood attacks (Hulk tool) — high-volume GET/POST requests with
randomized headers to exhaust web server thread pools.

Attack profile  : DoS_Hulk (297,642 samples — 49.6% of attack traffic)
Primary action  : Block IP
Secondary action: Isolate Server (only if server already unresponsive)
Avoid           : Kill Process (worsens availability)

Tools
-----
  block_ip_address        — add firewall rule to drop traffic from source IP
  scale_out_web_server    — trigger auto-scaling / restart of web service
  isolate_server          — last-resort isolation if server is unresponsive
  log_incident_action     — audit trail
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from agents import Agent, ModelSettings, function_tool

from backend.agent.shared_tools import broadcast_alert_to_client, log_incident_action

logger = logging.getLogger(__name__)


# ── Specialist tools ──────────────────────────────────────────────────────────

@function_tool
def block_ip_address(
    source_ip: str,
    reason: str,
    duration_minutes: int = 60,
) -> str:
    """
    Block all inbound traffic from the specified source IP at the perimeter
    firewall / load balancer.

    In production this would call the firewall vendor API or iptables:
        iptables -I INPUT -s {source_ip} -j DROP
        # or: cloud Security Group API call

    Parameters
    ----------
    source_ip        : The attacker's IP address to block.
    reason           : Human-readable reason for the block (for audit trail).
    duration_minutes : How long to maintain the block before auto-expiry (default 60 min).

    Returns JSON confirmation.
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="DoS_Hulk",
        action_taken=f"block_ip ({duration_minutes}min)",
        severity="HIGH",
        message=f"[DOS_HULK] IP blocked: {source_ip} for {duration_minutes}min. Reason: {reason}",
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
def scale_out_web_server(
    service_name: str,
    current_instance_count: int,
    target_instance_count: int,
) -> str:
    """
    Trigger horizontal scale-out of the affected web service to absorb
    remaining flood traffic while the IP block propagates.

    In production this would call:
        kubectl scale deployment/{service_name} --replicas={target_instance_count}
        # or: AWS Auto Scaling API / GCP MIG resize

    Parameters
    ----------
    service_name           : Name of the web service (e.g. 'nginx-web', 'api-server').
    current_instance_count : Current number of running instances.
    target_instance_count  : Desired instance count after scale-out.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=service_name,
        attack_type="DoS_Hulk",
        action_taken=f"scale_out ({current_instance_count}→{target_instance_count} instances)",
        severity="MEDIUM",
        message=f"[DOS_HULK] Scale-out triggered for '{service_name}': {current_instance_count} → {target_instance_count} instances.",
    )
    return json.dumps({
        "action": "scale_out_web_server",
        "service": service_name,
        "from_instances": current_instance_count,
        "to_instances": target_instance_count,
        "status": "scaling_initiated_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: kubectl scale / cloud auto-scaling API.",
    })


@function_tool
def isolate_server(
    dest_ip: str,
    service_name: str,
    reason: str,
) -> str:
    """
    Isolate the targeted web server from the network as a last resort when the
    server is already unresponsive and continued flooding risks cascading failure.

    WARNING: This is disruptive — the server becomes unreachable to ALL traffic.
    Only call this if the server is confirmed unresponsive.

    Parameters
    ----------
    dest_ip      : IP of the server to isolate.
    service_name : Human-readable service name for logs.
    reason       : Justification for the drastic isolation action.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="DoS_Hulk",
        action_taken="isolate_server",
        severity="CRITICAL",
        message=f"[DOS_HULK] CRITICAL: Server {dest_ip} ({service_name}) ISOLATED. Reason: {reason}",
    )
    return json.dumps({
        "action": "isolate_server",
        "target_ip": dest_ip,
        "service": service_name,
        "network_rule": f"DENY all traffic to/from {dest_ip}",
        "reason": reason,
        "status": "isolation_applied_stub",
        "severity": "CRITICAL",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: SDN controller / VLAN reassignment / security group removal.",
    })


# ── Agent definition ──────────────────────────────────────────────────────────

DOS_HULK_SYSTEM_PROMPT = """
You are the DoS Hulk Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: DoS_Hulk
DoS Hulk is an HTTP Denial-of-Service tool that floods web servers with massive volumes
of GET/POST requests using randomized, obfuscated headers to bypass simple filters.
Goal: exhaust server thread pools / connection limits.

## Detection Signature
- Very high forward packet count and forward byte count in short flow duration
- Low backward traffic (server struggling to respond)
- Many concurrent flows from few source IPs to destination port 80 or 443
- High reconstruction error from the autoencoder

## Your Decision Logic
1. ALWAYS call `block_ip_address` first — this is the primary containment action.
2. Call `scale_out_web_server` if you assess the flood may have already degraded service
   (recon_error > 0.1 or dqn_action includes "isolate").
3. ONLY call `isolate_server` as absolute last resort if the server is confirmed unresponsive.
   Do NOT isolate proactively — it causes unnecessary service disruption.
5. ALWAYS call `log_incident_action` to record your actions.
6. DO NOT call `kill_process` — killing the web server process worsens availability.

## Severity Assessment
- recon_error < 0.08: MEDIUM — block IP, monitor
- recon_error 0.08-0.15: HIGH — block IP, consider scale-out
- recon_error > 0.15: CRITICAL — block IP + scale-out, consider isolation

Always provide a clear incident summary explaining what you did and why.
"""

dos_hulk_agent = Agent(
    name="DoSHulkSpecialistAgent",
    instructions=DOS_HULK_SYSTEM_PROMPT,
    tools=[
        block_ip_address,
        scale_out_web_server,
        isolate_server,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
