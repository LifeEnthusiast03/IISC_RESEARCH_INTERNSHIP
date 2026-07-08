"""
backend/agent/ddos_loit_agent.py
──────────────────────────────────
DDoS LOIT Specialist Agent

Handles Distributed Denial-of-Service attacks using LOIC/HOIC-style tools —
many distributed sources flood a target simultaneously.

Attack profile  : DDoS_LOIT (95,729 samples — 15.9%)
Primary action  : Isolate Server (per-IP blocking won't scale to thousands of sources)
Secondary action: Block IP for top-N highest-volume sources as stopgap
Escalation      : ALWAYS emit human escalation alert — DDoS needs upstream provider

Tools
-----
  isolate_server_from_flood  — network-level service isolation / failover
  block_top_sources          — rate-limit / block highest-volume attacking IPs
  escalate_to_upstream_isp   — notify ISP/CDN scrubbing service (stub)
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
def isolate_server_from_flood(
    dest_ip: str,
    service_name: str,
    flood_source_count_estimate: int,
) -> str:
    """
    Isolate the targeted server/service from the DDoS flood by redirecting
    traffic to a scrubbing center or enabling null-routing.

    Unlike single-source DoS, DDoS involves thousands of sources — per-IP
    blocking is insufficient. Service isolation + upstream mitigation is the
    only effective autonomous response.

    Parameters
    ----------
    dest_ip                   : Target server IP being flooded.
    service_name              : Name of the service under attack.
    flood_source_count_estimate: Estimated number of distinct attacking source IPs.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="DDoS_LOIT",
        action_taken="isolate_server_from_flood",
        severity="CRITICAL",
        message=f"[DDOS_LOIT] CRITICAL: {dest_ip} ({service_name}) isolated from DDoS flood. Est. sources: {flood_source_count_estimate}.",
    )
    return json.dumps({
        "action": "isolate_server_from_flood",
        "target_ip": dest_ip,
        "service": service_name,
        "flood_source_estimate": flood_source_count_estimate,
        "mitigation": "null_route_and_scrubbing_center_redirect",
        "status": "isolation_initiated_stub",
        "severity": "CRITICAL",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: BGP blackhole (RTBH) or CDN scrubbing redirect.",
    })


@function_tool
def block_top_attack_sources(
    source_ips: list[str],
    reason: str,
) -> str:
    """
    Apply firewall drop rules to the top-N highest-volume attacking source IPs
    as a stopgap measure while upstream DDoS mitigation activates.

    This won't fully neutralise a true DDoS (too many sources) but reduces
    load enough to keep the service partially functional.

    Parameters
    ----------
    source_ips : List of source IPs to block (top-N by traffic volume).
    reason     : Reason string for the firewall rule audit log.
    """
    broadcast_alert_to_client(
        source_ip=str(source_ips[:3]),
        dest_ip="N/A",
        attack_type="DDoS_LOIT",
        action_taken=f"block_top_sources ({len(source_ips)} IPs)",
        severity="CRITICAL",
        message=f"[DDOS_LOIT] {len(source_ips)} attack source IPs blocked. Top sources: {source_ips[:5]}.",
    )
    return json.dumps({
        "action": "block_top_attack_sources",
        "blocked_ips": source_ips,
        "count": len(source_ips),
        "reason": reason,
        "status": "rules_applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: batch iptables rules or cloud WAF IP block list.",
    })


@function_tool
def escalate_to_upstream_isp(
    dest_ip: str,
    attack_bandwidth_estimate_gbps: float,
    isp_contact_email: str = "noc@upstream-isp.example.com",
) -> str:
    """
    Notify the upstream ISP or cloud DDoS mitigation provider to activate
    scrubbing / BGP blackholing for the targeted IP prefix.

    True DDoS mitigation requires upstream network changes outside the
    agent's direct control — this tool triggers that escalation.

    Parameters
    ----------
    dest_ip                        : The targeted IP being flooded.
    attack_bandwidth_estimate_gbps : Rough estimate of attack volume in Gbps.
    isp_contact_email              : NOC contact for the upstream provider.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=dest_ip,
        attack_type="DDoS_LOIT",
        action_taken="escalate_to_upstream_isp",
        severity="CRITICAL",
        message=f"[DDOS_LOIT] ISP escalation triggered for {dest_ip}. Est. bandwidth: ~{attack_bandwidth_estimate_gbps:.1f}Gbps. Contact: {isp_contact_email}.",
    )
    return json.dumps({
        "action": "escalate_to_upstream_isp",
        "target_ip": dest_ip,
        "estimated_bandwidth_gbps": attack_bandwidth_estimate_gbps,
        "isp_contact": isp_contact_email,
        "escalation_type": "DDoS_scrubbing_request",
        "status": "escalation_sent_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: automated API call to Cloudflare/AWS Shield/Akamai NOC.",
    })


DDOS_LOIT_SYSTEM_PROMPT = """
You are the DDoS LOIT Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: DDoS_LOIT
DDoS using LOIC/HOIC-style tools — MANY distributed sources flood a single target
simultaneously with TCP/UDP/HTTP requests. Unlike DoS Hulk (single source), this is
a COORDINATED attack across a botnet. Per-IP blocking cannot keep pace.

## Detection Signature
- Many source IPs, each with moderate individual traffic
- Destination sees aggregate flood; consistent fingerprint across sources (same tool)
- Shared destination port across all attacking flows
- High reconstruction error

## Your Decision Logic — CRITICAL: This is different from single-source DoS
1. ALWAYS call `isolate_server_from_flood` — per-IP blocking cannot neutralise DDoS.
   Upstream scrubbing or BGP blackholing is the only scalable response.
2. Call `block_top_attack_sources` with [source_ip] as a stopgap (partial credit,
   reduces load while isolation activates).
3. ALWAYS call `escalate_to_upstream_isp` — DDoS REQUIRES upstream provider involvement.
   This is the clearest case for human escalation in the entire IDS system.
5. Call `log_incident_action` for audit.

## Why You Cannot Fully Stop This Autonomously
A single agent at one enforcement point cannot block thousands of sources fast enough.
Your goal is damage limitation + triggering the humans and upstream providers who CAN
stop it. Be explicit about this limitation in your response.

Always communicate clearly: what you did autonomously, what you escalated, and
what the human SOC team needs to do next (contact ISP, update status page, etc.).
"""

ddos_loit_agent = Agent(
    name="DDoSLOITSpecialistAgent",
    instructions=DDOS_LOIT_SYSTEM_PROMPT,
    tools=[
        isolate_server_from_flood,
        block_top_attack_sources,
        escalate_to_upstream_isp,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
