"""
backend/agent/botnet_ares_agent.py
────────────────────────────────────
Botnet ARES Specialist Agent

Handles C2 beacon traffic — an infected host "phoning home" to its Command & Control
server with periodic heartbeat patterns.

Attack profile  : Botnet_ARES (5,508 samples — 0.92%)
Primary action  : Isolate Server (the infected HOST, not just the network path)
Secondary action: Block IP (the C2 destination) network-wide
Optional        : Kill Process (if endpoint integration available)

Tools
-----
  isolate_infected_host   — remove infected endpoint from network
  block_c2_destination    — block outbound traffic to the C2 server IP/domain
  kill_malicious_process  — terminate the beacon process (requires EDR integration)
  scan_for_lateral_spread — check other hosts for same C2 beacon pattern
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
def isolate_infected_host(
    source_ip: str,
    reason: str,
) -> str:
    """
    Immediately isolate the botnet-infected host from the network.

    The compromise lives ON the endpoint — blocking the outbound C2 IP
    alone leaves an infected machine on your network free to try other
    C2 channels (DNS tunneling, alternate IPs, domain generation algorithms).

    Isolation is the primary action for botnet infections.

    Parameters
    ----------
    source_ip : IP of the infected internal host to isolate.
    reason    : Evidence/reason for isolation (C2 beacon pattern).
    """
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip="N/A",
        attack_type="Botnet_ARES",
        action_taken="isolate_infected_host",
        severity="CRITICAL",
        message=f"[BOTNET_ARES] CRITICAL: Infected host {source_ip} isolated. Reason: {reason}",
    )
    return json.dumps({
        "action": "isolate_infected_host",
        "infected_host": source_ip,
        "network_rule": f"DENY all traffic to/from {source_ip}",
        "reason": reason,
        "severity": "CRITICAL",
        "timestamp": datetime.utcnow().isoformat(),
        "next_steps": [
            "Run EDR/AV scan on isolated host",
            "If severe: reimage the host",
            "Investigate for lateral movement to other hosts",
            "Identify initial infection vector",
        ],
        "note": "STUB — In production: VLAN reassignment / security group / SDN isolation.",
    })


@function_tool
def block_c2_destination(
    c2_ip_or_domain: str,
    reason: str,
) -> str:
    """
    Block network-wide outbound traffic to the identified C2 server
    (IP address or domain). Applied at the perimeter firewall / DNS sinkhole.

    This is a network-wide rule — it protects all hosts, not just the
    already-isolated infected one. Critical if other hosts may also be infected.

    Parameters
    ----------
    c2_ip_or_domain : C2 server IP address or domain name.
    reason          : Evidence (beacon pattern, threat intelligence match).
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=c2_ip_or_domain,
        attack_type="Botnet_ARES",
        action_taken="block_c2_destination",
        severity="HIGH",
        message=f"[BOTNET_ARES] Blocked C2 destination: {c2_ip_or_domain}. Reason: {reason}",
    )
    return json.dumps({
        "action": "block_c2_destination",
        "c2_target": c2_ip_or_domain,
        "rule_type": "egress_block_network_wide",
        "enforcement": "perimeter_firewall_and_dns_sinkhole",
        "reason": reason,
        "status": "applied_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "STUB — In production: iptables egress rule + DNS RPZ sinkhole entry.",
    })


@function_tool
def kill_malicious_process(
    infected_host_ip: str,
    process_name_hint: str = "unknown",
) -> str:
    """
    Terminate the malicious beacon process on the infected endpoint.

    NOTE: This requires EDR (Endpoint Detection & Response) agent integration
    on the target host. Without EDR, this action cannot be executed remotely.
    Consider this a stretch-goal / simulation for now.

    Parameters
    ----------
    infected_host_ip  : IP of the host running the malicious process.
    process_name_hint : Process name or pattern hint from threat intelligence.
    """
    broadcast_alert_to_client(
        source_ip=infected_host_ip,
        dest_ip="N/A",
        attack_type="Botnet_ARES",
        action_taken="kill_malicious_process",
        severity="HIGH",
        message=f"[BOTNET_ARES] Attempting to kill malicious process '{process_name_hint}' on {infected_host_ip}.",
    )
    return json.dumps({
        "action": "kill_malicious_process",
        "target_host": infected_host_ip,
        "process_hint": process_name_hint,
        "status": "requires_edr_integration_stub",
        "timestamp": datetime.utcnow().isoformat(),
        "note": (
            "STUB — Requires EDR agent (e.g. CrowdStrike, Defender ATP) on the host. "
            "In simulation: log the intent. In production: EDR API call to terminate process."
        ),
    })


@function_tool
def scan_for_lateral_spread(
    c2_ip_or_domain: str,
    network_subnet: str = "192.168.0.0/24",
) -> str:
    """
    Scan the internal network for other hosts communicating with the same C2
    server — identifying whether the infection has spread laterally.

    Botnet infections often indicate multiple compromised hosts on the same network.

    Parameters
    ----------
    c2_ip_or_domain : The known C2 server to search for in NetFlow/firewall logs.
    network_subnet  : Internal subnet to scan for lateral spread indicators.
    """
    broadcast_alert_to_client(
        source_ip="N/A",
        dest_ip=c2_ip_or_domain,
        attack_type="Botnet_ARES",
        action_taken="scan_for_lateral_spread",
        severity="MEDIUM",
        message=f"[BOTNET_ARES] Scanning {network_subnet} for lateral spread communicating with {c2_ip_or_domain}.",
    )
    return json.dumps({
        "action": "scan_for_lateral_spread",
        "c2_server": c2_ip_or_domain,
        "scanned_subnet": network_subnet,
        "result": "STUB — NetFlow analysis not available in simulation",
        "recommendation": (
            "In production: query SIEM for all internal hosts that communicated "
            f"with {c2_ip_or_domain} in the past 7 days. "
            "Any matching host is a potential botnet member."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    })


BOTNET_ARES_SYSTEM_PROMPT = """
You are the Botnet ARES Specialist Agent in an AI-powered Network Intrusion Detection System.

## Attack You Handle: Botnet_ARES
Traffic from a host ALREADY INFECTED with ARES botnet malware, communicating with its
Command & Control (C2) server. This is a "phone home" beacon — periodic outbound
connections at fixed intervals to receive commands or exfiltrate data.

## Detection Signature
- Periodic outbound connections at near-fixed intervals (heartbeat pattern)
- Very low inter-arrival time variance (deliberately timed beacons)
- Small set of external destination IPs/domains (C2 servers)
- Often to unusual ports or via HTTP/DNS tunneling
- Elevated reconstruction error

## Why This Is Different from DoS/Scan Attacks
This is an ENDPOINT COMPROMISE — the threat lives INSIDE YOUR NETWORK.
Blocking the C2 destination IP alone is insufficient because:
1. The infected host remains on your network
2. Botnet malware uses multiple C2 channels (domain generation, DNS tunneling, backup IPs)
3. The infected host can participate in DDoS attacks, exfiltrate data, spread laterally

## Your Decision Logic
1. Call `isolate_infected_host` (source_ip = the beacon source) — THIS IS PRIMARY.
   Isolation stops ALL C2 channels, not just the detected one.
2. Call `block_c2_destination` (dest_ip = the C2 server) — network-wide protection.
3. Call `scan_for_lateral_spread` to check if other hosts are also infected.
4. Call `kill_malicious_process` if EDR integration is available (optional/stretch goal).
6. Call `log_incident_action`.

## Reward Note
Isolation is strongly rewarded here (+10) even though it is disruptive, because botnet
presence represents standing risks: DDoS participation, data exfiltration, lateral movement.
The service disruption cost of isolation is worth it.

Always recommend in your response:
- Forensic imaging of the isolated host
- EDR/AV scan before bringing the host back online
- Investigation of the initial infection vector
"""

botnet_ares_agent = Agent(
    name="BotnetARESSpecialistAgent",
    instructions=BOTNET_ARES_SYSTEM_PROMPT,
    tools=[
        isolate_infected_host,
        block_c2_destination,
        kill_malicious_process,
        scan_for_lateral_spread,
        log_incident_action,
    ],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
