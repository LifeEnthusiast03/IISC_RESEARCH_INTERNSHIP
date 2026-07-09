"""
backend/agent/normal_traffic_agent.py
──────────────────────────────────────
Normal Traffic Baseline Agent

A *standalone* agent that is invoked exclusively for benign (non-anomalous)
network flows.  It is intentionally NOT wired into the Manager Agent's handoff
list — it is called directly from predict_route.py whenever is_anomaly=False.

Responsibilities
────────────────
1. Acknowledge and briefly describe the benign flow.
2. Call `log_traffic_baseline` to persist a lightweight baseline record.
3. Return a short, clear confirmation that no remediation is needed.

Why a separate agent?
─────────────────────
Keeping normal-traffic analysis decoupled from the threat-response pipeline
ensures the Manager Agent stays focused on attack routing and is never confused
by benign baseline data.  It also lets us evolve the baseline agent
independently (e.g. adding drift-detection tools later).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from agents import Agent, ModelSettings, function_tool

from backend.agent.shared_tools import broadcast_alert_to_client, log_incident_action

logger = logging.getLogger(__name__)


# ── Normal-traffic tools ───────────────────────────────────────────────────────

@function_tool
def log_traffic_baseline(
    source_ip: str,
    dest_ip: str,
    src_port: int,
    dst_port: int,
    reconstruction_error: float,
    flow_label: str = "BENIGN",
) -> str:
    """
    Record a benign network flow in the baseline traffic log.

    This is called for every non-anomalous flow so that:
      • A statistical baseline of normal traffic can be maintained.
      • Future drift-detection models have labelled reference data.
      • Auditors can verify the IDS is processing all traffic, not just attacks.

    Parameters
    ----------
    source_ip            : Originating IP address.
    dest_ip              : Destination IP address.
    src_port             : Source TCP/UDP port.
    dst_port             : Destination TCP/UDP port.
    reconstruction_error : Autoencoder reconstruction error for this flow.
                           Low values confirm the flow is well within
                           the model's learned normal distribution.
    flow_label           : Classification label (always 'BENIGN' here).

    Returns JSON confirmation string.
    """
    entry = {
        "event":                "baseline_log",
        "timestamp":            datetime.utcnow().isoformat(),
        "source_ip":            source_ip,
        "dest_ip":              dest_ip,
        "src_port":             src_port,
        "dst_port":             dst_port,
        "reconstruction_error": round(reconstruction_error, 6),
        "flow_label":           flow_label,
        "status":               "recorded",
    }
    logger.info("[NORMAL TRAFFIC] Baseline flow logged: %s", json.dumps(entry))

    # Broadcast a live update to the dashboard so the terminal shows
    # real-time confirmation during agent tool execution
    broadcast_alert_to_client(
        source_ip=source_ip,
        dest_ip=dest_ip,
        attack_type="BENIGN",
        action_taken="log_baseline",
        severity="LOW",
        message=(
            f"[NORMAL TRAFFIC] Baseline recorded: {source_ip}:{src_port} → "
            f"{dest_ip}:{dst_port} | recon_error={reconstruction_error:.6f} | label={flow_label}"
        ),
    )

    return json.dumps({
        "status":  "baseline_recorded",
        "entry":   entry,
        "message": (
            f"Benign flow from {source_ip}:{src_port} → {dest_ip}:{dst_port} "
            f"recorded (recon_error={reconstruction_error:.6f})."
        ),
    })


# ── Agent definition ───────────────────────────────────────────────────────────

NORMAL_TRAFFIC_SYSTEM_PROMPT = """
You are the Normal Traffic Baseline Agent in an AI-powered Network Intrusion
Detection System (IDS).

## Your Purpose
You are invoked ONLY when the ML pipeline has determined a network flow is
BENIGN (is_anomaly = False).  Your job is NOT threat response — it is
accurate acknowledgement and baseline logging.

## What You Must Do
1. Read the incident context carefully (source IP, dest IP, ports,
   reconstruction error).
2. Confirm that the reconstruction error is below the anomaly threshold,
   meaning the autoencoder considers this flow normal.
3. Call `log_traffic_baseline` once with the flow details.
4. Optionally call `log_incident_action` to leave an audit entry.
5. Return a SHORT, plain-English summary (3–5 sentences max) stating:
   - The flow is benign and no action was taken.
   - The reconstruction error and why it indicates normal traffic.
   - That the flow has been logged for baseline purposes.

## What You Must NOT Do
- Do NOT recommend any blocking, isolation, or remediation actions.
- Do NOT raise alerts or escalate.
- Do NOT hand off to any other agent — you are standalone.
- Do NOT produce long reports — keep your response concise.

## Tone
Professional, factual, reassuring. This is confirmation of normal system
operation, not an incident report.
"""

normal_traffic_agent = Agent(
    name="NormalTrafficBaselineAgent",
    instructions=NORMAL_TRAFFIC_SYSTEM_PROMPT,
    tools=[
        log_traffic_baseline,
        log_incident_action,
    ],
    model="gpt-4o-mini",          # lightweight model — benign flows need fast, cheap responses
    model_settings=ModelSettings(tool_choice="required"),
)
