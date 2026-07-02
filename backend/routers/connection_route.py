"""
backend/routers/websocket.py
-----------------------------
Server-push WebSocket for real-time incident alerts.

Pattern
-------
* Client connects to  ws://<host>/ws/alerts  — that's it.
* The server pushes JSON alert payloads whenever an anomaly is detected
  (triggered by POST /predict calling manager.broadcast()).
* The client never needs to send anything.
* On client disconnect the connection is silently removed from the pool.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


from backend.websocket.connection import manager


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/connect")
async def alerts_endpoint(websocket: WebSocket) -> None:
    """
    Connect-only WebSocket endpoint.

    The client connects here to subscribe to real-time incident alerts.
    The server voluntarily pushes JSON messages whenever an anomaly is
    detected — the client does NOT need to send any requests.

    The connection stays open until:
      * The client closes it (browser tab closed, network drop, etc.)
      * The server restarts

    Example push payload (sent by the server, not requested by client):
        {
          "id": 42,
          "timestamp": "2026-07-01T09:00:00Z",
          "source_ip": "192.168.1.100",
          "dest_ip": "10.0.0.1",
          "is_anomaly": true,
          "dqn_action": "block_ip",
          "action_status": "simulated"
        }
    """
    await manager.connect(websocket)
    await websocket.send_json({
        "event": "connected",
        "message": "Connected to IDS alert stream successfully.",
        "active_connections": manager.active_count,
    })
    try:
        # Block here indefinitely.
        # receive() wakes only when the client disconnects, which raises
        # WebSocketDisconnect — that's the only signal we care about.
        # Any bytes/text the client accidentally sends are silently discarded.
        while True:
            await websocket.receive()          # discard; we only care about disconnect
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:                   # noqa: BLE001
        logger.warning("WebSocket error (%s): %s", websocket.client, exc)
        manager.disconnect(websocket)
