"""
backend/websocket/connection.py
-------------------------------
WebSocket connection manager for real-time incident alerts.
"""

import json
import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages the pool of active WebSocket connections.

    The server calls broadcast() to push a message to every connected client.
    Clients do not need to send anything — this is a pure server-push channel.
    """

    def __init__(self) -> None:
        self._active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self._active.append(websocket)
        logger.info(
            "WebSocket client connected: %s  (total: %d)",
            websocket.client,
            len(self._active),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket from the active pool."""
        if websocket in self._active:
            self._active.remove(websocket)
        logger.info(
            "WebSocket client disconnected: %s  (total: %d)",
            websocket.client,
            len(self._active),
        )

    async def broadcast(self, payload: dict) -> None:
        """
        Push a JSON payload to every connected client.

        Any connection that errors during send (stale / closed) is
        removed from the pool automatically.
        """
        if not self._active:
            return

        message = json.dumps(payload, default=str)
        dead: list[WebSocket] = []

        for ws in list(self._active):
            try:
                await ws.send_text(message)
            except Exception:  # noqa: BLE001
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)

    @property
    def active_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._active)


# ── Singleton shared across the app ──────────────────────────────────────────
manager = ConnectionManager()
