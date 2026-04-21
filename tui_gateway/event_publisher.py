"""Best-effort WebSocket publisher transport for the PTY-side gateway.

The dashboard's `/api/pty` spawns `hermes --tui` as a child process, which
spawns its own ``tui_gateway.entry``.  Tool/reasoning/status events fire on
*that* gateway's transport — three processes removed from the dashboard
server itself.  To surface them in the dashboard sidebar (`/api/events`),
the PTY-side gateway opens a back-WS to the dashboard at startup and
mirrors every emit through this transport.

Wire protocol: newline-framed JSON dicts (the same shape the dispatcher
already passes to ``write``).  No JSON-RPC envelope here — the dashboard's
``/api/pub`` endpoint just rebroadcasts the bytes verbatim to subscribers.

Failure mode: silent.  The agent loop must never block waiting for the
sidecar to drain.  A dead WS short-circuits all subsequent writes.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

try:
    from websockets.sync.client import connect as ws_connect
except ImportError:  # pragma: no cover - websockets is a required install path
    ws_connect = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)


class WsPublisherTransport:
    __slots__ = ("_url", "_lock", "_ws", "_dead")

    def __init__(self, url: str, *, connect_timeout: float = 2.0) -> None:
        self._url = url
        self._lock = threading.Lock()
        self._ws: Optional[object] = None
        self._dead = False

        if ws_connect is None:
            self._dead = True

            return

        try:
            self._ws = ws_connect(url, open_timeout=connect_timeout, max_size=None)
        except Exception as exc:
            _log.debug("event publisher connect failed: %s", exc)
            self._dead = True
            self._ws = None

    def write(self, obj: dict) -> bool:
        if self._dead or self._ws is None:
            return False

        try:
            with self._lock:
                self._ws.send(json.dumps(obj, ensure_ascii=False))  # type: ignore[union-attr]

            return True
        except Exception as exc:
            _log.debug("event publisher write failed: %s", exc)
            self._dead = True
            self._ws = None

            return False

    def close(self) -> None:
        self._dead = True

        if self._ws is None:
            return

        try:
            self._ws.close()  # type: ignore[union-attr]
        except Exception:
            pass

        self._ws = None
