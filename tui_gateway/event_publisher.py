"""Best-effort WebSocket publisher transport for the PTY-side gateway.

The dashboard's `/api/pty` spawns `hermes --tui`, which spawns ``tui_gateway.entry`` — three
processes from the dashboard server. To surface events in the sidebar (`/api/events`), that gateway
opens a back-WS to the dashboard at startup and mirrors every emit through this transport as
newline-framed JSON (no JSON-RPC envelope; ``/api/pub`` rebroadcasts bytes verbatim). Failure mode:
silent — the agent loop must never block on the sidecar: ``send`` runs on a daemon thread, ``write``
returns after enqueueing (drop when full), a dead WS short-circuits all later writes.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
from typing import Optional

try:
    from websockets.sync.client import connect as ws_connect
except ImportError:  # pragma: no cover - websockets is a required install path
    ws_connect = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)

_DRAIN_STOP = object()
_QUEUE_MAX = 256


class WsPublisherTransport:
    __slots__ = ("_url", "_lock", "_ws", "_dead", "_q", "_worker")

    def __init__(self, url: str, *, connect_timeout: float = 2.0) -> None:
        self._url = url
        self._lock = threading.Lock()
        self._ws: Optional[object] = None
        self._dead = ws_connect is None
        self._q: queue.Queue[object] = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker: Optional[threading.Thread] = None
        if self._dead:
            return
        try:
            self._ws = ws_connect(url, open_timeout=connect_timeout, max_size=None)
        except Exception as exc:
            _log.debug("event publisher connect failed: %s", exc)
            self._dead = True
            return
        self._worker = threading.Thread(target=self._drain, name="hermes-ws-pub", daemon=True)
        self._worker.start()

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is _DRAIN_STOP:
                return
            if not isinstance(item, str) or self._ws is None:
                continue
            try:
                with self._lock:
                    if self._ws is not None:
                        self._ws.send(item)  # type: ignore[union-attr]
            except Exception as exc:
                _log.debug("event publisher write failed: %s", exc)
                self._dead = True
                self._ws = None

    def write(self, obj: dict) -> bool:
        if self._dead or self._ws is None or self._worker is None:
            return False
        try:
            self._q.put_nowait(json.dumps(obj, ensure_ascii=False))
            return True
        except queue.Full:
            return False

    def close(self) -> None:
        self._dead = True
        w = self._worker
        if w is not None and w.is_alive():
            # Best-effort: if the queue is wedged, the daemon thread dies with the process.
            with contextlib.suppress(queue.Full):
                self._q.put_nowait(_DRAIN_STOP)
            w.join(timeout=3.0)
        self._worker = None
        if self._ws is None:
            return
        with contextlib.suppress(Exception), self._lock:
            if self._ws is not None:
                self._ws.close()  # type: ignore[union-attr]
        self._ws = None
