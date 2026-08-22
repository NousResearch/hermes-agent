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
Actual ``send`` calls run on a daemon thread so the TeeTransport's
``write`` returns after enqueueing (best-effort; drop when the queue is full).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Optional

try:
    from websockets.sync.client import connect as ws_connect
except ImportError:  # pragma: no cover - websockets is a required install path
    ws_connect = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)

_DRAIN_STOP = object()

_QUEUE_MAX = 256
_CONNECT_RETRY_DELAYS_S = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)


class WsPublisherTransport:
    __slots__ = ("_url", "_lock", "_ws", "_dead", "_q", "_worker", "_connect_timeout")

    def __init__(self, url: str, *, connect_timeout: float = 2.0) -> None:
        self._url = url
        self._lock = threading.Lock()
        self._ws: Optional[object] = None
        self._dead = False
        self._q: queue.Queue[object] = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker: Optional[threading.Thread] = None
        self._connect_timeout = connect_timeout

        if ws_connect is None:
            self._dead = True

            return

        self._worker = threading.Thread(
            target=self._run,
            name="hermes-ws-pub",
            daemon=True,
        )
        self._worker.start()

    def _run(self) -> None:
        if not self._connect():
            self._dead = True

            return

        self._drain()

    def _connect(self) -> bool:
        attempts = len(_CONNECT_RETRY_DELAYS_S)

        for idx, delay in enumerate(_CONNECT_RETRY_DELAYS_S, start=1):
            if self._dead:
                return False

            if delay:
                time.sleep(delay)
                if self._dead:
                    return False

            try:
                self._ws = ws_connect(
                    self._url,
                    open_timeout=self._connect_timeout,
                    max_size=None,
                )

                return True
            except Exception as exc:
                _log.debug(
                    "event publisher connect failed (attempt %d/%d): %s",
                    idx,
                    attempts,
                    exc,
                )

        return False

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is _DRAIN_STOP:
                return
            if not isinstance(item, str):
                continue
            if self._ws is None:
                continue
            try:
                with self._lock:
                    if self._ws is not None:
                        self._ws.send(item)  # type: ignore[union-attr]
            except Exception as exc:
                _log.debug("event publisher write failed: %s", exc)
                self._dead = True
                self._ws = None
                return

    def write(self, obj: dict) -> bool:
        if self._dead or self._worker is None:
            return False

        line = json.dumps(obj, ensure_ascii=False)

        try:
            self._q.put_nowait(line)

            return True
        except queue.Full:
            return False

    def close(self) -> None:
        self._dead = True
        w = self._worker
        if w is not None and w.is_alive():
            try:
                self._q.put_nowait(_DRAIN_STOP)
            except queue.Full:
                # Best-effort: if the queue is wedged, the daemon thread
                # will be torn down with the process.
                pass
            w.join(timeout=3.0)
        self._worker = None

        if self._ws is None:
            return

        try:
            with self._lock:
                if self._ws is not None:
                    self._ws.close()  # type: ignore[union-attr]
        except Exception:
            pass

        self._ws = None
