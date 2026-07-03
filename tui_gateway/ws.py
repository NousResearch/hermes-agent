"""WebSocket transport for the tui_gateway JSON-RPC server.

Reuses :func:`tui_gateway.server.dispatch` verbatim so every RPC method, every
slash command, every approval/clarify/sudo flow, and every agent event flows
through the same handlers whether the client is Ink over stdio or an iOS /
web client over WebSocket.

Wire protocol
-------------
Identical to stdio: newline-delimited JSON-RPC in both directions. The server
emits a ``gateway.ready`` event immediately after connection accept, then
echoes responses/events for inbound requests. No framing differences.

Mounting
--------
    from fastapi import WebSocket
    from tui_gateway.ws import handle_ws

    @app.websocket("/api/ws")
    async def ws(ws: WebSocket):
        await handle_ws(ws)
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

from tui_gateway import server

_log = logging.getLogger(__name__)

# Max seconds an inline WS response waits for its queued frame to reach the
# socket before the caller resumes. Worker-thread writes enqueue only; the
# single writer task owns all socket sends.
_WS_WRITE_TIMEOUT_S = 10.0
_WS_LOG_PAYLOAD_PREVIEW = 240

# Per-token streaming frames are coalesced: buffered and flushed on a fixed
# cadence instead of waking/sending once per token. A model reply emits hundreds
# of these in a burst, and each one used to create its own event-loop task.
_STREAMING_EVENT_TYPES = frozenset({
    "message.delta",
    "reasoning.delta",
    "thinking.delta",
})
# Max time a streamed token waits in the buffer before flush (~30 fps). Short
# enough to stay imperceptible to the live token cadence.
_TOKEN_COALESCE_S = 0.033

# Observation-plane backpressure bounds. The websocket sidecar is a UI/event
# stream; it must never let arbitrary worker stdout/tool/progress storms create
# unbounded send tasks or in-memory frame lists. Critical control/RPC frames are
# kept on their own bounded priority lane; overload there means the connection is
# no longer safe to use and is closed explicitly.
_WS_NONCRITICAL_MAX_FRAMES = 256
_WS_CRITICAL_MAX_FRAMES = 64
_WS_SEND_BURST = 60
_WS_SEND_INTERVAL_S = 1.0 / 30.0

_CRITICAL_EVENT_TYPES = frozenset({
    "approval.request",
    "clarify.request",
    "sudo.request",
    "secret.request",
    "terminal.read.request",
    "terminal.write.request",
    "gateway.ready",
    "message.complete",
    "error",
})

_LATEST_ONLY_EVENT_TYPES = frozenset({
    "session.info",
    "status.update",
    "notification.show",
    "notification.clear",
    "pet.info",
    "pet.info.meta",
})

_PROGRESS_EVENT_TYPES = frozenset({
    "preview.restart.progress",
    "pet.generate.progress",
    "tool.generating",
})

# Keep starlette optional at import time; handle_ws uses the real class when
# it's available and falls back to a generic Exception sentinel otherwise.
try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:  # pragma: no cover - starlette is a required install path
    _WebSocketDisconnect = Exception  # type: ignore[assignment]


@dataclass(slots=True)
class _QueuedFrame:
    line: str
    waiter: asyncio.Future | None = None


class WSTransport:
    """Per-connection WS transport with a bounded single-writer queue.

    ``write`` is safe to call from any thread. It never sends directly and never
    creates a per-frame send task; all frames flow through one writer task owned
    by this transport. Non-critical UI frames are bounded/coalesced. Critical
    JSON-RPC/control frames are preserved on a priority lane; if that lane
    overflows, the transport closes explicitly instead of silently dropping
    control events.
    """

    def __init__(
        self,
        ws: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        peer: str = "unknown",
    ) -> None:
        self._ws = ws
        self._loop = loop
        self._peer = peer
        self._closed = False
        self._lock = threading.Lock()
        self._critical: deque[_QueuedFrame] = deque()
        self._streaming: deque[_QueuedFrame] = deque()
        self._normal: deque[_QueuedFrame] = deque()
        self._snapshots: OrderedDict[tuple[Any, ...], _QueuedFrame] = OrderedDict()
        self._dropped_noncritical = 0
        self._last_backpressure_event = 0.0
        self._writer_task: asyncio.Task | None = None
        self._wake_event: asyncio.Event | None = None
        self._stream_flush_handle: asyncio.TimerHandle | None = None
        self._next_stream_flush = 0.0
        self._last_send_at = 0.0
        self._sent_in_burst = 0
        self._loop.call_soon_threadsafe(self._ensure_writer)

    @staticmethod
    def _event_type(obj: dict) -> str | None:
        if not isinstance(obj, dict) or obj.get("method") != "event":
            return None
        params = obj.get("params")
        if not isinstance(params, dict):
            return None
        event_type = params.get("type")
        return event_type if isinstance(event_type, str) else None

    @staticmethod
    def _is_streaming_frame(obj: dict) -> bool:
        """True for high-frequency per-token frames eligible for coalescing."""
        return WSTransport._event_type(obj) in _STREAMING_EVENT_TYPES

    @staticmethod
    def _snapshot_key(obj: dict, event_type: str) -> tuple[Any, ...]:
        params = obj.get("params") if isinstance(obj, dict) else None
        payload = params.get("payload") if isinstance(params, dict) else None
        sid = params.get("session_id") if isinstance(params, dict) else None
        if isinstance(payload, dict):
            if event_type == "status.update":
                return (event_type, sid, payload.get("kind"))
            if event_type.startswith("notification."):
                return (event_type, sid, payload.get("key"))
            if event_type in _PROGRESS_EVENT_TYPES or event_type.endswith(".progress"):
                return (
                    "progress",
                    event_type,
                    sid,
                    payload.get("task_id"),
                    payload.get("token"),
                    payload.get("name"),
                )
        return (event_type, sid)

    @staticmethod
    def _noncritical_size(
        streaming: deque[_QueuedFrame],
        normal: deque[_QueuedFrame],
        snapshots: OrderedDict[tuple[Any, ...], _QueuedFrame],
    ) -> int:
        return len(streaming) + len(normal) + len(snapshots)

    @staticmethod
    def _finish_waiter(frame: _QueuedFrame, ok: bool) -> None:
        waiter = frame.waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(ok)

    def _ensure_writer(self) -> None:
        if self._closed:
            return
        if self._wake_event is None:
            self._wake_event = asyncio.Event()
        if self._writer_task is None or self._writer_task.done():
            self._writer_task = self._loop.create_task(self._writer_loop())

    def _wake_writer(self) -> None:
        if self._closed:
            return
        self._ensure_writer()
        if self._wake_event is not None:
            self._wake_event.set()

    def _schedule_wake(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._wake_writer)
        except RuntimeError:
            self._closed = True

    def _arm_stream_flush_locked(self) -> None:
        if self._closed or not self._streaming:
            return
        now = time.monotonic()
        if self._next_stream_flush <= now:
            self._next_stream_flush = now + _TOKEN_COALESCE_S
        if self._stream_flush_handle is None:
            delay = max(0.0, self._next_stream_flush - now)
            self._loop.call_soon_threadsafe(self._schedule_stream_flush, delay)

    def _schedule_stream_flush(self, delay: float) -> None:
        if self._closed or self._stream_flush_handle is not None:
            return
        self._stream_flush_handle = self._loop.call_later(delay, self._stream_flush_due)

    def _stream_flush_due(self) -> None:
        self._stream_flush_handle = None
        self._wake_writer()

    def _record_drop_locked(self, reason: str) -> None:
        self._dropped_noncritical += 1
        now = time.monotonic()
        if now - self._last_backpressure_event < 1.0:
            return
        self._last_backpressure_event = now
        line = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "stream.backpressure",
                    "payload": {
                        "dropped": self._dropped_noncritical,
                        "reason": reason,
                        "max_frames": _WS_NONCRITICAL_MAX_FRAMES,
                    },
                },
            },
            ensure_ascii=False,
        )
        self._snapshots[("stream.backpressure", None)] = _QueuedFrame(line)

    def _drop_one_noncritical_locked(self) -> bool:
        frame: _QueuedFrame | None = None
        if self._streaming:
            frame = self._streaming.popleft()
        elif self._normal:
            frame = self._normal.popleft()
        elif self._snapshots:
            _key, frame = self._snapshots.popitem(last=False)
        if frame is None:
            return False
        self._finish_waiter(frame, False)
        self._record_drop_locked("noncritical_queue_full")
        return True

    def _enqueue_locked(self, obj: dict, line: str, waiter: asyncio.Future | None = None) -> bool:
        event_type = self._event_type(obj)
        frame = _QueuedFrame(line, waiter)

        # JSON-RPC replies/errors and blocking user-control prompts must never be
        # silently dropped. If even this protected lane is full, explicitly
        # degrade by closing the transport rather than pretending delivery is OK.
        if event_type in _CRITICAL_EVENT_TYPES or (event_type is None and obj.get("id") is not None):
            if len(self._critical) >= _WS_CRITICAL_MAX_FRAMES:
                self._closed = True
                self._finish_waiter(frame, False)
                _log.error(
                    "ws critical queue overflow peer=%s max=%d — closing transport",
                    self._peer,
                    _WS_CRITICAL_MAX_FRAMES,
                )
                return False
            self._critical.append(frame)
            return True

        if self._is_streaming_frame(obj):
            while self._noncritical_size(self._streaming, self._normal, self._snapshots) >= _WS_NONCRITICAL_MAX_FRAMES:
                if not self._drop_one_noncritical_locked():
                    break
            self._streaming.append(frame)
            self._arm_stream_flush_locked()
            return True

        if event_type in _LATEST_ONLY_EVENT_TYPES or event_type in _PROGRESS_EVENT_TYPES or (
            isinstance(event_type, str) and event_type.endswith(".progress")
        ):
            key = self._snapshot_key(obj, event_type)
            old = self._snapshots.pop(key, None)
            if old is not None:
                self._finish_waiter(old, False)
            while self._noncritical_size(self._streaming, self._normal, self._snapshots) >= _WS_NONCRITICAL_MAX_FRAMES:
                if not self._drop_one_noncritical_locked():
                    break
            self._snapshots[key] = frame
            return True

        while self._noncritical_size(self._streaming, self._normal, self._snapshots) >= _WS_NONCRITICAL_MAX_FRAMES:
            if not self._drop_one_noncritical_locked():
                break
        self._normal.append(frame)
        return True

    def write(self, obj: dict) -> bool:
        if self._closed:
            return False
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            if self._closed:
                return False
            ok = self._enqueue_locked(obj, line)
        if ok:
            self._schedule_wake()
        return ok and not self._closed

    async def write_async(self, obj: dict) -> bool:
        """Queue from the owning event loop and wait until the frame is sent."""
        if self._closed:
            return False
        waiter = self._loop.create_future()
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            if self._closed:
                return False
            ok = self._enqueue_locked(obj, line, waiter)
        if not ok:
            return False
        self._wake_writer()
        try:
            return bool(await asyncio.wait_for(waiter, timeout=_WS_WRITE_TIMEOUT_S)) and not self._closed
        except asyncio.TimeoutError:
            _log.warning(
                "ws async write slow (loop stalled >%ss) peer=%s — frame left queued",
                _WS_WRITE_TIMEOUT_S,
                self._peer,
            )
            return not self._closed

    def _pop_next_frame_locked(self) -> tuple[_QueuedFrame | None, float | None]:
        if self._critical:
            return self._critical.popleft(), None
        if self._snapshots:
            _key, frame = self._snapshots.popitem(last=False)
            return frame, None
        now = time.monotonic()
        if self._streaming:
            if now < self._next_stream_flush:
                return None, self._next_stream_flush - now
            return self._streaming.popleft(), None
        if self._normal:
            return self._normal.popleft(), None
        return None, None

    async def _writer_loop(self) -> None:
        assert self._wake_event is not None
        try:
            while not self._closed:
                frame: _QueuedFrame | None
                sleep_for: float | None
                with self._lock:
                    frame, sleep_for = self._pop_next_frame_locked()

                if frame is None:
                    if sleep_for is None:
                        await self._wake_event.wait()
                        self._wake_event.clear()
                    else:
                        try:
                            await asyncio.wait_for(self._wake_event.wait(), timeout=sleep_for)
                            self._wake_event.clear()
                        except asyncio.TimeoutError:
                            pass
                    continue

                if self._sent_in_burst >= _WS_SEND_BURST:
                    elapsed = time.monotonic() - self._last_send_at
                    if elapsed < _WS_SEND_INTERVAL_S:
                        await asyncio.sleep(_WS_SEND_INTERVAL_S - elapsed)
                    self._sent_in_burst = 0

                ok = await self._safe_send(frame.line)
                self._finish_waiter(frame, ok)
                self._last_send_at = time.monotonic()
                self._sent_in_burst += 1
        except asyncio.CancelledError:
            pass
        finally:
            with self._lock:
                queued = [*self._critical, *self._streaming, *self._normal, *self._snapshots.values()]
                self._critical.clear()
                self._streaming.clear()
                self._normal.clear()
                self._snapshots.clear()
            for frame in queued:
                self._finish_waiter(frame, False)

    async def _safe_send(self, line: str) -> bool:
        try:
            await self._ws.send_text(line)
            return True
        except Exception as exc:
            self._closed = True
            _log.warning(
                "ws send failed peer=%s error_type=%s error=%s",
                self._peer,
                type(exc).__name__,
                exc,
            )
            return False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            queued = [*self._streaming, *self._normal, *self._snapshots.values()]
            self._streaming.clear()
            self._normal.clear()
            self._snapshots.clear()
        for frame in queued:
            self._finish_waiter(frame, False)

        def _cancel() -> None:
            handle = self._stream_flush_handle
            if handle is not None:
                handle.cancel()
                self._stream_flush_handle = None
            task = self._writer_task
            if task is not None and not task.done():
                task.cancel()
            if self._wake_event is not None:
                self._wake_event.set()

        try:
            if asyncio.get_running_loop() is self._loop:
                _cancel()
            else:
                self._loop.call_soon_threadsafe(_cancel)
        except RuntimeError:
            self._loop.call_soon_threadsafe(_cancel)


def _ws_peer_label(ws: Any) -> str:
    """Return ``host:port`` when available, else a stable placeholder."""
    client = getattr(ws, "client", None)
    if client is None:
        return "unknown"
    host = getattr(client, "host", None) or "unknown"
    port = getattr(client, "port", None)
    return f"{host}:{port}" if port is not None else host


def _disable_nagle(ws: Any) -> None:
    """Disable Nagle so streamed JSON-RPC frames go out individually.

    Without it the kernel coalesces the small per-token frames, so a burst after
    the model's think-pause lands on the client in one tick and no client-side
    smoothing can recover the cadence. GUI/WS only; chat platforms don't hit
    this path. Best-effort — skip silently if the socket isn't reachable.
    """
    try:
        scope = getattr(ws, "scope", None) or {}
        transport = (scope.get("extensions") or {}).get("transport") or getattr(ws, "transport", None)
        sock = transport.get_extra_info("socket") if transport is not None else None
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as exc:  # pragma: no cover - best-effort tuning
        _log.debug("ws TCP_NODELAY skip: %s", exc)


async def handle_ws(ws: Any) -> None:
    """Run one WebSocket session. Wire-compatible with ``tui_gateway.entry``."""
    peer = _ws_peer_label(ws)
    transport: WSTransport | None = None
    messages = 0
    parse_errors = 0
    dispatch_crashes = 0
    send_failures = 0
    disconnect_reason = "not_connected"

    try:
        await ws.accept()
        disconnect_reason = "connected"
        # Push small streamed frames out immediately instead of letting Nagle
        # batch them — keeps the live token cadence intact for GUI clients.
        _disable_nagle(ws)
        _log.info("ws accepted peer=%s", peer)

        transport = WSTransport(ws, asyncio.get_running_loop(), peer=peer)

        # The desktop app and dashboard chat reach the agent through this WS
        # sidecar, NOT through tui_gateway.entry.main() (the stdio TUI path that
        # spawns the background MCP discovery thread). Without starting it here,
        # discovery never runs in this process: _make_agent only *waits* on the
        # thread (wait_for_mcp_discovery), which no-ops when it was never
        # created, so the agent snapshots an MCP-less tool list and the only way
        # to surface MCP tools is a manual /reload-mcp. Start it once per
        # process here (idempotent, config-gated) before gateway.ready so the
        # first agent build can pick up already-spawning servers. (#38945)
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(
            logger=_log,
            thread_name="tui-ws-mcp-discovery",
        )

        ready_ok = await transport.write_async(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "gateway.ready",
                    "payload": {"skin": server.resolve_skin()},
                },
            }
        )
        if not ready_ok:
            disconnect_reason = "ready_send_failed"
            send_failures += 1
            _log.error("ws ready frame send failed peer=%s", peer)
            return

        while True:
            try:
                raw = await ws.receive_text()
            except _WebSocketDisconnect as exc:
                disconnect_reason = (
                    "client_disconnect("
                    f"code={getattr(exc, 'code', None)},"
                    f"reason={getattr(exc, 'reason', None)})"
                )
                break
            except Exception:
                disconnect_reason = "receive_failed"
                _log.exception("ws receive failed peer=%s", peer)
                break

            line = raw.strip()
            if not line:
                continue
            messages += 1

            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors += 1
                _log.warning(
                    "ws parse error peer=%s index=%d error=%s payload=%r",
                    peer,
                    messages,
                    exc,
                    line[:_WS_LOG_PAYLOAD_PREVIEW],
                )
                ok = await transport.write_async(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": "parse error"},
                        "id": None,
                    }
                )
                if not ok:
                    disconnect_reason = "send_failed_after_parse_error"
                    send_failures += 1
                    _log.warning("ws parse-error reply send failed peer=%s", peer)
                    break
                continue

            # dispatch() may schedule long handlers on the pool; it returns
            # None in that case and the worker writes the response itself via
            # the transport we pass in (a separate thread, so transport.write
            # is the safe path there). For inline handlers it returns the
            # response dict, which we write here from the loop.
            req_id = req.get("id") if isinstance(req, dict) else None
            req_method = req.get("method") if isinstance(req, dict) else None
            try:
                resp = await asyncio.to_thread(server.dispatch, req, transport)
            except Exception:
                dispatch_crashes += 1
                _log.exception(
                    "ws dispatch crash peer=%s id=%s method=%s",
                    peer,
                    req_id,
                    req_method,
                )
                ok = await transport.write_async(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32603, "message": "internal error"},
                        "id": req_id if req_id is not None else None,
                    }
                )
                if not ok:
                    disconnect_reason = "send_failed_after_dispatch_crash"
                    send_failures += 1
                    _log.warning(
                        "ws dispatch-crash reply send failed peer=%s id=%s method=%s",
                        peer,
                        req_id,
                        req_method,
                    )
                    break
                continue
            if resp is not None and not await transport.write_async(resp):
                disconnect_reason = "send_failed_after_response"
                send_failures += 1
                _log.warning(
                    "ws response send failed peer=%s id=%s method=%s",
                    peer,
                    req_id,
                    req_method,
                )
                break
    finally:
        reaped_sessions = 0
        detached_sessions = 0
        if transport is not None:
            transport.close()

            # Reap sessions this transport owned (close_on_disconnect sidecar
            # sessions) or detach the rest to the drop sentinel so later emits
            # don't crash into a closed socket or fall through to desktop stdout
            # logs. Detached sessions are handed to the grace-windowed WS-orphan
            # reaper inside _close_sessions_for_transport (a quick reconnect /
            # session.resume cancels it). This is the single WS-disconnect
            # teardown path.
            #
            # Offloaded: _close_session_by_id does a blocking worker.close()
            # (terminate + waits) plus a synchronous DB write — inline that
            # would freeze the uvicorn event loop for every other live
            # connection.
            try:
                reaped_sessions, detached_sessions = await asyncio.to_thread(
                    server._close_sessions_for_transport,
                    transport,
                    end_reason="ws_disconnect",
                )
            except Exception:
                _log.exception("ws transport teardown failed peer=%s", peer)
        try:
            await ws.close()
        except Exception as exc:
            _log.debug("ws close failed peer=%s error=%s", peer, exc)
        _log.info(
            "ws closed peer=%s reason=%s messages=%d parse_errors=%d "
            "dispatch_crashes=%d send_failures=%d reaped_sessions=%d detached_sessions=%d",
            peer,
            disconnect_reason,
            messages,
            parse_errors,
            dispatch_crashes,
            send_failures,
            reaped_sessions,
            detached_sessions,
        )
