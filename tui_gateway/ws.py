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
import queue as _queue
import socket
from typing import Any

from tui_gateway import server

_log = logging.getLogger(__name__)

_WS_LOG_PAYLOAD_PREVIEW = 240

# Max pending outbound frames in the writer queue.  When the client is
# completely stuck (TCP buffer full, renderer frozen) the queue acts as a
# shock absorber.  4096 is large enough that it should never fill in normal
# operation; if it does, the transport is effectively dead and we drop.
_WS_QUEUE_MAXSIZE = 4096

# Seconds to wait for the writer task to flush remaining frames on disconnect.
_WRITER_DRAIN_TIMEOUT_S = 2.0

# Keep starlette optional at import time; handle_ws uses the real class when
# it's available and falls back to a generic Exception sentinel otherwise.
try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:  # pragma: no cover - starlette is a required install path
    _WebSocketDisconnect = Exception  # type: ignore[assignment]


class WSTransport:
    """Per-connection WS transport with a dedicated writer task.

    Writes from any thread are non-blocking: frames are enqueued to a
    thread-safe queue and a single async writer task drains them in order.
    This decouples the read loop (and RPC dispatch) from write backpressure:
    a slow client can stall the writer, but never blocks inbound RPC reads
    or agent worker threads that emit streaming events.

    ``write`` is safe to call from any thread — it uses ``queue.put_nowait``
    which is intrinsically thread-safe.  ``write_async`` is the event-loop
    counterpart with identical semantics; both return immediately.
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
        self._outgoing: _queue.Queue[str | None] = _queue.Queue(
            maxsize=_WS_QUEUE_MAXSIZE
        )
        self._writer: asyncio.Task[None] | None = None
        self._dropped = 0  # metrics: frames dropped due to queue overflow

    # ── lifecycle ──────────────────────────────────────────────────────

    def start_writer(self) -> None:
        """Start the dedicated writer task.  Must be called on the event loop."""
        if self._writer is None:
            self._writer = self._loop.create_task(self._writer_loop())

    async def _writer_loop(self) -> None:
        """Drain the outgoing queue and send frames one at a time.

        Only this coroutine calls ``ws.send_text``, so there are never
        concurrent sends on the same socket — a fragile path under
        Starlette/uvicorn.  If ``send_text`` fails the transport is latched
        closed and the loop exits; the read loop in ``handle_ws`` detects
        ``_closed`` on its next iteration and tears down.
        """
        while True:
            try:
                item = await self._loop.run_in_executor(None, self._outgoing.get)
            except Exception:
                break
            if item is None:  # sentinel from close()
                break
            try:
                await self._ws.send_text(item)
            except Exception as exc:
                self._closed = True
                _log.warning(
                    "ws send failed peer=%s error_type=%s error=%s",
                    self._peer,
                    type(exc).__name__,
                    exc,
                )
                break

    def close(self) -> None:
        self._closed = True
        try:
            self._outgoing.put_nowait(None)  # wake up a blocked writer
        except _queue.Full:
            pass  # queue is full; writer will drain, see _closed, and stop

    # ── write interface ────────────────────────────────────────────────

    def write(self, obj: dict) -> bool:
        """Enqueue one JSON frame.  Non-blocking; safe from any thread.

        Returns ``True`` if the frame was enqueued (or dropped due to
        backpressure, which is not a transport failure).  Returns ``False``
        ONLY when the transport is closed — the ``Transport`` protocol's
        "peer is gone" signal.
        """
        if self._closed:
            return False
        line = json.dumps(obj, ensure_ascii=False)
        try:
            self._outgoing.put_nowait(line)
        except _queue.Full:
            self._dropped += 1
            _log.warning(
                "ws queue full (maxsize=%d) peer=%s — dropping streaming frame (dropped=%d)",
                _WS_QUEUE_MAXSIZE,
                self._peer,
                self._dropped,
            )
        return True

    async def write_async(self, obj: dict) -> bool:
        """Enqueue from the event loop.  Same semantics as :meth:`write`.

        On queue overflow this returns ``False`` so ``handle_ws`` treats the
        RPC response as undeliverable and disconnects — a completely stuck
        client (4096 pending frames) is effectively dead.
        """
        if self._closed:
            return False
        line = json.dumps(obj, ensure_ascii=False)
        try:
            self._outgoing.put_nowait(line)
        except _queue.Full:
            _log.warning(
                "ws queue full (maxsize=%d) peer=%s — RPC response dropped, disconnecting",
                _WS_QUEUE_MAXSIZE,
                self._peer,
            )
            return False
        return True


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
        transport.start_writer()

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
            # If the writer task died (send_text failure) the transport is
            # closed — tear down before trying to read from a dead socket.
            if transport._closed:
                disconnect_reason = "send_failed"
                send_failures += 1
                _log.warning(
                    "ws transport closed (writer failed) peer=%s", peer
                )
                break

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

            # Let the writer flush remaining frames (bounded), then proceed
            # to session teardown.  A stuck writer is cancelled so the
            # teardown path is not delayed.
            if transport._writer is not None:
                try:
                    await asyncio.wait_for(
                        transport._writer, timeout=_WRITER_DRAIN_TIMEOUT_S
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    transport._writer.cancel()

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
