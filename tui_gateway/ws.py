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
import concurrent.futures
import json
import logging
import socket
import threading
import time
from typing import Any

from tui_gateway import server
from tui_gateway.event_replay import replay_epoch

_log = logging.getLogger(__name__)

# Max seconds a pool-dispatched handler will block waiting for the event loop
# to flush a WS frame before we mark the transport dead. Protects handler
# threads from a wedged socket.
_WS_WRITE_TIMEOUT_S = 10.0
_WS_LOG_PAYLOAD_PREVIEW = 240

# Per-token streaming frames are coalesced: buffered and flushed as a batch on
# a short timer instead of waking the event loop once per token. A model reply
# emits hundreds of these in a burst, and each one is a loop wakeup competing
# with the agent turn for the GIL — coalescing cuts that churn (CF-2). The task
# that introduced this called them "agent.token"/"agent.thinking"; in this
# codebase the per-token frames are the ``*.delta`` stream events below. Keep
# this set to genuinely high-frequency, display-only events — anything a client
# must see promptly (tool/approval/status/completion frames) is non-streaming
# and flushes the buffer ahead of itself, so ordering is preserved.
_STREAMING_EVENT_TYPES = frozenset({
    "message.delta",
    "reasoning.delta",
    "thinking.delta",
})
# Max time a streamed token waits in the buffer before flush (~30 fps). Short
# enough to stay imperceptible to the live token cadence.
_TOKEN_COALESCE_S = 0.033

# Keep starlette optional at import time; handle_ws uses the real class when
# it's available and falls back to a generic Exception sentinel otherwise.
try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:  # pragma: no cover - starlette is a required install path
    _WebSocketDisconnect = Exception  # type: ignore[assignment]


class WSTransport:
    """Per-connection WS transport.

    ``write`` is safe to call from any thread *other than* the event loop
    thread that owns the socket. Pool workers (the only real caller) run in
    their own threads, so marshalling onto the loop via
    :func:`asyncio.run_coroutine_threadsafe` + ``future.result()`` is correct
    and deadlock-free there.

    When called from the loop thread itself (e.g. by ``handle_ws`` for an
    inline response) the same call would deadlock: we'd schedule work onto
    the loop we're currently blocking. We detect that case and fire-and-
    forget instead. Callers that need to know when the bytes are on the wire
    should use :meth:`write_async` from the loop thread.
    """

    def __init__(
        self,
        ws: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        peer: str = "unknown",
        auth_identity: dict | None = None,
    ) -> None:
        self._ws = ws
        self._loop = loop
        self._peer = peer
        #: Server-verified identity carried from the WS-upgrade credential
        #: (dashboard ticket / internal credential) — stamped by
        #: ``hermes_cli.web_server._ws_auth_reason`` onto the WS object and
        #: passed through ``handle_ws``. None for transports that
        #: authenticated via the legacy token path or stdio. RPC params can
        #: never populate this: it is the only identity authority for
        #: browser-controller registration.
        self.auth_identity = auth_identity
        self._closed = False
        self._last_inbound_at = time.monotonic()
        # Token-coalescing buffer (CF-2). Streamed token frames land here and a
        # short timer flushes the batch. The lock guards the buffer + the
        # "armed" flag against the worker threads that call write(); the timer
        # handle is only ever touched on the loop thread.
        self._token_lock = threading.Lock()
        self._pending_tokens: list[str] = []
        self._token_flush_handle: asyncio.TimerHandle | None = None
        self._token_flush_armed = False
        # Buffer mutation is protected by the thread lock above; actual socket
        # writes need an async boundary because several batches can be queued on
        # the owning loop while it recovers from a stall.
        self._send_lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_inbound_at(self) -> float:
        return self._last_inbound_at

    def mark_inbound(self) -> None:
        self._last_inbound_at = time.monotonic()

    @staticmethod
    def _is_streaming_frame(obj: dict) -> bool:
        """True for high-frequency per-token frames eligible for coalescing."""
        params = obj.get("params") if isinstance(obj, dict) else None
        if not isinstance(params, dict):
            return False
        return params.get("type") in _STREAMING_EVENT_TYPES

    def write(self, obj: dict) -> bool:
        if self._closed:
            return False

        line = json.dumps(obj, ensure_ascii=False)

        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False

        # Coalesce streamed token frames: buffer this frame and arm a short
        # flush timer instead of waking the loop right now. Cheap and
        # non-blocking — the worker returns immediately. Ordering is preserved
        # because every non-streaming frame (below) drains the buffer ahead of
        # itself.
        if self._is_streaming_frame(obj):
            with self._token_lock:
                self._pending_tokens.append(line)
                if not self._token_flush_armed:
                    self._token_flush_armed = True
                    # call_soon_threadsafe arms the call_later timer on the loop
                    # thread and is safe to call from a worker or the loop.
                    self._loop.call_soon_threadsafe(self._arm_token_flush)
            return not self._closed

        # Non-streaming frame (RPC response, control frame, non-token event):
        # append it behind any buffered tokens and flush the whole batch NOW so
        # it can never overtake the tokens that preceded it. The send is
        # scheduled INSIDE the lock so the on-the-wire order matches the buffer
        # order even if the coalesce timer fires on the loop at the same moment.
        from agent.async_utils import safe_schedule_threadsafe
        with self._token_lock:
            self._pending_tokens.append(line)
            batch = self._pending_tokens
            self._pending_tokens = []
            if on_loop:
                # Fire-and-forget — don't block the loop waiting on itself.
                self._loop.create_task(self._safe_send_many(batch))
                return True
            fut = safe_schedule_threadsafe(
                self._safe_send_many(batch), self._loop
            )
            if fut is None:
                self._closed = True
                return False

        try:
            fut.result(timeout=_WS_WRITE_TIMEOUT_S)
            return not self._closed
        except concurrent.futures.TimeoutError:  # builtin TimeoutError on 3.11+
            # The event loop is stalled (GIL-heavy agent turn, delegation
            # running N children), NOT the socket dead. The send coroutine is
            # already scheduled and will flush once the loop breathes — latching
            # _closed here permanently silenced live windows after one slow
            # write (the "subagent window shows zero streaming" bug). Unblock
            # the worker thread and keep the transport alive; _safe_send_many
            # latches on a real socket error when the frame actually fails.
            _log.warning(
                "ws write slow (loop stalled >%ss) peer=%s — frame left in flight",
                _WS_WRITE_TIMEOUT_S, self._peer,
            )
            return not self._closed
        except Exception as exc:
            self._closed = True
            _log.warning(
                "ws write failed peer=%s error_type=%s error=%s",
                self._peer, type(exc).__name__, exc,
            )
            return False

    def _arm_token_flush(self) -> None:
        """Arm the coalesce timer. Runs on the loop thread (call_soon_threadsafe)."""
        if self._closed:
            return
        self._token_flush_handle = self._loop.call_later(
            _TOKEN_COALESCE_S, self._flush_tokens
        )

    def _flush_tokens(self) -> None:
        """Send buffered tokens as one batch. Runs on the loop thread (timer).

        The send is scheduled under the lock so its wire order is fixed relative
        to a concurrent non-streaming flush in :meth:`write`.
        """
        with self._token_lock:
            self._token_flush_handle = None
            self._token_flush_armed = False
            if not self._pending_tokens or self._closed:
                self._pending_tokens = []
                return
            batch = self._pending_tokens
            self._pending_tokens = []
            self._loop.create_task(self._safe_send_many(batch))

    async def write_async(self, obj: dict) -> bool:
        """Send from the owning event loop. Awaits until the frame is on the wire."""
        if self._closed:
            return False
        # Flush any buffered streamed tokens ahead of this frame (RPC response /
        # control frame) as ONE serialized batch. Sending them in two lock
        # acquisitions would let a later batch slip between the pending tokens
        # and the frame that drained them.
        with self._token_lock:
            batch = self._pending_tokens
            self._pending_tokens = []
            batch.append(json.dumps(obj, ensure_ascii=False))
        await self._safe_send_many(batch)
        return not self._closed

    async def _safe_send_many(self, lines: list[str]) -> None:
        """Send one indivisible batch of pre-serialized frames in wire order."""
        async with self._send_lock:
            if self._closed:
                return
            try:
                for line in lines:
                    if self._closed:
                        return
                    await self._ws.send_text(line)
            except Exception as exc:
                # Latch while still holding the writer lock so queued batches
                # observe the failure before they get a chance to touch the
                # socket.
                self._closed = True
                _log.warning(
                    "ws send failed peer=%s error_type=%s error=%s",
                    self._peer, type(exc).__name__, exc,
                )

    def close(self) -> None:
        self._closed = True
        # Cancel any pending coalesce flush. close() runs on the loop thread
        # (the handle_ws finally), so touching the TimerHandle here is safe.
        handle = self._token_flush_handle
        if handle is not None:
            handle.cancel()
            self._token_flush_handle = None


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
            # Dead-peer detection: without keepalive a silently-dropped client
            # (SSH tunnel reset, client sleep) leaves the TCP leg half-open
            # forever, receive_text() blocks indefinitely, and the disconnect
            # teardown (detach + orphan reap + resume replay) never runs.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, "TCP_KEEPIDLE"):  # Linux
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            elif hasattr(socket, "TCP_KEEPALIVE"):  # macOS idle seconds
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30)
    except Exception as exc:  # pragma: no cover - best-effort tuning
        _log.debug("ws TCP_NODELAY skip: %s", exc)


async def handle_ws(
    ws: Any,
    *,
    auth_identity: dict | None = None,
    subprotocol: str | None = None,
) -> None:
    """Run one WebSocket session. Wire-compatible with ``tui_gateway.entry``.

    *auth_identity* is the server-minted ``{user_id, provider}`` recorded at
    WS-upgrade authentication (``hermes_cli.web_server._ws_auth_reason``); it
    is stored on the transport as ``WSTransport.auth_identity`` and is the
    only identity authority for browser-controller registration. Existing
    callers (stdio-free harnesses, the embedded TUI child) omit it and get a
    ``None`` transport identity — unchanged behaviour.
    """
    peer = _ws_peer_label(ws)
    transport: WSTransport | None = None
    messages = 0
    parse_errors = 0
    dispatch_crashes = 0
    send_failures = 0
    disconnect_reason = "not_connected"

    try:
        if subprotocol:
            await ws.accept(subprotocol=subprotocol)
        else:
            await ws.accept()
        disconnect_reason = "connected"
        # Push small streamed frames out immediately instead of letting Nagle
        # batch them — keeps the live token cadence intact for GUI clients.
        _disable_nagle(ws)
        _log.info("ws accepted peer=%s", peer)

        transport = WSTransport(
            ws,
            asyncio.get_running_loop(),
            peer=peer,
            auth_identity=auth_identity,
        )

        # resolve_skin() reads config + initializes the skin engine —
        # synchronous I/O + CPU work that should not block the event loop
        # during the cold-start window. Run it in the thread pool so the
        # WS read loop stays free to drain the frontend's initial RPC
        # burst (setup.status, session.list, ...) without a stall
        # (#60800). The skin payload is small (a dict of strings/arrays),
        # so the to_thread overhead is negligible.
        skin_payload = await asyncio.to_thread(server.resolve_skin)
        ready_ok = await transport.write_async(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "gateway.ready",
                    # change_events: this backend broadcasts pet.changed /
                    # cron.changed / sessions.changed, so clients can demote
                    # their legacy polls to slow backstops.
                    "payload": {
                        "skin": skin_payload,
                        "change_events": True,
                        "heartbeat": True,
                        # Replay-contract process identity: lets reconnecting
                        # clients detect a backend restart and reset their
                        # per-session seq watermarks (see event_replay).
                        "replay_epoch": replay_epoch(),
                    },
                },
            }
        )
        if ready_ok:
            # Live-apply skins Hermes activates mid-conversation.
            server._ensure_skin_watcher()
            # Track this peer for session-less global broadcasts (skin.changed
            # from the background watcher) — write_json can't route those.
            server.register_live_transport(transport)
        # Cross-backend liveness (#94895): register a heartbeat row so
        # the startup orphan sweep can distinguish "row owned by a live
        # but idle backend" from "row truly orphaned". The stdio TUI's
        # entry.main() does the same; idempotent + once-per-process so a
        # stdio TUI that already started the refresher is a no-op here.
        try:
            server._start_backend_heartbeat_refresher()
        except Exception:
            _log.warning("backend heartbeat refresher start failed", exc_info=True)
        # Same once-per-process startup pass for session rows orphaned by a
        # previous gateway process (#65194): the desktop app and web dashboard
        # reach the agent through this WS sidecar, not entry.main(). Idempotent
        # + config-gated inside, so a stdio TUI that already scheduled is a
        # no-op.
        try:
            server._schedule_startup_orphan_sweep()
        except Exception:
            _log.warning("startup orphan sweep scheduling failed", exc_info=True)
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
            transport.mark_inbound()
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

            if req_method == "gateway.ping":
                ok = await transport.write_async(
                    {
                        "jsonrpc": "2.0",
                        "result": {"ok": True},
                        "id": req_id,
                    }
                )
                if not ok:
                    disconnect_reason = "send_failed_after_heartbeat"
                    send_failures += 1
                    _log.warning("ws heartbeat reply send failed peer=%s id=%s", peer, req_id)
                    break
                continue

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
            server.unregister_live_transport(transport)

            # Owner-safely park browser controllers this transport registered.
            # A reconnect with the same stable identity may deliver a terminal
            # result for work already in flight; no new dispatch is admitted
            # while the controller is offline.
            #
            # Offloaded via to_thread: disconnect acquires the controller's
            # send_lock, which a worker-thread dispatch may hold while blocking
            # on THIS loop to transmit its frame (run_coroutine_threadsafe +
            # result(timeout=10)). Acquiring it synchronously here would park
            # the whole event loop behind that 10s send bridge.
            try:
                from gateway.browser_control_broker import (
                    get_browser_control_broker,
                )

                await asyncio.to_thread(
                    get_browser_control_broker().disconnect_owner, transport
                )
            except Exception:
                _log.exception("ws browser-controller disconnect failed peer=%s", peer)

            transport.close()

            try:
                await asyncio.to_thread(server._release_wake_for_transport, transport)
            except Exception:
                _log.exception("ws wake-word teardown failed peer=%s", peer)

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
