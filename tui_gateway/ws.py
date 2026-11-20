"""WebSocket transport for the tui_gateway JSON-RPC server: reuses :func:`tui_gateway.server.dispatch`
verbatim so every RPC, slash command, approval flow and agent event takes the same handlers as Ink over
stdio. Wire protocol is identical to stdio (newline-delimited JSON-RPC both ways; ``gateway.ready`` right
after accept). Mount as ``@app.websocket("/api/ws") async def ws(ws): await handle_ws(ws)``."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import concurrent.futures
import json
import logging
import socket
import threading
import time
from typing import Any, Callable

from tui_gateway import server
from agent.message_sanitization import _sanitize_surrogates
from tui_gateway.event_replay import replay_epoch
from tui_gateway.transport import serialize_frame
from tui_gateway.ws_legacy_fallback import dispatch_legacy, legacy_fallback_allowed

_log = logging.getLogger(__name__)

# Scale-to-zero: tell the (separate) gateway process a dashboard/desktop/TUI client is attached via
# the mtime of a marker file it reads in its idle predicate (gateway/scale_to_zero.py). Clients ping
# every 15s; one write per 5s per process is plenty.
_DASHBOARD_CLIENT_TOUCH_MIN_INTERVAL_S = 5.0
_dashboard_client_touched_at = 0.0
_dashboard_client_touch_lock = threading.Lock()
# The touch is file I/O and every caller sits on the event loop, where a slow disk stalls every
# connection, managed-worker RPCs included (#139011). It runs on its own single worker so it never
# ties up the pool asyncio.to_thread shares, and at most one write per path waits behind a slow one.
_dashboard_client_touch_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="dashboard-client-touch")
_dashboard_client_touch_pending: set = set()


def _touch_dashboard_client_marker(path) -> None:
    with _dashboard_client_touch_lock:
        # Cleared before the write: activity during a slow write queues exactly one follow-up.
        _dashboard_client_touch_pending.discard(path)
    from gateway.scale_to_zero import touch_dashboard_client_heartbeat
    touch_dashboard_client_heartbeat(path)  # best-effort, never raises


def _note_dashboard_client_activity(*, force: bool = False) -> None:
    """Refresh the dashboard-client liveness marker (throttled, best-effort, never blocks the caller)."""
    global _dashboard_client_touched_at
    now = time.monotonic()
    with _dashboard_client_touch_lock:
        if not force and now - _dashboard_client_touched_at < _DASHBOARD_CLIENT_TOUCH_MIN_INTERVAL_S:
            return
        _dashboard_client_touched_at = now
    path = None
    try:
        from gateway.scale_to_zero import dashboard_client_heartbeat_path
        # The process's own home, whatever profile scope the socket runs in (the watcher reads it).
        path = dashboard_client_heartbeat_path()
        with _dashboard_client_touch_lock:
            if path in _dashboard_client_touch_pending:
                return
            _dashboard_client_touch_pending.add(path)
        _dashboard_client_touch_pool.submit(_touch_dashboard_client_marker, path)
    except Exception:
        _dashboard_client_touch_pending.discard(path)  # a refused submit (shutdown) must not wedge it
        _log.debug("dashboard client heartbeat touch failed", exc_info=True)


def _sanitize_ws_text(text: str) -> str:
    """Return *text* that can be UTF-8 encoded for a WebSocket frame.

    ``json.dumps(..., ensure_ascii=False)`` happily emits lone UTF-16 surrogates; Starlette's
    ``send_text`` then raises ``UnicodeEncodeError``, which used to latch the whole connection
    closed. Same U+FFFD replacement every other Hermes transport applies.

    See #97288.
    """
    return _sanitize_surrogates(text) if text else text


# Max seconds a pool-dispatched handler blocks waiting for the loop to flush a WS frame before we
# give up waiting (the transport is NOT marked dead).
_WS_WRITE_TIMEOUT_S = 10.0
# Max seconds one send_text may await the socket once it is actually running on the loop. A healthy
# socket returns from send_text without waiting (the frame lands in the transport buffer); only kernel
# backpressure parks it, so a GIL/loop stall cannot start this clock. Deliberately 3x the worker wait
# above and under the client's 45s heartbeat deadline (apps/shared json-rpc-gateway): a peer that
# cannot drain ~48 KiB in 30s is gone, and closing here starts its reconnect instead of leaving every
# later frame and RPC reply parked behind the writer lock (#106369).
_WS_SEND_DEADLINE_S = 30.0
_WS_LOG_PAYLOAD_PREVIEW = 240
# Frames read ahead of a busy dispatcher. RPCs still dispatch one at a time in arrival order, but on a
# per-connection task so the read loop keeps answering gateway.ping while a handler blocks (a lock held
# by a long compaction, a GIL-heavy turn): the client's heartbeat deadline measures the transport, not
# handler latency (#108325). A full queue parks the reader — the old serial behaviour.
_WS_DISPATCH_QUEUE_MAX = 256
# These handlers only project already-owned state.  They may take seconds for a large stored
# transcript, so running them through the serial control lane turns an inexpensive interrupt or
# control read behind them into a client timeout.  Keep the exception deliberately narrow: mutating
# RPCs still retain arrival order, and no more than two expensive reads consume a connection.
# ``session.account_usage`` waits on a provider's usage API (a plugin hook up to its deadline).
_WS_CONCURRENT_READ_METHODS = frozenset({"session.context_breakdown", "session.transcript", "session.account_usage"})
_WS_CONCURRENT_READ_LIMIT = 2
# Reads admitted beyond the running ones; one more parks the reader (as the serial lane did).
_WS_WAITING_READ_LIMIT = 8
# A shared socket's frame naming a profile hot-serve is still starting (a chat opened on a profile
# created a moment ago) waits for that start in a lane of its own: on the serial lane every later
# control on the socket, other profiles' prompts and interrupts included, would queue behind it.
# Bounded under the Desktop's 30 s request deadline; a start still running then is refused
# retryably (``runtime_draining``).
_WS_PROFILE_START_WAIT_S = 20.0
# A viewer's answer to an open server→client request (its response frame, a batch clarify's
# ``clarify.lock``) settles in memory and never queues: the serial lane may be running the very handler
# that waits for it (a plugin slash command's ``ctx.ask_choice`` under ``slash.exec``).
_WS_ANSWER_METHODS = frozenset({"clarify.lock"})


def _answers_open_request(req: Any) -> bool:
    from tui_gateway.server_requests import is_response_frame
    return isinstance(req, dict) and (req.get("method") in _WS_ANSWER_METHODS or is_response_frame(req))

# Per-token streaming frames are coalesced: buffered and flushed as a batch on a short timer instead
# of waking the loop once per token (each wakeup competes with the agent turn for the GIL). Keep this
# set to genuinely high-frequency, display-only events — anything a client must see promptly
# (tool/approval/status/completion) is non-streaming and flushes the buffer ahead of itself, so
# ordering is preserved. _TOKEN_COALESCE_S: max buffer wait (~30 fps; imperceptible).
_STREAMING_EVENT_TYPES = frozenset({"message.delta", "reasoning.delta", "thinking.delta"})
_TOKEN_COALESCE_S = 0.033

# starlette stays optional at import time; fall back to a generic sentinel.
try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:  # pragma: no cover - starlette is a required install path
    _WebSocketDisconnect = Exception  # type: ignore[assignment]


class WSTransport:
    """Per-connection WS transport. ``write`` is safe from any thread *other than* the loop thread owning the
    socket (pool workers marshal onto the loop and block on the future); from the loop thread itself it would
    deadlock, so it detects that and fires-and-forgets. Loop-thread callers needing completion use ``write_async``."""

    def __init__(self, ws: Any, loop: asyncio.AbstractEventLoop, *, peer: str = "unknown",
                 auth_identity: dict | None = None) -> None:
        self._ws = ws
        self._loop = loop
        self._peer = peer
        #: Server-verified identity from the WS-upgrade credential, stamped by ``web_server_chat._ws_auth_reason``; None
        #: for legacy-token/stdio. RPC params can never populate it: sole identity authority for relay authorship
        #: and for the ``user_id`` the agent is built with (``server._session_auth_user_id``).
        self.auth_identity = auth_identity
        self.desktop_local = False
        #: This socket is bound to a session authority (``ws.handle_ws``): its session verbs speak the
        #: canonical wire, ``client.hello`` is answered by the authority, and ``gateway.capabilities``
        #: says so (``resumed_handshake``) instead of advertising a fold that would answer -32601.
        self.has_session_authority = False
        #: Etag of the skin this socket's ``gateway.ready`` frame carried, so the folded handshake
        #: answers the SAME identity the client compared against (``gateway/session_hello.py``).
        self.skin_etag = ""
        #: The served profile home whose skin this peer follows (None = the launch home); see change_watcher.
        self.skin_home: str | None = None
        self._closed = False
        # Token-coalescing buffer. The lock guards the buffer + "armed" flag against worker threads
        # calling write(); the timer handle is only ever touched on the loop thread.
        self._token_lock = threading.Lock()
        self._pending_tokens: list[str] = []
        self._token_flush_handle: asyncio.TimerHandle | None = None
        self._token_flush_armed = False
        # Socket writes need an async boundary: several batches can queue on the loop during a stall.
        self._send_lock = asyncio.Lock()
        self._abort_requested = False
        # One socket close per transport: the send-deadline path and abort() both reach for a 1011,
        # and under load both fire — latched on the loop thread so the peer gets a single close frame.
        self._close_scheduled = False

    def _on_loop(self) -> bool:
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    def write(self, obj: dict) -> bool:
        if self._closed:
            return False
        line = serialize_frame(obj, self._peer, _log)
        on_loop = self._on_loop()
        # Streamed token: buffer it and arm the flush timer; the worker returns immediately.
        # call_soon_threadsafe is safe from a worker or the loop.
        params = obj.get("params") if isinstance(obj, dict) else None
        if isinstance(params, dict) and params.get("type") in _STREAMING_EVENT_TYPES:
            with self._token_lock:
                self._pending_tokens.append(line)
                if not self._token_flush_armed:
                    self._token_flush_armed = True
                    self._loop.call_soon_threadsafe(self._arm_token_flush)
            return not self._closed
        # Non-streaming frame: append behind any buffered tokens and flush the whole batch NOW so it
        # can never overtake them. The send is scheduled INSIDE the lock so wire order matches buffer
        # order even if the coalesce timer fires on the loop at the same moment.
        from agent.async_utils import safe_schedule_threadsafe
        with self._token_lock:
            self._pending_tokens.append(line)
            batch, self._pending_tokens = self._pending_tokens, []
            if on_loop:
                self._loop.create_task(self._safe_send_many(batch))
                return True
            fut = safe_schedule_threadsafe(self._safe_send_many(batch), self._loop)
            if fut is None:
                self._closed = True
                return False
        try:
            fut.result(timeout=_WS_WRITE_TIMEOUT_S)
            return not self._closed
        except concurrent.futures.TimeoutError:  # builtin TimeoutError on 3.11+
            # The loop is stalled (GIL-heavy turn, delegation), NOT the socket dead: the send is already
            # scheduled and flushes once the loop breathes. Latching _closed here permanently silenced
            # live windows after one slow write; _safe_send_many latches on a real error.
            _log.warning("ws write slow (loop stalled >%ss) peer=%s — frame left in flight", _WS_WRITE_TIMEOUT_S, self._peer)
            return not self._closed
        except Exception as exc:
            self._closed = True
            _log.warning("ws write failed peer=%s error_type=%s error=%s", self._peer, type(exc).__name__, exc)
            return False

    def _arm_token_flush(self) -> None:  # loop thread
        if not self._closed:
            self._token_flush_handle = self._loop.call_later(_TOKEN_COALESCE_S, self._flush_tokens)

    def _flush_tokens(self) -> None:
        """Timer callback (loop thread): send buffered tokens as one batch, scheduled under the lock so
        wire order is fixed relative to a concurrent ``write``."""
        with self._token_lock:
            self._token_flush_handle = None
            self._token_flush_armed = False
            batch, self._pending_tokens = self._pending_tokens, []
            if batch and not self._closed:
                self._loop.create_task(self._safe_send_many(batch))

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def server_closed(self) -> bool:
        """The SERVER asked this peer's socket to close (send deadline, fanout overflow): the
        teardown line's ``initiated=server`` signal — latched here rather than inferred from a
        library close code, which the ASGI disconnect does not preserve."""
        return self._close_scheduled or self._abort_requested

    @property
    def ws(self) -> Any:
        """The WebSocket this transport writes to (what sign-out tracks a browser tab by)."""
        return self._ws

    async def write_async(self, obj: dict) -> bool:
        """Send from the owning loop; awaits until the frame is on the wire. Buffered tokens are flushed
        ahead of it in the SAME batch so nothing slips between."""
        if self._closed:
            return False
        with self._token_lock:
            batch, self._pending_tokens = self._pending_tokens, []
            batch.append(serialize_frame(obj, self._peer, _log))
        await self._safe_send_many(batch)
        return not self._closed

    async def write_many_async(self, frames: list[dict]) -> bool:
        """``write_async`` for an ordered batch: buffered tokens, then *frames*, in one send batch."""
        return await self.write_many_if_current_async(frames, None)

    async def write_many_if_current_async(self, frames: list[dict], is_current: Callable[[], bool] | None) -> bool:
        """Recheck the batch's owner at each send without gating other producers' buffered tokens."""
        if self._closed:
            return False
        with self._token_lock:
            batch, self._pending_tokens = self._pending_tokens, []
            unguarded_prefix = len(batch)
            batch.extend(serialize_frame(frame, self._peer, _log) for frame in frames)
        await self._safe_send_many(batch, is_current=is_current, unguarded_prefix=unguarded_prefix)
        return not self._closed

    async def _safe_send_many(self, lines: list[str], *, is_current: Callable[[], bool] | None = None,
                              unguarded_prefix: int = 0) -> None:
        """Send one indivisible batch of pre-serialized frames in wire order."""
        async with self._send_lock:
            if self._closed:
                return
            for index, line in enumerate(lines):
                if self._closed:
                    return
                if is_current is not None and index >= unguarded_prefix and not is_current():
                    return
                payload = _sanitize_ws_text(line)
                try:
                    await asyncio.wait_for(self._ws.send_text(payload), timeout=_WS_SEND_DEADLINE_S)
                except TimeoutError:
                    # The loop is responsive (the timer fired) but the socket never drained: unlike the
                    # loop-stall wait in write(), this is a dead peer. Latch under the writer lock so queued
                    # batches bail, and close the socket so handle_ws's read loop ends and its teardown
                    # (session detach/reap, client reconnect) runs. See #106369.
                    self._closed = True
                    _log.warning("ws send deadline exceeded (socket stalled, loop responsive) peer=%s deadline=%ss — closing",
                                 self._peer, _WS_SEND_DEADLINE_S)
                    self._loop.create_task(self._close_socket(1011, "send deadline"))
                    return
                except UnicodeEncodeError as exc:
                    # A single illegal UTF-8 frame (lone surrogate) must not tear down the socket.
                    _log.warning("ws send skipped invalid utf-8 frame peer=%s error=%s", self._peer, exc)
                    continue
                except Exception as exc:
                    # Latch while holding the writer lock so queued batches observe the failure first.
                    self._closed = True
                    _log.warning("ws send failed peer=%s error_type=%s error=%s", self._peer, type(exc).__name__, exc)
                    return

    def close(self) -> None:  # loop thread (handle_ws finally), so the TimerHandle is safe
        self._closed = True
        if self._token_flush_handle is not None:
            self._token_flush_handle.cancel()
            self._token_flush_handle = None

    def abort(self) -> None:
        """Close from any thread and drop the socket with 1011 so the client reconnects and replays
        (fanout overflow). One-shot: N mirrored sessions overflowing on this socket schedule one close."""
        self._closed = True
        with self._token_lock:
            if self._abort_requested:
                return
            self._abort_requested = True
        if self._on_loop():
            self._finish_abort()
            return
        # A loop that already shut down has nothing left to cancel or close.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._finish_abort)

    def _finish_abort(self) -> None:  # loop thread
        self.close()
        self._loop.create_task(self._close_socket(1011, "fanout overflow"))

    async def _close_socket(self, code: int, reason: str) -> None:
        """Close the peer socket so ``handle_ws``'s ``receive_text`` unblocks and its disconnect teardown
        runs. The server library bounds this (websockets ``close_timeout`` → abort). One-shot: the
        send-deadline path and ``abort()`` can both schedule this; the first to run on the loop closes,
        the rest return, so the peer never gets a second 1011."""
        if self._close_scheduled:
            return
        self._close_scheduled = True
        try:
            await self._ws.close(code=code)
        except Exception as exc:
            _log.debug("ws close after %s failed peer=%s error=%s", reason, self._peer, exc)


def _ws_peer_label(ws: Any) -> str:
    """``host:port`` when available, else a stable placeholder."""
    client = getattr(ws, "client", None)
    if client is None:
        return "unknown"
    host, port = getattr(client, "host", None) or "unknown", getattr(client, "port", None)
    return f"{host}:{port}" if port is not None else host


def _is_unknown_method(resp) -> bool:
    error = resp.get("error") if isinstance(resp, dict) else None
    return isinstance(error, dict) and error.get("code") == -32601


def _disable_nagle(ws: Any) -> None:
    """Disable Nagle + enable TCP keepalive on the raw socket (best-effort). Without TCP_NODELAY the kernel
    coalesces small per-token frames, so a burst after the model's think-pause lands in one tick and no
    client-side smoothing can recover the cadence. Without keepalive a silently-dropped client (SSH tunnel
    reset, sleep) leaves the leg half-open forever: receive_text() blocks and the disconnect teardown never runs."""
    try:
        scope = getattr(ws, "scope", None) or {}
        transport = (scope.get("extensions") or {}).get("transport") or getattr(ws, "transport", None)
        sock = transport.get_extra_info("socket") if transport is not None else None
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, "TCP_KEEPIDLE"):  # Linux
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            elif hasattr(socket, "TCP_KEEPALIVE"):  # macOS idle seconds
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30)
    except Exception as exc:  # pragma: no cover - best-effort tuning
        _log.debug("ws TCP_NODELAY skip: %s", exc)


def _authority_connection(ws: Any, transport: WSTransport, auth_identity: dict | None, operator: bool):
    """The session-authority connection for this socket (route scope, else app state), or None
    when this backend has no authority and requests go to the legacy ``server.dispatch``."""
    authority = (getattr(ws, 'scope', None) or {}).get('hermes.session_authority') or getattr(
        getattr(getattr(ws, 'app', None), 'state', None), 'session_authority', None)
    if authority is None:
        return None
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, transport, auth_identity or {}, operator=operator)


async def _dispatch_request(authority_connection: Any, req: Any, req_method: Any, transport: WSTransport) -> Any:
    """One request through the authority when attached, else the legacy dispatcher. A request from the
    local Desktop runs as work for it (the welcome trial and onboarding key on that), and so do the
    threads it hands off to: ``to_thread`` and the legacy pool copy this context."""
    from hermes_cli.process_identity import serving_local_desktop
    from tui_gateway.server_requests import resolve_pinned_response
    if authority_connection is not None and resolve_pinned_response(req, transport):
        # The answer to a request a sidecar RPC sent back to THIS socket (vault.preview.fill's page
        # scripts); the authority's own registry never holds those ids.
        return None
    with serving_local_desktop(getattr(transport, "desktop_local", False)):
        if authority_connection is None:
            return await asyncio.to_thread(server.dispatch, req, transport)
        resp = await authority_connection.dispatch(req)
        actor = authority_connection.actor
        if (_is_unknown_method(resp) and req_method in server._methods
                and legacy_fallback_allowed(actor, req_method)):
            # Session verbs live on the authority; everything else the sidecar still
            # registers (pet, wake word, connectors) keeps its legacy handler. A legacy
            # session writer keeps the authority's -32601 (ws_legacy_fallback), as does a
            # method neither side knows, which the client's capability fallbacks key on.
            resp = await asyncio.to_thread(dispatch_legacy, server, req, transport, actor)
        return resp


def _start_backend_liveness() -> None:
    """Backend heartbeat refresher + startup orphan sweep (idempotent, once per process); a
    failure of either is logged and never blocks the connection."""
    for start, what in (
        (server._start_backend_heartbeat_refresher, "backend heartbeat refresher start"),
        (server._schedule_startup_orphan_sweep, "startup orphan sweep scheduling"),
        (server._start_cache_expiry_compactor, "cache-expiry compactor start"),
    ):
        try:
            start()
        # health: allow BLE001 -- liveness boundary; logs the traceback (exc_info) via _log, a name ruff does not treat as a logger
        except Exception:
            _log.warning("%s failed", what, exc_info=True)


#: Close codes a CLIENT's own Close frame carries as a plain goodbye; uvicorn passes the peer's
#: code and reason through verbatim (``websockets_sansio_impl.py`` handle_parser_exception path).
_WS_PEER_FAREWELL_CLOSE_CODES = frozenset({1000, 1001})
#: 1005 after the handshake: the transport died with NO Close frame. uvicorn's sansio protocol
#: reports exactly this for any TCP loss, so a client that vanished and a server keepalive ping
#: that timed out are the SAME report (``connection_lost`` queues 1005). The line therefore names
#: both outcomes and leaves the split to ``silent_s`` against the surface's configured
#: ``ws_ping_interval + ws_ping_timeout`` budget — doubled to ``interval + 2*timeout`` where the
#: two-strike policy fits the tunnel idle cap (``hermes_cli/web_server_keepalive.py``, C2a).
#: The brief's "a keepalive failure is coded 1012"
#: (desktop-reconnect-under-1s.md, flagged unverified there) is FALSE for this stack: 1012 is
#: uvicorn's graceful-shutdown notice and the sansio keepalive kill lands here, as 1005 — which
#: is arithmetically consistent with the measured 40 s churn grid, because production's dashboard
#: binds 0.0.0.0 (non-loopback → pings ON at the 20/20 default), unlike the loopback test box.
_WS_UNATTRIBUTED_LOSS_CLOSE_CODE = 1005
#: 1012: uvicorn's graceful-shutdown notice (``websockets_sansio_impl.py`` shutdown) — a deploy.
_WS_SHUTDOWN_CLOSE_CODE = 1012


def _ws_attach_ready_watchers(transport: WSTransport, authority_connection: Any) -> None:
    """After a delivered gateway.ready: live-apply skins Hermes activates mid-conversation,
    watch cross-process lease moves (display.lease), and track this peer for session-less global
    broadcasts write_json can't route. A served profile's TUI repaints from ITS home's skin,
    never the launch home's."""
    server._ensure_skin_watcher()
    server._ensure_lease_watcher()
    if authority_connection is not None:
        transport.skin_home = server._skin_home_for(authority_connection.actor.profile_id)
    server.register_live_transport(transport)


def _ws_ready_frame(skin_payload: Any, *, has_session_authority: bool, skin_etag: str = "") -> dict:
    """The first frame after accept: the skin plus the flags the reconnect contract gates on.

    change_events: this backend broadcasts pet/cron/sessions.changed, so clients can demote
    legacy polls to backstops. replay_epoch lets reconnecting clients detect a backend restart
    and reset their per-session seq watermarks (event_replay). skin_etag is the content identity of
    ``skin``, so a reconnect holding this palette already knows whether it must fetch a body at all.
    resumed_handshake is TRUE only when this socket has a session authority to answer
    ``client.hello``: the client then folds its first frame (capabilities + every session's replay +
    the open requests) into ONE application round trip instead of a capabilities call plus one
    ``session.events.since`` per session.
    """
    return {
        "jsonrpc": "2.0", "method": "event",
        "params": {"type": "gateway.ready", "payload": {
            "skin": skin_payload, "skin_etag": skin_etag, "change_events": True, "heartbeat": True,
            "replay_epoch": replay_epoch(), "resumed_handshake": has_session_authority,
            **({"session_authority": True} if has_session_authority else {}),
        }},
    }


def _ws_close_fields(accepted_at, last_inbound_at, detected_at, reason: str, code: Any,
                     *, shutting_down: bool, server_closed: bool) -> tuple[float, float, str]:
    """(life_s, silent_s, initiated) for the ``ws closed`` line — C1 (t_c1f387b2).

    ``silent_s`` is the time the socket sat orphaned-silent: last inbound frame to the instant
    the close was DETECTED (a half-open socket's silence is exactly the server-visible detection
    lag). ``initiated`` names the closer, and claims only what the stack can actually show
    (each label driven through real uvicorn + real ``handle_ws`` in
    ``tests/tui_gateway/test_ws_close_initiated.py``):

    ``shutdown`` 1012 — uvicorn's graceful-shutdown notice, so every deploy or restart — or a
    cancelled handler task. ``server`` this server ended it: a failed reply/ready write, a
    read-lane failure, or its own socket close (send deadline, fanout overflow; latched by the
    transport, so no close code has to be trusted). ``peer`` a client Close frame saying goodbye
    (1000/1001). ``peer_or_keepalive`` 1005 — the connection died with no Close frame, which on
    the production transport is a vanished client and a failed keepalive ping alike; C2a splits
    that bucket with ``silent_s`` against the surface's ping budget. ``unknown`` anything else:
    no causal claim is manufactured from missing data.
    """
    life_s = (detected_at - accepted_at) if accepted_at is not None else 0.0
    silent_s = (detected_at - last_inbound_at) if last_inbound_at is not None else 0.0
    if shutting_down or code == _WS_SHUTDOWN_CLOSE_CODE:
        return life_s, silent_s, "shutdown"
    if reason.startswith("client_disconnect"):
        if server_closed:
            return life_s, silent_s, "server"
        if code in _WS_PEER_FAREWELL_CLOSE_CODES:
            return life_s, silent_s, "peer"
        if code == _WS_UNATTRIBUTED_LOSS_CLOSE_CODE:
            return life_s, silent_s, "peer_or_keepalive"
        return life_s, silent_s, "unknown"
    if "send" in reason or reason == "receive_failed":
        return life_s, silent_s, "server"
    return life_s, silent_s, "unknown"


def _ws_log_closed(peer: str, reason: str, counts: tuple[int, int, int, int, int, int],
                   accepted_at, last_inbound_at, detected_at, code: Any, *,
                   shutting_down: bool, server_closed: bool) -> None:
    """The teardown ``ws closed`` line: C1's close fields plus the per-connection counters."""
    messages, parse_errors, dispatch_crashes, send_failures, reaped, detached = counts
    life_s, silent_s, initiated = _ws_close_fields(
        accepted_at, last_inbound_at,
        detected_at if detected_at is not None else time.monotonic(),
        reason, code, shutting_down=shutting_down, server_closed=server_closed)
    _log.info(
        "ws closed peer=%s reason=%s messages=%d parse_errors=%d "
        "dispatch_crashes=%d send_failures=%d reaped_sessions=%d detached_sessions=%d "
        "life_s=%.2f silent_s=%.2f initiated=%s",
        peer, reason, messages, parse_errors, dispatch_crashes, send_failures, reaped, detached,
        life_s, silent_s, initiated,
    )



async def _ws_end_reads(concurrent_reads: set, started_reads: set) -> None:
    """Reads that never started are dropped (their client is gone); started ones finish before the
    authority closes, unless the connection itself was cancelled (server shutdown)."""
    task = asyncio.current_task()
    shutting_down = task is not None and task.cancelling()
    for read in concurrent_reads:
        if shutting_down or read not in started_reads:
            read.cancel()
    await asyncio.gather(*concurrent_reads, return_exceptions=True)


class _SendFailed(Exception):
    """Raised by handle_ws._reply when a reply could not be written: ends the read loop."""


async def handle_ws(ws: Any, *, auth_identity: dict | None = None, subprotocol: str | None = None,
                    operator: bool = False) -> None:
    """Run one WebSocket session. Wire-compatible with ``tui_gateway.entry``. *auth_identity* is the server-minted
    ``{user_id, provider}`` recorded at WS-upgrade auth, stored as ``WSTransport.auth_identity`` (the only identity
    authority for session ownership and relay authorship); callers that omit it (harnesses, embedded TUI child) get None."""
    peer, transport = _ws_peer_label(ws), None
    authority_connection = None
    messages = parse_errors = dispatch_crashes = send_failures = 0
    disconnect_reason = "not_connected"
    # C1 reconnect telemetry (t_c1f387b2), logging only: accept / last-inbound / detection stamps.
    accepted_at = last_inbound_at = disconnect_detected_at = None
    close_code = None

    async def _reply(frame: dict, reason: str, msg: str, *args: Any) -> None:
        """write_async; on failure record *reason*, log *msg* and end the read loop."""
        nonlocal disconnect_reason, send_failures
        if not await transport.write_async(frame):
            disconnect_reason = reason
            send_failures += 1
            _log.warning(msg, *args)
            raise _SendFailed

    def _error(code: int, message: str, req_id: Any) -> dict:
        return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}

    pending: asyncio.Queue = asyncio.Queue(maxsize=_WS_DISPATCH_QUEUE_MAX)
    dispatcher: asyncio.Task | None = None
    concurrent_reads: set[asyncio.Task] = set()
    started_reads: set[asyncio.Task] = set()
    concurrent_read_slots = asyncio.Semaphore(_WS_CONCURRENT_READ_LIMIT)
    # Running + waiting reads: taken by the reader before a read is queued, so a flood parks it.
    read_admission = asyncio.Semaphore(_WS_CONCURRENT_READ_LIMIT + _WS_WAITING_READ_LIMIT)
    # Set when the read lane could not send a reply: ends the reader without cancelling the
    # control handler the serial lane is running (it finishes, then the dispatcher stops).
    send_failed: asyncio.Future = asyncio.get_running_loop().create_future()
    stop = object()  # queue sentinel; a JSON ``null`` frame parses to None, so None can't be it
    # Frames waiting for a profile's start, per start, in arrival order; one lane task drains each.
    held: dict[str, collections.deque] = {}
    held_lanes: set[asyncio.Task] = set()
    # Held frames count against the serial lane's bound: once that many wait, the dispatcher waits for
    # room, the queue fills and the reader parks, as it does behind a busy handler.
    held_room = asyncio.Semaphore(_WS_DISPATCH_QUEUE_MAX)
    # Set once the client is gone: frames still waiting for a start are dropped, not run for nobody.
    client_gone = asyncio.Event()
    # The start each lane still waits for, and the lanes the disconnect dropped: decided when the client
    # leaves, so a start that ends a moment later (before the lane resumes) cannot run them.
    held_waiting: dict[str, asyncio.Event] = {}
    held_dropped: set[str] = set()

    def _client_left() -> None:
        """At the disconnect, with no yield: drop every lane whose start has not ended yet. A lane that
        already dispatches keeps going (a viewer leaving never stops a turn)."""
        client_gone.set()
        held_dropped.update(key for key, started in held_waiting.items() if not started.is_set())

    async def _dispatch_one(req: Any) -> None:
        """Dispatch one request and write its inline reply."""
        nonlocal dispatch_crashes
        req_id = req.get("id") if isinstance(req, dict) else None
        req_method = req.get("method") if isinstance(req, dict) else None
        # dispatch() may schedule long handlers on the pool; it returns None then and the worker
        # writes the response itself via transport.write (a separate thread, so that is the safe
        # path). Inline handlers return the response dict, written here from the loop.
        try:
            resp = await _dispatch_request(authority_connection, req, req_method, transport)
        # health: allow BLE001 -- dispatch boundary: logs the traceback (_log.exception) and answers -32603
        except Exception:
            dispatch_crashes += 1
            _log.exception("ws dispatch crash peer=%s id=%s method=%s", peer, req_id, req_method)
            await _reply(_error(-32603, "internal error", req_id), "send_failed_after_dispatch_crash",
                         "ws dispatch-crash reply send failed peer=%s id=%s method=%s", peer, req_id, req_method)
            return
        if resp is not None:
            await _reply(resp, "send_failed_after_response",
                         "ws response send failed peer=%s id=%s method=%s", peer, req_id, req_method)

    async def _dispatch_concurrent_read(req: Any) -> None:
        async with concurrent_read_slots:
            started_reads.add(asyncio.current_task())
            try:
                await _dispatch_one(req)
            except _SendFailed:
                # A failed response ends the reader even from the concurrent lane, never by
                # cancelling the serial lane's in-flight handler.
                if not send_failed.done():
                    send_failed.set_result(None)
                raise

    def _finish_concurrent_read(task: asyncio.Task) -> None:
        # Here, not in the task: a task cancelled before its first step never runs its body.
        read_admission.release()
        concurrent_reads.discard(task)
        started_reads.discard(task)
        with contextlib.suppress(asyncio.CancelledError, _SendFailed):
            task.result()

    async def _dispatch_after_start(key: str, started: asyncio.Event) -> None:
        """Dispatch the frames held for one start, in arrival order, once it ends or the wait runs out
        (the authority then refuses a profile still starting retryably). A frame for that profile read
        meanwhile joins the queue instead of overtaking it on the serial lane. A client that leaves
        before the start ends drops them: nobody reads those replies."""
        try:
            waits = {asyncio.ensure_future(started.wait()), asyncio.ensure_future(client_gone.wait())}
            try:
                await asyncio.wait(waits, timeout=_WS_PROFILE_START_WAIT_S, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for wait in waits:
                    wait.cancel()
            del held_waiting[key]  # dispatching from here on: a later disconnect lets these frames finish
            frames = held[key]
            if key in held_dropped:
                for _ in frames:
                    held_room.release()
                frames.clear()
            while frames:
                await _dispatch_one(frames[0])
                frames.popleft()
                held_room.release()
        except _SendFailed:
            if not send_failed.done():
                send_failed.set_result(None)
        finally:
            held_waiting.pop(key, None)  # cancelled while waiting (server shutdown)
            held_dropped.discard(key)
            del held[key]

    async def _hold_for_profile_start(req: Any) -> bool:
        if authority_connection is None or authority_connection.awaits_profile_start(req, held) is None:
            return False
        await _unless_dispatch_failed(held_room.acquire())
        # Asked again after waiting for room: the start may have ended and its lane drained meanwhile,
        # and then the frame goes to the serial lane behind the frames that lane already answered.
        start = authority_connection.awaits_profile_start(req, held)
        if start is None:
            held_room.release()
            return False
        key, started = start
        if key in held:
            held[key].append(req)
        else:
            held[key] = collections.deque([req])
            assert started is not None  # a start not held yet comes from the registry, with its event
            held_waiting[key] = started
            if client_gone.is_set() and not started.is_set():
                held_dropped.add(key)  # read before the disconnect, for a start that had not ended by it
            lane = asyncio.create_task(_dispatch_after_start(key, started))
            held_lanes.add(lane)
            lane.add_done_callback(held_lanes.discard)
        return True

    async def _end_held() -> None:
        """Frames held for a start that ended before the disconnect finish, as the serial lane finishes
        its queue (the others were dropped), unless the connection itself was cancelled (server shutdown)."""
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            for lane in held_lanes:
                lane.cancel()
        await asyncio.gather(*held_lanes, return_exceptions=True)

    async def _next_request() -> Any:
        """The next queued frame, or ``stop`` once the read lane failed to send a reply."""
        step = asyncio.ensure_future(pending.get())
        await asyncio.wait({step, send_failed}, return_when=asyncio.FIRST_COMPLETED)
        if step.done():
            return stop if send_failed.done() else step.result()
        step.cancel()
        return stop

    async def _dispatch_loop() -> None:
        """Dispatch controls in arrival order; bounded expensive read-only projections run alongside them."""
        while (req := await _next_request()) is not stop:
            req_method = req.get("method") if isinstance(req, dict) else None
            if req_method in _WS_CONCURRENT_READ_METHODS:
                task = asyncio.create_task(_dispatch_concurrent_read(req))
                concurrent_reads.add(task)
                task.add_done_callback(_finish_concurrent_read)
                continue
            if await _hold_for_profile_start(req):
                continue
            await _dispatch_one(req)

    async def _queue_request(req: Any) -> None:
        if _answers_open_request(req):
            await _dispatch_one(req)
            return
        if isinstance(req, dict) and req.get("method") in _WS_CONCURRENT_READ_METHODS:
            # A read past the waiting cap parks the reader, as the serial lane did; released
            # when its task ends (a read cancelled before it ran releases it too).
            await _unless_dispatch_failed(read_admission.acquire())
        await _unless_dispatch_failed(pending.put(req))

    async def _unless_dispatch_failed(aw: Any) -> Any:
        """Await *aw*, or raise _SendFailed once a reply could not be sent (the dispatcher ended on one,
        or the read lane failed one)."""
        assert dispatcher is not None
        step = asyncio.ensure_future(aw)
        try:
            await asyncio.wait({step, dispatcher, send_failed}, return_when=asyncio.FIRST_COMPLETED)
            if step.done():
                return step.result()
            raise _SendFailed
        finally:
            step.cancel()  # no-op once done

    try:
        await (ws.accept(subprotocol=subprotocol) if subprotocol else ws.accept())
        disconnect_reason = "connected"
        accepted_at = last_inbound_at = time.monotonic()
        # Mark the client attached before the (possibly slow) ready/skin setup so scale-to-zero sees it.
        _note_dashboard_client_activity(force=True)
        _disable_nagle(ws)
        _log.info("ws accepted peer=%s", peer)
        transport = WSTransport(ws, asyncio.get_running_loop(), peer=peer, auth_identity=auth_identity)
        authority_connection = _authority_connection(ws, transport, auth_identity, operator)
        transport.has_session_authority = authority_connection is not None
        # resolve_skin() is sync I/O + CPU; pooled so the read loop can drain the frontend's initial RPC burst.
        skin_payload, skin_etag = await asyncio.to_thread(server.resolve_skin_with_etag)
        # Stamped before the write so the folded handshake answers the etag THIS ready frame carried.
        transport.skin_etag = skin_etag
        ready_ok = await transport.write_async(_ws_ready_frame(
            skin_payload, skin_etag=skin_etag, has_session_authority=authority_connection is not None))
        if ready_ok:
            _ws_attach_ready_watchers(transport, authority_connection)
        # Cross-backend liveness: a heartbeat row lets the startup orphan sweep tell "live but idle
        # backend" from "truly orphaned". Idempotent and once-per-process, like the orphan sweep (the
        # desktop app and web dashboard reach the agent via this sidecar, not entry.main()).
        _start_backend_liveness()
        if not ready_ok:
            disconnect_reason = "ready_send_failed"
            disconnect_detected_at = time.monotonic()
            send_failures += 1
            _log.error("ws ready frame send failed peer=%s", peer)
            return

        dispatcher = asyncio.create_task(_dispatch_loop())
        while True:
            try:
                raw = await _unless_dispatch_failed(ws.receive_text())
                last_inbound_at = time.monotonic()
                _note_dashboard_client_activity()
            except _SendFailed:
                raise
            except _WebSocketDisconnect as exc:
                disconnect_detected_at = time.monotonic()
                close_code = getattr(exc, "code", None)
                disconnect_reason = f"client_disconnect(code={close_code},reason={getattr(exc, 'reason', None)})"
                break
            except Exception:
                disconnect_detected_at = time.monotonic()
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
                _log.warning("ws parse error peer=%s index=%d error=%s payload=%r", peer, messages, exc, line[:_WS_LOG_PAYLOAD_PREVIEW])
                await _reply(_error(-32700, "parse error", None), "send_failed_after_parse_error",
                             "ws parse-error reply send failed peer=%s", peer)
                continue
            if isinstance(req, dict) and req.get("method") == "gateway.ping":
                req_id = req.get("id")
                await _reply({"jsonrpc": "2.0", "result": {"ok": True}, "id": req_id}, "send_failed_after_heartbeat",
                             "ws heartbeat reply send failed peer=%s id=%s", peer, req_id)
                continue
            await _queue_request(req)
    except _SendFailed:
        disconnect_detected_at = time.monotonic()
        # A failed reply send ends the loop; never leave the healthy 'connected' reason standing
        # (C1: teardown must name the server as the initiator).
        disconnect_reason = "send_failed" if disconnect_reason == "connected" else disconnect_reason
    finally:
        _client_left()
        if dispatcher is not None:
            # Finish the in-flight handler and the frames read before the disconnect (as the serial read loop
            # did) before the teardown below parks this transport's sessions. A cancelled connection (server
            # shutdown) stops at once instead, as the read loop's cancelled await did.
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                dispatcher.cancel()
            else:
                with contextlib.suppress(_SendFailed):
                    await _unless_dispatch_failed(pending.put(stop))
            await asyncio.wait({dispatcher})
            failure = None if dispatcher.cancelled() else dispatcher.exception()
            if failure is not None and not isinstance(failure, _SendFailed):
                _log.error("ws dispatcher failed peer=%s", peer, exc_info=failure)
        await _ws_end_reads(concurrent_reads, started_reads)
        await _end_held()
        if authority_connection is not None:
            # After the dispatcher drained: an in-flight authority call must not see its connection closed.
            await authority_connection.close()
        reaped_sessions = detached_sessions = 0
        if transport is not None:
            server.unregister_live_transport(transport)
            transport.close()
            try:
                await asyncio.to_thread(server._release_wake_for_transport, transport)
            except Exception:
                _log.exception("ws wake-word teardown failed peer=%s", peer)
            # The single WS-disconnect teardown path: reap sessions this transport owned (close_on_disconnect
            # sidecars) or detach the rest to the drop sentinel so later emits don't hit a closed socket; detached
            # ones go to the grace-windowed orphan reaper (a quick resume cancels it). Offloaded: worker.close()
            # blocks (terminate + waits) plus a sync DB write, which inline would freeze the loop for every peer.
            try:
                reaped_sessions, detached_sessions = await asyncio.to_thread(
                    server._close_sessions_for_transport, transport, end_reason="ws_disconnect"
                )
            except Exception:
                _log.exception("ws transport teardown failed peer=%s", peer)
        try:
            await ws.close()
        except Exception as exc:
            _log.debug("ws close failed peer=%s error=%s", peer, exc)
        # C1 teardown fields (t_c1f387b2): life and orphan-silent time measured to the DETECTION
        # instant, not to the end of this (possibly slow) teardown.
        task = asyncio.current_task()
        _ws_log_closed(
            peer, disconnect_reason,
            (messages, parse_errors, dispatch_crashes, send_failures, reaped_sessions, detached_sessions),
            accepted_at, last_inbound_at, disconnect_detected_at, close_code,
            shutting_down=bool(task is not None and task.cancelling()),
            server_closed=bool(transport is not None and transport.server_closed),
        )
