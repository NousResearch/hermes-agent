"""voice.grok.* JSON-RPC handlers (docs/grok-live-voice/SPEC §4.1): the renderer-facing surface
of the Grok-Live engine. The xAI websocket itself lives in
``tools/voice_live_grok_bridge.py`` (one ``GrokLiveBridge`` per Hermes session, owned here);
audio rides base64 PCM16 chunks inside these methods/events (the ``wake.feed`` precedent),
NOT raw binary WS frames. Bodies are rebound onto server.py's globals (method_ctx.bind_module),
used bare — ``_caller_transport`` / ``_wake_resume_if_owner`` are methods_voice's, published
onto the same namespace.

Routing / delegation design (revised 2026-09-15, renderer-submits):

- The RENDERER is the single ``prompt.submit`` caller (gpt-live parity): its ``onDelegation``
  handler submits like a typed message, lazily creating the chat session on a fresh draft.
  The backend never submits — the old ``_grok_delegation_sink`` double-submitted every
  delegation and no-oped on a fresh draft's synthetic id (kanban t_07a77402).
- ``voice.grok.*`` events therefore cannot rely on ``_sessions[sid].transport`` alone: a
  fresh-draft bridge is keyed by an id that is not (yet) a session. ``_grok_emit_for`` falls
  back to the caller transport captured at ``voice.grok.start`` and refreshed by every
  ``voice.grok.audio`` chunk, so events always reach the app that opened the mic — never stdio.
- The spoken reply is pulled, not pushed: the renderer calls ``voice.grok.speak`` when its
  turn settles (the bridge's ``speak_reply`` — xAI force_message — is unchanged).
"""

from __future__ import annotations

import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

# One live xAI session per Hermes session; wake-word mic-lease owners per session (SPEC §6).
_grok_bridges_lock = threading.Lock()
_grok_bridges: dict = {}        # session_id -> GrokLiveBridge
_grok_wake_owners: dict = {}    # session_id -> transport that paused the wake detector
# session_id -> transport of the client that opened this voice bridge (fresh-draft routing:
# the renderer-submits design keys the bridge by an id that need not be a Hermes session).
_grok_client_transports: dict = {}
# Renderer-supplied routing key -> canonical bridge key (a start may use a synthetic id, the
# renderer later learns the real chat session id and re-keys its audio/speak calls onto it).
_grok_alias_lock = threading.Lock()
_grok_alias: dict = {}


def _grok_emit_for(session_id: str, transport=None):
    """Bind the bridge's event sink so ``voice.grok.*`` events reach the app that owns this
    voice conversation: ``write_json``'s full path (replay stamping + session-transport
    routing) when the id is a live Hermes session — existing-chat behavior, byte-identical —
    else the caller transport captured at start / refreshed by the latest upstream chunk,
    never stdio, which was the fresh-draft silent failure. The literal calls double as the
    emitter inventory the contract-completeness scan (tests/tui_gateway/contracts/
    test_generated.py) discovers — and whitelist the sink so the bridge can never emit an
    undeclared event name."""
    def _routed(session_id: str, event: str, payload: dict) -> None:
        """Full write_json path when the id is a live Hermes session (replay stamping +
        session-transport routing — existing-chat behavior, byte-identical); else the
        fresh-draft fallback below (write_json would drop the frame to stdio)."""
        if (_sessions.get(session_id) or {}).get("transport") is not None:
            _emit(event, session_id, payload)
            return
        frame = _event_frame(event, session_id, payload)
        from tui_gateway.event_replay import _stamp_event
        from tui_gateway.transport import current_transport
        _stamp_event(frame)
        with _grok_bridges_lock:
            t = _grok_client_transports.get(session_id)
        t = t or transport or current_transport()
        if t is not None:
            t.write(frame)

    def _sink(event_type: str, payload: dict) -> None:
        if event_type == "voice.grok.audio":
            _routed(session_id, "voice.grok.audio", payload)
        elif event_type == "voice.grok.transcript":
            _routed(session_id, "voice.grok.transcript", payload)
        elif event_type == "voice.grok.state":
            _routed(session_id, "voice.grok.state", payload)
        elif event_type == "voice.grok.delegation":
            _routed(session_id, "voice.grok.delegation", payload)
        else:
            logger.warning("voice.grok: bridge emitted undeclared event %r — dropped", event_type)
    return _sink


def _grok_refresh_client_transport(session_id: str, transport) -> None:
    """Latest caller transport wins (reconnect / renderer reload re-keys where events go)."""
    with _grok_bridges_lock:
        if session_id in _grok_client_transports:
            _grok_client_transports[session_id] = transport


def _grok_key_of(session_id: str) -> str:
    """Canonical bridge key for a renderer-supplied id (follows the alias set on re-key)."""
    with _grok_alias_lock:
        return _grok_alias.get(session_id, session_id)


def _grok_get_bridge(session_id: str):
    """The live bridge for ``session_id``; a dead one (error/idle after fatal) is reaped so the
    next ``voice.grok.start`` creates a fresh bridge instead of reusing a corpse."""
    sid = _grok_key_of(session_id)
    with _grok_bridges_lock:
        bridge = _grok_bridges.get(sid)
    if bridge is not None and not bridge.alive:
        with _grok_bridges_lock:
            _grok_bridges.pop(sid, None)
        return None
    return bridge


@method("voice.grok.status")
def _(rid, params: dict) -> dict:
    """Non-secret Grok-Live readiness verdict — mirrors ``GET /api/audio/voice-live-grok/status``
    for TUI/desktop-over-gateway parity (SPEC §11). Never carries the credential."""
    try:
        from tools.voice_live_grok import resolve_grok_live_status
        return _ok(rid, resolve_grok_live_status())
    except Exception as e:
        return _err(rid, 5026, str(e))


@method("voice.grok.start")
def _(rid, params: dict) -> dict:
    """Open (or reuse) the backend xAI realtime session for ``session_id``. Fail-fast when no
    credential resolves (SPEC §8/§9: an explicit start failure is reported, never retried); the
    connect handshake itself is async and reported via ``voice.grok.state`` events.

    The id is the chat's Hermes session id when one exists; on a FRESH DRAFT the renderer's
    synthetic id is accepted as the bridge key — events route to the caller transport captured
    here, and the renderer re-keys onto the real id via ``voice.grok.rekey`` once its submit
    mints the session."""
    session_id = str(params.get("session_id") or "").strip()
    if not session_id:
        return _err(rid, 4001, "voice.grok.start requires session_id")
    try:
        from tools.voice_live_grok import _grok_live_section, resolve_grok_live_status
        status = resolve_grok_live_status()
    except Exception as e:
        return _err(rid, 5026, str(e))
    if not status.get("available"):
        return _ok(rid, {"started": False, "reason": status.get("reason")})
    # Mic lease (SPEC §6): the same pause path voice.record uses before opening the mic.
    transport = _caller_transport()
    if _grok_get_bridge(session_id) is not None:
        with _grok_bridges_lock:
            _grok_client_transports[session_id] = transport
        return _ok(rid, {"started": True, "reused": True})
    try:
        from tools.wake_word import pause_listening
        if pause_listening(owner=transport):
            with _grok_bridges_lock:
                _grok_wake_owners[session_id] = transport
    except Exception as e:
        logger.debug("voice.grok.start: wake pause failed (best-effort): %s", e)
    from tools.voice_live_grok_bridge import GrokLiveBridge
    bridge = GrokLiveBridge(session_id=session_id, emit=_grok_emit_for(session_id, transport),
                            live_config=_grok_live_section())
    with _grok_bridges_lock:
        _grok_bridges[session_id] = bridge
        _grok_client_transports[session_id] = transport
    try:
        bridge.start()
    except Exception as e:
        with _grok_bridges_lock:
            _grok_bridges.pop(session_id, None)
            _grok_client_transports.pop(session_id, None)
        return _err(rid, 5026, str(e))
    logger.info("voice.grok.start: bridge starting for session=%s", session_id)
    return _ok(rid, {"started": True, "state": "connecting"})


@method("voice.grok.stop")
def _(rid, params: dict) -> dict:
    """User-initiated stop: close the xAI session (cancelling any pending reconnect) and hand
    the mic back to the wake-word detector. Symmetric with gpt-live's ``end()`` (SPEC §9)."""
    session_id = str(params.get("session_id") or "").strip()
    if not session_id:
        return _err(rid, 4001, "voice.grok.stop requires session_id")
    sid = _grok_key_of(session_id)
    with _grok_bridges_lock:
        bridge = _grok_bridges.pop(sid, None)
        owner = _grok_wake_owners.pop(sid, None)
        _grok_client_transports.pop(sid, None)
    with _grok_alias_lock:
        _grok_alias.pop(session_id, None)
        stale = [k for k, v in _grok_alias.items() if v == sid]
        for k in stale:
            _grok_alias.pop(k, None)
    if owner is not None:
        try:
            _wake_resume_if_owner(owner)
        except Exception as e:
            logger.debug("voice.grok.stop: wake resume failed (best-effort): %s", e)
    if bridge is None:
        return _ok(rid, {"stopped": False, "reason": "not_running"})
    return _ok(rid, {"stopped": bool(bridge.stop())})


@method("voice.grok.audio")
def _(rid, params: dict) -> dict:
    """One ~100ms upstream mic chunk (base64 PCM16 mono 24kHz) → relay toward xAI. The AEC gate
    and mute decision live server-side in the bridge (SPEC §7); backpressure drops, never
    queues (SPEC §4.1). Also refreshes the caller transport this bridge's events fall back to."""
    session_id = str(params.get("session_id") or "").strip()
    if not session_id:
        return _err(rid, 4001, "voice.grok.audio requires session_id")
    raw_b64 = params.get("pcm_b64") or ""
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        return _err(rid, 4001, "voice.grok.audio requires base64 pcm")
    import base64
    try:
        pcm = base64.b64decode(raw_b64, validate=False)
    except Exception as e:
        return _err(rid, 4001, f"invalid base64 pcm: {e}")
    if not pcm:
        return _ok(rid, {"accepted": False, "reason": "empty"})
    if len(pcm) > 96000:  # soft cap: 2s of 24 kHz int16 mono (a legit chunk is ~4.8KB)
        return _err(rid, 4001, "pcm frame too large")
    bridge = _grok_get_bridge(session_id)
    if bridge is None:
        return _ok(rid, {"accepted": False, "reason": "not_running"})
    _grok_refresh_client_transport(_grok_key_of(session_id), _caller_transport())
    return _ok(rid, bridge.send_mic(pcm))


@method("voice.grok.mute")
def _(rid, params: dict) -> dict:
    """Explicit mute toggle — independent of the AEC half-duplex gate (SPEC §4.1/§7)."""
    session_id = str(params.get("session_id") or "").strip()
    if not session_id:
        return _err(rid, 4001, "voice.grok.mute requires session_id")
    bridge = _grok_get_bridge(session_id)
    if bridge is None:
        return _err(rid, 4021, "no live grok-live session for session_id")
    return _ok(rid, {"muted": bridge.set_muted(bool(params.get("muted")))})


@method("voice.grok.speak")
def _(rid, params: dict) -> dict:
    """Renderer-submits design (SPEC §5 revised): the renderer is the sole ``prompt.submit``
    caller, so IT knows when the Hermes turn settled — and pulls the spoken reply through here.
    ``speak_reply`` makes xAI speak the text verbatim (force_message; no re-prompt, no model
    involvement on the xAI side). Presentation cleanup (markdown strip) stays renderer-side."""
    session_id = str(params.get("session_id") or "").strip()
    text = str(params.get("text") or "")
    if not session_id:
        return _err(rid, 4001, "voice.grok.speak requires session_id")
    if not text.strip():
        return _ok(rid, {"spoken": False, "reason": "empty"})
    bridge = _grok_get_bridge(session_id)
    if bridge is None:
        logger.info("voice.grok.speak: not_running session=%s chars=%d", session_id, len(text))
        return _ok(rid, {"spoken": False, "reason": "not_running"})
    spoken = bool(bridge.speak_reply(text))
    logger.info("voice.grok.speak: session=%s spoken=%s chars=%d", session_id, spoken, len(text))
    return _ok(rid, {"spoken": spoken})


@method("voice.grok.rekey")
def _(rid, params: dict) -> dict:
    """Fresh-draft completion: the renderer learned the chat's real Hermes session id (its
    submit minted it) and re-keys the bridge so events/delegation route by session from now
    on. Idempotent; a no-op when the voice session was already keyed by the real id."""
    from_id = str(params.get("from_session_id") or "").strip()
    to_id = str(params.get("to_session_id") or "").strip()
    if not from_id or not to_id or from_id == to_id:
        return _err(rid, 4001, "voice.grok.rekey requires distinct from_session_id/to_session_id")
    with _grok_bridges_lock:
        bridge = _grok_bridges.get(_grok_key_of(from_id))
        if bridge is None:
            return _ok(rid, {"rekeyed": False, "reason": "not_running"})
        old_key = _grok_key_of(from_id)
        if old_key != to_id:
            _grok_bridges[to_id] = _grok_bridges.pop(old_key)
            _grok_client_transports[to_id] = _grok_client_transports.pop(old_key, None)
            _grok_wake_owners[to_id] = _grok_wake_owners.pop(old_key, None)
            with _grok_alias_lock:
                _grok_alias[from_id] = to_id
    logger.info("voice.grok.rekey: bridge %s -> session=%s", from_id, to_id)
    return _ok(rid, {"rekeyed": True})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
