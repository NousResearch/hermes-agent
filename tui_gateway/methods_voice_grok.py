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
_profile_scoped = _registry.profile_scoped

# Emitter inventory for the wire-completeness oracle (tests/tui_gateway/contracts/
# test_generated.py imports this, the _CHANGE_WATCHES/_CHILD_DELTA_EVENTS precedent):
# the bridge emits through the _sink/_routed indirection below, whose call shape the
# literal scan cannot see, so the four names are declared here once. The _sink
# whitelist below drops anything else, so a new event name fails this way loudly.
GROK_LIVE_EVENTS = frozenset({
    "voice.grok.audio",
    "voice.grok.transcript",
    "voice.grok.state",
    "voice.grok.delegation",
})

# One live xAI session per Hermes session; wake-word mic-lease owners per session (SPEC §6).
_grok_bridges_lock = threading.Lock()
_grok_start_lock = threading.Lock()
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
    never stdio, which was the fresh-draft silent failure. The emitted names are declared
    once in ``GROK_LIVE_EVENTS`` (module level) for the contract-completeness scan — the
    ``_sink`` whitelist below drops anything undeclared, so the two cannot drift apart."""
    def _routed(session_id: str, event: str, payload: dict) -> None:
        """Full write_json path when the id is a live Hermes session (replay stamping +
        session-transport routing — existing-chat behavior, byte-identical); else the
        fresh-draft fallback below (write_json would drop the frame to stdio)."""
        if (_sessions.get(session_id) or {}).get("transport") is not None:
            _emit(event, session_id, payload)
            return
        frame = _event_frame(event, session_id, payload)
        from tui_gateway.event_replay import _stamp_event
        _stamp_event(frame)
        with _grok_bridges_lock:
            t = _grok_client_transports.get(session_id)
        t = t or transport
        if t is not None:
            t.write(frame)

    def _sink(event_type: str, payload: dict) -> None:
        # The closure keeps the original id so a bridge can survive a fresh-draft
        # rekey. Resolve it at emission time and canonicalize the payload too;
        # otherwise the envelope and payload can keep the abandoned synthetic id.
        routed_id = _grok_key_of(session_id)
        payload = {**payload, "session_id": routed_id}
        if event_type in GROK_LIVE_EVENTS:
            _routed(routed_id, event_type, payload)
        else:
            logger.warning("voice.grok: bridge emitted undeclared event %r — dropped", event_type)
    return _sink


def _grok_owns(session_id: str, transport) -> bool:
    """Only the client that opened a voice bridge may control or receive it."""
    with _grok_bridges_lock:
        return _grok_client_transports.get(session_id) is transport


def _grok_key_of(session_id: str) -> str:
    """Canonical bridge key for a renderer-supplied id (follows the alias set on re-key)."""
    with _grok_alias_lock:
        return _grok_alias.get(session_id, session_id)


def _grok_take_bridge(sid: str, expected=None):
    """Remove one bridge and all of its routing/lease state atomically."""
    with _grok_bridges_lock:
        if expected is not None and _grok_bridges.get(sid) is not expected:
            return None, None
        bridge = _grok_bridges.pop(sid, None)
        owner = _grok_wake_owners.pop(sid, None)
        _grok_client_transports.pop(sid, None)
    with _grok_alias_lock:
        for alias, target in list(_grok_alias.items()):
            if alias == sid or target == sid:
                _grok_alias.pop(alias, None)
    return bridge, owner


def _grok_resume_wake(owner, context: str) -> None:
    if owner is None:
        return
    try:
        _wake_resume_if_owner(owner)
    except Exception as e:
        logger.debug("voice.grok: %s wake resume failed (best-effort): %s", context, e)


def drop_grok_bridges_for_transport(transport) -> None:
    """Stop voice sessions owned by a disconnected renderer and release their mic leases."""
    with _grok_bridges_lock:
        sids = [sid for sid, owner in _grok_client_transports.items() if owner is transport]
    for sid in sids:
        bridge, owner = _grok_take_bridge(sid)
        _grok_resume_wake(owner, "disconnect")
        if bridge is not None:
            try:
                bridge.stop(timeout=2.0)
            except Exception:
                logger.debug("voice.grok: disconnect bridge stop failed", exc_info=True)


def _grok_get_bridge(session_id: str):
    """Return the live bridge for ``session_id`` and fully reap a dead one.

    Reaping must release the wake-word lease and routing/alias state as well as
    the bridge object; otherwise a failed xAI session can hold the microphone
    indefinitely and poison the next start.
    """
    sid = _grok_key_of(session_id)
    with _grok_bridges_lock:
        bridge = _grok_bridges.get(sid)
    if bridge is None or bridge.alive:
        return bridge

    with _grok_bridges_lock:
        if _grok_bridges.get(sid) is not bridge:
            return _grok_bridges.get(sid)
    _removed, owner = _grok_take_bridge(sid, expected=bridge)
    _grok_resume_wake(owner, "dead-bridge")
    return None


@method("voice.grok.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Non-secret Grok-Live readiness verdict — mirrors ``GET /api/audio/voice-live-grok/status``
    for TUI/desktop-over-gateway parity (SPEC §11). Never carries the credential."""
    try:
        from tools.voice_live_grok import resolve_grok_live_status
        return _ok(rid, resolve_grok_live_status())
    except Exception as e:
        return _err(rid, 5026, str(e))


@method("voice.grok.start")
@_profile_scoped
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
        from tools.voice_live_grok import _grok_live_section, _resolve_grok_credentials, resolve_grok_live_status
        status = resolve_grok_live_status()
    except Exception as e:
        return _err(rid, 5026, str(e))
    if not status.get("available"):
        return _ok(rid, {"started": False, "reason": status.get("reason")})
    live_config = _grok_live_section()
    credential = _resolve_grok_credentials(live_config)
    if not credential:
        return _ok(rid, {"started": False, "reason": status.get("reason")})
    # Serialize check/create/start so a double-click cannot orphan a second xAI
    # thread. The caller transport is fixed for the lifetime of the bridge.
    transport = _caller_transport()
    sid = _grok_key_of(session_id)
    target = _sessions.get(sid)
    target_transport = target.get("transport") if isinstance(target, dict) else None
    if target_transport is not None and target_transport is not transport:
        return _err(rid, 4021, "chat session is owned by another client")
    with _grok_start_lock:
        if _grok_get_bridge(sid) is not None:
            if not _grok_owns(sid, transport):
                return _err(rid, 4021, "voice bridge is owned by another client")
            return _ok(rid, {"started": True, "reused": True})
        # Mic lease (SPEC §6): the same pause path voice.record uses before opening the mic.
        wake_paused = False
        try:
            from tools.wake_word import pause_listening
            wake_paused = bool(pause_listening(owner=transport))
            if wake_paused:
                with _grok_bridges_lock:
                    _grok_wake_owners[sid] = transport
        except Exception as e:
            logger.debug("voice.grok.start: wake pause failed (best-effort): %s", e)
        from tools.voice_live_grok_bridge import GrokLiveBridge
        bridge = GrokLiveBridge(session_id=sid, emit=_grok_emit_for(sid, transport),
                                live_config=live_config, credential=credential)
        with _grok_bridges_lock:
            _grok_bridges[sid] = bridge
            _grok_client_transports[sid] = transport
        try:
            bridge.start()
        except Exception as e:
            with _grok_bridges_lock:
                _grok_bridges.pop(sid, None)
                _grok_client_transports.pop(sid, None)
                owner = _grok_wake_owners.pop(sid, None)
            if owner is not None:
                try:
                    _wake_resume_if_owner(owner)
                except Exception:
                    logger.debug("voice.grok.start: wake rollback failed", exc_info=True)
            return _err(rid, 5026, str(e))
    logger.info("voice.grok.start: bridge starting for session=%s", sid)
    return _ok(rid, {"started": True, "state": "connecting"})


@method("voice.grok.stop")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """User-initiated stop: close the xAI session (cancelling any pending reconnect) and hand
    the mic back to the wake-word detector. Symmetric with gpt-live's ``end()`` (SPEC §9)."""
    session_id = str(params.get("session_id") or "").strip()
    if not session_id:
        return _err(rid, 4001, "voice.grok.stop requires session_id")
    sid = _grok_key_of(session_id)
    with _grok_bridges_lock:
        bridge = _grok_bridges.get(sid)
        if bridge is not None and _grok_client_transports.get(sid) is not _caller_transport():
            return _err(rid, 4021, "voice bridge is owned by another client")
    bridge, owner = _grok_take_bridge(sid, expected=bridge)
    _grok_resume_wake(owner, "stop")
    if bridge is None:
        return _ok(rid, {"stopped": False, "reason": "not_running"})
    return _ok(rid, {"stopped": bool(bridge.stop())})


@method("voice.grok.audio")
@_profile_scoped
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
    sid = _grok_key_of(session_id)
    bridge = _grok_get_bridge(sid)
    if bridge is None:
        return _ok(rid, {"accepted": False, "reason": "not_running"})
    if not _grok_owns(sid, _caller_transport()):
        return _err(rid, 4021, "voice bridge is owned by another client")
    return _ok(rid, bridge.send_mic(pcm))


@method("voice.grok.mute")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Explicit mute toggle — independent of the AEC half-duplex gate (SPEC §4.1/§7)."""
    session_id = str(params.get("session_id") or "").strip()
    if not session_id:
        return _err(rid, 4001, "voice.grok.mute requires session_id")
    sid = _grok_key_of(session_id)
    bridge = _grok_get_bridge(sid)
    if bridge is None:
        return _err(rid, 4021, "no live grok-live session for session_id")
    if not _grok_owns(sid, _caller_transport()):
        return _err(rid, 4021, "voice bridge is owned by another client")
    return _ok(rid, {"muted": bridge.set_muted(bool(params.get("muted")))})


@method("voice.grok.speak")
@_profile_scoped
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
    sid = _grok_key_of(session_id)
    bridge = _grok_get_bridge(sid)
    if bridge is None:
        logger.info("voice.grok.speak: not_running session=%s chars=%d", session_id, len(text))
        return _ok(rid, {"spoken": False, "reason": "not_running"})
    if not _grok_owns(sid, _caller_transport()):
        return _err(rid, 4021, "voice bridge is owned by another client")
    spoken = bool(bridge.speak_reply(text))
    logger.info("voice.grok.speak: session=%s spoken=%s chars=%d", session_id, spoken, len(text))
    return _ok(rid, {"spoken": spoken})


@method("voice.grok.rekey")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Fresh-draft completion: the renderer learned the chat's real Hermes session id (its
    submit minted it) and re-keys the bridge so events/delegation route by session from now
    on. Idempotent; a no-op when the voice session was already keyed by the real id."""
    from_id = str(params.get("from_session_id") or "").strip()
    to_id = str(params.get("to_session_id") or "").strip()
    if not from_id or not to_id or from_id == to_id:
        return _err(rid, 4001, "voice.grok.rekey requires distinct from_session_id/to_session_id")
    transport = _caller_transport()
    target = _sessions.get(to_id)
    if not isinstance(target, dict) or target.get("transport") is not transport:
        return _err(rid, 4021, "target chat session is not owned by this client")
    with _grok_bridges_lock:
        old_key = _grok_key_of(from_id)
        bridge = _grok_bridges.get(old_key)
        if bridge is None:
            return _ok(rid, {"rekeyed": False, "reason": "not_running"})
        if _grok_client_transports.get(old_key) is not transport:
            return _err(rid, 4021, "voice bridge is owned by another client")
        if old_key != to_id:
            if to_id in _grok_bridges:
                return _ok(rid, {"rekeyed": False, "reason": "target_already_running"})
            _grok_bridges[to_id] = _grok_bridges.pop(old_key)
            _grok_client_transports[to_id] = _grok_client_transports.pop(old_key)
            wake_owner = _grok_wake_owners.pop(old_key, None)
            if wake_owner is not None:
                _grok_wake_owners[to_id] = wake_owner
            with _grok_alias_lock:
                _grok_alias[from_id] = to_id
    # Status and any payload created by the bridge itself must adopt the real id
    # too; moving only the registry leaves future events on the synthetic id.
    bridge.rekey(to_id)
    logger.info("voice.grok.rekey: bridge %s -> session=%s", from_id, to_id)
    return _ok(rid, {"rekeyed": True})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
