"""voice.grok.* JSON-RPC handlers (docs/grok-live-voice/SPEC §4.1): the renderer-facing surface
of the Grok-Live engine. The xAI websocket itself lives in
``tools/voice_live_grok_bridge.py`` (one ``GrokLiveBridge`` per Hermes session, owned here);
audio rides base64 PCM16 chunks inside these methods/events (the ``wake.feed`` precedent),
NOT raw binary WS frames. Bodies are rebound onto server.py's globals (method_ctx.bind_module),
used bare — ``_caller_transport`` / ``_wake_resume_if_owner`` are methods_voice's, published
onto the same namespace.
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


def _grok_emit_for(session_id: str):
    """Bind the bridge's event sink to this session so ``voice.grok.*`` events route to the
    transport that owns the Hermes session (server.write_json's session-id routing). The
    literal calls double as the emitter inventory the contract-completeness scan
    (tests/tui_gateway/contracts/test_generated.py) discovers — and whitelist the sink so the
    bridge can never emit an undeclared event name."""
    def _sink(event_type: str, payload: dict) -> None:
        if event_type == "voice.grok.audio":
            _emit("voice.grok.audio", session_id, payload)
        elif event_type == "voice.grok.transcript":
            _emit("voice.grok.transcript", session_id, payload)
        elif event_type == "voice.grok.state":
            _emit("voice.grok.state", session_id, payload)
        elif event_type == "voice.grok.delegation":
            _emit("voice.grok.delegation", session_id, payload)
        else:
            logger.warning("voice.grok: bridge emitted undeclared event %r — dropped", event_type)
    return _sink


def _grok_get_bridge(session_id: str):
    """The live bridge for ``session_id``; a dead one (error/idle after fatal) is reaped so the
    next ``voice.grok.start`` creates a fresh bridge instead of reusing a corpse."""
    with _grok_bridges_lock:
        bridge = _grok_bridges.get(session_id)
    if bridge is not None and not bridge.alive:
        with _grok_bridges_lock:
            _grok_bridges.pop(session_id, None)
        return None
    return bridge


def _grok_delegation_sink(session_id: str):
    """SPEC §5 step 3: a settled spoken exchange becomes a NORMAL Hermes turn on the open
    session — mirrors gpt-live's seam exactly (same ``surface: "voice-live"``, same per-turn
    note function ``tools/voice_live.py::voice_live_turn_note()`` reused verbatim via
    ``session_notifications._hud_surface_note`` — no new note text is authored here, the
    single biggest cache-safety guarantee per SPEC §5/§13#1). Never touches the system prompt,
    the toolset, or the agent's context — only ``prompt.submit``'s existing ``text``/
    ``voice_context``/``surface``/``queued`` wire params, exactly like a client-issued submit.

    Runs on the bridge's asyncio loop thread; ``prompt.submit`` itself just claims the turn and
    returns — the actual model call runs on its own daemon thread (``_run_prompt_submit``), so
    this call does not block the bridge."""
    def _sink(sid: str, delegation_id: str, prompt: str, context: str) -> None:
        bridge = _grok_get_bridge(sid)

        def _speak_reply(text: str, status: str) -> None:
            # A voice reply must be short/speakable prose per SPEC (the per-turn note in
            # voice_live_turn_note() already instructs the model to answer that way — this
            # is presentation cleanup of markdown/links only, not a persona rewrite).
            if bridge is None or not bridge.alive or status != "complete" or not text.strip():
                return
            try:
                from tools.tts_text_normalize import strip_markdown_for_tts
                spoken = strip_markdown_for_tts(text).strip()
            except Exception:
                spoken = text.strip()
            if spoken:
                bridge.speak_reply(spoken)

        session = _sessions.get(sid)
        if not isinstance(session, dict):
            return
        # Server-internal only (methods_prompt.py pops it, never a wire param); last delegation
        # in a session wins if a newer one supersedes an in-flight turn, matching gpt-live's
        # busyRef-gated interrupt-then-submit semantics (SPEC §13#2).
        session["_voice_reply_sink"] = _speak_reply
        submitted = _methods["prompt.submit"](f"grok-delegation-{delegation_id}", {
            "session_id": sid, "text": prompt, "surface": "voice-live",
            "voice_context": context, "queued": True})
        if "error" in submitted:
            logger.warning("grok-live: delegation prompt.submit failed for session=%s: %s",
                           sid, submitted["error"])
    return _sink


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
    connect handshake itself is async and reported via ``voice.grok.state`` events."""
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
    if _grok_get_bridge(session_id) is not None:
        return _ok(rid, {"started": True, "reused": True})
    # Mic lease (SPEC §6): the same pause path voice.record uses before opening the mic.
    transport = _caller_transport()
    try:
        from tools.wake_word import pause_listening
        if pause_listening(owner=transport):
            with _grok_bridges_lock:
                _grok_wake_owners[session_id] = transport
    except Exception as e:
        logger.debug("voice.grok.start: wake pause failed (best-effort): %s", e)
    from tools.voice_live_grok_bridge import GrokLiveBridge
    bridge = GrokLiveBridge(session_id=session_id, emit=_grok_emit_for(session_id),
                            delegation_sink=_grok_delegation_sink(session_id),
                            live_config=_grok_live_section())
    with _grok_bridges_lock:
        _grok_bridges[session_id] = bridge
    try:
        bridge.start()
    except Exception as e:
        with _grok_bridges_lock:
            _grok_bridges.pop(session_id, None)
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
    with _grok_bridges_lock:
        bridge = _grok_bridges.pop(session_id, None)
        owner = _grok_wake_owners.pop(session_id, None)
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
    queues (SPEC §4.1)."""
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


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
