"""Grok-Live voice relay contracts (``methods_voice_grok.py``) — SPEC §4.1.

The backend holds the xAI realtime websocket exclusively; the renderer relays base64 PCM16
chunks (~100ms, 24kHz mono) inside ordinary JSON-RPC calls/events — the ``wake.feed``
precedent, NOT raw binary WS frames (``apps/shared``'s channel is JSON-RPC-only).

Events live here (not in ``events.py``) so the whole ``voice.grok.*`` surface is one file,
mirroring how ``server_requests.py`` keeps ``request.cancel`` next to its requests. Every
payload carries ``session_id`` per SPEC §4.1 even though the event envelope also stamps it —
the desktop's grok-live store keys everything off the owning Hermes session.
"""

from __future__ import annotations

from .base import Params, Payload, Result, WireEnum
from .registry import event, method

# ── methods ─────────────────────────────────────────────────────────────────────


class VoiceGrokStatusParams(Params):
    profile: str | None = None


class VoiceGrokStatusResult(Result):
    """``tools/voice_live_grok.py::resolve_grok_live_status`` — the non-secret readiness verdict
    (never the credential). ``available`` means a credential resolves per SPEC §8, not that the
    websocket will succeed."""

    mode: str  # "chained" | "gpt-live" | "grok-live"
    available: bool
    reason: str | None = None
    model: str
    voice: str


method("voice.grok.status", params=VoiceGrokStatusParams, result=VoiceGrokStatusResult,
       doc="Grok-Live availability verdict (mirrors GET /api/audio/voice-live-grok/status).")


class VoiceGrokStartParams(Params):
    session_id: str
    profile: str | None = None


class VoiceGrokStartResult(Result):
    """Fail-fast answer only: ``started: false`` + ``reason`` when no credential resolves. The
    connect handshake is async — progress arrives as ``voice.grok.state`` events."""

    started: bool
    reason: str | None = None
    state: str | None = None  # "connecting" on a fresh start
    reused: bool | None = None  # true when the session already had a live bridge


method("voice.grok.start", params=VoiceGrokStartParams, result=VoiceGrokStartResult,
       doc="Open (or reuse) the backend xAI realtime session; pauses the wake-word mic lease.")


class VoiceGrokStopParams(Params):
    session_id: str
    profile: str | None = None


class VoiceGrokStopResult(Result):
    stopped: bool
    reason: str | None = None  # "not_running" when the session had no live bridge


method("voice.grok.stop", params=VoiceGrokStopParams, result=VoiceGrokStopResult,
       doc="Close the xAI session cleanly (cancel any pending reconnect) and resume wake-word.")


class VoiceGrokAudioParams(Params):
    """One ~100ms upstream mic chunk (PCM16 mono 24kHz, base64) to relay to xAI."""

    session_id: str
    pcm_b64: str
    seq: int | None = None
    profile: str | None = None


class VoiceGrokAudioResult(Result):
    """``dropped: true`` = upstream backpressure (the bridge never queues stale live mic audio;
    SPEC §4.1)."""

    accepted: bool
    dropped: bool | None = None
    reason: str | None = None  # "not_running" | "connecting" | "backpressure"


method("voice.grok.audio", params=VoiceGrokAudioParams, result=VoiceGrokAudioResult,
       doc="Relay one ~100ms base64 PCM16 mic chunk to the backend-held xAI session.")


class VoiceGrokMuteParams(Params):
    session_id: str
    muted: bool
    profile: str | None = None


class VoiceGrokMuteResult(Result):
    muted: bool


method("voice.grok.mute", params=VoiceGrokMuteParams, result=VoiceGrokMuteResult,
       doc="Explicit mic mute, independent of the server-side AEC half-duplex gate (SPEC §7).")


class VoiceGrokSpeakParams(Params):
    """Renderer-submits design (SPEC §5, revised): the renderer is the sole ``prompt.submit``
    caller, so once its own turn settles it asks the bridge to speak the finished reply — the
    bridge itself never touches ``prompt.submit`` (single-submitter invariant)."""

    session_id: str
    text: str
    profile: str | None = None


class VoiceGrokSpeakResult(Result):
    spoken: bool
    reason: str | None = None  # "not_running" when the bridge has no live session for session_id


method("voice.grok.speak", params=VoiceGrokSpeakParams, result=VoiceGrokSpeakResult,
       doc="Make the bridge speak Hermes' finished reply verbatim (xAI force_message, SPEC §5 step 5).")


class VoiceGrokRekeyParams(Params):
    """Fresh-draft completion: the renderer's submit minted the chat session, so it re-keys the
    bridge from the synthetic start id onto the real Hermes session id — events and delegation
    route by session from then on. Idempotent."""

    from_session_id: str
    to_session_id: str
    profile: str | None = None


class VoiceGrokRekeyResult(Result):
    rekeyed: bool
    reason: str | None = None  # "not_running" when no bridge exists for from_session_id


method("voice.grok.rekey", params=VoiceGrokRekeyParams, result=VoiceGrokRekeyResult,
       doc="Re-key the voice bridge onto the chat's real Hermes session id (fresh-draft start).")


# ── events (server → client) ─────────────────────────────────────────────────────


class VoiceGrokState(WireEnum):
    connecting = "connecting"
    listening = "listening"
    speaking = "speaking"
    thinking = "thinking"
    reconnecting = "reconnecting"
    degraded = "degraded"  # overlay: upstream backpressure drop streak (SPEC §4.1)
    error = "error"
    idle = "idle"


class VoiceGrokAudioPayload(Payload):
    """``voice_live_grok_bridge._relay_downstream`` — one ~100ms of Grok's spoken reply for
    renderer playback. Never dropped server-side (a dropped reply chunk is an audible glitch)."""

    session_id: str
    pcm_b64: str
    seq: int


class VoiceGrokTranscriptPayload(Payload):
    """``voice_live_grok_bridge._emit_transcript`` — mirrors xAI
    ``conversation.item.input_audio_transcription.*`` / ``response.output_audio_transcript.done``
    for the transcript UI (same shape gpt-live's ``LiveTranscriptFragment`` carries)."""

    session_id: str
    speaker: str  # "user" | "assistant"
    text: str
    item_id: str | None = None


class VoiceGrokStatePayload(Payload):
    """``voice_live_grok_bridge._set_state`` — server-driven conversation status (the ws lives
    on the backend, so unlike gpt-live the renderer cannot derive this locally)."""

    session_id: str
    state: VoiceGrokState
    reason: str | None = None


class VoiceGrokDelegationPayload(Payload):
    """``voice_live_grok_bridge._flush_delegation`` — a settled spoken utterance that is a real
    request (SPEC §5, renderer-submits revision). ``prompt`` is the user's last words (the turn
    text the renderer submits — the persisted user row); ``context`` is the recent spoken
    exchange riding the model input only (``voice_context``). The renderer's ``onDelegation``
    is the SINGLE ``prompt.submit`` caller — the backend never submits on its behalf."""

    session_id: str
    delegation_id: str
    prompt: str
    context: str


event("voice.grok.audio", VoiceGrokAudioPayload,
      doc="One ~100ms of Grok's spoken reply audio for playback (base64 PCM16 24kHz).")
event("voice.grok.transcript", VoiceGrokTranscriptPayload,
      doc="A user/assistant transcript fragment from the xAI realtime session.")
event("voice.grok.state", VoiceGrokStatePayload,
      doc="Grok-Live conversation state transition (server-driven).")
event("voice.grok.delegation", VoiceGrokDelegationPayload,
      doc="A settled spoken utterance delegated to Hermes as a normal turn.")
