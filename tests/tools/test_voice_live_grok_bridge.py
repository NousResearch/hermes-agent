"""Grok-Live realtime bridge: lifecycle, reconnect, AEC gate, backpressure, protocol contract.

Everything runs against FAKE websockets (no network, no real ~/.hermes — the credential
resolver is monkeypatched). The bridge owns a real thread + asyncio loop per instance; the
fakes are driven from the test thread through ``loop.call_soon_threadsafe``. Every emitted
``voice.grok.*`` payload is validated against the declared contract models
(``tui_gateway/contracts/prompt_voice_grok.py``) so the renderer's generated TS types and the
emitter cannot drift apart.
"""

import asyncio
import base64
import json
import time

import pytest

import tools.voice_live_grok as grok_config
from tools.voice_live_grok_bridge import (
    DEGRADED_DROP_STREAK, EVENT_AUDIO, EVENT_DELEGATION, EVENT_STATE, EVENT_TRANSCRIPT,
    GrokLiveBridge, LEAD_IN_MAX_CHUNKS, build_delegation, realtime_url,
    session_update_payload)
from tui_gateway.contracts.prompt_voice_grok import (
    VoiceGrokAudioPayload, VoiceGrokDelegationPayload, VoiceGrokStatePayload,
    VoiceGrokTranscriptPayload)

END = object()  # end-of-stream sentinel for the fake receiver queue

EVENT_MODELS = {
    EVENT_AUDIO: VoiceGrokAudioPayload,
    EVENT_TRANSCRIPT: VoiceGrokTranscriptPayload,
    EVENT_STATE: VoiceGrokStatePayload,
    EVENT_DELEGATION: VoiceGrokDelegationPayload,
}

LIVE_CONFIG = {
    "model": "grok-voice-latest", "voice": "eve", "auth": "auto", "instructions": "",
    "vad_threshold": 0.6, "silence_ms": 500, "prefix_ms": 250, "speed": 1.1,
}


class FakeWS:
    """Stand-in for a ``websockets`` connection: records sent frames, replays pushed events."""

    def __init__(self):
        self.sent = []
        self.loop = None
        self.queue = None
        self.closed = False
        self.send_gate = None  # optional asyncio.Event: send() blocks until set

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    def push(self, msg):
        """Thread-safe event injection: str (JSON event), bytes (audio), Exception, or END."""
        self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    def push_event(self, event: dict):
        self.push(json.dumps(event))

    def binary_frames(self):
        return [f for f in self.sent if isinstance(f, (bytes, bytearray))]

    def json_frames(self):
        return [json.loads(f) for f in self.sent if isinstance(f, str)]

    async def send(self, data):
        if self.send_gate is not None:
            await self.send_gate.wait()
        self.sent.append(data)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        while True:
            msg = await self.queue.get()
            if msg is END:
                return
            if isinstance(msg, Exception):
                raise msg
            yield msg


class FakeConnector:
    """``connect_factory`` stand-in: one Fresh FakeWS per call, url/headers recorded."""

    def __init__(self):
        self.calls = []
        self.sockets = []

    def __call__(self, url, headers):
        ws = FakeWS()
        self.calls.append((url, headers))
        self.sockets.append(ws)
        return ws


def wait_for(cond, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = cond()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")


def states(events):
    return [p["state"] for t, p in events if t == EVENT_STATE]


@pytest.fixture
def credential(monkeypatch):
    """The bridge resolves credentials through tools.voice_live_grok — patched, never disk."""
    monkeypatch.setattr(grok_config, "_resolve_grok_credentials", lambda live=None: "test-token")


@pytest.fixture
def harness():
    """Bridge factory + event capture; every bridge is stopped on teardown."""
    events = []
    bridges = []

    def emit(event_type, payload):
        # Protocol contract: the emitter must always match the declared payload model.
        EVENT_MODELS[event_type].model_validate(payload)
        events.append((event_type, payload))

    def make(**kwargs):
        kwargs.setdefault("session_id", "sess-1")
        kwargs.setdefault("emit", emit)
        kwargs.setdefault("live_config", dict(LIVE_CONFIG))
        kwargs.setdefault("reconnect_delay_s", 0.05)
        kwargs.setdefault("settle_s", 0.05)
        bridge = GrokLiveBridge(**kwargs)
        bridges.append(bridge)
        return bridge

    yield events, make
    for bridge in bridges:
        bridge.stop()


def establish(connector, events, make, socket_index=0, **bridge_kwargs):
    """Start a bridge and drive it to the ``listening`` state; returns the FakeWS."""
    bridge = make(connect_factory=connector, **bridge_kwargs)
    bridge.start()
    ws = wait_for(lambda: connector.sockets[socket_index]
                  if len(connector.sockets) > socket_index else None)
    wait_for(lambda: any("session.update" in f for f in ws.sent if isinstance(f, str)))
    ws.push_event({"type": "session.created", "session": {"id": "xai-sess"}})
    ws.push_event({"type": "session.updated"})
    wait_for(lambda: "listening" in states(events))
    return bridge, ws


# ── session.update payload (SPEC §4.2) ─────────────────────────────────────────


def test_session_update_payload_comes_from_config_and_has_no_tools():
    payload = session_update_payload(LIVE_CONFIG)
    assert payload["type"] == "session.update"
    session = payload["session"]
    assert session["reasoning"] == {"effort": "none"}
    assert session["turn_detection"] == {
        "type": "server_vad", "threshold": 0.6,
        "silence_duration_ms": 500, "prefix_padding_ms": 250}
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["input"]["transport"] == "binary"
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["speed"] == 1.1
    assert session["voice"] == "eve"
    assert "Hermes" in session["instructions"]  # the short persona, never the system prompt
    assert "tools" not in session  # decision #6: zero local function tools in v1


def test_persona_appends_extra_instructions():
    payload = session_update_payload({**LIVE_CONFIG, "instructions": "Speak Lao first."})
    assert payload["session"]["instructions"].endswith("Speak Lao first.")


def test_realtime_url_carries_the_model():
    assert realtime_url("grok-voice-latest") == "wss://api.x.ai/v1/realtime?model=grok-voice-latest"


# ── lifecycle (SPEC §9) ─────────────────────────────────────────────────────────


def test_start_handshake_reaches_listening(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)

    url, headers = connector.calls[0]
    assert url == "wss://api.x.ai/v1/realtime?model=grok-voice-latest"
    assert headers == {"Authorization": "Bearer test-token"}  # OAuth-first resolution, §8
    update = next(f for f in ws.json_frames() if f["type"] == "session.update")
    assert update["session"]["voice"] == "eve"
    assert states(events)[:2] == ["connecting", "listening"]
    assert bridge.status()["established"] is True


def test_start_without_credential_fails_fast_and_never_retries(monkeypatch, harness):
    monkeypatch.setattr(grok_config, "_resolve_grok_credentials", lambda live=None: "")
    events, make = harness
    connector = FakeConnector()
    bridge = make(connect_factory=connector)
    bridge.start()
    wait_for(lambda: "error" in states(events))
    time.sleep(0.15)  # a retry would show up as another connect call + reconnecting state
    assert connector.calls == []  # fail-fast BEFORE opening the socket
    assert "reconnecting" not in states(events)
    error = next(p for t, p in events if t == EVENT_STATE and p["state"] == "error")
    assert "no xAI credential" in error["reason"]
    assert not bridge.alive


def test_xai_error_during_handshake_is_a_start_failure_not_a_retry(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge = make(connect_factory=connector)
    bridge.start()
    ws = wait_for(lambda: connector.sockets[0] if connector.sockets else None)
    wait_for(lambda: any("session.update" in f for f in ws.sent if isinstance(f, str)))
    ws.push_event({"type": "error", "error": {"message": "invalid authorization"}})
    wait_for(lambda: "error" in states(events))
    time.sleep(0.15)
    assert len(connector.sockets) == 1  # SPEC §9: explicit start failure is NOT retried
    assert "reconnecting" not in states(events)
    assert not bridge.alive


def test_stop_closes_the_socket_and_emits_idle(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    assert bridge.stop() is True
    wait_for(lambda: "idle" in states(events))
    assert ws.closed
    assert not bridge.alive


# ── reconnect (SPEC §10) ─────────────────────────────────────────────────────────


def test_receiver_exit_is_fatal_and_reconnects_the_same_session(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws1 = establish(connector, events, make)
    ws1.push(END)  # remote ended the session without a user stop
    wait_for(lambda: "reconnecting" in states(events))
    ws2 = wait_for(lambda: connector.sockets[1] if len(connector.sockets) > 1 else None)
    # Fresh handshake on the same logical session: session.update is re-sent (xAI has no
    # resumable-session concept, SPEC §10).
    wait_for(lambda: any("session.update" in f for f in ws2.sent if isinstance(f, str)))
    assert connector.calls[1][0] == connector.calls[0][0]
    ws2.push_event({"type": "session.updated"})
    wait_for(lambda: states(events)[-1] == "listening")
    assert bridge.status()["reconnects"] == 1
    assert bridge.session_id == "sess-1"


def test_user_stop_during_the_reconnect_wait_cancels_the_retry(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws1 = establish(connector, events, make, reconnect_delay_s=30.0)
    ws1.push(END)
    wait_for(lambda: "reconnecting" in states(events))
    assert bridge.stop(timeout=5.0) is True  # must not wait out the 30s retry delay
    assert states(events)[-1] == "idle"
    assert len(connector.sockets) == 1


# ── upstream relay: AEC gate (SPEC §7) + mute + backpressure (SPEC §4.1) ────────


def test_mic_chunks_relay_as_binary_frames(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    chunk = b"\x01\x00" * 2400
    assert bridge.send_mic(chunk) == {"accepted": True}
    wait_for(lambda: ws.binary_frames() == [chunk])


def test_aec_gate_withholds_mic_during_playback_plus_tail(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make, echo_tail_s=0.2)
    first = b"\x01\x00" * 2400
    bridge.send_mic(first)
    wait_for(lambda: ws.binary_frames() == [first])

    ws.push(b"\x02\x00" * 2400)  # Grok starts speaking → gate closes
    wait_for(lambda: "speaking" in states(events))
    bridge.send_mic(b"\x03\x00" * 2400)  # accepted but must NOT reach xAI (echo)
    time.sleep(0.1)
    assert ws.binary_frames() == [first]

    time.sleep(0.25)  # past the 0.2s echo tail → gate re-opens
    bridge.send_mic(b"\x04\x00" * 2400)
    wait_for(lambda: len(ws.binary_frames()) == 2)


def test_explicit_mute_withholds_independently_of_the_gate(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    assert bridge.set_muted(True) is True
    assert bridge.send_mic(b"\x05\x00" * 2400) == {"accepted": True}
    time.sleep(0.1)
    assert ws.binary_frames() == []
    bridge.set_muted(False)
    bridge.send_mic(b"\x06\x00" * 2400)
    wait_for(lambda: len(ws.binary_frames()) == 1)


def test_send_mic_before_handshake_parks_in_lead_in_then_drains(credential, harness):
    """Connect-time audio is buffered (bounded), not dropped, and reaches xAI once
    the session is established — the wake-word user speaks from the moment the
    bridge starts, often during xAI's slow opening handshake."""
    events, make = harness
    connector = FakeConnector()
    bridge = make(connect_factory=connector)
    bridge.start()
    wait_for(lambda: connector.sockets)
    assert bridge.send_mic(b"\x00" * 100) == {"accepted": True, "reason": "lead_in"}
    # bounded: past LEAD_IN_MAX_CHUNKS it degrades to drop-not-queue
    for _ in range(LEAD_IN_MAX_CHUNKS + 5):
        bridge.send_mic(b"\x00" * 100)
    result = bridge.send_mic(b"\x00" * 100)
    assert result == {"accepted": False, "dropped": True, "reason": "backpressure"}
    # established: the parked audio drains into the mic queue
    bridge._established = True  # simulate session.updated without a real socket
    with bridge._state_lock:
        drained = list(bridge._lead_in)
    assert len(drained) == LEAD_IN_MAX_CHUNKS


def test_backpressure_drops_never_queues_and_surfaces_degraded(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    ws.send_gate = asyncio.Event()  # xAI leg is stuck: the pump parks inside send()

    bridge.send_mic(b"\x01\x00" * 2400)  # pump takes it and blocks in send()
    for _ in range(3):  # fill the 300ms queue (3 x 100ms chunks)
        assert bridge.send_mic(b"\x02\x00" * 2400) == {"accepted": True}
    wait_for(lambda: bridge._mic_q.qsize() == 3)

    drops = [bridge.send_mic(b"\x03\x00" * 2400) for _ in range(DEGRADED_DROP_STREAK)]
    assert all(d == {"accepted": False, "dropped": True, "reason": "backpressure"} for d in drops)
    wait_for(lambda: "degraded" in states(events))
    assert bridge.status()["dropped_chunks"] == DEGRADED_DROP_STREAK
    degraded = [p for t, p in events if t == EVENT_STATE and p["state"] == "degraded"]
    assert len(degraded) == 1  # surfaced once, not per drop

    # The xAI leg recovers: queued audio flushes, the streak resets, the base state returns.
    bridge._loop.call_soon_threadsafe(ws.send_gate.set)
    wait_for(lambda: len(ws.binary_frames()) == 4)
    wait_for(lambda: states(events)[-1] == "listening")
    bridge.send_mic(b"\x04\x00" * 2400)
    wait_for(lambda: len(ws.binary_frames()) == 5)


# ── downstream relay + events (SPEC §4.1/§4.2) ──────────────────────────────────


def test_downstream_audio_relays_as_event_and_drives_speaking_state(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    ws.push(b"\x07\x00" * 2400)
    wait_for(lambda: any(t == EVENT_AUDIO for t, _ in events))
    audio = next(p for t, p in events if t == EVENT_AUDIO)
    assert base64.b64decode(audio["pcm_b64"]) == b"\x07\x00" * 2400
    assert audio["seq"] == 1 and audio["session_id"] == "sess-1"
    assert states(events)[-1] == "speaking"
    ws.push_event({"type": "response.done"})
    wait_for(lambda: states(events)[-1] == "listening")


def test_base64_delta_event_shape_also_relays(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    payload = b"\x08\x00" * 2400
    ws.push_event({"type": "response.output_audio.delta",
                   "delta": base64.b64encode(payload).decode()})
    wait_for(lambda: any(t == EVENT_AUDIO for t, _ in events))
    audio = next(p for t, p in events if t == EVENT_AUDIO)
    assert base64.b64decode(audio["pcm_b64"]) == payload


def test_function_call_events_are_logged_and_ignored(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    ws.push_event({"type": "response.function_call_arguments.done",
                   "name": "get_system_status", "call_id": "c1"})
    time.sleep(0.1)  # must not crash the session (SPEC §4.2: v1 registers no tools)
    assert bridge.alive
    assert "error" not in states(events)
    assert states(events)[-1] == "listening"


# ── delegation (SPEC §5 revised: renderer-submits — event carries prompt+context) ─


def test_utterance_settles_into_a_delegation_event_with_prompt_and_context(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    ws.push_event({"type": "conversation.item.input_audio_transcription.updated",
                   "item_id": "i1", "delta": "what's "})
    ws.push_event({"type": "conversation.item.input_audio_transcription.updated",
                   "item_id": "i1", "delta": "the time"})
    ws.push_event({"type": "input_audio_buffer.speech_stopped"})
    wait_for(lambda: any(t == EVENT_DELEGATION for t, _ in events))

    delegation = next(p for t, p in events if t == EVENT_DELEGATION)
    assert delegation["session_id"] == "sess-1"
    assert delegation["delegation_id"].startswith("grok-")
    # prompt = the user's last words (the turn text the renderer submits), context = the
    # recent spoken exchange (voice_context, model input only).
    assert delegation["prompt"] == "what's the time"
    assert "User: what's the time" in delegation["context"]
    assert "thinking" in states(events)
    # Transcript fragments stream to the UI as they arrive, not only at flush time.
    user_fragments = [p["text"] for t, p in events
                      if t == EVENT_TRANSCRIPT and p["speaker"] == "user"]
    assert user_fragments == ["what's ", "the time"]


def test_completed_transcript_finalizes_the_item_and_assistant_turns_join_context(
        credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    ws.push_event({"type": "response.output_audio_transcript.done",
                   "item_id": "a1", "transcript": "It is noon."})
    ws.push_event({"type": "conversation.item.input_audio_transcription.completed",
                   "item_id": "i1", "transcript": "thanks"})
    ws.push_event({"type": "input_audio_buffer.speech_stopped"})
    wait_for(lambda: any(t == EVENT_DELEGATION for t, _ in events))
    delegation = next(p for t, p in events if t == EVENT_DELEGATION)
    assert delegation["prompt"] == "thanks"
    assert delegation["context"] == "Voice assistant: It is noon.\nUser: thanks"
    speakers = [(p["speaker"], p["text"]) for t, p in events if t == EVENT_TRANSCRIPT]
    assert ("assistant", "It is noon.") in speakers


def test_silence_flush_without_transcript_emits_no_delegation(credential, harness):
    events, make = harness
    connector = FakeConnector()
    bridge, ws = establish(connector, events, make)
    ws.push_event({"type": "input_audio_buffer.speech_stopped"})
    time.sleep(0.15)
    assert not any(t == EVENT_DELEGATION for t, _ in events)


# ── delegationPrompt parity (pure function) ──────────────────────────────────────


def test_build_delegation_merges_same_speaker_runs_like_the_renderer():
    fragments = [
        {"speaker": "user", "text": "hi "},
        {"speaker": "user", "text": "there"},
        {"speaker": "assistant", "text": "hello"},
        {"speaker": "user", "text": "what   time\nis it"},
    ]
    delegation = build_delegation(fragments)
    assert delegation["prompt"] == "what time is it"
    assert delegation["context"] == (
        "User: hi there\nVoice assistant: hello\nUser: what time is it")


def test_build_delegation_falls_back_to_context_tail_without_a_user_turn():
    delegation = build_delegation([{"speaker": "assistant", "text": "only the assistant spoke"}])
    assert delegation["prompt"] == "Voice assistant: only the assistant spoke"
