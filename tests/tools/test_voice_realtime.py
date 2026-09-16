"""Tests for the xAI realtime (S2S) voice input backend (tools.voice_realtime).

Covers both brains: ears (VAD + transcription into Hermes turns) and
supervisor (grok-voice + consult/steer). These tests drive the session
with a fake WebSocket + fake mic — no network, no audio hardware.
"""

import base64
import json
import queue
import threading
import time

import pytest

from tools.voice_realtime import RealtimeVoiceSession
from tools.voice_realtime_config import (
    DEFAULT_REALTIME_KEYTERMS,
    DEFAULT_REALTIME_MODEL,
    FRAME_MS,
    FRAME_SAMPLES,
    INPUT_SAMPLE_RATE,
    RealtimeConfig,
    RealtimeVoiceError,
    build_session_update,
    load_realtime_config,
    realtime_voice_enabled,
)

# Original function object, bound BEFORE the autouse fixture patches the
# module attribute — lets requirement tests exercise the real logic.
from tools.voice_realtime import check_realtime_requirements as _real_check_requirements


def _wait_until(cond, timeout=5.0, interval=0.02):
    """Poll ``cond`` until truthy or timeout; returns the final verdict."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestLoadRealtimeConfig:
    def test_defaults_from_empty_config(self):
        cfg = load_realtime_config({})
        assert cfg.model == DEFAULT_REALTIME_MODEL
        assert cfg.brain == "supervisor" and cfg.supervisor
        assert cfg.vad_threshold is None
        assert cfg.vad_silence_ms is None
        assert cfg.language_hint == ""
        assert cfg.keyterms == []
        assert cfg.idle_pause_seconds == 120.0
        assert cfg.discord_text_mirror is False

    def test_non_dict_voice_section_is_safe(self):
        # A hand-edited `voice: true` must not crash config loading.
        cfg = load_realtime_config(True)
        assert cfg.model == DEFAULT_REALTIME_MODEL

    def test_shipped_defaults_reproduce_the_dataclass_defaults(self):
        """DEFAULT_CONFIG's realtime block must be a complete, faithful
        spelling of RealtimeConfig: every setting the loader reads is
        declared there, and loading the shipped defaults changes nothing."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        shipped = DEFAULT_CONFIG["voice"]["realtime"]
        loaded = load_realtime_config(DEFAULT_CONFIG["voice"])
        assert loaded == RealtimeConfig()
        # Loader-only knobs (plus the enable flag and surface toggles that
        # are not RealtimeConfig fields) all have a documented default.
        for key in (
            "enabled", "model", "url", "brain", "voice", "instructions",
            "full_duplex", "vad_threshold", "vad_silence_ms",
            "vad_prefix_padding_ms", "language_hint", "keyterms",
            "idle_pause_seconds", "require_wake_name", "wake_names",
            "discord_text_mirror", "discord", "narrate_progress",
        ):
            assert key in shipped, key

    def test_overrides_applied(self):
        cfg = load_realtime_config({
            "realtime": {
                "model": "grok-voice-think-fast-2.0",
                "vad_threshold": 0.5,
                "vad_silence_ms": 1200,
                "vad_prefix_padding_ms": 500,
                "language_hint": "ja",
                "keyterms": ["Hermes", " Nous ", ""],
                "idle_pause_seconds": 0,
            },
        })
        assert cfg.model == "grok-voice-think-fast-2.0"
        assert cfg.vad_threshold == 0.5
        assert cfg.vad_silence_ms == 1200
        assert cfg.vad_prefix_padding_ms == 500
        assert cfg.language_hint == "ja"
        assert cfg.keyterms == ["Hermes", "Nous"]
        assert cfg.idle_pause_seconds == 0

    def test_yaml_shape_corruption_falls_back(self):
        # bool is an int subclass — `vad_threshold: true` must not become 1.0.
        cfg = load_realtime_config({
            "realtime": {
                "vad_threshold": True,
                "vad_silence_ms": "soon",
                "keyterms": "not-a-list",
            },
        })
        assert cfg.vad_threshold is None
        assert cfg.vad_silence_ms is None
        assert cfg.keyterms == []

    def test_out_of_range_threshold_rejected(self):
        cfg = load_realtime_config({"realtime": {"vad_threshold": 3.0}})
        assert cfg.vad_threshold is None

    def test_discord_text_mirror_default_false(self):
        assert load_realtime_config({}).discord_text_mirror is False
        assert load_realtime_config({"realtime": {}}).discord_text_mirror is False

    def test_discord_text_mirror_opt_in(self):
        cfg = load_realtime_config({"realtime": {"discord_text_mirror": True}})
        assert cfg.discord_text_mirror is True


class TestRealtimeVoiceEnabled:
    def test_disabled_by_default(self):
        assert realtime_voice_enabled({}) is False
        assert realtime_voice_enabled({"realtime": {}}) is False
        assert realtime_voice_enabled(None) is False
        assert realtime_voice_enabled({"realtime": "yes"}) is False

    def test_enabled(self):
        assert realtime_voice_enabled({"realtime": {"enabled": True}}) is True


# ---------------------------------------------------------------------------
# session.update payload
# ---------------------------------------------------------------------------

class TestBuildSessionUpdate:
    def test_full_payload_shape(self):
        payload = build_session_update(RealtimeConfig(brain="ears"))
        assert payload["type"] == "session.update"
        session = payload["session"]
        td = session["turn_detection"]
        assert td["type"] == "server_vad"
        # The relay must never answer: best-effort OpenAI-compat suppression.
        assert td["create_response"] is False
        assert session["reasoning"] == {"effort": "none"}
        audio_in = session["audio"]["input"]
        assert audio_in["format"] == {"type": "audio/pcm", "rate": INPUT_SAMPLE_RATE}
        assert audio_in["transcription"]["model"] == "grok-transcribe"
        assert "instructions" in session and session["instructions"]

    def test_vad_tuning_forwarded_only_when_set(self):
        payload = build_session_update(RealtimeConfig(
            vad_threshold=0.6, vad_silence_ms=900, vad_prefix_padding_ms=250,
        ))
        td = payload["session"]["turn_detection"]
        assert td["threshold"] == 0.6
        assert td["silence_duration_ms"] == 900
        assert td["prefix_padding_ms"] == 250

        bare = build_session_update(RealtimeConfig(brain="ears"))["session"]["turn_detection"]
        assert "threshold" not in bare
        assert "silence_duration_ms" not in bare
        assert "prefix_padding_ms" not in bare

    def test_language_hint_and_keyterms(self):
        payload = build_session_update(RealtimeConfig(
            language_hint="es-MX", keyterms=["Hermes", "Nous"],
        ))
        tr = payload["session"]["audio"]["input"]["transcription"]
        assert tr["language_hint"] == "es-MX"
        assert tr["keyterms"] == [*DEFAULT_REALTIME_KEYTERMS, "Nous"]

    def test_keyterms_respect_xai_per_term_and_count_caps(self):
        """xAI rejects a term over its per-term cap with an ``error`` event, which would spend
        the session's one-shot minimal-config retry on a payload that stays invalid — so every
        term the payload carries, from config or a caller merge, fits the cap."""
        from tools.voice_realtime_config import (
            MAX_KEYTERM_CHARS,
            MAX_KEYTERMS,
            load_realtime_config,
            merge_realtime_keyterms,
        )

        long_term = "x" * (MAX_KEYTERM_CHARS * 3)
        cfg = load_realtime_config({"realtime": {"keyterms": [long_term, "Nous"]}})
        payload = build_session_update(cfg)
        terms = payload["session"]["audio"]["input"]["transcription"]["keyterms"]
        assert all(len(t) <= MAX_KEYTERM_CHARS for t in terms)
        assert "Nous" in terms

        merged = merge_realtime_keyterms(
            [long_term], [f"term{i}" for i in range(MAX_KEYTERMS + 20)],
        )
        assert len(merged) <= MAX_KEYTERMS
        assert all(len(t) <= MAX_KEYTERM_CHARS for t in merged)

    def test_minimal_variant_drops_compat_extras_keeps_core(self):
        payload = build_session_update(
            RealtimeConfig(brain="ears", vad_threshold=0.7), minimal=True,
        )
        session = payload["session"]
        assert "create_response" not in session["turn_detection"]
        assert "reasoning" not in session
        # Core VAD + transcription survive the downgrade.
        assert session["turn_detection"]["type"] == "server_vad"
        assert session["turn_detection"]["threshold"] == 0.7
        assert session["audio"]["input"]["transcription"]["model"] == "grok-transcribe"

    def test_supervisor_payload_shape(self):
        from tools.voice_realtime_config import (
            CONSULT_TOOL_NAME,
            OUTPUT_SAMPLE_RATE,
            STEER_TOOL_NAME,
        )

        payload = build_session_update(RealtimeConfig(
            brain="supervisor", voice="ara", instructions_extra="Speak like a pirate.",
        ))
        session = payload["session"]
        # grok answers in supervisor mode — auto-responses must NOT be muted.
        assert "create_response" not in session["turn_detection"]
        assert session["voice"] == "ara"
        assert session["audio"]["output"]["format"]["rate"] == OUTPUT_SAMPLE_RATE
        tools = session["tools"]
        assert [t["name"] for t in tools] == [CONSULT_TOOL_NAME, STEER_TOOL_NAME]
        assert "task" in tools[0]["parameters"]["properties"]
        assert "instruction" in tools[1]["parameters"]["properties"]
        assert "NOT queued" in session["instructions"]
        assert "steer_hermes" in tools[0]["description"]
        assert CONSULT_TOOL_NAME in session["instructions"]
        assert STEER_TOOL_NAME in session["instructions"]
        assert "Speak like a pirate." in session["instructions"]
        # Input half is identical to ears mode: server VAD + transcription.
        assert session["turn_detection"]["type"] == "server_vad"
        assert session["audio"]["input"]["transcription"]["model"] == "grok-transcribe"

    def test_brain_config_parsing(self):
        """supervisor is the default; ears is the explicit opt-in; anything else keeps the default."""
        assert load_realtime_config({}).supervisor
        assert load_realtime_config({"realtime": {"brain": "supervisor"}}).supervisor
        assert load_realtime_config({"realtime": {"brain": "nonsense"}}).supervisor
        assert not load_realtime_config({"realtime": {"brain": "ears"}}).supervisor
        assert load_realtime_config({"realtime": {"brain": " Ears "}}).brain == "ears"
        cfg = load_realtime_config({"realtime": {"voice": " rex "}})
        assert cfg.voice == "rex"


# ---------------------------------------------------------------------------
# Session behavior — fake WS + fake mic, no network / audio hardware
# ---------------------------------------------------------------------------

class _FakeClosed(Exception):
    pass


_CLOSE = object()


class FakeWS:
    def __init__(self):
        self.sent = []
        self._inbox = queue.Queue()
        self.closed = False

    def send(self, payload):
        if self.closed:
            raise _FakeClosed("send on closed ws")
        self.sent.append(payload)

    def push(self, event: dict):
        self._inbox.put(json.dumps(event))

    def recv(self):
        item = self._inbox.get()
        if item is _CLOSE:
            raise _FakeClosed("connection closed")
        return item

    def close(self):
        self.closed = True
        self._inbox.put(_CLOSE)

    # test helpers -----------------------------------------------------
    def sent_events(self):
        return [json.loads(p) for p in self.sent]

    def sent_types(self):
        return [e.get("type") for e in self.sent_events()]


class FakeMic:
    instances = []

    def __init__(self, on_frame):
        self.on_frame = on_frame
        self.closed = False
        FakeMic.instances.append(self)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_mics():
    FakeMic.instances = []
    yield
    FakeMic.instances = []


@pytest.fixture(autouse=True)
def _fast_reconnect(monkeypatch):
    monkeypatch.setattr("tools.voice_realtime.RECONNECT_DELAYS", (0.01, 0.01))


@pytest.fixture(autouse=True)
def _fake_creds(monkeypatch):
    monkeypatch.setattr(
        "tools.voice_realtime.check_realtime_requirements",
        lambda **_kw: (True, ""),
    )
    import tools.xai_http as xai_http
    monkeypatch.setattr(
        xai_http, "resolve_xai_http_credentials",
        lambda **kw: {"api_key": "xai-test-key"},
    )


class _Harness:
    """Session + captured callbacks, wired to FakeWS/FakeMic."""

    def __init__(self, cfg=None, gate=None, hold=None, connect_fn=None):
        self.ws = FakeWS()
        self.transcripts = []
        self.speech_started = 0
        self.speech_stopped = 0
        self.states = []
        self.idle_pauses = 0
        self._gate = gate if gate is not None else (lambda: True)
        connect = connect_fn or (lambda url, headers: self.ws)
        self.connect_urls = []

        def _tracking_connect(url, headers):
            self.connect_urls.append((url, headers))
            return connect(url, headers)

        self.session = RealtimeVoiceSession(
            cfg or RealtimeConfig(brain="ears", idle_pause_seconds=0),
            on_transcript=self.transcripts.append,
            on_speech_started=self._inc_started,
            on_speech_stopped=self._inc_stopped,
            on_state=lambda s, d: self.states.append((s, d)),
            on_idle_pause=self._inc_idle,
            input_gate=lambda: self._gate(),
            activity_hold=hold,
            connect_fn=_tracking_connect,
            mic_factory=FakeMic,
        )

    def _inc_started(self):
        self.speech_started += 1

    def _inc_stopped(self):
        self.speech_stopped += 1

    def _inc_idle(self):
        self.idle_pauses += 1

    def start_and_connect(self):
        self.session.start()
        assert _wait_until(lambda: self.session.connected), "session never connected"
        return self.session


class TestSessionLifecycle:
    def test_net_thread_runs_in_the_callers_context(self, monkeypatch):
        """Credential resolution happens on the net thread; it must see the
        ContextVars (profile secret scope) active when start() was called,
        otherwise multiplexed profiles fall back to the default key."""
        import contextvars

        import tools.xai_http as xai_http

        scope = contextvars.ContextVar("test_profile_scope", default="unset")
        seen = []

        def _creds(**_kw):
            seen.append(scope.get())
            return {"api_key": "xai-test-key"}

        monkeypatch.setattr(xai_http, "resolve_xai_http_credentials", _creds)
        h = _Harness()
        token = scope.set("secondary-profile")
        try:
            h.start_and_connect()
        finally:
            scope.reset(token)
            h.session.stop()
        assert seen == ["secondary-profile"]

    def test_connects_sends_session_update_and_reports_state(self):
        h = _Harness()
        try:
            h.start_and_connect()
            assert ("connected", "") in h.states
            events = h.ws.sent_events()
            assert events, "nothing sent on connect"
            assert events[0]["type"] == "session.update"
            assert events[0]["session"]["turn_detection"]["type"] == "server_vad"
            url, headers = h.connect_urls[0]
            assert "model=" + DEFAULT_REALTIME_MODEL in url
            assert headers["Authorization"] == "Bearer xai-test-key"
        finally:
            h.session.stop()

    def test_start_requires_requirements(self, monkeypatch):
        monkeypatch.setattr(
            "tools.voice_realtime.check_realtime_requirements",
            lambda **_kw: (False, "nope"),
        )
        h = _Harness()
        with pytest.raises(RealtimeVoiceError):
            h.session.start()

    def test_stop_closes_mic_and_socket(self):
        h = _Harness()
        h.start_and_connect()
        h.session.stop()
        assert FakeMic.instances[0].closed
        assert h.ws.closed
        assert not h.session.alive

    def test_stop_during_blocked_connect_aborts_the_late_socket(self):
        """stop() while connect() is still dialling: when the dial finally returns, the socket
        must be closed unused — no session.update, no buffered audio, no "connected" state —
        and the net thread must exit instead of serving a session the surface tore down."""
        released = threading.Event()

        def _blocked_connect(url, headers):
            released.wait(timeout=5)
            return h.ws

        h = _Harness(connect_fn=_blocked_connect)
        h.session.start()
        h.session.set_armed(True)
        h.session._enqueue_frame(b"\x01\x01" * FRAME_SAMPLES)
        assert _wait_until(lambda: len(h.session._prebuffer) == 1)
        stopper = threading.Thread(target=h.session.stop)
        stopper.start()
        assert _wait_until(h.session._stop.is_set)
        states_at_stop = list(h.states)
        released.set()
        stopper.join(timeout=5)
        assert not stopper.is_alive()
        assert _wait_until(lambda: not h.session._net_thread.is_alive())
        assert h.states == states_at_stop
        assert ("connected", "") not in h.states
        assert h.ws.sent == []
        assert h.ws.closed

    def test_full_queue_drops_oldest_frame_not_newest(self):
        h = _Harness()
        h.session._frames = queue.Queue(maxsize=2)
        h.session._enqueue_frame(b"a")
        h.session._enqueue_frame(b"b")
        h.session._enqueue_frame(b"c")
        assert h.session._frames.get_nowait() == b"b"
        assert h.session._frames.get_nowait() == b"c"

    def test_reconnects_after_connection_loss(self):
        h = _Harness()
        try:
            h.start_and_connect()
            first_ws = h.ws
            h.ws = FakeWS()  # next connect gets a fresh socket
            first_ws.close()
            assert _wait_until(lambda: h.session.connected and h.ws.sent)
            assert any(s == "reconnecting" for s, _ in h.states)
            assert h.ws.sent_events()[0]["type"] == "session.update"
        finally:
            h.session.stop()

    def test_goes_dead_after_exhausting_retries(self):
        def _always_fail(url, headers):
            raise OSError("connect refused")

        h = _Harness(connect_fn=_always_fail)
        h.session.start()
        assert _wait_until(lambda: any(s == "dead" for s, _ in h.states))
        assert not h.session.alive

    def test_accept_then_close_server_backs_off_and_goes_dead(self):
        """A server that accepts the socket and closes it at once (rejected
        session, quota) must not be redialed in a tight loop forever: each
        short-lived connection spends a retry, then the session is dead."""
        connects = []

        class _HangsUpAfterHandshake(FakeWS):
            def recv(self):  # session.update was accepted; server then closes
                raise _FakeClosed("connection closed")

        def _accept_then_close(url, headers):
            ws = _HangsUpAfterHandshake()
            connects.append(ws)
            return ws

        h = _Harness(connect_fn=_accept_then_close)
        h.session.start()
        assert _wait_until(lambda: any(s == "dead" for s, _ in h.states))
        assert not h.session.alive
        # Initial connect + one redial per RECONNECT_DELAYS entry (patched to 2).
        assert len(connects) == 3


class TestSessionEvents:
    def _armed_session(self, **kw):
        h = _Harness(**kw)
        h.start_and_connect()
        h.session.set_armed(True)
        return h

    def test_transcript_delivered_when_armed(self):
        h = self._armed_session()
        try:
            h.ws.push({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "  list the files  ",
            })
            assert _wait_until(lambda: h.transcripts == ["list the files"])
        finally:
            h.session.stop()

    def _sync_events(self, h):
        """Barrier: wait until every already-pushed event has been handled.

        The recv loop is strictly ordered, so once the cancel for this
        response.created shows up, everything pushed before it was processed.
        """
        marker = len([t for t in h.ws.sent_types() if t == "response.cancel"]) + 1
        h.ws.push({"type": "response.created", "response": {"id": f"sync{marker}"}})
        assert _wait_until(
            lambda: h.ws.sent_types().count("response.cancel") >= marker
        ), "event sync barrier never completed"

    def test_transcript_dropped_when_disarmed_or_empty(self):
        h = self._armed_session()
        try:
            h.session.set_armed(False)
            h.ws.push({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "ignored",
            })
            self._sync_events(h)  # "ignored" fully processed while disarmed
            h.session.set_armed(True)
            h.ws.push({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "   ",
            })
            h.ws.push({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "real",
            })
            assert _wait_until(lambda: h.transcripts == ["real"])
        finally:
            h.session.stop()

    def test_transcript_dropped_when_gate_closed(self):
        gate_open = threading.Event()
        h = self._armed_session(gate=gate_open.is_set)
        try:
            h.ws.push({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "speaker bleed",
            })
            self._sync_events(h)  # bleed processed while the gate was closed
            gate_open.set()
            h.ws.push({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "real",
            })
            assert _wait_until(lambda: h.transcripts == ["real"])
        finally:
            h.session.stop()

    def test_speech_events_fire_when_armed_and_gated(self):
        h = self._armed_session()
        try:
            h.ws.push({"type": "input_audio_buffer.speech_started"})
            h.ws.push({"type": "input_audio_buffer.speech_stopped"})
            assert _wait_until(
                lambda: h.speech_started == 1 and h.speech_stopped == 1
            )
        finally:
            h.session.stop()

    def test_response_created_is_cancelled(self):
        """The relay never speaks: auto-responses are cancelled immediately."""
        h = self._armed_session()
        try:
            h.ws.push({"type": "response.created", "response": {"id": "r1"}})
            assert _wait_until(lambda: "response.cancel" in h.ws.sent_types())
        finally:
            h.session.stop()

    def test_error_event_downgrades_session_config_once(self):
        h = self._armed_session()
        try:
            h.ws.push({"type": "error", "error": {"message": "unknown field"}})
            h.ws.push({"type": "error", "error": {"message": "again"}})
            self._sync_events(h)  # both errors fully processed

            updates = [
                e for e in h.ws.sent_events() if e["type"] == "session.update"
            ]
            # Initial full config + exactly one minimal downgrade — the second
            # error must not spam further updates.
            assert len(updates) == 2
            minimal = updates[1]["session"]
            assert "create_response" not in minimal["turn_detection"]
            assert "reasoning" not in minimal
        finally:
            h.session.stop()

    def test_disarm_clears_server_buffer(self):
        h = self._armed_session()
        try:
            h.session.set_armed(False)
            assert "input_audio_buffer.clear" in h.ws.sent_types()
        finally:
            h.session.stop()


class TestAudioPump:
    def test_frames_stream_as_base64_appends(self):
        h = _Harness()
        try:
            h.start_and_connect()
            h.session.set_armed(True)
            frame = b"\x01\x02" * 160
            FakeMic.instances[0].on_frame(frame)

            def _decoded_appends():
                return [
                    base64.b64decode(e["audio"])
                    for e in h.ws.sent_events()
                    if e["type"] == "input_audio_buffer.append"
                ]

            assert _wait_until(lambda: frame in _decoded_appends())
        finally:
            h.session.stop()

    def test_frames_dropped_when_gate_closed(self):
        gate_open = threading.Event()
        h = _Harness(gate=gate_open.is_set)
        try:
            h.start_and_connect()
            h.session.set_armed(True)
            # RMS updates for every consumed frame regardless of gating, so a
            # loud frame doubles as the "frame 1 was processed" barrier.
            loud = (b"\x00\x40") * 320
            FakeMic.instances[0].on_frame(loud)
            assert _wait_until(lambda: h.session.current_rms > 1000)
            gate_open.set()
            marker = b"\x07\x07" * 160
            FakeMic.instances[0].on_frame(marker)

            def _decoded_appends():
                return [
                    base64.b64decode(e["audio"])
                    for e in h.ws.sent_events()
                    if e["type"] == "input_audio_buffer.append"
                ]

            assert _wait_until(lambda: marker in _decoded_appends())
            assert loud not in _decoded_appends()
        finally:
            h.session.stop()

    def test_rms_meter_updates(self):
        h = _Harness()
        try:
            h.start_and_connect()
            h.session.set_armed(True)
            loud = (b"\x00\x40") * 320  # int16 samples of 16384
            FakeMic.instances[0].on_frame(loud)
            assert _wait_until(lambda: h.session.current_rms > 1000)
        finally:
            h.session.stop()

    def _decoded_appends(self, h):
        return [
            base64.b64decode(e["audio"])
            for e in h.ws.sent_events()
            if e.get("type") == "input_audio_buffer.append"
        ]

    def test_idle_mic_streams_silence_pcm_when_armed(self):
        """No Discord RTP (empty mic queue) still feeds VAD with zero PCM."""
        h = _Harness()
        silence = b"\x00" * (FRAME_SAMPLES * 2)
        try:
            h.start_and_connect()
            h.session.set_armed(True)

            def _silence_appends():
                return [pcm for pcm in self._decoded_appends(h) if pcm == silence]

            assert _wait_until(lambda: len(_silence_appends()) >= 2)
            for pcm in _silence_appends():
                assert pcm == silence
                assert len(pcm) == FRAME_SAMPLES * 2
        finally:
            h.session.stop()

    def test_no_silence_when_disarmed(self):
        h = _Harness()
        try:
            h.start_and_connect()
            # Pump Empty cycles happen at FRAME_MS; wait a few before asserting.
            time.sleep(0.35)
            assert self._decoded_appends(h) == []
        finally:
            h.session.stop()

    def test_jittered_mic_frames_are_not_interleaved_with_silence(self):
        """A local mic delivers one frame per FRAME_MS with callback jitter.
        A late frame is not silence: exactly the mic's frames reach the
        socket, in order, with no zero frames spliced between them."""
        h = _Harness()
        silence = b"\x00" * (FRAME_SAMPLES * 2)
        try:
            h.start_and_connect()
            h.session.set_armed(True)
            mic = FakeMic.instances[0]
            frames = [bytes([i + 1]) * (FRAME_SAMPLES * 2) for i in range(12)]
            period = FRAME_MS / 1000.0
            next_at = time.monotonic()
            for i, frame in enumerate(frames):
                # ±4 ms jitter around the nominal period, like PortAudio.
                next_at += period + (0.004 if i % 2 else -0.004)
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                mic.on_frame(frame)
            assert _wait_until(lambda: len(self._decoded_appends(h)) >= len(frames))
            sent = self._decoded_appends(h)
            assert silence not in sent
            assert sent == frames
        finally:
            h.session.stop()

    def test_gated_tail_is_dropped_when_playback_ends_without_barge(self):
        """Frames captured behind the closed gate while speech plays are the
        assistant's own bleed; without a barge they must not be sent."""
        gate_open = threading.Event()  # the surface closes its gate during playback
        h = _Harness(
            cfg=RealtimeConfig(brain="supervisor", idle_pause_seconds=0),
            gate=gate_open.is_set,
        )
        try:
            h.start_and_connect()
            h.session.set_armed(True)
            sess = h.session
            mic = FakeMic.instances[0]
            sess._playing = True
            for _ in range(3):
                mic.on_frame(b"\x01\x01" * FRAME_SAMPLES)
            assert _wait_until(lambda: len(sess._gated_tail) == 3)
            assert self._decoded_appends(h) == []
            sess._playing = False  # playback finished normally, no barge
            gate_open.set()
            spoken = b"\x02\x02" * FRAME_SAMPLES
            mic.on_frame(spoken)
            assert _wait_until(lambda: self._decoded_appends(h) == [spoken])
            assert len(sess._gated_tail) == 0
        finally:
            h.session.stop()

    def test_no_silence_prebuffer_when_ws_is_none(self):
        released = threading.Event()

        def _blocked_connect(url, headers):
            released.wait(timeout=5)
            return FakeWS()

        h = _Harness(connect_fn=_blocked_connect)
        try:
            h.session.start()
            h.session.set_armed(True)
            time.sleep(0.35)
            assert h.session._ws is None
            assert list(h.session._prebuffer) == []
        finally:
            released.set()
            h.session.stop()

    def test_no_silence_when_gate_closed(self):
        gate_open = threading.Event()
        h = _Harness(gate=gate_open.is_set)
        try:
            h.start_and_connect()
            h.session.set_armed(True)
            time.sleep(0.35)
            assert self._decoded_appends(h) == []
        finally:
            h.session.stop()


class TestSupervisorBrain:
    def _supervisor_session(self):
        h = _Harness(cfg=RealtimeConfig(brain="supervisor", idle_pause_seconds=0))
        h.function_calls = []
        h.session._on_function_call = (
            lambda name, call_id, args: h.function_calls.append((name, call_id, args))
        )
        h.start_and_connect()
        h.session.set_armed(True)
        return h

    def test_responses_are_not_cancelled(self):
        h = self._supervisor_session()
        try:
            h.ws.push({"type": "response.created", "response": {"id": "r1"}})
            h.ws.push({"type": "input_audio_buffer.speech_started"})
            assert _wait_until(lambda: h.speech_started == 1)
            assert "response.cancel" not in h.ws.sent_types()
        finally:
            h.session.stop()

    def test_audio_deltas_are_queued_for_playback(self):
        h = self._supervisor_session()
        played = []
        h.session._enqueue_playout = played.append
        try:
            pcm = b"\x01\x02" * 10
            h.ws.push({
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(pcm).decode(),
            })
            assert _wait_until(lambda: played == [pcm])
        finally:
            h.session.stop()

    def test_function_call_event_dispatches(self):
        h = self._supervisor_session()
        try:
            h.ws.push({
                "type": "response.function_call_arguments.done",
                "name": "consult_hermes",
                "call_id": "c1",
                "arguments": json.dumps({"task": "list files"}),
            })
            assert _wait_until(lambda: len(h.function_calls) == 1)
            name, call_id, args = h.function_calls[0]
            assert (name, call_id) == ("consult_hermes", "c1")
            assert json.loads(args)["task"] == "list files"
        finally:
            h.session.stop()

    def test_send_function_output_delivers_result_then_response(self):
        h = self._supervisor_session()
        try:
            h.session.send_function_output("c9", "done: 3 files")

            def _types():
                return h.ws.sent_types()

            assert _wait_until(
                lambda: "conversation.item.create" in _types()
                and "response.create" in _types()
            )
            items = [
                e for e in h.ws.sent_events()
                if e["type"] == "conversation.item.create"
            ]
            assert items[0]["item"] == {
                "type": "function_call_output",
                "call_id": "c9",
                "output": "done: 3 files",
            }
            # Result lands before the follow-up response request.
            assert _types().index("conversation.item.create") < _types().index("response.create")
        finally:
            h.session.stop()

    def test_speak_verbatim_uses_force_message(self):
        h = self._supervisor_session()
        try:
            assert h.session.speak_verbatim("Running tests.") is True
            forced = [
                e for e in h.ws.sent_events()
                if e["type"] == "conversation.item.create"
                and e["item"].get("type") == "force_message"
            ]
            assert len(forced) == 1
            assert forced[0]["item"]["content"] == [
                {"type": "output_text", "text": "Running tests."}
            ]
        finally:
            h.session.stop()

    def test_speak_verbatim_refused_in_ears_mode(self):
        h = _Harness()
        h.start_and_connect()
        try:
            assert h.session.speak_verbatim("nope") is False
        finally:
            h.session.stop()

    def test_response_audio_flag_tracks_current_response(self):
        h = self._supervisor_session()
        try:
            h.ws.push({"type": "response.created", "response": {"id": "r1"}})
            h.ws.push({
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\x00\x01").decode(),
            })
            self._sync(h)
            assert h.session.last_response_had_audio is True
            # A new response resets the flag — a silent tool call must be
            # detectable even right after a spoken reply.
            h.ws.push({"type": "response.created", "response": {"id": "r2"}})
            self._sync(h)
            assert h.session.last_response_had_audio is False
        finally:
            h.session.stop()

    def _sync(self, h):
        """Barrier: server events are processed strictly in order, so a
        unique sentinel proves everything pushed before it was handled."""
        token = f"sync-{time.monotonic_ns()}"
        h.ws.push({
            "type": "response.function_call_arguments.done",
            "name": "sync_barrier",
            "call_id": token,
            "arguments": "{}",
        })
        assert _wait_until(
            lambda: any(call_id == token for _, call_id, _args in h.function_calls)
        ), "supervisor event sync barrier never completed"

    def test_speak_acknowledgment_is_instant_force_message(self):
        from tools.voice_realtime_config import _ACK_PHRASES

        h = self._supervisor_session()
        try:
            h.session.speak_acknowledgment()
            forced = [
                e for e in h.ws.sent_events()
                if e["type"] == "conversation.item.create"
                and e["item"].get("type") == "force_message"
            ]
            # Sent synchronously — no model turn, no deferred thread.
            assert len(forced) == 1
            assert forced[0]["item"]["content"][0]["text"] in _ACK_PHRASES
        finally:
            h.session.stop()

    def test_speech_started_clears_queued_playout(self):
        h = self._supervisor_session()
        try:
            h.session._playout_q.put(b"\x00\x01")
            h.ws.push({"type": "input_audio_buffer.speech_started"})
            assert _wait_until(lambda: h.session._playout_q.empty())
        finally:
            h.session.stop()


class TestLoudBarge:
    def _half_duplex_session(self):
        h = _Harness(cfg=RealtimeConfig(brain="supervisor", idle_pause_seconds=0))
        return h

    def test_sustained_loud_speech_over_playback_triggers_barge(self):
        h = self._half_duplex_session()
        sess = h.session
        sess._playing = True
        sess._active_response = True
        sess._playout_q.put(b"\x00\x01")
        sent = []
        sess._send_event = lambda event: sent.append(event) or True
        # Calibrate the bleed floor with quiet playback frames.
        sess._current_rms = 300
        for _ in range(5):
            sess._update_barge_detector()
        assert not sess.barge_active
        # User talks clearly over it — two consecutive hot frames trip it.
        sess._current_rms = 300 * 10
        sess._update_barge_detector()
        assert not sess.barge_active  # one frame isn't enough (noise spikes)
        sess._update_barge_detector()
        assert sess.barge_active
        assert sess._playout_q.empty()  # speech was cut
        assert {"type": "response.cancel"} in sent

    def test_bleed_alone_never_triggers(self):
        h = self._half_duplex_session()
        sess = h.session
        sess._playing = True
        sess._current_rms = 800
        for _ in range(50):
            sess._update_barge_detector()
        assert not sess.barge_active

    def test_detector_inactive_in_full_duplex_and_ears(self):
        for cfg in (
            RealtimeConfig(brain="supervisor", full_duplex=True),
            RealtimeConfig(brain="ears"),
        ):
            h = _Harness(cfg=cfg)
            sess = h.session
            sess._playing = True
            sess._current_rms = 32000
            for _ in range(5):
                sess._update_barge_detector()
            assert not sess.barge_active

    def test_floor_recalibrates_between_playbacks(self):
        h = self._half_duplex_session()
        sess = h.session
        sess._playing = True
        sess._current_rms = 500
        sess._update_barge_detector()
        assert sess._bleed_floor >= 500
        sess._playing = False
        sess._update_barge_detector()
        assert sess._bleed_floor == 0.0


class TestIdlePause:
    def test_idle_pause_disarms_and_notifies(self):
        h = _Harness(cfg=RealtimeConfig(brain="ears", idle_pause_seconds=5))
        h.session._last_voice_activity = time.monotonic() - 999
        h.session._armed.set()
        h.session._check_idle_pause()
        assert h.idle_pauses == 1
        assert not h.session.armed

    def test_activity_hold_blocks_idle_pause(self):
        h = _Harness(cfg=RealtimeConfig(brain="ears", idle_pause_seconds=5), hold=lambda: True)
        h.session._last_voice_activity = time.monotonic() - 999
        h.session._armed.set()
        h.session._check_idle_pause()
        assert h.idle_pauses == 0
        assert h.session.armed

    def test_zero_disables_idle_pause(self):
        h = _Harness(cfg=RealtimeConfig(brain="ears", idle_pause_seconds=0))
        h.session._last_voice_activity = time.monotonic() - 999
        h.session._armed.set()
        h.session._check_idle_pause()
        assert h.idle_pauses == 0
        assert h.session.armed

    def test_speech_events_refresh_idle_clock(self):
        h = _Harness(cfg=RealtimeConfig(brain="ears", idle_pause_seconds=5))
        h.session._last_voice_activity = time.monotonic() - 999
        h.session._armed.set()
        h.session._handle_event({"type": "input_audio_buffer.speech_started"})
        h.session._check_idle_pause()
        assert h.idle_pauses == 0
        assert h.session.armed


# ---------------------------------------------------------------------------
# Requirements — non-local surfaces (Discord VC) skip the sounddevice check
# ---------------------------------------------------------------------------

class TestRequireLocalAudio:
    def test_missing_sounddevice_blocks_only_local_surfaces(self, monkeypatch):
        import sys

        pytest.importorskip("numpy")
        pytest.importorskip("websockets")
        # None in sys.modules makes `import sounddevice` raise ImportError.
        monkeypatch.setitem(sys.modules, "sounddevice", None)

        ok_local, detail = _real_check_requirements()
        assert ok_local is False
        assert "sounddevice" in detail

        ok_remote, detail_remote = _real_check_requirements(require_local_audio=False)
        assert ok_remote is True, detail_remote


# ---------------------------------------------------------------------------
# Buffering playout sinks (Discord mixer): clear() + pending() protocol
# ---------------------------------------------------------------------------

class _BufferingSink:
    """Fake sink modelling the Discord mixer: writes accumulate in a buffer
    that keeps 'playing' (pending) until an external drain empties it."""

    def __init__(self):
        self.writes = []
        self.cleared = 0
        self.closed = False
        self._pending = False
        self.lock = threading.Lock()

    def write(self, chunk):
        with self.lock:
            self.writes.append(chunk)
            self._pending = True

    def clear(self):
        with self.lock:
            self.cleared += 1
            self._pending = False

    def pending(self):
        with self.lock:
            return self._pending

    def drain(self):
        with self.lock:
            self._pending = False

    def set_active(self, active):
        pass

    def close(self):
        self.closed = True


class TestBufferingSinkProtocol:
    def _session_with_sink(self):
        sink = _BufferingSink()
        h = _Harness(cfg=RealtimeConfig(brain="supervisor", idle_pause_seconds=0))
        h.session._playout_sink_factory = lambda: sink
        return h, sink

    def test_speaking_covers_sink_drain_and_clear_reaches_sink(self):
        h, sink = self._session_with_sink()
        try:
            h.session._enqueue_playout(b"\x00\x01")
            assert _wait_until(lambda: sink.writes), "sink never received audio"
            # Queue is empty but the sink still holds audio — the session is
            # audibly speaking (send_function_output's quiet-wait depends on
            # this to not talk over the in-flight answer).
            assert _wait_until(lambda: h.session._playout_q.empty())
            assert h.session.speaking is True
            # Barge-in path: clear_playout must clear the sink buffer too.
            h.session.clear_playout()
            assert sink.cleared == 1
            assert _wait_until(lambda: h.session.speaking is False)
        finally:
            h.session.stop()

    def test_playing_flag_holds_until_sink_drains(self):
        h, sink = self._session_with_sink()
        try:
            h.session._enqueue_playout(b"\x00\x01")
            assert _wait_until(lambda: sink.writes)
            assert _wait_until(lambda: h.session._playing is True)
            # Queue empty + sink pending → still playing.
            time.sleep(0.6)  # a few playout-loop idle cycles
            assert h.session._playing is True
            sink.drain()
            assert _wait_until(lambda: h.session._playing is False)
        finally:
            h.session.stop()
