"""Tests for the provider-agnostic streaming TTS backend (tools.tts_streaming)
and its dispatch through tools.tts_tool.stream_tts_to_speaker.

No live audio or network: the ElevenLabs/OpenAI SDKs, sounddevice, and the sync
synth path are all mocked. Covers the registry/resolver, provider availability,
the chunked-streamer playback path, and the universal per-sentence sync fallback.
"""

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

import tools.tts_streaming as ts

pytest.importorskip("numpy")


# ── SentenceChunker ──────────────────────────────────────────────────────


class TestSentenceChunker:
    def test_cuts_sentence_the_moment_its_boundary_arrives(self):
        c = ts.SentenceChunker()
        assert c.feed("This is the first full") == []
        assert c.feed(" sentence of it all. And") == ["This is the first full sentence of it all. "]
        assert c.flush() == ["And"]

    def test_short_fragment_rides_with_the_next_sentence(self):
        c = ts.SentenceChunker()
        # "Ha! " alone is under min_len — it must not become its own clip.
        assert c.feed("Ha! ") == []
        assert c.feed("That was a good one, honestly. ") == [
            "Ha! That was a good one, honestly. "
        ]

    def test_think_blocks_are_stripped_even_across_deltas(self):
        c = ts.SentenceChunker()
        assert c.feed("<think>secret reason") == []
        assert c.feed("ing</think>The actual spoken answer. ") == ["The actual spoken answer. "]

    def test_flush_drains_the_tail(self):
        c = ts.SentenceChunker()
        c.feed("no boundary here")
        assert c.flush() == ["no boundary here"]
        assert c.flush() == []

    def test_paragraph_break_is_a_boundary(self):
        c = ts.SentenceChunker()
        assert c.feed("A paragraph without punctuation\n\nnext one") == [
            "A paragraph without punctuation\n\n"
        ]


# ── Interruption latch ───────────────────────────────────────────────────


class TestSpeechInterruptedLatch:
    def test_take_pops_and_reports_recent_barge(self):
        ts.mark_speech_interrupted()
        assert ts.take_speech_interrupted() is True
        assert ts.take_speech_interrupted() is False  # one-shot

    def test_untouched_latch_is_false(self):
        ts._interrupted_at = None
        assert ts.take_speech_interrupted() is False

    def test_stale_barge_expires(self, monkeypatch):
        ts.mark_speech_interrupted()
        at = ts._interrupted_at
        monkeypatch.setattr(ts.time, "monotonic", lambda: at + ts._INTERRUPT_TTL_S + 1)
        assert ts.take_speech_interrupted() is False


# ── Registry + resolver ──────────────────────────────────────────────────


def _register_fake(monkeypatch, name, available=True, chunks=(b"\x00\x00",)):
    class _Fake(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return available

        def stream(self, text):
            yield from chunks

    monkeypatch.setitem(ts._REGISTRY, name, _Fake)
    return _Fake


def test_resolve_returns_configured_streamer(monkeypatch):
    _register_fake(monkeypatch, "faketts")
    prov = ts.resolve_streaming_provider({"provider": "faketts"})
    assert isinstance(prov, ts.StreamingTTSProvider)


def test_resolve_none_for_unregistered_provider(monkeypatch):
    # edge is a sync provider — not registered — so the dispatcher keeps its voice.
    assert ts.resolve_streaming_provider({"provider": "edge"}) is None


def test_resolve_none_when_provider_unavailable(monkeypatch):
    _register_fake(monkeypatch, "faketts", available=False)
    assert ts.resolve_streaming_provider({"provider": "faketts"}) is None


def test_resolve_honors_preferred_override(monkeypatch):
    _register_fake(monkeypatch, "faketts")
    prov = ts.resolve_streaming_provider({"provider": "edge"}, preferred="faketts")
    assert isinstance(prov, ts.StreamingTTSProvider)


def test_never_swaps_provider_for_streaming(monkeypatch):
    # A registered streamer must NOT be substituted when the user picked another
    # (non-streaming) provider — that would silently change their voice.
    _register_fake(monkeypatch, "elevenlabs")
    assert ts.resolve_streaming_provider({"provider": "edge"}) is None


# ── Built-in provider availability ───────────────────────────────────────


def test_elevenlabs_available_reflects_key(monkeypatch):
    # Key lookups now route through the provider-secret resolver
    # (config > env/.env > credential pool), not bare get_env_value.
    monkeypatch.setattr(ts, "_resolve_key", lambda env, pid: "key" if env == "ELEVENLABS_API_KEY" else "")
    assert ts.ElevenLabsStreamer.available() is True
    monkeypatch.setattr(ts, "_resolve_key", lambda env, pid: "")
    assert ts.ElevenLabsStreamer.available() is False


def test_openai_available_reflects_audio_key_resolution(monkeypatch):
    monkeypatch.setattr(ts, "_openai_config_api_key", lambda: "")
    monkeypatch.setattr(ts, "resolve_openai_audio_api_key", lambda: "voice-key")
    assert ts.OpenAIStreamer.available() is True
    monkeypatch.setattr(ts, "resolve_openai_audio_api_key", lambda: "")
    assert ts.OpenAIStreamer.available() is False
    # tts.openai.api_key from config.yaml counts too
    monkeypatch.setattr(ts, "_openai_config_api_key", lambda: "cfg-key")
    assert ts.OpenAIStreamer.available() is True


def test_openai_streamer_prefers_configured_base_url(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self):
            yield b"\x01\x00"

    class _StreamingCreate:
        @staticmethod
        def create(**kwargs):
            captured["request"] = kwargs
            return _Response()

    class _OpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.audio = MagicMock()
            self.audio.speech.with_streaming_response = _StreamingCreate()

    monkeypatch.setattr(ts, "resolve_openai_audio_api_key", lambda: "voice-key")
    monkeypatch.setattr(
        ts,
        "get_env_value",
        lambda key, *args: "https://env.example/v1" if key == "OPENAI_BASE_URL" else None,
    )
    monkeypatch.setattr("openai.OpenAI", _OpenAI)

    config = {
        "provider": "openai",
        "openai": {
            "base_url": "http://local-tts.example/v1",
            "model": "tts-1",
            "voice": "local-voice",
        },
    }
    streamer = ts.resolve_streaming_provider(config)

    assert streamer is not None
    assert list(streamer.stream("Streaming test.")) == [b"\x01\x00"]
    assert captured["client"] == {
        "api_key": "voice-key",
        "base_url": "http://local-tts.example/v1",
    }


def test_openai_streamer_prefers_configured_api_key(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self):
            yield b"\x01\x00"

    class _StreamingCreate:
        @staticmethod
        def create(**kwargs):
            return _Response()

    class _OpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.audio = MagicMock()
            self.audio.speech.with_streaming_response = _StreamingCreate()

    monkeypatch.setattr(ts, "resolve_openai_audio_api_key", lambda: "env-key")
    monkeypatch.setattr(ts, "get_env_value", lambda key, *args: None)
    monkeypatch.setattr("openai.OpenAI", _OpenAI)

    config = {
        "provider": "openai",
        "openai": {"api_key": "cfg-key", "base_url": "http://local-tts.example/v1"},
    }
    streamer = ts.resolve_streaming_provider(config)

    assert streamer is not None
    assert list(streamer.stream("Streaming test.")) == [b"\x01\x00"]
    assert captured["client"]["api_key"] == "cfg-key"


# ── Dispatch: chunked streamer path ──────────────────────────────────────


def _drain_queue(sentences):
    q = queue.Queue()
    for s in sentences:
        q.put(s)
    q.put(None)
    return q


def _sd_mock():
    sd = MagicMock()
    out = MagicMock()
    sd.OutputStream.return_value = out
    return sd, out


def test_streamer_path_writes_pcm_to_output(monkeypatch):
    from tools import tts_tool

    class _Fake(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            yield b"\x01\x00" * 50
            yield b"\x02\x00" * 50

    sd, out = _sd_mock()
    q = _drain_queue(["Hello there, this is a full sentence."])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=_Fake({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert out.write.called, "expected PCM chunks written to the output stream"
    assert done.is_set()


def test_stop_event_aborts_streaming(monkeypatch):
    from tools import tts_tool

    class _Fake(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            for _ in range(1000):
                yield b"\x00\x00" * 50

    sd, out = _sd_mock()
    stop, done = threading.Event(), threading.Event()
    stop.set()  # pre-set: no audio should be written
    q = _drain_queue(["A complete sentence here."])

    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=_Fake({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert not out.write.called
    assert done.is_set()


# ── Dispatch: universal per-sentence sync fallback ───────────────────────


def test_sync_fallback_speaks_each_sentence(monkeypatch):
    from tools import tts_tool

    spoken = []
    monkeypatch.setattr(tts_tool, "text_to_speech_tool",
                        lambda text, output_path: spoken.append(text))
    played = []
    fake_vm = MagicMock()
    fake_vm.play_audio_file.side_effect = lambda p: played.append(p)
    monkeypatch.setitem(__import__("sys").modules, "tools.voice_mode", fake_vm)
    monkeypatch.setattr("os.path.getsize", lambda p: 100)
    monkeypatch.setattr("os.path.isfile", lambda p: True)

    q = _drain_queue(["First full sentence here. ", "Second full sentence here. "])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=None):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert len(spoken) == 2, f"expected both sentences synthesized, got {spoken}"
    assert len(played) == 2
    assert done.is_set()


def test_display_callback_fires_without_audio(monkeypatch):
    from tools import tts_tool

    seen = []
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", lambda text, output_path: None)
    q = _drain_queue(["A sentence to display aloud."])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=None):
        tts_tool.stream_tts_to_speaker(q, stop, done, display_callback=seen.append)

    assert seen, "display_callback should fire even on the sync path"
    assert done.is_set()


# ── tts.streaming.provider config knob (salvaged from PR #47588) ─────────


def test_streaming_provider_knob_pins_streamer(monkeypatch):
    _register_fake(monkeypatch, "faketts")
    prov = ts.resolve_streaming_provider(
        {"provider": "edge", "streaming": {"provider": "faketts"}}
    )
    assert isinstance(prov, ts.StreamingTTSProvider)


def test_streaming_provider_knob_pin_unusable_returns_none(monkeypatch):
    _register_fake(monkeypatch, "faketts", available=False)
    # A pinned-but-unusable streamer must NOT fall through to another
    # provider — the user asked for that one specifically.
    _register_fake(monkeypatch, "otherstreamer")
    assert ts.resolve_streaming_provider(
        {"provider": "otherstreamer", "streaming": {"provider": "faketts"}}
    ) is None


def test_streaming_provider_auto_walks_priority(monkeypatch):
    # elevenlabs unavailable, gemini available → auto resolves gemini.
    _register_fake(monkeypatch, "elevenlabs", available=False)
    fake_gemini = _register_fake(monkeypatch, "gemini")
    _register_fake(monkeypatch, "openai")
    prov = ts.resolve_streaming_provider(
        {"provider": "edge", "streaming": {"provider": "auto"}}
    )
    assert isinstance(prov, fake_gemini)


def test_streaming_provider_auto_none_when_nothing_usable(monkeypatch):
    for name in ts._PROVIDER_PRIORITY:
        _register_fake(monkeypatch, name, available=False)
    assert ts.resolve_streaming_provider(
        {"provider": "edge", "streaming": {"provider": "auto"}}
    ) is None


# ── Credential routing: resolve_provider_secret, never bare env ──────────


def test_elevenlabs_available_routes_through_secret_resolver(monkeypatch):
    calls = []

    def _fake_resolve(env_var, provider_id):
        calls.append((env_var, provider_id))
        return "pool-key"

    monkeypatch.setattr(ts, "_resolve_key", _fake_resolve)
    assert ts.ElevenLabsStreamer.available() is True
    assert ("ELEVENLABS_API_KEY", "elevenlabs") in calls


def test_gemini_available_falls_back_to_google_key(monkeypatch):
    keys = {"GOOGLE_API_KEY": "g-key"}
    monkeypatch.setattr(ts, "_resolve_key", lambda env, pid: keys.get(env, ""))
    assert ts.GeminiStreamer.available() is True
    keys.clear()
    assert ts.GeminiStreamer.available() is False


def test_xai_available_uses_oauth_credential_resolver(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("tools.xai_http")
    fake.resolve_xai_http_credentials = lambda: {"api_key": "xai-key"}
    monkeypatch.setitem(sys.modules, "tools.xai_http", fake)
    assert ts.XAIStreamer.available() is True
    fake.resolve_xai_http_credentials = lambda: {"api_key": ""}
    assert ts.XAIStreamer.available() is False


# ── Gemini SSE parsing ────────────────────────────────────────────────────


def test_gemini_streamer_decodes_sse_pcm_chunks(monkeypatch):
    import base64
    import json as _json
    import sys
    import types

    pcm1, pcm2 = b"\x01\x00" * 40, b"\x02\x00" * 40

    def _event(pcm):
        return "data: " + _json.dumps({
            "candidates": [{"content": {"parts": [
                {"inlineData": {"data": base64.b64encode(pcm).decode()}}
            ]}}]
        })

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=True):
            yield _event(pcm1)
            yield ""  # SSE separator
            yield ": heartbeat"  # comment line
            yield _event(pcm2)

    captured = {}

    def _post(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        captured["stream"] = kwargs.get("stream")
        return _Resp()

    fake_requests = types.ModuleType("requests")
    fake_requests.post = _post
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setattr(ts, "_resolve_key", lambda env, pid: "g-key")

    streamer = ts.GeminiStreamer({}, {"voice": "Kore"})
    assert list(streamer.stream("Hello there.")) == [pcm1, pcm2]
    assert captured["params"]["alt"] == "sse"
    assert captured["params"]["key"] == "g-key"
    assert captured["stream"] is True, "Gemini SSE must use a bounded streamed body"


# ── xAI WebSocket bridge ─────────────────────────────────────────────────


def test_xai_streamer_yields_collected_frames(monkeypatch):
    frames = [b"\x01\x00" * 30, b"\x02\x00" * 30]
    streamer = ts.XAIStreamer.__new__(ts.XAIStreamer)
    streamer.tts_config, streamer.section = {}, {}
    monkeypatch.setattr(streamer, "_collect_async", lambda text: list(frames))
    assert list(streamer.stream("A sentence.")) == frames


# ── 16 MiB per-sentence stream cap ───────────────────────────────────────


def test_stream_cap_truncates_runaway_upstream(monkeypatch):
    monkeypatch.setattr(ts, "_STREAM_SENTENCE_BYTE_CAP", 100)

    def _endless():
        while True:
            yield b"\x00" * 64

    out = list(ts._capped(_endless(), "test"))
    assert len(out) == 1  # 64 ok, 128 > cap → stop
    assert sum(len(c) for c in out) <= 100
