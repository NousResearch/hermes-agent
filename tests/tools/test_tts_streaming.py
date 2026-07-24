"""Tests for the provider-agnostic streaming TTS backend (tools.tts_streaming)
and its dispatch through tools.tts_tool.stream_tts_to_speaker.

No live audio or network: the ElevenLabs/OpenAI SDKs, sounddevice, and the sync
synth path are all mocked. Covers the registry/resolver, provider availability,
the chunked-streamer playback path, and the universal per-sentence sync fallback.
"""

import queue
import threading
from unittest.mock import ANY, MagicMock, patch

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


def test_elevenlabs_available_requires_key_and_sdk(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(ts, "get_env_value", lambda k, *a: "key" if k == "ELEVENLABS_API_KEY" else None)
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: MagicMock())
    assert ts.ElevenLabsStreamer.available() is True
    monkeypatch.setattr(ts, "get_env_value", lambda k, *a: None)
    assert ts.ElevenLabsStreamer.available() is False


def test_elevenlabs_unavailable_when_sdk_import_fails(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(ts, "get_env_value", lambda k, *a: "key" if k == "ELEVENLABS_API_KEY" else None)
    monkeypatch.setattr(
        tts_tool,
        "_import_elevenlabs",
        MagicMock(side_effect=ImportError("missing elevenlabs")),
    )

    assert ts.ElevenLabsStreamer.available() is False


def test_openai_available_requires_resolved_audio_credentials_and_sdk(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: MagicMock())
    monkeypatch.setattr(
        tts_tool,
        "_resolve_openai_audio_client_config",
        lambda: ("voice-key", "https://api.openai.com/v1", False),
    )
    assert ts.OpenAIStreamer.available() is True


def test_openai_unavailable_when_sdk_import_fails(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(
        tts_tool,
        "_resolve_openai_audio_client_config",
        lambda: ("voice-key", "https://api.openai.com/v1", False),
    )
    monkeypatch.setattr(
        tts_tool,
        "_import_openai_client",
        MagicMock(side_effect=ImportError("missing openai")),
    )

    assert ts.OpenAIStreamer.available() is False


def test_openai_unavailable_when_audio_credentials_cannot_resolve(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: MagicMock())
    monkeypatch.setattr(
        tts_tool,
        "_resolve_openai_audio_client_config",
        MagicMock(side_effect=ValueError("missing credentials")),
    )

    assert ts.OpenAIStreamer.available() is False


def test_openai_stream_uses_resolved_audio_credentials_and_config_base_url(monkeypatch):
    from tools import tts_tool

    client_class = MagicMock()
    client = client_class.return_value
    response = client.audio.speech.with_streaming_response.create.return_value
    response.__enter__.return_value.iter_bytes.return_value = iter([b"pcm"])
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: client_class)
    monkeypatch.setattr(
        tts_tool,
        "_resolve_openai_audio_client_config",
        lambda: ("voice-key", "https://resolved.example/v1", False),
    )
    streamer = ts.OpenAIStreamer(
        {},
        {
            "base_url": "https://configured.example/v1",
            "model": "custom-tts",
            "voice": "nova",
        },
    )

    assert list(streamer.stream("hello")) == [b"pcm"]
    client_class.assert_called_once_with(
        api_key="voice-key",
        base_url="https://configured.example/v1",
    )
    client.audio.speech.with_streaming_response.create.assert_called_once_with(
        model="custom-tts",
        voice="nova",
        input="hello",
        response_format="pcm",
        extra_headers={"x-idempotency-key": ANY},
    )
    client.close.assert_called_once_with()


def test_openai_stream_keeps_managed_token_on_managed_gateway(monkeypatch):
    from tools import tts_tool

    client_class = MagicMock()
    client = client_class.return_value
    response = client.audio.speech.with_streaming_response.create.return_value
    response.__enter__.return_value.iter_bytes.return_value = iter([b"pcm"])
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: client_class)
    monkeypatch.setattr(
        tts_tool,
        "_resolve_openai_audio_client_config",
        lambda: ("managed-token", "https://managed.example/v1", True),
    )
    streamer = ts.OpenAIStreamer(
        {},
        {
            "base_url": "https://untrusted.example/v1",
            "model": "unsupported-model",
        },
    )

    assert list(streamer.stream("hello")) == [b"pcm"]
    client_class.assert_called_once_with(
        api_key="managed-token",
        base_url="https://managed.example/v1",
    )
    request = client.audio.speech.with_streaming_response.create
    assert request.call_args.kwargs["model"] == tts_tool.DEFAULT_OPENAI_MODEL
    assert request.call_args.kwargs["extra_headers"]["x-idempotency-key"]
    client.close.assert_called_once_with()


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
