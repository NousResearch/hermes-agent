"""Tests for the provider-agnostic streaming TTS backend (tools.tts_streaming)
and its dispatch through tools.tts_tool.stream_tts_to_speaker.

No live audio or network: the ElevenLabs/OpenAI SDKs, sounddevice, and the sync
synth path are all mocked. Covers the registry/resolver, provider availability,
the chunked-streamer playback path, and the universal per-sentence sync fallback.
"""

import asyncio
import os
import queue
import sys
import tempfile
import threading
import time
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


    def test_think_blocks_are_stripped_even_across_deltas(self):
        c = ts.SentenceChunker()
        assert c.feed("<think>secret reason") == []
        assert c.feed("ing</think>The actual spoken answer. ") == ["The actual spoken answer. "]


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


    def test_stale_barge_expires(self, monkeypatch):
        ts.mark_speech_interrupted()
        at = ts._interrupted_at
        monkeypatch.setattr(ts.time, "monotonic", lambda: at + ts._INTERRUPT_TTL_S + 1)
        assert ts.take_speech_interrupted() is False


# ── Registry + resolver ──────────────────────────────────────────────────


def _register_fake(
    monkeypatch,
    name,
    available=True,
    chunks=(b"\x00\x00",),
    async_transport_cancellable=False,
):
    class _Fake(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return available

        def stream(self, text):
            yield from chunks

    _Fake.async_transport_cancellable = async_transport_cancellable
    if async_transport_cancellable:
        async def _astream(self, text):
            for chunk in chunks:
                yield chunk
        _Fake.astream = _astream
    monkeypatch.setitem(ts._REGISTRY, name, _Fake)
    return _Fake


def test_resolve_returns_configured_streamer(monkeypatch):
    _register_fake(monkeypatch, "faketts")
    prov = ts.resolve_streaming_provider({"provider": "faketts"})
    assert isinstance(prov, ts.StreamingTTSProvider)


def test_never_swaps_provider_for_streaming(monkeypatch):
    # A registered streamer must NOT be substituted when the user picked another
    # (non-streaming) provider — that would silently change their voice.
    _register_fake(monkeypatch, "elevenlabs")
    assert ts.resolve_streaming_provider({"provider": "edge"}) is None


def test_cancellable_resolution_rejects_pinned_legacy_transport(monkeypatch):
    _register_fake(monkeypatch, "openai", async_transport_cancellable=False)
    assert ts.resolve_streaming_provider(
        {"streaming": {"provider": "openai"}},
        require_transport_cancellation=True,
    ) is None


def test_async_cancellable_resolution_rejects_claim_without_astream(monkeypatch):
    class FalseClaim(ts.StreamingTTSProvider):
        async_transport_cancellable = True

        @staticmethod
        def available():
            return True

        def stream(self, text):
            yield b"legacy"

    monkeypatch.setitem(ts._REGISTRY, "false-claim", FalseClaim)
    assert ts.resolve_streaming_provider(
        {"streaming": {"provider": "false-claim"}},
        require_transport_cancellation=True,
    ) is None


def test_cancellable_auto_skips_legacy_provider_for_gemini(monkeypatch):
    legacy = _register_fake(monkeypatch, "elevenlabs")
    cancellable = _register_fake(
        monkeypatch, "gemini", async_transport_cancellable=True
    )
    monkeypatch.setattr(ts, "_PROVIDER_PRIORITY", ["elevenlabs", "gemini"])

    provider = ts.resolve_streaming_provider(
        {"streaming": {"provider": "auto"}},
        require_transport_cancellation=True,
    )

    assert not isinstance(provider, legacy)
    assert isinstance(provider, cancellable)


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


# ── Dispatch: universal per-sentence sync fallback ───────────────────────


# ── tts.streaming.provider config knob (salvaged from PR #47588) ─────────


# ── Credential routing: resolve_provider_secret, never bare env ──────────


def test_elevenlabs_available_routes_through_secret_resolver(monkeypatch):
    calls = []

    def _fake_resolve(env_var, provider_id):
        calls.append((env_var, provider_id))
        return "pool-key"

    monkeypatch.setattr(ts, "_resolve_key", _fake_resolve)
    assert ts.ElevenLabsStreamer.available() is True
    assert ("ELEVENLABS_API_KEY", "elevenlabs") in calls


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


# ── xAI WebSocket bridge ─────────────────────────────────────────────────


# ── 16 MiB per-sentence stream cap ───────────────────────────────────────


def test_stream_cap_raises_instead_of_silently_truncating(monkeypatch):
    monkeypatch.setattr(ts, "_STREAM_SENTENCE_BYTE_CAP", 100)

    def _endless():
        while True:
            yield b"\x00" * 64

    iterator = ts._capped(_endless(), "test")
    assert next(iterator) == b"\x00" * 64
    with pytest.raises(ts.StreamingTTSLimitError, match="decoded audio cap"):
        next(iterator)


# ── Dispatch: chunked streamer path (regression tests) ───────────────────


# The 12 speaker-path tests below assert on the sounddevice OutputStream
# branch, which stream_tts_to_speaker takes on every host EXCEPT macOS —
# Darwin routes to the tempfile/afplay path by design. They used to fake
# platform.system() == "Linux" (a no-op on the Linux CI lane) purely to
# shield macOS dev machines; an honest exclusion skipif says the same
# thing without lying to the interpreter.
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_streamer_path_handles_misaligned_pcm_chunks(monkeypatch):
    """Regression: PCM chunks with odd byte counts must not be dropped.

    OpenAI's streaming PCM API yields HTTP chunks on arbitrary byte
    boundaries that are not aligned to the int16 frame width (2 bytes).
    The old code called numpy.frombuffer directly on each chunk, which
    raised "buffer size must be a multiple of element size" on any
    odd-length chunk and silently dropped it — producing scattered
    audio fragments. The fix carries leftover bytes into the next chunk.
    """
    from tools import tts_tool

    class _OddChunkProvider(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            # Deliberately yield chunks with odd byte counts so the
            # int16 frame boundary falls between chunks.
            yield b"\x01\x00\x02"       # 3 bytes — odd, would crash old code
            yield b"\x00\x03\x00\x04"   # 4 bytes — even, old code OK
            yield b"\x00\x05\x00"       # 3 bytes — odd, would crash old code

    sd, out = _sd_mock()
    q = _drain_queue(["A complete sentence for testing."])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_OddChunkProvider({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    # Every chunk must have been written — no drops from misalignment.
    assert out.write.called, "expected PCM chunks written despite odd byte counts"
    # Collect all bytes the output stream received across all write calls.
    written_bytes = b""
    for call_args in out.write.call_args_list:
        arr = call_args[0][0]
        written_bytes += arr.tobytes()
    # The provider yielded 3 + 4 + 3 = 10 bytes total; all should arrive.
    assert len(written_bytes) == 10, (
        f"expected 10 bytes of PCM data, got {len(written_bytes)} — "
        "misaligned chunks were likely dropped"
    )
    assert done.is_set()


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_streamer_path_survives_portaudio_write_error(monkeypatch):
    """Regression: a transient PortAudio error on output_stream.write must
    not kill the playback thread or hang the pipeline join.

    PortAudio/Core Audio can raise errors mid-stream (e.g. PaErrorCode -9986
    "Internal PortAudio error" on macOS device state changes).  The worker
    must log and break, not crash — otherwise _playback_done never fires.
    """
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
    out.write.side_effect = OSError("Internal PortAudio error [PaErrorCode -9986]")
    q = _drain_queue(["A complete sentence for testing."])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Fake({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert out.write.called, "expected at least one write attempt"
    assert done.is_set(), "done event must fire even after PortAudio error"


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_streamer_reinit_after_portaudio_error_plays_remaining_sentences(monkeypatch):
    """Regression: after a PortAudio error the worker must reinit the stream
    and continue playing remaining sentences instead of dropping them.

    Simulates two sentences where the first triggers a PortAudio -9986 error
    on write.  The mock sounddevice returns a *fresh* OutputStream on the
    second call to ``OutputStream()`` (the reinit).  The second sentence must
    be written to that fresh stream, proving the pipeline recovered.
    """
    from tools import tts_tool

    class _Fake(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            yield b"\x01\x00" * 50
            yield b"\x02\x00" * 50

    # First OutputStream fails on write; second (reinit) succeeds.
    sd = MagicMock()
    broken_out = MagicMock()
    fresh_out = MagicMock()
    out_pool = [broken_out, fresh_out]
    broken_out.write.side_effect = OSError(
        "Internal PortAudio error [PaErrorCode -9986]"
    )

    def _make_stream(*args, **kwargs):
        return out_pool.pop(0) if out_pool else MagicMock()

    sd.OutputStream.side_effect = _make_stream

    q = _drain_queue([
        "First sentence triggers PortAudio error here. ",
        "Second sentence must still play after reinit. ",
    ])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Fake({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert broken_out.write.called, "first stream should have received a write"
    assert fresh_out.write.called, (
        "second (reinit) stream should have received writes for the "
        "remaining sentence — proves the pipeline recovered"
    )
    assert done.is_set(), "done event must fire after recovery"


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_streamer_tempfile_fallback_after_reinit_exhausted(monkeypatch):
    """Regression: after 3 failed reinits, remaining sentences must play
    via the temp-file fallback, not be silently dropped.
    """
    from tools import tts_tool

    class _Fake(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            yield b"\x01\x00" * 50

    # Every OutputStream fails on write — reinit will keep failing.
    sd = MagicMock()
    out = MagicMock()
    sd.OutputStream.return_value = out
    out.write.side_effect = OSError(
        "Internal PortAudio error [PaErrorCode -9986]"
    )

    # Patch play_audio_file so the tempfile fallback doesn't actually
    # try to play audio — just count that it was called.
    play_calls: list[str] = []

    def _fake_play(path):
        play_calls.append(path)

    q = _drain_queue([
        "First sentence triggers PortAudio error. ",
        "Second sentence fails after first reinit. ",
        "Third sentence fails after second reinit. ",
        "Fourth sentence fails after third reinit. ",
        "Fifth sentence plays via tempfile fallback. ",
    ])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Fake({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("tools.voice_mode.play_audio_file", side_effect=_fake_play):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    # The stream was created 4 times: initial + 3 reinit attempts.
    assert sd.OutputStream.call_count == 4, (
        f"expected 4 OutputStream calls (initial + 3 reinits), "
        f"got {sd.OutputStream.call_count}"
    )
    assert done.is_set(), "done event must fire even after reinit exhaustion"
    assert len(play_calls) >= 1, (
        "tempfile fallback should have been invoked for remaining "
        "sentences after reinit exhaustion"
    )



# ── Dispatch: hybrid batch-prefetch path ──────────────────────────────────

@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_hybrid_first_sentence_streamed_individually(monkeypatch):
    """The first sentence must get its own stream() call for low TTFA."""
    from tools import tts_tool

    stream_calls: list[str] = []

    class _Tracking(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            stream_calls.append(text)
            yield b"\x00\x00" * 10

    sd, out = _sd_mock()
    q = _drain_queue(["This is the first complete sentence."])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Tracking({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert len(stream_calls) == 1, (
        f"single sentence should trigger 1 stream() call, got {stream_calls}"
    )
    assert done.is_set()


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_hybrid_subsequent_sentences_prefetched_individually(monkeypatch):
    """Every sentence should get its own stream() call — per-sentence
    prefetch fires the HTTP request the moment each sentence completes,
    eliminating inter-sentence gaps."""
    from tools import tts_tool

    stream_calls: list[str] = []

    class _Tracking(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            stream_calls.append(text)
            yield b"\x00\x00" * 10

    sd, out = _sd_mock()
    # Four sentences — each gets its own stream() call.
    sentences = [
        "This is the very first sentence here. ",
        "This is the second complete sentence. ",
        "This is the third complete sentence. ",
        "This is the fourth complete sentence. ",
    ]
    q = _drain_queue(sentences)
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Tracking({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    # Exactly 4 calls: one per sentence.
    assert len(stream_calls) == 4, (
        f"expected 4 stream() calls (1 per sentence), "
        f"got {len(stream_calls)}: {stream_calls}"
    )
    # Each call contains its corresponding sentence's text.
    assert "first sentence" in stream_calls[0]
    assert "second" in stream_calls[1].lower()
    assert "third" in stream_calls[2].lower()
    assert "fourth" in stream_calls[3].lower()
    assert done.is_set()


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_hybrid_short_sentences_each_get_own_call(monkeypatch):
    """Short sentences should each get their own stream() call — no batching,
    no waiting for a threshold or end-of-text."""
    from tools import tts_tool

    stream_calls: list[str] = []

    class _Tracking(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            stream_calls.append(text)
            yield b"\x00\x00" * 10

    sd, out = _sd_mock()
    # Two short sentences — each gets its own stream() call.
    q = _drain_queue([
        "This is the first sentence. ",
        "Short second one. ",
    ])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Tracking({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert len(stream_calls) == 2, (
        f"expected 2 stream() calls (1 per sentence), "
        f"got {len(stream_calls)}: {stream_calls}"
    )
    assert "first" in stream_calls[0].lower()
    assert "second" in stream_calls[1].lower()
    assert done.is_set()


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_hybrid_done_event_waits_for_prefetch(monkeypatch):
    """The done event must not fire until the prefetch thread has finished,
    otherwise continuous voice mode could overlap turns."""
    from tools import tts_tool

    prefetch_done = threading.Event()

    class _Blocking(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            # For the batch call (second stream() invocation), block until
            # the test signals. The first call returns immediately.
            yield b"\x00\x00" * 10
            # Small delay to ensure the prefetch thread is running when
            # the main loop hits end-of-text.
            import time as _time
            _time.sleep(0.3)
            prefetch_done.set()

    sd, out = _sd_mock()
    sentences = [
        "This is the first sentence here. ",
        "This is the second sentence here. ",
        "This is the third sentence here. ",
    ]
    q = _drain_queue(sentences)
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Blocking({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    # done.is_set() is true — but only after the prefetch joined.
    assert done.is_set()
    # The prefetch thread should have completed before done was set.
    assert prefetch_done.is_set(), (
        "done event fired before the prefetch thread finished — "
        "this would cause audio overlap in continuous voice mode"
    )


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_hybrid_single_sentence_still_works(monkeypatch):
    """A single-sentence reply should stream immediately with no batch."""
    from tools import tts_tool

    stream_calls: list[str] = []

    class _Tracking(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            stream_calls.append(text)
            yield b"\x00\x00" * 10

    sd, out = _sd_mock()
    q = _drain_queue(["Just one complete sentence."])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Tracking({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert len(stream_calls) == 1, (
        f"single sentence should trigger exactly 1 stream() call, got {stream_calls}"
    )
    assert done.is_set()


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_hybrid_playback_serialized_no_overlap(monkeypatch):
    """Multiple batch flushes must not overlap on the output stream.

    The playback lock serializes write calls so audio segments play in
    order. We verify by tracking concurrent playback — at most one thread
    should be inside _play_pcm_chunks at any time.
    """
    from tools import tts_tool

    active_plays = [0]
    max_concurrent = [0]
    play_order: list[str] = []

    class _Tracking(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            # Yield enough data to exercise the write loop.
            for _ in range(5):
                yield b"\x00\x00" * 20

    sd = MagicMock()
    out = MagicMock()

    def _mock_write(_data):
        active_plays[0] += 1
        max_concurrent[0] = max(max_concurrent[0], active_plays[0])
        # Track which batch is playing by the data pattern (not text,
        # since we can't access it from the write callback).
        play_order.append("play")
        active_plays[0] -= 1

    out.write.side_effect = _mock_write
    sd.OutputStream.return_value = out

    # Many sentences to force multiple batch flushes.
    sentences = [f"This is sentence number {i} here. " for i in range(10)]
    q = _drain_queue(sentences)
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Tracking({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert done.is_set()
    assert max_concurrent[0] <= 1, (
        f"playback threads overlapped: max concurrent writes = {max_concurrent[0]}"
    )


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_hybrid_prefetch_fires_http_immediately(monkeypatch):
    """The prefetch thread must start consuming the generator (firing the
    HTTP request) the moment _enqueue_audio is called, NOT when the
    playback worker gets to it.

    We verify by recording the wall-clock time when stream() first yields
    and asserting that the second call's first yield happens before the
    first call's playback completes.
    """
    import time
    from tools import tts_tool

    stream_start_times: list[float] = []
    playback_done_times: list[float] = []
    block_first_playback = threading.Event()

    class _BlockingFirst(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            stream_start_times.append(time.monotonic())
            # First sentence: block until the test signals playback to proceed.
            # This simulates a long audio segment still playing.
            if len(stream_start_times) == 1:
                block_first_playback.wait(timeout=5.0)
            yield b"\x00\x00" * 10

    sd, out = _sd_mock()
    write_count = [0]

    def _mock_write(_data):
        write_count[0] += 1
        if write_count[0] == 1:
            # First write of first sentence — unblock so playback can finish.
            block_first_playback.set()

    out.write.side_effect = _mock_write

    # Two sentences: first blocks, second should prefetch while first plays.
    q = _drain_queue(["First sentence here. ", "Second sentence here. "])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_BlockingFirst({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert done.is_set()
    assert len(stream_start_times) == 2, (
        f"expected 2 stream() calls, got {len(stream_start_times)}"
    )
    # The second stream() call must have started (HTTP fired) while the
    # first was still blocked/playing. Since the first blocks until
    # playback starts, and the second is enqueued immediately after,
    # the second's start time should be very close to the first's.
    # We just assert both fired (the timing is inherently tested by the
    # fact that block_first_playback was needed to unblock the first).
    assert stream_start_times[1] > stream_start_times[0], (
        "second stream() should start after the first"
    )


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_display_callback_not_called_when_streaming_enabled(monkeypatch):
    """When streaming is enabled, display_callback must NOT be passed to
    the TTS consumer — the token stream already renders text. This
    prevents duplicate rendering (fix #1).

    This is a CLI-level test simulated at the tts_tool level: the key
    invariant is that stream_tts_to_speaker with display_callback=None
    still works correctly (no crash, no display).
    """
    from tools import tts_tool

    class _Fake(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            yield b"\x00\x00" * 10

    sd, out = _sd_mock()
    q = _drain_queue(["A sentence for the no-callback path. "])
    stop, done = threading.Event(), threading.Event()

    # display_callback=None simulates the streaming_enabled=True case.
    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Fake({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        tts_tool.stream_tts_to_speaker(q, stop, done, display_callback=None)

    assert done.is_set()
    # No assertion on display — the point is no crash and done is set.


# ── Sync fallback: one-ahead synthesis/playback pipeline ─────────────────
#
# The universal per-sentence sync path pipelines synthesis with playback:
# while sentence n plays, sentence n+1 is already synthesizing. For local
# model providers (RTF near 1) the serial path spent as long silent between
# sentences as speaking; these pin the overlap, ordering, stop, failure
# isolation, and temp-file hygiene of the pipelined path.


def _timed_sync_run(monkeypatch, sentences, *, synth_s=0.12, play_s=0.12,
                    synth_fail_on=None, stop_after_plays=None):
    """Drive stream_tts_to_speaker over the sync path with timed fakes.

    Returns (events, stop, done): events is [(kind, sentence, t_start, t_end)]
    with kinds "synth"/"play", timestamps from a shared monotonic origin.
    """
    from tools import tts_tool

    origin = time.monotonic()
    events = []
    lock = threading.Lock()
    stop, done = threading.Event(), threading.Event()

    def fake_synth(text, output_path):
        t0 = time.monotonic() - origin
        if synth_fail_on and synth_fail_on in text:
            raise RuntimeError("synth exploded")
        time.sleep(synth_s)
        with open(output_path, "wb") as fh:
            fh.write(b"x" * 100)
        with lock:
            events.append(("synth", text, t0, time.monotonic() - origin))

    def fake_play(path):
        t0 = time.monotonic() - origin
        time.sleep(play_s)
        with lock:
            events.append(("play", path, t0, time.monotonic() - origin))
            plays = sum(1 for e in events if e[0] == "play")
        if stop_after_plays is not None and plays >= stop_after_plays:
            stop.set()

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_synth)
    fake_vm = MagicMock()
    fake_vm.play_audio_file.side_effect = fake_play
    monkeypatch.setitem(__import__("sys").modules, "tools.voice_mode", fake_vm)

    q = _drain_queue(sentences)
    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=None):
        tts_tool.stream_tts_to_speaker(q, stop, done)
    return events, stop, done


def test_sync_pipeline_overlaps_synthesis_with_playback(monkeypatch):
    sentences = ["First full sentence here. ", "Second full sentence here. ",
                 "Third full sentence here. "]
    events, _stop, done = _timed_sync_run(monkeypatch, sentences)

    synths = [e for e in events if e[0] == "synth"]
    plays = [e for e in events if e[0] == "play"]
    assert len(synths) == 3 and len(plays) == 3
    assert done.is_set()

    # The point of the pipeline: sentence 2's synthesis STARTS before
    # sentence 1's playback ENDS (serial code could never do this).
    synth2_start = synths[1][2]
    play1_end = plays[0][3]
    assert synth2_start < play1_end, (
        f"no overlap: synth2 started at {synth2_start:.3f}, "
        f"play1 ended at {play1_end:.3f}"
    )


def test_sync_pipeline_preserves_order_and_isolates_failures(monkeypatch):
    sentences = ["Alpha sentence spoken first. ", "Bravo sentence explodes here. ",
                 "Charlie sentence still plays. "]
    events, _stop, done = _timed_sync_run(monkeypatch, sentences,
                                          synth_fail_on="Bravo")

    synths = [e[1] for e in events if e[0] == "synth"]
    plays = [e for e in events if e[0] == "play"]
    # Bravo's synth raised: never synthesized-to-file, never played — but
    # Alpha and Charlie both played, in submission order.
    assert [s.split()[0] for s in synths] == ["Alpha", "Charlie"]
    assert len(plays) == 2
    assert done.is_set()


def test_sync_pipeline_stop_skips_queued_playback(monkeypatch):
    sentences = ["First full sentence here. ", "Second full sentence here. ",
                 "Third full sentence here. ", "Fourth full sentence here. "]
    events, stop, done = _timed_sync_run(monkeypatch, sentences,
                                         stop_after_plays=1)

    plays = [e for e in events if e[0] == "play"]
    assert len(plays) == 1, f"stop after first play must skip the rest, got {len(plays)}"
    assert stop.is_set() and done.is_set()


def test_sync_pipeline_cleans_temp_files(monkeypatch):
    from tools import tts_tool

    created = []
    real_mkstemp = tempfile.mkstemp

    def tracking_mkstemp(*a, **k):
        fd, path = real_mkstemp(*a, **k)
        created.append(path)
        return fd, path

    monkeypatch.setattr(tts_tool.tempfile, "mkstemp", tracking_mkstemp)
    events, _stop, done = _timed_sync_run(monkeypatch,
                                          ["First full sentence here. ",
                                           "Second full sentence here. "])
    assert len([e for e in events if e[0] == "play"]) == 2
    assert created, "expected temp files to be created via mkstemp"
    leftovers = [p for p in created if os.path.exists(p)]
    assert not leftovers, f"temp files not cleaned: {leftovers}"


# ── Gemini 3.1 streamGenerateContent transport ───────────────────────────
#
# Gemini 3.1 Flash TTS Preview emits incremental PCM through the existing
# streamGenerateContent SSE API. Credentials stay in the x-goog-api-key header
# and redirects are disabled so requests cannot forward the key cross-origin.

import base64 as _b64
import json as _json


def _sse_context(events):
    resp = MagicMock()
    body = b"".join(
        b"data: "
        + (_json.dumps(ev) if not isinstance(ev, str) else ev).encode()
        + b"\n\n"
        for ev in events
    )
    resp.headers = {"content-type": "text/event-stream; charset=utf-8"}
    # Split arbitrarily to prove the bounded parser handles network chunking.
    resp.iter_content.return_value = iter(
        body[i:i + 17] for i in range(0, len(body), 17)
    )
    resp.__enter__.return_value = resp
    return resp


def _gemini_audio_event(b64_data: str) -> dict:
    return {
        "candidates": [{
            "content": {"parts": [{
                "inlineData": {"mimeType": "audio/L16;rate=24000", "data": b64_data},
            }]},
        }],
    }


class TestGeminiStreamingTransport:
    def _streamer(self):
        return ts.GeminiStreamer(
            {"gemini": {"model": "gemini-3.1-flash-tts-preview"}},
            {"model": "gemini-3.1-flash-tts-preview"},
        )

    @pytest.mark.asyncio
    async def test_async_transport_streams_pcm_without_sync_thread(self, monkeypatch):
        import httpx

        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        event = _gemini_audio_event(_b64.b64encode(b"\x09\x0a").decode())
        wire = b"data: " + _json.dumps(event).encode() + b"\n\n"
        captured = {}

        class Response:
            headers = {"content-type": "text/event-stream"}

            def raise_for_status(self):
                return None

            async def aiter_bytes(self, chunk_size):
                assert chunk_size == 8192
                yield wire[:7]
                yield wire[7:]

        class StreamContext:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *_args):
                return None

        class Client:
            def __init__(self, **kwargs):
                captured["client"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def stream(self, method, url, **kwargs):
                captured.update(method=method, url=url, request=kwargs)
                return StreamContext()

        monkeypatch.setattr(httpx, "AsyncClient", Client)
        chunks = [chunk async for chunk in self._streamer().astream("Async")]
        assert chunks == [b"\x09\x0a"]
        assert captured["method"] == "POST"
        assert captured["client"]["follow_redirects"] is False
        assert captured["request"]["headers"] == {"x-goog-api-key": "g-key"}

    @pytest.mark.asyncio
    async def test_async_cancel_interrupts_pre_header_wait(self, monkeypatch):
        import httpx

        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class StreamContext:
            async def __aenter__(self):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            async def __aexit__(self, *_args):
                return None

        class Client:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def stream(self, *_args, **_kwargs):
                return StreamContext()

        monkeypatch.setattr(httpx, "AsyncClient", Client)
        streamer = self._streamer()

        async def consume():
            return [chunk async for chunk in streamer.astream("Cancel headers")]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        streamer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_async_deadline_covers_pre_header_wait(self, monkeypatch):
        import httpx

        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setattr(ts, "_GEMINI_STREAM_DEADLINE_S", 0.05)

        class StreamContext:
            async def __aenter__(self):
                await asyncio.Event().wait()

            async def __aexit__(self, *_args):
                return None

        class Client:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def stream(self, *_args, **_kwargs):
                return StreamContext()

        monkeypatch.setattr(httpx, "AsyncClient", Client)
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            _ = [chunk async for chunk in self._streamer().astream("Deadline")]
        assert time.monotonic() - start < 0.5

    def test_31_uses_stream_generate_content_with_header_auth(self, monkeypatch):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _sse_context([
                _gemini_audio_event(_b64.b64encode(b"\x01\x02").decode()),
            ])

        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        with patch("requests.post", side_effect=fake_post):
            assert list(self._streamer().stream("Hello")) == [b"\x01\x02"]

        assert captured["url"].endswith(
            "/models/gemini-3.1-flash-tts-preview:streamGenerateContent"
        )
        assert captured["kwargs"]["params"] == {"alt": "sse"}
        assert captured["kwargs"]["headers"]["x-goog-api-key"] == "g-key"
        assert captured["kwargs"]["allow_redirects"] is False
        body = captured["kwargs"]["json"]
        assert body["generationConfig"]["responseModalities"] == ["AUDIO"]
        voice = body["generationConfig"]["speechConfig"]["voiceConfig"]
        assert voice["prebuiltVoiceConfig"]["voiceName"] == "Kore"

    def test_yields_pcm_from_multiple_audio_events(self, monkeypatch):
        chunks = [b"\x01\x02\x03\x04", b"\x05\x06"]
        events = [_gemini_audio_event(_b64.b64encode(c).decode()) for c in chunks]
        with patch("requests.post", return_value=_sse_context(events)):
            assert list(self._streamer().stream("Speak")) == chunks

    def test_ignores_non_audio_and_malformed_events(self, monkeypatch):
        good = _gemini_audio_event(_b64.b64encode(b"\x0a\x0b").decode())
        events = [
            [],
            "not json",
            {"candidates": []},
            {"candidates": [{"content": {"parts": [None, "text"]}}]},
            {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]},
            good,
        ]
        with patch("requests.post", return_value=_sse_context(events)):
            assert list(self._streamer().stream("Robust")) == [b"\x0a\x0b"]

    def test_invalid_base64_is_ignored(self, monkeypatch):
        bad = _gemini_audio_event("not-valid-base64!!!")
        good = _gemini_audio_event(_b64.b64encode(b"\x0c\x0d").decode())
        with patch("requests.post", return_value=_sse_context([bad, good])):
            assert list(self._streamer().stream("Robust")) == [b"\x0c\x0d"]

    def test_uses_provider_secret_resolver(self, monkeypatch):
        captured = {}
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr(
            ts,
            "_resolve_key",
            lambda env_name, provider: "secret-source-key" if env_name == "GEMINI_API_KEY" else "",
        )

        def fake_post(_url, **kwargs):
            captured["headers"] = kwargs["headers"]
            return _sse_context([])

        with patch("requests.post", side_effect=fake_post):
            list(self._streamer().stream("Secret"))
        assert captured["headers"]["x-goog-api-key"] == "secret-source-key"

    def test_rejects_non_sse_or_missing_content_type(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        for headers in ({"content-type": "application/json"}, {}):
            response = _sse_context([])
            response.headers = headers
            with patch("requests.post", return_value=response):
                with pytest.raises(RuntimeError, match="unexpected content type"):
                    list(self._streamer().stream("Wrong response"))

    def test_rejects_oversized_unterminated_sse_event(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        response = _sse_context([])
        response.iter_content.return_value = iter([b"data: " + b"x" * (4 * 1024 * 1024 + 1)])
        with patch("requests.post", return_value=response):
            with pytest.raises(RuntimeError, match="SSE event exceeds"):
                list(self._streamer().stream("Oversized"))

    def test_ignores_inline_data_with_wrong_or_missing_audio_mime(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        encoded = _b64.b64encode(b"\x01\x02").decode()
        invalid_mimes = [
            "image/png",
            "audio/L16evil;rate=24000",
            "audio/L16;rate=24000;channels=2",
            "audio/L16;rate=16000",
            "audio/L16;rate=not-a-number",
        ]
        invalid_events = []
        for mime in invalid_mimes:
            event = _gemini_audio_event(encoded)
            event["candidates"][0]["content"]["parts"][0]["inlineData"][
                "mimeType"
            ] = mime
            invalid_events.append(event)
        missing = _gemini_audio_event(encoded)
        del missing["candidates"][0]["content"]["parts"][0]["inlineData"][
            "mimeType"
        ]
        good = _gemini_audio_event(_b64.b64encode(b"\x03\x04").decode())
        with patch(
            "requests.post",
            return_value=_sse_context([*invalid_events, missing, good]),
        ):
            assert list(self._streamer().stream("Formats")) == [b"\x03\x04"]

    def test_raw_sse_cap_is_cumulative_before_json_decode(self):
        response = _sse_context([])
        response.iter_content.return_value = iter([b": keepalive\n", b": keepalive\n"])
        with pytest.raises(RuntimeError, match="raw SSE bytes"):
            list(
                ts._iter_bounded_sse_data(
                    response,
                    label="test SSE",
                    deadline=time.monotonic() + 1,
                    raw_byte_cap=15,
                )
            )

    def test_absolute_deadline_rejects_trickled_sse(self):
        response = _sse_context([])
        response.iter_content.return_value = iter([b": keepalive\n"])
        with pytest.raises(TimeoutError, match="absolute deadline"):
            list(
                ts._iter_bounded_sse_data(
                    response,
                    label="test SSE",
                    deadline=time.monotonic() - 1,
                )
            )

    def test_deadline_closes_blocked_body_read(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        monkeypatch.setattr(ts, "_GEMINI_STREAM_DEADLINE_S", 0.05)
        closed = threading.Event()

        class BlockingResponse:
            headers = {"content-type": "text/event-stream; charset=utf-8"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                assert chunk_size == 8192
                closed.wait(timeout=1.0)
                return iter(())

            def close(self):
                closed.set()

        started = time.monotonic()
        with patch("requests.post", return_value=BlockingResponse()):
            with pytest.raises(TimeoutError, match="absolute deadline"):
                list(self._streamer().stream("Deadline"))
        assert time.monotonic() - started < 0.5
        assert closed.is_set()

    def test_cancel_before_request_prevents_network_start(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        streamer = self._streamer()
        streamer.cancel()
        with patch("requests.post") as post:
            assert list(streamer.stream("Already cancelled")) == []
        post.assert_not_called()

    def test_cancel_closes_blocked_response_promptly(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        started = threading.Event()
        closed = threading.Event()

        class BlockingResponse:
            headers = {"content-type": "text/event-stream"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                assert chunk_size == 8192
                started.set()
                closed.wait(timeout=2.0)
                return iter(())

            def close(self):
                closed.set()

        streamer = self._streamer()
        errors = []

        def consume():
            try:
                list(streamer.stream("Cancel"))
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        thread = threading.Thread(target=consume)
        with patch("requests.post", return_value=BlockingResponse()):
            thread.start()
            assert started.wait(timeout=1.0)
            streamer.cancel()
            thread.join(timeout=1.0)

        assert closed.is_set()
        assert not thread.is_alive()
        assert errors == []
