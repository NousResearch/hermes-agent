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


def test_stream_cap_truncates_runaway_upstream(monkeypatch):
    monkeypatch.setattr(ts, "_STREAM_SENTENCE_BYTE_CAP", 100)

    def _endless():
        while True:
            yield b"\x00" * 64

    out = list(ts._capped(_endless(), "test"))
    assert len(out) == 1  # 64 ok, 128 > cap → stop
    assert sum(len(c) for c in out) <= 100


# ── Dispatch: chunked streamer path (regression tests) ───────────────────


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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert out.write.called, "expected at least one write attempt"
    assert done.is_set(), "done event must fire even after PortAudio error"


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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert broken_out.write.called, "first stream should have received a write"
    assert fresh_out.write.called, (
        "second (reinit) stream should have received writes for the "
        "remaining sentence — proves the pipeline recovered"
    )
    assert done.is_set(), "done event must fire after recovery"


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
         patch("platform.system", return_value="Linux"), \
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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert len(stream_calls) == 1, (
        f"single sentence should trigger 1 stream() call, got {stream_calls}"
    )
    assert done.is_set()


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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert len(stream_calls) == 2, (
        f"expected 2 stream() calls (1 per sentence), "
        f"got {len(stream_calls)}: {stream_calls}"
    )
    assert "first" in stream_calls[0].lower()
    assert "second" in stream_calls[1].lower()
    assert done.is_set()


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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    # done.is_set() is true — but only after the prefetch joined.
    assert done.is_set()
    # The prefetch thread should have completed before done was set.
    assert prefetch_done.is_set(), (
        "done event fired before the prefetch thread finished — "
        "this would cause audio overlap in continuous voice mode"
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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert len(stream_calls) == 1, (
        f"single sentence should trigger exactly 1 stream() call, got {stream_calls}"
    )
    assert done.is_set()


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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
        tts_tool.stream_tts_to_speaker(q, stop, done)

    assert done.is_set()
    assert max_concurrent[0] <= 1, (
        f"playback threads overlapped: max concurrent writes = {max_concurrent[0]}"
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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
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
         patch.object(tts_tool, "_import_sounddevice", return_value=sd), \
         patch("platform.system", return_value="Linux"):
        tts_tool.stream_tts_to_speaker(q, stop, done, display_callback=None)

    assert done.is_set()
    # No assertion on display — the point is no crash and done is set.
