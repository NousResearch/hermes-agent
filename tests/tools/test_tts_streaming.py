"""Tests for the provider-agnostic streaming TTS backend (tools.tts_streaming)
and its dispatch through tools.tts_tool_speaker.stream_tts_to_speaker.

No live audio or network: the ElevenLabs/OpenAI SDKs, sounddevice, and the sync
synth path are all mocked. Covers the registry/resolver, provider availability,
the chunked-streamer playback path, and the universal per-sentence sync fallback.
"""

import os
import json
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


def test_openai_streamer_forwards_consent_attestation(monkeypatch):
    """The chunked path sends the same optional tts.openai body fields as the sync path (#99775);
    an unset key adds no extra_body so strict servers see an unchanged request."""
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
            captured["create"] = kwargs
            return _Response()

    class _OpenAI:
        def __init__(self, **kwargs):
            self.audio = MagicMock()
            self.audio.speech.with_streaming_response = _StreamingCreate()

    monkeypatch.setattr(ts, "resolve_openai_audio_api_key", lambda: "env-key")
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key, *args: None)
    monkeypatch.setattr("openai.OpenAI", _OpenAI)

    section = {"api_key": "k", "consent_attestation": "I have consent"}
    list(ts.OpenAIStreamer({"openai": section}, section).stream("hi"))
    assert captured["create"]["extra_body"] == {"consent_attestation": "I have consent"}

    list(ts.OpenAIStreamer({"openai": {"api_key": "k"}}, {"api_key": "k"}).stream("hi"))
    assert "extra_body" not in captured["create"]


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
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key, *args: None)
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




def test_xai_available_uses_oauth_credential_resolver(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("tools.xai_http")
    fake.resolve_xai_http_credentials = lambda **kw: {"api_key": "xai-key"}
    monkeypatch.setitem(sys.modules, "tools.xai_http", fake)
    assert ts.XAIStreamer.available() is True
    fake.resolve_xai_http_credentials = lambda **kw: {"api_key": ""}
    assert ts.XAIStreamer.available() is False


def test_xai_streaming_prefers_explicit_api_key(monkeypatch):
    """Metered TTS 403s on the subscription OAuth bearer — the streaming path must
    resolve credentials with prefer_api_key=True like the sync path (#87045)."""
    import sys
    import types

    calls = []

    fake = types.ModuleType("tools.xai_http")
    fake.resolve_xai_http_credentials = lambda **kw: calls.append(kw) or {"api_key": "k"}
    monkeypatch.setitem(sys.modules, "tools.xai_http", fake)

    ts.XAIStreamer.available()
    assert calls and all(c.get("prefer_api_key") is True for c in calls)

    # The tts_tool availability probe (wired as _BUILTIN_REQUIREMENTS["xai"]) must
    # resolve key-first too, or a configured key still spends the OAuth pool (#113727).
    from tools import tts_tool

    calls.clear()
    assert tts_tool._xai_requirements() is True
    assert calls and all(c.get("prefer_api_key") is True for c in calls)

    # _async_frames resolves before websockets.connect; an empty key raises first.
    calls.clear()
    ws_fake = types.ModuleType("websockets")
    monkeypatch.setitem(sys.modules, "websockets", ws_fake)
    fake.resolve_xai_http_credentials = lambda **kw: calls.append(kw) or {"api_key": ""}
    streamer = ts.XAIStreamer({}, {"voice_id": "v"})
    with pytest.raises(RuntimeError, match="No xAI credentials"):
        import asyncio
        asyncio.run(streamer._async_frames("hi").__anext__())
    assert calls and calls[0].get("prefer_api_key") is True


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
    from tools.tts_tool_speaker import stream_tts_to_speaker

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
        stream_tts_to_speaker(q, stop, done)

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
    from tools.tts_tool_speaker import stream_tts_to_speaker

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
        stream_tts_to_speaker(q, stop, done)

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
    from tools.tts_tool_speaker import stream_tts_to_speaker

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
        stream_tts_to_speaker(q, stop, done)

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
    from tools.tts_tool_speaker import stream_tts_to_speaker

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
        stream_tts_to_speaker(q, stop, done)

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
def test_speaker_honours_tts_streaming_min_len_for_short_cjk_opener(monkeypatch):
    """The CLI/TUI speaker cuts with the profile's tts.streaming.min_len (#96927): a 7-char CJK
    opener is streamed on its own instead of riding behind the second sentence."""
    from tools import tts_tool
    from tools.tts_tool_speaker import stream_tts_to_speaker

    stream_calls: list[str] = []

    class _Tracking(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            stream_calls.append(text)
            yield b"\x00\x00" * 10

    sd, _out = _sd_mock()
    q = _drain_queue(["记得，叫团团. ", "然后我们再说第二句话，这一句要长一些才行. "])
    stop, done = threading.Event(), threading.Event()

    with patch("tools.tts_streaming.resolve_streaming_provider",
               return_value=_Tracking({}, {})), \
         patch.object(tts_tool, "_load_tts_config", return_value={"streaming": {"min_len": 6}}), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        stream_tts_to_speaker(q, stop, done)

    assert stream_calls[0] == "记得，叫团团.", stream_calls
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
    from tools.tts_tool_speaker import stream_tts_to_speaker

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
        stream_tts_to_speaker(q, stop, done)

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
def test_hybrid_done_event_waits_for_prefetch(monkeypatch):
    """The done event must not fire until the prefetch thread has finished,
    otherwise continuous voice mode could overlap turns."""
    from tools import tts_tool
    from tools.tts_tool_speaker import stream_tts_to_speaker

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
        stream_tts_to_speaker(q, stop, done)

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
    from tools.tts_tool_speaker import stream_tts_to_speaker

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
        stream_tts_to_speaker(q, stop, done)

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
    from tools.tts_tool_speaker import stream_tts_to_speaker

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
        stream_tts_to_speaker(q, stop, done)

    assert done.is_set()
    assert max_concurrent[0] <= 1, (
        f"playback threads overlapped: max concurrent writes = {max_concurrent[0]}"
    )




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
    from tools.tts_tool_speaker import stream_tts_to_speaker

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
        stream_tts_to_speaker(q, stop, done)
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
    from tools import tts_tool_speaker

    created = []
    real_mkstemp = tempfile.mkstemp

    def tracking_mkstemp(*a, **k):
        fd, path = real_mkstemp(*a, **k)
        created.append(path)
        return fd, path

    monkeypatch.setattr(tts_tool_speaker.tempfile, "mkstemp", tracking_mkstemp)
    events, _stop, done = _timed_sync_run(monkeypatch,
                                          ["First full sentence here. ",
                                           "Second full sentence here. "])
    assert len([e for e in events if e[0] == "play"]) == 2
    assert created, "expected temp files to be created via mkstemp"
    leftovers = [p for p in created if os.path.exists(p)]
    assert not leftovers, f"temp files not cleaned: {leftovers}"


# ── #76466: honor the endpoint-reported PCM sample rate ────────────────────

class _FakeSpeechResponse:
    """Stand-in for the OpenAI SDK streaming response context manager."""

    def __init__(self, headers, chunks):
        self.headers, self._chunks = headers, chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self):
        yield from self._chunks


def _patch_openai_speech(monkeypatch, headers):
    calls = []

    class _Speech:
        class with_streaming_response:
            @staticmethod
            def create(**kw):
                calls.append(kw)
                return _FakeSpeechResponse(headers, [b"\x01\x00" * 100])

    class _Client:
        def __init__(self, **kw):
            self.audio = MagicMock(speech=_Speech())

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)
    return calls


def test_openai_streamer_honors_endpoint_reported_rate_in_wav_playback(monkeypatch):
    """Issue #76466: an OpenAI-compatible endpoint answering 44.1 kHz PCM (X-Audio-Sample-Rate)
    must drive the WAV header written for playback, not the construction-time expectation."""
    import wave
    from tools import tts_tool_speaker as sp

    _patch_openai_speech(monkeypatch, {"content-type": "audio/pcm", "x-audio-sample-rate": "44100"})
    streamer = ts.OpenAIStreamer({}, {"api_key": "sk-x", "pcm_sample_rate": "22050"})

    wav_rates = []

    def _fake_play(path):
        with wave.open(path, "rb") as wf:
            wav_rates.append(wf.getframerate())

    monkeypatch.setattr(sp._StreamerPlayback, "_device_usable", lambda self: False)
    with patch("tools.voice_mode.play_audio_file", side_effect=_fake_play):
        playback = sp._StreamerPlayback(streamer, threading.Event())
        playback.speak("One sentence.")
        playback.finish()
    assert streamer.sample_rate == 44100
    assert wav_rates == [44100]


@pytest.mark.parametrize(
    ("config", "headers", "expected"),
    [
        ({"pcm_sample_rate": "22050"}, {}, 22050),  # validated static expectation (PR #74021)
        ({"pcm_sample_rate": "bogus"}, {}, 24000),  # unparseable config falls back to the default
        ({}, {"content-type": "audio/pcm", "x-audio-sample-rate": "44100"}, 44100),
        ({}, {"content-type": "audio/L16; rate=16000"}, 16000),
        ({}, {"content-type": "audio/pcm"}, None),
    ],
)
def test_openai_pcm_sample_rate_resolution(config, headers, expected):
    """Issue #76466: static ``pcm_sample_rate`` is the pre-request expectation; the endpoint's
    response headers (explicit header or ``audio/L16; rate=``) are the post-request truth."""
    if headers:
        assert ts._sample_rate_from_headers(headers) == expected
    else:
        assert ts.OpenAIStreamer({}, {"api_key": "sk-x", **config}).sample_rate == expected


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS deliberately skips the sounddevice OutputStream path (PR #62601)",
)
def test_speaker_output_stream_opens_at_rate_learned_from_first_chunk(monkeypatch):
    """Issue #76466: the PortAudio device is opened after the first chunk arrived, at the rate
    the provider learned from the response, not at the construction-time default."""
    from tools import tts_tool
    from tools.tts_tool_speaker import stream_tts_to_speaker

    class _Learns(ts.StreamingTTSProvider):
        sample_rate = 24000

        @staticmethod
        def available():
            return True

        def stream(self, text):
            self.sample_rate = 44100  # what OpenAIStreamer does on the response headers
            yield b"\x01\x00" * 50

    sd, out = _sd_mock()
    q = _drain_queue(["The first sentence is long enough. ", "The second sentence is long enough too. "])
    stop, done = threading.Event(), threading.Event()
    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=_Learns({}, {})), \
         patch.object(tts_tool, "_import_sounddevice", return_value=sd):
        stream_tts_to_speaker(q, stop, done)
    assert done.is_set()
    assert [c.kwargs["samplerate"] for c in sd.OutputStream.call_args_list] == [44100]
    assert out.write.call_count == 2


def test_sync_pipeline_plays_the_artifact_the_tool_reported(monkeypatch, tmp_path):
    """A provider whose artifact lands off the requested path (command ``format`` suffix
    rewrite, or voice-compatible ffmpeg conversion) must still play: follow the reported
    ``file_path``/``file_paths`` instead of gating on the requested path (#115029)."""
    from tools import tts_tool
    from tools.tts_tool_speaker import stream_tts_to_speaker

    ogg = tmp_path / "sentence.ogg"
    ogg.write_bytes(b"x" * 32)

    def fake_synth(text, output_path):
        # The requested .mp3 stays a zero-byte mkstemp file; only the reported artifact is real.
        ogg_str = str(ogg)
        return json.dumps({
            "success": True,
            "file_path": ogg_str,
            "file_paths": [ogg_str],
        })

    played = []
    fake_vm = MagicMock()
    fake_vm.play_audio_file.side_effect = played.append
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_synth)
    monkeypatch.setitem(sys.modules, "tools.voice_mode", fake_vm)

    q = _drain_queue(["Hello there. "])
    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=None):
        stream_tts_to_speaker(q, threading.Event(), threading.Event())
    assert played == [str(ogg)], (
        "sentence dropped: playback ignored the reported artifact"
    )


def test_sync_pipeline_falls_back_to_requested_path_when_reported_missing(monkeypatch):
    """A tool envelope that reports nothing usable (None, non-JSON, missing files) keeps the
    legacy behavior: play the requested path when the tool wrote it there."""
    from tools import tts_tool
    from tools.tts_tool_speaker import stream_tts_to_speaker

    def fake_synth(text, output_path):
        with open(output_path, "wb") as fh:
            fh.write(b"x" * 100)
        return json.dumps({"success": False, "error": "shape without paths"})

    played = []
    fake_vm = MagicMock()

    def _record(path):
        played.append((path, os.path.getsize(path)))

    fake_vm.play_audio_file.side_effect = _record
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_synth)
    monkeypatch.setitem(sys.modules, "tools.voice_mode", fake_vm)

    q = _drain_queue(["Hello there. "])
    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=None):
        stream_tts_to_speaker(q, threading.Event(), threading.Event())
    # The temp file is unlinked after playback, so capture its size at play time.
    assert len(played) == 1 and played[0][1] > 0, (
        "requested-path fallback no longer plays"
    )
