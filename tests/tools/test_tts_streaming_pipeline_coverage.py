"""Coverage for the TTS streaming pipeline in ``tools/tts_tool.py``.

Covers the classes/functions introduced for per-sentence streaming:
``_SyncSentencePipeline`` (its ``__init__``, ``speak``, ``close``,
``_synthesize_to_tmp`` and ``_drain``) plus the orchestrator
``stream_tts_to_speaker`` in its sync-pipeline fallback form (when no
chunked streaming provider is available) and in its chunked-streamer form on
macOS (where ``output_stream`` is intentionally left ``None`` so audio goes
through the tempfile -> ``play_audio_file`` path).

No real audio device is ever touched: ``tools.voice_mode.play_audio_file``
is replaced by a recorder stub and the chunked streamer is a fake object that
yields tiny int16 PCM chunks.
"""

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import types
import wave

import pytest

import tools.tts_streaming as tts_streaming
import tools.tts_tool as tts_tool
from tools.tts_tool import _SyncSentencePipeline, stream_tts_to_speaker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tiny_wav(path: str, frames: bytes = b"\x00\x00" * 8, rate: int = 24000) -> None:
    """Write a tiny, valid 16-bit mono WAV to *path* so playback sees a
    non-empty, real audio file (the pipeline only checks ``getsize > 0``)."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)


def _make_tmp_wav_path() -> str:
    """Create a tiny WAV temp file and return its path (caller unlinks)."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    _write_tiny_wav(path)
    return path


def _settled_assert(baseline: int) -> None:
    """Assert no threads leaked past *baseline* (with a grace window)."""
    deadline = time.time() + 3.0
    while threading.active_count() > baseline and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= baseline, (
        f"leaked threads: active={threading.active_count()} baseline={baseline}"
    )


class _FakeStreamer:
    """Stands in for a chunked StreamingTTSProvider.

    Yields a few int16-aligned PCM chunks so the Darwin path can run the
    tempfile playback branch without any real audio device or numpy.
    """

    sample_rate = 24000
    channels = 1

    def __init__(self):
        self.streamed: list[str] = []

    def stream(self, text):
        self.streamed.append(text)

        def _gen():
            for chunk in (b"\x01\x00", b"\x02\x00", b"\x03\x00", b"\x04\x00"):
                yield chunk

        return _gen()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_voice_mode(monkeypatch):
    """Replace ``tools.voice_mode`` with a recorder stub.

    The pipeline imports ``play_audio_file`` lazily inside the worker/player
    threads, so swapping the module object in ``sys.modules`` lets those
    threads see the stub. Returns the list of played paths.
    """
    played: list[str] = []
    mod = types.ModuleType("tools.voice_mode")
    mod.play_audio_file = lambda path: played.append(path)
    mod.mark_audio_output_active = lambda _active: None
    monkeypatch.setitem(sys.modules, "tools.voice_mode", mod)
    return played


def _patch_sync_fallback(monkeypatch):
    """Force the per-sentence sync-pipeline fallback (no chunked streamer)."""
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(
        tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: None
    )


# ---------------------------------------------------------------------------
# _SyncSentencePipeline
# ---------------------------------------------------------------------------

def test_sync_pipeline_speaks_in_order_with_lookahead(fake_voice_mode, monkeypatch):
    """Sentences are synthesized FIFO and played FIFO with the lookahead bound."""
    played = fake_voice_mode
    synth_texts: list[str] = []

    def fake_tts(text, output_path, **kwargs):
        synth_texts.append(text)
        _write_tiny_wav(output_path)
        return {"ok": True}

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

    baseline = threading.active_count()
    pipe = _SyncSentencePipeline(threading.Event(), lookahead=2)
    sentences = [
        "First sentence here has enough length to speak.",
        "Second sentence here has enough length too.",
        "Third sentence here carries a bit more content.",
    ]
    for s in sentences:
        pipe.speak(s)
    pipe.close()

    assert synth_texts == sentences
    assert len(played) == len(sentences)
    _settled_assert(baseline)


def test_sync_pipeline_speak_returns_when_stop_set(fake_voice_mode, monkeypatch):
    """``speak`` must no-op once stop_event is set, and ``close`` still joins."""
    played = fake_voice_mode
    synth: list[str] = []

    def fake_tts(text, output_path, **kwargs):
        synth.append(text)
        _write_tiny_wav(output_path)
        return {"ok": True}

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

    baseline = threading.active_count()
    stop = threading.Event()
    stop.set()
    pipe = _SyncSentencePipeline(stop)
    pipe.speak("This should never be queued.")
    pipe.close()

    assert synth == []
    assert played == []
    _settled_assert(baseline)


def test_synthesize_to_tmp_returns_path_on_success(monkeypatch):
    """Real ``_synthesize_to_tmp`` returns a non-empty file when synthesis works."""
    def fake_tts(text, output_path, **kwargs):
        _write_tiny_wav(output_path)
        return {"ok": True}

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

    pipe = _SyncSentencePipeline.__new__(_SyncSentencePipeline)
    pipe._stop = threading.Event()
    result = pipe._synthesize_to_tmp("speak this aloud")
    assert result is not None
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 0
    os.unlink(result)


def test_synthesize_to_tmp_returns_none_when_stop_set(monkeypatch):
    """``_synthesize_to_tmp`` bails out to ``None`` when stop_event is set."""
    def fake_tts(text, output_path, **kwargs):
        _write_tiny_wav(output_path)
        return {"ok": True}

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

    pipe = _SyncSentencePipeline.__new__(_SyncSentencePipeline)
    pipe._stop = threading.Event()
    pipe._stop.set()
    assert pipe._synthesize_to_tmp("speak this aloud") is None


def test_synthesize_to_tmp_unlinks_tmp_on_failure(monkeypatch):
    """A synthesis exception is swallowed, the temp file is removed, None returned."""
    def failing_tts(text, output_path, **kwargs):
        raise RuntimeError("synthesis exploded")

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", failing_tts)

    pipe = _SyncSentencePipeline.__new__(_SyncSentencePipeline)
    pipe._stop = threading.Event()
    assert pipe._synthesize_to_tmp("speak this aloud") is None


# ---------------------------------------------------------------------------
# stream_tts_to_speaker — sync fallback
# ---------------------------------------------------------------------------

def test_stream_empty_input_flush(fake_voice_mode, monkeypatch):
    """A lone ``None`` sentinel flushes an empty buffer: nothing spoken, done set."""
    played = fake_voice_mode
    _patch_sync_fallback(monkeypatch)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    assert played == []
    _settled_assert(baseline)


def test_stream_sync_speaks_and_drains_remaining(fake_voice_mode, monkeypatch):
    """Sync fallback speaks buffered sentences on the sentinel and drains leftovers."""
    played = fake_voice_mode
    _patch_sync_fallback(monkeypatch)
    synth: list[str] = []

    def fake_tts(text, output_path, **kwargs):
        synth.append(text)
        _write_tiny_wav(output_path)
        return {"ok": True}

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    displayed: list[str] = []
    baseline = threading.active_count()

    text_queue.put("The first sentence is long enough to speak. And the second one too, here it is.")
    text_queue.put(None)
    text_queue.put("ignored-after-sentinel")  # drained (not fed) at the end

    stream_tts_to_speaker(text_queue, stop, done, display_callback=displayed.append)

    assert done.is_set()
    assert displayed
    assert len(synth) >= 1
    assert len(played) == len(synth)
    _settled_assert(baseline)


def test_stream_long_buffer_idle_flush(fake_voice_mode, monkeypatch):
    """A long buffer without a sentence boundary is flushed on queue idle."""
    played = fake_voice_mode
    _patch_sync_fallback(monkeypatch)

    def fake_tts(text, output_path, **kwargs):
        _write_tiny_wav(output_path)
        return {"ok": True}

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()

    def producer():
        text_queue.put("x" * 150)  # no sentence boundary; > long_flush_len(100)
        time.sleep(1.0)
        text_queue.put(None)

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()
    stream_tts_to_speaker(text_queue, stop, done)
    prod.join(timeout=5.0)

    assert done.is_set()
    assert len(played) == 1
    _settled_assert(baseline)


def test_stream_sync_skips_duplicate_sentences(fake_voice_mode, monkeypatch):
    """Near-identical repeated sentences should be spoken only once."""
    played = fake_voice_mode
    _patch_sync_fallback(monkeypatch)

    def fake_tts(text, output_path, **kwargs):
        _write_tiny_wav(output_path)
        return {"ok": True}

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()

    text_queue.put("Repeat sentence here. Repeat sentence here.")
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    assert len(played) == 1
    _settled_assert(baseline)


def test_stream_barge_in_stops_playback(fake_voice_mode, monkeypatch):
    """Setting stop_event mid-stream cancels pending playback and still sets done."""
    played = fake_voice_mode
    _patch_sync_fallback(monkeypatch)

    synth_started = threading.Event()
    release = threading.Event()

    def gated_synth(self, cleaned):
        synth_started.set()
        release.wait(timeout=5.0)
        return _make_tmp_wav_path()

    monkeypatch.setattr(_SyncSentencePipeline, "_synthesize_to_tmp", gated_synth)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()

    # One full sentence (so chunker produces it) plus a trailing fragment buffered.
    text_queue.put("First sentence right here. Second sentence goes here too.")

    th = threading.Thread(
        target=stream_tts_to_speaker, args=(text_queue, stop, done), daemon=True
    )
    th.start()

    assert synth_started.wait(3.0), "synthesis never started"
    stop.set()          # barge in before the pending sentence plays
    release.set()       # let the gated synthesis finish
    th.join(timeout=6.0)

    assert not th.is_alive()
    assert done.is_set()
    assert played == []  # nothing played after barge-in
    _settled_assert(baseline)


def test_stream_sets_done_when_synth_raises(fake_voice_mode, monkeypatch):
    """Even when synthesis raises, tts_done_event is set in the finally block."""
    played = fake_voice_mode
    _patch_sync_fallback(monkeypatch)

    def raising_synth(self, cleaned):
        raise RuntimeError("synthesis exploded")

    monkeypatch.setattr(_SyncSentencePipeline, "_synthesize_to_tmp", raising_synth)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()

    text_queue.put("A sufficiently long sentence to trigger per sentence processing.")
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    assert played == []
    _settled_assert(baseline)


def test_stream_sets_done_when_config_load_raises(fake_voice_mode, monkeypatch):
    """A failure inside the try block still lands in finally and sets done."""
    played = fake_voice_mode

    def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(tts_tool, "_load_tts_config", boom)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    assert played == []
    _settled_assert(baseline)


# ---------------------------------------------------------------------------
# stream_tts_to_speaker — chunked streamer on macOS (output_stream stays None)
# ---------------------------------------------------------------------------

def test_stream_chunked_darwin_plays_via_tempfile(fake_voice_mode, monkeypatch):
    """A chunked streamer with no real output stream plays through temp WAV."""
    played = fake_voice_mode
    streamer = _FakeStreamer()
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(
        tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: streamer
    )
    # Force a tiny per-request cap so the truncation path is exercised.
    monkeypatch.setattr(
        tts_tool, "_resolve_max_text_length", lambda provider, tts_config=None, **kw: 10
    )

    # Force the macOS output_stream=None route on any OS: with system()=="Darwin"
    # the code never creates an OutputStream and plays through temp WAVs.
    monkeypatch.setattr(tts_tool.platform, "system", lambda: "Darwin")

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    displayed: list[str] = []
    baseline = threading.active_count()

    text_queue.put("This is a very long sentence for the streaming provider to speak aloud.")
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done, display_callback=displayed.append)

    assert done.is_set()
    assert streamer.streamed, "stream() was never called"
    # Truncated to the 10-char cap.
    assert len(streamer.streamed[0]) == 10
    assert played, "chunked path should have played one tempfile"
    assert displayed
    _settled_assert(baseline)


class _GatedStreamer:
    """Streamer whose generator blocks mid-stream until released.

    Used to drive the prefetch-cancellation branch: the consumer thread yields
    one chunk, signals, then waits on *gate*; once ``stop_event`` is set and
    *gate* is released the next yield trips the mid-sentence cancellation.
    """

    sample_rate = 24000
    channels = 1

    def __init__(self):
        self.streamed: list[str] = []
        self.first_yielded = threading.Event()
        self.gate = threading.Event()

    def stream(self, text):
        self.streamed.append(text)

        def _gen():
            yield b"\x11\x00"
            self.first_yielded.set()
            self.gate.wait(5.0)
            yield b"\x22\x00"
            yield b"\x33\x00"

        return _gen()


def _patch_numpy(monkeypatch):
    """Stub ``numpy`` so the PortAudio worker can ``import`` it and call
    ``frombuffer(...).reshape(-1, 1)`` without a real numpy install.  The fake
    array only needs to carry the raw bytes so the fake stream's ``write`` can
    record them."""

    class _FBArr:
        def __init__(self, raw):
            self.raw = raw

        def reshape(self, *a):
            return self

    nm = types.ModuleType("numpy")
    nm.frombuffer = lambda buf, dtype="<i2": _FBArr(buf)
    monkeypatch.setitem(sys.modules, "numpy", nm)


def _patch_linux_output_stream(monkeypatch, streamer, sd, *, max_len=1000):
    """Enter the non-Darwin chunked path: force ``platform.system()`` to return
    ``"Linux"`` so ``stream_tts_to_speaker`` creates a real OutputStream, and
    supply a fake sounddevice module plus a numpy stub for the worker."""
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(
        tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: streamer
    )
    monkeypatch.setattr(tts_tool, "_import_sounddevice", lambda: sd)
    monkeypatch.setattr(tts_tool.platform, "system", lambda: "Linux")
    monkeypatch.setattr(tts_tool, "_resolve_max_text_length", lambda *a, **k: max_len)
    _patch_numpy(monkeypatch)


# ---------------------------------------------------------------------------
# _strip_markdown_for_tts — legacy regex fallback
# ---------------------------------------------------------------------------

def test_strip_markdown_uses_regex_fallback(monkeypatch):
    """When the shared normalizer raises, the legacy regex pipeline runs and
    strips markdown / think blocks / emoji, rewrites the link label, and drops
    bare URLs."""
    norm = types.ModuleType("tools.tts_text_normalize")

    def _boom(text, **kwargs):
        raise RuntimeError("normalizer unavailable")

    norm.prepare_spoken_text = _boom
    monkeypatch.setitem(sys.modules, "tools.tts_text_normalize", norm)

    src = (
        "**Bold** and [a link](http://example.com) then "
        "https://plain.url `code` and *ital*."
        "\n\n\n# Heading\n- item\n--- 😀 <think>hidden</think>"
    )
    out = tts_tool._strip_markdown_for_tts(src)

    assert out == out.strip()
    assert "Bold" in out
    assert "a link" in out
    assert "https://" not in out and "plain.url" not in out
    assert "<think" not in out and "hidden" not in out
    assert "😀" not in out
    assert out


# ---------------------------------------------------------------------------
# stream_tts_to_speaker — chunked streamer with a live (non-Darwin) output stream
# ---------------------------------------------------------------------------

def test_stream_chunked_with_output_stream_writes_pcm(fake_voice_mode, monkeypatch):
    """Non-Darwin chunked path: PCM frames are written through a live
    OutputStream (no temp-file playback) and the stream is closed in finally."""
    played = fake_voice_mode
    streamer = _FakeStreamer()
    created = []

    class _FakeStream:
        def __init__(self, samplerate, channels, dtype):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.started = False
            self.stopped = False
            self.closed = False
            self.writes: list[bytes] = []
            created.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

        def write(self, arr):
            self.writes.append(arr.raw)

    sd = types.ModuleType("sounddevice")
    sd.OutputStream = _FakeStream
    _patch_linux_output_stream(monkeypatch, streamer, sd)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()
    text_queue.put("A sentence long enough for the output-stream path to speak.")
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    assert created, "no OutputStream was created"
    assert created[0].started
    assert created[0].writes, "worker never wrote PCM to the stream"
    assert created[0].closed, "output stream was not closed in finally"
    assert played == [], "temp-file playback should not run with a live stream"
    _settled_assert(baseline)


def test_stream_chunked_recreates_stream_on_write_error(fake_voice_mode, monkeypatch):
    """A PortAudio write failure triggers a stream reinit and a retry."""
    played = fake_voice_mode
    streamer = _FakeStreamer()
    created = []

    class _FakeStream:
        def __init__(self, samplerate, channels, dtype):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.started = False
            self.stopped = False
            self.closed = False
            self.writes = []
            created.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

        def write(self, arr):
            # The first stream fails on its very first write; the recreated
            # stream works, so the retry (and every later frame) succeeds.
            if len(created) == 1 and not self.writes:
                self.writes.append("FAIL")
                raise OSError("device busy")
            self.writes.append(arr.raw)

    sd = types.ModuleType("sounddevice")
    sd.OutputStream = _FakeStream
    _patch_linux_output_stream(monkeypatch, streamer, sd)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()
    text_queue.put("A sentence that hits a transient device error then recovers.")
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    assert len(created) >= 2, "stream was not recreated on write error"
    assert created[1].writes, "recreated stream never received audio"
    assert created[1].closed
    _settled_assert(baseline)


def test_stream_chunked_falls_back_to_tempfile_when_reinit_fails(
    fake_voice_mode, monkeypatch
):
    """When the stream cannot be recreated, the remaining audio plays via a
    temp WAV file instead of the (broken) device."""
    played = fake_voice_mode
    streamer = _FakeStreamer()
    created = []

    class _FakeStream:
        def __init__(self, samplerate, channels, dtype):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.started = False
            self.stopped = False
            self.closed = False
            self.writes = []
            created.append(self)

        def start(self):
            if len(created) == 1:
                self.started = True
            else:
                raise OSError("cannot reinitialise device")

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

        def write(self, arr):
            self.writes.append(arr.raw)
            raise OSError("device gone")

    sd = types.ModuleType("sounddevice")
    sd.OutputStream = _FakeStream
    _patch_linux_output_stream(monkeypatch, streamer, sd)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()
    # Two sentences: the first exhausts the reinit budget (each write fails and
    # recreating the stream also fails), leaving the worker's _current_stream as
    # None so the SECOND segment takes the tempfile playback branch.
    text_queue.put("First sentence right here. Second sentence goes here too.")
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    assert played, "temp-file fallback never played the remaining audio"
    _settled_assert(baseline)


def test_stream_chunked_streamer_raise_is_swallowed(fake_voice_mode, monkeypatch):
    """A ``streamer.stream()`` exception is logged, the sentence is skipped,
    and ``tts_done_event`` is still set in finally."""
    played = fake_voice_mode

    class _RaisingStreamer:
        sample_rate = 24000
        channels = 1

        def stream(self, text):
            raise RuntimeError("streamer exploded")

    streamer = _RaisingStreamer()
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(
        tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: streamer
    )

    def _raise_max(p, t=None, **kw):
        raise RuntimeError("resolve boom")

    monkeypatch.setattr(tts_tool, "_resolve_max_text_length", _raise_max)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()
    text_queue.put("Sentence to trigger the failing streamer.")
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    assert played == []
    _settled_assert(baseline)


def test_stream_prefetch_cancels_on_stop(fake_voice_mode, monkeypatch):
    """Setting stop_event mid-prefetch cancels the in-flight segment (partial
    audio only) and nothing further is played."""
    played = fake_voice_mode
    streamer = _GatedStreamer()
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(
        tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: streamer
    )
    monkeypatch.setattr(tts_tool, "_resolve_max_text_length", lambda *a, **k: 1000)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()
    # A single run-on fragment (> long_flush_len) with no sentence boundary so
    # the idle-producer flush emits one sentence -> _enqueue_audio -> stream().
    text_queue.put("streaming " * 30)

    th = threading.Thread(
        target=stream_tts_to_speaker, args=(text_queue, stop, done), daemon=True
    )
    th.start()

    assert streamer.first_yielded.wait(3.0), "prefetch never started"
    stop.set()
    streamer.gate.set()
    th.join(timeout=6.0)

    assert not th.is_alive()
    assert done.is_set()
    _settled_assert(baseline)


def test_stream_finally_closes_sync_pipeline_even_on_error(fake_voice_mode, monkeypatch):
    """Even if the sync pipeline's ``close`` raises, the finally block still
    sets ``tts_done_event``."""
    played = fake_voice_mode
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(
        tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: None
    )

    class _CloseRaisingPipeline:
        def __init__(self, stop_event, *, lookahead=2):
            self.stop = stop_event

        def speak(self, cleaned):
            return None

        def close(self):
            raise RuntimeError("close boom")

    monkeypatch.setattr(tts_tool, "_SyncSentencePipeline", _CloseRaisingPipeline)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()
    text_queue.put("A sentence for the sync pipeline to speak.")
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    _settled_assert(baseline)


# ---------------------------------------------------------------------------
# stream_tts_to_speaker — temp-file playback error handling
# ---------------------------------------------------------------------------

def test_stream_play_via_tempfile_handles_playback_error(monkeypatch):
    """A failing ``play_audio_file`` inside the temp-file fallback is swallowed
    and the pipeline still completes."""
    mod = types.ModuleType("tools.voice_mode")

    def _boom(path):
        raise RuntimeError("no audio player")

    mod.play_audio_file = _boom
    mod.mark_audio_output_active = lambda _active: None
    monkeypatch.setitem(sys.modules, "tools.voice_mode", mod)

    streamer = _FakeStreamer()
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(
        tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: streamer
    )
    monkeypatch.setattr(tts_tool, "_resolve_max_text_length", lambda *a, **k: 1000)

    text_queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    baseline = threading.active_count()
    text_queue.put("A sentence whose playback is going to fail.")
    text_queue.put(None)

    stream_tts_to_speaker(text_queue, stop, done)

    assert done.is_set()
    _settled_assert(baseline)


# ---------------------------------------------------------------------------
# __main__ diagnostics block (run as a subprocess)
# ---------------------------------------------------------------------------

def test_main_diagnostics_block_runs():
    """Running ``tools/tts_tool.py`` as a script prints the diagnostics block
    and exits cleanly."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)

    proc = subprocess.run(
        [sys.executable, "tools/tts_tool.py"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Text-to-Speech Tool Module" in proc.stdout
    assert "Provider availability" in proc.stdout
    assert "Configured provider" in proc.stdout
