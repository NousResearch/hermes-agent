"""Unit tests for the off-device voice loop primitives.

Pure/hermetic: numpy only, no sockets, no models, no audio devices.
Covers the VAD trip/bleed behaviour and the :class:`_NetworkMicStream` framing +
endpointing shim.
"""

from __future__ import annotations

import json
import threading
import wave

import numpy as np
import pytest

from hermes_cli.web_routers._converse_loop import _NetworkMicStream, ConverseSession
from tools import voice_mode as vm


# ── _BargeDetector (via voice_mode; ConverseSession mirrors its wiring) ──

def _make_detector():
    return vm._BargeDetector(
        np, mult=vm.DEFAULT_BARGE_MULTIPLIER,
        calib_blocks=max(1, 450 // 30), trip_blocks=max(1, 300 // 30),
        grace_blocks=max(0, 500 // 30),
    )


def test_barge_detector_trips_on_speech_after_quiet_floor():
    det = _make_detector()
    # Calibrate on a quiet room (RMS well under the silence threshold).
    for _ in range(20):
        assert det.feed(80.0, playing=False) is None
    # A sustained burst of speech-level RMS (>> floor * multiplier) must trip.
    tripped = None
    for _ in range(20):
        phase = det.feed(6000.0, playing=False)
        if phase is not None:
            tripped = phase
            break
    assert tripped == "generation"


def test_barge_detector_ignores_bleed_during_grace_window():
    det = _make_detector()
    for _ in range(20):
        det.feed(80.0, playing=False)
    # Playback starts -> grace window opens. Speaker bleed (moderate RMS, below
    # the playback trigger clamp) during the grace window must NOT trip.
    tripped = False
    for _ in range(det.grace_blocks):
        # Bleed-level: above the quiet floor but below PLAYBACK_MIN_TRIGGER.
        if det.feed(1200.0, playing=True) is not None:
            tripped = True
    assert not tripped


# ── _NetworkMicStream framing ──

def test_network_mic_stream_reads_exact_blocks_from_odd_chunks():
    stop = threading.Event()
    stream = _NetworkMicStream(np, stop=stop)
    # Feed 1000 int16 samples in awkward byte-sized chunks.
    samples = np.arange(1000, dtype=np.int16)
    raw = samples.tobytes()
    for start in range(0, len(raw), 7):  # 7-byte chunks split samples across feeds
        stream.feed(raw[start:start + 7])

    block = 480
    got = []
    for _ in range(2):
        data, overflow = stream.read(block)
        assert overflow is False
        assert data.dtype == np.int16
        assert data.shape == (block,)
        got.append(data)
    # The two 480-sample reads reproduce the first 960 samples in order.
    np.testing.assert_array_equal(np.concatenate(got), samples[:960])


def test_network_mic_stream_zero_pads_and_returns_on_stop():
    stop = threading.Event()
    stream = _NetworkMicStream(np, stop=stop, poll_seconds=0.01)
    stream.feed(np.arange(100, dtype=np.int16).tobytes())
    stop.set()
    data, overflow = stream.read(480)
    assert overflow is False
    assert data.shape == (480,)
    # First 100 samples preserved, remainder zero-padded.
    np.testing.assert_array_equal(data[:100], np.arange(100, dtype=np.int16))
    assert np.all(data[100:] == 0)


# ── endpointing over the shim (canned speech-then-silence) ──

def test_capture_until_quiet_produces_wav_from_shim(tmp_path):
    stop = threading.Event()
    stream = _NetworkMicStream(np, stop=stop, poll_seconds=0.01)
    block = int(vm.SAMPLE_RATE * 0.03)  # 480

    # A few speech-level blocks then enough silence to endpoint.
    speech = (np.ones(block, dtype=np.int16) * 8000)
    silence = np.zeros(block, dtype=np.int16)
    endpoint_blocks = max(1, 1250 // 30)
    stream.feed(speech.tobytes())
    stream.feed(speech.tobytes())
    for _ in range(endpoint_blocks + 2):
        stream.feed(silence.tobytes())

    from collections import deque

    pre_roll: deque = deque(maxlen=4)
    wav_path = vm._capture_until_quiet(
        stream, np, block, pre_roll,
        endpoint_blocks=endpoint_blocks, max_blocks=max(1, 30_000 // 30),
    )
    assert wav_path
    with wave.open(wav_path, "rb") as wf:
        assert wf.getframerate() == vm.SAMPLE_RATE
        assert wf.getnchannels() == 1
        assert wf.getnframes() > 0
    vm._unlink_quietly(wav_path)


# ── ConverseSession barge-in / playing flag ──

def test_converse_session_barge_in_sets_interrupt_and_stops_tts():
    session = ConverseSession(np)
    tts_stop = threading.Event()
    session.set_playing(True, tts_stop=tts_stop)
    assert session.playing() is True
    session._trigger_barge_in()
    assert tts_stop.is_set()
    assert session.take_interrupted() is True
    # Popping the flag clears it.
    assert session.take_interrupted() is False
    assert session.playing() is False


def test_converse_session_stop_pushes_sentinel():
    session = ConverseSession(np)
    session.stop()
    assert session.stopped is True
    assert session.transcripts.get_nowait() is None


# ── converse synthesizer: streaming path + one-shot fallback ──

from tools.voice_converse_loop import (
    _decode_audio_file_to_pcm16, resolve_converse_synthesizer)


def _write_wav(path, *, rate=24000, ms=50, freq=440.0):
    """Write a tiny mono s16 WAV (a short sine tone) and return its sample count."""
    n = int(rate * ms / 1000)
    t = np.arange(n, dtype=np.float64) / rate
    samples = (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return n


def test_decode_audio_file_to_pcm16_roundtrips_wav(tmp_path):
    wav = tmp_path / "tone.wav"
    n = _write_wav(wav, rate=24000, ms=50)
    pcm = _decode_audio_file_to_pcm16(str(wav), target_rate=24000)
    assert pcm  # non-empty
    assert len(pcm) % 2 == 0  # int16-aligned
    # ~n samples out (allow a little slack for resampler edge frames).
    got = len(pcm) // 2
    assert abs(got - n) <= max(64, n // 10)


def test_decode_audio_file_to_pcm16_bad_file_returns_empty(tmp_path):
    bad = tmp_path / "not-audio.bin"
    bad.write_bytes(b"not an audio file")
    assert _decode_audio_file_to_pcm16(str(bad)) == b""


def test_resolve_converse_synthesizer_uses_streaming_when_available(monkeypatch):
    class _FakeStreamer:
        sample_rate = 24000

        def stream(self, text):
            yield b"\x01\x02\x03\x04"

    monkeypatch.setattr(
        "tools.tts_streaming.resolve_streaming_provider", lambda cfg: _FakeStreamer())
    synth = resolve_converse_synthesizer({})
    assert synth is not None
    assert synth.sample_rate == 24000
    assert list(synth.synth("hi")) == [b"\x01\x02\x03\x04"]


def test_resolve_converse_synthesizer_falls_back_to_one_shot(monkeypatch, tmp_path):
    # No streamer -> one-shot fallback that transcodes text_to_speech_tool's file.
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: None)

    wav = tmp_path / "reply.wav"
    _write_wav(wav, rate=24000, ms=40)

    def _fake_tts(text, *a, **k):
        return json.dumps({"success": True, "file_path": str(wav)})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", _fake_tts)

    synth = resolve_converse_synthesizer({})
    assert synth is not None
    assert synth.sample_rate == 24000
    pcm = b"".join(synth.synth("say something"))
    assert pcm and len(pcm) % 2 == 0
    # The temp file was unlinked after synthesis.
    assert not wav.exists()


def test_one_shot_fallback_yields_nothing_on_provider_failure(monkeypatch):
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: None)
    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        lambda text, *a, **k: json.dumps({"success": False, "error": "nope"}))
    synth = resolve_converse_synthesizer({})
    assert list(synth.synth("x")) == []


# ── output_rate resampling wrapper ──

from tools.voice_converse_loop import resample_synth


class _ToneSynth:
    """A synth that yields a fixed number of int16 mono samples at ``sample_rate``."""

    def __init__(self, sample_rate, n_samples=4800):
        self.sample_rate = sample_rate
        self._n = n_samples

    def synth(self, text):
        t = np.arange(self._n, dtype=np.float64) / self.sample_rate
        pcm = (np.sin(2 * np.pi * 220.0 * t) * 10000).astype(np.int16).tobytes()
        # Emit in a couple of chunks (and on an odd byte boundary) to exercise buffering.
        yield pcm[:1001]
        yield pcm[1001:]


def test_resample_synth_is_noop_when_rates_match():
    inner = _ToneSynth(24000)
    wrapped = resample_synth(inner, 24000)
    assert wrapped is inner  # identity — no wrapper allocated
    assert wrapped.sample_rate == 24000


def test_resample_synth_changes_byte_length_at_new_rate():
    inner = _ToneSynth(24000, n_samples=4800)  # 4800 samples @24k = 200 ms
    src_bytes = len(b"".join(inner.synth("hi")))
    wrapped = resample_synth(inner, 12000)  # half rate -> ~half the samples/bytes
    assert wrapped.sample_rate == 12000
    out_bytes = len(b"".join(wrapped.synth("hi")))
    assert out_bytes % 2 == 0  # int16-aligned
    # 200 ms at 12 kHz ≈ 2400 samples = 4800 bytes; allow slack for resampler edge frames.
    assert abs(out_bytes - src_bytes // 2) <= max(256, src_bytes // 10)
    assert out_bytes < src_bytes  # downsampling actually shrank the stream


def test_resample_synth_upsamples_to_more_bytes():
    inner = _ToneSynth(16000, n_samples=3200)  # 200 ms @16k
    src_bytes = len(b"".join(inner.synth("hi")))
    wrapped = resample_synth(inner, 24000)  # 1.5x rate -> ~1.5x bytes
    out_bytes = len(b"".join(wrapped.synth("hi")))
    assert out_bytes > src_bytes


# ── shared turn driver: history cap + control-frame ordering ──

import asyncio
import queue as _queue

from tools.voice_converse_loop import _HISTORY_MAX_MESSAGES, drive_converse_turns


class _FakeConverseSession:
    """Minimal ConverseSession stand-in for driving drive_converse_turns hermetically.

    Serves a fixed list of transcripts (then the None shutdown sentinel), and no-ops
    the playback/barge-in coordination so a turn runs start-to-finish without VAD.
    """

    def __init__(self, transcripts):
        self.transcripts = _queue.Queue()
        for t in transcripts:
            self.transcripts.put(t)
        self.transcripts.put(None)  # shutdown sentinel
        self.stopped = False

    def take_interrupted(self):
        return False

    def set_playing(self, value, *, tts_stop=None):
        return None


class _EchoSynth:
    sample_rate = 24000

    def synth(self, text):
        yield text.encode("utf-8")


def _run_driver(session, history, replies, idle_interval=0.0):
    """Drive drive_converse_turns to completion, returning the JSON frames sent."""
    sent: list = []

    async def _send_json(obj):
        sent.append(obj)

    async def _send_bytes(data):
        sent.append(("bytes", data))

    reply_iter = iter(replies)

    async def _run_turn(transcript, on_delta, *, interrupted):
        reply = next(reply_iter)
        on_delta(reply)  # stream the whole reply as one delta
        return reply, None

    async def _main():
        loop = asyncio.get_running_loop()
        await drive_converse_turns(
            session=session, synth=_EchoSynth(), cap=4000, loop=loop,
            send_json=_send_json, send_bytes=_send_bytes,
            run_turn=_run_turn, history=history, idle_interval=idle_interval)

    asyncio.run(_main())
    return sent


def test_drive_converse_turns_control_frame_ordering():
    session = _FakeConverseSession(["hello there."])
    history: list = []
    sent = _run_driver(session, history, ["Hi back."])

    # transcript -> speaking -> PCM bytes -> turn_done, in that exact order.
    types = [f.get("type") if isinstance(f, dict) else "bytes" for f in sent]
    assert types == ["transcript", "speaking", "bytes", "turn_done"]
    assert sent[0] == {"type": "transcript", "text": "hello there."}
    assert sent[2] == ("bytes", b"Hi back.")
    # The turn was recorded in history (user + assistant).
    assert history == [
        {"role": "user", "content": "hello there."},
        {"role": "assistant", "content": "Hi back."},
    ]


def test_drive_converse_turns_caps_history_tail():
    # Run enough turns that user+assistant messages would exceed the cap, and assert
    # the driver keeps only the last _HISTORY_MAX_MESSAGES (a bounded tail).
    n_turns = _HISTORY_MAX_MESSAGES  # 2 messages/turn -> 2N appended, well over the cap
    transcripts = [f"say {i}." for i in range(n_turns)]
    replies = [f"reply {i}." for i in range(n_turns)]
    history: list = []
    _run_driver(_FakeConverseSession(transcripts), history, replies)

    assert len(history) == _HISTORY_MAX_MESSAGES
    # The tail is retained: the last kept message is the final assistant reply.
    assert history[-1] == {"role": "assistant", "content": f"reply {n_turns - 1}."}
    assert history[0]["role"] == "user"  # cap preserves whole (user, assistant) pairs here


def test_drive_converse_turns_emits_idle_and_keeps_socket_open():
    # Session mode: no new utterance for a while. The driver emits periodic
    # {"type":"idle"} with a growing quiet_seconds (it never closes the socket) until a
    # real frame arrives; here the shutdown sentinel ends the loop after a few intervals.
    session = _FakeConverseSession([])
    session.transcripts = _queue.Queue()  # empty; no sentinel yet
    threading.Timer(0.17, lambda: session.transcripts.put(None)).start()

    sent = _run_driver(session, [], [], idle_interval=0.05)

    idles = [f for f in sent if isinstance(f, dict) and f.get("type") == "idle"]
    assert len(idles) >= 1
    quiets = [f["quiet_seconds"] for f in idles]
    assert all(isinstance(q, (int, float)) for q in quiets)
    assert quiets == sorted(quiets)  # cumulative quiet grows across pings
    assert not any(isinstance(f, dict) and f.get("type") == "turn_done" for f in sent)


def test_drive_converse_turns_stop_word_ends_exchange(monkeypatch):
    # Session mode: a spoken stop phrase is announced as {"type":"stop_word"} and the
    # agent turn is SKIPPED (the client decides to re-arm/sleep).
    monkeypatch.setattr(
        "tools.voice_mode_transcript.is_voice_stop_phrase",
        lambda t: t.strip().lower() == "goodbye")
    session = _FakeConverseSession(["goodbye"])
    history: list = []
    sent = _run_driver(session, history, ["SHOULD NOT RUN"], idle_interval=1.0)

    types = [f.get("type") if isinstance(f, dict) else "bytes" for f in sent]
    assert types == ["transcript", "stop_word"]
    assert sent[1] == {"type": "stop_word", "text": "goodbye"}
    assert history == []  # the stop phrase never became a turn


def test_drive_converse_turns_stop_word_ignored_in_continuous_mode(monkeypatch):
    # Continuous mode (idle_interval=0): stop-word handling is off — "goodbye" is a
    # normal turn with no {"type":"stop_word"}.
    monkeypatch.setattr("tools.voice_mode_transcript.is_voice_stop_phrase", lambda t: True)
    session = _FakeConverseSession(["goodbye"])
    history: list = []
    sent = _run_driver(session, history, ["Bye!"], idle_interval=0.0)

    types = [f.get("type") if isinstance(f, dict) else "bytes" for f in sent]
    assert "stop_word" not in types
    assert types == ["transcript", "speaking", "bytes", "turn_done"]


def test_drive_converse_turns_caps_tts_output():
    # A runaway reply must not be spoken in full: the driver caps the synthesized audio
    # near _MAX_TTS_CHARS_PER_TURN, while the full reply is still recorded in history.
    from tools.voice_converse_loop import _MAX_TTS_CHARS_PER_TURN

    long_reply = "This is a spoken sentence. " * 120  # ~3200 chars, well over the cap
    session = _FakeConverseSession(["say a lot."])
    history: list = []
    sent = _run_driver(session, history, [long_reply])

    pcm = b"".join(d[1] for d in sent if isinstance(d, tuple) and d[0] == "bytes")
    assert len(pcm) < len(long_reply)                       # actually capped
    assert len(pcm) <= _MAX_TTS_CHARS_PER_TURN + 200        # bounded near the cap
    assert history[-1] == {"role": "assistant", "content": long_reply}  # full reply kept
