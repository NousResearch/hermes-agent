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


# ── Smart Turn v3 adaptive endpoint ──

from tools.voice_converse_loop import parse_smart_turn_config  # noqa: E402


class _ScriptedDetector:
    """Stand-in for SmartTurnDetector: returns a scripted P(complete) per call."""

    def __init__(self, probs):
        self._probs = list(probs)
        self.threshold = 0.5
        self.calls = 0
        self.last_len = 0

    def turn_complete_probability(self, audio_f32):
        self.last_len = int(np.asarray(audio_f32).size)
        p = self._probs[min(self.calls, len(self._probs) - 1)]
        self.calls += 1
        return float(p)


def _adaptive_session(detector, *, endpoint_blocks=2, max_blocks=60):
    session = ConverseSession(np)  # endpoint_model defaults off -> no real model load
    session._turn_detector = detector
    session._endpoint_threshold = detector.threshold
    session._endpoint_blocks = endpoint_blocks
    session._max_blocks = max_blocks
    return session


def _feed_silence(session, blocks):
    silence = np.zeros(session._block, dtype=np.int16).tobytes()
    for _ in range(blocks):
        session.stream.feed(silence)


def test_parse_smart_turn_config_defaults_off():
    assert parse_smart_turn_config({}) == (False, 0.5)
    assert parse_smart_turn_config({"voice": {}}) == (False, 0.5)
    assert parse_smart_turn_config({"voice": {"endpoint": {"model": "none"}}})[0] is False
    assert parse_smart_turn_config(None) == (False, 0.5)


def test_parse_smart_turn_config_enabled_with_threshold():
    enabled, thr = parse_smart_turn_config(
        {"voice": {"endpoint": {"model": "smart-turn-v3", "threshold": 0.7}}})
    assert enabled is True
    assert thr == pytest.approx(0.7)
    # threshold clamps into [0, 1]
    assert parse_smart_turn_config(
        {"voice": {"endpoint": {"model": "x", "threshold": 5}}})[1] == 1.0


def test_adaptive_commits_immediately_when_turn_sounds_complete():
    det = _ScriptedDetector([0.95])
    session = _adaptive_session(det)
    _feed_silence(session, session._endpoint_blocks + 4)
    wav_path = session._capture_adaptive(silence_rms=float(vm.SILENCE_RMS_THRESHOLD))
    assert det.calls == 1  # committed on the first candidate pause
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnframes() > 0
    vm._unlink_quietly(wav_path)


def test_adaptive_holds_then_commits_capturing_more_audio():
    # Two "not done yet" verdicts, then complete: the mic stays open across the holds.
    det = _ScriptedDetector([0.1, 0.2, 0.9])
    session = _adaptive_session(det)
    _feed_silence(session, session._endpoint_blocks * 4 + 4)
    wav_path = session._capture_adaptive(silence_rms=float(vm.SILENCE_RMS_THRESHOLD))
    assert det.calls == 3  # held twice, committed on the third check
    vm._unlink_quietly(wav_path)


def test_adaptive_gives_up_after_max_holds():
    # Model never says complete; the hold budget bounds the wait (no run to _max_blocks).
    det = _ScriptedDetector([0.0])
    session = _adaptive_session(det, max_blocks=500)
    _feed_silence(session, session._endpoint_blocks * 6 + 4)
    wav_path = session._capture_adaptive(silence_rms=float(vm.SILENCE_RMS_THRESHOLD))
    from tools.voice_converse_loop import _MAX_ENDPOINT_HOLDS

    assert det.calls == _MAX_ENDPOINT_HOLDS  # committed once the budget was spent
    vm._unlink_quietly(wav_path)


def test_frames_to_f32_resamples_non_16k_capture():
    session = ConverseSession(np, input_rate=48000)
    frames = [np.full(session._block, 4000, dtype=np.int16) for _ in range(10)]
    out = session._frames_to_f32_16k(frames)
    total = session._block * 10
    assert out.dtype == np.float32
    assert abs(out.size - round(total * 16000 / 48000)) <= 1
    assert float(np.max(np.abs(out))) <= 1.0


# ── voice phase tracing ──

from tools import voice_tracing as vt  # noqa: E402


class _FakeSpan:
    def __init__(self, name, kind, start_time):
        self.name, self.kind, self.start_time = name, kind, start_time
        self.attributes, self.events, self.end_time = {}, [], None

    def set_attribute(self, k, v):
        self.attributes[k] = v

    def add_event(self, name, attrs, timestamp=None):
        self.events.append((name, attrs, timestamp))

    def end(self, end_time=None):
        self.end_time = end_time


class _FakeTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, kind=None, start_time=None, context=None):
        s = _FakeSpan(name, kind, start_time)
        s.parent = context
        self.spans.append(s)
        return s


class _Kind:
    SERVER = "server"
    INTERNAL = "internal"


def _patch_tracer(monkeypatch):
    tracer = _FakeTracer()
    monkeypatch.setattr(vt, "_get_tracer", lambda: (tracer, _Kind, lambda span: span))
    return tracer


def test_phase_recorder_times_phases_and_attrs():
    rec = vt.PhaseRecorder(input_rate=16000, session_mode=True)
    with rec.phase("voice.capture", **{"endpoint.mode": "smart_turn"}) as cap:
        cap.event("smart_turn.infer", p=0.2, holds=0)
        cap.set(**{"smart_turn.holds": 1})
    assert rec.phases[0].name == "voice.capture"
    assert rec.phases[0].end_ns >= rec.phases[0].start_ns
    assert rec.phases[0].attrs["endpoint.mode"] == "smart_turn"
    assert rec.phases[0].events[0][0] == "smart_turn.infer"
    assert rec.start_ns is not None and rec.end_ns is not None


def test_emit_turn_trace_builds_parent_and_child_spans(monkeypatch):
    tracer = _patch_tracer(monkeypatch)
    rec = vt.PhaseRecorder(session_mode=False)
    with rec.phase("voice.capture", **{"endpoint.mode": "fixed"}):
        pass
    rec.add_phase("voice.agent", 1_000, 2_000, **{"llm.ttft_ms": 1, "reply.chars": 5})
    rec.set(outcome="ok")
    vt.emit_turn_trace(rec)
    names = [s.name for s in tracer.spans]
    assert names[0] == "voice.turn" and tracer.spans[0].kind == "server"
    assert "voice.capture" in names and "voice.agent" in names
    parent = tracer.spans[0]
    assert parent.attributes["outcome"] == "ok"
    agent = next(s for s in tracer.spans if s.name == "voice.agent")
    assert agent.start_time == 1_000 and agent.end_time == 2_000
    assert agent.attributes["reply.chars"] == 5
    assert all(s.end_time is not None for s in tracer.spans)  # every span closed


def test_emit_turn_trace_noop_without_tracer(monkeypatch):
    monkeypatch.setattr(vt, "_get_tracer", lambda: (None, None, None))
    rec = vt.PhaseRecorder()
    with rec.phase("voice.capture"):
        pass
    vt.emit_turn_trace(rec)  # must not raise
    vt.emit_turn_trace(None)


def test_emit_turn_trace_drops_non_primitive_attributes(monkeypatch):
    tracer = _patch_tracer(monkeypatch)
    rec = vt.PhaseRecorder()
    with rec.phase("voice.capture", ok=True, bad={"nested": 1}, n=3):
        pass
    vt.emit_turn_trace(rec)
    cap = next(s for s in tracer.spans if s.name == "voice.capture")
    assert cap.attributes == {"ok": True, "n": 3}  # dict attr dropped


def test_capture_and_transcribe_records_capture_and_stt_phases(monkeypatch):
    det = _ScriptedDetector([0.9])
    session = _adaptive_session(det)
    _feed_silence(session, session._endpoint_blocks + 4)
    monkeypatch.setattr(
        vm, "transcribe_recording",
        lambda path, model=None: {"success": True, "transcript": "hello there"})
    text = session._capture_and_transcribe()
    assert text == "hello there"
    rec = session.take_recorder()
    phases = {p.name: p for p in rec.phases}
    assert set(phases) == {"voice.capture", "voice.stt"}
    assert phases["voice.capture"].attrs["endpoint.mode"] == "smart_turn"
    assert phases["voice.capture"].attrs["smart_turn.commit_reason"] == "complete"
    # Latency-breakdown attrs for tuning "endpointing is slow": the trailing-silence window,
    # speech time, and hold cost are all exposed on the capture span.
    cap = phases["voice.capture"].attrs
    assert cap["smart_turn.silence_ms"] > 0
    assert "capture.speech_ms" in cap
    assert cap["smart_turn.hold_ms"] == 0  # no holds on this immediate-commit turn
    assert phases["voice.stt"].attrs["stt.transcript_chars"] == len("hello there")
    assert phases["voice.stt"].attrs["stt.success"] is True


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

    def take_recorder(self):
        return None  # no capture-phase recorder in the driver-only fake

    def set_playing(self, value, *, tts_stop=None):
        return None

    def begin_turn(self):
        self.turns_begun = getattr(self, "turns_begun", 0) + 1

    def end_turn(self):
        self.turns_ended = getattr(self, "turns_ended", 0) + 1

    def begin_playback_tail(self, turn_id, fallback_seconds):
        self.tails_begun = getattr(self, "tails_begun", 0) + 1

    def end_playback_tail(self):
        pass

    def notify_drained(self, turn_id=None):
        pass


class _EchoSynth:
    sample_rate = 24000

    def synth(self, text):
        yield text.encode("utf-8")


def _run_driver(session, history, replies, quiet_interval=0.0):
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
            run_turn=_run_turn, history=history, quiet_interval=quiet_interval)

    asyncio.run(_main())
    return sent


def test_driver_uses_bundled_recorder_not_stale_session_slot(monkeypatch):
    # Regression: the trace must be built from the recorder BOUND to the transcript, never a
    # shared session slot the VAD worker may have overwritten with the next (empty) capture while
    # the driver was backed up behind a long turn. Otherwise a real turn emits with no
    # capture/STT spans and a mis-timed parent (the empty-turn artifact in Tempo).
    from tools.voice_converse_loop import _Utterance

    tracer = _patch_tracer(monkeypatch)
    recA = vt.PhaseRecorder(session_mode=False)
    with recA.phase("voice.capture", **{"endpoint.mode": "fixed", "tag": "A"}):
        pass
    with recA.phase("voice.stt", **{"stt.transcript_chars": 5}):
        pass

    session = _FakeConverseSession([])
    session.transcripts = _queue.Queue()
    session.transcripts.put(_Utterance("hello", recA))
    session.transcripts.put(None)
    # A mismatched newer recorder in the slot must be IGNORED (the old bug read this).
    session.take_recorder = lambda: vt.PhaseRecorder(session_mode=False)

    _run_driver(session, [], ["hi"])
    names = [s.name for s in tracer.spans]
    assert "voice.capture" in names and "voice.stt" in names  # from recA, not the empty slot
    cap = next(s for s in tracer.spans if s.name == "voice.capture")
    assert cap.attributes.get("tag") == "A"
    assert "voice.agent" in names  # driver appended agent/TTS to the SAME (bound) recorder


def test_drive_converse_turns_control_frame_ordering():
    session = _FakeConverseSession(["hello there."])
    history: list = []
    sent = _run_driver(session, history, ["Hi back."])

    # transcript -> thinking -> speaking -> PCM bytes -> turn_done, in that exact order.
    types = [f.get("type") if isinstance(f, dict) else "bytes" for f in sent]
    assert types == ["transcript", "thinking", "speaking", "bytes", "turn_done"]
    assert sent[0]["type"] == "transcript" and sent[0]["text"] == "hello there."
    assert sent[3] == ("bytes", b"Hi back.")
    # Every turn frame carries the same turn_id (correlation id, present even with tracing off).
    turn_ids = {f["turn_id"] for f in sent if isinstance(f, dict) and "turn_id" in f}
    assert len(turn_ids) == 1
    assert sent[0]["turn_id"] and sent[-1]["turn_id"] == sent[0]["turn_id"]
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


def test_drive_converse_turns_brackets_the_turn_with_begin_end():
    # The driver must suppress quiet accrual around a turn: begin_turn before running it,
    # end_turn after turn_done. A stop-word (no turn) must do neither.
    session = _FakeConverseSession(["hello there."])
    _run_driver(session, [], ["Hi back."])
    assert getattr(session, "turns_begun", 0) == 1
    assert getattr(session, "turns_ended", 0) == 1


def test_drive_converse_turns_forwards_quiet_ticks_and_keeps_socket_open():
    # Session mode: the SESSION (not a wall-clock timeout in the driver) emits QuietTick
    # markers as the RECEIVED stream stays silent. The driver forwards each as
    # {"type":"quiet", quiet_seconds} and never closes the socket, until the shutdown sentinel.
    from tools.voice_converse_loop import QuietTick

    session = _FakeConverseSession([])
    session.transcripts = _queue.Queue()
    for q in (0.5, 1.0, 1.5):
        session.transcripts.put(QuietTick(q))
    session.transcripts.put(None)  # shutdown sentinel ends the loop

    sent = _run_driver(session, [], [], quiet_interval=0.5)

    quiets = [f for f in sent if isinstance(f, dict) and f.get("type") == "quiet"]
    assert [f["quiet_seconds"] for f in quiets] == [0.5, 1.0, 1.5]  # forwarded verbatim, in order
    assert not any(isinstance(f, dict) and f.get("type") == "turn_done" for f in sent)


def test_drive_converse_turns_stop_word_ends_exchange(monkeypatch):
    # Session mode: a spoken stop phrase is announced as {"type":"stop_word"} and the
    # agent turn is SKIPPED (the client decides to re-arm/sleep).
    monkeypatch.setattr(
        "tools.voice_mode_transcript.is_voice_stop_phrase",
        lambda t: t.strip().lower() == "goodbye")
    session = _FakeConverseSession(["goodbye"])
    history: list = []
    sent = _run_driver(session, history, ["SHOULD NOT RUN"], quiet_interval=1.0)

    types = [f.get("type") if isinstance(f, dict) else "bytes" for f in sent]
    assert types == ["transcript", "stop_word"]
    assert sent[1]["type"] == "stop_word" and sent[1]["text"] == "goodbye"
    assert history == []  # the stop phrase never became a turn
    assert getattr(session, "turns_begun", 0) == 0  # a stop-word does not begin a turn


def test_drive_converse_turns_times_out_hung_turn(monkeypatch):
    # A hung agent turn (LLM call that never returns, no error) must not wedge the socket: the
    # driver abandons it at the deadline and STILL emits an error + turn_done so the client
    # returns to listening instead of "thinking" forever.
    monkeypatch.setattr("tools.voice_converse_loop._resolve_turn_timeout", lambda: 0.3)
    session = _FakeConverseSession(["are you there?"])
    sent: list = []

    async def _sj(o):
        sent.append(o)

    async def _sb(d):
        sent.append(("bytes", d))

    async def _hang(transcript, on_delta, *, interrupted):
        await asyncio.Event().wait()  # never completes
        return "", None

    async def _main():
        loop = asyncio.get_running_loop()
        await drive_converse_turns(
            session=session, synth=_EchoSynth(), cap=4000, loop=loop,
            send_json=_sj, send_bytes=_sb, run_turn=_hang, history=[], quiet_interval=0.0)

    asyncio.run(_main())
    types = [f.get("type") if isinstance(f, dict) else "bytes" for f in sent]
    assert "error" in types              # the hang surfaced as an error
    assert types[-1] == "turn_done"      # and the turn ALWAYS ends


def test_drive_converse_turns_flushes_queue_on_signoff():
    # When the agent signs off (expects_more=False), utterances that queued behind the turn must
    # be dropped, not run — otherwise their replies arrive after the client has slept.
    session = _FakeConverseSession(["are we done?", "a queued leftover"])
    sent = _run_driver(session, [], ["All set. Over and out."], quiet_interval=1.0)

    transcripts = [f["text"] for f in sent if isinstance(f, dict) and f.get("type") == "transcript"]
    assert transcripts == ["are we done?"]          # the leftover never became a turn
    turn_dones = [f for f in sent if isinstance(f, dict) and f.get("type") == "turn_done"]
    assert len(turn_dones) == 1 and turn_dones[0].get("expects_more") is False


def test_driver_tail_barge_emits_interrupted_not_a_turn():
    # A trip during the playback tail arrives as _TailBarge → the driver emits `interrupted`
    # (client stops playback) and runs NO turn (the reply's echo is never transcribed/answered).
    from tools.voice_converse_loop import _TailBarge

    session = _FakeConverseSession([])
    session.transcripts = _queue.Queue()
    session.transcripts.put(_TailBarge("turn-x"))
    session.transcripts.put(None)
    sent = _run_driver(session, [], [])
    types = [f.get("type") if isinstance(f, dict) else "bytes" for f in sent]
    assert types == ["interrupted"]
    assert sent[0] == {"type": "interrupted", "turn_id": "turn-x"}
    assert getattr(session, "turns_begun", 0) == 0  # no agent turn ran


def test_playback_tail_suppresses_quiet_until_drained():
    from tools.voice_converse_loop import QuietTick

    session = ConverseSession(np, quiet_interval=0.5)  # 0.03s of received-silence per call

    def _drain_ticks():
        out = []
        while True:
            try:
                out.append(session.transcripts.get_nowait())
            except Exception:  # noqa: BLE001 - queue.Empty
                break
        return [t for t in out if isinstance(t, QuietTick)]

    session.begin_playback_tail("t1", fallback_seconds=100)
    for _ in range(50):  # 1.5s of received silence — but tail suppresses accrual
        session._account_received_silence()
    assert _drain_ticks() == []                       # no quiet during the playback tail
    session.notify_drained("other")                   # mismatched id: tail stays active
    for _ in range(50):
        session._account_received_silence()
    assert _drain_ticks() == []
    session.notify_drained("t1")                       # correct id ends the tail
    for _ in range(50):
        session._account_received_silence()
    assert len(_drain_ticks()) >= 1                    # quiet resumes after drain


def test_voice_system_prompt_signoff_instruction():
    from tools.voice_converse_loop import voice_system_prompt
    # Session mode (allow_signoff) teaches the model the sign-off; continuous mode does not.
    with_signoff = voice_system_prompt("Sakura", allow_signoff=True)
    assert "Over and out" in with_signoff and "stops listening" in with_signoff
    assert "Over and out" not in voice_system_prompt("Sakura", allow_signoff=False)


def _turn_done_frame(sent):
    return next(f for f in sent if isinstance(f, dict) and f.get("type") == "turn_done")


def test_drive_converse_turns_signoff_sets_expects_more_false():
    # Session mode: the agent ends its reply with the sign-off phrase → turn_done carries
    # expects_more=false so a wake-word client sleeps immediately. No separate frame.
    session = _FakeConverseSession(["are we done?"])
    sent = _run_driver(session, [], ["All set. Over and out."], quiet_interval=1.0)
    assert _turn_done_frame(sent).get("expects_more") is False
    assert not any(isinstance(f, dict) and f.get("type") == "conversation_end" for f in sent)


def _spoken_bytes(sent):
    return b"".join(f[1] for f in sent if isinstance(f, tuple) and f[0] == "bytes")


def test_drive_converse_turns_signoff_is_not_spoken_but_still_flags():
    # Session mode: the sign-off is a control marker — stripped from the spoken audio (the
    # _EchoSynth echoes the TTS text as bytes) while still setting expects_more=false.
    session = _FakeConverseSession(["bye"])
    sent = _run_driver(session, [], ["See you later. Over and out."], quiet_interval=1.0)
    spoken = _spoken_bytes(sent).lower()
    assert b"see you later" in spoken
    assert b"over and out" not in spoken  # not spoken aloud
    assert _turn_done_frame(sent).get("expects_more") is False


def test_drive_converse_turns_signoff_spoken_in_continuous_mode():
    # Continuous mode: no sign-off handling, so the phrase is spoken normally (not a marker).
    session = _FakeConverseSession(["bye"])
    sent = _run_driver(session, [], ["See you later. Over and out."], quiet_interval=0.0)
    assert b"over and out" in _spoken_bytes(sent).lower()


def test_drive_converse_turns_question_sets_expects_more_true():
    # Session mode: the agent asked a follow-up question → turn_done carries expects_more=true
    # so the client keeps the mic hot instead of sleeping on the next quiet.
    session = _FakeConverseSession(["set a timer"])
    sent = _run_driver(session, [], ["Sure — for how long?"], quiet_interval=1.0)
    assert _turn_done_frame(sent).get("expects_more") is True


def test_drive_converse_turns_plain_reply_has_no_expects_more():
    # A plain statement gives no signal: the field is ABSENT (client's quiet timer governs).
    session = _FakeConverseSession(["what's the weather?"])
    sent = _run_driver(session, [], ["It's sunny and 72."], quiet_interval=1.0)
    assert "expects_more" not in _turn_done_frame(sent)


def test_drive_converse_turns_continuous_mode_never_sets_expects_more():
    # Continuous mode (quiet_interval=0): the flag is meaningless, so it's never set, even for a
    # sign-off or a question.
    for reply in ("All set. Over and out.", "For how long?"):
        session = _FakeConverseSession(["hi"])
        sent = _run_driver(session, [], [reply], quiet_interval=0.0)
        assert "expects_more" not in _turn_done_frame(sent)


def test_drive_converse_turns_stop_word_ignored_in_continuous_mode(monkeypatch):
    # Continuous mode (quiet_interval=0): stop-word handling is off — "goodbye" is a
    # normal turn with no {"type":"stop_word"}.
    monkeypatch.setattr("tools.voice_mode_transcript.is_voice_stop_phrase", lambda t: True)
    session = _FakeConverseSession(["goodbye"])
    history: list = []
    sent = _run_driver(session, history, ["Bye!"], quiet_interval=0.0)

    types = [f.get("type") if isinstance(f, dict) else "bytes" for f in sent]
    assert "stop_word" not in types
    assert types == ["transcript", "thinking", "speaking", "bytes", "turn_done"]


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
