"""Local live transcription (tools.voice_partial): session contract, worker behaviour, lock safety.

``stt.streaming`` already answers "transcribe while I speak" for providers with a realtime wire
(openai / xai / elevenlabs). The local faster-whisper backend has no such wire, so this module
answers the same switch offline: re-decode a capped tail of the running capture on a cadence.
Invariants this file pins:

* The session satisfies the live-session contract ``tools.voice_mode`` already drives
  (``set_input_rate`` / ``start`` / ``end_audio`` / ``cancel`` / ``finalize`` / ``provider``).
* It is **preview-only**: ``finalize()`` yields no transcript and ``_park_live_session`` retires
  it instead of parking it, so the take is always transcribed in full.
* It opens only when the profile's STT resolves to local — a provider with its own live wire must
  not get a whisper model loaded behind the user's back. ``stt.streaming`` off = no loop.
* A tick landing while a decode is in flight is dropped, never queued.
* Empty / hallucinated / identical decodes never re-surface.
* The decode holds ``_stt_inference_lock`` across the lazy ``transcribe()`` AND segment iteration,
  and skips its tick while an idle model unload is in flight.
* ``AudioRecorder.tail_wav_path`` returns the NEWEST audio, capped, thread-safe under live appends.
"""

from __future__ import annotations

import array
import os
import sys
import threading
import time
import wave
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import MagicMock

import pytest

if "faster_whisper" not in sys.modules:
    _stub = ModuleType("faster_whisper")
    _stub.WhisperModel = MagicMock(name="WhisperModel")
    _stub.__spec__ = ModuleSpec("faster_whisper", loader=None)
    sys.modules["faster_whisper"] = _stub


class FakeSegment:
    def __init__(self, text: str):
        self.text = text
        self.no_speech_prob = 0.0
        self.avg_logprob = 0.0


class FakeRecorder:
    """Recorder double exposing only what the partial worker is allowed to touch."""

    def __init__(self, tail_paths=None, recording=True):
        self._paths = list(tail_paths or [])
        self.is_recording = recording
        self.elapsed_seconds = 10.0
        self.tail_calls = 0

    def tail_wav_path(self, max_seconds):
        self.tail_calls += 1
        return self._paths.pop(0) if self._paths else None


def _force_local_stt(monkeypatch, provider="local", streaming=True, interval=0.05):
    """Pin the profile's STT resolution + the voice.partial tuning for a deterministic worker.

    ``_partial_cfg`` is patched rather than the config values: production clamps the tuning to
    sane floors (a re-decode is a full whisper pass), and these tests need many ticks in well
    under a second. The clamp itself is pinned by test_tuning_is_clamped_to_usable_floors.
    """
    import hermes_cli.config as hermes_config
    import tools.transcription_tools as tt
    import tools.voice_partial as vp

    cfg = {
        "stt": {"enabled": True, "provider": provider, "streaming": streaming},
        "voice": {"partial": {"interval_seconds": interval, "tail_seconds": 5.0, "min_seconds": 0.1}},
    }
    monkeypatch.setattr(hermes_config, "load_config", lambda *a, **k: dict(cfg))
    monkeypatch.setattr(tt, "_load_stt_config", lambda: dict(cfg)["stt"])
    monkeypatch.setattr(tt, "_get_provider", lambda _cfg=None: provider)
    monkeypatch.setattr(vp, "_partial_cfg", lambda: dict(cfg)["voice"]["partial"])
    return cfg


def _wait_until(fn, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(0.02)
    return False


def _stub_local_model(monkeypatch, model):
    """Replace the model load + local kwarg/segment helpers so a decode is pure-Python."""
    import tools.transcription_local as tl
    import tools.transcription_tools as tt

    monkeypatch.setattr(tt, "_get_or_load_local_model", lambda name, cfg: model)
    monkeypatch.setattr(tt, "_resolve_stt_language", lambda provider, cfg: "en")
    monkeypatch.setattr(tl, "build_local_transcribe_kwargs", lambda cfg: {})
    monkeypatch.setattr(tl, "_normalize_local_stt_language", lambda lang, supported: "en")
    monkeypatch.setattr(tl, "_join_confident_segments", lambda segs, cfg: " ".join(s.text for s in segs))


# ── live-session contract ───────────────────────────────────────────────────

def test_session_satisfies_the_live_session_contract(monkeypatch):
    """tools.voice_mode drives exactly these members; a rename upstream breaks this test."""
    _force_local_stt(monkeypatch)
    from tools.voice_partial import open_local_partial_session

    session = open_local_partial_session(FakeRecorder(["tail.wav"]), lambda text: None)
    assert session is not None
    for member in ("cancel", "end_audio", "finalize", "preview_only", "provider", "set_input_rate"):
        assert hasattr(session, member), f"live-session contract missing {member}"
    assert session.preview_only is True

    session.set_input_rate(48000)
    result = session.finalize()
    assert result["provider"] == "local"
    # Preview-only: a parked session would let a capped tail masquerade as the take's text.
    assert not result.get("success")
    assert not (result.get("transcript") or "").strip()


def test_preview_only_session_is_never_parked():
    """_park_live_session retires a preview session instead of storing it for _live_result."""
    from tools.voice_mode import _LIVE_SESSIONS, _park_live_session

    class PreviewSession:
        preview_only = True
        provider = "local"

        def __init__(self):
            self.retired = False

        def end_audio(self):
            self.retired = True

        def cancel(self):
            self.retired = True

    session = PreviewSession()
    _park_live_session("/tmp/hermes-preview-take.wav", session)
    assert session.retired is True
    assert "/tmp/hermes-preview-take.wav" not in _LIVE_SESSIONS


def test_streaming_off_does_not_open_a_local_session(monkeypatch):
    """``stt.streaming`` is the switch; off means no re-decode loop, exactly as for live wires."""
    from tools.voice_partial import open_local_partial_session

    _force_local_stt(monkeypatch, streaming=False)
    assert open_local_partial_session(FakeRecorder(["tail.wav"]), lambda text: None) is None


def test_cloud_stt_provider_does_not_load_a_local_model(monkeypatch):
    """A provider with a live wire keeps its own path; arming here would load a whisper model the
    user never selected."""
    from tools.voice_partial import open_local_partial_session

    _force_local_stt(monkeypatch, provider="openai")
    assert open_local_partial_session(FakeRecorder(["tail.wav"]), lambda text: None) is None


def test_recorder_without_tail_support_and_missing_callback_are_skipped(monkeypatch):
    """Termux-style recorders have no frame buffer to snapshot; no callback means nothing to feed."""
    from tools.voice_partial import open_local_partial_session

    _force_local_stt(monkeypatch)

    class NoTailSupport:
        is_recording = True
        elapsed_seconds = 10.0

    assert open_local_partial_session(NoTailSupport(), lambda text: None) is None
    assert open_local_partial_session(FakeRecorder(["tail.wav"]), None) is None


def test_tuning_is_clamped_to_usable_floors(monkeypatch):
    """A re-decode is a full whisper pass: absurd tuning is clamped, not trusted. (The worker
    tests patch _partial_cfg to get sub-second cadences; this pins what a real config gets.)"""
    import hermes_cli.config as hermes_config
    import tools.voice_partial as vp

    monkeypatch.setattr(hermes_config, "load_config", lambda *a, **k: {
        "voice": {"partial": {"interval_seconds": 0.01, "tail_seconds": 0.5, "min_seconds": 0.2}}})
    cfg = vp._partial_cfg()
    assert cfg["interval_seconds"] == vp._FLOORS["interval_seconds"]
    assert cfg["tail_seconds"] == vp._FLOORS["tail_seconds"]
    assert cfg["min_seconds"] == vp._FLOORS["min_seconds"]

    monkeypatch.setattr(hermes_config, "load_config", lambda *a, **k: {})
    assert vp._partial_cfg() == vp._DEFAULTS

    # Non-numeric junk falls back rather than crashing the worker.
    monkeypatch.setattr(hermes_config, "load_config", lambda *a, **k: {
        "voice": {"partial": {"interval_seconds": "often"}}})
    assert vp._partial_cfg()["interval_seconds"] == vp._DEFAULTS["interval_seconds"]


# ── worker behaviour ────────────────────────────────────────────────────────

def test_worker_emits_partials_and_end_audio_retires_it(monkeypatch):
    import tools.voice_partial as vp

    _force_local_stt(monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(vp, "_decode_tail", lambda wav: f"partial {len(seen) + 1}")

    session = vp.open_local_partial_session(FakeRecorder(["a.wav", "b.wav", "c.wav"]), seen.append)
    assert session is not None
    assert vp.get_partial_state()["enabled"] is True

    assert _wait_until(lambda: len(seen) >= 2), "no partial surfaced while the mic was open"
    session.end_audio()
    assert vp.get_partial_state()["enabled"] is False
    assert session._worker is None or not session._worker.is_alive()


def test_tick_landing_mid_decode_is_dropped_not_queued(monkeypatch):
    """A slow decode must never build a queue of partials behind the mic."""
    import tools.voice_partial as vp

    _force_local_stt(monkeypatch, interval=0.05)
    gate = threading.Event()
    decodes: list[str] = []

    def slow_decode(wav_path):
        decodes.append(wav_path)
        gate.wait(0.6)
        return "one partial"

    monkeypatch.setattr(vp, "_decode_tail", slow_decode)
    session = vp.open_local_partial_session(FakeRecorder([f"t{i}.wav" for i in range(50)]), lambda text: None)
    assert session is not None

    time.sleep(0.3)  # several intervals elapse while the first decode is blocked
    assert len(decodes) == 1, "a second decode started while the first was still running"
    gate.set()
    session.end_audio()


def test_empty_or_hallucinated_partial_is_not_emitted(monkeypatch):
    import tools.voice_partial as vp

    _force_local_stt(monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(vp, "_decode_tail", lambda wav_path: None)

    session = vp.open_local_partial_session(FakeRecorder(["a.wav", "b.wav"]), seen.append)
    time.sleep(0.2)
    session.end_audio()
    assert seen == []
    assert vp.get_partial_state()["partial"] == ""


def test_unchanged_partial_does_not_re_render(monkeypatch):
    """Nothing new said → no new callback, so surfaces don't repaint identical text."""
    import tools.voice_partial as vp

    _force_local_stt(monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(vp, "_decode_tail", lambda wav_path: "the same words")

    session = vp.open_local_partial_session(FakeRecorder([f"t{i}.wav" for i in range(20)]), seen.append)
    time.sleep(0.3)
    session.end_audio()
    assert seen == ["the same words"]


def test_decode_failure_records_the_error_and_keeps_the_worker(monkeypatch):
    import tools.voice_partial as vp

    _force_local_stt(monkeypatch)
    attempts: list[int] = []

    def exploding_decode(wav_path):
        attempts.append(1)
        raise RuntimeError("decode exploded")

    monkeypatch.setattr(vp, "_decode_tail", exploding_decode)
    session = vp.open_local_partial_session(FakeRecorder([f"t{i}.wav" for i in range(10)]), lambda text: None)
    assert _wait_until(lambda: len(attempts) >= 2), "one bad tick must not kill the worker"
    assert "decode exploded" in vp.get_partial_state()["last_error"]
    session.end_audio()


def test_worker_stops_when_the_recorder_is_no_longer_capturing(monkeypatch):
    import tools.voice_partial as vp

    _force_local_stt(monkeypatch)
    rec = FakeRecorder([f"t{i}.wav" for i in range(50)], recording=True)
    monkeypatch.setattr(vp, "_decode_tail", lambda wav_path: "partial")
    session = vp.open_local_partial_session(rec, lambda text: None)
    assert session is not None
    assert _wait_until(lambda: rec.tail_calls >= 1)

    rec.is_recording = False
    calls_after = rec.tail_calls
    time.sleep(0.2)
    assert rec.tail_calls == calls_after, "the worker kept re-decoding a stopped capture"
    session.end_audio()


# ── inference lock + unload handshake ───────────────────────────────────────

def test_decode_tail_holds_inference_lock_across_the_lazy_decode(monkeypatch, tmp_path):
    """faster-whisper decodes on first next(): the lock must span transcribe() AND iteration."""
    import tools.transcription_tools as tt
    import tools.voice_partial as vp

    wav = tmp_path / "tail.wav"
    wav.write_bytes(b"\0" * 8)
    lock_held_at_decode: list[bool] = []

    class Model:
        supported_languages = {"en": "English"}

        def transcribe(self, path, **kwargs):
            lock_held_at_decode.append(tt._stt_inference_lock.locked())
            return iter([FakeSegment("hello there")]), MagicMock()

    _stub_local_model(monkeypatch, Model())
    assert vp._decode_tail(str(wav)) == "hello there"
    assert lock_held_at_decode == [True], "the lazy transcribe() ran outside the inference lock"


def test_decode_tail_skips_its_tick_during_an_idle_unload(monkeypatch, tmp_path):
    """The idle watcher is retiring the model — skip the tick instead of reloading it."""
    import tools.transcription_tools as tt
    import tools.voice_partial as vp

    wav = tmp_path / "tail.wav"
    wav.write_bytes(b"\0" * 8)
    loads: list[str] = []

    def counting_load(name, cfg):
        loads.append(name)
        return None

    _stub_local_model(monkeypatch, None)
    monkeypatch.setattr(tt, "_get_or_load_local_model", counting_load)
    monkeypatch.setattr(tt, "_model_unload_in_progress", True)

    assert vp._decode_tail(str(wav)) is None
    assert loads == [], "a mid-flight idle unload was fought by reloading the model"


def test_concurrent_decodes_are_serialized(monkeypatch, tmp_path):
    """A partial and the final pass must never drive one CT2 model at the same time."""
    import tools.voice_partial as vp

    wav = tmp_path / "tail.wav"
    wav.write_bytes(b"\0" * 8)
    inside = 0
    overlap = threading.Event()
    guard = threading.Lock()

    class Model:
        supported_languages = {"en": "English"}

        def transcribe(self, path, **kwargs):
            nonlocal inside
            with guard:
                inside += 1
                if inside > 1:
                    overlap.set()
            time.sleep(0.15)
            with guard:
                inside -= 1
            return iter([FakeSegment("ok")]), MagicMock()

    _stub_local_model(monkeypatch, Model())
    threads = [threading.Thread(target=vp._decode_tail, args=(str(wav),)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert not overlap.is_set(), "two decodes ran concurrently on one model"


# ── tail snapshot (AudioRecorder.tail_wav_path) ─────────────────────────────

def _frames(count: int, block: int, base: int = 0, indexed: bool = False):
    """count blocks of `block` int16 samples; with indexed=True frame i holds base+i, which is how
    the tail tests tell newest frames from oldest. array('h') is int16, exactly what sounddevice
    hands AudioRecorder."""
    return [array.array("h", [base + (i if indexed else 0)] * block) for i in range(count)]


def _make_recorder(frames):
    from tools.voice_mode import AudioRecorder
    rec = AudioRecorder()
    rec._frames = list(frames)
    return rec


def test_tail_wav_returns_newest_audio_capped():
    rec = _make_recorder(_frames(40, 1000, base=1, indexed=True))  # frames hold 1..40, 2.5 s total
    path = rec.tail_wav_path(2.0)
    assert path is not None
    try:
        with wave.open(path, "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 16000
            data = array.array("h")
            data.frombytes(w.readframes(w.getnframes()))
        assert len(data) <= 32000  # capped at 2.0 s @ 16 kHz
        assert len(data) >= 31000  # whole frames kept: no mid-frame chop
        # The tail is the NEWEST audio: newest frame holds 40, oldest surviving holds 9.
        assert data[-1] == 40
        assert data[0] == 9
    finally:
        os.unlink(path)


def test_tail_wav_none_when_too_short():
    rec = _make_recorder(_frames(1, 500))  # 500 samples < the 3200-byte (100 ms) floor
    assert rec.tail_wav_path(2.0) is None


def test_tail_wav_survives_concurrent_frame_append():
    """The snapshot reads the frame list under the recorder lock — it must not race the audio
    callback that appends to it."""
    rec = _make_recorder(_frames(20, 1000, base=1))
    stop = threading.Event()
    errors: list[Exception] = []
    paths: list[str] = []

    def appender():
        while not stop.is_set():
            with rec._lock:
                rec._frames.append(array.array("h", [99] * 1000))
            time.sleep(0.005)

    def snapshotter():
        while not stop.is_set():
            try:
                path = rec.tail_wav_path(1.0)
                if path:
                    paths.append(path)
            except Exception as exc:  # noqa: BLE001 - the point under test: a snapshot must not raise
                errors.append(exc)
            time.sleep(0.005)

    threads = [threading.Thread(target=fn) for fn in (appender, snapshotter)]
    for t in threads:
        t.start()
    time.sleep(0.4)
    stop.set()
    for t in threads:
        t.join(5)

    assert errors == [], f"the snapshot raced the live append: {errors}"
    assert paths, "no snapshot was produced while the mic was appending"
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass
