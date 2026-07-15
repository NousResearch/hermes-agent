"""Tests for tools.wake_word — the "Hey Hermes" hotword detector.

No live audio or network: the sounddevice import is faked, engines are stubbed,
and lazy-dep availability is monkeypatched. Covers config resolution, engine
dispatch, the requirements probe, the detector fire/cooldown loop, and the
process-wide singleton lifecycle.
"""

import multiprocessing
import threading
import time
import types
from pathlib import Path

import pytest

import tools.wake_word as ww


# ── Config helpers ───────────────────────────────────────────────────────


def test_config_defaults_and_clamping():
    assert ww._provider({}) == "openwakeword"
    assert ww._provider({"provider": "Porcupine"}) == "porcupine"
    assert ww._sensitivity({"sensitivity": 5}) == 1.0
    assert ww._sensitivity({"sensitivity": -1}) == 0.0
    assert ww._sensitivity({"sensitivity": "nope"}) == 0.5
    assert ww.wake_phrase({"phrase": "hey hermes"}) == "hey hermes"
    assert ww.wake_phrase({}) == "hey jarvis"


def test_wake_surface_enabled_gate():
    # Disabled → never, regardless of surface.
    assert ww.wake_surface_enabled("cli", {"enabled": False, "surface": "cli"}) is False
    # auto → every surface is eligible; ownership still admits only one.
    for s in ("cli", "tui", "gui"):
        assert ww.wake_surface_enabled(s, {"enabled": True, "surface": "auto"}) is True
    # Pinned surface → only that one.
    cfg = {"enabled": True, "surface": "tui"}
    assert ww.wake_surface_enabled("tui", cfg) is True
    assert ww.wake_surface_enabled("cli", cfg) is False
    assert ww.wake_surface_enabled("gui", cfg) is False
    # Missing/blank surface defaults to auto.
    assert ww.wake_surface_enabled("gui", {"enabled": True}) is True


def test_looks_like_path():
    assert ww._looks_like_path("models/hey_hermes.onnx")
    assert ww._looks_like_path("custom.ppn")
    assert not ww._looks_like_path("hey_jarvis")


def test_load_wake_word_config_is_a_dict_with_defaults():
    # Wired into DEFAULT_CONFIG, so a real load returns the section shape.
    cfg = ww.load_wake_word_config()
    assert isinstance(cfg, dict)
    assert cfg.get("enabled") is False
    assert cfg.get("provider") == "openwakeword"


def test_load_wake_word_config_guards_non_dict(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"wake_word": "oops"}
    )
    assert ww.load_wake_word_config() == {}


# ── Engine dispatch ──────────────────────────────────────────────────────


def test_build_engine_dispatch(monkeypatch):
    monkeypatch.setattr(ww, "_OpenWakeWordEngine", lambda cfg: "oww")
    monkeypatch.setattr(ww, "_PorcupineEngine", lambda cfg: "pv")
    assert ww._build_engine({"provider": "openwakeword"}) == "oww"
    assert ww._build_engine({"provider": "porcupine"}) == "pv"
    with pytest.raises(ValueError):
        ww._build_engine({"provider": "bogus"})


# ── Requirements probe ───────────────────────────────────────────────────


def test_requirements_openwakeword_available(monkeypatch):
    monkeypatch.setattr(ww, "_audio_available", lambda: True)
    monkeypatch.setattr("tools.lazy_deps.is_available", lambda f: True)
    r = ww.check_wake_word_requirements(
        {"provider": "openwakeword", "phrase": "hey hermes"}
    )
    assert r["available"] is True
    assert r["provider"] == "openwakeword"
    assert r["phrase"] == "hey hermes"


def test_requirements_porcupine_needs_access_key(monkeypatch):
    monkeypatch.delenv("PORCUPINE_ACCESS_KEY", raising=False)
    monkeypatch.setattr(ww, "_audio_available", lambda: True)
    monkeypatch.setattr("tools.lazy_deps.is_available", lambda f: True)
    r = ww.check_wake_word_requirements({"provider": "porcupine"})
    assert r["available"] is False
    assert r["access_key_set"] is False
    assert "PORCUPINE_ACCESS_KEY" in r["hint"]


def test_requirements_unavailable_without_audio(monkeypatch):
    monkeypatch.setattr(ww, "_audio_available", lambda: False)
    monkeypatch.setattr("tools.lazy_deps.is_available", lambda f: True)
    r = ww.check_wake_word_requirements({"provider": "openwakeword"})
    assert r["available"] is False
    assert r["audio_available"] is False


# ── Detector loop ────────────────────────────────────────────────────────


class _FakeStream:
    """Always-readable input stream that yields trivial frames."""

    def __init__(self, **_kw):
        self.closed = False

    def start(self):
        pass

    def read(self, n):
        time.sleep(0.01)
        return [0] * n, False

    def stop(self):
        pass

    def close(self):
        self.closed = True


class _FakeEngine:
    frame_length = 4

    def __init__(self, fire=True):
        self._fire = fire
        self.closed = False
        self.resets = 0

    def process(self, frame):
        return self._fire

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


def _fake_audio(monkeypatch):
    fake_sd = types.SimpleNamespace(InputStream=lambda **kw: _FakeStream(**kw))
    monkeypatch.setattr(ww, "_import_audio", lambda: (fake_sd, None))


def test_detector_fires_once_under_cooldown(monkeypatch):
    _fake_audio(monkeypatch)
    calls = []
    eng = _FakeEngine(fire=True)
    det = ww.WakeWordDetector(eng, lambda: calls.append(1), cooldown=10.0)
    det.start()
    time.sleep(0.25)
    det.stop()
    assert len(calls) == 1  # high cooldown suppresses repeats
    assert eng.closed is True
    assert det.running is False


def test_detector_refires_after_cooldown(monkeypatch):
    _fake_audio(monkeypatch)
    calls = []
    det = ww.WakeWordDetector(_FakeEngine(fire=True), lambda: calls.append(1), cooldown=0.05)
    det.start()
    time.sleep(0.3)
    det.stop()
    assert len(calls) >= 2


def test_detector_no_fire_when_engine_quiet(monkeypatch):
    _fake_audio(monkeypatch)
    calls = []
    det = ww.WakeWordDetector(_FakeEngine(fire=False), lambda: calls.append(1))
    det.start()
    time.sleep(0.15)
    det.stop()
    assert calls == []


def test_detector_resets_engine_on_each_start(monkeypatch):
    # Clearing the engine buffer on (re)start is what stops a resume right after
    # a voice turn from re-firing on stale audio (the runaway wake loop).
    _fake_audio(monkeypatch)
    eng = _FakeEngine(fire=False)
    det = ww.WakeWordDetector(eng, lambda: None)
    det.start()
    time.sleep(0.05)
    det.pause()
    det.resume()
    time.sleep(0.05)
    det.stop()
    assert eng.resets >= 2  # initial start + resume


def test_detector_pause_resume(monkeypatch):
    _fake_audio(monkeypatch)
    det = ww.WakeWordDetector(_FakeEngine(fire=False), lambda: None)
    det.start()
    time.sleep(0.05)
    assert det.running is True
    det.pause()
    assert det.running is False
    det.resume()
    time.sleep(0.05)
    assert det.running is True
    det.stop()
    assert det.running is False


# ── Singleton lifecycle ──────────────────────────────────────────────────


def test_singleton_lifecycle(monkeypatch, tmp_path):
    _fake_audio(monkeypatch)
    monkeypatch.setattr(ww, "_build_engine", lambda cfg: _FakeEngine(fire=False))
    monkeypatch.setattr(ww, "_lock_path", lambda: tmp_path / "wake.lock")
    owner = object()

    assert ww.is_listening() is False
    det = ww.start_listening(lambda: None, owner=owner, config={})
    time.sleep(0.05)
    assert ww.is_listening() is True
    assert ww.owns_listener(owner) is True

    # Re-entrant start returns the same detector and re-arms it.
    det2 = ww.start_listening(lambda: None, owner=owner, config={})
    assert det2 is det

    assert ww.pause_listening(owner=owner) is True
    assert ww.is_listening() is False
    assert ww.resume_listening(owner=owner) is True
    time.sleep(0.05)
    assert ww.is_listening() is True

    assert ww.stop_listening(owner=owner) is True
    assert ww.is_listening() is False


def test_second_owner_cannot_mutate_listener(monkeypatch, tmp_path):
    _fake_audio(monkeypatch)
    monkeypatch.setattr(ww, "_build_engine", lambda cfg: _FakeEngine(fire=False))
    monkeypatch.setattr(ww, "_lock_path", lambda: tmp_path / "wake.lock")
    owner, intruder = object(), object()
    first_callback = lambda: None

    detector = ww.start_listening(first_callback, owner=owner, config={})
    with pytest.raises(ww.WakeWordInUse):
        ww.start_listening(lambda: None, owner=intruder, config={})

    assert detector.on_wake is first_callback
    assert ww.pause_listening(owner=intruder) is False
    assert ww.resume_listening(owner=intruder) is False
    assert ww.stop_listening(owner=intruder) is False
    assert ww.owns_listener(owner) is True
    assert ww.stop_listening(owner=owner) is True


def test_detection_callback_can_pause_and_close_stream(monkeypatch, tmp_path):
    streams = []

    def _stream(**kw):
        stream = _FakeStream(**kw)
        streams.append(stream)
        return stream

    fake_sd = types.SimpleNamespace(InputStream=_stream)
    monkeypatch.setattr(ww, "_import_audio", lambda: (fake_sd, None))
    monkeypatch.setattr(ww, "_build_engine", lambda cfg: _FakeEngine(fire=True))
    monkeypatch.setattr(ww, "_lock_path", lambda: tmp_path / "wake.lock")
    owner = object()
    paused = threading.Event()

    def _on_wake():
        if ww.pause_listening(owner=owner):
            paused.set()

    ww.start_listening(_on_wake, owner=owner, config={})
    assert paused.wait(2)
    assert ww.is_listening() is False
    assert streams[0].closed is True
    assert ww.stop_listening(owner=owner) is True


def test_startup_failure_releases_owner_and_machine_lock(monkeypatch, tmp_path):
    class _BrokenSoundDevice:
        @staticmethod
        def InputStream(**_kw):
            raise OSError("no microphone")

    lock_path = tmp_path / "wake.lock"
    monkeypatch.setattr(ww, "_import_audio", lambda: (_BrokenSoundDevice, None))
    monkeypatch.setattr(ww, "_build_engine", lambda cfg: _FakeEngine(fire=False))
    monkeypatch.setattr(ww, "_lock_path", lambda: lock_path)
    owner = object()

    with pytest.raises(RuntimeError, match="Failed to open"):
        ww.start_listening(lambda: None, owner=owner, config={})

    assert ww.owns_listener(owner) is False
    handle = ww._acquire_machine_lock(lock_path)
    ww._release_machine_lock(handle)


def test_stream_failure_releases_owner_and_machine_lock(monkeypatch, tmp_path):
    class _FailingStream(_FakeStream):
        def read(self, _n):
            raise OSError("device disconnected")

    fake_sd = types.SimpleNamespace(InputStream=lambda **kw: _FailingStream(**kw))
    engine = _FakeEngine(fire=False)
    lock_path = tmp_path / "wake.lock"
    monkeypatch.setattr(ww, "_import_audio", lambda: (fake_sd, None))
    monkeypatch.setattr(ww, "_build_engine", lambda cfg: engine)
    monkeypatch.setattr(ww, "_lock_path", lambda: lock_path)
    owner = object()

    ww.start_listening(lambda: None, owner=owner, config={})
    deadline = time.time() + 2
    while ww.owns_listener(owner) and time.time() < deadline:
        time.sleep(0.01)

    assert ww.owns_listener(owner) is False
    assert engine.closed is True
    handle = ww._acquire_machine_lock(lock_path)
    ww._release_machine_lock(handle)


def _hold_machine_lock(path: str, ready, release) -> None:
    from tools import wake_word

    handle = wake_word._acquire_machine_lock(Path(path))
    ready.set()
    release.wait(10)
    assert handle is not None


def test_machine_lock_is_released_when_owner_process_exits(tmp_path):
    lock_path = tmp_path / "wake.lock"
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    process = ctx.Process(
        target=_hold_machine_lock,
        args=(str(lock_path), ready, release),
    )
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(ww.WakeWordInUse):
            ww._acquire_machine_lock(lock_path)
        release.set()
        process.join(10)
        assert process.exitcode == 0
        handle = ww._acquire_machine_lock(lock_path)
        ww._release_machine_lock(handle)
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
        process.join(10)
