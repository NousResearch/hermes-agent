"""Tests for tools.wake_word — the "Hey Hermes" hotword detector.

No live audio or network: the sounddevice import is faked, engines are stubbed,
and lazy-dep availability is monkeypatched. Covers config resolution, engine
dispatch, the requirements probe, the detector fire/cooldown loop, and the
process-wide singleton lifecycle.
"""

import multiprocessing
import os
import sys
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
    assert ww._input_device({}) is None
    assert ww._input_device({"input_device": 7}) == 7
    assert ww._input_device({"input_device": " Microphone Array "}) == "Microphone Array"
    assert ww._input_device({"input_device": ""}) is None
    assert ww._input_device({"input_device": False}) is None
    assert ww._sensitivity({"sensitivity": 5}) == 1.0
    assert ww._sensitivity({"sensitivity": -1}) == 0.0
    # Invalid input falls back to the configured default, not a hardcoded 0.5.
    assert ww._sensitivity({"sensitivity": "nope"}) == ww._DEFAULTS["sensitivity"]
    assert ww._sensitivity({}) == ww._DEFAULTS["sensitivity"]
    assert ww.wake_phrase({"phrase": "hey hermes"}) == "hey hermes"
    assert ww.wake_phrase({}) == "hey hermes"


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


def _voice_loop_ready(monkeypatch, stt=True, tts=True):
    """Pin the STT/TTS probes so requirements tests don't depend on the
    test venv's installed voice stack."""
    monkeypatch.setattr(ww, "_stt_ready", lambda: stt)
    monkeypatch.setattr(ww, "_tts_ready", lambda: tts)


def test_requirements_openwakeword_available(monkeypatch):
    _voice_loop_ready(monkeypatch)
    monkeypatch.setattr(ww, "_audio_available", lambda: True)
    monkeypatch.setattr("tools.lazy_deps.is_available", lambda f: True)
    r = ww.check_wake_word_requirements(
        {"provider": "openwakeword", "phrase": "hey hermes"}
    )
    assert r["available"] is True
    assert r["provider"] == "openwakeword"
    assert r["phrase"] == "hey hermes"


def test_tts_ready_is_a_probe_never_an_installer(monkeypatch):
    """_tts_ready must NOT trigger lazy pip installs from a status poll.

    Regression: check_tts_requirements → _import_edge_tts → lazy_deps.ensure
    ran pip inside wake.status; a slow/failed install froze the poll and
    unmounted the desktop ear. Uninstalled-but-lazy-installable counts as
    ready WITHOUT calling ensure/check.
    """
    import types as _types

    monkeypatch.setattr(
        ww, "_tts_ready", ww.__dict__["_tts_ready"]
    )  # use the real implementation
    fake_tts = _types.SimpleNamespace(
        _get_provider=lambda cfg: "edge",
        _load_tts_config=lambda: {},
        check_tts_requirements=lambda: (_ for _ in ()).throw(
            AssertionError("check_tts_requirements must not run when deps are missing")
        ),
    )
    monkeypatch.setitem(sys.modules, "tools.tts_tool", fake_tts)

    # Deps missing + lazy installs allowed → ready (installs at first speak).
    monkeypatch.setattr("tools.lazy_deps.is_available", lambda f: False)
    monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
    assert ww._tts_ready() is True

    # Deps missing + lazy installs disabled → not ready.
    monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: False)
    assert ww._tts_ready() is False

    # Deps present → falls through to the real requirements check.
    fake_tts.check_tts_requirements = lambda: True
    monkeypatch.setattr("tools.lazy_deps.is_available", lambda f: True)
    assert ww._tts_ready() is True


def test_requirements_fresh_install_lazy_allowed(monkeypatch):
    """Deps missing + lazy installs allowed → available, so /wake on can
    reach the engine constructor's ``lazy_deps.ensure()`` call.

    Regression: the audio probe imports sounddevice/numpy — packages the
    lazy installer would fetch — so gating ``available`` on it made the
    lazy-install path unreachable on a fresh machine (the /wake on handler
    printed the pip hint and bailed before ensure() ever ran).
    """
    def _boom():
        raise AssertionError("audio probe must not run while deps are missing")

    _voice_loop_ready(monkeypatch)
    monkeypatch.setattr(ww, "_audio_available", _boom)
    monkeypatch.setattr("tools.lazy_deps.is_available", lambda f: False)
    monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
    r = ww.check_wake_word_requirements({"provider": "openwakeword"})
    assert r["available"] is True
    assert r["deps_available"] is False
    assert r["hint"] == ""


def test_requirements_deps_present_but_no_audio_hint(monkeypatch):
    """Once deps ARE installed, a failing audio probe blocks with a mic hint
    (lazy installs can't fix a missing audio device)."""
    monkeypatch.setattr(ww, "_audio_available", lambda: False)
    monkeypatch.setattr("tools.lazy_deps.is_available", lambda f: True)
    monkeypatch.setattr("tools.lazy_deps._allow_lazy_installs", lambda: True)
    r = ww.check_wake_word_requirements({"provider": "openwakeword"})
    assert r["available"] is False
    assert "audio device" in r["hint"]


# ── openWakeWord engine (bundled model + base-model fetch) ───────────────


def _install_fake_openwakeword(monkeypatch):
    """Swap in a fake ``openwakeword`` so the engine builds with no network.

    Returns a ``calls`` dict recording every ``download_models`` invocation.
    """
    calls = {"download": []}

    class _FakeModel:
        def __init__(self, wakeword_models, inference_framework="onnx"):
            self.wakeword_models = list(wakeword_models)
            self.models = {"hey_hermes": object()}

        def predict(self, frame):
            return {"hey_hermes": 0.0}

        def reset(self):
            pass

    oww = types.ModuleType("openwakeword")
    oww.utils = types.SimpleNamespace(
        download_models=lambda names=[]: calls["download"].append(list(names))
    )
    model_mod = types.ModuleType("openwakeword.model")
    model_mod.Model = _FakeModel

    monkeypatch.setitem(sys.modules, "openwakeword", oww)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_mod)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **k: None)
    return calls


def test_openwakeword_ensures_base_models_for_custom_path(monkeypatch):
    # Regression: a custom ``.onnx`` path used to skip download_models entirely,
    # so a fresh install crashed at load time on a missing melspectrogram.onnx.
    # The base feature models must be ensured for a custom path too.
    calls = _install_fake_openwakeword(monkeypatch)
    eng = ww._OpenWakeWordEngine(
        {"provider": "openwakeword", "openwakeword": {"model": "/models/hey_hermes.onnx"}}
    )
    assert calls["download"] == [["/models/hey_hermes.onnx"]]
    assert eng._labels == ["hey_hermes"]


def test_bundled_hey_hermes_model_ships_on_disk():
    # The "hey hermes" wake word works out of the box only if the model is
    # actually bundled. Both framework artifacts must exist and be non-trivial.
    for framework in ("onnx", "tflite"):
        path = ww._bundled_wakeword_path(framework)
        assert os.path.exists(path), path
        assert os.path.getsize(path) > 1024, path


# ── platform-aware backend selection (openWakeWord onnx is broken on macOS ARM64,
#    upstream dscripka/openWakeWord#336) ────────────────────────────────────────

def test_default_framework_is_tflite_on_macos_arm64(monkeypatch):
    monkeypatch.setattr(ww.sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    assert ww.default_inference_framework() == "tflite"


def test_explicit_framework_kept_off_broken_platform(monkeypatch):
    # An operator who pins a backend keeps it everywhere ONNX actually works.
    calls = _install_fake_openwakeword(monkeypatch)
    monkeypatch.setattr(ww.sys, "platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    ww._OpenWakeWordEngine(
        {"provider": "openwakeword", "openwakeword": {"inference_framework": "onnx"}}
    )
    (downloaded,) = calls["download"]
    assert downloaded == [ww._bundled_wakeword_path("onnx")]


def test_empty_framework_falls_back_to_platform_default(monkeypatch):
    monkeypatch.setattr(ww.sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    assert ww.resolve_inference_framework({}) == "tflite"
    assert ww.resolve_inference_framework({"openwakeword": {"inference_framework": ""}}) == "tflite"
    monkeypatch.setattr(ww.sys, "platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert ww.resolve_inference_framework({}) == "onnx"


# ── ambient-speech rejection: consecutive-frame confirmation ──────────────────

def _openwakeword_engine_with_scores(monkeypatch, cfg_wake, scores):
    """Build a real _OpenWakeWordEngine whose predict() replays ``scores``."""
    seq = iter(scores)

    class _ScriptedModel:
        def __init__(self, wakeword_models, inference_framework="onnx"):
            self.models = {"hey_hermes": object()}

        def predict(self, frame):
            return {"hey_hermes": next(seq)}

        def reset(self):
            pass

    oww = types.ModuleType("openwakeword")
    oww.utils = types.SimpleNamespace(download_models=lambda names=[]: None)
    model_mod = types.ModuleType("openwakeword.model")
    model_mod.Model = _ScriptedModel
    monkeypatch.setitem(sys.modules, "openwakeword", oww)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_mod)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **k: None)
    monkeypatch.setattr(ww, "ensure_tflite_runtime", lambda: True)
    return ww._OpenWakeWordEngine({"provider": "openwakeword", **cfg_wake})


# ── sherpa-onnx open-vocabulary engine ───────────────────────────────────


def _install_fake_sherpa(monkeypatch, tmp_path):
    """Fake sherpa_onnx + a fake model dir so the engine builds offline."""
    calls = {"text2token": [], "spotter": [], "results": []}

    model_dir = tmp_path / "kws-model"
    model_dir.mkdir()
    for name in (
        "tokens.txt",
        "bpe.model",
        "encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        "joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
    ):
        (model_dir / name).write_bytes(b"x")

    class _FakeStream:
        def accept_waveform(self, sample_rate, samples):
            pass

    class _FakeSpotter:
        def __init__(self, **kwargs):
            calls["spotter"].append(kwargs)

        def create_stream(self):
            return _FakeStream()

        def is_ready(self, stream):
            return bool(calls["results"])

        def decode_stream(self, stream):
            pass

        def get_result(self, stream):
            return calls["results"].pop(0) if calls["results"] else ""

        def reset_stream(self, stream):
            pass

    def _fake_text2token(phrases, tokens, tokens_type, bpe_model):
        calls["text2token"].append(list(phrases))
        return [p.split() for p in phrases]

    sherpa = types.ModuleType("sherpa_onnx")
    sherpa.KeywordSpotter = _FakeSpotter
    sherpa.text2token = _fake_text2token
    monkeypatch.setitem(sys.modules, "sherpa_onnx", sherpa)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **k: None)

    # numpy is an optional voice-extra dep, lazy-installed at runtime — CI's
    # hermetic slices don't have it. process() only calls asarray(...)/32768,
    # so a minimal stub keeps these tests runnable without the real package.
    if "numpy" not in sys.modules:
        class _FakeArr(list):
            def __truediv__(self, other):
                return self

        np_stub = types.ModuleType("numpy")
        np_stub.float32 = "float32"
        np_stub.asarray = lambda x, dtype=None: _FakeArr(x)
        monkeypatch.setitem(sys.modules, "numpy", np_stub)
    return calls, model_dir


# ── Multi-profile phrase routing ─────────────────────────────────────────


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


class _Frame(list):
    """List with numpy-ish abs()/max() so the silence probe sees real peaks."""

    def __abs__(self):
        return _Frame(abs(x) for x in self)

    def max(self):
        return max(self) if self else 0


class _SilentStream(_FakeStream):
    """Stream that always yields near-zero frames (dead macOS mic)."""

    def read(self, n):
        time.sleep(0.005)
        return _Frame([0] * n), False


class _LoudStream(_FakeStream):
    """Stream that yields audible frames."""

    def read(self, n):
        time.sleep(0.005)
        return _Frame([500] * n), False


def test_detector_opens_configured_input_device_and_reports_backend(monkeypatch):
    opened = []

    def _stream(**kwargs):
        opened.append(kwargs)
        return _LoudStream(**kwargs)

    fake_sd = types.SimpleNamespace(
        InputStream=_stream,
        query_devices=lambda selector, kind: {
            "name": "Microphone Array",
            "hostapi": 2,
            "max_input_channels": 2,
            "default_samplerate": 48000.0,
        },
        query_hostapis=lambda index: {"name": "Windows WASAPI"},
    )
    monkeypatch.setattr(ww, "_import_audio", lambda: (fake_sd, None))

    det = ww.WakeWordDetector(
        _FakeEngine(fire=False),
        lambda: None,
        input_device="Microphone Array",
    )
    det.start()
    try:
        assert opened[0]["device"] == "Microphone Array"
        assert det.input_device_details == {
            "selector": "Microphone Array",
            "name": "Microphone Array",
            "hostapi_index": 2,
            "hostapi": "Windows WASAPI",
            "max_input_channels": 2,
            "default_samplerate": 48000.0,
        }
    finally:
        det.stop()


def test_windows_silent_hint_names_selected_device(monkeypatch):
    monkeypatch.setattr(ww.sys, "platform", "win32")
    hint = ww.silent_audio_hint(
        {
            "selector": 3,
            "name": "Microphone Array",
            "hostapi": "Windows WASAPI",
        }
    )
    assert "Microphone Array (Windows WASAPI)" in hint
    assert "wake_word.input_device" in hint
    assert "macOS" not in hint


def test_detector_flags_silent_stream_and_recovers(monkeypatch):
    """A stream of zeros sets audio_silent; audible input clears it."""
    monkeypatch.setattr(ww, "_SILENCE_ALERT_SECONDS", 0.001)  # trip on the first frame
    stream_cls = {"cls": _SilentStream}
    fake_sd = types.SimpleNamespace(InputStream=lambda **kw: stream_cls["cls"](**kw))
    monkeypatch.setattr(ww, "_import_audio", lambda: (fake_sd, None))

    det = ww.WakeWordDetector(_FakeEngine(fire=False), lambda: None)
    det.start()
    try:
        deadline = time.monotonic() + 2.0
        while not det.audio_silent and time.monotonic() < deadline:
            time.sleep(0.01)
        assert det.audio_silent is True
        assert ww.audio_is_silent() is False  # module accessor needs the singleton

        monkeypatch.setattr(ww, "_detector", det)
        assert ww.audio_is_silent() is True

        # Audio returns (permission granted / real mic) — flag clears.
        det.pause()
        stream_cls["cls"] = _LoudStream
        det.resume()
        deadline = time.monotonic() + 2.0
        while det.audio_silent and time.monotonic() < deadline:
            time.sleep(0.01)
        assert det.audio_silent is False
    finally:
        monkeypatch.setattr(ww, "_detector", None)
        det.stop()


# ── Singleton lifecycle ──────────────────────────────────────────────────


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
