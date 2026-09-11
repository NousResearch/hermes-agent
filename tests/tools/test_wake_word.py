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

import pm
import importlib

pm_ensure = importlib.import_module("pm.ensure")
import tools.wake_word as ww


# ── Config helpers ───────────────────────────────────────────────────────


def test_config_defaults_and_clamping():
    assert ww._provider({}) == ww._provider(ww.load_wake_word_config())
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
    from tools.wake_word_engines import _looks_like_path
    assert _looks_like_path("models/hey_hermes.onnx")
    assert _looks_like_path("custom.ppn")
    assert not _looks_like_path("hey_jarvis")


@pytest.mark.parametrize("system,machine,expected", [
    ("win32", "ARM64", "porcupine"),
    ("win32", "AMD64", "openwakeword"),
    ("darwin", "x86_64", "sherpa"),
    ("darwin", "arm64", "openwakeword"),
    ("linux", "x86_64", "openwakeword"),
    ("linux", "aarch64", "openwakeword"),
])
@pytest.mark.parametrize("saved", [None, "wake_word: {}\n", "wake_word:\n  provider: auto\n"])
def test_loaded_wake_defaults_resolve_supported_provider(tmp_path, monkeypatch, system, machine, expected, saved):
    from functools import partial

    from pm.extras import extra_supported

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    if saved is not None:
        path.write_text(saved, encoding="utf-8")
    cfg = ww.load_wake_word_config()
    assert cfg["provider"] == "auto"
    supported = partial(extra_supported, environment={"sys_platform": system, "platform_machine": machine},
                        importable=lambda _: False)
    assert ww._provider(cfg, supported=supported) == expected
    assert ww._provider({}, supported=supported) == expected
    assert supported(ww._PROVIDERS[expected][1])
    assert not ww.wake_surface_enabled("gui", cfg)
    assert cfg["provider"] == "auto"
    if saved is not None:
        assert path.read_text(encoding="utf-8") == saved


def test_load_wake_word_config_guards_non_dict(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"wake_word": "oops"}
    )
    assert ww.load_wake_word_config() == {}


# ── Engine dispatch ──────────────────────────────────────────────────────


def test_build_engine_dispatch(monkeypatch):
    monkeypatch.setattr(ww, "_OpenWakeWordEngine", lambda cfg: "oww")
    monkeypatch.setattr(ww, "_PorcupineEngine", lambda cfg: "pv")
    monkeypatch.setattr(ww, "_SherpaKwsEngine", lambda cfg: "sherpa")
    expected = {"openwakeword": "oww", "porcupine": "pv", "sherpa": "sherpa"}[ww._provider({})]
    assert ww._build_engine(ww.load_wake_word_config()) == expected
    assert ww._build_engine({"provider": "auto"}) == expected
    assert ww._build_engine({"provider": "openwakeword"}) == "oww"
    assert ww._build_engine({"provider": "porcupine"}) == "pv"
    with pytest.raises(ValueError):
        ww._build_engine({"provider": "bogus"})


def test_engine_classes_are_owned_by_the_extracted_module():
    """wake_word imports the extracted engines and must not shadow them with
    copies — two class families carrying thresholds/model lookup/cleanup is
    exactly the bug class where fixes to the apparent owner do nothing."""
    from tools import wake_word_engines as engines

    assert ww._Engine is engines._Engine
    assert ww._OpenWakeWordEngine is engines._OpenWakeWordEngine
    assert ww._SherpaKwsEngine is engines._SherpaKwsEngine
    assert ww._PorcupineEngine is engines._PorcupineEngine


def test_engine_construction_ensures_audio_io_only_for_local_capture(monkeypatch, tmp_path):
    """Constructor-to-capture admission: an engine constructor ensures its own
    wake-* extra always, but audio-io (sounddevice+numpy) ONLY when the resolved
    capture mode is local. Client capture (desktop streams PCM via wake.feed)
    must never trigger installation of local audio libraries.

    Regression (Q033): the active engine constructors ensured only wake-*, so a
    freshly installed per-engine extra without sounddevice/numpy failed at
    capture; the extracted _ensure_dep fixed that but was dead code because
    wake_word shadowed the extracted classes.
    """
    from tools import wake_word_engines as engines

    ensured: list[str] = []

    def _fake_ensure_import(feature, *a, **k):
        ensured.append(feature)

    monkeypatch.setattr(pm, "ensure_import", _fake_ensure_import)
    monkeypatch.setattr(pm, "available", lambda feature: feature in ensured)

    class _FakeModel:
        id = "hey_hermes"

        @staticmethod
        def from_model(model_path, libtensorflowlite_c_path=None):
            return _FakeModel()

        def process_streaming(self, embeddings):
            return iter(())

        def reset(self):
            pass

        def close(self):
            pass

    class _FakeFeatures:
        @staticmethod
        def from_builtin(models_dir=None, libtensorflowlite_c_path=None):
            return _FakeFeatures()

        def process_streaming(self, audio_chunk):
            return iter(())

        def reset(self):
            pass

        def close(self):
            pass

    mod = types.ModuleType("pyopen_wakeword")
    mod.OpenWakeWord = _FakeModel
    mod.OpenWakeWordFeatures = _FakeFeatures
    monkeypatch.setitem(sys.modules, "pyopen_wakeword", mod)

    cfg = {"provider": "openwakeword"}

    # Client capture: engine extra only, never audio-io.
    engines._OpenWakeWordEngine({**cfg, "capture": "client"})
    assert "wake-openwakeword" in ensured
    assert "audio-io" not in ensured

    # Local capture: engine extra plus the capture deps.
    ensured.clear()
    engines._OpenWakeWordEngine({**cfg, "capture": "local"})
    assert "wake-openwakeword" in ensured
    assert "audio-io" in ensured

    # The caller has already selected client capture even when config says auto.
    ensured.clear()
    monkeypatch.setattr(ww, "_lock_path", lambda: tmp_path / "wake.lock")
    owner = object()
    try:
        ww.start_listening(lambda: None, owner=owner, config=cfg, external_audio=True)
        assert "wake-openwakeword" in ensured
        assert "audio-io" not in ensured
    finally:
        ww.stop_listening(owner=owner)


# ── Requirements probe ───────────────────────────────────────────────────


def _voice_loop_ready(monkeypatch, stt=True, tts=True):
    """Pin the STT/TTS probes so requirements tests don't depend on the
    test venv's installed voice stack."""
    monkeypatch.setattr(ww, "_stt_ready", lambda: stt)
    monkeypatch.setattr(ww, "_tts_ready", lambda: tts)
    monkeypatch.setattr("pm.extras._PLATFORM_GATES", {})


@pytest.mark.parametrize("system,machine", [("win32", "ARM64"), ("darwin", "x86_64")])
@pytest.mark.parametrize("provider", ["auto", *ww._PROVIDERS])
def test_loaded_provider_requirements_preserve_choices_and_require_keys(tmp_path, monkeypatch, system, machine, provider):
    from functools import partial

    from pm.extras import extra_supported

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("PORCUPINE_ACCESS_KEY", raising=False)
    saved = f"wake_word:\n  provider: {provider}\n  capture: client\n"
    path = tmp_path / "config.yaml"
    path.write_text(saved, encoding="utf-8")
    cfg = ww.load_wake_word_config()
    supported = partial(extra_supported, environment={"sys_platform": system, "platform_machine": machine},
                        importable=lambda _: False)
    monkeypatch.setattr(pm, "available", lambda _: False)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(ww, "_stt_ready", lambda: True)
    monkeypatch.setattr(ww, "_tts_ready", lambda: True)
    result = ww.check_wake_word_requirements(cfg, supported=supported)
    selected = ww._provider(cfg, supported=supported)
    assert result["provider"] == selected
    if provider != "auto":
        assert selected == provider
    if not supported(ww._PROVIDERS[selected][1]):
        assert not result["available"]
        assert "not supported on this platform" in result["hint"]
    elif selected == "porcupine":
        assert not result["available"]
        assert not result["access_key_set"]
        assert "PORCUPINE_ACCESS_KEY" in result["hint"]
        monkeypatch.setenv("PORCUPINE_ACCESS_KEY", "test-key")
        assert ww.check_wake_word_requirements(cfg, supported=supported)["available"]
    else:
        assert result["available"]
    assert not ww.wake_surface_enabled("gui", cfg)
    assert cfg["provider"] == provider
    assert path.read_text(encoding="utf-8") == saved


def test_requirements_openwakeword_available(monkeypatch):
    _voice_loop_ready(monkeypatch)
    monkeypatch.setattr(ww, "_audio_available", lambda: True)
    monkeypatch.setattr(pm, "available", lambda f: True)
    r = ww.check_wake_word_requirements(
        {"provider": "openwakeword", "phrase": "hey hermes"}
    )
    assert r["available"] is True
    assert r["provider"] == "openwakeword"
    assert r["phrase"] == "hey hermes"


def test_tts_ready_is_a_probe_never_an_installer(monkeypatch):
    """_tts_ready must NOT trigger lazy pip installs from a status poll.

    Regression: check_tts_requirements → _import_edge_tts → pm.ensure_import
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
    monkeypatch.setattr(pm, "available", lambda f: False)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: True)
    assert ww._tts_ready() is True

    # Deps missing + lazy installs disabled → not ready.
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: False)
    assert ww._tts_ready() is False

    # Deps present → falls through to the real requirements check.
    fake_tts.check_tts_requirements = lambda: True
    monkeypatch.setattr(pm, "available", lambda f: True)
    assert ww._tts_ready() is True


def test_requirements_fresh_install_lazy_allowed(monkeypatch):
    """Deps missing + lazy installs allowed → available, so /wake on can
    reach the engine constructor's ``pm.ensure_import()`` call.

    Regression: the audio probe imports sounddevice/numpy — packages the
    lazy installer would fetch — so gating ``available`` on it made the
    lazy-install path unreachable on a fresh machine (the /wake on handler
    printed the pip hint and bailed before ensure() ever ran).
    """
    def _boom():
        raise AssertionError("audio probe must not run while deps are missing")

    _voice_loop_ready(monkeypatch)
    monkeypatch.setattr(ww, "_audio_available", _boom)
    monkeypatch.setattr(pm, "available", lambda f: False)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: True)
    r = ww.check_wake_word_requirements({"provider": "openwakeword"})
    assert r["available"] is True
    assert r["deps_available"] is False
    assert r["hint"] == ""


@pytest.mark.parametrize("capture", ["local", "client"])
@pytest.mark.parametrize("provider", ["openwakeword", "oww", "local"])
def test_requirements_reject_unsupported_engine_without_attempting_install(monkeypatch, capture, provider):
    import pm.extras as extras

    _voice_loop_ready(monkeypatch)
    monkeypatch.setattr(extras, "_PLATFORM_GATES", {"wake-openwakeword": "python_version < '0'"})
    monkeypatch.setattr(extras, "_importable", lambda anchor: False)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setenv("PORCUPINE_ACCESS_KEY", "test-key")

    def no_install(*args, **kwargs):
        pytest.fail("a requirements probe must not install dependencies")

    monkeypatch.setattr(pm_ensure, "sync_venv", no_install)
    result = ww.check_wake_word_requirements({"provider": provider, "capture": capture})
    assert result["available"] is False
    assert "not supported" in result["hint"]
    assert "wake_word.provider" in result["hint"]
    assert "sherpa" in result["hint"] and "porcupine" in result["hint"]
    assert "uv sync" not in result["hint"]
    for alternative in ("sherpa", "porcupine"):
        assert ww.check_wake_word_requirements({"provider": alternative, "capture": capture})["available"] is True


def test_requirements_lazy_disabled_returns_remedy_not_nameerror(monkeypatch):
    """Deps missing + lazy installs disabled → unavailable WITH a remedy hint.

    Regression (C29): a competing legacy hint ladder also ran on this path and
    referenced the deleted lazy-deps module by bare name — a NameError crashed
    the status probe instead of returning ``hint``.
    """
    _voice_loop_ready(monkeypatch)
    monkeypatch.setattr(ww, "_audio_available", lambda: True)
    monkeypatch.setattr(pm, "available", lambda f: False)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: False)
    r = ww.check_wake_word_requirements({"provider": "openwakeword"})
    assert r["available"] is False
    assert r["deps_available"] is False
    assert "uv sync --frozen --extra wake-openwakeword" in r["hint"]


def test_requirements_deps_present_but_no_audio_hint(monkeypatch):
    """Once deps ARE installed, a failing audio probe blocks with a mic hint
    (lazy installs can't fix a missing audio device)."""
    _voice_loop_ready(monkeypatch)
    monkeypatch.setattr(ww, "_audio_available", lambda: False)
    monkeypatch.setattr(ww, "_local_input_device_ready", lambda: False)
    monkeypatch.setattr(pm, "available", lambda f: True)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: True)
    r = ww.check_wake_word_requirements({"provider": "openwakeword", "capture": "local"})
    assert r["available"] is False
    assert "audio device" in r["hint"] or "microphone" in r["hint"].lower()


# ── openWakeWord engine (pyopen-wakeword; bundled model, no runtime fetch) ──


def test_openwakeword_custom_model_path_used(monkeypatch):
    # A custom ``model`` path passes through to pyopen-wakeword as-is. The
    # shared feature models come from the wheel (from_builtin) — there is no
    # download_models step to regress.
    captured = {}

    class _FakeModel:
        def __init__(self, model_path, libtensorflowlite_c_path=None):
            self.id = os.path.splitext(os.path.basename(str(model_path)))[0]
            captured["path"] = str(model_path)

        @staticmethod
        def from_model(model_path, libtensorflowlite_c_path=None):
            return _FakeModel(model_path, libtensorflowlite_c_path)

        def process_streaming(self, embeddings):
            return iter(())

        def reset(self):
            pass

        def close(self):
            pass

    class _FakeFeatures:
        @staticmethod
        def from_builtin(models_dir=None, libtensorflowlite_c_path=None):
            return _FakeFeatures()

        def process_streaming(self, audio_chunk):
            return iter(())

        def reset(self):
            pass

        def close(self):
            pass

    mod = types.ModuleType("pyopen_wakeword")
    mod.OpenWakeWord = _FakeModel
    mod.OpenWakeWordFeatures = _FakeFeatures
    monkeypatch.setitem(sys.modules, "pyopen_wakeword", mod)
    monkeypatch.setattr(pm, "ensure_import", lambda *a, **k: None)
    eng = ww._OpenWakeWordEngine(
        {"provider": "openwakeword", "openwakeword": {"model": "/models/hey_hermes.tflite"}}
    )
    assert captured["path"] == "/models/hey_hermes.tflite"
    assert eng._labels == ["hey_hermes"]


def test_bundled_hey_hermes_model_ships_on_disk():
    # The "hey hermes" wake word works out of the box only if the model is
    # actually bundled. pyopen-wakeword runs TFLite only.
    path = ww._bundled_wakeword_path()
    assert os.path.exists(path), path
    assert os.path.getsize(path) > 1024, path


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
    monkeypatch.setattr(pm, "ensure_import", lambda *a, **k: None)

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
    reads = []
    processed = []

    class _NativeRateStream(_LoudStream):
        def read(self, n):
            reads.append(n)
            return super().read(n)

    class _RecordingEngine(_FakeEngine):
        def process(self, frame):
            processed.append(frame)
            return False

    def _stream(**kwargs):
        opened.append(kwargs)
        return _NativeRateStream(**kwargs)

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
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(ww, "_import_audio", lambda: (fake_sd, np))

    det = ww.WakeWordDetector(
        _RecordingEngine(fire=False),
        lambda: None,
        input_device="Microphone Array",
    )
    det.start()
    try:
        assert opened[0]["device"] == "Microphone Array"
        assert opened[0]["samplerate"] == 48000
        assert opened[0]["blocksize"] == 12
        deadline = time.monotonic() + 2.0
        while not processed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert reads[0] == 12
        assert len(processed[0]) == 4
        assert processed[0].tolist() == [500] * 4
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


@pytest.mark.platforms("windows")
def test_windows_silent_hint_names_selected_device():
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


@pytest.mark.platforms("macos")
def test_macos_silent_hint_points_at_privacy_settings():
    """On macOS a silent stream is almost always the TCC mic permission, so the
    hint names System Settings rather than the device."""
    hint = ww.silent_audio_hint(
        {"selector": 1, "name": "MacBook Pro Microphone", "hostapi": "Core Audio"}
    )
    assert "Privacy & Security" in hint
    assert "Microphone" in hint


@pytest.mark.platforms("linux")
def test_linux_silent_hint_names_selected_device():
    hint = ww.silent_audio_hint(
        {"selector": 2, "name": "HD Audio Capture", "hostapi": "ALSA"}
    )
    assert "HD Audio Capture (ALSA)" in hint
    assert "Privacy & Security" not in hint


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


# ── Client capture (remote desktop mic → wake.feed) ──────────────────────


def test_resolve_capture_mode_auto_and_prefer_client(monkeypatch):
    monkeypatch.setattr(ww, "_local_input_device_ready", lambda: False)
    # auto without prefer_client stays local (CLI/TUI/status semantics)
    assert ww.resolve_capture_mode({"capture": "auto"}) == "local"
    assert ww.resolve_capture_mode({"capture": "auto"}, prefer_client=True) == "client"
    assert ww.resolve_capture_mode({"capture": "local"}, prefer_client=True) == "local"
    assert ww.resolve_capture_mode({"capture": "client"}) == "client"
    assert ww.resolve_capture_mode({"capture": "auto"}, force_local=True) == "local"
    monkeypatch.setattr(ww, "_local_input_device_ready", lambda: True)
    assert ww.resolve_capture_mode({"capture": "auto"}) == "local"
    # A working backend mic wins under auto even for a preferring surface, so
    # local desktops keep PortAudio + wake_word.input_device selection.
    assert ww.resolve_capture_mode({"capture": "auto"}, prefer_client=True) == "local"
    # Explicit client still forces streaming (backend mic exists but is wrong).
    assert ww.resolve_capture_mode({"capture": "client"}, prefer_client=True) == "client"


def test_requirements_client_capture_without_local_mic(monkeypatch):
    monkeypatch.setattr(ww, "_audio_available", lambda: False)
    monkeypatch.setattr(ww, "_local_input_device_ready", lambda: False)
    monkeypatch.setattr(ww, "_stt_ready", lambda: True)
    monkeypatch.setattr(ww, "_tts_ready", lambda: True)
    monkeypatch.setattr(pm, "available", lambda f: True)
    monkeypatch.setattr(pm_ensure, "lazy_installs_allowed", lambda: False)

    reqs = ww.check_wake_word_requirements({"capture": "client", "provider": "openwakeword"})
    assert reqs["available"] is True
    assert reqs["capture"] == "client"


def test_client_capture_feed_fires(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")

    monkeypatch.setattr(ww, "_build_engine", lambda cfg: _FakeEngine(fire=True))
    monkeypatch.setattr(ww, "_lock_path", lambda: tmp_path / "wake.lock")
    # External mode must not import sounddevice
    monkeypatch.setattr(
        ww,
        "_import_audio",
        lambda: (_ for _ in ()).throw(OSError("no local mic")),
    )
    owner = object()
    fired = threading.Event()

    def _on_wake():
        fired.set()

    ww.start_listening(_on_wake, owner=owner, config={}, external_audio=True)
    assert ww.is_listening() is True
    info = ww.detector_frame_info()
    fl = int(info["frame_length"])
    # Non-silent frame so silence flag does not dominate
    pcm = (np.ones(fl, dtype=np.int16) * 5000).tobytes()
    assert ww.feed_audio(owner=owner, pcm_int16=pcm) is True
    assert fired.wait(2.0)
    assert ww.stop_listening(owner=owner) is True


def test_feed_audio_rejects_wrong_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(ww, "_build_engine", lambda cfg: _FakeEngine(fire=False))
    monkeypatch.setattr(ww, "_lock_path", lambda: tmp_path / "wake.lock")
    owner = object()
    ww.start_listening(lambda: None, owner=owner, config={}, external_audio=True)
    assert ww.feed_audio(owner=object(), pcm_int16=b"\x00\x00") is False
    assert ww.stop_listening(owner=owner) is True
