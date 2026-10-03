"""CPU-baseline gate for the SIMD native wheels (numpy 2.4 / ctranslate2).

NumPy 2.4 raised its x86-64 cpu-baseline to x86-64-v2 (SSE4.1/SSE4.2/POPCNT) and
ctranslate2's wheels dispatch above that too, so on pre-v2 cores the first
``import numpy`` (or ``WhisperModel`` load) raises SIGILL — a signal Python cannot
catch, which takes the whole ``hermes serve`` process down (#109771). These tests
pin every gate that must refuse instead: the probe itself, the provider resolver
(auto-fallback to local_command/cloud), the explicit resolvers, the transcribe
entry point, the whisper loader, the audio import, and the wake-word ladder.
"""

import sys
import types

import pytest

from tools.native_cpu_compat import x86_64_local_voice_native_unsupported_reason

V2_FLAGS = "flags\t\t: fpu mmx sse sse2 sse3 ssse3 sse4_1 sse4_2 popcnt\n"
PRE_V2_FLAGS = "flags\t\t: fpu mmx sse sse2 sse3 ssse3\n"  # AMD E2-2000-class: no SSE4/POPCNT


class TestCpuBaselineProbe:
    def test_pre_v2_cpu_is_reported_unsupported(self):
        reason = x86_64_local_voice_native_unsupported_reason(
            system="Linux", machine="x86_64", cpuinfo_text=PRE_V2_FLAGS)
        assert reason is not None
        assert "x86-64-v2" in reason
        assert "sse4_1" in reason and "sse4_2" in reason and "popcnt" in reason

    def test_v2_cpu_passes(self):
        assert x86_64_local_voice_native_unsupported_reason(
            system="Linux", machine="x86_64", cpuinfo_text=V2_FLAGS) is None

    def test_unreadable_cpuinfo_fails_open(self):
        assert x86_64_local_voice_native_unsupported_reason(
            system="Linux", machine="x86_64", cpuinfo_text="processor\t: 0\n") is None

    def test_non_x86_64_is_never_gated(self):
        assert x86_64_local_voice_native_unsupported_reason(
            system="Linux", machine="aarch64", cpuinfo_text=PRE_V2_FLAGS) is None

    def test_macos_is_never_gated(self):
        assert x86_64_local_voice_native_unsupported_reason(
            system="Darwin", machine="x86_64", cpuinfo_text=PRE_V2_FLAGS) is None


@pytest.fixture
def pre_v2_cpu(monkeypatch):
    """Simulate a pre-x86-64-v2 Linux host for every module that probes it."""
    import tools.native_cpu_compat as ncc

    monkeypatch.setattr(ncc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ncc.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(ncc.Path, "read_text", lambda self, **kw: PRE_V2_FLAGS)
    return ncc


class TestProviderResolution:
    """The resolver must fall back to local_command/cloud instead of 'local'."""

    def test_auto_detect_skips_faster_whisper(self, pre_v2_cpu, monkeypatch):
        import tools.transcription_tools as tt

        monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", True)
        monkeypatch.setattr(tt, "_HAS_OPENAI", True)
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(tt, "_has_local_command", lambda: False)
            assert tt._detect_local_backend() is None
            assert tt._get_provider({}) == "groq"
            mp.setattr(tt, "_has_local_command", lambda: True)
            assert tt._detect_local_backend() == "local_command"
            assert tt._get_provider({}) == "local_command"

    def test_explicit_local_command_does_not_fall_through_to_whisper(self, pre_v2_cpu, monkeypatch):
        import tools.transcription_tools as tt

        monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", True)
        monkeypatch.setattr(tt, "_has_local_command", lambda: False)
        assert tt._resolve_explicit_local_command() == "none"


class TestTranscribeLocalRefuses:
    def test_error_envelope_instead_of_crash(self, pre_v2_cpu):
        import tools.transcription_tools as tt

        result = tt._transcribe_local("/tmp/any.wav", "base")

        assert result["success"] is False
        assert "x86-64-v2" in result["error"]

    def test_no_lazy_install_runs(self, pre_v2_cpu, monkeypatch):
        import tools.transcription_tools as tt

        monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", False)
        monkeypatch.setattr(tt, "_try_lazy_install_stt",
                            lambda: pytest.fail("pip must not run for SIMD wheels on a pre-v2 CPU"))
        result = tt._transcribe_local("/tmp/any.wav", "base")
        assert result["success"] is False


class TestLocalWhisperLoaderGate:
    def test_pre_v2_cpu_refuses_before_the_native_import(self, pre_v2_cpu, monkeypatch):
        import tools.transcription_local as tl

        def _boom(*a, **kw):
            raise AssertionError("faster_whisper must not be imported on a pre-v2 CPU")

        fake_fw = types.ModuleType("faster_whisper")
        setattr(fake_fw, "WhisperModel", _boom)
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

        with pytest.raises(tl.CpuBaselineError, match="x86-64-v2"):
            tl._load_local_whisper_model("base")

    def test_v2_cpu_proceeds_to_the_loader(self, monkeypatch):
        import tools.native_cpu_compat as ncc
        import tools.transcription_local as tl

        monkeypatch.setattr(ncc.platform, "system", lambda: "Linux")
        monkeypatch.setattr(ncc.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(ncc.Path, "read_text", lambda self, **kw: V2_FLAGS)

        class _Model:
            def __init__(self, *a, **kw):
                pass

        fake_fw = types.ModuleType("faster_whisper")
        setattr(fake_fw, "WhisperModel", _Model)
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

        assert tl._load_local_whisper_model("base") is not None


class TestVoiceModeAudioGate:
    def test_audio_check_does_not_import_numpy(self, pre_v2_cpu, monkeypatch):
        original_import = __import__

        def fail_numpy_import():
            raise AssertionError("numpy must not be imported on unsupported legacy x86")

        monkeypatch.setitem(sys.modules, "numpy", None)
        monkeypatch.setattr(
            "builtins.__import__",
            lambda name, *args, **kwargs: fail_numpy_import()
            if name == "numpy"
            else original_import(name, *args, **kwargs),
        )

        from tools.voice_mode import _audio_available

        assert _audio_available() is False

    def test_voice_requirements_reports_cpu_guard(self, pre_v2_cpu, monkeypatch):
        monkeypatch.setattr(
            "tools.voice_mode.detect_audio_environment",
            lambda: {"available": True, "warnings": [], "notices": []},
        )
        monkeypatch.setattr(
            "tools.transcription_tools._get_provider", lambda cfg: "none")

        from tools.voice_mode import check_voice_requirements

        result = check_voice_requirements()

        assert result["available"] is False
        assert "Audio capture: MISSING" in result["details"]
        assert "x86-64-v2" in result["details"]


class TestWakeWordGate:
    def test_wake_word_unavailable_with_cpu_hint(self, pre_v2_cpu, monkeypatch):
        import tools.wake_word as ww

        def fake_supported(feature, **kw):
            return True

        monkeypatch.setattr(ww, "load_wake_word_config", lambda: {})
        monkeypatch.setattr("pm.available", lambda extra: True, raising=False)
        monkeypatch.setattr("pm.install.lazy_installs_allowed", lambda: True, raising=False)

        result = ww.check_wake_word_requirements(supported=fake_supported)

        assert result["available"] is False
        assert "x86-64-v2" in result["hint"]
