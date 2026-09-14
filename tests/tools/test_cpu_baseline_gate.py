"""CPU-baseline gate for the SIMD native wheels (numpy 2.4 / ctranslate2).

NumPy 2.4 raised its x86-64 cpu-baseline to x86-64-v2 (SSE4.1/SSE4.2/POPCNT) and
ctranslate2's wheels dispatch above that too, so on pre-v2 cores the first
``import numpy`` (or ``WhisperModel`` load) raises SIGILL — a signal Python cannot
catch, which takes the whole ``hermes serve`` process down (#109771). These tests
pin the three gates that must refuse instead: lazy-install (before pip), the local
whisper loader (before the import) and the transcribe entry point.
"""

import sys
import types

import pytest

import tools.lazy_deps as ld
import tools.transcription_local as tl
import tools.transcription_tools as tt


V2_FLAGS = "flags\t\t: fpu mmx sse sse2 sse3 ssse3 sse4_1 sse4_2 popcnt\n"
PRE_V2_FLAGS = "flags\t\t: fpu mmx sse sse2 sse3 ssse3\n"  # AMD E2-2000-class: no SSE4/POPCNT


def _pre_v2_linux(monkeypatch):
    monkeypatch.setattr(ld.sys, "platform", "linux")
    monkeypatch.setattr(ld.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(ld, "_read_cpuinfo_flags", lambda: PRE_V2_FLAGS)


class TestCpuBaselineProbe:
    def test_pre_v2_cpu_is_reported_unsupported(self, monkeypatch):
        _pre_v2_linux(monkeypatch)
        reason = ld._cpu_baseline_missing_reason()
        assert reason is not None
        assert "x86-64-v2" in reason

    def test_v2_cpu_passes(self, monkeypatch):
        monkeypatch.setattr(ld.sys, "platform", "linux")
        monkeypatch.setattr(ld.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(ld, "_read_cpuinfo_flags", lambda: V2_FLAGS)
        assert ld._cpu_baseline_missing_reason() is None

    def test_unreadable_cpuinfo_fails_open(self, monkeypatch):
        _pre_v2_linux(monkeypatch)
        monkeypatch.setattr(ld, "_read_cpuinfo_flags", lambda: "")
        assert ld._cpu_baseline_missing_reason() is None

    def test_non_x86_64_is_never_gated(self, monkeypatch):
        monkeypatch.setattr(ld.sys, "platform", "linux")
        monkeypatch.setattr(ld.platform, "machine", lambda: "aarch64")
        assert ld._cpu_baseline_missing_reason() is None

    def test_macos_is_never_gated(self, monkeypatch):
        monkeypatch.setattr(ld.sys, "platform", "darwin")
        assert ld._cpu_baseline_missing_reason() is None

    def test_simd_features_are_gated_but_not_others(self, monkeypatch):
        _pre_v2_linux(monkeypatch)
        for feature in ("stt.faster_whisper", "wake.openwakeword", "wake.sherpa", "wake.porcupine"):
            assert ld._unsupported_feature_reason(feature) is not None
        assert ld._unsupported_feature_reason("platform.telegram") is None


class TestLazyInstallGate:
    def test_pre_v2_cpu_blocks_whisper_install_before_pip(self, monkeypatch):
        _pre_v2_linux(monkeypatch)
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_lazy_install_target", lambda: None)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip must not run for SIMD wheels on a pre-v2 CPU"),
        )

        with pytest.raises(ld.FeatureUnavailable) as excinfo:
            ld.ensure("stt.faster_whisper", prompt=False)

        assert "x86-64-v2" in str(excinfo.value)

    def test_pre_v2_cpu_blocks_wake_engines_too(self, monkeypatch):
        _pre_v2_linux(monkeypatch)
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_lazy_install_target", lambda: None)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip must not run for SIMD wheels on a pre-v2 CPU"),
        )

        for feature in ("wake.openwakeword", "wake.sherpa", "wake.porcupine"):
            with pytest.raises(ld.FeatureUnavailable):
                ld.ensure(feature, prompt=False)


class TestLocalWhisperLoaderGate:
    def test_pre_v2_cpu_refuses_before_the_native_import(self, monkeypatch):
        _pre_v2_linux(monkeypatch)
        monkeypatch.setattr(tl.platform, "system", lambda: "Linux")

        def _boom():
            raise AssertionError("faster_whisper must not be imported on a pre-v2 CPU")

        fake_fw = types.ModuleType("faster_whisper")
        setattr(fake_fw, "WhisperModel", _boom)
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

        with pytest.raises(tl.CpuBaselineError, match="x86-64-v2"):
            tl._load_local_whisper_model("base")

    def test_v2_cpu_proceeds_to_the_loader(self, monkeypatch):
        monkeypatch.setattr(ld.sys, "platform", "linux")
        monkeypatch.setattr(ld.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(ld, "_read_cpuinfo_flags", lambda: V2_FLAGS)
        monkeypatch.setattr(tl.platform, "system", lambda: "Linux")

        class _Model:
            def __init__(self, *a, **kw):
                pass

        fake_fw = types.ModuleType("faster_whisper")
        setattr(fake_fw, "WhisperModel", _Model)
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

        assert tl._load_local_whisper_model("base") is not None


class TestTranscribeLocalRefuses:
    def test_error_envelope_instead_of_crash(self, monkeypatch):
        _pre_v2_linux(monkeypatch)

        result = tt._transcribe_local("/tmp/any.wav", "base")

        assert result["success"] is False
        assert "x86-64-v2" in result["error"]
