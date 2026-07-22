"""
Tests for the native SuperTonic 3 TTS provider.

These tests pin the resolution / caching / dispatch paths for SuperTonic
without requiring the ``supertonic`` package to actually be installed
(the synthesis step is monkey-patched to avoid needing the ONNX model).
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HERMES_SESSION_PLATFORM",):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def clear_supertonic_cache():
    """Reset the module-level model cache between tests."""
    from tools import tts_tool as _tt
    _tt._supertonic_model_cache.clear()
    yield
    _tt._supertonic_model_cache.clear()


@pytest.fixture
def mock_supertonic_module():
    """Inject a fake supertonic + numpy + soundfile module that return stub objects."""
    fake_tts_instance = MagicMock()
    # synthesize() returns (wav, dur) where wav is (1, N) and dur is ndarray
    fake_tts_instance.synthesize.return_value = (
        [[0.0] * 44100],  # 1s of silence at 44100Hz, shape (1, N)
        MagicMock(__getitem__=lambda self, idx: 1.0),  # dur[0] = 1.0
    )
    fake_tts_instance.get_voice_style.return_value = MagicMock()
    fake_tts_cls = MagicMock(return_value=fake_tts_instance)
    fake_supertonic = MagicMock()
    fake_supertonic.TTS = fake_tts_cls

    # Stub numpy — squeeze on (1, N) returns 1D
    import numpy as np
    fake_np = MagicMock(wraps=np)

    # Stub soundfile
    fake_sf = MagicMock()
    def _fake_write(path, audio, samplerate):
        Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake")
    fake_sf.write = _fake_write

    with patch.dict(
        "sys.modules",
        {"supertonic": fake_supertonic, "soundfile": fake_sf},
    ):
        yield fake_tts_instance, fake_tts_cls


# ---------------------------------------------------------------------------
# Registry / constants
# ---------------------------------------------------------------------------

class TestSuperTonicRegistration:
    def test_supertonic_is_a_builtin_provider(self):
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS
        assert "supertonic" in BUILTIN_TTS_PROVIDERS

    def test_supertonic_has_a_text_length_cap(self):
        from tools.tts_tool import PROVIDER_MAX_TEXT_LENGTH
        assert PROVIDER_MAX_TEXT_LENGTH.get("supertonic", 0) > 0


# ---------------------------------------------------------------------------
# _check_supertonic_available
# ---------------------------------------------------------------------------

class TestCheckSuperTonicAvailable:
    def test_returns_bool_without_raising(self):
        from tools.tts_tool import _check_supertonic_available
        # We don't care about the current environment's answer — just that
        # the probe never raises on a machine without supertonic installed.
        assert isinstance(_check_supertonic_available(), bool)

    def test_reports_available_when_package_present(self, monkeypatch):
        import importlib.util
        from tools.tts_tool import _check_supertonic_available

        fake_spec = MagicMock()
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda name: fake_spec if name == "supertonic" else None,
        )
        assert _check_supertonic_available() is True

    def test_reports_unavailable_when_package_missing(self, monkeypatch):
        import importlib.util
        from tools.tts_tool import _check_supertonic_available

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert _check_supertonic_available() is False


# ---------------------------------------------------------------------------
# _generate_supertonic_tts — stubbed so we don't need supertonic installed
# ---------------------------------------------------------------------------

class TestGenerateSuperTonicTts:
    def test_successful_wav_generation(self, tmp_path, mock_supertonic_module):
        from tools.tts_tool import _generate_supertonic_tts

        fake_tts, fake_cls = mock_supertonic_module
        output_path = str(tmp_path / "test.wav")
        result = _generate_supertonic_tts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.wav").exists()
        fake_cls.assert_called_once_with(auto_download=True)
        fake_tts.synthesize.assert_called_once()

    def test_config_passes_voice_lang_steps(self, tmp_path, mock_supertonic_module):
        from tools.tts_tool import _generate_supertonic_tts

        fake_tts, _ = mock_supertonic_module
        config = {
            "supertonic": {
                "voice": "F1",
                "lang": "fr",
                "steps": 2,
            }
        }
        _generate_supertonic_tts("Bonjour", str(tmp_path / "out.wav"), config)

        call_kwargs = fake_tts.synthesize.call_args.kwargs
        assert call_kwargs["lang"] == "fr"
        assert call_kwargs["steps"] == 2
        # get_voice_style was called with the configured voice
        fake_tts.get_voice_style.assert_called_once_with(voice_name="F1")

    def test_default_voice_lang_steps(self, tmp_path, mock_supertonic_module):
        from tools.tts_tool import (
            DEFAULT_SUPERTONIC_VOICE,
            DEFAULT_SUPERTONIC_LANG,
            DEFAULT_SUPERTONIC_STEPS,
            _generate_supertonic_tts,
        )

        fake_tts, _ = mock_supertonic_module
        _generate_supertonic_tts("Hi", str(tmp_path / "out.wav"), {})

        fake_tts.get_voice_style.assert_called_once_with(voice_name=DEFAULT_SUPERTONIC_VOICE)
        call_kwargs = fake_tts.synthesize.call_args.kwargs
        assert call_kwargs["lang"] == DEFAULT_SUPERTONIC_LANG
        assert call_kwargs["steps"] == DEFAULT_SUPERTONIC_STEPS

    def test_model_is_cached_across_calls(self, tmp_path, mock_supertonic_module):
        from tools.tts_tool import _generate_supertonic_tts

        _, fake_cls = mock_supertonic_module
        _generate_supertonic_tts("One", str(tmp_path / "a.wav"), {})
        _generate_supertonic_tts("Two", str(tmp_path / "b.wav"), {})

        # Same cache key → TTS class instantiated exactly once
        assert fake_cls.call_count == 1

    def test_non_wav_extension_triggers_ffmpeg_conversion(
        self, tmp_path, mock_supertonic_module, monkeypatch
    ):
        """Non-.wav output path causes WAV -> target ffmpeg conversion."""
        from tools import tts_tool as _tt

        calls = []

        def fake_shutil_which(cmd):
            return "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None

        def fake_run(cmd, check=False, timeout=None, **kw):
            calls.append(cmd)
            out_path = cmd[-1]
            Path(out_path).write_bytes(b"fake-mp3-data")
            return MagicMock(returncode=0)

        monkeypatch.setattr(_tt.shutil, "which", fake_shutil_which)
        monkeypatch.setattr(_tt.subprocess, "run", fake_run)

        output_path = str(tmp_path / "test.mp3")
        result = _tt._generate_supertonic_tts("Hi", output_path, {})

        assert result == output_path
        assert len(calls) == 1
        assert calls[0][0] == "/usr/bin/ffmpeg"

    def test_missing_supertonic_raises_import_error(self, tmp_path, monkeypatch):
        """When supertonic package is not installed, _import_supertonic raises."""
        monkeypatch.setitem(sys.modules, "supertonic", None)
        from tools.tts_tool import _generate_supertonic_tts

        with pytest.raises((ImportError, TypeError)):
            _generate_supertonic_tts("Hi", str(tmp_path / "out.wav"), {})


# ---------------------------------------------------------------------------
# text_to_speech_tool end-to-end (provider == "supertonic")
# ---------------------------------------------------------------------------

class TestTextToSpeechToolWithSuperTonic:
    def test_dispatches_to_supertonic(self, tmp_path, monkeypatch, mock_supertonic_module):
        from tools import tts_tool
        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setattr(tts_tool, "_import_supertonic", lambda: mock_supertonic_module[1])

        cfg = {"provider": "supertonic", "supertonic": {"voice": "M1"}}
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(text="hi", output_path=str(tmp_path / "clip.wav"))
        data = json.loads(result)

        assert data["success"] is True, data
        assert data["provider"] == "supertonic"
        assert Path(data["file_path"]).exists()

    def test_missing_package_surfaces_error(self, tmp_path, monkeypatch):
        from tools import tts_tool
        from tools.tts_tool import text_to_speech_tool

        def raise_import():
            raise ImportError("No module named 'supertonic'")

        monkeypatch.setattr(tts_tool, "_import_supertonic", raise_import)

        cfg = {"provider": "supertonic"}
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(text="hi", output_path=str(tmp_path / "clip.wav"))
        data = json.loads(result)

        assert data["success"] is False
        assert "supertonic" in data["error"]


# ---------------------------------------------------------------------------
# check_tts_requirements
# ---------------------------------------------------------------------------

class TestCheckTtsRequirementsSuperTonic:
    def test_supertonic_install_satisfies_requirements(self, monkeypatch):
        from tools import tts_tool
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "supertonic"})
        monkeypatch.setattr(tts_tool, "_check_supertonic_available", lambda: False)
        assert check_tts_requirements() is False

        monkeypatch.setattr(tts_tool, "_check_supertonic_available", lambda: True)
        assert check_tts_requirements() is True