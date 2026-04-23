"""Tests for the KittenTTS local provider in tools/tts_tool.py."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HERMES_SESSION_PLATFORM",):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def clear_kittentts_cache():
    """Reset the module-level model cache between tests."""
    from tools import tts_tool as _tt
    _tt._kittentts_model_cache.clear()
    yield
    _tt._kittentts_model_cache.clear()


@pytest.fixture
def mock_kittentts_module():
    """Inject a fake kittentts + soundfile module that return stub objects."""
    fake_model = MagicMock()
    # 24kHz float32 PCM at ~2s of silence
    fake_model.generate.return_value = np.zeros(48000, dtype=np.float32)
    fake_cls = MagicMock(return_value=fake_model)
    fake_kittentts = MagicMock()
    fake_kittentts.KittenTTS = fake_cls

    # Stub soundfile — the real package isn't installed in CI venv, and
    # _generate_kittentts does `import soundfile as sf` at runtime.
    fake_sf = MagicMock()
    def _fake_write(path, audio, samplerate):
        # Emulate writing a real file so downstream path checks succeed.
        import pathlib
        pathlib.Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake")
    fake_sf.write = _fake_write

    with patch.dict(
        "sys.modules",
        {"kittentts": fake_kittentts, "soundfile": fake_sf},
    ):
        yield fake_model, fake_cls


class TestGenerateKittenTts:
    def test_successful_wav_generation(self, tmp_path, mock_kittentts_module):
        from tools.tts_tool import _generate_kittentts

        fake_model, fake_cls = mock_kittentts_module
        output_path = str(tmp_path / "test.wav")
        result = _generate_kittentts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.wav").exists()
        fake_cls.assert_called_once()
        fake_model.generate.assert_called_once()

    def test_config_passes_voice_speed_cleantext(self, tmp_path, mock_kittentts_module):
        from tools.tts_tool import _generate_kittentts

        fake_model, _ = mock_kittentts_module
        config = {
            "kittentts": {
                "model": "KittenML/kitten-tts-mini-0.8",
                "voice": "Luna",
                "speed": 1.25,
                "clean_text": False,
            }
        }
        _generate_kittentts("Hi there", str(tmp_path / "out.wav"), config)

        call_kwargs = fake_model.generate.call_args.kwargs
        assert call_kwargs["voice"] == "Luna"
        assert call_kwargs["speed"] == 1.25
        assert call_kwargs["clean_text"] is False

    def test_default_model_and_voice(self, tmp_path, mock_kittentts_module):
        from tools.tts_tool import (
            DEFAULT_KITTENTTS_MODEL,
            DEFAULT_KITTENTTS_VOICE,
            _generate_kittentts,
        )

        fake_model, fake_cls = mock_kittentts_module
        _generate_kittentts("Hi", str(tmp_path / "out.wav"), {})

        fake_cls.assert_called_once_with(DEFAULT_KITTENTTS_MODEL)
        assert fake_model.generate.call_args.kwargs["voice"] == DEFAULT_KITTENTTS_VOICE

    def test_model_is_cached_across_calls(self, tmp_path, mock_kittentts_module):
        from tools.tts_tool import _generate_kittentts

        _, fake_cls = mock_kittentts_module
        _generate_kittentts("One", str(tmp_path / "a.wav"), {})
        _generate_kittentts("Two", str(tmp_path / "b.wav"), {})

        # Same model name → class instantiated exactly once
        assert fake_cls.call_count == 1

    def test_different_models_are_cached_separately(self, tmp_path, mock_kittentts_module):
        from tools.tts_tool import _generate_kittentts

        _, fake_cls = mock_kittentts_module
        _generate_kittentts(
            "A", str(tmp_path / "a.wav"),
            {"kittentts": {"model": "KittenML/kitten-tts-nano-0.8-int8"}},
        )
        _generate_kittentts(
            "B", str(tmp_path / "b.wav"),
            {"kittentts": {"model": "KittenML/kitten-tts-mini-0.8"}},
        )

        assert fake_cls.call_count == 2

    def test_non_wav_extension_triggers_ffmpeg_conversion(
        self, tmp_path, mock_kittentts_module, monkeypatch
    ):
        """Non-.wav output path causes WAV → target ffmpeg conversion."""
        from tools import tts_tool as _tt

        calls = []

        def fake_shutil_which(cmd):
            return "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None

        def fake_run(cmd, check=False, timeout=None, **kw):
            calls.append(cmd)
            # Emulate ffmpeg writing the output file
            import pathlib
            out_path = cmd[-1]
            pathlib.Path(out_path).write_bytes(b"fake-mp3-data")
            return MagicMock(returncode=0)

        monkeypatch.setattr(_tt.shutil, "which", fake_shutil_which)
        monkeypatch.setattr(_tt.subprocess, "run", fake_run)

        output_path = str(tmp_path / "test.mp3")
        result = _tt._generate_kittentts("Hi", output_path, {})

        assert result == output_path
        assert len(calls) == 1
        assert calls[0][0] == "/usr/bin/ffmpeg"

    def test_missing_kittentts_raises_import_error(self, tmp_path, monkeypatch):
        """When kittentts package is not installed, _import_kittentts raises."""
        import sys
        monkeypatch.setitem(sys.modules, "kittentts", None)
        from tools.tts_tool import _generate_kittentts

        with pytest.raises((ImportError, TypeError)):
            _generate_kittentts("Hi", str(tmp_path / "out.wav"), {})


class TestCheckKittenttsAvailable:
    def test_reports_available_when_package_present(self, monkeypatch):
        import importlib.util
        from tools.tts_tool import _check_kittentts_available

        fake_spec = MagicMock()
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda name: fake_spec if name == "kittentts" else None,
        )
        assert _check_kittentts_available() is True

    def test_reports_unavailable_when_package_missing(self, monkeypatch):
        import importlib.util
        from tools.tts_tool import _check_kittentts_available

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert _check_kittentts_available() is False


class TestDispatcherBranch:
    def test_kittentts_not_installed_returns_helpful_error(self, monkeypatch, tmp_path):
        """When provider=kittentts but package missing, return JSON error with setup hint."""
        import sys
        monkeypatch.setitem(sys.modules, "kittentts", None)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.tts_tool import text_to_speech_tool

        # Write a config telling it to use kittentts
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"tts": {"provider": "kittentts"}})
        )

        result = json.loads(text_to_speech_tool(text="Hello"))
        assert result["success"] is False
        assert "kittentts" in result["error"].lower()
        assert "hermes setup tts" in result["error"].lower()
