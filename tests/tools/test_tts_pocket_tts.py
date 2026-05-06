"""Tests for the Pocket TTS local provider in tools/tts_tool.py."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tools import tts_tool
from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    PROVIDER_MAX_TEXT_LENGTH,
    _check_pocket_tts_available,
    _generate_pocket_tts,
    _pocket_tts_model_cache,
    check_tts_requirements,
    text_to_speech_tool,
)


@pytest.fixture(autouse=True)
def _reset_pocket_tts_cache():
    tts_tool._pocket_tts_model_cache.clear()
    yield
    tts_tool._pocket_tts_model_cache.clear()


@pytest.fixture
def mock_pocket_tts():
    """Inject a fake pocket_tts module."""
    fake_audio = MagicMock()
    fake_audio.numpy.return_value = np.zeros(24000, dtype=np.float32)

    fake_model = MagicMock()
    fake_model.sample_rate = 24000
    fake_model.generate_audio.return_value = fake_audio
    fake_model.get_state_for_audio_prompt.return_value = {"voice": "mock"}

    fake_tts_model_cls = MagicMock()
    fake_tts_model_cls.load_model.return_value = fake_model

    fake_module = MagicMock()
    fake_module.TTSModel = fake_tts_model_cls

    with patch.dict("sys.modules", {"pocket_tts": fake_module}):
        yield fake_model, fake_tts_model_cls


@pytest.fixture
def mock_scipy():
    fake_wavfile = MagicMock()
    fake_scipy_io = MagicMock()
    fake_scipy_io.wavfile = fake_wavfile
    fake_scipy = MagicMock()
    fake_scipy.io = fake_scipy_io
    fake_scipy.io.wavfile = fake_wavfile

    with patch.dict("sys.modules", {
        "scipy": fake_scipy,
        "scipy.io": fake_scipy_io,
        "scipy.io.wavfile": fake_wavfile,
    }):
        yield fake_wavfile


# ---------------------------------------------------------------------------
# Registry / constants
# ---------------------------------------------------------------------------

class TestPocketTtsRegistration:
    def test_pocket_tts_is_a_builtin_provider(self):
        assert "pocket_tts" in BUILTIN_TTS_PROVIDERS

    def test_pocket_tts_has_a_text_length_cap(self):
        assert PROVIDER_MAX_TEXT_LENGTH.get("pocket_tts", 0) > 0


# ---------------------------------------------------------------------------
# _check_pocket_tts_available
# ---------------------------------------------------------------------------

class TestCheckPocketTtsAvailable:
    def test_reports_available_when_package_present(self, monkeypatch):
        import importlib.util
        fake_spec = MagicMock()
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda name: fake_spec if name == "pocket_tts" else None,
        )
        assert _check_pocket_tts_available() is True

    def test_reports_unavailable_when_package_missing(self, monkeypatch):
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert _check_pocket_tts_available() is False

    def test_handles_import_error_gracefully(self, monkeypatch):
        import importlib.util
        def raise_import_error(name):
            raise ImportError("forced failure")
        monkeypatch.setattr(importlib.util, "find_spec", raise_import_error)
        assert _check_pocket_tts_available() is False


# ---------------------------------------------------------------------------
# _generate_pocket_tts
# ---------------------------------------------------------------------------

class TestGeneratePocketTts:
    def test_generates_wav_with_defaults(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, fake_tts_model_cls = mock_pocket_tts
        out = str(tmp_path / "out.wav")

        result = _generate_pocket_tts("Hello", out, {})

        fake_tts_model_cls.load_model.assert_called_once_with(language="english", temp=0.7)
        fake_model.get_state_for_audio_prompt.assert_called_once_with("alba")
        fake_model.generate_audio.assert_called_once_with({"voice": "mock"}, "Hello")
        mock_scipy.write.assert_called_once_with(out, 24000, fake_model.generate_audio.return_value.numpy.return_value)
        assert result == out

    def test_respects_language_config(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, fake_tts_model_cls = mock_pocket_tts
        config = {"pocket_tts": {"language": "french"}}

        _generate_pocket_tts("Bonjour", str(tmp_path / "out.wav"), config)

        fake_tts_model_cls.load_model.assert_called_once_with(language="french", temp=0.7)

    def test_24l_variant_appended_for_non_english(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, fake_tts_model_cls = mock_pocket_tts
        config = {"pocket_tts": {"language": "italian", "use_24l": True}}

        _generate_pocket_tts("Ciao", str(tmp_path / "out.wav"), config)

        fake_tts_model_cls.load_model.assert_called_once_with(language="italian_24l", temp=0.7)

    def test_24l_not_appended_for_english(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, fake_tts_model_cls = mock_pocket_tts
        config = {"pocket_tts": {"language": "english", "use_24l": True}}

        _generate_pocket_tts("Hello", str(tmp_path / "out.wav"), config)

        fake_tts_model_cls.load_model.assert_called_once_with(language="english", temp=0.7)

    def test_24l_not_doubled(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, fake_tts_model_cls = mock_pocket_tts
        config = {"pocket_tts": {"language": "french_24l", "use_24l": True}}

        _generate_pocket_tts("Bonjour", str(tmp_path / "out.wav"), config)

        fake_tts_model_cls.load_model.assert_called_once_with(language="french_24l", temp=0.7)

    def test_respects_voice_config(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, _ = mock_pocket_tts
        config = {"pocket_tts": {"voice": "cosette"}}

        _generate_pocket_tts("Hello", str(tmp_path / "out.wav"), config)

        fake_model.get_state_for_audio_prompt.assert_called_once_with("cosette")

    def test_voice_file_overrides_voice_name(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, _ = mock_pocket_tts
        config = {"pocket_tts": {"voice": "alba", "voice_file": "/path/to/ref.wav"}}

        _generate_pocket_tts("Hello", str(tmp_path / "out.wav"), config)

        fake_model.get_state_for_audio_prompt.assert_called_once_with("/path/to/ref.wav")

    def test_respects_temp_config(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, fake_tts_model_cls = mock_pocket_tts
        config = {"pocket_tts": {"temp": 0.5}}

        _generate_pocket_tts("Hello", str(tmp_path / "out.wav"), config)

        fake_tts_model_cls.load_model.assert_called_once_with(language="english", temp=0.5)

    def test_model_cached_across_calls(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, fake_tts_model_cls = mock_pocket_tts
        config = {"pocket_tts": {"language": "english"}}

        _generate_pocket_tts("First", str(tmp_path / "a.wav"), config)
        _generate_pocket_tts("Second", str(tmp_path / "b.wav"), config)

        assert fake_tts_model_cls.load_model.call_count == 1

    def test_different_languages_get_separate_cache_entries(self, tmp_path, mock_pocket_tts, mock_scipy):
        fake_model, fake_tts_model_cls = mock_pocket_tts

        _generate_pocket_tts("Hello", str(tmp_path / "a.wav"), {"pocket_tts": {"language": "english"}})
        _generate_pocket_tts("Bonjour", str(tmp_path / "b.wav"), {"pocket_tts": {"language": "french"}})

        assert fake_tts_model_cls.load_model.call_count == 2

    def test_mp3_output_with_ffmpeg(self, tmp_path, mock_pocket_tts, mock_scipy, monkeypatch):
        fake_model, _ = mock_pocket_tts
        out_mp3 = str(tmp_path / "out.mp3")
        expected_wav = str(tmp_path / "out.wav")

        def fake_which(cmd):
            return "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None

        ffmpeg_calls = []

        def fake_run(cmd, check=False, timeout=None, **kw):
            ffmpeg_calls.append(cmd)
            Path(cmd[-1]).write_bytes(b"fake-mp3-data")
            return MagicMock(returncode=0)

        monkeypatch.setattr(tts_tool.shutil, "which", fake_which)
        monkeypatch.setattr(tts_tool.subprocess, "run", fake_run)
        # Ensure the WAV path exists so os.remove doesn't fail
        Path(expected_wav).write_bytes(b"fake-wav")

        result = _generate_pocket_tts("Hello", out_mp3, {})

        assert len(ffmpeg_calls) == 1
        assert ffmpeg_calls[0][0] == "/usr/bin/ffmpeg"
        assert not Path(expected_wav).exists()
        assert result == out_mp3

    def test_mp3_output_without_ffmpeg_falls_back_to_rename(self, tmp_path, mock_pocket_tts, mock_scipy, monkeypatch):
        fake_model, _ = mock_pocket_tts
        out_mp3 = str(tmp_path / "out.mp3")
        expected_wav = str(tmp_path / "out.wav")

        monkeypatch.setattr(tts_tool.shutil, "which", lambda cmd: None)

        renamed = []
        real_rename = os.rename

        def fake_rename(src, dst):
            renamed.append((src, dst))

        monkeypatch.setattr(tts_tool.os, "rename", fake_rename)

        _generate_pocket_tts("Hello", out_mp3, {})

        assert len(renamed) == 1
        assert renamed[0] == (expected_wav, out_mp3)


# ---------------------------------------------------------------------------
# text_to_speech_tool dispatcher
# ---------------------------------------------------------------------------

class TestPocketTtsDispatcher:
    def test_not_installed_returns_helpful_error(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "pocket_tts", None)

        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"tts": {"provider": "pocket_tts"}})
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = json.loads(text_to_speech_tool(text="Hello"))

        assert result["success"] is False
        assert "pocket-tts" in result["error"]
        assert "pip install" in result["error"]


# ---------------------------------------------------------------------------
# check_tts_requirements
# ---------------------------------------------------------------------------

class TestCheckTtsRequirementsPocketTts:
    def test_pocket_tts_satisfies_requirements(self, monkeypatch):
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: (_ for _ in ()).throw(ImportError()))
        monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: (_ for _ in ()).throw(ImportError()))
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: (_ for _ in ()).throw(ImportError()))
        monkeypatch.setattr(tts_tool, "_import_mistral_client", lambda: (_ for _ in ()).throw(ImportError()))
        monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)
        monkeypatch.setattr(tts_tool, "_check_kittentts_available", lambda: False)
        monkeypatch.setattr(tts_tool, "_check_piper_available", lambda: False)
        monkeypatch.setattr(tts_tool, "_has_any_command_tts_provider", lambda: False)
        monkeypatch.setattr(tts_tool, "_has_openai_audio_backend", lambda: False)
        for env in ("MINIMAX_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY",
                    "GOOGLE_API_KEY", "MISTRAL_API_KEY", "ELEVENLABS_API_KEY"):
            monkeypatch.delenv(env, raising=False)

        monkeypatch.setattr(tts_tool, "_check_pocket_tts_available", lambda: False)
        assert check_tts_requirements() is False

        monkeypatch.setattr(tts_tool, "_check_pocket_tts_available", lambda: True)
        assert check_tts_requirements() is True
