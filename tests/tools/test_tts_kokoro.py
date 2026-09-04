"""
Tests for the native Kokoro TTS provider.

These tests pin the resolution / caching / dispatch paths for Kokoro
without requiring the ``kokoro-onnx`` package to actually be installed
(the synthesis step is monkey-patched to avoid needing the ONNX wheel).
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool
from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    DEFAULT_KOKORO_VOICE,
    PROVIDER_MAX_TEXT_LENGTH,
    _check_kokoro_available,
    _resolve_kokoro_model_paths,
    check_tts_requirements,
    text_to_speech_tool,
)


# ---------------------------------------------------------------------------
# Registry / constants
# ---------------------------------------------------------------------------

class TestKokoroRegistration:
    def test_kokoro_is_a_builtin_provider(self):
        assert "kokoro" in BUILTIN_TTS_PROVIDERS

    def test_kokoro_has_a_text_length_cap(self):
        assert PROVIDER_MAX_TEXT_LENGTH.get("kokoro", 0) > 0


# ---------------------------------------------------------------------------
# _check_kokoro_available
# ---------------------------------------------------------------------------

class TestCheckKokoroAvailable:
    def test_returns_bool_without_raising(self):
        # We don't care about the current environment's answer — just that
        # the probe never raises on a machine without kokoro-onnx installed.
        assert isinstance(_check_kokoro_available(), bool)


# ---------------------------------------------------------------------------
# _resolve_kokoro_model_paths
# ---------------------------------------------------------------------------

class TestResolveKokoroModelPaths:
    def test_custom_model_path_returned_as_is(self, tmp_path):
        model = tmp_path / "custom.onnx"
        model.write_bytes(b"fake onnx bytes")
        voices = tmp_path / "voices.bin"
        voices.write_bytes(b"fake voices")

        config = {"model_path": str(model), "voices_path": str(voices)}
        model_str, voices_str = _resolve_kokoro_model_paths(config, tmp_path)
        assert model_str == str(model)
        assert voices_str == str(voices)

    def test_custom_model_path_missing_raises(self, tmp_path):
        config = {"model_path": str(tmp_path / "nonexistent.onnx")}
        with pytest.raises(RuntimeError, match="model_path does not exist"):
            _resolve_kokoro_model_paths(config, tmp_path)

    def test_unknown_model_variant_raises(self, tmp_path):
        config = {"model": "quantum"}
        with pytest.raises(RuntimeError, match="Unknown Kokoro model variant"):
            _resolve_kokoro_model_paths(config, tmp_path)

    def test_int8_variant_downloads(self, tmp_path, monkeypatch):
        downloaded = []

        def fake_download(filename, dest_dir, timeout=600):
            p = dest_dir / filename
            p.write_bytes(b"fake")
            downloaded.append(filename)
            return p

        monkeypatch.setattr(tts_tool, "_download_kokoro_file", fake_download)
        config = {}
        model_str, voices_str = _resolve_kokoro_model_paths(config, tmp_path)
        assert "kokoro-v1.0.int8.onnx" in model_str
        assert "voices-v1.0.bin" in voices_str
        assert "kokoro-v1.0.int8.onnx" in downloaded
        assert "voices-v1.0.bin" in downloaded

    def test_fp32_variant_downloads(self, tmp_path, monkeypatch):
        downloaded = []

        def fake_download(filename, dest_dir, timeout=600):
            p = dest_dir / filename
            p.write_bytes(b"fake")
            downloaded.append(filename)
            return p

        monkeypatch.setattr(tts_tool, "_download_kokoro_file", fake_download)
        config = {"model": "fp32"}
        model_str, _ = _resolve_kokoro_model_paths(config, tmp_path)
        assert "kokoro-v1.0.onnx" in model_str
        assert model_str.endswith("kokoro-v1.0.onnx")


# ---------------------------------------------------------------------------
# _generate_kokoro_tts — stubbed so we don't need kokoro-onnx installed
# ---------------------------------------------------------------------------

class _StubKokoro:
    """Stand-in for kokoro_onnx.Kokoro used by the synthesis tests."""

    loaded: list[str] = []
    calls: list[tuple] = []
    from_session_calls: list[tuple] = []

    def __init__(self, model_path, voices_path):
        self.model_path = model_path
        self.voices_path = voices_path
        _StubKokoro.loaded.append(model_path)

    @classmethod
    def from_session(cls, session, voices_path):
        cls.from_session_calls.append((session, voices_path))
        instance = cls.__new__(cls)
        instance.voices_path = voices_path
        instance.model_path = "<session>"
        cls.loaded.append("<session>")
        return instance

    def create(self, text, voice="af_heart", speed=1.0, lang="en-us"):
        _StubKokoro.calls.append((text, voice, speed, lang))
        # Return a simple list — the soundfile stub doesn't care about type.
        # Avoid importing numpy here: re-importing numpy's C extension in the
        # same process after monkeypatching sys.modules raises ImportError.
        samples = [0.0] * 240  # 10ms of silence at 24kHz
        return samples, 24000


@pytest.fixture(autouse=True)
def _reset_kokoro_cache():
    """Clear the module-level model cache between tests."""
    tts_tool._kokoro_model_cache.clear()
    _StubKokoro.loaded = []
    _StubKokoro.calls = []
    _StubKokoro.from_session_calls = []
    yield
    tts_tool._kokoro_model_cache.clear()


@pytest.fixture
def mock_soundfile():
    """Stub soundfile — the real package isn't installed in CI venv, and
    _generate_kokoro_tts does `import soundfile as sf` at runtime.
    """
    fake_sf = MagicMock()

    def _fake_write(path, audio, samplerate):
        Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake")

    fake_sf.write = _fake_write
    with patch.dict("sys.modules", {"soundfile": fake_sf}):
        yield fake_sf


class TestGenerateKokoroTts:
    def _prepare_model_files(self, tmp_path):
        model = tmp_path / "kokoro-v1.0.int8.onnx"
        model.write_bytes(b"fake model")
        voices = tmp_path / "voices-v1.0.bin"
        voices.write_bytes(b"fake voices")
        return model, voices

    def test_loads_model_and_writes_wav(self, tmp_path, monkeypatch, mock_soundfile):
        model, voices = self._prepare_model_files(tmp_path)
        monkeypatch.setattr(tts_tool, "_import_kokoro", lambda: _StubKokoro)

        out_path = str(tmp_path / "out.wav")
        config = {"kokoro": {"model_path": str(model), "voices_path": str(voices)}}

        result = tts_tool._generate_kokoro_tts("hello", out_path, config)

        assert result == out_path
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0
        assert _StubKokoro.loaded == [str(model)]
        assert _StubKokoro.calls[0][0] == "hello"

    def test_model_cache_reused_across_calls(self, tmp_path, monkeypatch, mock_soundfile):
        model, voices = self._prepare_model_files(tmp_path)
        monkeypatch.setattr(tts_tool, "_import_kokoro", lambda: _StubKokoro)

        config = {"kokoro": {"model_path": str(model), "voices_path": str(voices)}}
        tts_tool._generate_kokoro_tts("one", str(tmp_path / "a.wav"), config)
        tts_tool._generate_kokoro_tts("two", str(tmp_path / "b.wav"), config)

        # Kokoro() constructor should have been called exactly once.
        assert _StubKokoro.loaded == [str(model)]
        # But both create() calls went through.
        assert [c[0] for c in _StubKokoro.calls] == ["one", "two"]

    def test_voice_passed_through_to_create(self, tmp_path, monkeypatch, mock_soundfile):
        model, voices = self._prepare_model_files(tmp_path)
        monkeypatch.setattr(tts_tool, "_import_kokoro", lambda: _StubKokoro)

        config = {"kokoro": {
            "model_path": str(model),
            "voices_path": str(voices),
            "voice": "am_michael",
            "lang": "en-us",
            "speed": 1.2,
        }}
        tts_tool._generate_kokoro_tts("hi", str(tmp_path / "out.wav"), config)

        assert _StubKokoro.calls[0][1] == "am_michael"
        assert _StubKokoro.calls[0][2] == 1.2
        assert _StubKokoro.calls[0][3] == "en-us"

    def test_use_cuda_falls_back_to_cpu_on_failure(self, tmp_path, monkeypatch, mock_soundfile):
        model, voices = self._prepare_model_files(tmp_path)
        monkeypatch.setattr(tts_tool, "_import_kokoro", lambda: _StubKokoro)

        # Make onnxruntime import fail so the CUDA path raises.
        monkeypatch.setitem(sys.modules, "onnxruntime", None)

        config = {"kokoro": {
            "model_path": str(model),
            "voices_path": str(voices),
            "use_cuda": True,
        }}
        result = tts_tool._generate_kokoro_tts("hi", str(tmp_path / "out.wav"), config)
        assert Path(result).exists()
        # Should have fallen back to the CPU constructor path.
        assert _StubKokoro.loaded == [str(model)]
        assert _StubKokoro.from_session_calls == []


# ---------------------------------------------------------------------------
# text_to_speech_tool end-to-end (provider == "kokoro")
# ---------------------------------------------------------------------------

class TestTextToSpeechToolWithKokoro:
    def test_dispatches_to_kokoro(self, tmp_path, monkeypatch, mock_soundfile):
        model = tmp_path / "kokoro-v1.0.int8.onnx"
        model.write_bytes(b"fake model")
        voices = tmp_path / "voices-v1.0.bin"
        voices.write_bytes(b"fake voices")

        monkeypatch.setattr(tts_tool, "_import_kokoro", lambda: _StubKokoro)

        cfg = {"provider": "kokoro", "kokoro": {
            "model_path": str(model), "voices_path": str(voices),
        }}
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(text="hi", output_path=str(tmp_path / "clip.wav"))
        data = json.loads(result)

        assert data["success"] is True, data
        assert data["provider"] == "kokoro"
        assert Path(data["file_path"]).exists()

    def test_missing_package_surfaces_error(self, tmp_path, monkeypatch):
        def raise_import():
            raise ImportError("No module named 'kokoro_onnx'")

        monkeypatch.setattr(tts_tool, "_import_kokoro", raise_import)

        cfg = {"provider": "kokoro"}
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(text="hi", output_path=str(tmp_path / "clip.wav"))
        data = json.loads(result)

        assert data["success"] is False
        assert "kokoro-onnx" in data["error"]


# ---------------------------------------------------------------------------
# check_tts_requirements
# ---------------------------------------------------------------------------

class TestCheckTtsRequirementsKokoro:
    def test_kokoro_install_satisfies_requirements(self, monkeypatch):
        # Drop every other provider so we can isolate the kokoro signal.
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "kokoro"})
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

        # Now toggle the kokoro check on and off.
        monkeypatch.setattr(tts_tool, "_check_kokoro_available", lambda: False)
        assert check_tts_requirements() is False

        monkeypatch.setattr(tts_tool, "_check_kokoro_available", lambda: True)
        assert check_tts_requirements() is True
