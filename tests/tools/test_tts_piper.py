"""
Tests for the native Piper TTS provider.

These tests pin the resolution / caching / dispatch paths for Piper
without requiring the ``piper-tts`` package to actually be installed
(the synthesis step is monkey-patched to avoid needing the ONNX wheel).
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool
from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    DEFAULT_PIPER_VOICE,
    PROVIDER_MAX_TEXT_LENGTH,
    _check_piper_available,
    _resolve_piper_voice_path,
    check_tts_requirements,
    text_to_speech_tool,
)


# ---------------------------------------------------------------------------
# Registry / constants
# ---------------------------------------------------------------------------

class TestPiperRegistration:
    def test_piper_is_a_builtin_provider(self):
        assert "piper" in BUILTIN_TTS_PROVIDERS

    def test_piper_has_a_text_length_cap(self):
        assert PROVIDER_MAX_TEXT_LENGTH.get("piper", 0) > 0


# ---------------------------------------------------------------------------
# _check_piper_available
# ---------------------------------------------------------------------------

class TestCheckPiperAvailable:
    def test_returns_bool_without_raising(self):
        # We don't care about the current environment's answer — just that
        # the probe never raises on a machine without piper installed.
        assert isinstance(_check_piper_available(), bool)


# ---------------------------------------------------------------------------
# _resolve_piper_voice_path
# ---------------------------------------------------------------------------

class TestResolvePiperVoicePath:
    def test_direct_onnx_path_returned_as_is(self, tmp_path):
        model = tmp_path / "custom.onnx"
        model.write_bytes(b"fake onnx bytes")
        result = _resolve_piper_voice_path(str(model), tmp_path)
        assert result == str(model)

    def test_cached_voice_name_not_redownloaded(self, tmp_path):
        """If both <voice>.onnx and <voice>.onnx.json exist in the
        download dir, no subprocess is spawned."""
        voice = "en_US-test-medium"
        (tmp_path / f"{voice}.onnx").write_bytes(b"model")
        (tmp_path / f"{voice}.onnx.json").write_text("{}")

        with patch("tools.tts_tool.subprocess.run") as mock_run:
            result = _resolve_piper_voice_path(voice, tmp_path)

        mock_run.assert_not_called()
        assert result == str(tmp_path / f"{voice}.onnx")

    def test_missing_voice_triggers_download(self, tmp_path):
        voice = "en_US-new-medium"

        def fake_run(cmd, *a, **kw):
            # Simulate a successful download: write the expected files.
            (tmp_path / f"{voice}.onnx").write_bytes(b"model")
            (tmp_path / f"{voice}.onnx.json").write_text("{}")
            return MagicMock(returncode=0, stderr="", stdout="")

        with patch("tools.tts_tool.subprocess.run", side_effect=fake_run) as mock_run:
            result = _resolve_piper_voice_path(voice, tmp_path)

        mock_run.assert_called_once()
        # Verify the command shape: python -m piper.download_voices <voice> --download-dir <dir>
        call_args = mock_run.call_args.args[0]
        assert "piper.download_voices" in " ".join(call_args)
        assert voice in call_args
        assert "--download-dir" in call_args
        assert str(tmp_path) in call_args
        assert result == str(tmp_path / f"{voice}.onnx")

    def test_download_failure_raises_runtime(self, tmp_path):
        voice = "en_US-broken-medium"
        fake_result = MagicMock(returncode=1, stderr="voice not found", stdout="")
        with patch("tools.tts_tool.subprocess.run", return_value=fake_result):
            with pytest.raises(RuntimeError, match="Piper voice download failed"):
                _resolve_piper_voice_path(voice, tmp_path)

    def test_download_success_but_missing_file_raises(self, tmp_path):
        voice = "en_US-weird-medium"
        fake_result = MagicMock(returncode=0, stderr="", stdout="")
        # Subprocess "succeeds" but doesn't actually write the files.
        with patch("tools.tts_tool.subprocess.run", return_value=fake_result):
            with pytest.raises(RuntimeError, match="completed but .+ is missing"):
                _resolve_piper_voice_path(voice, tmp_path)

    def test_empty_voice_falls_back_to_default_name(self, tmp_path):
        (tmp_path / f"{DEFAULT_PIPER_VOICE}.onnx").write_bytes(b"model")
        (tmp_path / f"{DEFAULT_PIPER_VOICE}.onnx.json").write_text("{}")
        result = _resolve_piper_voice_path("", tmp_path)
        assert result.endswith(f"{DEFAULT_PIPER_VOICE}.onnx")


# ---------------------------------------------------------------------------
# _generate_piper_tts — stubbed so we don't need piper-tts installed
# ---------------------------------------------------------------------------

class _StubPiperVoice:
    """Stand-in for piper.PiperVoice used by the synthesis tests."""

    loaded: list[str] = []
    calls: list[tuple] = []

    @classmethod
    def load(cls, model_path, use_cuda=False):
        cls.loaded.append(model_path)
        instance = cls()
        instance.model_path = model_path
        instance.use_cuda = use_cuda
        return instance

    def synthesize_wav(self, text, wav_file, syn_config=None):
        # Minimal valid WAV: an empty frame set is fine for our size check.
        # The wave module accepts any frames; we just need the file to exist
        # with non-zero bytes after close.
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 1024)
        _StubPiperVoice.calls.append((text, getattr(self, "model_path", ""), syn_config))


@pytest.fixture(autouse=True)
def _reset_piper_cache():
    """Clear the module-level voice cache between tests."""
    tts_tool._piper_voice_cache.clear()
    _StubPiperVoice.loaded = []
    _StubPiperVoice.calls = []
    yield
    tts_tool._piper_voice_cache.clear()


class TestGeneratePiperTts:
    def _prepare_voice_files(self, tmp_path, voice=DEFAULT_PIPER_VOICE):
        model = tmp_path / f"{voice}.onnx"
        model.write_bytes(b"model")
        (tmp_path / f"{voice}.onnx.json").write_text("{}")
        return model

    def test_loads_voice_and_writes_wav(self, tmp_path, monkeypatch):
        model = self._prepare_voice_files(tmp_path)
        monkeypatch.setattr(tts_tool, "_import_piper", lambda: _StubPiperVoice)

        out_path = str(tmp_path / "out.wav")
        config = {"piper": {"voice": str(model)}}

        result = tts_tool._generate_piper_tts("hello", out_path, config)

        assert result == out_path
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0
        assert _StubPiperVoice.loaded == [str(model)]
        assert _StubPiperVoice.calls[0][0] == "hello"

    def test_voice_cache_reused_across_calls(self, tmp_path, monkeypatch):
        model = self._prepare_voice_files(tmp_path)
        monkeypatch.setattr(tts_tool, "_import_piper", lambda: _StubPiperVoice)

        config = {"piper": {"voice": str(model)}}
        tts_tool._generate_piper_tts("one", str(tmp_path / "a.wav"), config)
        tts_tool._generate_piper_tts("two", str(tmp_path / "b.wav"), config)

        # load() should have been called exactly once for the same model+cuda key.
        assert _StubPiperVoice.loaded == [str(model)]
        # But both synthesize calls went through.
        assert [c[0] for c in _StubPiperVoice.calls] == ["one", "two"]

    def test_voice_name_triggers_download(self, tmp_path, monkeypatch):
        """A config voice of ``en_US-lessac-medium`` should be resolved via
        _resolve_piper_voice_path (which would normally download)."""
        monkeypatch.setattr(tts_tool, "_import_piper", lambda: _StubPiperVoice)

        def fake_resolve(voice, download_dir):
            model = download_dir / f"{voice}.onnx"
            model.write_bytes(b"model")
            return str(model)

        monkeypatch.setattr(tts_tool, "_resolve_piper_voice_path", fake_resolve)

        config = {"piper": {"voice": "en_US-lessac-medium", "voices_dir": str(tmp_path)}}
        result = tts_tool._generate_piper_tts("hi", str(tmp_path / "out.wav"), config)

        assert Path(result).exists()
        assert _StubPiperVoice.loaded[0].endswith("en_US-lessac-medium.onnx")

    def test_advanced_knobs_passed_as_synconfig(self, tmp_path, monkeypatch):
        model = self._prepare_voice_files(tmp_path)
        monkeypatch.setattr(tts_tool, "_import_piper", lambda: _StubPiperVoice)

        # Fake SynthesisConfig so we can assert the knobs flowed through.
        fake_syn_cls = MagicMock()

        class FakePiperModule:
            SynthesisConfig = fake_syn_cls

        # The SynthesisConfig import happens inline inside _generate_piper_tts
        # via ``from piper import SynthesisConfig``. Inject a fake piper
        # module so that import resolves.
        monkeypatch.setitem(sys.modules, "piper", FakePiperModule)

        config = {
            "piper": {
                "voice": str(model),
                "length_scale": 2.0,
                "volume": 0.8,
            },
        }
        tts_tool._generate_piper_tts(
            "slow voice", str(tmp_path / "out.wav"), config,
        )

        # SynthesisConfig was constructed with the advanced knobs.
        fake_syn_cls.assert_called_once()
        kwargs = fake_syn_cls.call_args.kwargs
        assert kwargs["length_scale"] == 2.0
        assert kwargs["volume"] == 0.8


# ---------------------------------------------------------------------------
# text_to_speech_tool end-to-end (provider == "piper")
# ---------------------------------------------------------------------------

class TestTextToSpeechToolWithPiper:
    def test_dispatches_to_piper(self, tmp_path, monkeypatch):
        model = tmp_path / f"{DEFAULT_PIPER_VOICE}.onnx"
        model.write_bytes(b"model")
        (tmp_path / f"{DEFAULT_PIPER_VOICE}.onnx.json").write_text("{}")

        monkeypatch.setattr(tts_tool, "_import_piper", lambda: _StubPiperVoice)

        cfg = {"provider": "piper", "piper": {"voice": str(model)}}
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(text="hi", output_path=str(tmp_path / "clip.wav"))
        data = json.loads(result)

        assert data["success"] is True, data
        assert data["provider"] == "piper"
        assert Path(data["file_path"]).exists()

    def test_missing_package_surfaces_error(self, tmp_path, monkeypatch):
        def raise_import():
            raise ImportError("No module named 'piper'")

        monkeypatch.setattr(tts_tool, "_import_piper", raise_import)

        cfg = {"provider": "piper"}
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(text="hi", output_path=str(tmp_path / "clip.wav"))
        data = json.loads(result)

        assert data["success"] is False
        assert "piper-tts" in data["error"]


# ---------------------------------------------------------------------------
# check_tts_requirements
# ---------------------------------------------------------------------------

class TestCheckTtsRequirementsPiper:
    def test_piper_install_satisfies_requirements(self, monkeypatch):
        # Drop every other provider so we can isolate the piper signal.
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: (_ for _ in ()).throw(ImportError()))
        monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: (_ for _ in ()).throw(ImportError()))
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: (_ for _ in ()).throw(ImportError()))
        monkeypatch.setattr(tts_tool, "_import_mistral_client", lambda: (_ for _ in ()).throw(ImportError()))
        monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)
        monkeypatch.setattr(tts_tool, "_check_kittentts_available", lambda: False)
        monkeypatch.setattr(tts_tool, "_has_any_command_tts_provider", lambda: False)
        monkeypatch.setattr(tts_tool, "_has_openai_audio_backend", lambda: False)
        for env in ("MINIMAX_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY",
                    "GOOGLE_API_KEY", "MISTRAL_API_KEY", "ELEVENLABS_API_KEY"):
            monkeypatch.delenv(env, raising=False)

        # Now toggle the piper check on and off.
        monkeypatch.setattr(tts_tool, "_check_piper_available", lambda: False)
        assert check_tts_requirements() is False

        monkeypatch.setattr(tts_tool, "_check_piper_available", lambda: True)
        assert check_tts_requirements() is True
