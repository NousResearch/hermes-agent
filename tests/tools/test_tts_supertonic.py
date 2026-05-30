"""
Tests for the native Supertonic TTS provider.

These tests pin the registration / caching / dispatch paths for Supertonic
without requiring the ``supertonic`` package to actually be installed (the
synthesis step is monkey-patched to avoid needing the ONNX wheel or the
~400MB model download).
"""

import json
import wave
from pathlib import Path

import pytest

from tools import tts_tool
from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    DEFAULT_SUPERTONIC_VOICE,
    PROVIDER_MAX_TEXT_LENGTH,
    _check_supertonic_available,
    check_tts_requirements,
    text_to_speech_tool,
)


# ---------------------------------------------------------------------------
# Registry / constants
# ---------------------------------------------------------------------------

class TestSupertonicRegistration:
    def test_supertonic_is_a_builtin_provider(self):
        assert "supertonic" in BUILTIN_TTS_PROVIDERS

    def test_supertonic_has_a_text_length_cap(self):
        assert PROVIDER_MAX_TEXT_LENGTH.get("supertonic", 0) > 0


# ---------------------------------------------------------------------------
# _check_supertonic_available
# ---------------------------------------------------------------------------

class TestCheckSupertonicAvailable:
    def test_returns_bool_without_raising(self):
        # We don't care about the current environment's answer — just that
        # the probe never raises on a machine without supertonic installed.
        assert isinstance(_check_supertonic_available(), bool)


# ---------------------------------------------------------------------------
# _generate_supertonic_tts — stubbed so we don't need supertonic installed
# ---------------------------------------------------------------------------

def _write_min_wav(path: str) -> None:
    """Write a minimal valid WAV so file-size checks pass."""
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b"\x00\x00" * 1024)


class _StubSupertonicTTS:
    """Stand-in for supertonic.TTS used by the synthesis tests."""

    instances: list["_StubSupertonicTTS"] = []
    synth_calls: list[dict] = []

    def __init__(self, auto_download=False):
        self.auto_download = auto_download
        _StubSupertonicTTS.instances.append(self)

    def get_voice_style(self, voice_name):
        return {"voice_name": voice_name}

    def get_voice_style_from_path(self, path):
        return {"voice_style_path": path}

    def synthesize(self, text, voice_style, lang, total_steps, speed, **extra):
        _StubSupertonicTTS.synth_calls.append({
            "text": text,
            "voice_style": voice_style,
            "lang": lang,
            "total_steps": total_steps,
            "speed": speed,
            "extra": extra,
        })
        return ("FAKE_WAV", 44100)

    def save_audio(self, wav, out):
        _write_min_wav(out)


@pytest.fixture(autouse=True)
def _reset_supertonic_cache(monkeypatch):
    """Clear the module-level engine cache and stub state between tests."""
    tts_tool._supertonic_tts_cache.clear()
    _StubSupertonicTTS.instances = []
    _StubSupertonicTTS.synth_calls = []
    monkeypatch.setattr(tts_tool, "_import_supertonic", lambda: _StubSupertonicTTS)
    yield
    tts_tool._supertonic_tts_cache.clear()


class TestGenerateSupertonicTts:
    def test_loads_engine_and_writes_wav(self, tmp_path):
        out_path = str(tmp_path / "out.wav")
        config = {"supertonic": {"voice": "M2", "lang": "pl"}}

        result = tts_tool._generate_supertonic_tts("cześć", out_path, config)

        assert result == out_path
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0
        call = _StubSupertonicTTS.synth_calls[0]
        assert call["text"] == "cześć"
        assert call["lang"] == "pl"
        assert call["voice_style"] == {"voice_name": "M2"}

    def test_engine_cache_reused_across_calls(self, tmp_path):
        config = {"supertonic": {"voice": "M1"}}
        tts_tool._generate_supertonic_tts("one", str(tmp_path / "a.wav"), config)
        tts_tool._generate_supertonic_tts("two", str(tmp_path / "b.wav"), config)

        # TTS() should have been constructed exactly once (model is global).
        assert len(_StubSupertonicTTS.instances) == 1
        assert [c["text"] for c in _StubSupertonicTTS.synth_calls] == ["one", "two"]

    def test_defaults_applied_when_config_empty(self, tmp_path):
        result = tts_tool._generate_supertonic_tts("hi", str(tmp_path / "out.wav"), {})
        assert Path(result).exists()
        call = _StubSupertonicTTS.synth_calls[0]
        assert call["voice_style"] == {"voice_name": DEFAULT_SUPERTONIC_VOICE}
        assert call["lang"] == "en"
        assert call["speed"] == 1.0
        assert call["total_steps"] == 8

    def test_speed_and_total_steps_clamped(self, tmp_path):
        config = {"supertonic": {"speed": 9.0, "total_steps": 99}}
        tts_tool._generate_supertonic_tts("hi", str(tmp_path / "out.wav"), config)
        call = _StubSupertonicTTS.synth_calls[0]
        assert call["speed"] == 2.0       # clamped to SUPERTONIC_SPEED_MAX
        assert call["total_steps"] == 12  # clamped to SUPERTONIC_TOTAL_STEPS_MAX

        _StubSupertonicTTS.synth_calls.clear()
        config = {"supertonic": {"speed": 0.1, "total_steps": 1}}
        tts_tool._generate_supertonic_tts("hi", str(tmp_path / "out2.wav"), config)
        call = _StubSupertonicTTS.synth_calls[0]
        assert call["speed"] == 0.7       # clamped to SUPERTONIC_SPEED_MIN
        assert call["total_steps"] == 5   # clamped to SUPERTONIC_TOTAL_STEPS_MIN

    def test_voice_style_path_takes_precedence(self, tmp_path):
        style = tmp_path / "my_voice.style"
        style.write_bytes(b"fake style")
        config = {"supertonic": {"voice": "M1", "voice_style_path": str(style)}}
        tts_tool._generate_supertonic_tts("hi", str(tmp_path / "out.wav"), config)
        call = _StubSupertonicTTS.synth_calls[0]
        assert call["voice_style"] == {"voice_style_path": str(style)}

    def test_optional_knobs_forwarded_when_set(self, tmp_path):
        config = {
            "supertonic": {
                "max_chunk_length": 120,
                "silence_duration": 0.25,
            },
        }
        tts_tool._generate_supertonic_tts("hi", str(tmp_path / "out.wav"), config)
        extra = _StubSupertonicTTS.synth_calls[0]["extra"]
        assert extra["max_chunk_length"] == 120
        assert extra["silence_duration"] == 0.25


# ---------------------------------------------------------------------------
# text_to_speech_tool end-to-end (provider == "supertonic")
# ---------------------------------------------------------------------------

class TestTextToSpeechToolWithSupertonic:
    def test_dispatches_to_supertonic(self, tmp_path, monkeypatch):
        cfg = {"provider": "supertonic", "supertonic": {"voice": "M1"}}
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(text="hi", output_path=str(tmp_path / "clip.wav"))
        data = json.loads(result)

        assert data["success"] is True, data
        assert data["provider"] == "supertonic"
        assert Path(data["file_path"]).exists()

    def test_missing_package_surfaces_error(self, tmp_path, monkeypatch):
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

class TestCheckTtsRequirementsSupertonic:
    def test_supertonic_install_satisfies_requirements(self, monkeypatch):
        # Drop every other provider so we can isolate the supertonic signal.
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

        # Now toggle the supertonic check on and off.
        monkeypatch.setattr(tts_tool, "_check_supertonic_available", lambda: False)
        assert check_tts_requirements() is False

        monkeypatch.setattr(tts_tool, "_check_supertonic_available", lambda: True)
        assert check_tts_requirements() is True
