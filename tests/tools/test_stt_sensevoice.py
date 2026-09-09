"""SenseVoiceSmall GGUF provider contracts."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import transcription_tools
from tools.transcription_sensevoice import _sensevoice_binary


def test_sensevoice_dispatches_locally_with_configured_gguf(tmp_path, monkeypatch):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    model = tmp_path / "sensevoice-small-q8.gguf"
    model.write_bytes(b"GGUF")
    vad = tmp_path / "fsmn-vad.gguf"
    vad.write_bytes(b"GGUF")
    binary = tmp_path / "llama-funasr-sensevoice"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return SimpleNamespace(stdout="粵語 mixed English\n", stderr="")

    config = {
        "provider": "sensevoice",
        "sensevoice": {
            "binary": str(binary),
            "model": str(model),
            "vad_model": str(vad),
            "backend": "cpu",
        },
    }
    monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: config)
    with patch("tools.transcription_sensevoice._run_quiet", side_effect=fake_run):
        result = transcription_tools.transcribe_audio(str(audio))

    assert result == {
        "success": True,
        "transcript": "粵語 mixed English",
        "provider": "sensevoice",
    }
    assert captured["command"] == [
        str(binary), "-m", str(model), "-a", str(audio),
        "--vad", str(vad), "--backend", "cpu",
    ]
    assert captured["kwargs"]["timeout"] == 300
    assert "OPENAI_API_KEY" not in captured["kwargs"]["env"]


def test_sensevoice_reports_missing_model_before_spawning(tmp_path, monkeypatch):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")
    config = {
        "provider": "sensevoice",
        "sensevoice": {"binary": "llama-funasr-sensevoice", "model": ""},
    }
    monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: config)

    with patch("tools.transcription_sensevoice._run_quiet") as run:
        result = transcription_tools.transcribe_audio(str(audio))

    assert result["success"] is False
    assert "stt.sensevoice.model" in result["error"]
    run.assert_not_called()


@pytest.mark.linux_only
def test_sensevoice_rejects_non_executable_explicit_binary(tmp_path):
    binary = tmp_path / "llama-funasr-sensevoice"
    binary.write_bytes(b"binary")
    binary.chmod(0o644)

    assert os.access(binary, os.X_OK) is False
    assert _sensevoice_binary(str(binary)) is None