import os
from pathlib import Path

import pytest

from tools import transcription_tools


def test_missing_api_key_returns_error_and_logs_error(caplog, tmp_path: Path, monkeypatch) -> None:
    """Force OpenAI provider so we exercise the OpenAI backend error path."""
    caplog.set_level("ERROR")
    audio_file = tmp_path / "audio.ogg"
    audio_file.write_bytes(b"fake-audio")

    # Ensure provider resolution selects OpenAI and no key is present
    monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "", prepend=False)
    monkeypatch.setenv("HERMES_STT_PROVIDER", "openai", prepend=False)

    # Avoid hitting faster-whisper in tests
    monkeypatch.setattr(transcription_tools, "_HAS_FASTER_WHISPER", False, raising=False)
    monkeypatch.setattr(transcription_tools, "_HAS_OPENAI", True, raising=False)

    result = transcription_tools._transcribe_openai(str(audio_file), model_name="whisper-1")

    assert result["success"] is False
    assert result["transcript"] == ""
    assert "VOICE_TOOLS_OPENAI_KEY not set" in result["error"]
    assert any(
        "VOICE_TOOLS_OPENAI_KEY is not set" in message for message in caplog.messages
    )


def test_unsupported_extension_returns_clear_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "test-key")
    bad_file = tmp_path / "audio.txt"
    bad_file.write_text("not audio")

    result = transcription_tools.transcribe_audio(str(bad_file))

    assert result["success"] is False
    assert "Unsupported format" in result["error"]


def test_env_model_override_used_when_model_not_provided(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import patch

    # Force OpenAI provider through config/env
    monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "test-key")
    monkeypatch.setenv("VOICE_TOOLS_STT_MODEL", "gpt-4o-mini-transcribe")
    monkeypatch.setenv("HERMES_STT_PROVIDER", "openai", prepend=False)

    # Avoid local backend
    monkeypatch.setattr(transcription_tools, "_HAS_FASTER_WHISPER", False, raising=False)
    monkeypatch.setattr(transcription_tools, "_HAS_OPENAI", True, raising=False)

    audio_file = tmp_path / "audio.ogg"
    audio_file.write_bytes(b"fake-audio")

    with patch("tools.transcription_tools.OpenAI") as MockClient:
        instance = MockClient.return_value
        instance.audio.transcriptions.create.return_value = "hello world"

        result = transcription_tools.transcribe_audio(str(audio_file))

    assert result["success"] is True
    assert result["transcript"] == "hello world"
    instance.audio.transcriptions.create.assert_called_once()
    _, kwargs = instance.audio.transcriptions.create.call_args
    # Model should come from VOICE_TOOLS_STT_MODEL env var
    assert kwargs["model"] == "gpt-4o-mini-transcribe"


def test_unknown_model_emits_warning_but_still_calls_api(tmp_path: Path, monkeypatch, caplog) -> None:
    from unittest.mock import patch

    # Set log level for the transcription_tools logger specifically
    caplog.set_level("WARNING", logger="tools.transcription_tools")

    # Force OpenAI provider
    monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "test-key")
    monkeypatch.setenv("HERMES_STT_PROVIDER", "openai", prepend=False)
    monkeypatch.setattr(transcription_tools, "_HAS_FASTER_WHISPER", False, raising=False)
    monkeypatch.setattr(transcription_tools, "_HAS_OPENAI", True, raising=False)

    audio_file = tmp_path / "audio.ogg"
    audio_file.write_bytes(b"fake-audio")

    # Mock the logger to verify warning is called
    with patch("tools.transcription_tools.OpenAI") as MockClient, \
         patch("tools.transcription_tools.logger") as mock_logger:
        instance = MockClient.return_value
        instance.audio.transcriptions.create.return_value = "ok"

        result = transcription_tools.transcribe_audio(
            str(audio_file),
            model="custom-whisper-x",
        )

    assert result["success"] is True
    assert "ok" in result["transcript"]
    instance.audio.transcriptions.create.assert_called_once()
    # Verify warning was called with the model name
    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args
    assert "custom-whisper-x" in str(call_args), f"Expected 'custom-whisper-x' in warning call args: {call_args}"

