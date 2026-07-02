"""Regression tests for #54589: OpenAI-compatible TTS mp3 + local Opus transcode."""

from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool


@pytest.fixture(autouse=True)
def _openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _run_openai_tts(output_path: str, tts_config=None):
    mock_response = MagicMock()
    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = mock_response
    mock_cls = MagicMock(return_value=mock_client)

    with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), patch(
        "tools.tts_tool._resolve_openai_audio_client_config",
        return_value=("test-key", None),
    ):
        result = tts_tool._generate_openai_tts(
            "Hello",
            output_path,
            tts_config or {"openai": {}},
        )
    return result, mock_client, mock_response


def test_openai_tts_mp3_target_requests_mp3(tmp_path):
    out = tmp_path / "speech.mp3"
    result, mock_client, mock_response = _run_openai_tts(str(out))

    create_kwargs = mock_client.audio.speech.create.call_args[1]
    assert create_kwargs["response_format"] == "mp3"
    mock_response.stream_to_file.assert_called_once_with(str(out))
    assert result == str(out)


def test_openai_tts_ogg_target_synthesizes_mp3_then_transcodes(tmp_path, monkeypatch):
    ogg = tmp_path / "speech.ogg"
    mp3 = tmp_path / "speech.mp3"

    def fake_convert(path: str) -> str:
        assert path == str(mp3)
        ogg.write_bytes(b"ogg")
        return str(ogg)

    monkeypatch.setattr(tts_tool, "_convert_to_opus", fake_convert)

    result, mock_client, mock_response = _run_openai_tts(str(ogg))

    create_kwargs = mock_client.audio.speech.create.call_args[1]
    assert create_kwargs["response_format"] == "mp3"
    mock_response.stream_to_file.assert_called_once_with(str(mp3))
    assert result == str(ogg)


def test_openai_tts_ogg_target_falls_back_to_mp3_when_transcode_unavailable(
    tmp_path, monkeypatch
):
    ogg = tmp_path / "speech.ogg"
    mp3 = tmp_path / "speech.mp3"

    monkeypatch.setattr(tts_tool, "_convert_to_opus", lambda _path: None)

    result, mock_client, mock_response = _run_openai_tts(str(ogg))

    create_kwargs = mock_client.audio.speech.create.call_args[1]
    assert create_kwargs["response_format"] == "mp3"
    mock_response.stream_to_file.assert_called_once_with(str(mp3))
    assert result == str(mp3)


def test_openai_telegram_integration_uses_transcoded_opus(tmp_path, monkeypatch):
    """End-to-end through text_to_speech_tool on Telegram session."""
    ogg = tmp_path / "speech.ogg"

    def fake_openai(_text, output_path, _tts_config):
        assert output_path == str(ogg)
        ogg.write_bytes(b"ogg")
        return str(ogg)

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "openai"})
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_openai_tts", fake_openai)

    import json

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(ogg)))

    assert result["success"] is True
    assert result["file_path"] == str(ogg)
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{ogg}"
