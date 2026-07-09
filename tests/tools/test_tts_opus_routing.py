import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from gateway.session_context import _UNSET, _VAR_MAP
from tools import tts_tool


def _reset_session_context() -> None:
    for var in _VAR_MAP.values():
        var.set(_UNSET)


@pytest.fixture(autouse=True)
def _clean_session_platform(monkeypatch):
    _reset_session_context()
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    yield
    _reset_session_context()


async def _write_edge_output(_text: str, output_path: str, _tts_config: dict) -> str:
    Path(output_path).write_bytes(b"mp3")
    return output_path


def test_edge_cli_preserves_native_mp3(tmp_path, monkeypatch):
    out = tmp_path / "speech.mp3"
    convert = Mock()

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _write_edge_output)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

    assert result["success"] is True
    assert result["file_path"] == str(out)
    assert result["voice_compatible"] is False
    assert result["media_tag"] == f"MEDIA:{out}"
    convert.assert_not_called()


def test_edge_telegram_converts_to_opus_voice(tmp_path, monkeypatch):
    out = tmp_path / "speech.mp3"
    opus = tmp_path / "speech.ogg"

    def fake_convert(path: str) -> str:
        assert path == str(out)
        opus.write_bytes(b"ogg")
        return str(opus)

    convert = Mock(side_effect=fake_convert)

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _write_edge_output)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

    assert result["success"] is True
    assert result["file_path"] == str(opus)
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{opus}"
    convert.assert_called_once_with(str(out))


def test_openai_compatible_telegram_transcodes_mp3_for_opus_voice(
    tmp_path, monkeypatch
):
    out = tmp_path / "speech.ogg"
    synth = tmp_path / "speech.mp3"
    create = MagicMock()

    def fake_stream(path: str) -> None:
        Path(path).write_bytes(b"mp3")

    response = MagicMock()
    response.stream_to_file.side_effect = fake_stream
    create.return_value = response

    mock_client = MagicMock()
    mock_client.audio.speech.create = create
    mock_client_cls = MagicMock(return_value=mock_client)

    def fake_convert(path: str) -> str:
        assert path == str(synth)
        out.write_bytes(b"ogg")
        return str(out)

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {
            "provider": "openai",
            "openai": {"base_url": "https://compatible.example/v1"},
        },
    )
    monkeypatch.setattr(
        tts_tool,
        "_resolve_openai_audio_client_config",
        lambda: ("key", None, False),
    )
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: mock_client_cls)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", fake_convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

    assert result["success"] is True
    assert result["file_path"] == str(out)
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{out}"
    assert create.call_args.kwargs["response_format"] == "mp3"
    response.stream_to_file.assert_called_once_with(str(synth))
