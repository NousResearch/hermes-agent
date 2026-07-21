import json
from pathlib import Path
from unittest.mock import Mock

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


def test_telegram_target_native_provider_aligns_explicit_path_to_ogg(tmp_path, monkeypatch):
    requested = tmp_path / "speech.mp3"
    generated_paths = []

    def fake_gemini(text: str, output_path: str, tts_config: dict) -> str:
        generated_paths.append(output_path)
        assert output_path.endswith(".ogg")
        Path(output_path).write_bytes(b"ogg")
        return output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
    monkeypatch.setattr(tts_tool, "_generate_gemini_tts", fake_gemini)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", Mock())

    result = json.loads(
        tts_tool.text_to_speech_tool(
            "hello",
            output_path=str(requested),
            target_platform="telegram",
        )
    )

    assert generated_paths == [str(requested.with_suffix(".ogg"))]
    assert result["success"] is True
    assert result["file_path"] == str(requested.with_suffix(".ogg"))
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{requested.with_suffix('.ogg')}"
    tts_tool._convert_to_opus.assert_not_called()


def test_telegram_target_native_provider_failure_reports_attempted_path(tmp_path, monkeypatch):
    requested = tmp_path / "speech.mp3"
    attempted = requested.with_suffix(".ogg")

    def fake_gemini(text: str, output_path: str, tts_config: dict) -> str:
        assert output_path == str(attempted)
        Path(output_path).write_bytes(b"partial ogg")
        raise RuntimeError("provider failed")

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
    monkeypatch.setattr(tts_tool, "_generate_gemini_tts", fake_gemini)

    result = json.loads(
        tts_tool.text_to_speech_tool(
            "hello",
            output_path=str(requested),
            target_platform="telegram",
        )
    )

    assert result["success"] is False
    assert result["attempted_file_path"] == str(attempted)
    assert "file_path" not in result
    assert "media_tag" not in result


def test_telegram_target_conversion_provider_starts_mp3_and_returns_ogg(tmp_path, monkeypatch):
    requested = tmp_path / "speech.ogg"
    generation_path = requested.with_suffix(".mp3")
    opus = requested.with_suffix(".ogg")
    generated_paths = []

    async def fake_edge(text: str, output_path: str, tts_config: dict) -> str:
        generated_paths.append(output_path)
        assert output_path == str(generation_path)
        Path(output_path).write_bytes(b"mp3")
        return output_path

    def fake_convert(path: str) -> str:
        assert path == str(generation_path)
        opus.write_bytes(b"ogg")
        return str(opus)

    convert = Mock(side_effect=fake_convert)

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", fake_edge)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    result = json.loads(
        tts_tool.text_to_speech_tool(
            "hello",
            output_path=str(requested),
            target_platform="telegram",
        )
    )

    assert generated_paths == [str(generation_path)]
    assert result["success"] is True
    assert result["file_path"] == str(opus)
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{opus}"
    convert.assert_called_once_with(str(generation_path))


def test_target_platform_absent_preserves_manual_native_output_path(tmp_path, monkeypatch):
    requested = tmp_path / "speech.mp3"
    generated_paths = []

    def fake_gemini(text: str, output_path: str, tts_config: dict) -> str:
        generated_paths.append(output_path)
        assert output_path.endswith(".mp3")
        Path(output_path).write_bytes(b"mp3")
        return output_path

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
    monkeypatch.setattr(tts_tool, "_generate_gemini_tts", fake_gemini)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(requested)))

    assert generated_paths == [str(requested)]
    assert result["success"] is True
    assert result["file_path"] == str(requested)
    assert result["voice_compatible"] is False


def test_telegram_target_command_provider_keeps_configured_format(tmp_path, monkeypatch):
    requested = tmp_path / "speech.mp3"
    convert = Mock()

    def fake_command(text, output_path, provider_name, config, tts_config):
        Path(output_path).write_bytes(b"mp3")
        return output_path

    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {
            "provider": "cmd",
            "providers": {
                "cmd": {
                    "type": "command",
                    "command": "fake",
                    "output_format": "mp3",
                    "voice_compatible": False,
                },
            },
        },
    )
    monkeypatch.setattr(tts_tool, "_generate_command_tts", fake_command)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    result = json.loads(
        tts_tool.text_to_speech_tool(
            "hello",
            output_path=str(requested),
            target_platform="telegram",
        )
    )

    assert result["success"] is True
    assert result["file_path"] == str(requested)
    assert result["voice_compatible"] is False
    assert result["media_tag"] == f"MEDIA:{requested}"
    convert.assert_not_called()


def test_telegram_target_plugin_provider_keeps_non_voice_output(tmp_path, monkeypatch):
    requested = tmp_path / "speech.mp3"
    convert = Mock()

    def fake_plugin(text, output_path, provider, tts_config):
        Path(output_path).write_bytes(b"mp3")
        return output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "cartesia"})
    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", fake_plugin)
    monkeypatch.setattr(tts_tool, "_plugin_provider_is_voice_compatible", lambda provider: False)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    result = json.loads(
        tts_tool.text_to_speech_tool(
            "hello",
            output_path=str(requested),
            target_platform="telegram",
        )
    )

    assert result["success"] is True
    assert result["file_path"] == str(requested)
    assert result["voice_compatible"] is False
    assert result["media_tag"] == f"MEDIA:{requested}"
    convert.assert_not_called()
