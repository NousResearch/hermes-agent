import json
import sys
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


def test_edge_matrix_converts_to_opus_voice(tmp_path, monkeypatch):
    """Matrix voice bubbles need Ogg/Opus too (MSC3245, issue #14841)."""
    out = tmp_path / "speech.mp3"
    opus = tmp_path / "speech.ogg"

    def fake_convert(path: str) -> str:
        assert path == str(out)
        opus.write_bytes(b"ogg")
        return str(opus)

    convert = Mock(side_effect=fake_convert)

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "matrix")
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


# --- an opted-in command provider keeps its native bytes unless the platform needs Opus ---

_OPT_IN_CMD = (
    f'"{sys.executable}" -c "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])" '
    "{input_path} {output_path}"
)


def _opt_in_command_cfg() -> dict:
    """A command provider that opts into voice delivery but emits MP3 (the langauto shape)."""
    return {
        "provider": "opt-in-cli",
        "providers": {
            "opt-in-cli": {
                "type": "command",
                "command": _OPT_IN_CMD,
                "output_format": "mp3",
                "voice_compatible": True,
            },
        },
    }


def test_opted_in_command_provider_keeps_mp3_on_slack(tmp_path, monkeypatch):
    """Slack plays MP3 inline; forcing Opus there lands as a plain file attachment."""
    out = tmp_path / "cmdtone.mp3"
    convert = Mock()

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "slack")
    monkeypatch.setattr(tts_tool, "_load_tts_config", _opt_in_command_cfg)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

    assert result["success"] is True
    assert result["file_path"] == str(out)
    assert result["voice_compatible"] is False
    assert result["media_tag"] == f"MEDIA:{out}"
    convert.assert_not_called()


@pytest.mark.parametrize("platform", ["telegram", "discord"])
def test_opted_in_command_provider_still_converts_on_opus_platforms(tmp_path, monkeypatch, platform):
    """Opus-requiring platforms (Discord uploads a hard-coded voice-message.ogg) still convert."""
    out = tmp_path / "cmdtone.mp3"
    opus = tmp_path / "cmdtone.ogg"

    def fake_convert(path: str) -> str:
        assert path == str(out)
        opus.write_bytes(b"ogg")
        return str(opus)

    convert = Mock(side_effect=fake_convert)

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    monkeypatch.setattr(tts_tool, "_load_tts_config", _opt_in_command_cfg)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

    assert result["success"] is True
    assert result["file_path"] == str(opus)
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{opus}"
    convert.assert_called_once_with(str(out))


def test_opted_in_plugin_provider_keeps_mp3_without_opus_platform(tmp_path, monkeypatch):
    """Same gate for the plugin backend branch (command_provider_config is None there)."""
    mp3 = tmp_path / "plugin.mp3"
    mp3.write_bytes(b"mp3")
    convert = Mock()

    monkeypatch.setattr(tts_tool, "_plugin_provider_is_voice_compatible", lambda _name: True)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    path, voice_compatible = tts_tool._finalize_voice_delivery(
        str(mp3), "cartesia", None, want_opus=False)

    assert path == str(mp3)
    assert voice_compatible is False
    convert.assert_not_called()
