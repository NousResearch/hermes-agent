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


def _python_copy_command() -> str:
    """A command provider that copies the input text file to the output path.

    Uses ``sys.executable`` rather than a hard-coded ``python3`` so the test is
    portable to native Windows, matching ``tests/tools/test_tts_command_providers.py``.
    """
    interpreter = sys.executable
    return (
        f'"{interpreter}" -c "'
        "import sys;"
        "open(sys.argv[2],'wb').write(b'mp3')\" {input_path} {output_path}"
    )


def test_command_provider_telegram_auto_voice(tmp_path, monkeypatch):
    """A command provider on Telegram auto-converts to an Opus voice bubble
    even without an explicit voice_compatible opt-in (matches built-ins)."""
    out = tmp_path / "clip.mp3"
    opus = tmp_path / "clip.ogg"

    def fake_convert(path: str) -> str:
        opus.write_bytes(b"ogg")
        return str(opus)

    cfg = {
        "provider": "fal",
        "providers": {
            "fal": {
                "type": "command",
                "command": _python_copy_command(),
                "output_format": "mp3",
            },
        },
    }
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", Mock(side_effect=fake_convert))

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))
    assert result["success"] is True
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{opus}"


def test_command_provider_non_telegram_stays_document(tmp_path, monkeypatch):
    """Off Telegram, a command provider without opt-in stays a document."""
    out = tmp_path / "clip.mp3"
    cfg = {
        "provider": "fal",
        "providers": {
            "fal": {
                "type": "command",
                "command": _python_copy_command(),
                "output_format": "mp3",
            },
        },
    }
    convert = Mock()
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))
    assert result["success"] is True
    assert result["voice_compatible"] is False
    convert.assert_not_called()


def test_command_provider_explicit_opt_out_on_telegram(tmp_path, monkeypatch):
    """An explicit voice_compatible: false forces document delivery even on
    Telegram, overriding the auto-voice default."""
    out = tmp_path / "clip.mp3"
    cfg = {
        "provider": "fal",
        "providers": {
            "fal": {
                "type": "command",
                "command": _python_copy_command(),
                "output_format": "mp3",
                "voice_compatible": False,
            },
        },
    }
    convert = Mock()
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))
    assert result["success"] is True
    assert result["voice_compatible"] is False
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
