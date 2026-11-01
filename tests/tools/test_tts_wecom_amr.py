"""Tests for WeCom AMR voice-bubble support (#78595).

WeCom (企业微信) renders native voice bubbles only as ``audio/amr``. Two
gateway/TTS gaps prevented AMR from reaching the adapter's existing
``send_voice`` path: the gateway audio-extension whitelist missed ``.amr``,
and the TTS ``voice_compatible`` logic transcoded every non-Ogg file to Opus
— a format WeCom cannot render as a voice bubble.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from gateway.platforms.base import _AUDIO_EXTS, _AUDIO_MIME_TYPES
from tools import tts_tool
from tools.tts_tool import text_to_speech_tool


# ---------------------------------------------------------------------------
# Bug 1: gateway audio whitelist must recognize .amr
# ---------------------------------------------------------------------------

def test_audio_mime_types_include_amr():
    assert ".amr" in _AUDIO_MIME_TYPES
    assert _AUDIO_MIME_TYPES[".amr"] == "audio/amr"
    assert ".amr" in _AUDIO_EXTS


# ---------------------------------------------------------------------------
# Bug 2: TTS must not transcode AMR to Opus on WeCom
# ---------------------------------------------------------------------------

def _python_copy_command(output_placeholder: str = "{output_path}") -> str:
    """Cross-platform shell command that copies {input_path} -> output."""
    import shlex
    import sys

    interpreter = sys.executable
    return (
        f'"{interpreter}" -c "import shutil, sys; '
        f'shutil.copyfile(sys.argv[1], sys.argv[2])" '
        f'{{input_path}} {output_placeholder}'
    )


def _command_cfg(provider_name: str, output_format: str) -> dict:
    return {
        "provider": provider_name,
        "providers": {
            provider_name: {
                "type": "command",
                "command": _python_copy_command(),
                "output_format": output_format,
                "voice_compatible": True,
            },
        },
    }


def test_command_tts_wecom_keeps_amr_voice(tmp_path, monkeypatch):
    """On WeCom, a voice_compatible command provider emitting .amr must keep
    the AMR file and report voice-compatible — no Opus transcode."""
    out = tmp_path / "clip.amr"
    # ffmpeg missing / transcode failing: _convert_to_opus returns None
    convert = Mock(return_value=None)

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "wecom")
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)
    with patch("tools.tts_tool._load_tts_config", return_value=_command_cfg("py-copy-amr", "amr")):
        result = text_to_speech_tool(text="hi", output_path=str(out))

    data = json.loads(result)
    assert data["success"] is True, data
    assert data["voice_compatible"] is True
    assert data["file_path"].endswith(".amr"), data["file_path"]
    assert data["media_tag"].startswith("[[audio_as_voice]]")
    convert.assert_not_called()


def test_command_tts_telegram_still_transcodes_amr_to_opus(tmp_path, monkeypatch):
    """The Opus transcode path for Opus-native platforms must be unchanged:
    on Telegram, an AMR-emitting command provider is still transcoded."""
    out = tmp_path / "clip.amr"
    opus = tmp_path / "clip.ogg"

    def fake_convert(path: str) -> str:
        assert path == str(out)
        opus.write_bytes(b"ogg")
        return str(opus)

    convert = Mock(side_effect=fake_convert)

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)
    with patch("tools.tts_tool._load_tts_config", return_value=_command_cfg("py-copy-amr", "amr")):
        result = text_to_speech_tool(text="hi", output_path=str(out))

    data = json.loads(result)
    assert data["success"] is True, data
    assert data["voice_compatible"] is True
    assert data["file_path"].endswith(".ogg"), data["file_path"]
    convert.assert_called_once_with(str(out))
