import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from gateway.session_context import _UNSET, _VAR_MAP, reset_session_env, set_session_env
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


def test_edge_direct_voice_override_beats_configured_voice(tmp_path, monkeypatch):
    out = tmp_path / "speech.mp3"
    seen = {}

    async def fake_edge(text: str, output_path: str, tts_config: dict) -> str:
        seen["voice"] = tts_config["edge"]["voice"]
        Path(output_path).write_bytes(b"mp3")
        return output_path

    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "edge", "edge": {"voice": "ro-RO-EmilNeural"}},
    )
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", fake_edge)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", Mock())

    result = json.loads(
        tts_tool.text_to_speech_tool(
            "hello",
            output_path=str(out),
            voice="en-US-AriaNeural",
        )
    )

    assert result["success"] is True
    assert result["voice"] == "en-US-AriaNeural"
    assert seen["voice"] == "en-US-AriaNeural"


@pytest.mark.asyncio
async def test_edge_tts_honors_session_voice_override(tmp_path, monkeypatch):
    seen = {}

    class FakeCommunicate:
        def __init__(self, text, **kwargs):
            seen["text"] = text
            seen["kwargs"] = kwargs

        async def save(self, output_path):
            Path(output_path).write_bytes(b"mp3")

    monkeypatch.setattr(
        tts_tool,
        "_import_edge_tts",
        lambda: type("FakeEdge", (), {"Communicate": FakeCommunicate}),
    )
    token = set_session_env("HERMES_VOICE_TTS_VOICE_OVERRIDE", "en-US-AriaNeural")
    try:
        out = tmp_path / "speech.mp3"
        result = await tts_tool._generate_edge_tts(
            "Hello world",
            str(out),
            {"edge": {"voice": "ro-RO-EmilNeural"}},
        )
    finally:
        reset_session_env("HERMES_VOICE_TTS_VOICE_OVERRIDE", token)

    assert result == str(out)
    assert seen["kwargs"]["voice"] == "en-US-AriaNeural"
