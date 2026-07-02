import asyncio
import json
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

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


def test_edge_explicit_ogg_writes_temp_mp3_then_converts_to_target(tmp_path, monkeypatch):
    out = tmp_path / "speech.ogg"
    saved_paths: list[str] = []
    convert_calls: list[tuple[str, str]] = []

    class FakeCommunicate:
        def __init__(self, text: str, **kwargs):
            assert text == "hello"
            assert kwargs["voice"] == tts_tool.DEFAULT_EDGE_VOICE

        async def save(self, path: str) -> None:
            saved_paths.append(path)
            Path(path).write_bytes(b"mp3 bytes")

    def fake_convert(input_path: str, output_path: str | None = None) -> str:
        assert output_path == str(out)
        assert output_path is not None
        assert input_path != str(out)
        assert input_path.endswith(".mp3")
        convert_calls.append((input_path, output_path))
        out.write_bytes(b"OggS converted")
        return str(out)

    monkeypatch.setattr(
        tts_tool,
        "_import_edge_tts",
        lambda: SimpleNamespace(Communicate=FakeCommunicate),
    )
    monkeypatch.setattr(tts_tool, "_convert_to_opus", fake_convert)

    result = asyncio.run(tts_tool._generate_edge_tts("hello", str(out), {}))

    assert result == str(out)
    assert out.read_bytes().startswith(b"OggS")
    assert len(saved_paths) == 1
    assert saved_paths[0] != str(out)
    assert not Path(saved_paths[0]).exists()
    assert convert_calls == [(saved_paths[0], str(out))]


def test_telegram_explicit_ogg_for_non_native_provider_is_transcoded(tmp_path, monkeypatch):
    out = tmp_path / "speech.ogg"
    convert_calls: list[tuple[str, str | None]] = []

    def fake_generate_piper(_text: str, output_path: str, _tts_config: dict) -> str:
        assert output_path == str(out)
        Path(output_path).write_bytes(b"OggS fake vorbis")
        return output_path

    def fake_convert(input_path: str, output_path: str | None = None) -> str:
        assert output_path is not None
        assert input_path != str(out)
        assert output_path != str(out)
        convert_calls.append((input_path, output_path))
        Path(output_path).write_bytes(b"OggS opus")
        return output_path

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "piper"})
    monkeypatch.setattr(tts_tool, "_import_piper", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_piper_tts", fake_generate_piper)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", fake_convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

    assert result["success"] is True
    assert result["file_path"] == str(out)
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{out}"
    assert out.read_bytes() == b"OggS opus"
    assert len(convert_calls) == 1
