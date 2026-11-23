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


def test_edge_photon_converts_to_m4a_voice(tmp_path, monkeypatch):
    """iMessage voice notes are AAC/MP4, not Opus: a voice platform may want another container."""
    out = tmp_path / "speech.mp3"
    m4a = tmp_path / "speech.m4a"

    def fake_convert(path: str) -> str:
        assert path == str(out)
        m4a.write_bytes(b"m4a")
        return str(m4a)

    convert = Mock(side_effect=fake_convert)
    opus = Mock()

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "photon")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _write_edge_output)
    monkeypatch.setattr(tts_tool, "_convert_to_m4a", convert)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", opus)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

    assert result["success"] is True
    assert result["file_path"] == str(m4a)
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{m4a}"
    convert.assert_called_once_with(str(out))
    opus.assert_not_called()


def test_native_opus_provider_converts_for_non_opus_voice_platform(tmp_path, monkeypatch):
    """A provider that can emit Opus still needs transcoding where the bubble wants another
    container — it wrote MP3 because .ogg would be the wrong target."""
    mp3 = tmp_path / "speech.mp3"
    mp3.write_bytes(b"mp3")
    m4a = tmp_path / "speech.m4a"
    monkeypatch.setattr(tts_tool, "_convert_to_m4a", lambda _p: str(m4a))

    path, voice_compatible = tts_tool._finalize_voice_delivery(
        str(mp3), "elevenlabs", None, "m4a")

    assert (path, voice_compatible) == (str(m4a), True)

