"""A failed text_to_speech call must never destroy a file that already lived at output_path.

The model picks ``output_path``; before this fix a provider failure swept that path as a
"partial artifact" even when it was the user's pre-existing file.
"""

import json
from pathlib import Path

import pytest

from gateway.session_context import _UNSET, _VAR_MAP
from tools import tts_tool


@pytest.fixture(autouse=True)
def _edge_without_session_platform(monkeypatch):
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


@pytest.fixture
def audio_dir(tmp_path):
    path = tmp_path / "audio"  # tmp_path also holds the conftest's isolated HERMES_HOME
    path.mkdir()
    return path


async def _partial_write_then_fail(_text: str, output_path: str, _cfg: dict) -> str:
    Path(output_path).write_bytes(b"partial")
    raise RuntimeError("provider unreachable")


async def _write_audio(_text: str, output_path: str, _cfg: dict) -> str:
    Path(output_path).write_bytes(b"new-audio")
    return output_path


def test_failed_synthesis_keeps_existing_file_and_cleans_partial(audio_dir, monkeypatch):
    existing = audio_dir / "notes.mp3"
    existing.write_bytes(b"user data")
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _partial_write_then_fail)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(existing)))

    assert result["success"] is False
    assert existing.read_bytes() == b"user data"
    assert sorted(p.name for p in audio_dir.iterdir()) == ["notes.mp3"]


def test_failed_synthesis_to_new_path_leaves_nothing_behind(audio_dir, monkeypatch):
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _partial_write_then_fail)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(audio_dir / "new.mp3")))

    assert result["success"] is False
    assert list(audio_dir.iterdir()) == []


def test_successful_synthesis_still_overwrites_existing_file(audio_dir, monkeypatch):
    existing = audio_dir / "notes.mp3"
    existing.write_bytes(b"user data")
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _write_audio)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(existing)))

    assert result["success"] is True
    assert result["file_path"] == str(existing)
    assert existing.read_bytes() == b"new-audio"
    assert sorted(p.name for p in audio_dir.iterdir()) == ["notes.mp3"]
