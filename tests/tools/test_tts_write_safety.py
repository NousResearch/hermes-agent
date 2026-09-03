"""TTS output-path errors preserve the file-safety denial class."""

import json
from pathlib import Path

from tools.tts_tool import _text_to_speech_single, text_to_speech_tool


def _assert_safe_root_error(result: str, safe_root: Path) -> None:
    data = json.loads(result)
    assert data["success"] is False
    assert "outside HERMES_WRITE_SAFE_ROOT" in data["error"]
    assert str(safe_root) in data["error"]
    assert "credential" not in data["error"]


def test_single_tts_reports_safe_root_denial(tmp_path, monkeypatch):
    safe_root = tmp_path / "allowed"
    safe_root.mkdir()
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

    result = _text_to_speech_single(
        "hello",
        output_path=str(tmp_path / "outside.mp3"),
        tts_config_override={"provider": "edge"},
    )

    _assert_safe_root_error(result, safe_root)


def test_chunking_wrapper_reports_safe_root_denial(tmp_path, monkeypatch):
    safe_root = tmp_path / "allowed"
    safe_root.mkdir()
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

    result = text_to_speech_tool(
        "hello",
        output_path=str(tmp_path / "outside.mp3"),
    )

    _assert_safe_root_error(result, safe_root)


def test_single_tts_keeps_credential_denial_distinct(monkeypatch):
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)

    result = _text_to_speech_single(
        "hello",
        output_path=str(Path.home() / ".ssh" / "id_rsa"),
        tts_config_override={"provider": "edge"},
    )

    data = json.loads(result)
    assert data["success"] is False
    assert "protected system/credential file" in data["error"]
    assert "HERMES_WRITE_SAFE_ROOT" not in data["error"]


def test_single_tts_reports_approval_required_path(monkeypatch):
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)

    result = _text_to_speech_single(
        "hello",
        output_path=str(Path.home() / ".ssh" / "config"),
        tts_config_override={"provider": "edge"},
    )

    data = json.loads(result)
    assert data["success"] is False
    assert "requires explicit approval" in data["error"]
    assert "credential" not in data["error"]
