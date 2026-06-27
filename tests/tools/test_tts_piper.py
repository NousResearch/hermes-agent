"""Tests for the Piper local provider in tools/tts_tool.py."""

import json
from unittest.mock import MagicMock


def test_check_piper_available_with_path(monkeypatch):
    from tools import tts_tool as _tt

    monkeypatch.setattr(_tt.shutil, "which", lambda cmd: "/usr/bin/piper" if cmd == "piper" else None)
    assert _tt._check_piper_available({"piper": {"command": "piper"}}) is True


def test_generate_piper_tts_invokes_cli_and_converts(tmp_path, monkeypatch):
    from tools import tts_tool as _tt

    model = tmp_path / "bmo.onnx"
    config = tmp_path / "bmo.onnx.json"
    model.write_bytes(b"fake-model")
    config.write_text("{}")
    calls = []

    def fake_which(cmd):
        return {"piper": "/usr/bin/piper", "ffmpeg": "/usr/bin/ffmpeg"}.get(cmd)

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[0] == "/usr/bin/piper":
            (tmp_path / "out.wav").write_bytes(b"RIFFfake")
        else:
            (tmp_path / "out.mp3").write_bytes(b"fake-mp3")
        return MagicMock(returncode=0)

    monkeypatch.setattr(_tt.shutil, "which", fake_which)
    monkeypatch.setattr(_tt.subprocess, "run", fake_run)

    result = _tt._generate_piper_tts(
        "Hello BMO",
        str(tmp_path / "out.mp3"),
        {"piper": {"model": str(model), "config": str(config), "speaker": 0}},
    )

    assert result == str(tmp_path / "out.mp3")
    assert calls[0][0][:5] == ["/usr/bin/piper", "--model", str(model), "--output_file", str(tmp_path / "out.wav")]
    assert calls[0][1]["input"] == "Hello BMO"
    assert "--config" in calls[0][0]
    assert "--speaker" in calls[0][0]
    assert calls[1][0][0] == "/usr/bin/ffmpeg"
    assert (tmp_path / "out.mp3").exists()


def test_generate_piper_requires_model(tmp_path, monkeypatch):
    from tools import tts_tool as _tt

    monkeypatch.setattr(_tt.shutil, "which", lambda cmd: "/usr/bin/piper" if cmd == "piper" else None)
    try:
        _tt._generate_piper_tts("Hello", str(tmp_path / "out.wav"), {"piper": {}})
    except ValueError as exc:
        assert "tts.piper.model" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_dispatcher_returns_helpful_error_when_piper_missing(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(_tt.shutil, "which", lambda cmd: None)
    (tmp_path / "config.yaml").write_text(
        "tts:\n  provider: piper\n  piper:\n    model: /tmp/bmo.onnx\n"
    )

    result = json.loads(_tt.text_to_speech_tool("Hello"))
    assert result["success"] is False
    assert "piper" in result["error"].lower()
    assert "piper-tts" in result["error"].lower()
