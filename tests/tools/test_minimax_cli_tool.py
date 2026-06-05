import json
from pathlib import Path

import tools.minimax_cli_tool as mmx


class Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_minimax_cli_status_reports_quota_and_auth(monkeypatch, tmp_path):
    monkeypatch.setattr(mmx.shutil, "which", lambda name: "/usr/bin/mmx" if name == "mmx" else None)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["mmx", "auth", "status"]:
            return Completed('{"method":"api-key"}')
        if cmd == ["mmx", "quota"]:
            return Completed('{"category_remains":[{"category":"image_generation","current_interval_total_count":200,"current_interval_usage_count":1}]}')
        raise AssertionError(cmd)

    monkeypatch.setattr(mmx.subprocess, "run", fake_run)

    result = json.loads(mmx.minimax_cli_status())

    assert result["ok"] is True
    assert result["mmx_path"] == "/usr/bin/mmx"
    assert result["auth"]["method"] == "api-key"
    assert result["quota"]["category_remains"][0]["category"] == "image_generation"
    assert calls == [["mmx", "auth", "status"], ["mmx", "quota"]]


def test_minimax_image_generate_saves_to_managed_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(mmx, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(mmx.shutil, "which", lambda name: "/usr/bin/mmx" if name == "mmx" else None)

    def fake_run(cmd, **kwargs):
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_bytes(b"fake-image-bytes")
        return Completed('{"ok":true}')

    monkeypatch.setattr(mmx.subprocess, "run", fake_run)

    result = json.loads(mmx.minimax_image_generate("Emerald icon", aspect_ratio="square"))

    assert result["ok"] is True
    assert result["provider"] == "minimax-cli"
    assert result["model"] == "image-01"
    assert result["path"].startswith(str(tmp_path))
    assert Path(result["path"]).read_bytes() == b"fake-image-bytes"


def test_minimax_text_chat_extracts_text_and_hides_thinking(monkeypatch):
    monkeypatch.setattr(mmx.shutil, "which", lambda name: "/usr/bin/mmx" if name == "mmx" else None)

    payload = {
        "model": "MiniMax-M2.7-highspeed",
        "content": [
            {"type": "thinking", "thinking": "private chain"},
            {"type": "text", "text": "Visible answer"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    monkeypatch.setattr(
        mmx.subprocess,
        "run",
        lambda cmd, **kwargs: Completed(json.dumps(payload)),
    )

    result = json.loads(mmx.minimax_text_chat("hello", max_tokens=300))

    assert result["ok"] is True
    assert result["text"] == "Visible answer"
    assert "private chain" not in json.dumps(result)
    assert result["usage"] == {"input_tokens": 1, "output_tokens": 2}
