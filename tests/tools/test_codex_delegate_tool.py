import json
import subprocess
from pathlib import Path


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_delegate_to_codex_uses_non_pty_exec_by_default(monkeypatch, tmp_path):
    from tools import codex_delegate_tool as tool

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1:3] == ["exec", "--help"]:
            return _Completed(stdout="Run Codex non-interactively")
        return _Completed(returncode=0, stdout='{"event":"done"}\n', stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    monkeypatch.setenv("HOME", str(tmp_path))

    result = json.loads(tool.delegate_to_codex_tool("write a noop", str(tmp_path)))

    assert result["ok"] is True
    assert result["execution_mode"] == "non_pty"
    assert calls[-1][0][0].endswith("codex")
    assert calls[-1][0][1:6] == ["exec", "--json", "--cd", str(tmp_path), "--skip-git-repo-check"]
    assert calls[-1][0][-1] == "write a noop"
    assert calls[-1][1]["stdin"] is None
    assert calls[-1][1]["timeout"] == tool.DEFAULT_TIMEOUT_SECONDS


def test_delegate_to_codex_timeout_returns_structured_error(monkeypatch, tmp_path):
    from tools import codex_delegate_tool as tool

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=3, output="partial", stderr="slow")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    result = json.loads(tool.delegate_to_codex_tool("slow task", str(tmp_path), timeout=3))

    assert result["ok"] is False
    assert result["error"] == "timeout"
    assert result["timeout_seconds"] == 3
    assert "partial" in result["stdout"]
    assert "slow" in result["stderr"]


def test_delegate_to_codex_rejects_missing_workdir(tmp_path):
    from tools import codex_delegate_tool as tool

    missing = tmp_path / "missing"
    result = json.loads(tool.delegate_to_codex_tool("task", str(missing)))

    assert result["ok"] is False
    assert result["error"] == "invalid_working_directory"
