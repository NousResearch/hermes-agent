import json
import subprocess

import tools.claude_cli_tool as claude_cli_tool


def test_coerce_allowed_tools_from_list():
    assert claude_cli_tool._coerce_allowed_tools(["Read", "Bash(git *)"]) == "Read,Bash(git *)"


def test_claude_cli_run_tool_parses_success(monkeypatch):
    monkeypatch.setattr(claude_cli_tool, "BRIDGE_SCRIPT_PATH", claude_cli_tool.Path("/tmp/bridge.py"))
    monkeypatch.setattr(claude_cli_tool, "_path_exists", lambda _path: True)
    monkeypatch.setattr(claude_cli_tool.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(claude_cli_tool, "WRAPPER_PATH", claude_cli_tool.Path("/tmp/hermes-call-claude"))

    def _fake_run(cmd, capture_output, text, timeout):
        payload = {
            "success": True,
            "result": "OK",
            "session_id": "abc123",
            "command": cmd,
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(claude_cli_tool.subprocess, "run", _fake_run)
    result = json.loads(
        claude_cli_tool.claude_cli_run_tool(
            prompt="Reply with exactly: OK",
            workdir="/tmp",
            allowed_tools=["Read", "Bash(git *)"],
            timeout_seconds=30,
        )
    )
    assert result["success"] is True
    assert result["result"] == "OK"
    assert "--allowed-tools" in result["command"]


def test_claude_cli_run_tool_returns_error_on_non_json(monkeypatch):
    monkeypatch.setattr(claude_cli_tool, "BRIDGE_SCRIPT_PATH", claude_cli_tool.Path("/tmp/bridge.py"))
    monkeypatch.setattr(claude_cli_tool, "_path_exists", lambda _path: True)
    monkeypatch.setattr(claude_cli_tool.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(claude_cli_tool, "WRAPPER_PATH", claude_cli_tool.Path("/tmp/hermes-call-claude"))

    def _fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 1, stdout="not-json", stderr="boom")

    monkeypatch.setattr(claude_cli_tool.subprocess, "run", _fake_run)
    result = json.loads(claude_cli_tool.claude_cli_run_tool(prompt="hi"))
    assert "error" in result
    assert "non-JSON" in result["error"]
