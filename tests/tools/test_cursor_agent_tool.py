import json
import subprocess

import tools.cursor_agent as cursor_agent_tool
from tools.cursor_agent import cursor_agent, check_cursor_agent_requirements
from toolsets import resolve_toolset


class FakeCompletedProcess:
    returncode = 0
    stdout = json.dumps(
        {
            "success": True,
            "agent_id": "agent-test",
            "run_id": "run-test",
            "status": "finished",
            "result": "done",
        }
    )
    stderr = ""


def _mark_bridge_script_present(monkeypatch, tmp_path):
    script = tmp_path / "cursor-agent.mjs"
    script.write_text("// test bridge\n")
    monkeypatch.setattr(cursor_agent_tool, "BRIDGE_SCRIPT", script)


def test_cursor_agent_toolset_resolves():
    assert "cursor_agent" in resolve_toolset("cursor_agent")


def test_cursor_agent_requirements_need_node_and_key(monkeypatch):
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.setattr(cursor_agent_tool.shutil, "which", lambda name: "/usr/bin/node")
    assert check_cursor_agent_requirements() is False

    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    assert check_cursor_agent_requirements() is True

    monkeypatch.setattr(cursor_agent_tool.shutil, "which", lambda name: None)
    assert check_cursor_agent_requirements() is False


def test_cursor_agent_does_not_accept_or_pass_api_key_argument(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setenv("CURSOR_API_KEY", "secret-test-key")
    monkeypatch.setattr(cursor_agent_tool.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(cursor_agent_tool, "_bridge_ready", lambda: True)
    _mark_bridge_script_present(monkeypatch, tmp_path)

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["payload"] = json.loads(kwargs["input"])
        captured["env"] = kwargs["env"]
        return FakeCompletedProcess()

    monkeypatch.setattr(cursor_agent_tool.subprocess, "run", fake_run)

    result = json.loads(cursor_agent(prompt="hello"))

    assert result["success"] is True
    assert "secret-test-key" not in captured["cmd"]
    assert "api_key" not in captured["payload"]
    assert captured["env"]["CURSOR_API_KEY"] == "secret-test-key"


def test_cursor_agent_validates_run_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.setattr(cursor_agent_tool.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(cursor_agent_tool, "_bridge_ready", lambda: True)
    _mark_bridge_script_present(monkeypatch, tmp_path)

    result = json.loads(cursor_agent(prompt=""))
    assert "error" in result
    assert "prompt is required" in result["error"]


def test_cursor_agent_reports_missing_dependencies(monkeypatch, tmp_path):
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.setattr(cursor_agent_tool.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(cursor_agent_tool, "_bridge_ready", lambda: False)
    _mark_bridge_script_present(monkeypatch, tmp_path)

    result = json.loads(cursor_agent(prompt="hello"))
    assert result["success"] is False
    assert result["setup_needed"] is True
    assert "npm install" in result["error"]


def test_cursor_agent_timeout_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.setattr(cursor_agent_tool.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(cursor_agent_tool, "_bridge_ready", lambda: True)
    _mark_bridge_script_present(monkeypatch, tmp_path)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(cursor_agent_tool.subprocess, "run", fake_run)
    result = json.loads(cursor_agent(prompt="hello", timeout_seconds=3))
    assert result["success"] is False
    assert "timed out" in result["error"]
    assert result["timeout_seconds"] == 3
