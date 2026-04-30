import json
from pathlib import Path

from tools import mcporter_tool
from tools.registry import registry


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "mcporter.json"
    cfg.write_text('{"mcpServers":{"web-reader":{"url":"https://example.test/mcp"}}}\n')
    return cfg


def test_mcporter_tools_registered():
    names = set(registry.get_all_tool_names())
    assert {"mcporter_list", "mcporter_schema", "mcporter_call"} <= names
    assert registry.get_toolset_for_tool("mcporter_call") == "mcporter"


def test_check_requirements_false_without_config(monkeypatch):
    monkeypatch.delenv("MCPORTER_CONFIG", raising=False)
    monkeypatch.setattr(mcporter_tool, "_resolve_config_path", lambda value=None: None)
    assert mcporter_tool._check_requirements() is False


def test_mcporter_call_uses_subprocess_argv_no_shell(monkeypatch, tmp_path):
    cfg_path = _write_config(tmp_path)
    calls = []
    monkeypatch.setenv("MCPORTER_CONFIG", str(cfg_path))
    monkeypatch.setattr(mcporter_tool.shutil, "which", lambda cmd: "/usr/bin/npx")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        class Proc:
            returncode = 0
            stdout = '{"ok": true}'
            stderr = ""
        return Proc()

    monkeypatch.setattr(mcporter_tool.subprocess, "run", fake_run)
    raw = mcporter_tool._call_handler({"server": "web-reader", "tool": "webReader", "arguments": {"url": "https://example.com"}})
    parsed = json.loads(raw)
    assert parsed["result"] == {"ok": True}
    command, kwargs = calls[0]
    assert command[:3] == ["npx", "-y", "mcporter"]
    assert "--config" in command
    assert command[command.index("--config") + 1] == str(cfg_path)
    assert command[command.index("call") + 1] == "web-reader.webReader"
    assert json.loads(command[command.index("--args") + 1]) == {"url": "https://example.com"}
    assert "shell" not in kwargs
    assert kwargs["capture_output"] is True


def test_mcporter_call_rejects_non_object_arguments():
    raw = mcporter_tool._call_handler({"server": "s", "tool": "t", "arguments": "bad"})
    assert json.loads(raw)["error"] == "arguments must be an object"


def test_mcporter_error_redacts_authorization(monkeypatch, tmp_path):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setenv("MCPORTER_CONFIG", str(cfg_path))
    monkeypatch.setattr(mcporter_tool.shutil, "which", lambda cmd: "/usr/bin/npx")

    def fake_run(command, **kwargs):
        class Proc:
            returncode = 1
            stdout = ""
            stderr = "Authorization: Bearer secret-token-123 failed"
        return Proc()

    monkeypatch.setattr(mcporter_tool.subprocess, "run", fake_run)
    raw = mcporter_tool._schema_handler({"server": "web-reader"})
    err = json.loads(raw)["error"]
    assert "secret-token-123" not in err
    assert "[REDACTED]" in err


def test_mcporter_list_returns_text_when_not_json(monkeypatch, tmp_path):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setenv("MCPORTER_CONFIG", str(cfg_path))
    monkeypatch.setattr(mcporter_tool.shutil, "which", lambda cmd: "/usr/bin/npx")

    def fake_run(command, **kwargs):
        class Proc:
            returncode = 0
            stdout = "mcporter 0.9.0 — Listing 1 server\n- web-reader (1 tool)\n"
            stderr = ""
        return Proc()

    monkeypatch.setattr(mcporter_tool.subprocess, "run", fake_run)
    raw = mcporter_tool._list_handler({})
    assert "web-reader" in json.loads(raw)["result"]
