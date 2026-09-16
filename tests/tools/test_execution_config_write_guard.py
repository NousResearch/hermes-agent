"""Execution tools must not persist changes to the active security config."""

import json
import shlex
import sys

import hermes_cli.config as hermes_config
from tools.code_execution_tool import execute_code


def test_execute_code_restores_active_config_after_direct_write(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    original = b"approvals:\n  mode: smart\n"
    config_path.write_bytes(original)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    hermes_config._LOAD_CONFIG_CACHE.clear()

    result = json.loads(execute_code(
        code=f"open({str(config_path)!r}, 'w', encoding='utf-8').write('approvals:\\n  mode: off\\n')",
        task_id="config-write-guard",
        reset=True,
    ))

    assert result["status"] == "error"
    assert "modified the active Hermes config.yaml" in result["error"]
    assert config_path.read_bytes() == original


def test_terminal_restores_active_config_after_hidden_write(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    original = b"hooks:\n  pre_tool_call: []\n"
    config_path.write_bytes(original)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    hermes_config._LOAD_CONFIG_CACHE.clear()

    from tools.terminal_tool import terminal_tool

    script = (
        "import os; "
        "p=os.path.join(os.environ['HERMES_HOME'], 'config.yaml'); "
        "open(p, 'w', encoding='utf-8').write('hooks: {}\\n')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    result = json.loads(terminal_tool(command, task_id="terminal-config-write-guard"))

    assert result["exit_code"] == 126
    assert "modified the active Hermes config.yaml" in result["error"]
    assert config_path.read_bytes() == original
