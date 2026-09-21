"""Approval must inspect the path the file tool actually writes, across task cwds."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("tool_name", ["write_file", "patch"])
@pytest.mark.parametrize("relative_path", ["config", "alias"])
def test_task_relative_ssh_write_requires_approval(tmp_path, monkeypatch, tool_name, relative_path):
    from tools import file_tools, terminal_tool
    from tools.registry import registry

    home = tmp_path / "home"
    ssh = home / ".ssh"
    ssh.mkdir(parents=True)
    target = ssh / "config"
    target.write_text("Host original\n")
    (ssh / "alias").symlink_to(target)
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(launch))
    monkeypatch.chdir(launch)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(file_tools, "_file_ops_cache", {})
    terminal_tool.register_task_env_overrides("ssh-edit", {"cwd": str(ssh)})
    args = {"path": relative_path, "content": "Host changed\n"} if tool_name == "write_file" else {
        "path": relative_path, "mode": "replace", "old_string": "original", "new_string": "changed",
    }

    result = json.loads(registry.dispatch(tool_name, args, task_id="ssh-edit"))

    assert target.read_text() == "Host original\n", result
    assert "SSH config" in result.get("error", ""), result
