"""Tests for terminal workspace-root enforcement."""

import json
from unittest.mock import patch


def test_get_env_config_defaults_local_cwd_to_workspace_root(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(workspace))

    from tools.terminal_tool import _get_env_config

    config = _get_env_config()
    assert config["cwd"] == str(workspace)
    assert config["workspace_root"] == str(workspace)


def test_get_env_config_ignores_workspace_root_for_docker(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(workspace))

    from tools.terminal_tool import _get_env_config

    config = _get_env_config()
    assert config["cwd"] == "/root"
    assert config["workspace_root"] == ""


def test_terminal_tool_blocks_workdir_outside_workspace_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    config = {
        "env_type": "local",
        "cwd": str(workspace),
        "workspace_root": str(workspace),
        "timeout": 60,
    }

    with (
        patch("tools.terminal_tool._get_env_config", return_value=config),
        patch("tools.terminal_tool._start_cleanup_thread") as mock_cleanup,
        patch("tools.terminal_tool._create_environment") as mock_create,
    ):
        from tools.terminal_tool import terminal_tool

        result = json.loads(
            terminal_tool("pwd", task_id="workspace-root-test", workdir=str(outside))
        )

    assert result["status"] == "blocked"
    assert "workspace_root" in result["error"]
    mock_cleanup.assert_not_called()
    mock_create.assert_not_called()
