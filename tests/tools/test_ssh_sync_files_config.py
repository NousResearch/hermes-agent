"""terminal.ssh_sync_files: off means no ~/.hermes upload to the remote and no sync-back onto the host."""

import os
from unittest.mock import MagicMock, patch

import pytest

from tools import terminal_tool, terminal_tool_backends
from tools.environments import ssh as ssh_env
from tools.environments.ssh import SSHEnvironment


@pytest.fixture
def mocked_ssh(monkeypatch):
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(SSHEnvironment, "_detect_remote_home", lambda self: "/home/testuser")
    monkeypatch.setattr(SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(SSHEnvironment, "stop_all_processes", lambda self: None)


def test_sync_files_false_builds_no_sync_manager(mocked_ssh):
    with patch.object(ssh_env, "FileSyncManager") as manager_cls:
        env = SSHEnvironment(host="h", user="u", sync_files=False)
        assert env._sync_manager is None
        manager_cls.assert_not_called()
        env.cleanup()  # nothing to sync back; must not raise


def test_sync_files_default_builds_sync_manager_and_syncs_back(mocked_ssh):
    with patch.object(ssh_env, "FileSyncManager") as manager_cls:
        env = SSHEnvironment(host="h", user="u")
        manager_cls.assert_called_once()
        manager = manager_cls.return_value
        assert env._sync_manager is manager
        env.cleanup()
        manager.sync_back.assert_called_once()


@pytest.mark.parametrize("raw,expected", [
    (False, False), ("false", False), ("False", False), (0, False), ("0", False),
    (True, True), ("true", True), (1, True), (None, True),
])
def test_build_ssh_env_coerces_sync_files(raw, expected):
    cfg = {"host": "h", "user": "u"}
    if raw is not None:
        cfg["sync_files"] = raw  # None stands for "key missing"
    with patch.object(terminal_tool_backends, "_SSHEnvironment") as env_cls:
        terminal_tool_backends._build_ssh_env(cwd="~", timeout=10, ssh_config=cfg)
    assert env_cls.call_args.kwargs["sync_files"] is expected


@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off"])
def test_env_var_false_disables(monkeypatch, raw):
    monkeypatch.setenv("TERMINAL_SSH_SYNC_FILES", raw)
    assert terminal_tool._get_env_config()["ssh_sync_files"] is False


@pytest.mark.parametrize("raw", ["true", "1"])
def test_env_var_true_enables(monkeypatch, raw):
    monkeypatch.setenv("TERMINAL_SSH_SYNC_FILES", raw)
    assert terminal_tool._get_env_config()["ssh_sync_files"] is True


def test_env_var_unset_defaults_true(monkeypatch):
    monkeypatch.delenv("TERMINAL_SSH_SYNC_FILES", raising=False)
    assert terminal_tool._get_env_config()["ssh_sync_files"] is True


def _reads_false(monkeypatch, value):
    monkeypatch.setenv("TERMINAL_SSH_SYNC_FILES", value)
    return terminal_tool._get_env_config()["ssh_sync_files"] is False


def test_config_set_map_bridges_false(monkeypatch):
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP, _terminal_env_value

    assert TERMINAL_CONFIG_ENV_MAP["ssh_sync_files"] == "TERMINAL_SSH_SYNC_FILES"
    assert _reads_false(monkeypatch, _terminal_env_value(False))


def test_cli_map_bridges_false(monkeypatch):
    import cli

    assert cli._TERMINAL_ENV_MAPPINGS["ssh_sync_files"] == "TERMINAL_SSH_SYNC_FILES"
    assert _reads_false(monkeypatch, str(False))


def test_gateway_bridge_writes_false(monkeypatch):
    from gateway.run import _bridge_terminal_config_to_env

    monkeypatch.delenv("TERMINAL_SSH_SYNC_FILES", raising=False)
    try:
        _bridge_terminal_config_to_env({"ssh_sync_files": False})
        assert "TERMINAL_SSH_SYNC_FILES" in os.environ
        assert terminal_tool._get_env_config()["ssh_sync_files"] is False
    finally:
        os.environ.pop("TERMINAL_SSH_SYNC_FILES", None)


@pytest.mark.parametrize("raw,expected", [("false", False), ("true", True), (None, True)])
def test_env_var_reaches_ssh_environment(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("TERMINAL_SSH_SYNC_FILES", raising=False)
    else:
        monkeypatch.setenv("TERMINAL_SSH_SYNC_FILES", raw)
    cfg = {**terminal_tool._get_env_config(), "ssh_host": "h", "ssh_user": "u"}
    ssh_config = terminal_tool_backends._ssh_config_from_config(cfg)
    with patch.object(terminal_tool_backends, "_SSHEnvironment") as env_cls:
        terminal_tool_backends._build_ssh_env(cwd="~", timeout=10, ssh_config=ssh_config)
    assert env_cls.call_args.kwargs["sync_files"] is expected
