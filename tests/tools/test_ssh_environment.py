"""Tests for the SSH remote execution environment backend."""

import json
import os
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.environments.ssh import SSHEnvironment

_SSH_HOST = os.getenv("TERMINAL_SSH_HOST", "")
_SSH_USER = os.getenv("TERMINAL_SSH_USER", "")
_SSH_PORT = int(os.getenv("TERMINAL_SSH_PORT", "22"))
_SSH_KEY = os.getenv("TERMINAL_SSH_KEY", "")

_has_ssh = bool(_SSH_HOST and _SSH_USER)

requires_ssh = pytest.mark.skipif(
    not _has_ssh,
    reason="TERMINAL_SSH_HOST / TERMINAL_SSH_USER not set",
)


def _run(command, task_id="ssh_test", **kwargs):
    from tools.terminal_tool import terminal_tool
    return json.loads(terminal_tool(command, task_id=task_id, **kwargs))


def _cleanup(task_id="ssh_test"):
    from tools.terminal_tool import cleanup_vm
    cleanup_vm(task_id)


class TestBuildSSHCommand:

    @pytest.fixture(autouse=True)
    def _mock_connection(self, monkeypatch):
        monkeypatch.setattr("tools.environments.ssh.subprocess.run",
                            lambda *a, **k: subprocess.CompletedProcess([], 0))
        monkeypatch.setattr("tools.environments.ssh.subprocess.Popen",
                            lambda *a, **k: MagicMock(stdout=iter([]),
                                                      stderr=iter([]),
                                                      stdin=MagicMock()))
        monkeypatch.setattr("tools.environments.ssh.time.sleep", lambda _: None)

    def test_base_flags(self):
        env = SSHEnvironment(host="h", user="u")
        cmd = " ".join(env._build_ssh_command())
        for flag in ("ControlMaster=auto", "ControlPersist=300",
                      "BatchMode=yes", "StrictHostKeyChecking=accept-new"):
            assert flag in cmd

    def test_custom_port(self):
        env = SSHEnvironment(host="h", user="u", port=2222)
        cmd = env._build_ssh_command()
        assert "-p" in cmd and "2222" in cmd

    def test_key_path(self):
        env = SSHEnvironment(host="h", user="u", key_path="/k")
        cmd = env._build_ssh_command()
        assert "-i" in cmd and "/k" in cmd

    def test_user_host_suffix(self):
        env = SSHEnvironment(host="h", user="u")
        assert env._build_ssh_command()[-1] == "u@h"


class TestTerminalToolConfig:
    def test_ssh_persistent_default_false(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_SSH_PERSISTENT", raising=False)
        from tools.terminal_tool import _get_env_config
        assert _get_env_config()["ssh_persistent"] is False

    def test_ssh_persistent_true(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_SSH_PERSISTENT", "true")
        from tools.terminal_tool import _get_env_config
        assert _get_env_config()["ssh_persistent"] is True


def _setup_ssh_env(monkeypatch, persistent: bool):
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", _SSH_HOST)
    monkeypatch.setenv("TERMINAL_SSH_USER", _SSH_USER)
    monkeypatch.setenv("TERMINAL_SSH_PERSISTENT", "true" if persistent else "false")
    if _SSH_PORT != 22:
        monkeypatch.setenv("TERMINAL_SSH_PORT", str(_SSH_PORT))
    if _SSH_KEY:
        monkeypatch.setenv("TERMINAL_SSH_KEY", _SSH_KEY)


@requires_ssh
class TestOneShotSSH:

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        _setup_ssh_env(monkeypatch, persistent=False)
        yield
        _cleanup()

    def test_echo(self):
        r = _run("echo hello")
        assert r["exit_code"] == 0
        assert "hello" in r["output"]

    def test_exit_code(self):
        r = _run("exit 42")
        assert r["exit_code"] == 42

    def test_state_does_not_persist(self):
        _run("export HERMES_ONESHOT_TEST=yes")
        r = _run("echo $HERMES_ONESHOT_TEST")
        assert r["output"].strip() == ""


@requires_ssh
class TestPersistentSSH:

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        _setup_ssh_env(monkeypatch, persistent=True)
        yield
        _cleanup()

    def test_echo(self):
        r = _run("echo hello-persistent")
        assert r["exit_code"] == 0
        assert "hello-persistent" in r["output"]

    def test_env_var_persists(self):
        _run("export HERMES_PERSIST_TEST=works")
        r = _run("echo $HERMES_PERSIST_TEST")
        assert r["output"].strip() == "works"

    def test_cwd_persists(self):
        _run("cd /tmp")
        r = _run("pwd")
        assert r["output"].strip() == "/tmp"

    def test_exit_code(self):
        r = _run("(exit 42)")
        assert r["exit_code"] == 42

    def test_stderr(self):
        r = _run("echo oops >&2")
        assert r["exit_code"] == 0
        assert "oops" in r["output"]

    def test_multiline_output(self):
        r = _run("echo a; echo b; echo c")
        lines = r["output"].strip().splitlines()
        assert lines == ["a", "b", "c"]

    def test_timeout_then_recovery(self):
        r = _run("sleep 999", timeout=2)
        assert r["exit_code"] == 124
        r = _run("echo alive")
        assert r["exit_code"] == 0
        assert "alive" in r["output"]

    def test_large_output(self):
        r = _run("seq 1 1000")
        assert r["exit_code"] == 0
        lines = r["output"].strip().splitlines()
        assert len(lines) == 1000
        assert lines[0] == "1"
        assert lines[-1] == "1000"
