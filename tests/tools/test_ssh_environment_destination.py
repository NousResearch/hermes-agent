"""SSHEnvironment argv safety for OpenSSH destination option smuggling."""

import subprocess
from pathlib import Path

import pytest

from tools.environments.ssh import SSHEnvironment
from tools.environments.ssh_destination import format_ssh_destination


def test_ssh_environment_rejects_leading_dash_host_before_connect(monkeypatch):
    monkeypatch.setattr("tools.environments.ssh._ensure_ssh_available", lambda: None)
    called = {"connect": False}

    def _boom(self):
        called["connect"] = True
        raise AssertionError("must not connect with poisoned destination")

    monkeypatch.setattr(SSHEnvironment, "_establish_connection", _boom)
    with pytest.raises(ValueError, match="must not start with"):
        SSHEnvironment(host="-oProxyCommand=evil", user="alice")
    assert called["connect"] is False


def test_build_ssh_command_places_double_dash_before_destination():
    env = object.__new__(SSHEnvironment)
    env.port = 22
    env.key_path = ""
    env.control_socket = Path("/tmp/hermes-ssh-test.sock")
    env._destination = format_ssh_destination("example.com", "alice")
    cmd = SSHEnvironment._build_ssh_command(env)
    dash = cmd.index("--")
    assert cmd[dash + 1] == "alice@example.com"
    assert all(not arg.startswith("-oProxy") for arg in cmd)


def test_scp_upload_places_double_dash_before_paths(monkeypatch):
    env = object.__new__(SSHEnvironment)
    env.port = 22
    env.key_path = ""
    env.control_socket = Path("/tmp/hermes-ssh-test.sock")
    env._destination = format_ssh_destination("example.com", "alice")
    monkeypatch.setattr(
        SSHEnvironment,
        "_build_ssh_command",
        lambda self, extra_args=None: ["ssh", "--", "alice@example.com"],
    )
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.setdefault("cmds", []).append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("tools.environments.ssh.subprocess.run", fake_run)
    SSHEnvironment._scp_upload(env, "/local/file", "/home/alice/remote")
    scp_cmds = [c for c in captured["cmds"] if c and c[0] == "scp"]
    assert scp_cmds, "expected an scp subprocess"
    cmd = scp_cmds[-1]
    dash = cmd.index("--")
    assert cmd[dash + 1] == "/local/file"
    assert cmd[dash + 2] == "alice@example.com:/home/alice/remote"
