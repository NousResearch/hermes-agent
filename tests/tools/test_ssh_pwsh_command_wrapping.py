"""Tests for PowerShell command wrapping in ssh_pwsh backend."""

import base64
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.environments import ssh as ssh_env
from tools.environments import ssh_pwsh as ssh_pwsh_env
from tools.environments.ssh_pwsh import SSHPwshEnvironment


def _mock_completed(stdout=b"", stderr=b"", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(ssh_env.subprocess, "run",
                        lambda *a, **k: _mock_completed(stdout=b"ok\r\n"))
    monkeypatch.setattr(ssh_env.subprocess, "Popen",
                        lambda *a, **k: MagicMock(stdout=iter([]), stderr=iter([]),
                                                  stdin=MagicMock(), returncode=0,
                                                  poll=lambda: 0,
                                                  communicate=lambda **kw: (b"", b"")))
    monkeypatch.setattr(ssh_env.BaseEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(ssh_env, "FileSyncManager",
                        lambda **kw: type("M", (), {
                            "sync": lambda self, **k: None,
                            "sync_back": lambda self, **k: None,
                        })())
    e = SSHPwshEnvironment(host="h", user="u")
    e._remote_home = "C:\\Users\\test"
    return e


class TestWrapCommand:

    def test_contains_set_location(self, env):
        wrapped = env._wrap_command("dir", "C:\\Users\\test")
        assert "Set-Location" in wrapped
        assert "C:\\Users\\test" in wrapped

    def test_contains_invoke_expression(self, env):
        wrapped = env._wrap_command("Get-ChildItem", "C:\\")
        assert "Invoke-Expression" in wrapped
        assert "Get-ChildItem" in wrapped

    def test_captures_exit_code(self, env):
        wrapped = env._wrap_command("dir", "C:\\")
        assert "$LASTEXITCODE" in wrapped

    def test_emits_cwd_marker(self, env):
        wrapped = env._wrap_command("dir", "C:\\")
        assert env._cwd_marker in wrapped

    def test_sources_snapshot_when_ready(self, env):
        env._snapshot_ready = True
        wrapped = env._wrap_command("dir", "C:\\")
        assert ". " in wrapped
        assert "hermes-snap-" in wrapped

    def test_no_source_when_snapshot_not_ready(self, env):
        env._snapshot_ready = False
        wrapped = env._wrap_command("dir", "C:\\")
        assert "hermes-snap-" not in wrapped

    def test_escapes_single_quotes(self, env):
        wrapped = env._wrap_command("echo 'hello'", "C:\\")
        assert "''hello''" in wrapped

    def test_exits_with_captured_code(self, env):
        wrapped = env._wrap_command("dir", "C:\\")
        assert "exit $script:__hermes_ec" in wrapped


class TestRunBash:

    def test_uses_encoded_command(self, env, monkeypatch):
        captured = []
        monkeypatch.setattr(ssh_pwsh_env, "_popen_bash",
                            lambda cmd, stdin_data=None: (
                                captured.append(cmd) or MagicMock()
                            ))
        env._run_bash("echo hello")
        assert len(captured) == 1
        cmd = captured[0]
        assert "pwsh" in cmd
        assert "-NoProfile" in cmd
        assert "-EncodedCommand" in cmd
        encoded_idx = cmd.index("-EncodedCommand") + 1
        encoded = cmd[encoded_idx]
        decoded = base64.b64decode(encoded).decode("utf-16-le")
        assert "echo hello" in decoded
