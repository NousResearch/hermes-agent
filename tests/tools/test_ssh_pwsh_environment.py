"""Tests for the SSH PowerShell remote execution environment backend."""

import base64
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.environments import ssh as ssh_env


def _mock_completed(stdout=b"", stderr=b"", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def _decode_pwsh_cmd(cmd_list):
    """Decode EncodedCommand from SSH command list to get the PowerShell script."""
    try:
        # Find -EncodedCommand in the command list
        for i, arg in enumerate(cmd_list):
            if arg == "-EncodedCommand" and i + 1 < len(cmd_list):
                encoded = cmd_list[i + 1]
                # Decode base64 UTF-16LE
                return base64.b64decode(encoded).decode("utf-16-le")
    except Exception:
        pass
    return ""


@pytest.fixture
def mock_ssh_deps(monkeypatch):
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")

    call_log = []

    def fake_run(cmd, **kwargs):
        call_log.append(cmd)
        # Decode EncodedCommand to check the actual PowerShell script
        pwsh_script = _decode_pwsh_cmd(cmd)
        cmd_str = " ".join(str(c) for c in cmd) + " " + pwsh_script
        
        if "Write-Output 'ok'" in pwsh_script:
            return _mock_completed(stdout=b"ok\r\n")
        if "USERPROFILE" in pwsh_script:
            return _mock_completed(stdout=b"C:\\Users\\test\r\n")
        if "env:TEMP" in pwsh_script:
            return _mock_completed(stdout=b"C:\\Users\\test\\AppData\\Local\\Temp\r\n")
        if "New-Item" in pwsh_script and "Directory" in pwsh_script:
            return _mock_completed()
        return _mock_completed()

    monkeypatch.setattr(ssh_env.subprocess, "run", fake_run)
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

    from tools.environments.ssh_pwsh import SSHPwshEnvironment
    yield call_log


class TestShellDetection:

    def test_prefers_pwsh_over_powershell(self, mock_ssh_deps):
        from tools.environments.ssh_pwsh import SSHPwshEnvironment
        env = SSHPwshEnvironment(host="h", user="u")
        assert env._pwsh_cmd == "pwsh"

    def test_falls_back_to_powershell(self, mock_ssh_deps):
        from tools.environments.ssh_pwsh import SSHPwshEnvironment

        original_run = ssh_env.subprocess.run

        def run_no_pwsh(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "pwsh" in cmd_str:
                return _mock_completed(stderr=b"pwsh: command not found", returncode=127)
            return original_run(cmd, **kwargs)

        ssh_env.subprocess.run = run_no_pwsh
        try:
            env = SSHPwshEnvironment(host="h", user="u")
            assert env._pwsh_cmd == "powershell"
        finally:
            ssh_env.subprocess.run = original_run

    def test_raises_when_no_shell_found(self, mock_ssh_deps):
        from tools.environments.ssh_pwsh import SSHPwshEnvironment

        def run_fail(cmd, **kwargs):
            return _mock_completed(stderr=b"not found", returncode=127)

        ssh_env.subprocess.run = run_fail
        with pytest.raises(RuntimeError, match="pwsh/PowerShell not found"):
            SSHPwshEnvironment(host="h", user="u")


class TestRemoteHomeDetection:

    def test_detects_windows_home(self, mock_ssh_deps):
        from tools.environments.ssh_pwsh import SSHPwshEnvironment
        env = SSHPwshEnvironment(host="h", user="u")
        assert env._remote_home == "C:\\Users\\test"


class TestRemoteTempDetection:

    def test_detects_windows_temp(self, mock_ssh_deps):
        from tools.environments.ssh_pwsh import SSHPwshEnvironment
        env = SSHPwshEnvironment(host="h", user="u")
        assert "Users" in env._remote_temp or "Temp" in env._remote_temp

    def test_get_temp_dir_returns_remote_temp(self, mock_ssh_deps):
        from tools.environments.ssh_pwsh import SSHPwshEnvironment
        env = SSHPwshEnvironment(host="h", user="u")
        temp_dir = env.get_temp_dir()
        assert "$env:TEMP" not in temp_dir


class TestBuildSSHCommand:

    def test_inherits_ssh_flags(self, mock_ssh_deps):
        from tools.environments.ssh_pwsh import SSHPwshEnvironment
        env = SSHPwshEnvironment(host="h", user="u")
        cmd = " ".join(env._build_ssh_command())
        for flag in ("ControlMaster=auto", "ControlPersist=300",
                      "BatchMode=yes", "StrictHostKeyChecking=accept-new"):
            assert flag in cmd


class TestFactoryRegistration:

    def test_create_ssh_pwsh_environment(self, mock_ssh_deps):
        from tools.terminal_tool import _create_environment
        ssh_config = {"host": "h", "user": "u", "port": 22}
        env = _create_environment(
            "ssh_pwsh", image="", cwd="~", timeout=60,
            ssh_config=ssh_config,
        )
        from tools.environments.ssh_pwsh import SSHPwshEnvironment
        assert isinstance(env, SSHPwshEnvironment)

    def test_ssh_pwsh_requires_ssh_config(self):
        from tools.terminal_tool import _create_environment
        with pytest.raises(ValueError, match="ssh_host"):
            _create_environment(
                "ssh_pwsh", image="", cwd="~", timeout=60,
                ssh_config=None,
            )
