"""Tests for the SSH remote execution environment backend."""

import json
import os
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.environments.ssh import SSHEnvironment
from tools.environments import ssh as ssh_env

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
        monkeypatch.setattr("tools.environments.base.time.sleep", lambda _: None)

    def test_base_flags(self, monkeypatch):
        # ControlMaster flags are POSIX-only (#73927): assert them only
        # where multiplexing is enabled so the test passes on Windows too.
        monkeypatch.setattr(ssh_env, "_SSH_MULTIPLEX", True)
        env = SSHEnvironment(host="h", user="u")
        cmd = " ".join(env._build_ssh_command())
        for flag in ("ControlMaster=auto", "ControlPersist=300",
                      "BatchMode=yes", "StrictHostKeyChecking=accept-new"):
            assert flag in cmd

    def test_controlmaster_gated_off_on_windows(self, monkeypatch):
        """#73927: Windows OpenSSH has no Unix-domain ControlMaster, so the
        ControlPath/ControlMaster/ControlPersist options must be omitted —
        passing them fails the connection with 'getsockname failed'."""
        monkeypatch.setattr(ssh_env, "_SSH_MULTIPLEX", False)
        env = SSHEnvironment(host="h", user="u")
        cmd = " ".join(env._build_ssh_command())
        assert "ControlMaster" not in cmd
        assert "ControlPath" not in cmd
        assert "ControlPersist" not in cmd
        # Non-multiplex flags must still be present — the backend works,
        # just without connection pooling.
        assert "BatchMode=yes" in cmd
        assert "StrictHostKeyChecking=accept-new" in cmd
        assert env._build_ssh_command()[-1] == "u@h"


    def test_user_host_suffix(self):
        env = SSHEnvironment(host="h", user="u")
        assert env._build_ssh_command()[-1] == "u@h"


class TestControlSocketPath:
    """Regression tests for issue #11840.

    macOS caps Unix domain socket paths at 104 bytes (sun_path). SSH
    appends a 16-byte random suffix to the control socket path when
    operating in ControlMaster mode. An IPv6 host embedded in the
    filename plus the deeply-nested macOS $TMPDIR easily blows past
    the limit, causing every tool call to fail immediately.
    """

    @pytest.fixture(autouse=True)
    def _mock_connection(self, monkeypatch):
        monkeypatch.setattr("tools.environments.ssh.subprocess.run",
                            lambda *a, **k: subprocess.CompletedProcess([], 0))
        monkeypatch.setattr("tools.environments.ssh.subprocess.Popen",
                            lambda *a, **k: MagicMock(stdout=iter([]),
                                                      stderr=iter([]),
                                                      stdin=MagicMock()))
        monkeypatch.setattr("tools.environments.base.time.sleep", lambda _: None)

    # SSH appends ``.XXXXXXXXXXXXXXXX`` (17 bytes) to the ControlPath in
    # ControlMaster mode; the macOS sun_path field is 104 bytes including
    # the NUL terminator, so the usable path length is 103 bytes.
    _SSH_CONTROLMASTER_SUFFIX = 17
    _MAX_SUN_PATH = 103

    def test_fits_under_macos_socket_limit_with_ipv6_host(self, monkeypatch):
        """A realistic macOS $TMPDIR + IPv6 host must still produce a
        control socket path that fits once SSH appends its ControlMaster
        suffix (see issue #11840)."""
        # Simulate the macOS $TMPDIR shape from the issue traceback —
        # 48 bytes, the typical length of ``/var/folders/XX/YYYYYYYYY/T``.
        fake_tmp = "/var/folders/2t/wbkw5yb158jc3zhswgl7tz9c0000gn/T"
        monkeypatch.setattr("tools.environments.ssh.tempfile.gettempdir",
                            lambda: fake_tmp)
        # The simulated path doesn't exist on the test host — skip the
        # real mkdir so __init__ can proceed.
        from pathlib import Path as _Path
        monkeypatch.setattr(_Path, "mkdir", lambda *a, **k: None)

        env = SSHEnvironment(
            host="9373:9b91:4480:558d:708e:e601:24e8:d8d0",
            user="hermes",
            port=22,
        )

        total_len = len(str(env.control_socket)) + self._SSH_CONTROLMASTER_SUFFIX
        assert total_len <= self._MAX_SUN_PATH, (
            f"control socket path would exceed the {self._MAX_SUN_PATH}-byte "
            f"Unix domain socket limit once SSH appends its 16-byte suffix: "
            f"{env.control_socket} (+{self._SSH_CONTROLMASTER_SUFFIX} = {total_len})"
        )

    def test_path_is_deterministic_across_instances(self):
        """Same (user, host, port) must yield the same control socket so
        ControlMaster reuse works across reconnects."""
        first = SSHEnvironment(host="example.com", user="alice", port=2222)
        second = SSHEnvironment(host="example.com", user="alice", port=2222)
        assert first.control_socket == second.control_socket

    def test_path_differs_for_different_targets(self):
        """Different (user, host, port) triples must produce different paths."""
        base = SSHEnvironment(host="h", user="u", port=22).control_socket
        assert SSHEnvironment(host="h", user="u", port=23).control_socket != base
        assert SSHEnvironment(host="h", user="v", port=22).control_socket != base
        assert SSHEnvironment(host="g", user="u", port=22).control_socket != base

    def test_control_dir_mode_is_private(self, tmp_path, monkeypatch):
        """ControlPath directory must be 0o700 so other local users cannot
        plant sockets under /tmp/hermes-ssh (#80284)."""
        monkeypatch.setattr(ssh_env, "_SSH_MULTIPLEX", True)
        monkeypatch.setattr(
            "tools.environments.ssh.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        env = SSHEnvironment(host="example.com", user="alice")
        assert env.control_dir == tmp_path / "hermes-ssh"
        assert env.control_dir.is_dir()
        mode = env.control_dir.stat().st_mode & 0o777
        assert mode == 0o700, f"expected 0o700, got {oct(mode)}"
        assert env._use_multiplex is True


class TestControlPathTrust:
    """#80299: do not trust a leftover ControlPath socket after dir harden."""

    def _unix_socket(self, path):
        import socket

        path.unlink(missing_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        return sock

    def test_prepare_skips_posix_mode_when_not_multiplexing(self, tmp_path):
        control = tmp_path / "hermes-ssh"
        control.mkdir()
        control.chmod(0o777)
        use, shared = ssh_env._prepare_control_dir(control, multiplex=False)
        assert use is False
        assert shared is False
        assert control.is_dir()

    def test_unowned_control_dir_is_rejected(self, tmp_path, monkeypatch):
        control = tmp_path / "hermes-ssh"
        control.mkdir(mode=0o700)
        real_uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
        with pytest.raises(RuntimeError, match="not owned"):
            ssh_env._prepare_control_dir(control, multiplex=True)

    def test_symlink_control_dir_is_rejected(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        control = tmp_path / "hermes-ssh"
        control.symlink_to(real)
        with pytest.raises(RuntimeError, match="symlink"):
            ssh_env._prepare_control_dir(control, multiplex=True)

    def test_chmod_ineffective_disables_multiplex(self, tmp_path, monkeypatch):
        control = tmp_path / "hermes-ssh"
        control.mkdir()
        control.chmod(0o777)
        monkeypatch.setattr(type(control), "chmod", lambda self, mode: None)
        use, shared = ssh_env._prepare_control_dir(control, multiplex=True)
        assert shared is True
        assert use is False

    def test_foreign_socket_is_unlinked(self, tmp_path, monkeypatch):
        sock_path = tmp_path / "x.sock"
        sock = self._unix_socket(sock_path)
        try:
            real_uid = os.getuid()
            monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
            ssh_env._quarantine_untrusted_control_socket(
                sock_path, dir_was_shared=False
            )
            assert not sock_path.exists()
        finally:
            sock.close()

    def test_regular_file_at_socket_path_is_unlinked(self, tmp_path):
        sock_path = tmp_path / "x.sock"
        sock_path.write_text("planted", encoding="utf-8")
        ssh_env._quarantine_untrusted_control_socket(
            sock_path, dir_was_shared=False
        )
        assert not sock_path.exists()

    def test_owned_socket_in_private_dir_is_kept(self, tmp_path):
        sock_path = tmp_path / "x.sock"
        sock = self._unix_socket(sock_path)
        try:
            ssh_env._quarantine_untrusted_control_socket(
                sock_path, dir_was_shared=False
            )
            assert sock_path.exists()
        finally:
            sock.close()
            sock_path.unlink(missing_ok=True)

    def test_owned_socket_from_shared_dir_is_unlinked(self, tmp_path):
        sock_path = tmp_path / "x.sock"
        sock = self._unix_socket(sock_path)
        try:
            ssh_env._quarantine_untrusted_control_socket(
                sock_path, dir_was_shared=True
            )
            assert not sock_path.exists()
        finally:
            sock.close()

    def test_chmod_ineffective_init_omits_controlpath(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ssh_env, "_SSH_MULTIPLEX", True)
        monkeypatch.setattr(
            "tools.environments.ssh.subprocess.run",
            lambda *a, **k: __import__("subprocess").CompletedProcess([], 0),
        )
        monkeypatch.setattr(
            "tools.environments.ssh.subprocess.Popen",
            lambda *a, **k: MagicMock(
                stdout=iter([]), stderr=iter([]), stdin=MagicMock()
            ),
        )
        monkeypatch.setattr("tools.environments.base.time.sleep", lambda _: None)
        monkeypatch.setattr(
            "tools.environments.ssh.tempfile.gettempdir",
            lambda: str(tmp_path),
        )
        control = tmp_path / "hermes-ssh"
        control.mkdir()
        control.chmod(0o777)
        monkeypatch.setattr(type(control), "chmod", lambda self, mode: None)
        env = SSHEnvironment(host="h", user="u")
        assert env._use_multiplex is False
        cmd = " ".join(env._build_ssh_command())
        assert "ControlPath" not in cmd
        assert "ControlMaster" not in cmd


class TestTerminalToolConfig:
    def test_ssh_persistent_default_true(self, monkeypatch):
        """SSH persistent defaults to True (via TERMINAL_PERSISTENT_SHELL)."""
        monkeypatch.delenv("TERMINAL_SSH_PERSISTENT", raising=False)
        monkeypatch.delenv("TERMINAL_PERSISTENT_SHELL", raising=False)
        from tools.terminal_tool import _get_env_config
        assert _get_env_config()["ssh_persistent"] is True


    def test_ssh_persistent_respects_config(self, monkeypatch):
        """TERMINAL_PERSISTENT_SHELL=false disables SSH persistent by default."""
        monkeypatch.delenv("TERMINAL_SSH_PERSISTENT", raising=False)
        monkeypatch.setenv("TERMINAL_PERSISTENT_SHELL", "false")
        from tools.terminal_tool import _get_env_config
        assert _get_env_config()["ssh_persistent"] is False


class TestSSHPreflight:
    def test_ensure_ssh_available_raises_clear_error_when_missing(self, monkeypatch):
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: None)

        with pytest.raises(RuntimeError, match="SSH is not installed or not in PATH"):
            ssh_env._ensure_ssh_available()


    def test_ssh_environment_connects_when_ssh_exists(self, monkeypatch):
        called = {"count": 0}

        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")

        def _fake_establish(self):
            called["count"] += 1

        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", _fake_establish)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/alice")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
        monkeypatch.setattr(ssh_env, "FileSyncManager", lambda **kw: type("M", (), {"sync": lambda self, **k: None})())

        env = ssh_env.SSHEnvironment(host="example.com", user="alice")

        assert called["count"] == 1
        assert env.host == "example.com"
        assert env.user == "alice"


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


    def test_large_output(self):
        r = _run("seq 1 1000")
        assert r["exit_code"] == 0
        lines = r["output"].strip().splitlines()
        assert len(lines) == 1000
        assert lines[0] == "1"
        assert lines[-1] == "1000"
