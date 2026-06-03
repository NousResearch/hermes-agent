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


class TestTerminalToolConfig:
    def test_ssh_persistent_default_true(self, monkeypatch):
        """SSH persistent defaults to True (via TERMINAL_PERSISTENT_SHELL)."""
        monkeypatch.delenv("TERMINAL_SSH_PERSISTENT", raising=False)
        monkeypatch.delenv("TERMINAL_PERSISTENT_SHELL", raising=False)
        from tools.terminal_tool import _get_env_config
        assert _get_env_config()["ssh_persistent"] is True

    def test_ssh_persistent_explicit_false(self, monkeypatch):
        """Per-backend env var overrides the global default."""
        monkeypatch.setenv("TERMINAL_SSH_PERSISTENT", "false")
        from tools.terminal_tool import _get_env_config
        assert _get_env_config()["ssh_persistent"] is False

    def test_ssh_persistent_explicit_true(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_SSH_PERSISTENT", "true")
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

    def test_ssh_environment_checks_availability_before_connect(self, monkeypatch):
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: None)
        monkeypatch.setattr(
            ssh_env.SSHEnvironment,
            "_establish_connection",
            lambda self: pytest.fail("_establish_connection should not run when ssh is missing"),
        )

        with pytest.raises(RuntimeError, match="openssh-client"):
            ssh_env.SSHEnvironment(host="example.com", user="alice")

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


class TestNonUTF8SubprocessOutput:
    """Regression tests for issue #37130.

    Windows OpenSSH targets running Git Bash/MSYS shells can emit non-UTF8
    bytes on stdout/stderr. The backend decoded subprocess output in strict
    text mode (``text=True`` without ``errors=``), so a single undecodable
    byte raised ``UnicodeDecodeError`` during ``_ensure_remote_dirs()`` —
    before any user command ran — bricking all terminal/file/search tools
    even though a manual SSH connection to the same host worked fine.
    """

    _BAD_BYTES = b"setup output \xff\xfe done\n"

    @staticmethod
    def _decoding_run(stdout=b"", stderr=b"", returncode=0):
        """A ``subprocess.run`` stand-in that reproduces real text-mode decoding.

        On the unfixed backend (no ``errors="replace"``) this raises
        ``UnicodeDecodeError`` exactly as the real ``subprocess`` would; once
        the backend passes ``errors="replace"`` it decodes with U+FFFD.
        """
        def _run(cmd, *a, **kwargs):
            out, err = stdout, stderr
            if kwargs.get("text") or kwargs.get("universal_newlines"):
                enc = kwargs.get("encoding") or "utf-8"
                errs = kwargs.get("errors", "strict")
                out = stdout.decode(enc, errors=errs)
                err = stderr.decode(enc, errors=errs)
            return subprocess.CompletedProcess(cmd, returncode, out, err)
        return _run

    @pytest.fixture
    def _ssh_env_factory(self, monkeypatch):
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _n: "/usr/bin/ssh")
        monkeypatch.setattr("tools.environments.base.time.sleep", lambda _s: None)
        from pathlib import Path as _Path
        monkeypatch.setattr(_Path, "mkdir", lambda *a, **k: None)
        monkeypatch.setattr(ssh_env, "FileSyncManager", lambda **kw: MagicMock())
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session",
                            lambda self: None)

        def _factory(stdout=b"", stderr=b"", returncode=0):
            monkeypatch.setattr(
                "tools.environments.ssh.subprocess.run",
                self._decoding_run(stdout, stderr, returncode),
            )
            return ssh_env.SSHEnvironment(host="winhost", user="hermes")
        return _factory

    def test_init_survives_non_utf8_output(self, _ssh_env_factory):
        """Full __init__ (connect + detect home + _ensure_remote_dirs) must
        not raise on non-UTF8 subprocess output — the issue #37130 crash."""
        env = _ssh_env_factory(stdout=self._BAD_BYTES)
        assert env.host == "winhost"

    def test_init_survives_non_utf8_on_stderr(self, _ssh_env_factory):
        env = _ssh_env_factory(stderr=self._BAD_BYTES, returncode=0)
        assert env.user == "hermes"

    def test_detect_home_replaces_bad_bytes(self, _ssh_env_factory):
        env = _ssh_env_factory(stdout=b"/home/hermes\xff\n")
        assert env._remote_home == "/home/hermes�"

    def test_ensure_remote_dirs_does_not_raise(self, _ssh_env_factory):
        """Directly re-exercise the method named in the issue traceback."""
        env = _ssh_env_factory(stdout=self._BAD_BYTES)
        env._ensure_remote_dirs()

    def test_every_text_run_call_uses_errors_replace(self, monkeypatch, tmp_path):
        """All text-mode ``subprocess.run`` calls in the backend — including
        the file-sync sites that __init__ does not reach — must pass
        ``errors="replace"`` so non-UTF8 output can never crash them."""
        calls = []

        def _record(cmd, *a, **kwargs):
            calls.append(kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        def _fake_popen(*a, **k):
            m = MagicMock()
            m.returncode = 0
            m.poll.return_value = 0
            m.communicate.return_value = (b"", b"")
            m.stderr.read.return_value = b""
            return m

        monkeypatch.setattr(ssh_env.shutil, "which", lambda _n: "/usr/bin/ssh")
        monkeypatch.setattr("tools.environments.base.time.sleep", lambda _s: None)
        from pathlib import Path as _Path
        monkeypatch.setattr(_Path, "mkdir", lambda *a, **k: None)
        monkeypatch.setattr(ssh_env, "FileSyncManager", lambda **kw: MagicMock())
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session",
                            lambda self: None)
        monkeypatch.setattr("tools.environments.ssh.subprocess.run", _record)
        monkeypatch.setattr("tools.environments.ssh.subprocess.Popen", _fake_popen)

        env = ssh_env.SSHEnvironment(host="winhost", user="hermes")

        src = tmp_path / "payload.txt"
        src.write_text("x")
        env._ssh_delete(["~/.hermes/a"])
        env._scp_upload(str(src), "~/.hermes/b")
        env._ssh_bulk_upload([(str(src), f"{env._remote_home}/.hermes/sub/c")])

        text_calls = [k for k in calls if k.get("text")]
        assert text_calls, "expected text-mode subprocess.run calls"
        for kwargs in text_calls:
            assert kwargs.get("errors") == "replace", (
                f"text-mode subprocess.run missing errors='replace': {kwargs}"
            )
            assert kwargs.get("encoding") == "utf-8", (
                f"text-mode subprocess.run missing encoding='utf-8': {kwargs}"
            )


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
