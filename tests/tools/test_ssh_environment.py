"""Tests for the SSH remote execution environment backend."""

import json
import os
import signal
import subprocess
import time
import uuid
from unittest.mock import MagicMock

import psutil
import pytest

from tools.environments.base import _build_process_tree_kill_command
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


def _wait_until(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _pid_is_live(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


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


class TestSSHRemoteProcessCleanup:
    @pytest.mark.skipif(os.name == "nt", reason="POSIX process-tree semantics")
    def test_tree_cleanup_command_leaves_no_live_orphan(self):
        root = subprocess.Popen(
            [
                "/bin/bash",
                "-c",
                "trap 'wait; exit 0' TERM; sleep 60 & wait",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        descendants = []
        try:
            def _capture_descendants():
                try:
                    descendants[:] = [
                        child.pid
                        for child in psutil.Process(root.pid).children(recursive=True)
                    ]
                except psutil.NoSuchProcess:
                    descendants[:] = []
                return bool(descendants)

            assert _wait_until(_capture_descendants)

            result = subprocess.run(
                [
                    "/bin/bash",
                    "-c",
                    _build_process_tree_kill_command(root_pid=root.pid),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode == 0, result.stderr
            assert _wait_until(
                lambda: root.poll() is not None
                and all(not _pid_is_live(pid) for pid in descendants)
            )
        finally:
            if root.poll() is None:
                try:
                    os.killpg(root.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                root.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def test_tracked_tree_termination_uses_ssh_connection(self, monkeypatch):
        env = object.__new__(SSHEnvironment)
        env._build_ssh_command = MagicMock(return_value=["ssh", "u@h"])
        run = MagicMock(
            return_value=subprocess.CompletedProcess([], 0, "", "")
        )
        monkeypatch.setattr(ssh_env.subprocess, "run", run)

        result = env.terminate_process_tree(82477, timeout=3)

        assert result == {"output": "", "returncode": 0}
        env._build_ssh_command.assert_called_once_with()
        tree_kill_calls = [
            call
            for call in run.call_args_list
            if call.args
            and call.args[0][:2] == ["ssh", "u@h"]
            and "__hermes_root=82477" in call.args[0][-1]
        ]
        assert len(tree_kill_calls) == 1
        tree_kill_call = tree_kill_calls[0]
        remote_command = tree_kill_call.args[0][-1]
        assert "__hermes_root=82477" in remote_command
        assert "ps -eo pid=,ppid=" in remote_command
        assert "kill -TERM" in remote_command
        assert "kill -KILL" in remote_command
        assert tree_kill_call.kwargs["timeout"] == 3


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


    def test_background_registry_kill_kills_remote_child_tree(self):
        from tools.process_registry import process_registry

        marker = f"/tmp/hermes-background-child-{uuid.uuid4().hex}.pid"
        session_id = ""
        try:
            started = _run(
                f"sleep 999 & child=$!; echo $child > {marker}; wait $child",
                background=True,
            )
            session_id = started["session_id"]
            ready = _run(
                f"for attempt in $(seq 1 50); do "
                f"[ -s {marker} ] && exit 0; sleep 0.1; done; exit 1",
                timeout=10,
            )
            assert ready["exit_code"] == 0

            killed = process_registry.kill_process(session_id)
            assert killed["status"] == "killed"

            check = _run(
                f"pid=$(cat {marker}); "
                "state=$(ps -o stat= -p \"$pid\" 2>/dev/null); "
                "case \"$state\" in ''|*Z*) echo dead ;; *) echo alive:$state ;; esac",
                timeout=10,
            )
            assert check["output"].strip() == "dead"
        finally:
            if session_id:
                process_registry.kill_process(session_id)
            _run(
                f"if [ -r {marker} ]; then "
                f"pid=$(cat {marker}); kill -KILL \"$pid\" 2>/dev/null || true; fi; "
                f"rm -f {marker}",
                timeout=10,
            )

    def test_large_output(self):
        r = _run("seq 1 1000")
        assert r["exit_code"] == 0
        lines = r["output"].strip().splitlines()
        assert len(lines) == 1000
        assert lines[0] == "1"
        assert lines[-1] == "1000"
