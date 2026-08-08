"""Foreground PTY support (issue #81756).

``pty=true`` used to be honored only by the background path
(``process_registry.spawn_local(use_pty=...)``); the foreground path called
``env.execute()`` with no PTY flag, so interactive CLIs silently ran on
pipes.  These tests pin the new plumbing:

- ``terminal_tool`` forwards ``pty`` to ``env.execute(use_pty=...)``.
- ``LocalEnvironment._run_bash`` spawns via ``_spawn_pty_process`` when
  ``use_pty`` is set (and falls back to pipes when PTY is unavailable).
- ``_PtyProcessHandle`` adapts a ptyprocess/pywinpty handle to the
  ``ProcessHandle`` protocol so the shared ``_wait_for_process`` machinery
  (drain, interrupt, timeout) works unchanged.
"""

import json
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools.environments.base as base_module
import tools.terminal_tool as terminal_tool_module
from tools.environments.base import _PtyProcessHandle
from tools.environments.local import LocalEnvironment


# =========================================================================
# _PtyProcessHandle
# =========================================================================


class _FakePtyProc:
    """Duck-typed stand-in for ptyprocess.PtyProcess / winpty.PtyProcess."""

    def __init__(self, chunks, exitstatus=0, pid=4242):
        self._chunks = list(chunks)
        self.exitstatus = exitstatus
        self.pid = pid
        self.alive = True

    def isalive(self):
        return self.alive

    def read(self, _n):
        if self._chunks:
            return self._chunks.pop(0)
        self.alive = False
        raise EOFError

    def wait(self):
        self.alive = False
        return self.exitstatus

    def terminate(self, force=False):
        self.alive = False


class TestPtyProcessHandle:
    def test_streams_output_and_exit_code(self):
        pty = _FakePtyProc([b"caf\xc3", b"\xa9\n", b"done\n"], exitstatus=3)
        handle = _PtyProcessHandle(pty)
        out = handle.stdout.read()
        assert out == "café\ndone\n"
        assert handle.wait(timeout=5) == 3
        assert handle.returncode == 3
        assert handle.poll() == 3

    def test_pid_exposed(self):
        handle = _PtyProcessHandle(_FakePtyProc([b"x"], pid=777))
        assert handle.pid == 777

    def test_kill_terminates_pty(self):
        pty = _FakePtyProc([], pid=888)
        handle = _PtyProcessHandle(pty)
        handle.kill()
        assert pty.alive is False

    def test_poll_none_while_running(self):
        """poll() must return None while the PTY child is still alive so the
        shared _wait_for_process poll loop keeps waiting."""

        class _ForeverPty:
            def __init__(self):
                self.alive = True

            def isalive(self):
                return self.alive

            def read(self, _n):
                return b"x" * 100

            def wait(self):
                self.alive = False
                return 0

            def terminate(self, force=False):
                self.alive = False

        pty = _ForeverPty()
        handle = _PtyProcessHandle(pty)
        # Drain the pipe so the reader thread never blocks on a full buffer.
        drainer = threading.Thread(
            target=lambda: handle.stdout.read(), daemon=True
        )
        drainer.start()
        try:
            assert handle.poll() is None
        finally:
            handle.kill()


def test_spawn_pty_process_uses_configured_shell_and_env(monkeypatch):
    captured = {}

    class _FakeCls:
        @classmethod
        def spawn(cls, argv, cwd=None, env=None, dimensions=(24, 80)):
            captured["argv"] = argv
            captured["cwd"] = cwd
            captured["env"] = env
            captured["dimensions"] = dimensions
            return _FakePtyProc([b"hi\n"], pid=999)

    monkeypatch.setattr(base_module.platform, "system", lambda: "Linux")
    fake_module = SimpleNamespace(PtyProcess=_FakeCls)
    with patch.dict("sys.modules", {"ptyprocess": fake_module}):
        handle = base_module._spawn_pty_process(
            "echo hi",
            cwd="/tmp",
            run_env={"PATH": "/usr/bin", "HOME": "/root"},
            shell="/bin/bash",
        )

    assert captured["argv"] == ["/bin/bash", "-c", "set +m; echo hi"]
    assert captured["cwd"] == "/tmp"
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured["env"]["HOME"] == "/root"
    assert isinstance(handle, _PtyProcessHandle)


def test_spawn_pty_process_returns_none_when_library_missing(monkeypatch):
    monkeypatch.setattr(base_module.platform, "system", lambda: "Linux")
    with patch("builtins.__import__", side_effect=ImportError("no ptyprocess")):
        assert (
            base_module._spawn_pty_process(
                "echo hi", cwd="/tmp", run_env={}, shell="/bin/bash"
            )
            is None
        )


# =========================================================================
# LocalEnvironment._run_bash PTY branch
# =========================================================================


def _make_env(tmp_path):
    with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    return env


class _PtyProcessHandleStub:
    """Minimal ProcessHandle-compatible stub with a real stdout pipe."""

    def __init__(self, text="pty-output\n"):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, text.encode())
        os.close(write_fd)
        self._stdout = os.fdopen(read_fd, "r", encoding="utf-8", errors="replace")
        self.returncode = 0

    @property
    def stdout(self):
        return self._stdout

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_local_run_bash_uses_pty_when_requested(tmp_path, monkeypatch):
    """use_pty=True must route through _spawn_pty_process, not Popen."""
    env = _make_env(tmp_path)

    with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
         patch("tools.environments.base._spawn_pty_process",
               return_value=_PtyProcessHandleStub()) as spawn_pty, \
         patch("subprocess.Popen", side_effect=AssertionError("pipe path used for pty")):
        proc = env._run_bash("echo hi", use_pty=True)

    spawn_pty.assert_called_once()
    captured_args, captured_kwargs = spawn_pty.call_args
    assert captured_kwargs["shell"] == "/bin/bash"
    assert "echo hi" in captured_args[0]
    assert proc.poll() == 0


def test_local_run_bash_falls_back_to_pipes_without_pty_library(tmp_path, monkeypatch):
    """PTY library missing → pipe path, never an exception (matches the
    background spawn_local contract)."""
    env = _make_env(tmp_path)
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        proc.stdout = os.fdopen(read_fd, "r", encoding="utf-8", errors="replace")
        return proc

    with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
         patch("tools.environments.base._spawn_pty_process", return_value=None), \
         patch("subprocess.Popen", side_effect=fake_popen):
        proc = env._run_bash("echo hi", use_pty=True)

    assert captured["args"] == ["/bin/bash", "-c", "echo hi"]
    assert proc.poll() == 0


def test_local_run_bash_pty_skipped_when_stdin_piped(tmp_path, monkeypatch):
    """stdin_data and PTY are mutually exclusive — the pipe path must win so
    ``gh auth login --with-token`` (etc.) can still receive piped input."""
    env = _make_env(tmp_path)

    with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
         patch("tools.environments.base._spawn_pty_process") as spawn_pty, \
         patch("subprocess.Popen") as popen:
        env._run_bash("gh auth login --with-token", use_pty=True, stdin_data="tok\n")

    spawn_pty.assert_not_called()
    popen.assert_called_once()


def test_local_execute_pty_end_to_end(tmp_path, monkeypatch):
    """Full execute() with use_pty=True must return captured PTY output
    through the shared _wait_for_process machinery."""
    env = _make_env(tmp_path)

    with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
         patch("tools.environments.base._spawn_pty_process",
               return_value=_PtyProcessHandleStub("hello-from-pty\n")), \
         patch("tools.terminal_tool._interrupt_event", threading.Event()):
        result = env.execute("echo hello", use_pty=True)

    assert result["returncode"] == 0
    assert "hello-from-pty" in result["output"]


# =========================================================================
# terminal_tool foreground routing
# =========================================================================


def _base_config(tmp_path):
    return {
        "env_type": "local",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 30,
    }


def _run_foreground(command, config, monkeypatch, **kwargs):
    """Run terminal_tool() foreground against a mocked env; return (result, env)."""
    import tools.self_repo_guard as self_repo_guard

    monkeypatch.setattr(self_repo_guard, "get_running_source_root", lambda: None)
    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "ok", "returncode": 0}
    mock_env.cwd = config["cwd"]

    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(
            patch("tools.terminal_tool._get_env_config", return_value=config)
        )
        stack.enter_context(patch("tools.terminal_tool._start_cleanup_thread"))
        stack.enter_context(
            patch("tools.terminal_tool._active_environments", {"default": mock_env})
        )
        stack.enter_context(patch("tools.terminal_tool._last_activity", {"default": 0}))
        stack.enter_context(patch("tools.terminal_tool._session_cwd", {}))
        stack.enter_context(
            patch(
                "tools.terminal_tool._check_all_guards",
                return_value={"approved": True},
            )
        )
        result = json.loads(terminal_tool_module.terminal_tool(command=command, **kwargs))
    return result, mock_env


def test_foreground_pty_true_is_forwarded_to_env_execute(monkeypatch, tmp_path):
    """Issue #81756: foreground pty=true must reach env.execute(use_pty=True)."""
    config = _base_config(tmp_path)
    result, mock_env = _run_foreground(
        "python3 -c \"print(input())\"", config, monkeypatch, pty=True
    )

    assert result["exit_code"] == 0
    mock_env.execute.assert_called_once()
    assert mock_env.execute.call_args.kwargs.get("use_pty") is True


def test_foreground_pty_false_routes_to_pipes(monkeypatch, tmp_path):
    config = _base_config(tmp_path)
    _result, mock_env = _run_foreground(
        "echo hello", config, monkeypatch, pty=False
    )

    mock_env.execute.assert_called_once()
    assert mock_env.execute.call_args.kwargs.get("use_pty") is False


def test_foreground_pty_disabled_for_pipe_stdin_command(monkeypatch, tmp_path):
    """gh auth login --with-token needs piped stdin — PTY must be dropped in
    foreground exactly like background, with a pty_note explaining why."""
    config = _base_config(tmp_path)
    result, mock_env = _run_foreground(
        "gh auth login --hostname github.com --git-protocol https --with-token",
        config,
        monkeypatch,
        pty=True,
    )

    assert mock_env.execute.call_args.kwargs.get("use_pty") is False
    assert "pty_note" in result
    assert "PTY disabled" in result["pty_note"]


def test_foreground_pty_true_without_pty_flag_stays_false(monkeypatch, tmp_path):
    """The default (no pty arg) must keep the historical pipe behavior."""
    config = _base_config(tmp_path)
    _result, mock_env = _run_foreground("echo hello", config, monkeypatch)

    mock_env.execute.assert_called_once()
    assert mock_env.execute.call_args.kwargs.get("use_pty") is False
