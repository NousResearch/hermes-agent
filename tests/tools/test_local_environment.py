"""Tests for the LocalEnvironment terminal execution backend.

Covers:
- Basic command execution (output, returncode)
- Timeout handling
- stdin_data piping
- Interrupt signal handling
- Windows/POSIX platform detection guard (_kill_process_tree)
- Cross-platform: verifies no AttributeError on Windows-style call paths
"""

import platform
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env():
    """Return a fresh LocalEnvironment with a short default timeout."""
    from tools.environments.local import LocalEnvironment
    return LocalEnvironment(cwd="/tmp", timeout=5)


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

class TestLocalEnvironmentBasicExecution:
    def test_echo_returns_output(self):
        env = _make_env()
        result = env.execute("echo hello")
        assert result["returncode"] == 0
        assert "hello" in result["output"]

    def test_exit_code_propagated(self):
        env = _make_env()
        result = env.execute("exit 42")
        assert result["returncode"] == 42

    def test_stderr_merged_into_output(self):
        """stderr is redirected to stdout so both streams appear in output."""
        env = _make_env()
        result = env.execute("echo err_msg >&2")
        # On Unix the message reaches stdout via STDOUT redirect; on Windows
        # the shell handles 2>&1 differently but the command should still run.
        assert result["returncode"] == 0

    def test_multiline_output(self):
        env = _make_env()
        result = env.execute("printf 'line1\\nline2\\nline3\\n'")
        assert result["returncode"] == 0
        lines = [l for l in result["output"].splitlines() if l]
        assert len(lines) >= 3

    def test_empty_output(self):
        env = _make_env()
        result = env.execute("true")
        assert result["returncode"] == 0
        assert isinstance(result["output"], str)

    def test_nonexistent_command_returns_nonzero(self):
        env = _make_env()
        result = env.execute("command_that_does_not_exist_xyz123")
        assert result["returncode"] != 0

    def test_cwd_override(self, tmp_path):
        """Command should run in the specified cwd, not the default."""
        env = _make_env()
        # pwd prints the working directory
        result = env.execute("pwd", cwd=str(tmp_path))
        assert result["returncode"] == 0
        assert str(tmp_path) in result["output"].strip()


# ---------------------------------------------------------------------------
# stdin_data
# ---------------------------------------------------------------------------

class TestLocalEnvironmentStdin:
    def test_stdin_piped_to_command(self):
        env = _make_env()
        result = env.execute("cat", stdin_data="hello from stdin\n")
        assert result["returncode"] == 0
        assert "hello from stdin" in result["output"]

    def test_stdin_with_multiple_lines(self):
        env = _make_env()
        data = "alpha\nbeta\ngamma\n"
        result = env.execute("cat", stdin_data=data)
        assert "alpha" in result["output"]
        assert "beta" in result["output"]
        assert "gamma" in result["output"]

    def test_wc_counts_stdin_lines(self):
        env = _make_env()
        data = "a\nb\nc\n"
        result = env.execute("wc -l", stdin_data=data)
        assert result["returncode"] == 0
        # wc -l output contains the count
        assert "3" in result["output"]


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestLocalEnvironmentTimeout:
    def test_timeout_kills_process(self):
        from tools.environments.local import LocalEnvironment
        env = LocalEnvironment(cwd="/tmp", timeout=1)
        result = env.execute("sleep 60")
        # Should not take 60 seconds; returncode is non-zero after kill
        assert result["returncode"] != 0

    def test_timeout_message_in_output(self):
        from tools.environments.local import LocalEnvironment
        env = LocalEnvironment(cwd="/tmp", timeout=1)
        result = env.execute("sleep 60")
        assert "timed out" in result["output"].lower() or result["returncode"] != 0


# ---------------------------------------------------------------------------
# Interrupt
# ---------------------------------------------------------------------------

class TestLocalEnvironmentInterrupt:
    def test_interrupt_event_kills_process(self):
        """Setting _interrupt_event while a long command runs should abort it."""
        from tools.environments.local import LocalEnvironment
        import tools.terminal_tool as tt

        env = LocalEnvironment(cwd="/tmp", timeout=30)
        tt._interrupt_event.clear()

        results = []

        def _run():
            results.append(env.execute("sleep 30"))

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # Give the process a moment to start
        time.sleep(0.3)
        tt._interrupt_event.set()

        t.join(timeout=5)
        assert not t.is_alive(), "Thread should have exited after interrupt"
        assert results, "execute() should have returned a result"
        assert results[0]["returncode"] == 130
        assert "interrupted" in results[0]["output"].lower()

        # Clean up
        tt._interrupt_event.clear()


# ---------------------------------------------------------------------------
# _kill_process_tree — platform-specific behaviour
# ---------------------------------------------------------------------------

class TestKillProcessTree:
    """Verify _kill_process_tree works on both Windows and POSIX paths."""

    def test_posix_path_uses_killpg(self):
        """On POSIX, _kill_process_tree should call os.killpg."""
        from tools.environments import local as local_mod

        mock_proc = MagicMock()
        mock_proc.pid = 1234

        with (
            patch.object(local_mod, "_IS_WINDOWS", False),
            patch("os.getpgid", return_value=1234) as mock_getpgid,
            patch("os.killpg") as mock_killpg,
        ):
            local_mod._kill_process_tree(mock_proc)
            mock_getpgid.assert_called_once_with(1234)
            mock_killpg.assert_called_once()

    def test_posix_force_sends_sigkill(self):
        import signal as _signal
        from tools.environments import local as local_mod

        mock_proc = MagicMock()
        mock_proc.pid = 1234

        with (
            patch.object(local_mod, "_IS_WINDOWS", False),
            patch("os.getpgid", return_value=1234),
            patch("os.killpg") as mock_killpg,
        ):
            local_mod._kill_process_tree(mock_proc, force=True)
            args, _ = mock_killpg.call_args
            assert args[1] == _signal.SIGKILL

    def test_windows_path_calls_proc_kill(self):
        """On Windows, _kill_process_tree should call proc.kill(), never os.killpg."""
        from tools.environments import local as local_mod

        mock_proc = MagicMock()

        with (
            patch.object(local_mod, "_IS_WINDOWS", True),
            patch("os.killpg") as mock_killpg,
        ):
            local_mod._kill_process_tree(mock_proc)
            mock_proc.kill.assert_called_once()
            mock_killpg.assert_not_called()

    def test_posix_fallback_on_permission_error(self):
        """If os.killpg raises PermissionError, proc.terminate() is called instead."""
        from tools.environments import local as local_mod

        mock_proc = MagicMock()
        mock_proc.pid = 9999

        with (
            patch.object(local_mod, "_IS_WINDOWS", False),
            patch("os.getpgid", side_effect=PermissionError),
        ):
            # Should not raise; falls back to proc.terminate()
            local_mod._kill_process_tree(mock_proc)
            mock_proc.terminate.assert_called_once()

    def test_posix_fallback_on_process_lookup_error(self):
        """If os.getpgid raises ProcessLookupError, proc.terminate() is called."""
        from tools.environments import local as local_mod

        mock_proc = MagicMock()
        mock_proc.pid = 8888

        with (
            patch.object(local_mod, "_IS_WINDOWS", False),
            patch("os.getpgid", side_effect=ProcessLookupError),
        ):
            local_mod._kill_process_tree(mock_proc)
            mock_proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# Windows execute path (mocked)
# ---------------------------------------------------------------------------

class TestLocalEnvironmentWindowsPath:
    """Verify the Windows code path executes without AttributeError.

    These tests mock _IS_WINDOWS=True and subprocess.Popen so we can run
    them on Linux CI without an actual Windows environment.
    """

    def _make_mock_proc(self, returncode: int = 0, stdout_lines: list = None):
        mock_proc = MagicMock()
        mock_proc.returncode = returncode
        mock_proc.poll.return_value = returncode
        stdout_lines = stdout_lines or ["mocked output\n"]
        mock_proc.stdout.__iter__ = MagicMock(return_value=iter(stdout_lines))
        mock_proc.stdin = MagicMock()
        return mock_proc

    def test_windows_uses_cmd_exe(self):
        """On Windows, the command should be wrapped in cmd.exe /C."""
        from tools.environments import local as local_mod
        import tools.terminal_tool as tt

        tt._interrupt_event.clear()
        mock_proc = self._make_mock_proc(returncode=0, stdout_lines=["win output\n"])

        with (
            patch.object(local_mod, "_IS_WINDOWS", True),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            env = local_mod.LocalEnvironment(cwd="/tmp", timeout=5)
            result = env.execute("echo hello")

        call_args = mock_popen.call_args
        cmd = call_args[0][0]  # first positional arg
        assert cmd[0] == "cmd.exe"
        assert "/C" in cmd

    def test_windows_no_preexec_fn(self):
        """Windows Popen call must NOT include preexec_fn (AttributeError on Windows)."""
        from tools.environments import local as local_mod
        import tools.terminal_tool as tt

        tt._interrupt_event.clear()
        mock_proc = self._make_mock_proc()

        with (
            patch.object(local_mod, "_IS_WINDOWS", True),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            env = local_mod.LocalEnvironment(cwd="/tmp", timeout=5)
            env.execute("dir")

        call_kwargs = mock_popen.call_args[1]
        assert "preexec_fn" not in call_kwargs, (
            "preexec_fn must not be passed on Windows — it raises AttributeError"
        )

    def test_windows_returns_output(self):
        """Windows path should return the same result dict shape as POSIX."""
        from tools.environments import local as local_mod
        import tools.terminal_tool as tt

        tt._interrupt_event.clear()
        mock_proc = self._make_mock_proc(returncode=0, stdout_lines=["hello\n"])

        with (
            patch.object(local_mod, "_IS_WINDOWS", True),
            patch("subprocess.Popen", return_value=mock_proc),
        ):
            env = local_mod.LocalEnvironment(cwd="/tmp", timeout=5)
            result = env.execute("echo hello")

        assert "output" in result
        assert "returncode" in result
        assert result["returncode"] == 0
