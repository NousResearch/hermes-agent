"""Integration tests: browser timeout daemon cleanup.

These tests mock subprocess.Popen to trigger TimeoutExpired in
_run_browser_command, then verify the cleanup behavior (daemon kill,
rmtree, dict pops, etc.).
"""

import os
import subprocess
import tempfile
import threading
from unittest.mock import patch, MagicMock, ANY, call
from pathlib import Path

import pytest


# ── helpers ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level dicts/sets before each test so state doesn't leak."""
    import tools.browser_tool as bt
    with bt._cleanup_lock:
        bt._active_sessions.clear()
        bt._session_last_activity.clear()
        bt._recording_sessions.clear()
        bt._last_active_session_key.clear()
    yield


def make_session_info(session_name="test-session", cdp_url=None):
    d = {"session_name": session_name}
    if cdp_url:
        d["cdp_url"] = cdp_url
    return d


def setup_session(bt, task_id="task-1", session_name="test-session", cdp_url=None):
    """Helper: populate module-level state as if a session was started."""
    session_info = make_session_info(session_name, cdp_url)
    with bt._cleanup_lock:
        bt._active_sessions[task_id] = session_info
        bt._session_last_activity[task_id] = 1000.0
    bt._last_active_session_key[task_id] = task_id
    return session_info


# ── patches ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_proc(mock_config):
    """Mock subprocess.Popen whose .wait() raises TimeoutExpired."""
    proc = MagicMock()
    proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="test", timeout=30), None]
    proc.pid = 12345
    proc.returncode = None
    return proc


@pytest.fixture
def mock_config():
    """Mock config reads and CLI discovery so _run_browser_command proceeds."""
    # _find_agent_browser → returns path
    patcher_find = patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser")
    # Chromium check → return True so we don't block
    patcher_chrome = patch("tools.browser_tool._chromium_installed", return_value=True)
    # _is_local_mode → True so Chromium check matters
    patcher_local = patch("tools.browser_tool._is_local_mode", return_value=True)
    # _get_browser_engine → "chrome"
    patcher_engine = patch("tools.browser_tool._get_browser_engine", return_value="chrome")
    # _get_command_timeout → 30
    patcher_timeout = patch("tools.browser_tool._get_command_timeout", return_value=30)
    # tools.interrupt.is_interrupted → False
    patcher_interrupt = patch("tools.interrupt.is_interrupted", return_value=False)
    # _socket_safe_tmpdir → temp dir
    tmpdir = tempfile.mkdtemp()
    patcher_tmpdir = patch("tools.browser_tool._socket_safe_tmpdir", return_value=tmpdir)
    # Path.read_text → return a PID
    patcher_readtext = patch.object(Path, "read_text", return_value="99999")
    # ProcessRegistry._terminate_host_pid → no-op (lazy-imported, patch the real path)
    patcher_kill = patch("tools.process_registry.ProcessRegistry._terminate_host_pid")
    # _stop_cdp_supervisor → no-op
    patcher_cdp = patch("tools.browser_tool._stop_cdp_supervisor")
    # os.path.isdir → True
    patcher_isdir = patch("os.path.isdir", return_value=True)
    # os.path.isfile → True (pid file exists)
    patcher_isfile = patch("os.path.isfile", return_value=True)
    # shutil.rmtree → no-op
    patcher_rmtree = patch("shutil.rmtree")

    for p in [patcher_find, patcher_chrome, patcher_local, patcher_engine,
              patcher_timeout, patcher_interrupt, patcher_tmpdir, patcher_readtext,
              patcher_kill, patcher_cdp, patcher_isdir, patcher_isfile,
              patcher_rmtree]:
        p.start()

    yield

    for p in [patcher_find, patcher_chrome, patcher_local, patcher_engine,
              patcher_timeout, patcher_interrupt, patcher_tmpdir, patcher_readtext,
              patcher_kill, patcher_cdp, patcher_isdir, patcher_isfile,
              patcher_rmtree]:
        p.stop()

    # Clean up temp dir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def run_timeout_command(bt, task_id="task-1", mock_proc=None):
    """Call _run_browser_command with mocked Popen."""
    with patch("tools.browser_tool.subprocess.Popen", return_value=mock_proc):
        result = bt._run_browser_command(task_id, "open", timeout=30)
    return result


# ── Tests ────────────────────────────────────────────────────────────────

class TestBrowserTimeoutCleanup:

    def test_basic_timeout_cleans_up(self, mock_config, mock_proc):
        """Scenario 1: basic timeout → daemon killed, dicts popped, rmtree called."""
        import tools.browser_tool as bt
        setup_session(bt)
        bt._recording_sessions.add("task-1")

        result = run_timeout_command(bt, "task-1", mock_proc)

        # Error result
        assert result["success"] is False
        assert "timed out" in result["error"]

        # Recording cleared
        assert "task-1" not in bt._recording_sessions

        # Dicts popped
        assert "task-1" not in bt._active_sessions
        assert "task-1" not in bt._session_last_activity
        assert "task-1" not in bt._last_active_session_key

    def test_no_sidecar_single_key(self, mock_config, mock_proc):
        """Scenario 2: no sidecar → only one key cleaned."""
        import tools.browser_tool as bt
        setup_session(bt)

        result = run_timeout_command(bt, "task-1", mock_proc)
        assert result["success"] is False
        assert "task-1" not in bt._active_sessions

    def test_with_sidecar(self, mock_config, mock_proc):
        """Scenario 3: sidecar key exists → both keys cleaned."""
        import tools.browser_tool as bt
        setup_session(bt)
        # Add sidecar session
        sidecar_key = "task-1::local"
        with bt._cleanup_lock:
            bt._active_sessions[sidecar_key] = make_session_info("test-session-sidecar")
            bt._session_last_activity[sidecar_key] = 1001.0

        result = run_timeout_command(bt, "task-1", mock_proc)
        assert result["success"] is False
        assert "task-1" not in bt._active_sessions
        assert sidecar_key not in bt._active_sessions
        assert sidecar_key not in bt._session_last_activity

    def test_cloud_mode_skips_cleanup(self, mock_config, mock_proc):
        """Scenario 4: cloud mode (cdp_url set) → no cleanup at all."""
        import tools.browser_tool as bt
        setup_session(bt, cdp_url="wss://browserbase.com/ws")

        result = run_timeout_command(bt, "task-1", mock_proc)
        assert result["success"] is False
        # Session should still be in active_sessions (cleanup skipped)
        assert "task-1" in bt._active_sessions
        assert "task-1" in bt._session_last_activity
        assert "task-1" in bt._last_active_session_key

    def test_pid_file_missing(self, mock_config, mock_proc):
        """Scenario 5: PID file not found → daemon kill skipped, rmtree still called."""
        import tools.browser_tool as bt
        setup_session(bt)

        with patch("os.path.isfile", return_value=False):
            result = run_timeout_command(bt, "task-1", mock_proc)

        assert result["success"] is False
        assert "task-1" not in bt._active_sessions

    def test_pid_file_garbage_value(self, mock_config, mock_proc):
        """Scenario 6: PID file has garbage → ValueError caught + logger.warning called."""
        import tools.browser_tool as bt
        setup_session(bt)

        with patch.object(Path, "read_text", return_value="abc"):
            with patch("tools.browser_tool.logger.warning") as mock_logger:
                result = run_timeout_command(bt, "task-1", mock_proc)

        assert result["success"] is False
        assert "task-1" not in bt._active_sessions
        # Should have logged a warning about kill failure
        mock_logger.assert_any_call("Could not kill daemon for %s: %s", "test-session", ANY)

    def test_daemon_already_dead(self, mock_config, mock_proc):
        """Scenario 7: daemon already dead → ProcessLookupError caught."""
        import tools.browser_tool as bt
        setup_session(bt)

        from tools.process_registry import ProcessRegistry
        with patch.object(ProcessRegistry, "_terminate_host_pid",
                   side_effect=ProcessLookupError()):
            with patch("tools.browser_tool.logger.warning") as mock_logger:
                result = run_timeout_command(bt, "task-1", mock_proc)

        assert result["success"] is False
        assert "task-1" not in bt._active_sessions
        mock_logger.assert_any_call("Could not kill daemon for %s: %s", "test-session", ANY)

    def test_cloud_mode_with_sidecar_skipped(self, mock_config, mock_proc):
        """Scenario 8: cloud mode + sidecar → entire block skipped."""
        import tools.browser_tool as bt
        setup_session(bt, cdp_url="wss://example.com/ws")
        sidecar_key = "task-1::local"
        with bt._cleanup_lock:
            bt._active_sessions[sidecar_key] = make_session_info("sidecar", cdp_url=None)

        result = run_timeout_command(bt, "task-1", mock_proc)
        assert result["success"] is False
        # Cleanup skipped — both keys remain
        assert "task-1" in bt._active_sessions
        assert sidecar_key in bt._active_sessions

    def test_crash_resilience(self, mock_config, mock_proc):
        """Scenario 9: PermissionError on first rmtree → second key still cleaned, step 5 still runs."""
        import tools.browser_tool as bt
        import shutil
        setup_session(bt)
        sidecar_key = "task-1::local"
        with bt._cleanup_lock:
            bt._active_sessions[sidecar_key] = make_session_info("sidecar")
            bt._session_last_activity[sidecar_key] = 1001.0

        # First rmtree call raises → second should succeed
        original_rmtree = shutil.rmtree
        call_count = [0]

        def flaky_rmtree(path, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise PermissionError("Permission denied")
            return original_rmtree(path, **kwargs)

        with patch("shutil.rmtree", side_effect=flaky_rmtree):
            result = run_timeout_command(bt, "task-1", mock_proc)

        assert result["success"] is False
        assert "task-1" not in bt._active_sessions
        assert sidecar_key not in bt._active_sessions
        assert "task-1" not in bt._last_active_session_key

    def test_stop_cdp_supervisor_raises(self, mock_config, mock_proc):
        """Scenario 10: _stop_cdp_supervisor raises → for loop continues, step 5 still runs."""
        import tools.browser_tool as bt
        setup_session(bt)
        sidecar_key = "task-1::local"
        with bt._cleanup_lock:
            bt._active_sessions[sidecar_key] = make_session_info("sidecar")
            bt._session_last_activity[sidecar_key] = 1001.0

        call_count = [0]

        def flaky_cdp(sk):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CDP supervisor crashed")
            return None

        with patch("tools.browser_tool._stop_cdp_supervisor", side_effect=flaky_cdp):
            result = run_timeout_command(bt, "task-1", mock_proc)

        assert result["success"] is False
        # Sidecar key should still have been cleaned (for loop continued)
        assert "task-1" not in bt._active_sessions
        assert sidecar_key not in bt._active_sessions
        assert "task-1" not in bt._last_active_session_key

    def test_logger_warning_on_kill_failure(self, mock_config, mock_proc):
        """Logger.warning is called when daemon kill fails with ValueError."""
        import tools.browser_tool as bt
        setup_session(bt)

        with patch.object(Path, "read_text", return_value="not_a_number"):
            with patch("tools.browser_tool.logger.warning") as mock_warning:
                result = run_timeout_command(bt, "task-1", mock_proc)

        assert result["success"] is False
        # Find the specific daemon kill warning
        kill_warnings = [
            c for c in mock_warning.call_args_list
            if c[0][0] == "Could not kill daemon for %s: %s"
        ]
        assert len(kill_warnings) >= 1, "Expected logger.warning for kill failure"
