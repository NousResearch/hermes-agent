"""
Comprehensive tests for _run_browser_command timeout handler cleanup logic.

The timeout handler (lines ~2105-2158 in browser_tool.py) was enhanced to:
1. Clear recording state
2. Stop CDP supervisors
3. Kill daemon processes via PID file
4. Remove socket directories
5. Pop tracking dictionaries

This test file covers all 10 scenarios from the spec plus edge cases.
"""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import inspect
from unittest.mock import MagicMock, PropertyMock, patch, ANY

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_globals(bt):
    """Reset all module-level state so tests start clean."""
    bt._active_sessions.clear()
    bt._recording_sessions.clear()
    bt._session_last_activity.clear()
    bt._last_active_session_key.clear()
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    bt._cached_terminate_daemon = True
    bt._terminate_daemon_resolved = True
    bt._LOCAL_SUFFIX = "::local"


@pytest.fixture(autouse=True)
def _clean_state():
    import tools.browser_tool as bt
    _reset_globals(bt)
    yield
    _reset_globals(bt)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_timeout_proc(first_raise: bool = True):
    """Create a proc mock where wait() raises TimeoutExpired on first call, succeeds on second.

    The _run_browser_command code calls proc.wait(timeout=...) first, then
    proc.kill() + proc.wait() again inside the except block.
    """
    from unittest.mock import MagicMock
    proc = MagicMock()
    call_count = [0]

    def _wait(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            raise subprocess.TimeoutExpired(cmd="agent-browser", timeout=30)
        return 0  # second call succeeds

    proc.wait.side_effect = _wait
    proc.returncode = None
    return proc


@pytest.fixture
def mock_proc():
    """A subprocess.Popen-like mock that raises TimeoutExpired on first wait call."""
    return _make_timeout_proc()


@pytest.fixture
def mock_popen(mock_proc):
    """Patch subprocess.Popen to return our mock_proc."""
    with patch("subprocess.Popen", return_value=mock_proc) as m:
        yield m


@pytest.fixture
def local_session_info():
    """Session info without cdp_url (local mode — cleanup should happen)."""
    return {"session_name": "hermes_test_session"}


@pytest.fixture
def cloud_session_info():
    """Session info WITH cdp_url (cloud mode — cleanup should be skipped)."""
    return {"session_name": "hermes_cloud_session", "cdp_url": "wss://remote.example.com/ws"}


def _setup_local_session(bt, task_id="test_task", session_name="hermes_test_session"):
    """Helper: populate _active_sessions with a local session."""
    bt._active_sessions[task_id] = {"session_name": session_name}
    bt._recording_sessions.add(task_id)
    bt._session_last_activity[task_id] = 1000.0
    bt._last_active_session_key[task_id] = task_id
    return session_name


def _setup_pid_file(task_id, session_name, tmpdir_path):
    """Helper: create a socket dir with a PID file, return (sdir, pid)."""
    from tools.browser_tool import _socket_safe_tmpdir

    sdir = os.path.join(tmpdir_path or _socket_safe_tmpdir(), f"agent-browser-{session_name}")
    os.makedirs(sdir, exist_ok=True)
    pid = 12345
    with open(os.path.join(sdir, f"{session_name}.pid"), "w") as f:
        f.write(str(pid))
    return sdir, pid


# ===========================================================================
# BASE INFRASTRUCTURE
# ===========================================================================

class TestCleanupFixtures:

    def test_reset_state_is_clean(self):
        import tools.browser_tool as bt
        assert len(bt._active_sessions) == 0
        assert len(bt._recording_sessions) == 0
        assert len(bt._session_last_activity) == 0
        assert len(bt._last_active_session_key) == 0

    def test_function_exists(self):
        """Verify _run_browser_command exists and has the expected timeout handler."""
        import tools.browser_tool as bt
        assert hasattr(bt, "_run_browser_command")
        src = bt._run_browser_command.__code__
        assert src.co_filename.endswith("browser_tool.py")


# ===========================================================================
# SCENARIO 1 — Basic timeout (local mode, no sidecar)
# ===========================================================================

class TestScenario1_BasicTimeout:

    def test_full_cleanup_flow(self, mock_popen, tmpdir):
        """Mock proc.wait → TimeoutExpired. Assert all cleanup steps execute."""
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_test_session"

        # Setup: populate session state
        _setup_local_session(bt, task_id, session_name)

        # Setup: create fake socket dir with PID file
        sdir, pid = _setup_pid_file(task_id, session_name, str(tmpdir))

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor") as mock_stop_cdp:
                with patch("shutil.rmtree") as mock_rmtree:
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid"
                            ) as mock_terminate:
                                result = bt._run_browser_command(
                                    task_id, "navigate", ["https://example.com"]
                                )

        # 1. _recording_sessions.discard called
        assert task_id not in bt._recording_sessions

        # 2. _stop_cdp_supervisor called with task_id
        mock_stop_cdp.assert_called_once_with(task_id)

        # 3. ProcessRegistry._terminate_host_pid called with correct PID
        mock_terminate.assert_called_once_with(pid)

        # 4. shutil.rmtree called with correct socket dir
        mock_rmtree.assert_called_once_with(sdir, ignore_errors=True)

        # 5. _active_sessions[task_id] popped
        assert task_id not in bt._active_sessions

        # 6. _session_last_activity[task_id] popped
        assert task_id not in bt._session_last_activity

        # 7. _last_active_session_key[task_id] popped
        assert task_id not in bt._last_active_session_key

        # Result should indicate failure
        assert result["success"] is False
        assert "timed out" in result["error"].lower()


# ===========================================================================
# SCENARIO 2 — No sidecar: only one key in for loop
# ===========================================================================

class TestScenario2_NoSidecar:

    def test_single_key_only(self, mock_popen, tmpdir):
        """Verify _stop_cdp_supervisor called exactly once (no ::local key)."""
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_test_session"

        _setup_local_session(bt, task_id, session_name)

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor") as mock_stop_cdp:
                with patch("shutil.rmtree"):
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            bt._run_browser_command(task_id, "navigate", ["https://example.com"])

        # Only one call, with the bare task_id (not ::local)
        mock_stop_cdp.assert_called_once_with(task_id)


# ===========================================================================
# SCENARIO 3 — With sidecar (::local key)
# ===========================================================================

class TestScenario3_WithSidecar:

    def test_sidecar_key_also_cleaned(self, mock_popen, tmpdir):
        """Pre-populate _active_sessions with f'{task_id}::local'."""
        import tools.browser_tool as bt

        task_id = "test_task"
        local_key = f"{task_id}::local"
        session_name = "hermes_test_session"

        # Setup: both keys in _active_sessions
        _setup_local_session(bt, task_id, session_name)
        bt._active_sessions[local_key] = {"session_name": session_name + "_local"}
        bt._session_last_activity[local_key] = 2000.0

        # Create two socket dirs
        sdir1 = os.path.join(str(tmpdir), f"agent-browser-{session_name}")
        os.makedirs(sdir1, exist_ok=True)
        pid1 = 12345
        with open(os.path.join(sdir1, f"{session_name}.pid"), "w") as f:
            f.write(str(pid1))

        sdir2 = os.path.join(str(tmpdir), f"agent-browser-{session_name}_local")
        os.makedirs(sdir2, exist_ok=True)
        pid2 = 67890
        with open(os.path.join(sdir2, f"{session_name}_local.pid"), "w") as f:
            f.write(str(pid2))

        call_count = {"stop_cdp": 0, "terminate": 0, "rmtree": 0}

        def count_stop_cdp(key):
            call_count["stop_cdp"] += 1

        def count_terminate(pid):
            call_count["terminate"] += 1

        def count_rmtree(path, **kwargs):
            call_count["rmtree"] += 1

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor", side_effect=count_stop_cdp):
                with patch("shutil.rmtree", side_effect=count_rmtree):
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid",
                                side_effect=count_terminate,
                            ):
                                bt._run_browser_command(
                                    task_id, "navigate", ["https://example.com"]
                                )

        # for loop runs twice
        assert call_count["stop_cdp"] == 2
        assert call_count["terminate"] == 2
        assert call_count["rmtree"] == 2

        # Both keys popped from dicts
        assert task_id not in bt._active_sessions
        assert local_key not in bt._active_sessions
        assert task_id not in bt._session_last_activity
        assert local_key not in bt._session_last_activity


# ===========================================================================
# SCENARIO 4 — Cloud mode (cdp_url set): cleanup entirely skipped
# ===========================================================================

class TestScenario4_CloudMode:

    def test_cleanup_skipped_in_cloud_mode(self, mock_popen, tmpdir):
        """session_info has cdp_url — no kill, no rmtree, no pop."""
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_cloud_session"

        _setup_local_session(bt, task_id, session_name)

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor") as mock_stop_cdp:
                with patch("shutil.rmtree") as mock_rmtree:
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name, "cdp_url": "wss://remote.example.com/ws"},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid"
                            ) as mock_terminate:
                                result = bt._run_browser_command(
                                    task_id, "navigate", ["https://example.com"]
                                )

        # No cleanup should have happened
        mock_stop_cdp.assert_not_called()
        mock_terminate.assert_not_called()
        mock_rmtree.assert_not_called()

        # Sessions should NOT be popped (they're cloud-managed)
        assert task_id in bt._active_sessions
        assert task_id in bt._session_last_activity
        assert task_id in bt._last_active_session_key

        # Result should still be timeout failure
        assert result["success"] is False


# ===========================================================================
# SCENARIO 5 — PID file not found: daemon kill skipped, rmtree still called
# ===========================================================================

class TestScenario5_PidFileNotFound:

    def test_daemon_kill_skipped_when_no_pid_file(self, mock_popen, tmpdir):
        """os.path.isfile returns False → rmtree still called."""
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_test_session"

        _setup_local_session(bt, task_id, session_name)
        sdir = os.path.join(str(tmpdir), f"agent-browser-{session_name}")
        os.makedirs(sdir, exist_ok=True)
        # No PID file created

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor"):
                with patch("shutil.rmtree") as mock_rmtree:
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid"
                            ) as mock_terminate:
                                bt._run_browser_command(
                                    task_id, "navigate", ["https://example.com"]
                                )

        # Daemon kill skipped
        mock_terminate.assert_not_called()

        # rmtree still called
        mock_rmtree.assert_called_once_with(sdir, ignore_errors=True)

        # logger.warning should NOT be called for daemon failure
        # (no warning needed when PID file just doesn't exist)


# ===========================================================================
# SCENARIO 6 — PID file has garbage: ValueError caught, rmtree still called
# ===========================================================================

class TestScenario6_GarbagePid:

    def test_garbage_pid_caught_and_logged(self, mock_popen, tmpdir):
        """PID file content 'abc' → ValueError caught, rmtree still called."""
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_test_session"

        _setup_local_session(bt, task_id, session_name)
        sdir = os.path.join(str(tmpdir), f"agent-browser-{session_name}")
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, f"{session_name}.pid"), "w") as f:
            f.write("abc")  # Garbage PID

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor"):
                with patch("shutil.rmtree") as mock_rmtree:
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid"
                            ) as mock_terminate:
                                with patch("tools.browser_tool.logger.warning") as mock_log_warn:
                                    bt._run_browser_command(
                                        task_id, "navigate", ["https://example.com"]
                                    )

        # Daemon kill NOT called (ValueError before reaching it, or caught)
        # Actually: ValueError from int() is caught inside the try/except at line 2145
        # Let's trace: Path(pid_file).read_text() returns "abc", int("abc") raises ValueError
        # The ValueError is caught by `except (ProcessLookupError, ValueError, ...)` at line 2145
        # So _terminate_host_pid is NEVER called
        mock_terminate.assert_not_called()

        # logger.warning called with the daemon-kill-failure message
        mock_log_warn.assert_any_call(
            "browser '%s' timed out — daemon kill skipped: %s", ANY, ANY
        )

        # rmtree still called
        mock_rmtree.assert_called_once_with(sdir, ignore_errors=True)


# ===========================================================================
# SCENARIO 7 — Daemon already dead: ProcessLookupError caught
# ===========================================================================

class TestScenario7_DaemonAlreadyDead:

    def test_process_lookup_error_handled(self, mock_popen, tmpdir):
        """_terminate_host_pid raises ProcessLookupError → caught, rmtree still called."""
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_test_session"

        _setup_local_session(bt, task_id, session_name)
        sdir, pid = _setup_pid_file(task_id, session_name, str(tmpdir))

        def raise_lookup(*args, **kwargs):
            raise ProcessLookupError(f"No process with PID {pid}")

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor"):
                with patch("shutil.rmtree") as mock_rmtree:
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid",
                                side_effect=raise_lookup,
                            ):
                                with patch("tools.browser_tool.logger.warning") as mock_log_warn:
                                    bt._run_browser_command(
                                        task_id, "navigate", ["https://example.com"]
                                    )

        # logger.warning called with the daemon-kill-failure message
        mock_log_warn.assert_any_call(
            "browser '%s' timed out — daemon kill skipped: %s", ANY, ANY
        )

        # rmtree still called
        mock_rmtree.assert_called_once_with(sdir, ignore_errors=True)


# ===========================================================================
# SCENARIO 8 — Cloud mode with sidecar hybrid: cdp_url set, whole block skipped
# ===========================================================================

class TestScenario8_CloudModeWithSidecar:

    def test_cloud_mode_skipped_even_with_sidecar(self, mock_popen, tmpdir):
        """Cloud mode skips cleanup even when ::local key exists in _active_sessions."""
        import tools.browser_tool as bt

        task_id = "test_task"
        local_key = f"{task_id}::local"
        session_name = "hermes_cloud_session"

        # Setup: populate with cloud session info (cdp_url set)
        bt._active_sessions[task_id] = {"session_name": session_name, "cdp_url": "wss://remote.example.com/ws"}
        bt._active_sessions[local_key] = {"session_name": session_name + "_local"}
        bt._recording_sessions.add(task_id)
        bt._recording_sessions.add(local_key)
        bt._session_last_activity[task_id] = 1000.0
        bt._session_last_activity[local_key] = 2000.0
        bt._last_active_session_key[task_id] = task_id

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor") as mock_stop_cdp:
                with patch("shutil.rmtree") as mock_rmtree:
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name, "cdp_url": "wss://remote.example.com/ws"},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid"
                            ) as mock_terminate:
                                result = bt._run_browser_command(
                                    task_id, "navigate", ["https://example.com"]
                                )

        # No cleanup should happen at all
        mock_stop_cdp.assert_not_called()
        mock_terminate.assert_not_called()
        mock_rmtree.assert_not_called()

        # Sessions should remain intact
        assert task_id in bt._active_sessions
        assert local_key in bt._active_sessions

        # But _recording_sessions should still have entries (discard was gated by session_keys loop)
        # Actually: the _recording_sessions.discard is inside the `if not session_info.get("cdp_url")` block
        # So it should NOT be discarded
        assert task_id in bt._recording_sessions
        assert local_key in bt._recording_sessions


# ===========================================================================
# SCENARIO 9 — Crash resilience: rmtree raises on first key, second still done
# ===========================================================================

class TestScenario9_CrashResilience:

    def test_rmtree_crash_does_not_block_second_key(self, mock_popen, tmpdir):
        """shutil.rmtree raises PermissionError on first key; second still cleaned up."""
        import tools.browser_tool as bt

        task_id = "test_task"
        local_key = f"{task_id}::local"
        session_name = "hermes_test_session"

        # Setup: both keys in _active_sessions
        _setup_local_session(bt, task_id, session_name)
        bt._active_sessions[local_key] = {"session_name": session_name + "_local"}
        bt._session_last_activity[local_key] = 2000.0

        sdir1 = os.path.join(str(tmpdir), f"agent-browser-{session_name}")
        os.makedirs(sdir1, exist_ok=True)
        pid1 = 12345
        with open(os.path.join(sdir1, f"{session_name}.pid"), "w") as f:
            f.write(str(pid1))

        sdir2 = os.path.join(str(tmpdir), f"agent-browser-{session_name}_local")
        os.makedirs(sdir2, exist_ok=True)
        pid2 = 67890
        with open(os.path.join(sdir2, f"{session_name}_local.pid"), "w") as f:
            f.write(str(pid2))

        rmtree_calls = []

        def crashing_rmtree(path, **kwargs):
            rmtree_calls.append(path)
            if len(rmtree_calls) == 1:
                raise PermissionError(f"Permission denied: {path}")

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor"):
                with patch("shutil.rmtree", side_effect=crashing_rmtree):
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid"
                            ):
                                with patch("tools.browser_tool.logger.warning") as mock_log_warn:
                                    result = bt._run_browser_command(
                                        task_id, "navigate", ["https://example.com"]
                                    )

        # Both rmtree calls attempted
        assert len(rmtree_calls) == 2

        # Warning should mention the error on the first key (test_task, not ::local)
        mock_log_warn.assert_any_call(
            "Error during browser session cleanup for %s: %s", task_id, ANY
        )

        # First key (test_task) was NOT popped because rmtree crashed inside the try block
        # before _active_sessions.pop(sk) executed. Only the second key gets properly cleaned up.
        assert task_id in bt._active_sessions  # crash prevented pop
        assert local_key not in bt._active_sessions  # second key popped normally

        # _session_last_activity: same story
        assert task_id in bt._session_last_activity
        assert local_key not in bt._session_last_activity

        # _last_active_session_key.pop still executed (it's outside the for loop)
        assert task_id not in bt._last_active_session_key

        # Error result returned
        assert result["success"] is False


# ===========================================================================
# SCENARIO 10 — _stop_cdp_supervisor raises on first key
# ===========================================================================

class TestScenario10_SupervisorRaise:

    def test_stop_cdp_crash_does_not_block_rest(self, mock_popen, tmpdir):
        """_stop_cdp_supervisor raises RuntimeError; remaining keys processed."""
        import tools.browser_tool as bt

        task_id = "test_task"
        local_key = f"{task_id}::local"
        session_name = "hermes_test_session"

        _setup_local_session(bt, task_id, session_name)
        bt._active_sessions[local_key] = {"session_name": session_name + "_local"}
        bt._session_last_activity[local_key] = 2000.0

        sdir1 = os.path.join(str(tmpdir), f"agent-browser-{session_name}")
        os.makedirs(sdir1, exist_ok=True)
        pid1 = 12345
        with open(os.path.join(sdir1, f"{session_name}.pid"), "w") as f:
            f.write(str(pid1))

        sdir2 = os.path.join(str(tmpdir), f"agent-browser-{session_name}_local")
        os.makedirs(sdir2, exist_ok=True)
        pid2 = 67890
        with open(os.path.join(sdir2, f"{session_name}_local.pid"), "w") as f:
            f.write(str(pid2))

        stop_cdp_calls = []

        def crashing_stop(key):
            stop_cdp_calls.append(key)
            if len(stop_cdp_calls) == 1:
                raise RuntimeError(f"Supervisor failed for {key}")

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor", side_effect=crashing_stop):
                with patch("shutil.rmtree") as mock_rmtree:
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid"
                            ):
                                with patch("tools.browser_tool.logger.warning") as mock_log_warn:
                                    result = bt._run_browser_command(
                                        task_id, "navigate", ["https://example.com"]
                                    )

        # Both stop_cdp calls attempted
        assert len(stop_cdp_calls) == 2
        assert stop_cdp_calls == [task_id, local_key]

        # Warning logged for the crash
        mock_log_warn.assert_any_call(
            "Error during browser session cleanup for %s: %s", task_id, ANY
        )

        # Only one rmtree call (second key; first key crashed before reaching rmtree)
        assert mock_rmtree.call_count == 1

        # First key NOT popped (exception jumped past pop), second key popped
        assert task_id in bt._active_sessions
        assert local_key not in bt._active_sessions

        # _last_active_session_key.pop still executed
        assert task_id not in bt._last_active_session_key

        assert result["success"] is False


# ===========================================================================
# ADDITIONAL EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_session_info_with_session_name_null(self, mock_popen, tmpdir):
        """session_name is empty string → skip daemon kill and rmtree."""
        import tools.browser_tool as bt

        task_id = "test_task"
        bt._active_sessions[task_id] = {"session_name": ""}
        bt._recording_sessions.add(task_id)
        bt._session_last_activity[task_id] = 1000.0
        bt._last_active_session_key[task_id] = task_id

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor") as mock_stop_cdp:
                with patch("shutil.rmtree") as mock_rmtree:
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": ""},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid"
                            ) as mock_terminate:
                                result = bt._run_browser_command(
                                    task_id, "navigate", ["https://example.com"]
                                )

        # _stop_cdp_supervisor still called
        mock_stop_cdp.assert_called_once_with(task_id)

        # No daemon kill or rmtree (session_name empty)
        mock_terminate.assert_not_called()
        mock_rmtree.assert_not_called()

        # Sessions popped
        assert task_id not in bt._active_sessions
        assert task_id not in bt._session_last_activity
        assert task_id not in bt._last_active_session_key

    def test_no_session_info_at_all(self, mock_popen, tmpdir):
        """_active_sessions has no entry for task_id → sinfo is None → skip."""
        import tools.browser_tool as bt

        task_id = "test_task"
        # Don't populate _active_sessions at all
        bt._last_active_session_key[task_id] = task_id

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor") as mock_stop_cdp:
                with patch("shutil.rmtree") as mock_rmtree:
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": "my_session"},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid"
                            ) as mock_terminate:
                                result = bt._run_browser_command(
                                    task_id, "navigate", ["https://example.com"]
                                )

        # _stop_cdp_supervisor still called (it's before the sinfo check)
        mock_stop_cdp.assert_called_once_with(task_id)

        # No daemon kill or rmtree (sinfo was None)
        mock_terminate.assert_not_called()
        mock_rmtree.assert_not_called()

        # Pop still happens (it's unconditional under lock after try/except)
        assert task_id not in bt._active_sessions
        assert task_id not in bt._session_last_activity
        assert task_id not in bt._last_active_session_key

    def test_sidecar_key_not_in_active_sessions_skipped(self, mock_popen, tmpdir):
        """sidecar key not present in _active_sessions → only one key in loop."""
        import tools.browser_tool as bt

        task_id = "test_task"
        local_key = f"{task_id}::local"
        session_name = "hermes_test_session"

        _setup_local_session(bt, task_id, session_name)
        # DO NOT add local_key to _active_sessions

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor") as mock_stop_cdp:
                with patch("shutil.rmtree"):
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            bt._run_browser_command(
                                task_id, "navigate", ["https://example.com"]
                            )

        # Only one call to _stop_cdp_supervisor
        mock_stop_cdp.assert_called_once_with(task_id)


# ===========================================================================
# LOG VERIFICATION TESTS (Scenarios 5-7)
# ===========================================================================

class TestLogVerification:

    def test_log_warning_on_garbage_pid(self, mock_popen, tmpdir):
        """Scenario 6: logger.warning called with daemon-kill-failure message for garbage PID."""
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_test_session"
        _setup_local_session(bt, task_id, session_name)
        sdir = os.path.join(str(tmpdir), f"agent-browser-{session_name}")
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, f"{session_name}.pid"), "w") as f:
            f.write("not_a_number")

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor"):
                with patch("shutil.rmtree"):
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch("tools.browser_tool.logger.warning") as mock_log_warn:
                                bt._run_browser_command(
                                    task_id, "navigate", ["https://example.com"]
                                )

        # Check that the specific daemon-kill-failure warning was emitted
        daemon_warnings = [
            call for call in mock_log_warn.call_args_list
            if "daemon kill skipped" in str(call[0])
        ]
        assert len(daemon_warnings) >= 1, (
            f"Expected 'daemon kill skipped' warning, got: {mock_log_warn.call_args_list}"
        )

    def test_log_warning_on_process_lookup(self, mock_popen, tmpdir):
        """Scenario 7: logger.warning called with daemon-kill-failure for ProcessLookupError."""
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_test_session"
        _setup_local_session(bt, task_id, session_name)
        sdir, pid = _setup_pid_file(task_id, session_name, str(tmpdir))

        def raise_lookup(*args, **kwargs):
            raise ProcessLookupError(f"pid {pid} not found")

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor"):
                with patch("shutil.rmtree"):
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid",
                                side_effect=raise_lookup,
                            ):
                                with patch("tools.browser_tool.logger.warning") as mock_log_warn:
                                    bt._run_browser_command(
                                        task_id, "navigate", ["https://example.com"]
                                    )

        daemon_warnings = [
            call for call in mock_log_warn.call_args_list
            if "daemon kill skipped" in str(call[0])
        ]
        assert len(daemon_warnings) >= 1, (
            f"Expected 'daemon kill skipped' warning, got: {mock_log_warn.call_args_list}"
        )

    def test_log_warning_on_permission_error(self, mock_popen, tmpdir):
        """PermissionError from terminate_host_pid is also caught and logged."""
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_test_session"
        _setup_local_session(bt, task_id, session_name)
        sdir, pid = _setup_pid_file(task_id, session_name, str(tmpdir))

        def raise_perm(*args, **kwargs):
            raise PermissionError("Operation not permitted")

        with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmpdir)):
            with patch.object(bt, "_stop_cdp_supervisor"):
                with patch("shutil.rmtree"):
                    with patch(
                        "tools.browser_tool._find_agent_browser",
                        return_value="/usr/bin/agent-browser",
                    ):
                        with patch(
                            "tools.browser_tool._get_session_info",
                            return_value={"session_name": session_name},
                        ):
                            with patch(
                                "tools.process_registry.ProcessRegistry._terminate_host_pid",
                                side_effect=raise_perm,
                            ):
                                with patch("tools.browser_tool.logger.warning") as mock_log_warn:
                                    bt._run_browser_command(
                                        task_id, "navigate", ["https://example.com"]
                                    )

        daemon_warnings = [
            call for call in mock_log_warn.call_args_list
            if "daemon kill skipped" in str(call[0])
        ]
        assert len(daemon_warnings) >= 1, (
            f"Expected 'daemon kill skipped' warning, got: {mock_log_warn.call_args_list}"
        )


# ===========================================================================
# CODE STRUCTURE CHECKS
# ===========================================================================

class TestCodeStructure:

    def test_cleanup_block_exists(self):
        """Verify the TimeoutExpired handler has the cleanup block."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt._run_browser_command)

        # The cleanup block should have these elements
        assert "_recording_sessions.discard(sk)" in src
        assert "_stop_cdp_supervisor(sk)" in src
        assert "ProcessRegistry._terminate_host_pid" in src
        assert "shutil.rmtree(sdir" in src
        assert "_last_active_session_key.pop(task_id, None)" in src

    def test_cleanup_wrapped_in_cloud_check(self):
        """Cleanup should be inside 'if not session_info.get(\"cdp_url\"):'."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt._run_browser_command)

        # Find the cloud-check line
        lines = src.split("\n")
        cloud_check_lines = [i for i, l in enumerate(lines) if 'not session_info.get(\"cdp_url\")' in l]
        assert len(cloud_check_lines) >= 1, (
            "Expected cloud-guard including 'not session_info.get(\"cdp_url\")' before cleanup block"
        )

    def test_inner_try_except_exists(self):
        """The for loop body should be in try/except Exception."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt._run_browser_command)

        # Should have an inner try-except inside the for loop for crash resilience
        assert "try:" in src
        assert "except Exception as e:" in src
        assert "logger.warning" in src
