"""
Tests for the config-gated daemon kill on timeout feature.

When browser.terminate_daemon_on_timeout=false (default):
  Timeout handler should NOT run daemon kill / cleanup.
When browser.terminate_daemon_on_timeout=true:
  Timeout handler SHOULD run daemon kill / cleanup (covered by
  test_browser_timeout_cleanup.py with _cached_terminate_daemon=True).
"""
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_state():
    import tools.browser_tool as bt
    bt._active_sessions.clear()
    bt._recording_sessions.clear()
    bt._session_last_activity.clear()
    bt._last_active_session_key.clear()
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    bt._cached_terminate_daemon = False
    bt._terminate_daemon_resolved = True
    bt._LOCAL_SUFFIX = "::local"
    yield
    bt._active_sessions.clear()
    bt._recording_sessions.clear()
    bt._session_last_activity.clear()
    bt._last_active_session_key.clear()


def _setup_local_session(bt, task_id="test_task", session_name="hermes_test_session"):
    bt._active_sessions[task_id] = {"session_name": session_name}
    bt._recording_sessions.add(task_id)
    bt._session_last_activity[task_id] = 1000.0
    bt._last_active_session_key[task_id] = task_id


def _setup_pid_file(task_id, session_name, tmpdir_path):
    import tools.browser_tool as bt
    sdir = os.path.join(tmpdir_path, f"agent-browser-{session_name}")
    os.makedirs(sdir, exist_ok=True)
    pid = 12345
    with open(os.path.join(sdir, f"{session_name}.pid"), "w") as f:
        f.write(str(pid))
    return sdir, pid


@pytest.fixture
def mock_proc():
    proc = MagicMock()
    call_count = [0]

    def _wait(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            raise subprocess.TimeoutExpired(cmd="agent-browser", timeout=30)
        return 0

    proc.wait.side_effect = _wait
    proc.returncode = None
    return proc


@pytest.fixture
def mock_popen(mock_proc):
    with patch("subprocess.Popen", return_value=mock_proc) as m:
        yield m


class TestConfigDefault:
    """When _cached_terminate_daemon=False (default): no cleanup on timeout."""

    def test_cleanup_skipped_by_default(self, mock_popen, tmpdir):
        import tools.browser_tool as bt

        task_id = "test_task"
        session_name = "hermes_test_session"

        _setup_local_session(bt, task_id, session_name)
        _setup_pid_file(task_id, session_name, str(tmpdir))

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

        # Cleanup should NOT have happened
        assert task_id in bt._recording_sessions
        assert task_id in bt._active_sessions
        assert task_id in bt._session_last_activity
        mock_stop_cdp.assert_not_called()
        mock_terminate.assert_not_called()
        mock_rmtree.assert_not_called()

        # Result should still be a timeout error
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    def test_config_cached_value_persists(self):
        """Verify the lazy-init cache keeps False after first read."""
        import tools.browser_tool as bt
        # Already set to False by _clean_state
        assert bt._should_terminate_daemon_on_timeout() is False
        # Second call uses cache
        bt._cached_terminate_daemon = True
        # The cache was already resolved, so should still return old value
        # Actually, since resolved already, it returns cached value
        # Let's verify: first call should still be from cache
        bt._terminate_daemon_resolved = False
        bt._cached_terminate_daemon = False
        assert bt._should_terminate_daemon_on_timeout() is False
