"""Regression tests for browser session cleanup and screenshot recovery."""

from unittest.mock import patch


class TestScreenshotPathRecovery:
    def test_extracts_standard_absolute_path(self):
        from tools.browser_tool import _extract_screenshot_path_from_text

        assert (
            _extract_screenshot_path_from_text("Screenshot saved to /tmp/foo.png")
            == "/tmp/foo.png"
        )

    def test_extracts_quoted_absolute_path(self):
        from tools.browser_tool import _extract_screenshot_path_from_text

        assert (
            _extract_screenshot_path_from_text(
                "Screenshot saved to '/Users/david/.hermes/browser_screenshots/shot.png'"
            )
            == "/Users/david/.hermes/browser_screenshots/shot.png"
        )


class TestBrowserCleanup:
    def setup_method(self):
        from tools import browser_tool

        self.browser_tool = browser_tool
        self.orig_active_sessions = browser_tool._active_sessions.copy()
        self.orig_session_last_activity = browser_tool._session_last_activity.copy()
        self.orig_recording_sessions = browser_tool._recording_sessions.copy()
        self.orig_cleanup_done = browser_tool._cleanup_done

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._recording_sessions.clear()
        self.browser_tool._recording_sessions.update(self.orig_recording_sessions)
        self.browser_tool._cleanup_done = self.orig_cleanup_done

    def test_cleanup_browser_clears_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-1"] = 123.0

        with (
            patch("tools.browser_tool._maybe_stop_recording") as mock_stop,
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            browser_tool.cleanup_browser("task-1")

        assert "task-1" not in browser_tool._active_sessions
        assert "task-1" not in browser_tool._session_last_activity
        mock_stop.assert_called_once_with("task-1")
        mock_run.assert_called_once_with("task-1", "close", [], timeout=10)

    def test_explicit_session_forwards_verified_daemon_start_guard(self, tmp_path):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        socket_dir = tmp_path / "agent-browser-sess-1"
        socket_dir.mkdir()
        (socket_dir / "sess-1.pid").write_text("12345")

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}),
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("gateway.status._pid_exists", side_effect=[True, False]),
            patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=777),
            patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as terminate,
            patch("tools.browser_tool._reap_session_chromium"),
            patch("tools.browser_tool._remove_browser_socket_dir_if_safe") as remove_socket,
        ):
            browser_tool.cleanup_browser("task-1")

        terminate.assert_called_once_with(12345, expected_start=777)
        remove_socket.assert_called_once_with(str(socket_dir), "sess-1")

    def test_explicit_session_preserves_metadata_when_daemon_unverifiable(self, tmp_path):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        socket_dir = tmp_path / "agent-browser-sess-1"
        socket_dir.mkdir()
        (socket_dir / "sess-1.pid").write_text("12345")

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}),
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("gateway.status._pid_exists", return_value=True),
            patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=None),
            patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as terminate,
            patch("tools.browser_tool._reap_session_chromium"),
            patch("tools.browser_tool._remove_browser_socket_dir_if_safe") as remove_socket,
        ):
            browser_tool.cleanup_browser("task-1")

        terminate.assert_not_called()
        remove_socket.assert_not_called()
        assert socket_dir.exists()

    def test_explicit_session_preserves_metadata_when_daemon_survives(self, tmp_path):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        socket_dir = tmp_path / "agent-browser-sess-1"
        socket_dir.mkdir()
        (socket_dir / "sess-1.pid").write_text("12345")

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}),
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("gateway.status._pid_exists", side_effect=[True, True]),
            patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=777),
            patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as terminate,
            patch("tools.browser_tool._reap_session_chromium"),
            patch("tools.browser_tool._remove_browser_socket_dir_if_safe") as remove_socket,
        ):
            browser_tool.cleanup_browser("task-1")

        terminate.assert_called_once_with(12345, expected_start=777)
        remove_socket.assert_not_called()
        assert socket_dir.exists()

    def test_cleanup_camofox_managed_persistence_skips_close(self):
        """When camofox mode + managed persistence, soft_cleanup fires instead of close."""
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-1"] = 123.0

        with (
            patch("tools.browser_tool._is_camofox_mode", return_value=True),
            patch("tools.browser_tool._maybe_stop_recording") as mock_stop,
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ),
            patch("tools.browser_tool.os.path.exists", return_value=False),
            patch(
                "tools.browser_camofox.camofox_soft_cleanup",
                return_value=True,
            ) as mock_soft,
            patch("tools.browser_camofox.camofox_close") as mock_close,
        ):
            browser_tool.cleanup_browser("task-1")

        mock_soft.assert_called_once_with("task-1")
        mock_close.assert_not_called()

    def test_cleanup_camofox_no_persistence_calls_close(self):
        """When camofox mode but managed persistence is off, camofox_close fires."""
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-1"] = 123.0

        with (
            patch("tools.browser_tool._is_camofox_mode", return_value=True),
            patch("tools.browser_tool._maybe_stop_recording") as mock_stop,
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ),
            patch("tools.browser_tool.os.path.exists", return_value=False),
            patch(
                "tools.browser_camofox.camofox_soft_cleanup",
                return_value=False,
            ) as mock_soft,
            patch("tools.browser_camofox.camofox_close") as mock_close,
        ):
            browser_tool.cleanup_browser("task-1")

        mock_soft.assert_called_once_with("task-1")
        mock_close.assert_called_once_with("task-1")

    def test_emergency_cleanup_clears_all_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._cleanup_done = False
        browser_tool._active_sessions["task-1"] = {"session_name": "sess-1"}
        browser_tool._active_sessions["task-2"] = {"session_name": "sess-2"}
        browser_tool._session_last_activity["task-1"] = 1.0
        browser_tool._session_last_activity["task-2"] = 2.0
        browser_tool._recording_sessions.update({"task-1", "task-2"})

        with (
            patch("tools.browser_tool.cleanup_all_browsers") as mock_cleanup_all,
            patch("tools.browser_tool._reap_orphaned_browser_chromes") as mock_chrome_reaper,
            patch("tools.browser_tool._reap_orphaned_browser_sessions") as mock_daemon_reaper,
        ):
            browser_tool._emergency_cleanup_all_sessions()

        mock_cleanup_all.assert_called_once_with()
        mock_chrome_reaper.assert_called_once_with(
            min_age_seconds=browser_tool.BROWSER_SESSION_INACTIVITY_TIMEOUT
        )
        mock_daemon_reaper.assert_called_once_with()
        assert browser_tool._active_sessions == {}
        assert browser_tool._session_last_activity == {}
        assert browser_tool._recording_sessions == set()
        assert browser_tool._cleanup_done is True
