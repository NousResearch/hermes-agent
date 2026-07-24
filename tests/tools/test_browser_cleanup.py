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
        self.orig_cloud_cleanup_states = browser_tool._cloud_cleanup_states.copy()
        self.orig_cloud_cleanup_pending = browser_tool._cloud_cleanup_pending.copy()
        self.orig_cloud_cleanup_in_progress = browser_tool._cloud_cleanup_in_progress.copy()

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._recording_sessions.clear()
        self.browser_tool._recording_sessions.update(self.orig_recording_sessions)
        self.browser_tool._cleanup_done = self.orig_cleanup_done
        self.browser_tool._cloud_cleanup_states.clear()
        self.browser_tool._cloud_cleanup_states.update(self.orig_cloud_cleanup_states)
        self.browser_tool._cloud_cleanup_pending.clear()
        self.browser_tool._cloud_cleanup_pending.update(self.orig_cloud_cleanup_pending)
        self.browser_tool._cloud_cleanup_in_progress.clear()
        self.browser_tool._cloud_cleanup_in_progress.update(self.orig_cloud_cleanup_in_progress)

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
        mock_run.assert_called_once_with(
            "task-1",
            "close",
            [],
            timeout=10,
            _allow_cleanup_pending=True,
        )

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

    def test_failed_remote_close_is_retried_without_reusing_local_session(self):
        browser_tool = self.browser_tool

        class Provider:
            attempts = 0

            def close_session(self, _session_id):
                self.attempts += 1
                return self.attempts > 1

        provider = Provider()
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": "remote-1",
            "_cloud_cleanup_key": "cleanup-1",
        }
        browser_tool._session_last_activity["task-1"] = 123.0
        browser_tool._cloud_cleanup_states["cleanup-1"] = (
            provider,
            "remote-1",
            None,
            "/owner",
        )

        with (
            patch("tools.browser_tool._run_browser_command", return_value={}),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            browser_tool.cleanup_browser("task-1")
            assert "task-1" not in browser_tool._active_sessions
            assert "cleanup-1" in browser_tool._cloud_cleanup_pending
            browser_tool._retry_pending_cloud_cleanups()

        assert provider.attempts == 2
        assert "cleanup-1" not in browser_tool._cloud_cleanup_states
        assert "cleanup-1" not in browser_tool._cloud_cleanup_pending

    def test_emergency_cleanup_clears_all_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._cleanup_done = False
        browser_tool._active_sessions["task-1"] = {"session_name": "sess-1"}
        browser_tool._active_sessions["task-2"] = {"session_name": "sess-2"}
        browser_tool._session_last_activity["task-1"] = 1.0
        browser_tool._session_last_activity["task-2"] = 2.0
        browser_tool._recording_sessions.update({"task-1", "task-2"})

        with patch("tools.browser_tool.cleanup_all_browsers") as mock_cleanup_all:
            browser_tool._emergency_cleanup_all_sessions()

        mock_cleanup_all.assert_called_once_with()
        assert browser_tool._active_sessions == {}
        assert browser_tool._session_last_activity == {}
        assert browser_tool._recording_sessions == set()
        assert browser_tool._cleanup_done is True
