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

        with patch("tools.browser_tool.cleanup_all_browsers") as mock_cleanup_all:
            browser_tool._emergency_cleanup_all_sessions()

        mock_cleanup_all.assert_called_once_with()
        assert browser_tool._active_sessions == {}
        assert browser_tool._session_last_activity == {}
        assert browser_tool._recording_sessions == set()


class TestNavigateFailureClearsActivity:
    """Regression for the activity-timestamp-on-failed-open bug.

    `_get_session_info` calls `_update_session_activity(task_id)` *before*
    the actual open command runs. If the open command then blows up
    (timeout, refused connection, agent-browser daemon crash, etc.), the
    session's last-activity timestamp gets refreshed anyway, so the
    background inactivity-cleanup thread sees the session as "freshly
    used" forever and never reaps it. On a long-running gateway that
    leaves orphaned Chromium processes behind (see issue #32047).

    The contract: after a failed browser_navigate, the session's
    last-activity timestamp must NOT be present (or must be older than
    the inactivity timeout), so the cleanup thread can reap it on its
    next tick.
    """

    def setup_method(self):
        from tools import browser_tool

        self.browser_tool = browser_tool
        self.orig_active_sessions = browser_tool._active_sessions.copy()
        self.orig_session_last_activity = browser_tool._session_last_activity.copy()

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)

    def test_navigate_timeout_clears_activity_timestamp(self):
        """Open command raises -> _session_last_activity must be popped."""
        browser_tool = self.browser_tool
        nav_key = "default"  # bare task_id when no cloud provider

        def fake_get_session_info(task_id):
            # Simulate _get_session_info's documented side effect: it
            # stamps the activity timestamp on the way in.
            browser_tool._update_session_activity(task_id)
            return {
                "session_name": "fake-sess",
                "_first_nav": True,
            }

        with (
            patch(
                "tools.browser_tool._get_session_info",
                side_effect=fake_get_session_info,
            ),
            patch(
                "tools.browser_tool._get_open_command_timeout",
                return_value=120,
            ),
            patch(
                "tools.browser_tool._maybe_start_recording",
            ),
            patch(
                "tools.browser_tool._run_browser_command",
                side_effect=TimeoutError("agent-browser open timed out"),
            ),
        ):
            import pytest

            with pytest.raises(TimeoutError):
                browser_tool.browser_navigate("https://example.com")

        # The whole point: nav_key must NOT have an activity timestamp
        # after a failed open, so the reaper can clean it up.
        assert nav_key not in browser_tool._session_last_activity, (
            "browser_navigate failure must clear _session_last_activity so "
            "the inactivity reaper can reap the orphaned session. "
            f"Found: {browser_tool._session_last_activity!r}"
        )

    def test_navigate_success_refreshes_activity_timestamp(self):
        """Open command succeeds -> activity timestamp must be recent.

        Guards against the opposite regression: if the success path
        stops updating the timestamp (or updates it to a stale value),
        a healthy session gets reaped while still in use.
        """
        import time

        browser_tool = self.browser_tool
        nav_key = "default"  # bare task_id when no cloud provider
        before = time.time()

        def fake_get_session_info(task_id):
            browser_tool._update_session_activity(task_id)
            return {
                "session_name": "fake-sess",
                "_first_nav": True,
            }

        with (
            patch(
                "tools.browser_tool._get_session_info",
                side_effect=fake_get_session_info,
            ),
            patch(
                "tools.browser_tool._get_open_command_timeout",
                return_value=120,
            ),
            patch(
                "tools.browser_tool._maybe_start_recording",
            ),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True, "data": {"title": "x", "url": "https://example.com"}},
            ),
            # Bypass the heavy post-success branches (SSRF, bot-detection
            # heuristics, snapshot, etc.) — we only care about activity
            # bookkeeping here. Patch returns nothing to keep the response
            # dict minimal.
            patch("tools.browser_tool._copy_fallback_warning"),
        ):
            # Call through the real function; mock the side-effect-heavy
            # branches below.
            browser_tool.browser_navigate("https://example.com")

        # We can't assert the exact timestamp (it depends on whether
        # _get_session_info's update was overwritten by the new
        # success-path call, or vice versa). We can only assert it's
        # recent AND the session is tracked for reaping.
        assert nav_key in browser_tool._session_last_activity
        assert browser_tool._session_last_activity[nav_key] >= before
