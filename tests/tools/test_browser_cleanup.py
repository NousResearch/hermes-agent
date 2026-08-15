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
        self.orig_last_active_session_key = browser_tool._last_active_session_key.copy()
        self.orig_recording_sessions = browser_tool._recording_sessions.copy()
        self.orig_cleanup_done = browser_tool._cleanup_done

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._last_active_session_key.clear()
        self.browser_tool._last_active_session_key.update(
            self.orig_last_active_session_key
        )
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
            cleaned = browser_tool.cleanup_browser("task-1")

        assert cleaned is True
        assert "task-1" not in browser_tool._active_sessions
        assert "task-1" not in browser_tool._session_last_activity
        mock_stop.assert_called_once_with("task-1")
        mock_run.assert_called_once_with("task-1", "close", [], timeout=10)

    def test_cleanup_shared_cdp_closes_pinned_tab_before_session(self):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-cdp"] = {
            "session_name": "sess-cdp",
            "bb_session_id": None,
            "cdp_url": "ws://127.0.0.1:9222/devtools/browser/shared",
        }

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            cleaned = browser_tool.cleanup_browser("task-cdp")

        assert cleaned is True
        assert "task-cdp" not in browser_tool._active_sessions
        assert mock_run.call_args_list == [
            (("task-cdp", "tab", ["close"]), {"timeout": 10}),
            (("task-cdp", "close", []), {"timeout": 10}),
        ]

    def test_cleanup_shared_cdp_accepts_already_gone_tab(self):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-gone"] = {
            "session_name": "sess-gone",
            "bb_session_id": None,
            "cdp_url": "ws://shared",
        }
        browser_tool._session_last_activity["task-gone"] = 123.0

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                side_effect=[
                    {
                        "success": False,
                        "code": "tab_gone",
                        "error": "Pinned tab is no longer available",
                    },
                    {"success": True},
                ],
            ) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            cleaned = browser_tool.cleanup_browser("task-gone")

        assert cleaned is True
        assert "task-gone" not in browser_tool._active_sessions
        assert "task-gone" not in browser_tool._session_last_activity
        assert mock_run.call_args_list == [
            (("task-gone", "tab", ["close"]), {"timeout": 10}),
            (("task-gone", "close", []), {"timeout": 10}),
        ]

    def test_cleanup_shared_cdp_transient_failure_retains_ownership(self):
        browser_tool = self.browser_tool
        session = {
            "session_name": "sess-retry",
            "bb_session_id": None,
            "cdp_url": "ws://shared",
        }
        browser_tool._active_sessions["task-retry"] = session
        browser_tool._session_last_activity["task-retry"] = 123.0
        browser_tool._last_active_session_key["task-retry"] = "task-retry"

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": False, "error": "temporary disconnect"},
            ) as mock_run,
        ):
            cleaned = browser_tool.cleanup_browser("task-retry")

        assert cleaned is False
        assert browser_tool._active_sessions["task-retry"] is session
        assert browser_tool._session_last_activity["task-retry"] == 123.0
        assert browser_tool._last_active_session_key["task-retry"] == "task-retry"
        mock_run.assert_called_once_with(
            "task-retry", "tab", ["close"], timeout=10
        )

    def test_cleanup_shared_cdp_retries_then_becomes_idempotent(self):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-repeat"] = {
            "session_name": "sess-repeat",
            "bb_session_id": None,
            "cdp_url": "ws://shared",
        }
        browser_tool._session_last_activity["task-repeat"] = 123.0

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                side_effect=[
                    {"success": False, "error": "temporary disconnect"},
                    {"success": True},
                    {"success": True},
                ],
            ) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            assert browser_tool.cleanup_browser("task-repeat") is False
            assert "task-repeat" in browser_tool._active_sessions
            assert browser_tool.cleanup_browser("task-repeat") is True
            assert browser_tool.cleanup_browser("task-repeat") is True

        assert "task-repeat" not in browser_tool._active_sessions
        assert mock_run.call_args_list == [
            (("task-repeat", "tab", ["close"]), {"timeout": 10}),
            (("task-repeat", "tab", ["close"]), {"timeout": 10}),
            (("task-repeat", "close", []), {"timeout": 10}),
        ]

    def test_successful_sidecar_cleanup_preserves_primary_binding(self):
        browser_tool = self.browser_tool
        sidecar = "task-sidecar::local"
        browser_tool._active_sessions[sidecar] = {
            "session_name": "sess-sidecar",
            "bb_session_id": None,
        }
        browser_tool._last_active_session_key["task-sidecar"] = "task-sidecar"

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ),
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            assert browser_tool.cleanup_browser(sidecar) is True

        assert browser_tool._last_active_session_key["task-sidecar"] == "task-sidecar"


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
