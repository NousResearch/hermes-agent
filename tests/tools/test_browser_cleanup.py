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
        self.orig_shared_cdp_sessions = getattr(browser_tool, "_shared_cdp_sessions", {}).copy()
        self.orig_session_command_locks = getattr(browser_tool, "_session_command_locks", {}).copy()

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._recording_sessions.clear()
        self.browser_tool._recording_sessions.update(self.orig_recording_sessions)
        self.browser_tool._cleanup_done = self.orig_cleanup_done
        if hasattr(self.browser_tool, "_shared_cdp_sessions"):
            self.browser_tool._shared_cdp_sessions.clear()
            self.browser_tool._shared_cdp_sessions.update(self.orig_shared_cdp_sessions)
        if hasattr(self.browser_tool, "_session_command_locks"):
            self.browser_tool._session_command_locks.clear()
            self.browser_tool._session_command_locks.update(self.orig_session_command_locks)

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

    def test_cleanup_browser_falls_back_to_legacy_default_pid_file(self):
        """Legacy CDP sessions wrote default.pid; cleanup should still terminate them."""
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "cdp_abcd1234",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-1"] = 123.0

        def fake_isfile(path):
            return path.endswith("default.pid")

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}),
            patch("tools.browser_tool.os.path.exists", return_value=True),
            patch("tools.browser_tool.os.path.isfile", side_effect=fake_isfile),
            patch("tools.browser_tool.Path.read_text", return_value="4242"),
            patch("tools.browser_tool.os.kill") as mock_kill,
            patch("tools.browser_tool.shutil.rmtree") as mock_rmtree,
            patch("tools.browser_tool._socket_safe_tmpdir", return_value="/tmp"),
        ):
            browser_tool.cleanup_browser("task-1")

        mock_kill.assert_called_once_with(4242, browser_tool.signal.SIGTERM)
        mock_rmtree.assert_called_once_with("/tmp/agent-browser-cdp_abcd1234", ignore_errors=True)

    def test_cleanup_browser_shared_cdp_non_last_task_only_detaches(self):
        browser_tool = self.browser_tool
        shared_key = "browserless:/data/hermes-profiles/pasclaw"
        shared_session = {
            "session_name": "cdp_shared_session",
            "bb_session_id": None,
            "cdp_url": "wss://browserless.example/chromium?token=abc",
            "_shared_cdp_key": shared_key,
        }
        browser_tool._active_sessions["task-1"] = shared_session
        browser_tool._active_sessions["task-2"] = shared_session
        browser_tool._session_last_activity["task-1"] = 100.0
        browser_tool._session_last_activity["task-2"] = 200.0
        browser_tool._shared_cdp_sessions = {
            shared_key: {
                "session_info": shared_session,
                "task_ids": {"task-1", "task-2"},
            }
        }
        browser_tool._session_command_locks = {}

        with (
            patch("tools.browser_tool._maybe_stop_recording") as mock_stop,
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
            patch("tools.browser_tool.shutil.rmtree") as mock_rmtree,
            patch("tools.browser_tool.os.kill") as mock_kill,
        ):
            browser_tool.cleanup_browser("task-1")

        assert "task-1" not in browser_tool._active_sessions
        assert "task-1" not in browser_tool._session_last_activity
        assert browser_tool._active_sessions["task-2"] is shared_session
        assert browser_tool._shared_cdp_sessions[shared_key]["task_ids"] == {"task-2"}
        mock_stop.assert_not_called()
        mock_run.assert_not_called()
        mock_rmtree.assert_not_called()
        mock_kill.assert_not_called()

    def test_cleanup_inactive_shared_cdp_stale_alias_does_not_close_active_alias(self):
        browser_tool = self.browser_tool
        shared_key = "browserless:/data/hermes-profiles/pasclaw"
        shared_session = {
            "session_name": "cdp_shared_session",
            "bb_session_id": None,
            "cdp_url": "wss://browserless.example/chromium?token=abc",
            "_shared_cdp_key": shared_key,
        }
        browser_tool._active_sessions["stale-task"] = shared_session
        browser_tool._active_sessions["active-task"] = shared_session
        browser_tool._session_last_activity["stale-task"] = 0.0
        browser_tool._session_last_activity["active-task"] = 999.0
        browser_tool._shared_cdp_sessions = {
            shared_key: {
                "session_info": shared_session,
                "task_ids": {"stale-task", "active-task"},
            }
        }
        browser_tool._session_command_locks = {}

        with (
            patch("tools.browser_tool.BROWSER_SESSION_INACTIVITY_TIMEOUT", 300),
            patch("tools.browser_tool.time.time", return_value=1000.0),
            patch("tools.browser_tool._maybe_stop_recording") as mock_stop,
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
            patch("tools.browser_tool.shutil.rmtree") as mock_rmtree,
            patch("tools.browser_tool.os.kill") as mock_kill,
        ):
            browser_tool._cleanup_inactive_browser_sessions()

        assert "stale-task" not in browser_tool._active_sessions
        assert "stale-task" not in browser_tool._session_last_activity
        assert browser_tool._active_sessions["active-task"] is shared_session
        assert browser_tool._shared_cdp_sessions[shared_key]["task_ids"] == {"active-task"}
        mock_stop.assert_not_called()
        mock_run.assert_not_called()
        mock_rmtree.assert_not_called()
        mock_kill.assert_not_called()

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
