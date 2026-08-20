"""Regression tests for browser session cleanup and screenshot recovery."""

from pathlib import Path
from unittest.mock import patch

from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


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
        self.orig_session_profile_homes = browser_tool._session_profile_homes.copy()
        self.orig_cleanup_failure_counts = browser_tool._cleanup_failure_counts.copy()
        self.orig_recording_sessions = browser_tool._recording_sessions.copy()
        self.orig_cleanup_done = browser_tool._cleanup_done

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._session_profile_homes.clear()
        self.browser_tool._session_profile_homes.update(self.orig_session_profile_homes)
        self.browser_tool._cleanup_failure_counts.clear()
        self.browser_tool._cleanup_failure_counts.update(
            self.orig_cleanup_failure_counts
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
        browser_tool._session_profile_homes["task-1"] = "/profile-a"
        browser_tool._cleanup_failure_counts["task-1"] = 2

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
        assert "task-1" not in browser_tool._session_profile_homes
        assert "task-1" not in browser_tool._cleanup_failure_counts
        mock_stop.assert_called_once_with("task-1")
        mock_run.assert_called_once_with("task-1", "close", [], timeout=10)


    def test_emergency_cleanup_clears_all_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._cleanup_done = False
        browser_tool._active_sessions["task-1"] = {"session_name": "sess-1"}
        browser_tool._active_sessions["task-2"] = {"session_name": "sess-2"}
        browser_tool._session_last_activity["task-1"] = 1.0
        browser_tool._session_last_activity["task-2"] = 2.0
        browser_tool._session_profile_homes.update(
            {"task-1": "/profile-a", "task-2": "/profile-b"}
        )
        browser_tool._cleanup_failure_counts.update({"task-1": 1, "task-2": 2})
        browser_tool._recording_sessions.update({"task-1", "task-2"})

        with patch("tools.browser_tool.cleanup_all_browsers") as mock_cleanup_all:
            browser_tool._emergency_cleanup_all_sessions()

        mock_cleanup_all.assert_called_once_with()
        assert browser_tool._active_sessions == {}
        assert browser_tool._session_last_activity == {}
        assert browser_tool._session_profile_homes == {}
        assert browser_tool._cleanup_failure_counts == {}
        assert browser_tool._recording_sessions == set()
        assert browser_tool._cleanup_done is True

    def test_inactivity_cleanup_restores_each_sessions_profile_scope(
        self, tmp_path, monkeypatch
    ):
        browser_tool = self.browser_tool
        profile_a = tmp_path / "profile-a"
        profile_b = tmp_path / "profile-b"
        profile_a.mkdir()
        profile_b.mkdir()
        (profile_a / ".env").write_text("CAMOFOX_URL=https://a.example\n")
        (profile_b / ".env").write_text("CAMOFOX_URL=https://b.example\n")

        monkeypatch.setattr(browser_tool, "_session_profile_homes", {}, raising=False)
        monkeypatch.setattr(browser_tool, "_cleanup_failure_counts", {}, raising=False)
        monkeypatch.setattr(browser_tool, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 30)
        monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)

        secret_scope.set_multiplex_active(True)
        try:
            for task_id, profile_home in (
                ("task-a", profile_a),
                ("task-b", profile_b),
            ):
                home_token = set_hermes_home_override(str(profile_home))
                scope_token = secret_scope.set_secret_scope(
                    secret_scope.build_profile_secret_scope(profile_home)
                )
                try:
                    browser_tool._update_session_activity(task_id)
                    browser_tool._session_last_activity[task_id] = 0.0
                finally:
                    secret_scope.reset_secret_scope(scope_token)
                    reset_hermes_home_override(home_token)

            cleaned = []

            def cleanup_in_owning_scope(task_id):
                cleaned.append(
                    (
                        task_id,
                        Path(browser_tool.get_hermes_home()),
                        secret_scope.get_secret("CAMOFOX_URL"),
                    )
                )

            monkeypatch.setattr(browser_tool, "cleanup_browser", cleanup_in_owning_scope)
            browser_tool._cleanup_inactive_browser_sessions()
        finally:
            secret_scope.set_multiplex_active(False)

        assert cleaned == [
            ("task-a", profile_a, "https://a.example"),
            ("task-b", profile_b, "https://b.example"),
        ]
        assert browser_tool._session_last_activity == {}

    def test_inactivity_cleanup_force_reaps_after_three_failures(self, monkeypatch):
        browser_tool = self.browser_tool
        monkeypatch.setattr(browser_tool, "_session_profile_homes", {}, raising=False)
        monkeypatch.setattr(browser_tool, "_cleanup_failure_counts", {}, raising=False)
        monkeypatch.setattr(browser_tool, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 30)
        monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)
        browser_tool._session_last_activity["task-1"] = 0.0

        def fail_cleanup(_task_id):
            raise RuntimeError("persistent teardown failure")

        force_reaped = []

        def force_reap(task_id):
            force_reaped.append(task_id)
            browser_tool._session_last_activity.pop(task_id, None)

        monkeypatch.setattr(browser_tool, "cleanup_browser", fail_cleanup)
        monkeypatch.setattr(
            browser_tool, "_force_reap_browser_session", force_reap, raising=False
        )

        browser_tool._cleanup_inactive_browser_sessions()
        browser_tool._cleanup_inactive_browser_sessions()
        assert force_reaped == []
        assert browser_tool._session_last_activity == {"task-1": 0.0}

        browser_tool._cleanup_inactive_browser_sessions()

        assert force_reaped == ["task-1"]
        assert browser_tool._cleanup_failure_counts == {}

    def test_session_activity_resets_cleanup_failure_budget(self, monkeypatch):
        browser_tool = self.browser_tool
        monkeypatch.setattr(browser_tool.time, "time", lambda: 200.0)
        browser_tool._cleanup_failure_counts["task-1"] = 2

        browser_tool._update_session_activity("task-1")

        assert browser_tool._session_last_activity["task-1"] == 200.0
        assert "task-1" not in browser_tool._cleanup_failure_counts
