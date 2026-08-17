"""Regression tests for browser session cleanup and screenshot recovery."""

import json
import time

from unittest.mock import MagicMock, patch


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
        self.orig_retired_browser_tasks = getattr(
            browser_tool, "_retired_browser_tasks", set()
        ).copy()
        self.orig_task_states = browser_tool._browser_task_states.copy()
        self.orig_task_generations = browser_tool._browser_task_generations.copy()
        self.orig_cleanup_reasons = browser_tool._browser_task_cleanup_reasons.copy()
        self.orig_provider_cleanups = browser_tool._pending_provider_cleanups.copy()
        self.orig_recording_sessions = browser_tool._recording_sessions.copy()
        self.orig_cleanup_done = browser_tool._cleanup_done
        self.close_target_patch = patch(
            "tools.browser_tool._close_shared_cdp_target_confirmed",
            return_value=True,
        )
        self.close_target_mock = self.close_target_patch.start()

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._last_active_session_key.clear()
        self.browser_tool._last_active_session_key.update(
            self.orig_last_active_session_key
        )
        if hasattr(self.browser_tool, "_retired_browser_tasks"):
            self.browser_tool._retired_browser_tasks.clear()
            self.browser_tool._retired_browser_tasks.update(
                self.orig_retired_browser_tasks
            )
        self.browser_tool._browser_task_states.clear()
        self.browser_tool._browser_task_states.update(self.orig_task_states)
        self.browser_tool._browser_task_generations.clear()
        self.browser_tool._browser_task_generations.update(self.orig_task_generations)
        self.browser_tool._browser_task_cleanup_reasons.clear()
        self.browser_tool._browser_task_cleanup_reasons.update(
            self.orig_cleanup_reasons
        )
        self.browser_tool._pending_provider_cleanups.clear()
        self.browser_tool._pending_provider_cleanups.update(
            self.orig_provider_cleanups
        )
        self.browser_tool._recording_sessions.clear()
        self.browser_tool._recording_sessions.update(self.orig_recording_sessions)
        self.browser_tool._cleanup_done = self.orig_cleanup_done
        self.close_target_patch.stop()

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
        mock_run.assert_called_once_with(
            "task-1", "close", [], timeout=10, _allow_cleanup=True
        )

    def test_cleanup_shared_cdp_closes_pinned_tab_before_session(self):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-cdp"] = {
            "session_name": "sess-cdp",
            "bb_session_id": None,
            "cdp_url": "ws://127.0.0.1:9222/devtools/browser/shared",
            "target_id": "TARGET-OWNED",
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
            (("task-cdp", "close", []), {"timeout": 10, "_allow_cleanup": True}),
        ]

    def test_cleanup_shared_cdp_accepts_already_gone_tab(self):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-gone"] = {
            "session_name": "sess-gone",
            "bb_session_id": None,
            "cdp_url": "ws://shared",
            "target_id": "TARGET-OWNED",
        }
        browser_tool._session_last_activity["task-gone"] = 123.0

        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            cleaned = browser_tool.cleanup_browser("task-gone")

        assert cleaned is True
        assert "task-gone" not in browser_tool._active_sessions
        assert "task-gone" not in browser_tool._session_last_activity
        assert mock_run.call_args_list == [
            (("task-gone", "close", []), {"timeout": 10, "_allow_cleanup": True}),
        ]

    def test_cleanup_shared_cdp_transient_failure_retains_ownership(self):
        browser_tool = self.browser_tool
        session = {
            "session_name": "sess-retry",
            "bb_session_id": None,
            "cdp_url": "ws://shared",
            "target_id": "TARGET-OWNED",
        }
        browser_tool._active_sessions["task-retry"] = session
        browser_tool._session_last_activity["task-retry"] = 123.0
        browser_tool._last_active_session_key["task-retry"] = "task-retry"

        self.close_target_mock.return_value = False
        cleaned = browser_tool.cleanup_browser("task-retry")

        assert cleaned is False
        assert browser_tool._active_sessions["task-retry"] is session
        assert browser_tool._session_last_activity["task-retry"] == 123.0
        assert browser_tool._last_active_session_key["task-retry"] == "task-retry"
        assert "task-retry" not in browser_tool._retired_browser_tasks
        assert session["_cleanup_retry_pending"] is True
        self.close_target_mock.return_value = True

    def test_cleanup_shared_cdp_retries_then_becomes_idempotent(self):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-repeat"] = {
            "session_name": "sess-repeat",
            "bb_session_id": None,
            "cdp_url": "ws://shared",
            "target_id": "TARGET-OWNED",
        }
        browser_tool._session_last_activity["task-repeat"] = 123.0

        self.close_target_mock.side_effect = [False, True]
        with (
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            assert browser_tool.cleanup_browser("task-repeat") is False
            assert "task-repeat" in browser_tool._active_sessions
            assert browser_tool.cleanup_browser("task-repeat") is True
            assert browser_tool.cleanup_browser("task-repeat") is True

        assert "task-repeat" not in browser_tool._active_sessions
        assert mock_run.call_args_list == [
            (("task-repeat", "close", []), {"timeout": 10, "_allow_cleanup": True}),
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


def test_inactivity_reaper_spares_task_owned_shared_cdp(monkeypatch):
    from tools import browser_tool

    now = time.time()
    task_id = "task-shared-cdp"
    monkeypatch.setattr(
        browser_tool,
        "_active_sessions",
        {
            task_id: {
                "session_name": "cdp_owned",
                "bb_session_id": None,
                "cdp_url": "ws://127.0.0.1:9222/devtools/browser/shared",
                "target_id": "TARGET-OWNED",
                "features": {"cdp_override": True},
            }
        },
    )
    monkeypatch.setattr(
        browser_tool,
        "_session_last_activity",
        {task_id: now - browser_tool.BROWSER_SESSION_INACTIVITY_TIMEOUT - 1},
    )
    cleanup = MagicMock(return_value=True)
    monkeypatch.setattr(browser_tool, "cleanup_browser", cleanup)
    monkeypatch.setattr(browser_tool.time, "time", lambda: now)

    browser_tool._cleanup_inactive_browser_sessions()

    cleanup.assert_not_called()
    assert task_id in browser_tool._active_sessions
    assert task_id in browser_tool._session_last_activity

    # Once terminal cleanup has been requested and failed, the retained owner
    # must remain eligible for a later background retry.
    browser_tool._active_sessions[task_id]["_cleanup_retry_pending"] = True
    cleanup.reset_mock()

    browser_tool._cleanup_inactive_browser_sessions()

    cleanup.assert_called_once_with(
        task_id,
        reason=browser_tool.BrowserCleanupReason.INACTIVITY,
    )


def test_inactivity_reaper_still_cleans_local_session(monkeypatch):
    from tools import browser_tool

    now = time.time()
    task_id = "task-local"
    monkeypatch.setattr(
        browser_tool,
        "_active_sessions",
        {
            task_id: {
                "session_name": "local_owned",
                "bb_session_id": None,
                "cdp_url": None,
                "features": {"local": True},
            }
        },
    )
    monkeypatch.setattr(
        browser_tool,
        "_session_last_activity",
        {task_id: now - browser_tool.BROWSER_SESSION_INACTIVITY_TIMEOUT - 1},
    )
    cleanup = MagicMock(return_value=True)
    monkeypatch.setattr(browser_tool, "cleanup_browser", cleanup)
    monkeypatch.setattr(browser_tool.time, "time", lambda: now)

    browser_tool._cleanup_inactive_browser_sessions()

    cleanup.assert_called_once_with(
        task_id,
        reason=browser_tool.BrowserCleanupReason.INACTIVITY,
    )
    assert task_id not in browser_tool._session_last_activity


def test_successful_cleanup_tombstones_non_navigation_without_recreation(
    monkeypatch,
):
    from tools import browser_tool

    task_id = "task-terminal"
    monkeypatch.setattr(
        browser_tool,
        "_close_shared_cdp_target_confirmed",
        lambda _cdp_url, _target_id: True,
    )
    monkeypatch.setattr(
        browser_tool,
        "_active_sessions",
        {
            task_id: {
                "session_name": "cdp_terminal",
                "bb_session_id": None,
                "cdp_url": "ws://shared",
                "target_id": "TARGET-OWNED",
                "features": {"cdp_override": True},
            }
        },
    )
    monkeypatch.setattr(browser_tool, "_session_last_activity", {task_id: 1.0})
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {task_id: task_id})
    monkeypatch.setattr(browser_tool, "_retired_browser_tasks", set(), raising=False)
    monkeypatch.setattr(browser_tool, "_maybe_stop_recording", lambda _task: None)
    monkeypatch.setattr(browser_tool.os.path, "exists", lambda _path: False)
    with patch(
        "tools.browser_tool._run_browser_command",
        return_value={"success": True},
    ):
        assert browser_tool.cleanup_browser(task_id) is True
    assert task_id in browser_tool._retired_browser_tasks

    get_session = MagicMock(
        return_value={
            "session_name": "replacement",
            "bb_session_id": None,
            "cdp_url": "ws://shared",
            "target_id": "TARGET-OWNED",
        }
    )
    find_browser = MagicMock(return_value="/usr/bin/agent-browser")
    popen = MagicMock()
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_session_info", get_session)
    monkeypatch.setattr(browser_tool, "_find_agent_browser", find_browser)
    monkeypatch.setattr(browser_tool.subprocess, "Popen", popen)

    result = json.loads(browser_tool.browser_snapshot(task_id=task_id))

    assert result["success"] is False
    assert result["code"] == "browser_session_retired"
    assert result["data"]["terminal"] is True
    get_session.assert_not_called()
    find_browser.assert_not_called()
    popen.assert_not_called()


def test_retired_eval_does_not_use_stale_supervisor_or_subprocess(monkeypatch):
    from tools import browser_tool
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    task_id = "task-retired-eval"
    monkeypatch.setattr(browser_tool, "_retired_browser_tasks", {task_id})
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    supervisor_get = MagicMock()
    run_command = MagicMock()
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "get", supervisor_get)
    monkeypatch.setattr(browser_tool, "_run_browser_command", run_command)

    result = json.loads(
        browser_tool.browser_console(expression="1 + 1", task_id=task_id)
    )

    assert result["code"] == "browser_session_retired"
    assert result["data"]["terminal"] is True
    supervisor_get.assert_not_called()
    run_command.assert_not_called()
