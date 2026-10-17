"""Regression tests for browser session cleanup and screenshot recovery."""

from unittest.mock import patch
from tools import browser_tool_lifecycle as bt_lifecycle


class TestScreenshotPathRecovery:
    def test_extracts_standard_absolute_path(self):
        from tools.browser_tool_snapshot import _extract_screenshot_path_from_text

        assert (
            _extract_screenshot_path_from_text("Screenshot saved to /tmp/foo.png")
            == "/tmp/foo.png"
        )

    def test_extracts_quoted_absolute_path(self):
        from tools.browser_tool_snapshot import _extract_screenshot_path_from_text

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
                "tools.browser_tool_session._run_browser_command",
                return_value={"success": True},
            ) as mock_run,
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            bt_lifecycle.cleanup_browser("task-1")

        assert "task-1" not in browser_tool._active_sessions
        assert "task-1" not in browser_tool._session_last_activity
        mock_stop.assert_called_once_with("task-1")
        assert mock_run.call_args.args[:2] == ("task-1", "close")


    def test_emergency_cleanup_clears_all_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._cleanup_done = False
        browser_tool._active_sessions["task-1"] = {"session_name": "sess-1"}
        browser_tool._active_sessions["task-2"] = {"session_name": "sess-2"}
        browser_tool._session_last_activity["task-1"] = 1.0
        browser_tool._session_last_activity["task-2"] = 2.0
        browser_tool._recording_sessions.update({"task-1", "task-2"})

        with patch("tools.browser_tool_lifecycle.cleanup_all_browsers") as mock_cleanup_all:
            bt_lifecycle._emergency_cleanup_all_sessions()

        mock_cleanup_all.assert_called_once_with()
        assert browser_tool._active_sessions == {}
        assert browser_tool._session_last_activity == {}
        assert browser_tool._recording_sessions == set()
        assert browser_tool._cleanup_done is True


class TestInactivityJanitorMultiplex:
    """#86402 / #100738: the process-global janitor thread has no profile scope."""

    def setup_method(self):
        from agent import secret_scope
        from tools import browser_tool

        self.bt = browser_tool
        self.saved = {
            name: getattr(browser_tool, name).copy()
            for name in (
                "_active_sessions", "_session_last_activity",
                "_session_owner_homes", "_cleanup_failures", "_recording_sessions",
            )
        }
        self.orig_timeout = browser_tool.BROWSER_SESSION_INACTIVITY_TIMEOUT
        browser_tool.BROWSER_SESSION_INACTIVITY_TIMEOUT = 0
        for name in self.saved:
            getattr(browser_tool, name).clear()
        secret_scope.set_multiplex_active(True)

    def teardown_method(self):
        from agent import secret_scope

        secret_scope.set_multiplex_active(False)
        self.bt.BROWSER_SESSION_INACTIVITY_TIMEOUT = self.orig_timeout
        for name, saved in self.saved.items():
            live = getattr(self.bt, name)
            live.clear()
            live.update(saved)

    def test_janitor_tears_down_under_owner_profile_scope(self, tmp_path, monkeypatch):
        from agent import secret_scope
        from hermes_constants import (
            get_hermes_home, reset_hermes_home_override, set_hermes_home_override,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("CAMOFOX_URL", raising=False)
        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        p1 = tmp_path / "profiles" / "p1"
        p1.mkdir(parents=True)
        (p1 / ".env").write_text("CAMOFOX_URL=http://127.0.0.1:1\n", encoding="utf-8")

        # Profile p1's turn opens the session; the janitor later runs unscoped.
        home_tok = set_hermes_home_override(str(p1))
        scope_tok = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(p1))
        try:
            bt_lifecycle._update_session_activity("t1")
            self.bt._active_sessions["t1"] = {"session_name": "s1", "bb_session_id": None}
        finally:
            secret_scope.reset_secret_scope(scope_tok)
            reset_hermes_home_override(home_tok)
        self.bt._session_last_activity["t1"] -= 10

        seen = {}

        def fake_close(task_id, cmd, args, timeout=None):
            seen["home"] = str(get_hermes_home())
            seen["url"] = secret_scope.get_secret("CAMOFOX_URL")
            return {"success": True}

        with (
            patch("tools.browser_tool_session._run_browser_command", side_effect=fake_close),
            patch("tools.browser_camofox._delete", return_value={}),
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            bt_lifecycle._cleanup_inactive_browser_sessions()

        assert seen == {"home": str(p1), "url": "http://127.0.0.1:1"}
        assert "t1" not in self.bt._session_last_activity
        assert "t1" not in self.bt._active_sessions
        assert "t1" not in self.bt._session_owner_homes

    def test_repeated_failures_force_reap_and_close_cloud_session(self):
        from unittest.mock import MagicMock

        self.bt._active_sessions["t1"] = {"session_name": "s1", "bb_session_id": "bb-1"}
        self.bt._session_last_activity["t1"] = 1.0
        provider = MagicMock()

        with (
            patch("tools.browser_tool_lifecycle.cleanup_browser", side_effect=RuntimeError("boom")),
            patch("tools.browser_tool_cloud._get_cloud_provider", return_value=provider),
            patch("tools.browser_tool.os.path.exists", return_value=False),
        ):
            for _ in range(self.bt.MAX_INACTIVITY_CLEANUP_FAILURES - 1):
                bt_lifecycle._cleanup_inactive_browser_sessions()
            # An activity touch must NOT reset the failure budget.
            bt_lifecycle._update_session_activity("t1")
            self.bt._session_last_activity["t1"] = 1.0
            assert self.bt._cleanup_failures["t1"] == self.bt.MAX_INACTIVITY_CLEANUP_FAILURES - 1
            assert "t1" in self.bt._active_sessions
            provider.close_session.assert_not_called()

            bt_lifecycle._cleanup_inactive_browser_sessions()

        provider.close_session.assert_called_once_with("bb-1")
        assert "t1" not in self.bt._active_sessions
        assert "t1" not in self.bt._session_last_activity
        assert "t1" not in self.bt._cleanup_failures

    def _run_janitor_with_mid_selection_action(self, action):
        """Run the janitor once with ``action`` injected between the stale-list build and
        the per-session recheck — the exact window of #138797. The first ``time.time()``
        call is the selection snapshot; the second (the recheck) fires ``action`` first.
        ``action`` must not take ``_cleanup_lock`` (the recheck holds it — a locked
        ``_update_session_activity`` would deadlock; touch via a direct store instead).
        A positive timeout is required: with the 0s default even a just-touched
        session reads as `elapsed > 0`."""
        import time as _time

        self.bt.BROWSER_SESSION_INACTIVITY_TIMEOUT = 60
        real_time = _time.time
        calls = {"n": 0}

        def fake_time():
            calls["n"] += 1
            if calls["n"] == 1:
                return real_time() + 10_000.0  # far-future snapshot: activity 1.0 looks stale
            if calls["n"] == 2:
                action()
            return real_time()

        with (
            patch.object(bt_lifecycle.time, "time", side_effect=fake_time),
            patch("tools.browser_tool_lifecycle.cleanup_browser") as mock_cleanup,
        ):
            bt_lifecycle._cleanup_inactive_browser_sessions()
        return mock_cleanup

    def test_janitor_skips_session_touched_after_selection(self):
        """A command touching activity after the janitor built its stale list must save the
        session: the recheck under the lock sees the fresh timestamp and skips (#138797)."""
        self.bt._active_sessions["t1"] = {"session_name": "s1", "bb_session_id": None}
        self.bt._session_last_activity["t1"] = 1.0  # stale at selection time

        real_time = bt_lifecycle.time.time

        def touch():
            # What _update_session_activity would store, minus the lock it cannot take here.
            self.bt._session_last_activity["t1"] = real_time()

        mock_cleanup = self._run_janitor_with_mid_selection_action(touch)

        mock_cleanup.assert_not_called()
        assert "t1" in self.bt._active_sessions
        assert "t1" in self.bt._session_last_activity

    def test_janitor_still_cleans_untouched_stale_session(self):
        """Counterpart to the skip: with no touch in the window the session is still torn
        down (the recheck must not turn the janitor into a no-op)."""
        self.bt._active_sessions["t1"] = {"session_name": "s1", "bb_session_id": None}
        self.bt._session_last_activity["t1"] = 1.0

        mock_cleanup = self._run_janitor_with_mid_selection_action(lambda: None)

        mock_cleanup.assert_called_once()

    def test_janitor_skips_session_untracked_after_selection(self):
        """A session whose tracking vanished in the window (cleaned by another path) is
        skipped, not force-cleaned through a missing entry (#138797)."""
        self.bt._active_sessions["t1"] = {"session_name": "s1", "bb_session_id": None}
        self.bt._session_last_activity["t1"] = 1.0

        def untrack():
            self.bt._session_last_activity.pop("t1", None)
            self.bt._active_sessions.pop("t1", None)

        mock_cleanup = self._run_janitor_with_mid_selection_action(untrack)

        mock_cleanup.assert_not_called()
        assert "t1" not in self.bt._session_last_activity


class TestAtexitStopSwallowsInterrupt:
    def test_second_ctrl_c_during_join_does_not_propagate(self, monkeypatch):
        """A second Ctrl+C while the atexit hook waits on the janitor must not escape as a
        traceback (#10764): the thread is a daemon, the interpreter is already exiting."""
        from tools import browser_tool

        class _InterruptedJoin:
            def join(self, timeout=None):
                raise KeyboardInterrupt

        monkeypatch.setattr(browser_tool, "_cleanup_thread", _InterruptedJoin())
        monkeypatch.setattr(browser_tool, "_cleanup_running", True)
        bt_lifecycle._stop_browser_cleanup_thread()  # must not raise
        assert browser_tool._cleanup_running is False


class TestAtexitOriginUnimportable:
    def test_hooks_stay_silent_when_origin_fresh_import_fails(self, monkeypatch):
        """Mid-`hermes update` the on-disk tree can be half-new (new config.py importing
        a name the old utils.py lacks yet); the fresh import in origin_module then raises
        and the atexit hooks must stay silent (#112437)."""
        import tools.browser_tool_origin as origin_mod

        def _boom(_depth=2):
            raise ImportError("cannot import name 'file_signature' from 'utils'")

        monkeypatch.setattr(origin_mod, "origin_module", _boom)
        bt_lifecycle._emergency_cleanup_all_sessions()  # must not raise
        bt_lifecycle._stop_browser_cleanup_thread()  # must not raise
