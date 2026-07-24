"""Regression tests for browser session cleanup and screenshot recovery."""

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
        assert browser_tool._cleanup_done is True


def _fake_chrome_proc(pid, cmdline, parent_name=None):
    """Build a psutil.Process stand-in for the orphan-chrome scanner.

    ``parent_name=None`` models a browser whose daemon died (psutil returns
    ``None`` once the parent is gone / reparented past a subreaper); a string
    models a live parent with that process name.
    """
    proc = MagicMock()
    proc.pid = pid
    proc.info = {"pid": pid, "cmdline": cmdline}
    if parent_name is None:
        proc.parent.return_value = None
    else:
        parent = MagicMock()
        parent.name.return_value = parent_name
        proc.parent.return_value = parent
    return proc


class TestOrphanedChromeReaper:
    """Regression for Chromium orphaned by an abnormal daemon death.

    When the agent-browser daemon dies without shutting Chromium down (OOM
    kill / crash / SIGKILL — the common failure mode on memory-starved
    hosts), the browser tree is reparented to init and every existing
    cleanup path goes blind:

      * ``_cleanup_single_browser_session`` kills via the *daemon* PID's
        child tree — ``_terminate_host_pid`` returns silently when that PID
        is already dead — then removes the socket dir, destroying the only
        pid breadcrumb;
      * ``_reap_orphaned_browser_sessions`` only globs daemon socket dirs
        (``agent-browser-h_*``/``cdp_*``/``hermes_*``), never the browser's
        ``agent-browser-chrome-*`` user-data dir.

    Verified live: SIGKILL the daemon after a successful navigate and the
    full Chromium tree (12+ processes) survives session cleanup
    indefinitely.  The contract: the periodic sweep must tree-kill any main
    Chromium process carrying an ``agent-browser-chrome-*`` user-data dir
    whose parent is no longer a live agent-browser daemon, and must leave
    daemon-owned browsers and unrelated Chrome installs alone.
    """

    ORPHAN_CMDLINE = [
        "/opt/chromium/chrome",
        "--headless=new",
        "--user-data-dir=/tmp/agent-browser-chrome-abc123",
        "--no-sandbox",
    ]

    def test_orphaned_main_chrome_is_reaped(self):
        from tools import browser_tool

        orphan = _fake_chrome_proc(4242, self.ORPHAN_CMDLINE, parent_name=None)

        with (
            patch("psutil.process_iter", return_value=[orphan]),
            patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid"
            ) as mock_kill,
            patch("tools.browser_tool.shutil.rmtree") as mock_rmtree,
        ):
            reaped = browser_tool._reap_orphaned_chrome_processes()

        assert reaped == 1
        mock_kill.assert_called_once_with(4242)
        mock_rmtree.assert_called_once_with(
            "/tmp/agent-browser-chrome-abc123", ignore_errors=True
        )

    def test_daemon_owned_chrome_is_left_alone(self):
        """A browser whose parent is a live agent-browser daemon is in use."""
        from tools import browser_tool

        owned = _fake_chrome_proc(
            4243, self.ORPHAN_CMDLINE, parent_name="agent-browser-l"
        )

        with (
            patch("psutil.process_iter", return_value=[owned]),
            patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid"
            ) as mock_kill,
        ):
            reaped = browser_tool._reap_orphaned_chrome_processes()

        assert reaped == 0
        mock_kill.assert_not_called()

    def test_helper_and_unrelated_processes_are_skipped(self):
        """--type= helpers die with the main tree; foreign Chromes are not ours."""
        from tools import browser_tool

        helper = _fake_chrome_proc(
            4244,
            self.ORPHAN_CMDLINE + ["--type=renderer"],
            parent_name=None,
        )
        foreign = _fake_chrome_proc(
            4245,
            ["/usr/bin/google-chrome", "--user-data-dir=/home/u/.config/chrome"],
            parent_name=None,
        )
        no_cmdline = _fake_chrome_proc(4246, [], parent_name=None)

        with (
            patch("psutil.process_iter", return_value=[helper, foreign, no_cmdline]),
            patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid"
            ) as mock_kill,
        ):
            reaped = browser_tool._reap_orphaned_chrome_processes()

        assert reaped == 0
        mock_kill.assert_not_called()

    def test_orphan_reap_runs_chrome_sweep_even_without_socket_dirs(self):
        """The sweep must not be short-circuited by the empty-socket-dir return.

        An orphaned browser leaves NO socket dir behind (the daemon's dir is
        removed by session cleanup), so the sweep has to run precisely when
        the socket-dir scan finds nothing.
        """
        from tools import browser_tool

        with (
            patch("glob.glob", return_value=[]),
            patch(
                "tools.browser_tool._reap_orphaned_chrome_processes"
            ) as mock_sweep,
        ):
            browser_tool._reap_orphaned_browser_sessions()

        mock_sweep.assert_called_once_with()
