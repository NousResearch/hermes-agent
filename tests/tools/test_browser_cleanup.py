"""Regression tests for browser session cleanup and screenshot recovery."""

from unittest.mock import Mock, patch


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


class TestWindowsAgentBrowserHostCleanup:
    socket_a = r"C:\\Temp\\agent-browser-a"
    socket_b = r"C:\\Temp\\agent-browser-b"

    def _proc(self, env=None, error=None):
        proc = Mock(pid=123)
        proc.name.side_effect = error or (lambda: "agent-browser-win32-x64.exe")
        proc.cmdline.side_effect = error or (lambda: [])
        proc.environ.side_effect = error or (lambda: env or {})
        return proc

    def _run(self, monkeypatch, processes, start=101, terminate=None):
        import psutil
        from tools import browser_tool
        from tools.process_registry import ProcessRegistry
        monkeypatch.setattr(browser_tool.sys, "platform", "win32")
        monkeypatch.setattr(psutil, "process_iter", lambda _attrs: processes)
        monkeypatch.setattr("gateway.status.get_process_start_time", lambda _pid: start)
        terminate = terminate or Mock()
        monkeypatch.setattr(ProcessRegistry, "_terminate_host_pid", terminate)
        browser_tool._cleanup_windows_agent_browser_host(self.socket_a)
        return terminate

    def test_per_session_cleanup_invokes_host_cleanup(self, monkeypatch):
        from tools import browser_tool
        browser_tool._active_sessions["task"] = {"session_name": "a", "bb_session_id": None}
        helper = Mock()
        monkeypatch.setattr(browser_tool, "_cleanup_windows_agent_browser_host", helper)
        with (patch("tools.browser_tool._maybe_stop_recording"),
              patch("tools.browser_tool._run_browser_command"),
              patch("tools.browser_tool.os.path.exists", return_value=True),
              patch("tools.browser_tool.os.path.isfile", return_value=False),
              patch("tools.browser_tool.shutil.rmtree")):
            browser_tool.cleanup_browser("task")
        helper.assert_called_once()

    def test_matching_binding_terminates_with_start_token(self, monkeypatch):
        proc = self._proc({"AGENT_BROWSER_SOCKET_DIR": self.socket_a})
        self._run(monkeypatch, [proc]).assert_called_once_with(123, expected_start=101)

    def test_different_binding_is_preserved(self, monkeypatch):
        self._run(monkeypatch, [self._proc({"AGENT_BROWSER_SOCKET_DIR": self.socket_b})]).assert_not_called()

    def test_missing_binding_is_preserved(self, monkeypatch):
        self._run(monkeypatch, [self._proc()]).assert_not_called()

    def test_missing_start_token_is_skipped(self, monkeypatch):
        self._run(monkeypatch, [self._proc({"AGENT_BROWSER_SOCKET_DIR": self.socket_a})], start=None).assert_not_called()

    def test_recycled_pid_is_rejected_by_registry(self, monkeypatch):
        from tools import process_registry
        from tools.process_registry import ProcessRegistry
        monkeypatch.setattr(process_registry, "_IS_WINDOWS", True)
        monkeypatch.setattr(ProcessRegistry, "_is_host_pid_alive", lambda _pid: True)
        monkeypatch.setattr(ProcessRegistry, "_safe_host_start_time", lambda _pid: 202)
        taskkill = Mock()
        monkeypatch.setattr(process_registry.subprocess, "run", taskkill)
        self._run(monkeypatch, [self._proc({"AGENT_BROWSER_SOCKET_DIR": self.socket_a})], terminate=ProcessRegistry._terminate_host_pid)
        taskkill.assert_not_called()

    def test_inaccessible_process_does_not_crash(self, monkeypatch):
        import psutil
        self._run(monkeypatch, [self._proc(error=psutil.AccessDenied(pid=123))])
