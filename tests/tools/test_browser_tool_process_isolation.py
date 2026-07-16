import json
import os

import tools.browser_tool as browser_tool
import tools.interrupt as interrupt


def test_browser_popen_extra_starts_new_session_on_posix(monkeypatch):
    monkeypatch.setattr(browser_tool.os, "name", "posix")

    assert browser_tool._browser_popen_extra() == {"start_new_session": True}


def test_browser_popen_extra_keeps_windows_no_console_without_new_group(monkeypatch):
    class FakeStartupInfo:
        def __init__(self):
            self.dwFlags = 0

    monkeypatch.setattr(browser_tool.os, "name", "nt")
    monkeypatch.setattr(browser_tool.subprocess, "STARTUPINFO", FakeStartupInfo, raising=False)
    monkeypatch.setattr(browser_tool.subprocess, "STARTF_USESTDHANDLES", 0x100, raising=False)

    extra = browser_tool._browser_popen_extra()

    assert extra["creationflags"] == 0x08000000
    assert extra["close_fds"] is True
    assert isinstance(extra["startupinfo"], FakeStartupInfo)
    assert extra["startupinfo"].dwFlags == 0x100
    assert "start_new_session" not in extra


def test_browser_process_isolation_kwargs_cover_both_popen_paths(monkeypatch, tmp_path):
    """Both production launch paths must pass the shared isolation kwargs."""
    popen_calls = []
    isolation = {"start_new_session": "sentinel"}

    class FakeProcess:
        returncode = 0

        def __init__(self, argv, **kwargs):
            popen_calls.append((argv, kwargs))
            os.write(kwargs["stdout"], json.dumps({"success": True}).encode())

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(browser_tool, "_browser_popen_extra", lambda: isolation)
    monkeypatch.setattr(browser_tool.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(browser_tool, "_find_agent_browser", lambda: "/bin/agent-browser")
    monkeypatch.setattr(browser_tool, "_requires_real_termux_browser_install", lambda command: False)
    monkeypatch.setattr(browser_tool, "_is_local_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda task_id: {
        "session_name": "normal", "cdp_url": "ws://example.invalid"
    })
    monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(browser_tool, "_write_owner_pid", lambda *args: None)
    monkeypatch.setattr(browser_tool, "_build_browser_env", lambda: {})
    monkeypatch.setattr(browser_tool, "_merge_browser_path", lambda value: value)
    monkeypatch.setattr(interrupt, "is_interrupted", lambda: False)

    result = browser_tool._run_browser_command("task", "snapshot")
    assert result["success"] is True
    assert len(popen_calls) == 1
    assert popen_calls[0][1]["start_new_session"] == "sentinel"

    popen_calls.clear()
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *args, **kwargs: {"success": True, "data": {"result": "https://example.com"}},
    )
    monkeypatch.setattr(browser_tool, "_chromium_installed", lambda: True)

    result = browser_tool._run_chrome_fallback_command("task", "snapshot", [], 5)
    assert result["success"] is True
    assert len(popen_calls) == 3  # open, requested command, close
    assert all(call[1]["start_new_session"] == "sentinel" for call in popen_calls)
