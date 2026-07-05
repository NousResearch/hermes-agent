import tools.browser_tool as browser_tool


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


def test_browser_command_popen_sites_use_shared_isolation_helper():
    source = browser_tool.__loader__.get_source(browser_tool.__name__)
    # Helper definition + two Popen call sites in chrome fallback and normal
    # agent-browser command execution.
    assert source.count("_browser_popen_extra()") >= 3
