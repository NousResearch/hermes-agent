"""Regression tests for local browser executable discovery."""

def _reset_browser_discovery(browser_tool):
    browser_tool._cached_chromium_installed = None
    browser_tool._cached_chromium_executable = None


def test_playwright_chromium_cache_resolves_concrete_executable(tmp_path, monkeypatch):
    from tools import browser_tool

    home = tmp_path / "home"
    chrome = home / ".cache" / "ms-playwright" / "chromium-1223" / "chrome-linux" / "chrome"
    chrome.parent.mkdir(parents=True)
    chrome.write_text("#!/bin/sh\n")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    monkeypatch.setattr(browser_tool.shutil, "which", lambda _name: None)
    _reset_browser_discovery(browser_tool)

    assert browser_tool._chromium_installed() is True
    assert browser_tool._get_chromium_executable() == str(chrome)


def test_playwright_headless_shell_cache_resolves_concrete_executable(tmp_path, monkeypatch):
    from tools import browser_tool

    cache = tmp_path / "ms-playwright"
    headless = cache / "chromium_headless_shell-1223" / "chrome-linux" / "headless_shell"
    headless.parent.mkdir(parents=True)
    headless.write_text("#!/bin/sh\n")

    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(cache))
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    monkeypatch.setattr(browser_tool.shutil, "which", lambda _name: None)
    _reset_browser_discovery(browser_tool)

    assert browser_tool._chromium_installed() is True
    assert browser_tool._get_chromium_executable() == str(headless)


def test_explicit_agent_browser_executable_path_is_respected(tmp_path, monkeypatch):
    from tools import browser_tool

    explicit = tmp_path / "custom-chrome"
    explicit.write_text("#!/bin/sh\n")

    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(explicit))
    monkeypatch.setattr(browser_tool.shutil, "which", lambda _name: None)
    _reset_browser_discovery(browser_tool)

    assert browser_tool._chromium_installed() is True
    assert browser_tool._get_chromium_executable() == str(explicit)


def test_cache_reset_clears_chromium_executable(monkeypatch):
    from tools import browser_tool

    browser_tool._cached_chromium_installed = True
    browser_tool._cached_chromium_executable = "/tmp/chrome"

    monkeypatch.setattr(browser_tool, "cleanup_browser", lambda _task_id: None)
    monkeypatch.setattr(browser_tool, "_active_sessions", {})

    browser_tool.cleanup_all_browsers()

    assert browser_tool._cached_chromium_installed is None
    assert browser_tool._cached_chromium_executable is None
