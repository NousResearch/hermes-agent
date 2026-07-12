"""Tests for per-session browser command context caching."""

from unittest.mock import patch

import pytest


def _install_session(browser_tool, task_id="task-ctx", session_name="sess-ctx", cdp_url=None):
    session_info = {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": cdp_url,
        "features": {"local": cdp_url is None},
        "session_key": task_id,
        "owner_task_id": task_id,
    }
    browser_tool._active_sessions[task_id] = session_info
    return session_info


@pytest.fixture
def browser_tool_for_context_cache(monkeypatch, tmp_path):
    from tools import browser_tool

    orig_active_sessions = browser_tool._active_sessions.copy()
    orig_context_cache = browser_tool._browser_command_context_cache.copy()
    orig_session_activity = browser_tool._session_last_activity.copy()
    orig_recording_sessions = browser_tool._recording_sessions.copy()
    orig_last_active = browser_tool._last_active_session_key.copy()

    browser_tool._active_sessions.clear()
    browser_tool._browser_command_context_cache.clear()
    browser_tool._session_last_activity.clear()
    browser_tool._recording_sessions.clear()
    browser_tool._last_active_session_key.clear()

    monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_needs_chromium_sandbox_bypass", lambda: False)
    monkeypatch.setattr(
        browser_tool,
        "_merge_browser_path",
        lambda existing_path="": f"merged:{existing_path}",
    )

    try:
        yield browser_tool
    finally:
        browser_tool._active_sessions.clear()
        browser_tool._active_sessions.update(orig_active_sessions)
        browser_tool._browser_command_context_cache.clear()
        browser_tool._browser_command_context_cache.update(orig_context_cache)
        browser_tool._session_last_activity.clear()
        browser_tool._session_last_activity.update(orig_session_activity)
        browser_tool._recording_sessions.clear()
        browser_tool._recording_sessions.update(orig_recording_sessions)
        browser_tool._last_active_session_key.clear()
        browser_tool._last_active_session_key.update(orig_last_active)


def test_browser_command_context_cache_hit_reuses_stable_context(browser_tool_for_context_cache):
    browser_tool = browser_tool_for_context_cache
    session_info = _install_session(browser_tool)

    first = browser_tool._get_browser_command_context(
        "task-ctx",
        session_info,
        "/usr/local/bin/agent-browser",
        "chrome",
    )
    second = browser_tool._get_browser_command_context(
        "task-ctx",
        session_info,
        "/usr/local/bin/agent-browser",
        "chrome",
    )

    assert first == second
    assert first["cmd_prefix"] == ["/usr/local/bin/agent-browser"]
    assert first["backend_args"] == ["--session", "sess-ctx", "--engine", "chrome"]
    assert first["task_socket_dir"].endswith("/agent-browser-sess-ctx")


def test_browser_command_context_returns_defensive_copies(browser_tool_for_context_cache, monkeypatch):
    browser_tool = browser_tool_for_context_cache
    session_info = _install_session(browser_tool)
    first = browser_tool._get_browser_command_context(
        "task-ctx",
        session_info,
        "/usr/local/bin/agent-browser",
        "auto",
    )
    first["cmd_prefix"].append("--mutated")
    first["backend_args"][0] = "--cdp"

    second = browser_tool._get_browser_command_context(
        "task-ctx",
        session_info,
        "/usr/local/bin/agent-browser",
        "auto",
    )

    assert second["cmd_prefix"] == ["/usr/local/bin/agent-browser"]
    assert second["backend_args"] == ["--session", "sess-ctx"]
    assert first["cmd_prefix"] is not second["cmd_prefix"]
    assert first["backend_args"] is not second["backend_args"]


def test_engine_override_uses_separate_cached_context(browser_tool_for_context_cache, monkeypatch):
    browser_tool = browser_tool_for_context_cache
    session_info = _install_session(browser_tool)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "lightpanda")

    assert browser_tool._effective_browser_engine(None) == "lightpanda"
    assert browser_tool._effective_browser_engine("AUTO") == "auto"
    assert browser_tool._effective_browser_engine("chrome") == "chrome"
    assert browser_tool._effective_browser_engine("invalid") == "auto"

    lightpanda_context = browser_tool._get_browser_command_context(
        "task-ctx",
        session_info,
        "/usr/local/bin/agent-browser",
        browser_tool._effective_browser_engine(None),
    )
    auto_context = browser_tool._get_browser_command_context(
        "task-ctx",
        session_info,
        "/usr/local/bin/agent-browser",
        browser_tool._effective_browser_engine("auto"),
    )
    lightpanda_again = browser_tool._get_browser_command_context(
        "task-ctx",
        session_info,
        "/usr/local/bin/agent-browser",
        browser_tool._effective_browser_engine(None),
    )

    assert lightpanda_context["backend_args"] == [
        "--session",
        "sess-ctx",
        "--engine",
        "lightpanda",
    ]
    assert auto_context["backend_args"] == ["--session", "sess-ctx"]
    assert lightpanda_again["backend_args"] == lightpanda_context["backend_args"]


def test_browser_command_env_refreshes_between_cached_commands(
    browser_tool_for_context_cache,
    monkeypatch,
):
    browser_tool = browser_tool_for_context_cache
    session_info = _install_session(browser_tool)
    current_key = {"value": "old-key"}

    def fake_build_browser_env():
        return {
            "PATH": "/base/bin",
            "BROWSER_USE_API_KEY": current_key["value"],
        }

    monkeypatch.setattr(browser_tool, "_build_browser_env", fake_build_browser_env)

    first_context = browser_tool._get_browser_command_context(
        "task-ctx",
        session_info,
        "/usr/local/bin/agent-browser",
        "chrome",
    )
    first_env = browser_tool._build_browser_command_env(first_context["task_socket_dir"])

    current_key["value"] = "reloaded-key"
    second_context = browser_tool._get_browser_command_context(
        "task-ctx",
        session_info,
        "/usr/local/bin/agent-browser",
        "chrome",
    )
    second_env = browser_tool._build_browser_command_env(second_context["task_socket_dir"])

    assert first_context == second_context
    assert first_env["BROWSER_USE_API_KEY"] == "old-key"
    assert second_env["BROWSER_USE_API_KEY"] == "reloaded-key"
    assert second_env["AGENT_BROWSER_SOCKET_DIR"] == second_context["task_socket_dir"]


def test_cleanup_browser_invalidates_command_context_cache(browser_tool_for_context_cache, monkeypatch):
    browser_tool = browser_tool_for_context_cache
    session_info = _install_session(browser_tool, task_id="task-clean", session_name="sess-clean")
    browser_tool._get_browser_command_context(
        "task-clean",
        session_info,
        "/usr/local/bin/agent-browser",
        "auto",
    )
    browser_tool._get_browser_command_context(
        "task-clean",
        session_info,
        "/usr/local/bin/agent-browser",
        "lightpanda",
    )
    assert browser_tool._browser_command_context_cache

    with (
        patch("tools.browser_tool._maybe_stop_recording"),
        patch("tools.browser_tool._run_browser_command", return_value={"success": True}) as mock_run,
        patch("tools.browser_tool.os.path.exists", return_value=False),
    ):
        browser_tool.cleanup_browser("task-clean")

    assert "task-clean" not in browser_tool._active_sessions
    assert not [
        cache_key
        for cache_key in browser_tool._browser_command_context_cache
        if cache_key[0] == "task-clean"
    ]
    mock_run.assert_called_once_with("task-clean", "close", [], timeout=10)
