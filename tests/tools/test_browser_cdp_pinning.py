"""Command-construction tests for agent-browser CDP tab pinning."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import ANY, MagicMock

from tools import browser_tool


TAB_GONE_RESULT = {
    "success": False,
    "error": "Pinned tab is no longer available",
    "code": "tab_gone",
    "data": {
        "targetId": "ABC123",
        "lastUrl": "https://example.com/path",
    },
}


def _run_commands(
    monkeypatch,
    tmp_path,
    sessions: dict[str, dict[str, Any]],
    calls: list[tuple[str, str, list[str]]],
    *,
    browser_cmd: str = "/usr/bin/agent-browser",
    environments: list[dict[str, str]] | None = None,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    commands: list[list[str]] = []
    results: list[dict[str, Any]] = []

    proc = MagicMock()
    proc.returncode = 0
    proc.wait.return_value = 0

    def capture_popen(command, **kwargs):
        commands.append(command)
        if environments is not None:
            environments.append(dict(kwargs["env"]))
        os.write(kwargs["stdout"], json.dumps({"success": True}).encode())
        return proc

    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda **_kwargs: browser_cmd,
    )
    monkeypatch.setattr(browser_tool, "_requires_real_termux_browser_install", lambda _cmd: False)
    monkeypatch.setattr(browser_tool, "_is_local_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_session_info", sessions.__getitem__)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(browser_tool, "_is_headed_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(browser_tool, "_write_owner_pid", lambda *_args: None)
    monkeypatch.setattr(browser_tool, "_build_browser_env", lambda: {"PATH": "/usr/bin"})
    monkeypatch.setattr(browser_tool, "_merge_browser_path", lambda path: path)
    monkeypatch.setattr(browser_tool, "_needs_chromium_sandbox_bypass", lambda: False)
    monkeypatch.setattr(browser_tool.subprocess, "Popen", capture_popen)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    for task_id, command, args in calls:
        results.append(browser_tool._run_browser_command(task_id, command, args))

    return commands, results


def test_npx_specs_keep_generic_and_pin_tab_capabilities_separate():
    assert browser_tool.AGENT_BROWSER_NPX_SPEC == "agent-browser@0.26.0"
    assert browser_tool.AGENT_BROWSER_PIN_TAB_NPX_SPEC == "agent-browser@0.34.0"


def test_cdp_command_uses_named_session_and_pins_tab(monkeypatch, tmp_path):
    cdp_url = "ws://127.0.0.1:9222/devtools/browser/shared"
    sessions = {
        "task-a": {
            "session_name": "cdp_task_a",
            "cdp_url": cdp_url,
        }
    }

    commands, results = _run_commands(
        monkeypatch,
        tmp_path,
        sessions,
        [("task-a", "snapshot", ["-c"])],
    )

    assert results == [{"success": True}]
    assert commands == [[
        "/usr/bin/agent-browser",
        "--session",
        "cdp_task_a",
        "--cdp",
        cdp_url,
        "--pin-tab",
        "--json",
        "snapshot",
        "-c",
    ]]


def test_shared_cdp_command_disables_daemon_idle_timeout(monkeypatch, tmp_path):
    sessions = {
        "task-a": {
            "session_name": "cdp_task_a",
            "bb_session_id": None,
            "cdp_url": "ws://127.0.0.1:9222/devtools/browser/shared",
            "features": {"cdp_override": True},
        }
    }
    environments: list[dict[str, str]] = []

    _commands, results = _run_commands(
        monkeypatch,
        tmp_path,
        sessions,
        [("task-a", "snapshot", ["-c"])],
        environments=environments,
    )

    assert results == [{"success": True}]
    assert environments[0]["AGENT_BROWSER_IDLE_TIMEOUT_MS"] == "0"


def test_local_command_keeps_configured_daemon_idle_timeout(monkeypatch, tmp_path):
    sessions = {
        "task-local": {
            "session_name": "local_task",
            "bb_session_id": None,
            "cdp_url": None,
            "features": {"local": True},
        }
    }
    environments: list[dict[str, str]] = []

    _commands, results = _run_commands(
        monkeypatch,
        tmp_path,
        sessions,
        [("task-local", "snapshot", ["-c"])],
        environments=environments,
    )

    assert results == [{"success": True}]
    assert environments[0]["AGENT_BROWSER_IDLE_TIMEOUT_MS"] == str(
        browser_tool.BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000
    )


def test_non_cdp_command_keeps_existing_session_shape(monkeypatch, tmp_path):
    sessions = {
        "task-local": {
            "session_name": "local_task",
            "cdp_url": None,
        }
    }

    commands, results = _run_commands(
        monkeypatch,
        tmp_path,
        sessions,
        [("task-local", "snapshot", ["-c"])],
    )

    assert results == [{"success": True}]
    assert commands == [[
        "/usr/bin/agent-browser",
        "--session",
        "local_task",
        "--json",
        "snapshot",
        "-c",
    ]]


def test_cdp_npx_command_uses_pin_spec_and_npm_policy(monkeypatch, tmp_path):
    cdp_url = "ws://127.0.0.1:9222/devtools/browser/shared"
    sessions = {
        "task-npx": {
            "session_name": "cdp_task_npx",
            "cdp_url": cdp_url,
        }
    }
    environments: list[dict[str, str]] = []
    monkeypatch.setattr(browser_tool, "_resolve_npx_bin", lambda: "/managed/bin/npx")

    commands, results = _run_commands(
        monkeypatch,
        tmp_path,
        sessions,
        [("task-npx", "snapshot", ["-c"])],
        browser_cmd=browser_tool.NPX_AGENT_BROWSER_SENTINEL,
        environments=environments,
    )

    assert results == [{"success": True}]
    assert commands == [[
        "/managed/bin/npx",
        "--ignore-scripts",
        "--prefer-offline",
        "-y",
        "agent-browser@0.34.0",
        "--session",
        "cdp_task_npx",
        "--cdp",
        cdp_url,
        "--pin-tab",
        "--json",
        "snapshot",
        "-c",
    ]]
    assert len(environments) == 1
    assert environments[0]["PATH"] == "/usr/bin"
    assert environments[0]["npm_config_engine_strict"] == "true"
    assert environments[0]["npm_config_min_release_age"] == "0"


def test_two_tasks_pin_distinct_sessions_on_same_cdp_endpoint(monkeypatch, tmp_path):
    cdp_url = "ws://127.0.0.1:9222/devtools/browser/shared"
    sessions = {
        "task-a": {"session_name": "cdp_task_a", "cdp_url": cdp_url},
        "task-b": {"session_name": "cdp_task_b", "cdp_url": cdp_url},
    }

    commands, results = _run_commands(
        monkeypatch,
        tmp_path,
        sessions,
        [
            ("task-a", "snapshot", []),
            ("task-b", "snapshot", []),
        ],
    )

    assert results == [{"success": True}, {"success": True}]
    assert commands == [
        [
            "/usr/bin/agent-browser",
            "--session",
            "cdp_task_a",
            "--cdp",
            cdp_url,
            "--pin-tab",
            "--json",
            "snapshot",
        ],
        [
            "/usr/bin/agent-browser",
            "--session",
            "cdp_task_b",
            "--cdp",
            cdp_url,
            "--pin-tab",
            "--json",
            "snapshot",
        ],
    ]


def test_high_level_snapshot_preserves_structured_tab_gone(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: dict(TAB_GONE_RESULT),
    )

    result = json.loads(browser_tool.browser_snapshot(task_id="task-a"))

    assert result == TAB_GONE_RESULT


def test_high_level_console_does_not_mask_structured_tab_gone(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda _task_id: False)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: dict(TAB_GONE_RESULT),
    )

    result = json.loads(browser_tool.browser_console(task_id="task-a"))

    assert result == TAB_GONE_RESULT


def test_fixed_cdp_supervisor_waits_for_pinned_target(monkeypatch):
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    start = MagicMock()
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://shared")
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "get_or_start", start)

    browser_tool._ensure_cdp_supervisor("task-a")

    start.assert_not_called()


def test_fixed_cdp_supervisor_receives_pinned_target(monkeypatch):
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    start = MagicMock()
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://shared")
    monkeypatch.setattr(browser_tool, "_get_dialog_policy_config", lambda: ("must_respond", 300.0))
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "get_or_start", start)

    browser_tool._ensure_cdp_supervisor("task-a", target_id="TARGET-A")

    start.assert_called_once_with(
        task_id="task-a",
        cdp_url="ws://shared",
        target_id="TARGET-A",
        dialog_policy="must_respond",
        dialog_timeout_s=300.0,
        publish_guard=ANY,
    )
    assert callable(start.call_args.kwargs["publish_guard"])


def test_pinned_cdp_target_id_uses_agent_browser_active_page(monkeypatch):
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: {
            "success": True,
            "data": {
                "tabs": [
                    {"active": False, "type": "page", "targetId": "OTHER"},
                    {"active": True, "type": "page", "targetId": "TARGET-A"},
                ]
            },
        },
    )

    assert browser_tool._pinned_cdp_target_id("task-a") == "TARGET-A"


def test_navigate_fails_if_auto_snapshot_reports_tab_gone(monkeypatch):
    session = {
        "session_name": "cdp_task_a",
        "cdp_url": "ws://shared",
        "_first_nav": False,
    }
    command_results = iter(
        [
            {
                "success": True,
                "data": {
                    "url": "https://example.com/final",
                    "title": "Example final",
                },
            },
            dict(TAB_GONE_RESULT),
        ]
    )
    monkeypatch.setattr(
        browser_tool,
        "_navigation_session_key",
        lambda task_id, _url: task_id,
    )
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda _task: session)
    monkeypatch.setattr(browser_tool, "_pinned_cdp_target_id", lambda _task: "TARGET-A")
    monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: next(command_results),
    )

    result = json.loads(
        browser_tool.browser_navigate("https://example.com", task_id="task-a")
    )

    assert result == {
        "success": False,
        "error": TAB_GONE_RESULT["error"],
        "url": "https://example.com/final",
        "title": "Example final",
        "code": "tab_gone",
        "data": TAB_GONE_RESULT["data"],
    }


def test_explicit_navigation_after_cleanup_creates_fresh_owned_session(monkeypatch):
    task_id = "task-retired"
    old_session = {
        "session_name": "cdp_old",
        "bb_session_id": None,
        "cdp_url": "ws://shared",
        "features": {"cdp_override": True},
        "session_key": task_id,
        "owner_task_id": task_id,
    }
    monkeypatch.setattr(browser_tool, "_active_sessions", {task_id: old_session})
    monkeypatch.setattr(browser_tool, "_session_last_activity", {task_id: 1.0})
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {task_id: task_id})
    monkeypatch.setattr(browser_tool, "_retired_browser_tasks", set(), raising=False)
    monkeypatch.setattr(browser_tool, "_maybe_stop_recording", lambda _task: None)
    monkeypatch.setattr(browser_tool.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: {"success": True},
    )

    assert browser_tool.cleanup_browser(task_id) is True
    assert task_id in browser_tool._retired_browser_tasks

    monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://shared")
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_maybe_start_recording", lambda _task: None)
    monkeypatch.setattr(browser_tool, "_pinned_cdp_target_id", lambda _task: None)
    monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda *_a, **_kw: None)
    command_calls: list[tuple[str, str, list[str]]] = []

    def run_command(session_key, command, args, **_kwargs):
        command_calls.append((session_key, command, args))
        if command == "open":
            return {
                "success": True,
                "data": {"url": "https://example.com", "title": "Example"},
            }
        return {"success": True, "data": {"snapshot": "fresh", "refs": {}}}

    monkeypatch.setattr(browser_tool, "_run_browser_command", run_command)

    result = json.loads(
        browser_tool.browser_navigate("https://example.com", task_id=task_id)
    )

    assert result["success"] is True
    assert task_id not in browser_tool._retired_browser_tasks
    fresh_session = browser_tool._active_sessions[task_id]
    assert fresh_session is not old_session
    assert fresh_session["session_name"] != old_session["session_name"]
    assert fresh_session["owner_task_id"] == task_id
    assert fresh_session["session_key"] == task_id
    assert browser_tool._last_active_session_key[task_id] == task_id
    assert command_calls == [
        (task_id, "open", ["https://example.com"]),
        (task_id, "snapshot", ["-c"]),
    ]


def test_failed_explicit_restart_restores_retired_boundary(monkeypatch):
    task_id = "task-retired-failed-restart"
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_session_last_activity", {})
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    monkeypatch.setattr(browser_tool, "_retired_browser_tasks", {task_id}, raising=False)
    monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "ws://shared")
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_maybe_start_recording", lambda _task: None)
    monkeypatch.setattr(browser_tool, "_maybe_stop_recording", lambda _task: None)
    monkeypatch.setattr(browser_tool.os.path, "exists", lambda _path: False)
    command_calls: list[tuple[str, str, list[str]]] = []

    def run_command(session_key, command, args, **_kwargs):
        command_calls.append((session_key, command, args))
        if command == "open":
            return {"success": False, "error": "navigation failed"}
        return {"success": True}

    monkeypatch.setattr(browser_tool, "_run_browser_command", run_command)

    result = json.loads(
        browser_tool.browser_navigate("https://example.com", task_id=task_id)
    )

    assert result == {"success": False, "error": "navigation failed"}
    assert task_id in browser_tool._retired_browser_tasks
    assert task_id not in browser_tool._active_sessions
    assert task_id not in browser_tool._session_last_activity
    assert task_id not in browser_tool._last_active_session_key
    assert command_calls == [
        (task_id, "open", ["https://example.com"]),
        (task_id, "tab", ["close"]),
        (task_id, "close", []),
    ]
