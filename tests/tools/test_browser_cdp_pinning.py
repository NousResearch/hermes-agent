"""Command-construction tests for agent-browser CDP tab pinning."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock

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
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    commands: list[list[str]] = []
    results: list[dict[str, Any]] = []

    proc = MagicMock()
    proc.returncode = 0
    proc.wait.return_value = 0

    def capture_popen(command, **kwargs):
        commands.append(command)
        os.write(kwargs["stdout"], json.dumps({"success": True}).encode())
        return proc

    monkeypatch.setattr(browser_tool, "_find_agent_browser", lambda: "/usr/bin/agent-browser")
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


def test_managed_npx_spec_includes_first_tab_pinning_release():
    assert browser_tool.AGENT_BROWSER_NPX_SPEC == "agent-browser@^0.34.0"


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
