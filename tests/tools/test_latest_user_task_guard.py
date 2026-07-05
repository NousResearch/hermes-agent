"""Regression tests for the latest-user-request authority gate."""

from __future__ import annotations

import json

import model_tools
from agent.runtime_evidence import clear_runtime_evidence, current_runtime_evidence, seed_turn_runtime_evidence


def test_stale_pre_compaction_task_is_blocked_after_newer_user_request(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        latest_user_request="Check Notion only.",
        compaction_summary_active=True,
    )
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda function_name, function_args, **kwargs: json.dumps({"status": "ok", "output": "unexpected"}),
    )

    result = model_tools.handle_function_call(
        "terminal",
        {"command": "gh pr checks 17"},
        task_id="task-phase7-1",
        session_id="sess-phase7-1",
        user_task="Continue PR work.",
    )

    payload = json.loads(result)
    assert "Blocked stale task continuation after compaction" in payload["error"]
    assert "Latest real user request: Check Notion only." in payload["error"]
    assert "Stale/active task attempted: Continue PR work." in payload["error"]
    evidence = current_runtime_evidence()
    assert evidence is not None
    assert evidence["latest_user_request"] == "Check Notion only."
    assert evidence["active_user_task"] == "Continue PR work."
    clear_runtime_evidence()


def test_matching_latest_user_request_is_allowed(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        latest_user_request="Check Notion only.",
        compaction_summary_active=True,
    )
    calls = []

    def _dispatch(function_name, function_args, **kwargs):
        calls.append((function_name, function_args))
        return json.dumps({"status": "ok", "output": "allowed"})

    monkeypatch.setattr(model_tools.registry, "dispatch", _dispatch)

    result = model_tools.handle_function_call(
        "terminal",
        {"command": "echo hello"},
        task_id="task-phase7-2",
        session_id="sess-phase7-2",
        user_task="Check Notion only.",
    )

    assert json.loads(result)["output"] == "allowed"
    assert calls == [("terminal", {"command": "echo hello"})]
    evidence = current_runtime_evidence()
    assert evidence is not None
    assert evidence["latest_user_request"] == "Check Notion only."
    clear_runtime_evidence()




def test_task_sensitive_command_is_allowed_when_latest_request_matches_after_compaction():
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        latest_user_request="Check Notion only.",
        compaction_summary_active=True,
    )

    block = model_tools.enforce_latest_user_request_authority(
        function_name="terminal",
        function_args={"command": "gh pr checks 17"},
        user_task="Check Notion only.",
    )

    assert block is None
    clear_runtime_evidence()

def test_tool_dispatch_keeps_latest_user_request_evidence_current(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence()
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda function_name, function_args, **kwargs: json.dumps({"status": "ok", "output": "done"}),
    )

    result = model_tools.handle_function_call(
        "terminal",
        {"command": "echo hello"},
        task_id="task-phase7-3",
        session_id="sess-phase7-3",
        user_task="Check Notion only.",
    )

    assert json.loads(result)["output"] == "done"
    evidence = current_runtime_evidence()
    assert evidence is not None
    assert evidence["latest_user_request"] == "Check Notion only."
    clear_runtime_evidence()


def test_stale_task_after_summarized_context_is_blocked(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        latest_user_request="Review the Notion page only.",
        compaction_handoff_active=True,
        context_compacted=True,
    )
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda function_name, function_args, **kwargs: json.dumps({"status": "ok", "output": "unexpected"}),
    )

    result = model_tools.handle_function_call(
        "write_file",
        {"path": "docs/runtime.md", "content": "stale"},
        task_id="task-phase7-4",
        session_id="sess-phase7-4",
        user_task="Continue PR work.",
    )

    payload = json.loads(result)
    assert "Blocked stale task continuation after compaction" in payload["error"]
    assert "Latest real user request: Review the Notion page only." in payload["error"]
    clear_runtime_evidence()


def test_unrelated_normal_commands_still_pass(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        latest_user_request="Check Notion only.",
        compaction_summary_active=True,
    )
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda function_name, function_args, **kwargs: json.dumps({"status": "ok", "output": "hello"}),
    )

    result = model_tools.handle_function_call(
        "terminal",
        {"command": "echo hello"},
        task_id="task-phase7-5",
        session_id="sess-phase7-5",
        user_task="Continue PR work.",
    )

    assert json.loads(result)["output"] == "hello"
    clear_runtime_evidence()
