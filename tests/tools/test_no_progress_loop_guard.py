"""Regression tests for the no-progress loop detector."""

from __future__ import annotations

import json

import model_tools
from agent.runtime_evidence import clear_runtime_evidence, current_runtime_evidence, seed_turn_runtime_evidence


def _dispatch_result(payloads):
    items = list(payloads)

    def _run(function_name, function_args, **kwargs):
        if items:
            return items.pop(0)
        return payloads[-1]

    return _run


def test_repeated_identical_ci_checks_block_after_threshold(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(latest_user_request="Check whether CI is green.")
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        _dispatch_result([
            json.dumps({"status": "pending", "output": "checks pending"}),
            json.dumps({"status": "pending", "output": "checks pending"}),
            json.dumps({"status": "pending", "output": "checks pending"}),
        ]),
    )

    first = model_tools.handle_function_call(
        "terminal",
        {"command": "gh pr checks 17"},
        task_id="task-1",
        session_id="sess-1",
        user_task="Check CI",
    )
    second = model_tools.handle_function_call(
        "terminal",
        {"command": "gh pr checks 17"},
        task_id="task-1",
        session_id="sess-1",
        user_task="Check CI",
    )
    third = model_tools.handle_function_call(
        "terminal",
        {"command": "gh pr checks 17"},
        task_id="task-1",
        session_id="sess-1",
        user_task="Check CI",
    )

    assert json.loads(first)["status"] == "pending"
    assert json.loads(second)["status"] == "pending"
    third_payload = json.loads(third)
    assert "no-progress loop" in third_payload["error"]
    clear_runtime_evidence()


def test_repeated_identical_failed_patch_attempts_block(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(latest_user_request="Patch the target file.")
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        _dispatch_result([
            json.dumps({"error": "patch failed", "status": "blocked"}),
            json.dumps({"error": "patch failed", "status": "blocked"}),
            json.dumps({"error": "patch failed", "status": "blocked"}),
        ]),
    )

    first = model_tools.handle_function_call(
        "patch",
        {"path": "notes.txt", "patch": "@@ -1 +1 @@"},
        task_id="task-2",
        session_id="sess-2",
        user_task="Patch the target file.",
    )
    second = model_tools.handle_function_call(
        "patch",
        {"path": "notes.txt", "patch": "@@ -1 +1 @@"},
        task_id="task-2",
        session_id="sess-2",
        user_task="Patch the target file.",
    )
    third = model_tools.handle_function_call(
        "patch",
        {"path": "notes.txt", "patch": "@@ -1 +1 @@"},
        task_id="task-2",
        session_id="sess-2",
        user_task="Patch the target file.",
    )

    assert json.loads(first)["status"] == "blocked"
    assert json.loads(second)["status"] == "blocked"
    third_payload = json.loads(third)
    assert "no-progress loop" in third_payload["error"]
    clear_runtime_evidence()


def test_repeated_identical_status_checks_without_change_block(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(latest_user_request="Inspect repository status.")
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        _dispatch_result([
            json.dumps({"status": "ok", "output": "clean"}),
            json.dumps({"status": "ok", "output": "clean"}),
            json.dumps({"status": "ok", "output": "clean"}),
        ]),
    )

    first = model_tools.handle_function_call(
        "terminal",
        {"command": "git status --short"},
        task_id="task-3",
        session_id="sess-3",
        user_task="Inspect repository status.",
    )
    second = model_tools.handle_function_call(
        "terminal",
        {"command": "git status --short"},
        task_id="task-3",
        session_id="sess-3",
        user_task="Inspect repository status.",
    )
    third = model_tools.handle_function_call(
        "terminal",
        {"command": "git status --short"},
        task_id="task-3",
        session_id="sess-3",
        user_task="Inspect repository status.",
    )

    assert json.loads(first)["status"] == "ok"
    assert json.loads(second)["status"] == "ok"
    third_payload = json.loads(third)
    assert "no-progress loop" in third_payload["error"]
    clear_runtime_evidence()


def test_polling_with_changed_evidence_is_allowed(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(latest_user_request="Wait for CI to move.")
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        _dispatch_result([
            json.dumps({"status": "pending", "output": "checks pending"}),
            json.dumps({"status": "pending", "output": "checks still pending, new log line"}),
            json.dumps({"status": "pending", "output": "checks still pending, new log line"}),
        ]),
    )

    first = model_tools.handle_function_call(
        "terminal",
        {"command": "gh pr checks 17"},
        task_id="task-4",
        session_id="sess-4",
        user_task="Wait for CI to move.",
    )
    second = model_tools.handle_function_call(
        "terminal",
        {"command": "gh pr checks 17"},
        task_id="task-4",
        session_id="sess-4",
        user_task="Wait for CI to move.",
    )
    third = model_tools.handle_function_call(
        "terminal",
        {"command": "gh pr checks 17"},
        task_id="task-4",
        session_id="sess-4",
        user_task="Wait for CI to move.",
    )

    assert json.loads(first)["output"] == "checks pending"
    assert json.loads(second)["output"] == "checks still pending, new log line"
    assert json.loads(third)["output"] == "checks still pending, new log line"
    evidence = current_runtime_evidence()
    assert evidence is not None
    loop_state = evidence.get("no_progress_loop_state")
    assert loop_state is not None
    assert loop_state["repeat_cycles"] == 1
    clear_runtime_evidence()


def test_unrelated_normal_commands_pass(monkeypatch):
    clear_runtime_evidence()
    seed_turn_runtime_evidence(latest_user_request="Say hello.")
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda function_name, function_args, **kwargs: json.dumps({"status": "ok", "output": "hello"}),
    )

    result = model_tools.handle_function_call(
        "terminal",
        {"command": "echo hello"},
        task_id="task-5",
        session_id="sess-5",
        user_task="Say hello.",
    )

    assert json.loads(result)["output"] == "hello"
    evidence = current_runtime_evidence()
    assert evidence is None or "no_progress_loop_state" not in evidence
    clear_runtime_evidence()
