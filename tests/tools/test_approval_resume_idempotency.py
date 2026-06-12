from __future__ import annotations

import json

import pytest

from tools import approval as A


@pytest.fixture
def isolated_gateway_session(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(A, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_approval_ledger_path", lambda: tmp_path / "approval-ledger.json")

    session_key = "fgd-193-session"
    token = A.set_current_session_key(session_key)
    with A._lock:
        A._gateway_queues.pop(session_key, None)
        A._gateway_notify_cbs.pop(session_key, None)
        A._session_approved.pop(session_key, None)
    try:
        yield session_key, tmp_path / "approval-ledger.json"
    finally:
        A.reset_current_session_key(token)
        with A._lock:
            A._gateway_queues.pop(session_key, None)
            A._gateway_notify_cbs.pop(session_key, None)
            A._session_approved.pop(session_key, None)


def _load_ledger(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_gateway_approval_record_created_before_prompt_and_stores_no_raw_command(isolated_gateway_session):
    session_key, ledger_path = isolated_gateway_session
    command = "printf 'super-secret-token-123'"
    seen = {}

    def notify(_approval_data):
        seen["record"] = A._get_approval_ledger_record(
            session_key,
            command,
            "terminal",
            ["dangerous-test"],
        )
        with A._lock:
            entry = A._gateway_queues[session_key][-1]
            entry.result = "deny"
            entry.event.set()

    result = A._await_gateway_decision(
        session_key,
        notify,
        {
            "command": command,
            "pattern_key": "dangerous-test",
            "pattern_keys": ["dangerous-test"],
            "description": "dangerous test",
            "tool_type": "terminal",
        },
        surface="gateway",
    )

    assert result["resolved"] is True
    assert result["choice"] == "deny"
    assert result["approval_record_id"] == seen["record"]["id"]
    assert seen["record"]["status"] == "pending"

    raw_ledger = ledger_path.read_text(encoding="utf-8")
    assert "super-secret-token-123" not in raw_ledger
    assert command not in raw_ledger
    assert _load_ledger(ledger_path)[seen["record"]["id"]]["status"] == "denied"


def test_approve_once_marks_approved_once_before_execution_then_completed(isolated_gateway_session):
    session_key, ledger_path = isolated_gateway_session
    command = "rm -rf ./build-cache"

    def notify(_approval_data):
        with A._lock:
            entry = A._gateway_queues[session_key][-1]
        resolved = A.resolve_gateway_approval(session_key, "once")
        assert resolved == 1
        record = A._get_approval_ledger_record(
            session_key,
            command,
            "terminal",
            ["dangerous-test"],
        )
        assert record["status"] == "approved_once"
        assert entry.event.is_set()

    decision = A._await_gateway_decision(
        session_key,
        notify,
        {
            "command": command,
            "pattern_key": "dangerous-test",
            "pattern_keys": ["dangerous-test"],
            "description": "dangerous test",
            "tool_type": "terminal",
        },
        surface="gateway",
    )

    assert decision["choice"] == "once"
    record = A._get_approval_ledger_record(session_key, command, "terminal", ["dangerous-test"])
    assert record["status"] == "approved_once"

    assert A.mark_gateway_approval_executing(record["id"])
    assert _load_ledger(ledger_path)[record["id"]]["status"] == "executing"

    assert A.mark_gateway_approval_completed(record["id"], exit_code=0)
    completed = _load_ledger(ledger_path)[record["id"]]
    assert completed["status"] == "completed"
    assert completed["exit_code"] == 0


def test_execute_code_same_approved_once_after_resume_returns_stale_without_reprompt(isolated_gateway_session):
    session_key, _ledger_path = isolated_gateway_session
    calls = []

    def approve_once(_approval_data):
        calls.append("first")
        assert A.resolve_gateway_approval(session_key, "once") == 1

    with A._lock:
        A._gateway_notify_cbs[session_key] = approve_once

    code = "print('hello from execute_code')"
    first = A.check_execute_code_guard(code, "local")
    assert first["approved"] is True
    assert first.get("approval_record_id")

    def should_not_prompt(_approval_data):  # pragma: no cover - assertion below proves this is unused
        calls.append("second")
        raise AssertionError("duplicate approval prompt should be suppressed")

    with A._lock:
        A._gateway_notify_cbs[session_key] = should_not_prompt

    second = A.check_execute_code_guard(code, "local")
    assert second["approved"] is False
    assert second["outcome"] == "stale"
    assert "approval was interrupted" in second["message"]
    assert calls == ["first"]


def test_repeated_same_fingerprint_returns_approval_loop_detected(isolated_gateway_session):
    session_key, _ledger_path = isolated_gateway_session
    command = "rm -rf ./node_modules"

    A._prepare_gateway_approval_record(
        session_key=session_key,
        command=command,
        tool_type="terminal",
        pattern_keys=["dangerous-test"],
        description="dangerous test",
    )

    first_duplicate = A._handle_duplicate_gateway_approval(
        session_key=session_key,
        command=command,
        tool_type="terminal",
        pattern_keys=["dangerous-test"],
    )

    assert first_duplicate is not None
    assert first_duplicate["approved"] is False
    assert first_duplicate["outcome"] == "stale"

    duplicate = A._handle_duplicate_gateway_approval(
        session_key=session_key,
        command=command,
        tool_type="terminal",
        pattern_keys=["dangerous-test"],
    )

    assert duplicate is not None
    assert duplicate["approved"] is False
    assert duplicate["outcome"] == "approval_loop_detected"
    assert "APPROVAL LOOP DETECTED" in duplicate["message"]


def test_terminal_and_execute_code_paths_remain_guarded_by_approval_module():
    terminal_source = (A.__file__)
    assert terminal_source.endswith("approval.py")

    import inspect
    from tools import code_execution_tool, terminal_tool

    terminal_text = inspect.getsource(terminal_tool)
    execute_text = inspect.getsource(code_execution_tool)

    assert "approval_record_id" in terminal_text
    assert "mark_gateway_approval_executing" in terminal_text
    assert "mark_gateway_approval_completed" in terminal_text
    assert "check_execute_code_guard" in execute_text
    assert "mark_gateway_approval_executing" in execute_text
    assert "mark_gateway_approval_completed" in execute_text
