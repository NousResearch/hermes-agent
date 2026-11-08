"""Approvals that no surface can show fail closed instead of parking a phantom request.

With ``approvals.mode: manual`` and neither a gateway notifier nor a CLI panel for the session
(finite helpers, ``delegate_task`` children, headless workers) nobody can ever answer the prompt.
The guard must say so (``approval_delivery: "unavailable"``, ``approval_pending: False``, outcome
``notify_failed``), run nothing, and leave no pending entry behind for ``/approve`` to find.
Explicit denials and hard-policy refusals keep their own results.
"""

from __future__ import annotations

import json

import pytest

from tools import approval as A
from tools import approval_context, approval_smart

SESSION = "undelivered-test-session"
DANGEROUS = "rm -rf /home/user/project"


@pytest.fixture
def no_surface(monkeypatch):
    """Gateway-style ask mode, manual approvals, a bound session key, no notifier, no CLI panel."""
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    for var in ("HERMES_INTERACTIVE", "HERMES_CRON_SESSION", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    from tools.terminal_tool import set_approval_callback
    set_approval_callback(None)  # no CLI panel
    token = approval_context.set_current_session_key(SESSION)
    with A._lock:
        A._pending.pop(SESSION, None)
        A._gateway_queues.pop(SESSION, None)
        A._gateway_notify_cbs.pop(SESSION, None)
        A._session_approved.pop(SESSION, None)
    try:
        yield SESSION
    finally:
        approval_context.reset_current_session_key(token)
        with A._lock:
            A._pending.pop(SESSION, None)


def _assert_undelivered(result: dict, session_key: str) -> None:
    assert result["approved"] is False
    assert result["approval_delivery"] == "unavailable"
    assert result["approval_pending"] is False
    assert result["status"] == "approval_required"
    assert result["outcome"] == "notify_failed"
    assert result["user_consent"] is False
    with A._lock:
        assert session_key not in A._pending, "an unanswerable request must not be parked"
    assert A.has_blocking_approval(session_key) is False
    assert A.get_pending_gateway_approval(session_key) is None


def test_dangerous_command_without_a_surface_is_undelivered(no_surface):
    _assert_undelivered(A.check_all_command_guards(DANGEROUS, "local"), no_surface)


def test_execute_code_without_a_surface_is_undelivered(no_surface):
    _assert_undelivered(A.check_execute_code_guard("print('hi')", "local"), no_surface)


def test_undelivered_result_tells_the_model_nothing_ran_and_not_to_wait(no_surface):
    result = A.check_all_command_guards(DANGEROUS, "local")
    assert result["message"].startswith("BLOCKED:")
    assert "did not run" in result["message"]
    assert "STOP" not in result["message"] and "pending" not in result["message"].lower()


def test_smart_denied_undelivered_stays_one_operation(no_surface, monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda _command, _description: "deny")

    for result in (A.check_all_command_guards(DANGEROUS, "local"),
                   A.check_execute_code_guard("print('hi')", "local")):
        _assert_undelivered(result, no_surface)
        assert result["smart_denied"] is True
        assert result["allow_permanent"] is False


def test_terminal_tool_reports_the_undelivered_approval(no_surface):
    from tools import terminal_tool

    payload = json.loads(terminal_tool.terminal_tool(command=DANGEROUS))

    assert payload["status"] == "blocked"
    assert payload["approval_pending"] is False
    assert payload["approval_delivery"] == "unavailable"
    assert payload["error"].startswith("BLOCKED:")
    with A._lock:
        assert no_surface not in A._pending


def test_explicit_user_deny_is_unchanged(no_surface):
    """A human who answered "deny" gets the denial result, not the undelivered one."""
    def deny_cb(_approval_data):
        with A._lock:
            entry = A._gateway_queues[no_surface][-1]
            entry.result = "deny"
            entry.event.set()

    with A._lock:
        A._gateway_notify_cbs[no_surface] = deny_cb
    try:
        result = A.check_all_command_guards(DANGEROUS, "local")
    finally:
        with A._lock:
            A._gateway_notify_cbs.pop(no_surface, None)

    assert result["approved"] is False
    assert result["outcome"] == "denied"
    assert "approval_delivery" not in result
    assert "approval_pending" not in result


def test_hard_policy_refusal_is_unchanged(no_surface):
    """A hardline command is refused by policy before any approval is attempted."""
    result = A.check_all_command_guards("rm -rf /", "local")

    assert result["approved"] is False
    assert "approval_delivery" not in result
    assert result.get("outcome") != "notify_failed"
    assert "hardline" in result["message"].lower() or "blocked" in result["message"].lower()
