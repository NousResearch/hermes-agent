"""Approval-guard integration for task/run-scoped Kanban grants."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

import tools.approval as approval


@pytest.fixture(autouse=True)
def single_query_worker(monkeypatch: pytest.MonkeyPatch):
    approval._permanent_approved.clear()
    approval.clear_session("default")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_relay")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "claim-7")
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_ID", "apr-relay-test-001")
    monkeypatch.setenv("HERMES_PROFILE", "worker-o")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(approval, "_get_single_query_approval_mode", lambda: "deny")
    yield
    approval._permanent_approved.clear()
    approval.clear_session("default")


def _allow_tirith():
    return patch(
        "tools.tirith_security.check_command_security",
        return_value={"action": "allow", "findings": [], "summary": ""},
    )


def _receipt(operations: list[str]) -> dict:
    return {
        "approval_id": "apr-relay-test-001",
        "change_id": "chg-relay-test-001",
        "task_id": "t_relay",
        "run_id": 7,
        "operations": operations,
    }


def test_exact_delegated_terminal_operation_bypasses_no_other_gate(monkeypatch):
    command = "rm -rf /tmp/review-copy"
    is_dangerous, pattern_key, _ = approval.detect_dangerous_command(command)
    assert is_dangerous and pattern_key

    consume = Mock(return_value=_receipt([pattern_key]))
    monkeypatch.setattr(approval, "_consume_kanban_task_approval", consume)

    with _allow_tirith():
        result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is True
    assert result["delegated_approved"] is True
    assert result["approval_id"] == "apr-relay-test-001"
    consume.assert_called_once_with([pattern_key], action_class="command")


def test_unlisted_terminal_operation_stays_fail_closed(monkeypatch):
    command = "rm -rf /tmp/review-copy"
    monkeypatch.setattr(approval, "_consume_kanban_task_approval", Mock(return_value=None))

    with _allow_tirith():
        result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is False
    assert "single_query_mode" in result["message"]


def test_tirith_operation_must_be_independently_allowlisted(monkeypatch):
    fake_tirith = {
        "action": "block",
        "findings": [{
            "rule_id": "homograph-url",
            "severity": "HIGH",
            "title": "Homograph URL",
            "description": "lookalike characters",
        }],
        "summary": "homograph",
    }
    consume = Mock(return_value=None)
    monkeypatch.setattr(approval, "_consume_kanban_task_approval", consume)

    with (
        patch("tools.approval.detect_dangerous_command", return_value=(False, None, None)),
        patch("tools.tirith_security.check_command_security", return_value=fake_tirith),
    ):
        result = approval.check_all_command_guards("curl https://xn--example", "local")

    assert result["approved"] is False
    consume.assert_called_once_with(["tirith:homograph-url"], action_class="command")


def test_execute_code_requires_its_own_delegated_operation(monkeypatch):
    consume = Mock(return_value=_receipt(["execute_code"]))
    monkeypatch.setattr(approval, "_consume_kanban_task_approval", consume)

    result = approval.check_execute_code_guard("print('review')", "local")

    assert result["approved"] is True
    assert result["delegated_approved"] is True
    consume.assert_called_once_with(["execute_code"], action_class="command")


def test_plugin_escalation_requires_exact_delegated_rule(monkeypatch):
    operation = "plugin_rule:policy:review-read"
    consume = Mock(return_value=_receipt([operation]))
    monkeypatch.setattr(approval, "_consume_kanban_task_approval", consume)

    result = approval.request_tool_approval(
        "read_file",
        "read approved review artifact",
        rule_key="policy:review-read",
    )

    assert result["approved"] is True
    assert result["delegated_approved"] is True
    consume.assert_called_once_with([operation], action_class="command")


def test_hardline_floor_precedes_delegated_grant(monkeypatch):
    consume = Mock(return_value=_receipt(["anything"]))
    monkeypatch.setattr(approval, "_consume_kanban_task_approval", consume)

    result = approval.check_all_command_guards("rm -rf /", "local")

    assert result["approved"] is False
    consume.assert_not_called()


def test_user_deny_precedes_delegated_grant(monkeypatch):
    consume = Mock(return_value=_receipt(["anything"]))
    monkeypatch.setattr(approval, "_consume_kanban_task_approval", consume)
    monkeypatch.setattr(approval, "_match_user_deny_rule", lambda _command: "forbidden")

    result = approval.check_all_command_guards("rm -rf /tmp/review-copy", "local")

    assert result["approved"] is False
    assert result["user_deny"] is True
    assert "user-defined deny rule" in result["message"]
    consume.assert_not_called()
