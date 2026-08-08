"""Behavior contracts for goal-scoped approval persistence and risk changes."""

from __future__ import annotations

import contextvars
import logging


def _active_envelope() -> dict:
    return {
        "authorization_id": "fixture-auth",
        "goal": "complete the routine release workflow",
        "goal_status": "active",
        "scope": "AUTO_EXECUTE",
        "created_at": 1.0,
    }


def test_goal_authorization_roundtrip_and_status() -> None:
    from hermes_cli.goals import GoalManager, GoalState

    state = GoalState(
        goal="ship safely",
        authorization_id="auth-123",
        authorization_scope="AUTO_EXECUTE",
        authorization_created_at=123.0,
    )
    restored = GoalState.from_json(state.to_json())
    assert restored.authorization_envelope() == {
        "authorization_id": "auth-123",
        "goal": "ship safely",
        "goal_status": "active",
        "scope": "AUTO_EXECUTE",
        "created_at": 123.0,
    }

    manager = GoalManager.__new__(GoalManager)
    manager._state = restored
    line = manager.approval_status_line()
    assert "AUTO_EXECUTE=authorized" in line
    assert "OWNER_APPROVAL=required" in line
    assert "DENY=always blocked" in line
    assert "delegation=inherited" in line


def test_delegated_context_inherits_goal_authorization() -> None:
    from tools.approval import (
        get_goal_authorization,
        reset_goal_authorization,
        set_goal_authorization,
    )

    token = set_goal_authorization(_active_envelope())
    try:
        child_context = contextvars.copy_context()
        inherited = child_context.run(get_goal_authorization)
        assert inherited is not None
        assert inherited["authorization_id"] == "fixture-auth"
    finally:
        reset_goal_authorization(token)


def test_twenty_routine_operations_need_zero_prompts_then_risk_change_needs_one(
    monkeypatch, caplog
) -> None:
    from tools import approval

    routine_operations = [
        "pwd",
        "rg -n TODO src tests",
        "git status --short",
        "sed -n '1,80p' app.py",
        "python -m py_compile app.py",
        "python -m pytest -q tests/test_app.py",
        "sudo -n true",
        "sudo -n systemctl daemon-reload",
        "sudo -n systemctl restart demo.service",
        "systemctl status demo.service --no-pager",
        "git add app.py tests/test_app.py",
        "git commit -m 'fix: routine repair'",
        "git push origin HEAD",
        "gh pr checks 23",
        "gh issue edit 23 --add-label completed",
        "git diff --check",
        "journalctl -u demo.service -n 20 --no-pager",
        "sudo -n visudo -cf /etc/sudoers",
        "python -m pytest -q tests/test_app.py",
        "gh pr comment 23 --body 'routine verification complete'",
    ]
    prompt_count = 0

    def approve_once(*_args, **_kwargs):
        nonlocal prompt_count
        prompt_count += 1
        return "once"

    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    auth_token = approval.set_goal_authorization(_active_envelope())
    session_token = approval.set_current_session_key("goal-e2e")
    interactive_token = approval.set_hermes_interactive_context(True)
    caplog.set_level(logging.INFO, logger="tools.approval")
    try:
        for command in routine_operations:
            decision = approval.check_all_command_guards(
                command, "local", approval_callback=approve_once
            )
            assert decision["approved"] is True, command
        assert prompt_count == 0

        high_risk = approval.check_all_command_guards(
            "git push --force origin main", "local", approval_callback=approve_once
        )
        assert high_risk["approved"] is True
        assert prompt_count == 1

        denied = approval.check_all_command_guards(
            "shutdown -h now", "local", approval_callback=approve_once
        )
        assert denied["approved"] is False
        assert denied.get("hardline") is True
        assert prompt_count == 1
    finally:
        approval.reset_hermes_interactive_context(interactive_token)
        approval.reset_current_session_key(session_token)
        approval.reset_goal_authorization(auth_token)

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("AUTO_EXECUTE reason=") for message in messages)
    assert any(message.startswith("OWNER_APPROVAL reason=") for message in messages)
    assert any(message.startswith("DENY reason=") for message in messages)


def test_goal_authorization_does_not_bypass_tirith_warning(monkeypatch) -> None:
    from tools import approval
    from tools import tirith_security

    prompt_count = 0

    def approve_once(*_args, **_kwargs):
        nonlocal prompt_count
        prompt_count += 1
        return "once"

    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(
        tirith_security,
        "check_command_security",
        lambda _command: {
            "action": "warn",
            "findings": [
                {
                    "rule_id": "fixture-content-risk",
                    "severity": "HIGH",
                    "title": "Fixture content security risk",
                    "description": "Requires explicit owner review.",
                }
            ],
            "summary": "fixture warning",
        },
    )
    auth_token = approval.set_goal_authorization(_active_envelope())
    session_token = approval.set_current_session_key("goal-tirith-e2e")
    interactive_token = approval.set_hermes_interactive_context(True)
    try:
        decision = approval.check_all_command_guards(
            "printf safe", "local", approval_callback=approve_once
        )
        assert decision["approved"] is True
        assert prompt_count == 1
        assert decision.get("goal_authorized") is not True
    finally:
        approval.reset_hermes_interactive_context(interactive_token)
        approval.reset_current_session_key(session_token)
        approval.reset_goal_authorization(auth_token)


def test_real_money_language_is_always_owner_gated() -> None:
    from tools import approval

    token = approval.set_goal_authorization(_active_envelope())
    try:
        for action in (
            "place bet on fixture",
            "withdraw winnings",
            "purchase subscription",
            "transfer funds",
        ):
            assert approval.classify_goal_action(action) == (
                approval.OWNER_APPROVAL,
                "real-money or payment action",
            )
    finally:
        approval.reset_goal_authorization(token)


def test_bounded_absolute_workspace_pythonpath_is_goal_authorized(monkeypatch) -> None:
    from tools import approval
    from tools import tirith_security

    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(
        tirith_security,
        "check_command_security",
        lambda _command: {
            "action": "block",
            "findings": [
                {
                    "rule_id": "interpreter_hijack_env",
                    "severity": "HIGH",
                    "title": "Interpreter hijack environment variable: PYTHONPATH",
                    "description": "fixture",
                }
            ],
            "summary": "fixture",
        },
    )
    auth_token = approval.set_goal_authorization(_active_envelope())
    session_token = approval.set_current_session_key("goal-pythonpath-e2e")
    interactive_token = approval.set_hermes_interactive_context(True)
    prompts = 0

    def callback(*_args, **_kwargs):
        nonlocal prompts
        prompts += 1
        return "once"

    try:
        safe = approval.check_all_command_guards(
            "export PYTHONPATH=/home/ubuntu/OddsEdge/src; python -m pytest -q",
            "local",
            approval_callback=callback,
        )
        assert safe["approved"] is True
        assert safe.get("goal_authorized") is True
        assert prompts == 0

        unsafe = approval.check_all_command_guards(
            "PYTHONPATH=.:/tmp/plugin python task.py",
            "local",
            approval_callback=callback,
        )
        assert unsafe["approved"] is True
        assert unsafe.get("goal_authorized") is not True
        assert prompts == 1
    finally:
        approval.reset_hermes_interactive_context(interactive_token)
        approval.reset_current_session_key(session_token)
        approval.reset_goal_authorization(auth_token)
