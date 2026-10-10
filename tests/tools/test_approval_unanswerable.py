"""Approvals no permitted person can answer fail closed at once instead of waiting out the timeout."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools import approval as mod
from tools import approval_context
from tools.approval_gateway_wait import _await_gateway_decision

SESSION_KEY = "agent:main:telegram:dm:42"
REASON = "approval buttons in this chat are limited to admins, so it can't be granted here."


@pytest.fixture(autouse=True)
def _gateway_session(monkeypatch):
    stores = (mod._gateway_queues, mod._gateway_notify_cbs, mod._gateway_unanswerable,
              mod._session_approved, mod._pending)
    for store in stores:
        store.clear()
    for name in ("HERMES_YOLO_MODE", "HERMES_INTERACTIVE", "HERMES_CRON_SESSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION_KEY)
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"mode": "manual", "gateway_timeout": 300, "timeout": 300})
    yield
    for store in stores:
        store.clear()


def _register_unanswerable():
    notified = []
    mod.register_gateway_notify(SESSION_KEY, notified.append, unanswerable=REASON)
    return notified


def test_terminal_guard_denies_at_once_with_the_reason():
    notified = _register_unanswerable()

    start = time.monotonic()
    result = mod.check_all_command_guards("rm -rf .git", "local")

    assert time.monotonic() - start < 2.0
    assert result["approved"] is False
    assert result["outcome"] == "unanswerable"
    assert REASON in result["message"] and "do not retry" in result["message"].lower()
    assert notified == []
    assert not mod._gateway_queues.get(SESSION_KEY)


def test_execute_code_guard_denies_at_once():
    notified = _register_unanswerable()

    result = mod.check_execute_code_guard("import os; os.system('rm -rf /')", "local")

    assert result["outcome"] == "unanswerable"
    assert notified == []


def test_plugin_escalation_gate_denies_at_once():
    notified = _register_unanswerable()
    mod.approve_session(SESSION_KEY, "plugin-rule")

    result = mod.request_tool_approval("write_file", "plugin flagged this write", rule_key="plugin-rule")

    assert result["outcome"] == "unanswerable"
    assert notified == []


def test_session_grant_does_not_approve_an_unanswerable_session():
    _, pattern_key, _ = mod.detect_dangerous_command("rm -rf .git")
    mod.approve_session(SESSION_KEY, pattern_key)
    _register_unanswerable()

    assert mod.check_all_command_guards("rm -rf .git", "local")["outcome"] == "unanswerable"


def test_smart_approval_does_not_approve_an_unanswerable_session(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"mode": "smart", "gateway_timeout": 300, "timeout": 300})
    monkeypatch.setattr(mod, "_smart_verdict", lambda *a, **k: "approve")
    _register_unanswerable()

    assert mod.check_all_command_guards("rm -rf .git", "local")["outcome"] == "unanswerable"


def test_gateway_wait_reports_unanswerable_as_undeliverable():
    """Callers that don't know the outcome (MCP elicitation) still see a failed delivery."""
    notified = _register_unanswerable()

    decision = _await_gateway_decision(SESSION_KEY, notified.append, {"command": "x", "pattern_key": "k"})

    assert decision["notify_failed"] is True and decision["unanswerable"] == REASON
    assert notified == []


def test_marker_is_per_turn():
    _register_unanswerable()
    mod.unregister_gateway_notify(SESSION_KEY)
    assert mod.unanswerable_reason(SESSION_KEY) is None

    _register_unanswerable()
    mod.register_gateway_notify(SESSION_KEY, lambda data: None)  # next turn, answerable again
    assert mod.unanswerable_reason(SESSION_KEY) is None


def test_answerable_session_still_prompts(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_config",
                        lambda: {"mode": "manual", "gateway_timeout": 1, "timeout": 1})
    notified = []
    mod.register_gateway_notify(SESSION_KEY, notified.append)

    result = mod.check_all_command_guards("rm -rf .git", "local")

    assert result["outcome"] == "timeout"
    assert len(notified) == 1


# ── The gateway asks the adapter, per turn ────────────────────────────────────────────────────


class _Adapter:
    def __init__(self, reason):
        self._reason = reason

    def exec_approval_unanswerable(self, source):
        if isinstance(self._reason, Exception):
            raise self._reason
        return self._reason


class _NoHook:
    pass


@pytest.mark.parametrize("adapter,expected", [
    (_Adapter(REASON), REASON),
    (_Adapter(None), None),
    (_Adapter(""), None),
    (_Adapter(RuntimeError("boom")), None),
    (_NoHook(), None),
    (MagicMock(), None),
])
def test_unanswerable_approval_reason_asks_the_adapter_class(adapter, expected):
    from gateway.run_turn_runner_approval import unanswerable_approval_reason

    assert unanswerable_approval_reason(adapter, SimpleNamespace(chat_id="42")) == expected


def _run_turn(adapter, monkeypatch):
    from gateway.run_turn_runner import TurnRunner

    monkeypatch.setattr("gateway.run._wrap_current_message_with_observed_context", lambda msg, ctx: msg)
    seen = []
    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        session_key=SESSION_KEY, _status_adapter=adapter, _status_chat_id="42",
        message="hi", session_id="sid", source=SimpleNamespace(user_id="1", user_name="u", is_bot=False, chat_id="42"),
        title_user_message=None, persist_user_display_kind=None, persist_user_display_metadata=None,
        moa_config=None, inbound_message_id=None, mute_notification_reply=False,
    )
    runner._native_image_run_message = lambda: "hi"
    runner._approval_notify_sync = lambda data: None
    agent = SimpleNamespace(run_conversation=lambda msg, **kw: seen.append(mod.unanswerable_reason(SESSION_KEY)))
    runner._run_conversation_with_approval(agent, [], None, None, None)
    return seen


@pytest.mark.parametrize("adapter,expected", [(_Adapter(REASON), REASON), (_NoHook(), None)])
def test_turn_marks_the_session_only_while_running(adapter, expected, monkeypatch):
    assert _run_turn(adapter, monkeypatch) == [expected]
    assert mod.unanswerable_reason(SESSION_KEY) is None
