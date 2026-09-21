"""The smart-approval retry, driven through the REAL guard.

``_smart_approve``'s retry contract only matters if the real gate — ``check_all_command_guards`` /
``check_execute_code_guard`` → ``_human_decision`` → ``_smart_gate`` → ``_smart_verdict`` → the
guardian call — still recovers a truncated answer and still hands a completed abstention to the
human. Only the LLM boundary (``agent.auxiliary_client.call_llm``) is faked here, so the retry, the
verdict mapping, the hook/redaction wrapper and the escalation path all run for real
(tools/AGENTS.md: approval/security-boundary tools are E2E'd with real imports against a temp
``HERMES_HOME``).
"""

from __future__ import annotations

import pytest

import agent.auxiliary_client as aux
from tools import approval as A
from tools import approval_context
import tools.approval_detection as approval_detection

SESSION_KEY = "smart-retry-gate-session"


class _Resp:
    """Minimal LLM response: only the fields the guardian reads."""

    def __init__(self, content, finish_reason):
        choice = type("_Choice", (), {})()
        choice.message = type("_Msg", (), {"content": content})()
        choice.finish_reason = finish_reason
        self.choices = [choice]


@pytest.fixture
def smart_gate_session(monkeypatch):
    """A gateway session in ``smart`` mode with every command flagged as dangerous."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    for var in ("HERMES_INTERACTIVE", "HERMES_CRON_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    flagged = lambda command: (True, "gate-retry-danger", f"risk:{command}")  # noqa: E731
    monkeypatch.setattr(A, "detect_dangerous_command", flagged)
    monkeypatch.setattr(approval_detection, "detect_dangerous_command", flagged)
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )

    token = approval_context.set_current_session_key(SESSION_KEY)
    A._reset_denials(SESSION_KEY)
    with A._lock:
        for key in ("gate-retry-danger", "execute_code"):
            A._permanent_approved.discard(key)
            A._session_approved.get(SESSION_KEY, set()).discard(key)
        A._gateway_queues.pop(SESSION_KEY, None)
        A._gateway_notify_cbs.pop(SESSION_KEY, None)
    try:
        yield SESSION_KEY
    finally:
        approval_context.reset_current_session_key(token)
        A._reset_denials(SESSION_KEY)
        with A._lock:
            A._gateway_queues.pop(SESSION_KEY, None)
            A._gateway_notify_cbs.pop(SESSION_KEY, None)


def _guardian(monkeypatch, responses):
    """Fake only the LLM boundary; returns the recorded call kwargs."""
    calls = []

    def fake(*_args, **kwargs):
        calls.append(kwargs)
        return _Resp(*responses[min(len(calls) - 1, len(responses) - 1)])

    monkeypatch.setattr(aux, "call_llm", fake)
    return calls


def _human_answers(choice, asked):
    """Answer the pending gateway approval with *choice*, recording that a human was asked."""

    def cb(_approval_data):
        asked.append(choice)
        with A._lock:
            entries = A._gateway_queues.get(SESSION_KEY, [])
            if entries:
                entries[-1].result = choice
                entries[-1].event.set()

    with A._lock:
        A._gateway_notify_cbs[SESSION_KEY] = cb


# ── terminal guard ───────────────────────────────────────────────────────────


def test_truncated_answer_recovers_through_the_real_terminal_guard(smart_gate_session, monkeypatch):
    """Empty + finish_reason="length" then APPROVE → the command runs, no human is asked."""
    calls = _guardian(monkeypatch, [("", "length"), ("APPROVE", "stop")])
    asked = []
    _human_answers("deny", asked)

    result = A.check_all_command_guards("rm -rf /tmp/gate-retry-target", "local")

    assert result["approved"] is True
    assert result.get("smart_approved") is True
    assert len(calls) == 2, "the truncation must be retried once"
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]
    assert asked == [], "a recovered truncation must not reach the human prompt"


def test_completed_abstention_reaches_the_human_instead_of_a_second_verdict(smart_gate_session, monkeypatch):
    """A non-empty abstention must NOT be retried into an auto-approval.

    The second response would be accepted by the old (over-broad) retry; here it must never be
    requested: the guardian is asked once and the human decides.
    """
    calls = _guardian(monkeypatch, [("I cannot determine this", "stop"), ("APPROVE", "stop")])
    asked = []
    _human_answers("deny", asked)

    result = A.check_all_command_guards("rm -rf /tmp/gate-retry-target", "local")

    assert len(calls) == 1, "a completed abstention must not buy a second guardian verdict"
    assert asked == ["deny"], "the completed abstention must be handed to the human"
    assert result["approved"] is False
    assert "smart_approved" not in result


def test_human_approval_of_an_abstention_is_not_reported_as_a_smart_approval(smart_gate_session, monkeypatch):
    """When the human approves, the approval is theirs — not the guardian's second call."""
    calls = _guardian(monkeypatch, [("I cannot determine this", "stop"), ("APPROVE", "stop")])
    asked = []
    _human_answers("once", asked)

    result = A.check_all_command_guards("rm -rf /tmp/gate-retry-target", "local")

    assert asked == ["once"]
    assert len(calls) == 1
    assert result["approved"] is True
    assert "smart_approved" not in result


# ── execute_code guard ───────────────────────────────────────────────────────


def test_completed_abstention_reaches_the_human_via_the_execute_code_guard(smart_gate_session, monkeypatch):
    """The whole-script guard shares the retry contract (tools/AGENTS.md: cover both surfaces)."""
    calls = _guardian(monkeypatch, [("I cannot determine this", "stop"), ("APPROVE", "stop")])
    asked = []
    _human_answers("deny", asked)

    result = A.check_execute_code_guard("print('gate-retry')", "local")

    assert len(calls) == 1
    assert asked == ["deny"]
    assert result["approved"] is False
    assert "smart_approved" not in result
