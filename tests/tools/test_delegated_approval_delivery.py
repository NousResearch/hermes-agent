"""A background child refused once stays refused, and its result says it is blocked, not done.

A denied, withdrawn or timed-out approval is not consent. Inside one background delegation the next
gated action must not raise a fresh card (the child would otherwise re-ask until someone taps Allow),
and a child whose action never ran must not report ``completed`` to its parent or to a goal.
"""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import approval, async_delegation as ad
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    ad._reset_for_tests()
    home = set_hermes_home_override(str(tmp_path))
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval.approval_context, "_get_approval_timeout", lambda: 5)
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (None, False, True, False))
    yield monkeypatch
    ad.interrupt_all()
    ad._reset_for_tests()
    reset_hermes_home_override(home)


def _dispatch_child(key, operation):
    """Run ``operation`` as a background delegation of ``key``'s chat after its turn has ended."""
    done, ask, results = threading.Event(), threading.Event(), []

    def child():
        assert ask.wait(5)
        results.extend(operation())
        done.set()
        return {"status": "completed"}

    handle = ad.dispatch_async_delegation(goal="fixture", context=None, toolsets=None, role="leaf",
                                          model=None, session_key=key, runner=child)
    assert handle["status"] == "dispatched"
    approval.unregister_gateway_notify(key)  # the parent turn's finally
    ask.set()
    assert done.wait(10)
    return results


def _transport(cards, answer):
    def notify(data):
        cards.append(dict(data))
        answer(data)
    notify.background_notify = notify  # what a gateway chat turn publishes for its children
    return notify


@pytest.mark.parametrize("refusal", ["deny", "withdraw", "timeout"])
def test_a_refused_background_child_is_not_asked_again(gateway, refusal):
    key = "agent:test:refused-child"
    cards = []
    answers = {
        "deny": lambda d: approval.resolve_gateway_approval(key, "deny", request_id=d["request_id"]),
        "withdraw": lambda d: approval.withdraw_gateway_approval(key, d["request_id"], "fixture owner stopped"),
        "timeout": lambda d: None,
    }
    if refusal == "timeout":
        gateway.setattr(approval.approval_context, "_get_approval_timeout", lambda: 0.2)
    token = set_current_session_key(key)
    approval.register_gateway_notify(key, _transport(cards, answers[refusal]))
    try:
        first, second = _dispatch_child(key, lambda: [
            approval.check_execute_code_guard("print('first')", "local"),
            approval.check_execute_code_guard("print('second')", "local")])
    finally:
        approval.unregister_gateway_notify(key)
        reset_current_session_key(token)

    expected = {"deny": "denied", "withdraw": "cancelled", "timeout": "timeout"}[refusal]
    assert len(cards) == 1, "a refused child raised another approval card"
    assert first["approved"] is False and second["approved"] is False
    assert first["outcome"] == second["outcome"] == expected
    assert not approval.has_blocking_approval(key)


def test_one_childs_refusal_does_not_silence_a_sibling_delegation(gateway):
    key = "agent:test:sibling-child"
    cards = []

    def answer(data):
        approval.resolve_gateway_approval(key, "deny" if len(cards) == 1 else "once", request_id=data["request_id"])

    token = set_current_session_key(key)
    try:
        approval.register_gateway_notify(key, _transport(cards, answer))
        [refused] = _dispatch_child(key, lambda: [approval.check_execute_code_guard("print(1)", "local")])
        approval.register_gateway_notify(key, _transport(cards, answer))  # the chat's next turn
        [allowed] = _dispatch_child(key, lambda: [approval.check_execute_code_guard("print(2)", "local")])
    finally:
        approval.unregister_gateway_notify(key)
        reset_current_session_key(token)

    assert refused["outcome"] == "denied"
    assert allowed["approved"] is True
    assert len(cards) == 2


def test_a_denied_terminal_command_makes_the_child_blocked_not_completed(gateway):
    from tools import terminal_tool

    key = "agent:test:terminal-deny"
    token = set_current_session_key(key)
    approval.register_gateway_notify(key, _transport(
        [], lambda d: approval.resolve_gateway_approval(key, "deny", request_id=d["request_id"])))
    try:
        with pytest.raises(terminal_tool._Rejected) as rejected:
            terminal_tool._run_approval_guards("rm -rf .git", "local", {}, force=False)
    finally:
        approval.unregister_gateway_notify(key)
        reset_current_session_key(token)

    tool_result = json.loads(rejected.value.result_json)
    assert tool_result["status"] == "blocked"
    assert tool_result["approval_outcome"] == "denied"

    messages = [_call("c1", "rm -rf .git"), {"role": "tool", "tool_call_id": "c1", "content": rejected.value.result_json}]
    entry = _entry(messages, "The cleanup is done.")
    assert (entry["status"], entry["exit_reason"]) == ("blocked", "approval_required")
    assert entry["approval_blockers"][0]["approval_outcome"] == "denied"
    assert entry["error"]


@pytest.mark.parametrize("tool_result", [
    {"status": "pending_approval", "approval_pending": True, "command": "fixture"},
    *({"status": "blocked", "approval_outcome": o} for o in ("denied", "cancelled", "timeout", "notify_failed")),
])
def test_an_unexecuted_approval_is_never_reported_as_completed(tool_result):
    entry = _entry([_call("c1", "fixture"), _result("c1", tool_result)], "Done")
    assert entry["status"] == "blocked"
    assert entry["exit_reason"] == "approval_required"
    assert ad._batch_status({"results": [entry, {"status": "completed"}]}) == "blocked"


def test_a_blocked_command_without_an_approval_outcome_is_not_an_approval_blocker():
    entry = _entry([_call("c1", "cd /missing"), _result("c1", {"status": "blocked", "error": "workdir"})], "Done")
    assert entry["status"] == "completed"
    assert "approval_blockers" not in entry


def test_only_a_successful_rerun_of_the_same_action_clears_its_blocker():
    from tools.delegate_tool_child_run import _approval_blockers
    messages = [_call("a", "fixture"), _result("a", {"status": "blocked", "approval_outcome": "denied"}),
                _call("b", "read-only"), _result("b", {"exit_code": 0})]
    assert len(_approval_blockers(messages)) == 1
    messages += [_call("c", "fixture"), _result("c", {"exit_code": 0})]
    assert _approval_blockers(messages) == []


def test_a_blocked_child_stays_visible_with_its_blockers_after_it_ends(gateway):
    from tools.process_registry import process_registry

    blocker = {"status": "blocked", "approval_outcome": "denied"}
    handle = ad.dispatch_async_delegation_batch(
        goals=["deploy", "docs"], context=None, toolsets=None, role="leaf", model=None, session_key="",
        parent_session_id="parent-1", max_async_children=2,
        runner=lambda: {"results": [
            {"task_index": 0, "status": "blocked", "approval_blockers": [blocker], "error": "approval"},
            {"task_index": 1, "status": "completed", "summary": "ok"}], "total_duration_seconds": 0})
    assert handle["status"] == "dispatched"
    deadline = time.monotonic() + 10
    while process_registry.completion_queue.get(timeout=10)["delegation_id"] != handle["delegation_id"]:
        assert time.monotonic() < deadline

    [row] = ad.failed_delegations_for_session(parent_session_id="parent-1")
    assert (row["delegation_id"], row["goal"], row["status"]) == (handle["delegation_id"], "deploy", "blocked")
    assert row["approval_blockers"] == [blocker]


def _call(call_id, command):
    return {"role": "assistant", "tool_calls": [
        {"id": call_id, "function": {"name": "terminal", "arguments": json.dumps({"command": command})}}]}


def _result(call_id, data):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(data)}


def _entry(messages, final_response):
    return _build_result_entry(SimpleNamespace(), {"completed": True, "final_response": final_response,
                                                   "messages": messages}, 0, 0.1, _SchemaOutcome(None, None, [], 0))
