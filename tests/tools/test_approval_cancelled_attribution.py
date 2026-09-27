"""A gateway approval wait that ends without a user answer must not be reported as a user deny.

Teardown paths (a parent's ``delegate_task`` finishing, ``unregister_gateway_notify`` at turn
end, a ``/stop``) end the wait fail-closed, but the tool result carries ``outcome="cancelled"``
and the cause instead of "denied by user" (#112026, #22992).
"""
import threading
import time

import pytest

from tools import approval as mod
from tools import approval_context
from tools.interrupt import set_interrupt

SESSION_KEY = "test-cancelled-attribution"


@pytest.fixture
def gateway_session(monkeypatch):
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    for k in ("HERMES_CRON_SESSION", "HERMES_YOLO_MODE", "HERMES_INTERACTIVE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION_KEY)
    # ``--yolo`` is frozen at import from HERMES_YOLO_MODE; a host shell running yolo must not
    # auto-approve the gate under test.
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual", "timeout": 60})
    hooks = []
    original = approval_context._fire_approval_hook
    monkeypatch.setattr(approval_context, "_fire_approval_hook",
                        lambda name, **kw: (hooks.append((name, kw)), original(name, **kw)))
    yield hooks
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()


def _run_gate_until_pending():
    """Start the real command gate on a worker thread and return (thread, tid, result_holder)
    once the prompt has been handed to the gateway notifier."""
    notified = threading.Event()
    mod.register_gateway_notify(SESSION_KEY, lambda _data: notified.set())
    holder = {}

    def worker():
        holder["tid"] = threading.get_ident()
        holder["result"] = mod.check_all_command_guards("rm -rf .git", "local")

    thread = threading.Thread(target=worker)
    thread.start()
    assert notified.wait(timeout=10), "approval was never enqueued"
    return thread, holder


def _assert_withdrawn(result, cause):
    assert result["approved"] is False
    assert result.get("user_consent") is False
    assert result["outcome"] == "cancelled"
    assert "denied by user" not in result["message"].lower()
    assert cause in result["message"]
    assert "NOT consented" in result["message"]  # still fail-closed for the model


def test_teardown_interrupt_reports_cause_not_user_deny(gateway_session):
    """A delegation teardown raises the child's interrupt bit with a fixed cause; the approval
    result relays that cause and does not claim the user denied the command."""
    thread, holder = _run_gate_until_pending()
    set_interrupt(True, holder["tid"], reason="parent delegation ended")
    try:
        thread.join(timeout=10)
    finally:
        set_interrupt(False, holder["tid"])
    assert not thread.is_alive()
    _assert_withdrawn(holder["result"], "parent delegation ended")
    posts = [kw for name, kw in gateway_session if name == "post_approval_response"]
    assert posts[-1]["choice"] == "cancelled"
    assert not mod.has_blocking_approval(SESSION_KEY)


def test_turn_end_unregister_reports_withdrawn_prompt(gateway_session):
    """``unregister_gateway_notify`` at turn end wakes the wait with no decision: the result is
    a withdrawn prompt, while an explicit /deny on the same gate still reads as a user deny."""
    thread, holder = _run_gate_until_pending()
    mod.unregister_gateway_notify(SESSION_KEY)
    thread.join(timeout=10)
    assert not thread.is_alive()
    _assert_withdrawn(holder["result"], "the turn ended before the prompt was answered")

    thread, holder = _run_gate_until_pending()
    mod.resolve_gateway_approval(SESSION_KEY, "deny")
    thread.join(timeout=10)
    denied = holder["result"]
    assert denied["outcome"] == "denied"
    assert "denied by user" in denied["message"]


def test_session_boundary_teardown_reports_withdrawn_prompt(gateway_session):
    """``clear_session`` (/new, /reset, auto-reset) wakes the wait with no decision: the result
    is a withdrawn prompt, not a user deny."""
    thread, holder = _run_gate_until_pending()
    mod.clear_session(SESSION_KEY)
    thread.join(timeout=10)
    assert not thread.is_alive()
    _assert_withdrawn(holder["result"], "the session ended before the prompt was answered")


def test_coalesced_follower_inherits_the_leaders_cancellation(gateway_session):
    """A follower coalesced onto an interrupted leader wakes with the leader's cause, not a deny."""
    leader_thread, leader = _run_gate_until_pending()
    follower = {}
    follower_thread = threading.Thread(
        target=lambda: follower.__setitem__("result", mod.check_all_command_guards("rm -rf .git", "local")))
    follower_thread.start()  # identical command → coalesces onto the leader, no second prompt
    deadline = time.monotonic() + 10
    while not any(n == "pre_approval_request" and kw.get("coalesced") for n, kw in gateway_session):
        assert time.monotonic() < deadline, "follower never coalesced onto the leader"
        time.sleep(0.05)
    set_interrupt(True, leader["tid"], reason="parent delegation ended")
    try:
        leader_thread.join(timeout=10)
        follower_thread.join(timeout=10)
    finally:
        set_interrupt(False, leader["tid"])
    assert not leader_thread.is_alive() and not follower_thread.is_alive()
    _assert_withdrawn(leader["result"], "parent delegation ended")
    _assert_withdrawn(follower["result"], "parent delegation ended")


@pytest.fixture
def cli_session(monkeypatch):
    mod._session_approved.clear()
    for k in ("HERMES_CRON_SESSION", "HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual", "timeout": 60})
    hooks = []
    monkeypatch.setattr(approval_context, "_fire_approval_hook", lambda name, **kw: hooks.append((name, kw)))
    return hooks


def test_cli_callback_failure_reports_undelivered_prompt_not_user_deny(cli_session):
    """The CLI residual of #22992: a prompt that never reached a human (the approval callback
    raised) is 'cancelled' with its cause, not 'User denied this command'."""
    def broken_callback(command, description, **kwargs):
        raise TypeError("callback signature mismatch")

    result = mod.check_all_command_guards("rm -rf .git", "local", approval_callback=broken_callback)
    _assert_withdrawn(result, "the approval callback failed: TypeError")
    assert "user denied" not in result["message"].lower()
    posts = [kw for name, kw in cli_session if name == "post_approval_response"]
    assert posts[-1]["choice"] == "cancelled"
