"""Delegated sub-agent approvals on a gateway session reach the parent chat.

A child runs under its own session identity (``platform=subagent``) while the notify
callback is registered only for the parent turn. Async delegation returns before the
child asks, and that turn then unregisters the callback. The prompt has to post on the
parent session anyway, and a child with no route must fail closed instead of returning
``pending_approval`` that nothing can resolve.
"""

from __future__ import annotations

import contextvars
import json
import threading

import pytest

from agent.delegation_context import delegated_child_context
from gateway.session_context import clear_session_vars, set_session_vars
from tools import approval as A
from tools import approval_context
from tools.approval_detection import detect_dangerous_command
from tools.delegate_tool_config import _subagent_auto_deny

PARENT = "agent:main:discord:channel:1"
CHILD = "subagent:echo-hi"
TASK = "echo hi from the sub-agent"
COMMAND = "sh -c 'echo hi'"


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    for name in ("HERMES_INTERACTIVE", "HERMES_CRON_SESSION", "HERMES_YOLO_MODE", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    # Short enough that a failed test cannot sit on the approval wait, long enough to observe the block.
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 15)
    monkeypatch.setattr("tools.delegate_tool._load_config", dict)
    saved = (
        set(A._permanent_approved), dict(A._session_approved), dict(A._pending),
        dict(A._gateway_notify_cbs), dict(A._gateway_queues), dict(A._gateway_notify_pins),
        set(A._gateway_notify_release_pending),
    )
    A._permanent_approved.clear()
    A._session_approved.clear()
    A._pending.clear()
    A._gateway_notify_cbs.clear()
    A._gateway_queues.clear()
    A._gateway_notify_pins.clear()
    A._gateway_notify_release_pending.clear()
    tokens = set_session_vars(platform="discord", session_key=PARENT, session_id=PARENT)
    key_token = approval_context.set_current_session_key(PARENT)
    try:
        yield
    finally:
        approval_context.reset_current_session_key(key_token)
        clear_session_vars(tokens)
        (perm, session, pending, cbs, queues, pins, release) = saved
        A._permanent_approved.clear()
        A._permanent_approved.update(perm)
        A._session_approved.clear()
        A._session_approved.update(session)
        A._pending.clear()
        A._pending.update(pending)
        A._gateway_notify_cbs.clear()
        A._gateway_notify_cbs.update(cbs)
        A._gateway_queues.clear()
        A._gateway_queues.update(queues)
        A._gateway_notify_pins.clear()
        A._gateway_notify_pins.update(pins)
        A._gateway_notify_release_pending.clear()
        A._gateway_notify_release_pending.update(release)


def _prompts():
    seen = []
    notified = threading.Event()

    def notify(data):
        seen.append(dict(data))
        notified.set()

    return seen, notified, notify


def _run_child(fn):
    """Run ``fn`` on a worker that inherits nothing, the way a sub-agent thread does."""
    holder = {}
    done = threading.Event()

    def worker():
        try:
            from tools.terminal_tool import set_approval_callback
            set_approval_callback(_subagent_auto_deny)
            tokens = set_session_vars(platform="discord", session_key=PARENT, session_id=PARENT)
            parent_key = approval_context.set_current_session_key(PARENT)
            try:
                with delegated_child_context("child-session"), A.delegated_gateway_approval_scope(TASK):
                    child_key = approval_context.set_current_session_key(CHILD)
                    try:
                        holder["result"] = fn()
                    finally:
                        approval_context.reset_current_session_key(child_key)
            finally:
                approval_context.reset_current_session_key(parent_key)
                clear_session_vars(tokens)
                set_approval_callback(None)
        except Exception as exc:  # surfaced by the assertion on holder
            holder["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    return holder, done, thread


def test_dangerous_shell_c_is_the_pattern_under_test():
    dangerous, _key, description = detect_dangerous_command(COMMAND)
    assert dangerous is True
    assert "c" in description.lower()


@pytest.mark.platforms("posix")
def test_child_blocks_on_parent_callback_and_approve_runs_the_command(gateway):
    seen, notified, notify = _prompts()
    A.register_gateway_notify(PARENT, notify)
    holder, done, thread = _run_child(
        lambda: __import__("tools.terminal_tool", fromlist=["terminal_tool"]).terminal_tool(
            COMMAND, task_id="subagent-approval-echo",
        )
    )
    assert notified.wait(timeout=10), "parent notify callback was not called"
    assert len(seen) == 1
    assert seen[0]["delegated_child"] is True
    assert seen[0]["subagent_task"] == TASK
    assert seen[0]["description"].startswith(f"sub-agent: {TASK}")
    assert not done.is_set()
    assert A.list_gateway_approvals(CHILD) == []
    assert A.list_gateway_approvals(PARENT)

    assert A.resolve_gateway_approval(PARENT, "once") == 1
    assert done.wait(timeout=10)
    thread.join(timeout=10)
    assert "error" not in holder, holder.get("error")
    result = json.loads(holder["result"])
    assert result["exit_code"] == 0, result
    assert "hi" in result["output"]
    assert not A.has_blocking_approval(PARENT)
    assert PARENT not in A._pending and CHILD not in A._pending


def test_child_still_prompts_after_the_parent_turn_unregisters(gateway):
    seen, notified, notify = _prompts()
    A.register_gateway_notify(PARENT, notify)
    assert A.pin_gateway_notify(PARENT) is True
    A.unregister_gateway_notify(PARENT)
    assert A._gateway_notify_cb(PARENT) is not None

    holder, done, thread = _run_child(lambda: A.check_all_command_guards(COMMAND, "local"))
    assert notified.wait(timeout=10)
    assert not done.is_set()
    assert seen[0]["delegated_child"] is True

    assert A.resolve_gateway_approval(PARENT, "once") == 1
    assert done.wait(timeout=10)
    thread.join(timeout=10)
    assert holder["result"]["approved"] is True
    A.unpin_gateway_notify(PARENT)
    assert A._gateway_notify_cb(PARENT) is None
    assert not A.has_blocking_approval(PARENT)


def test_deny_and_timeout_are_blocked_and_leave_nothing_pending(gateway, monkeypatch):
    seen, notified, notify = _prompts()
    A.register_gateway_notify(PARENT, notify)
    holder, done, thread = _run_child(lambda: A.check_all_command_guards(COMMAND, "local"))
    assert notified.wait(timeout=10)
    assert A.resolve_gateway_approval(PARENT, "deny", reason="no") == 1
    assert done.wait(timeout=10)
    thread.join(timeout=10)
    denied = holder["result"]
    assert denied["approved"] is False and denied["outcome"] == "denied"
    assert denied["message"].startswith("BLOCKED:")
    assert denied.get("status") != "pending_approval"
    assert "no" in denied["message"]
    assert not A.has_blocking_approval(PARENT)
    assert PARENT not in A._pending

    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 0)
    seen.clear()
    notified.clear()
    holder, done, thread = _run_child(lambda: A.check_all_command_guards(COMMAND, "local"))
    assert notified.wait(timeout=10)
    assert done.wait(timeout=10)
    thread.join(timeout=10)
    timed_out = holder["result"]
    assert timed_out["approved"] is False and timed_out["outcome"] == "timeout"
    assert timed_out["message"].startswith("BLOCKED:")
    assert timed_out.get("status") != "pending_approval"
    assert not A.has_blocking_approval(PARENT)
    assert len(seen) == 1


def test_no_parent_callback_is_a_definitive_block(gateway):
    with delegated_child_context("child-session"), A.delegated_gateway_approval_scope(TASK):
        child_key = approval_context.set_current_session_key(CHILD)
        try:
            blocked = A.check_all_command_guards(COMMAND, "local")
        finally:
            approval_context.reset_current_session_key(child_key)
    assert blocked["approved"] is False and blocked["outcome"] == "blocked"
    assert blocked.get("status") != "pending_approval"
    assert blocked.get("approval_pending") is not True
    assert "main session" in blocked["message"]
    assert "subagent_auto_approve" in blocked["message"]
    assert TASK[:40] in blocked["message"]
    assert "denied by user" not in blocked["message"].lower()
    assert A.list_gateway_approvals(PARENT) == []
    assert A.list_gateway_approvals(CHILD) == []
    assert PARENT not in A._pending and CHILD not in A._pending

    # The parent session itself, with no notifier yet, still parks a reviewable pending approval.
    approval_context_key = approval_context.set_current_session_key(PARENT)
    try:
        pending = A.check_all_command_guards(COMMAND, "local")
    finally:
        approval_context.reset_current_session_key(approval_context_key)
    assert pending.get("status") == "pending_approval"
    assert pending.get("approval_pending") is True
    assert A._pending[PARENT]["command"] == COMMAND


def test_subagent_auto_approve_skips_the_prompt(gateway, monkeypatch):
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {"subagent_auto_approve": True})
    seen, _notified, notify = _prompts()
    A.register_gateway_notify(PARENT, notify)
    with delegated_child_context("child-session"), A.delegated_gateway_approval_scope(TASK):
        result = A.check_all_command_guards(COMMAND, "local")
    assert result["approved"] is True and result.get("user_approved") is True
    assert seen == []
    assert not A.has_blocking_approval(PARENT)


def test_unregister_while_pinned_cancels_only_the_parent_wait(gateway):
    parent_notified = threading.Event()
    child_notified = threading.Event()

    def notify(data):
        (child_notified if data.get("delegated_child") else parent_notified).set()

    A.register_gateway_notify(PARENT, notify)
    assert A.pin_gateway_notify(PARENT) is True

    parent_holder = {}
    parent_done = threading.Event()

    def parent_wait():
        parent_holder["result"] = A.check_all_command_guards(COMMAND, "local")
        parent_done.set()

    parent = threading.Thread(target=contextvars.copy_context().run, args=(parent_wait,))
    parent.start()
    assert parent_notified.wait(timeout=10)

    holder, done, thread = _run_child(lambda: A.check_all_command_guards("bash -lc 'echo hi'", "local"))
    assert child_notified.wait(timeout=10)

    A.unregister_gateway_notify(PARENT)
    assert parent_done.wait(timeout=10)
    parent.join(timeout=10)
    assert parent_holder["result"]["outcome"] == "cancelled"
    assert not done.is_set()
    assert A._gateway_notify_cb(PARENT) is not None

    assert A.resolve_gateway_approval(PARENT, "deny") == 1
    assert done.wait(timeout=10)
    thread.join(timeout=10)
    assert holder["result"]["outcome"] == "denied"
    A.unpin_gateway_notify(PARENT)
    assert A._gateway_notify_cb(PARENT) is None


def test_unscoped_hold_does_not_pin_the_default_fallback(gateway):
    """An empty approval context must not pin the literal key ``default``."""
    from gateway.session_context import scoped_session_key
    from tools import async_delegation as ad
    from tools.approval_context import reset_current_session_key, set_current_session_key

    token = set_current_session_key("")
    try:
        with scoped_session_key(""):
            A.register_gateway_notify("default", lambda _data: None)
            assert ad._hold_gateway_approval(PARENT) == ("", False)
            assert A._gateway_notify_pins.get("default", 0) == 0
            A.unpin_gateway_notify("default")
            assert A._gateway_notify_cb("default") is not None
    finally:
        reset_current_session_key(token)


def test_packed_button_token_resolves_only_that_prompt(gateway):
    """A click on one card must not approve a different command still queued on the session."""
    import threading

    from tools.approval import pack_button_approval, resolve_button_approval

    posted = [threading.Event(), threading.Event()]
    seen: list[dict] = []

    def notify(data):
        seen.append(dict(data))
        posted[len(seen) - 1].set()

    holders: list[dict] = [{}, {}]
    commands = ("rm -rf build", "sh -c 'echo hi'")

    def wait(idx):
        holders[idx]["result"] = A._await_gateway_decision(
            PARENT, notify, {"command": commands[idx], "description": "d", "pattern_key": "k",
                             "pattern_keys": ["k"]},
        )

    threads = [threading.Thread(target=wait, args=(i,)) for i in (0, 1)]
    try:
        threads[0].start()
        assert posted[0].wait(timeout=10)
        threads[1].start()
        assert posted[1].wait(timeout=10)
        assert {item["command"] for item in seen} == set(commands)
        assert resolve_button_approval(pack_button_approval(PARENT, "not-a-live-id"), "once") == 0
        assert A.has_blocking_approval(PARENT)
        second = seen[1]["request_id"]
        assert resolve_button_approval(pack_button_approval(PARENT, second), "deny") == 1
        threads[1].join(timeout=10)
        assert holders[1]["result"]["choice"] == "deny"
        assert threads[0].is_alive()
        assert resolve_button_approval(pack_button_approval(PARENT, seen[0]["request_id"]), "deny") == 1
        threads[0].join(timeout=10)
        assert holders[0]["result"]["choice"] == "deny"
        assert not A.has_blocking_approval(PARENT)
    finally:
        for entry_id in (item.get("request_id") for item in seen):
            if entry_id:
                resolve_button_approval(pack_button_approval(PARENT, entry_id), "deny")
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=10)


def test_async_dispatch_holds_the_notifier_until_the_child_finishes(gateway):
    from tests.waiting import wait_until
    from tools import async_delegation as ad

    seen_cb = {}
    entered = threading.Event()
    release = threading.Event()

    def runner():
        entered.set()
        assert release.wait(timeout=10)
        seen_cb["during"] = A._gateway_notify_cb(PARENT)
        return {"status": "completed", "summary": "done"}

    A.register_gateway_notify(PARENT, lambda _data: None)
    ad._reset_for_tests()
    try:
        handle = ad.dispatch_async_delegation(
            goal=TASK, context=None, toolsets=None, role="leaf", model="m",
            session_key=PARENT, runner=runner, max_async_children=1,
        )
        assert handle["status"] == "dispatched", handle
        assert entered.wait(timeout=10)
        A.unregister_gateway_notify(PARENT)
        assert A._gateway_notify_cb(PARENT) is not None
        release.set()
        wait_until(lambda: A._gateway_notify_cb(PARENT) is None, what="async child released the pin")
        assert seen_cb["during"] is not None
    finally:
        release.set()
        ad._reset_for_tests()
