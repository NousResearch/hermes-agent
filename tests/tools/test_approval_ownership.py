"""Approval cancellation follows executions, not the shared conversation key.

The detached-route precedent is #108700; these tests exercise the current
cancelled/request/settle protocol rather than its older denial-based teardown.
"""

import queue
import threading
from types import SimpleNamespace

import pytest

from tools import approval, approval_gateway_wait
from tools.approval_gateway_wait import _await_gateway_decision
from tools.approval_ownership import (
    gateway_approval_owner,
    register_gateway_approval_owner,
    retain_gateway_approval_owner,
)
from tools.delegate_tool_child_run import _signal_child_stop


@pytest.mark.parametrize("boundary", ["execution", "child_stop", "session_clear", "legacy_unregister"])
@pytest.mark.parametrize("phase", ["pre_enqueue", "pending", "in_notify", "deadline", "answered"])
def test_closure_revokes_only_its_scope_and_cannot_admit_late_prompts(boundary, phase, monkeypatch):
    key = "approval-owner-scope"
    sent = queue.Queue()
    unblock_notify = threading.Event()
    results, settled, threads = {}, {}, []
    poll_expired, resume_poll = threading.Event(), threading.Event()
    original_poll = approval_gateway_wait._poll_event

    def deadline_checked(event, session_key, *, interrupt_log):
        with approval._lock:
            entry = next((e for e in approval._gateway_queues.get(key, []) if e.event is event), None)
        if entry is not None and entry.data["description"] == "ending":
            poll_expired.set()
            assert resume_poll.wait(10)
            return "timeout"
        return original_poll(event, session_key, interrupt_log=interrupt_log)

    if phase == "deadline":
        monkeypatch.setattr(approval_gateway_wait, "_poll_event", deadline_checked)

    def notify(data):
        # A real transport may consult the queue; neither admission nor closure
        # may hold the core lock across this callback or its blocked send.
        assert approval.has_blocking_approval(key)
        label = data["description"]
        settled[label] = []
        assert approval.register_gateway_settle(key, data["request_id"], settled[label].append)
        sent.put(data)
        if phase == "in_notify" and label == "ending":
            assert unblock_notify.wait(10)

    ending = register_gateway_approval_owner(key, notify)
    token = gateway_approval_owner.set(ending)
    try:
        sibling = retain_gateway_approval_owner(key)
    finally:
        gateway_approval_owner.reset(token)
    assert sibling is not None
    newer = register_gateway_approval_owner(key, notify)

    def wait(owner, label):
        token = gateway_approval_owner.set(owner)
        try:
            results[label] = _await_gateway_decision(key, approval._gateway_notify_cb(key), {
                # Identical commands in independent executions must not coalesce:
                # withdrawing one must not withdraw a still-live sibling's prompt.
                "command": "synthetic operation (never executed)",
                "description": label, "pattern_key": "test", "pattern_keys": ["test"],
            })
        finally:
            gateway_approval_owner.reset(token)

    def start(owner, label):
        thread = threading.Thread(target=wait, args=(owner, label))
        threads.append(thread)
        thread.start()
        return thread

    def close():
        if boundary == "execution":
            ending.close()
        elif boundary == "child_stop":
            _signal_child_stop(SimpleNamespace(_gateway_approval_owner=ending, _interrupt_requested=False))
        elif boundary == "session_clear":
            approval.clear_session(key)
        else:
            approval.unregister_gateway_notify(key)

    full_session = boundary in {"session_clear", "legacy_unregister"}
    try:
        sibling_thread = start(sibling, "sibling")
        sibling_request = sent.get(timeout=10)
        newer_thread = start(newer, "newer")
        newer_request = sent.get(timeout=10)
        if phase == "pre_enqueue":
            close()
            ending_thread = start(ending, "ending")
        else:
            ending_thread = start(ending, "ending")
            ending_request = sent.get(timeout=10)
            if phase == "deadline":
                assert poll_expired.wait(10)
            if phase == "answered":
                assert approval.resolve_gateway_approval(key, "once", request_id=ending_request["request_id"]) == 1
            close()
            assert approval.resolve_gateway_approval(key, "once", request_id=ending_request["request_id"]) == 0
        unblock_notify.set()
        resume_poll.set()
        ending_thread.join(10)
        assert not ending_thread.is_alive()
        if phase == "answered":
            assert results["ending"]["choice"] == "once"
            assert not results["ending"].get("cancelled")
            assert settled["ending"] == ["resolved"]
        else:
            assert results["ending"]["choice"] is None
            assert results["ending"]["cancelled"]
            if phase != "pre_enqueue":
                assert settled["ending"] == ["session_closed"]

        # An old cleanup is idempotent, never a teardown of a new registration.
        ending.close()
        if full_session:
            for thread in (sibling_thread, newer_thread):
                thread.join(10)
                assert not thread.is_alive()
            assert results["sibling"]["cancelled"] and results["newer"]["cancelled"]
            assert not approval.has_blocking_approval(key)
            newer = register_gateway_approval_owner(key, notify)
        else:
            assert approval._gateway_notify_cb(key) is newer
            for request in (sibling_request, newer_request):
                assert approval.resolve_gateway_approval(key, "once", request_id=request["request_id"]) == 1
            for thread in (sibling_thread, newer_thread):
                thread.join(10)
                assert not thread.is_alive()
            assert results["sibling"]["choice"] == results["newer"]["choice"] == "once"

        # A captured old context cannot borrow the new callback, even after reset.
        late = start(ending, "late")
        late.join(10)
        assert not late.is_alive()
        assert results["late"]["cancelled"] and results["late"]["choice"] is None
        assert sent.empty()
        assert approval._gateway_notify_cb(key) is newer
    finally:
        unblock_notify.set()
        resume_poll.set()
        approval.unregister_gateway_notify(key)
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive()
