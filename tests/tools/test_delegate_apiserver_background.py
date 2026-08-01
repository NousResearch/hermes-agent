"""delegate_task(background=true) on stateless API-server sessions.

Previously async_delivery_supported()=False forced SYNCHRONOUS execution for
every background dispatch on the API server, blocking the whole turn. Now
that background completions can wake the originating session via the
/v1/chat/completions self-post (gateway/wake.py), a session-continuable
turn (raw session id bound as the api_server chat_id) dispatches async; only
session-id-less one-shot requests keep the sync fallback.

The wake target must be captured from the request-scoped chat_id binding,
NOT from HERMES_SESSION_ID: constructing a child agent calls
set_current_session_id(child.session_id), clobbering the HERMES_SESSION_ID
ContextVar and os.environ with the subagent's internal id before the
dispatch code reads it — the fake child build below reproduces that clobber.
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from gateway.session_context import set_session_vars
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_queue_and_context(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    while not process_registry.completion_queue.empty():
        try:
            process_registry.completion_queue.get_nowait()
        except Exception:
            break
    yield
    # Restore ContextVars to the pristine "never set" sentinel rather than
    # clear_session_vars()'s explicit-"" state, which would mask env vars for
    # unrelated tests running later in the same worker.
    import gateway.session_context as sc

    for var in sc._VAR_MAP.values():
        var.set(sc._UNSET)
    sc._SESSION_ASYNC_DELIVERY.set(sc._UNSET)
    # set_current_session_id (invoked by the clobber-reproducing fake child
    # build) writes os.environ directly — scrub it so it can't leak into
    # other test modules.
    import os

    os.environ.pop("HERMES_SESSION_ID", None)
    while not process_registry.completion_queue.empty():
        try:
            process_registry.completion_queue.get_nowait()
        except Exception:
            break


def _drain_one(timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.02)
    return None


def _fake_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    return parent


def _patch_delegate(monkeypatch):
    import tools.delegate_tool as dt

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    def fast_child(task_index, goal, child=None, parent_agent=None, **kw):
        return {
            "task_index": 0, "status": "completed", "summary": f"done: {goal}",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    def clobbering_build_child(**kw):
        # Reproduce what the real _build_child_agent -> AIAgent -> agent_init
        # path does: it synchronizes the child's internal session id into the
        # HERMES_SESSION_ID ContextVar + os.environ, clobbering the spawner's
        # id ~milliseconds before delegate_tool dispatches the batch.
        from gateway.session_context import set_current_session_id

        set_current_session_id("20260715_child1")
        return fake_child

    monkeypatch.setattr(dt, "_build_child_agent", clobbering_build_child)
    monkeypatch.setattr(dt, "_run_single_child", fast_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    return dt


def test_inject_batch_emits_fast_child_before_slow_child(monkeypatch):
    """Exercise delegate_tool's real per-child callback seam, not just ledger APIs."""
    dt = _patch_delegate(monkeypatch)
    slow_gate = __import__("threading").Event()

    def staggered_child(task_index, goal, child=None, parent_agent=None, **kw):
        if task_index == 1:
            slow_gate.wait(timeout=5)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": f"done: {goal}",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "completed",
        }

    monkeypatch.setattr(dt, "_run_single_child", staggered_child)
    set_session_vars(
        platform="telegram",
        chat_id="7",
        session_key="agent:main:telegram:dm:7",
        session_id="parent-sess",
        async_delivery=True,
    )
    parent = _fake_parent()
    parent._active_turn_id = "turn-live"
    parsed = json.loads(
        dt.delegate_task(
            tasks=[{"goal": "fast child"}, {"goal": "slow child"}],
            background=True,
            result_delivery="inject",
            parent_agent=parent,
        )
    )
    assert parsed["status"] == "dispatched", parsed

    first = _drain_one(timeout=2)
    assert first is not None
    assert first["delivery_event_key"] == "task:0"
    assert first["parent_turn_id"] == "turn-live"
    assert first["results"][0]["summary"] == "done: fast child"
    import tools.async_delegation as ad

    claim = ad.claim_event_delivery(first, "test-consumer")
    assert claim
    ad.complete_event_delivery(first, claim)

    slow_gate.set()
    second = _drain_one(timeout=2)
    assert second is not None
    assert second["delivery_event_key"] == "task:1"
    assert second["results"][0]["summary"] == "done: slow child"
    claim = ad.claim_event_delivery(second, "test-consumer")
    assert claim
    ad.complete_event_delivery(second, claim)
    deadline = time.time() + 2
    while ad.active_count() and time.time() < deadline:
        time.sleep(0.01)
    assert ad.active_count() == 0


def test_after_turn_batch_delivers_ready_children_without_waiting_and_coalesces(monkeypatch):
    """Ready siblings share one turn; an unfinished sibling cannot block them."""
    dt = _patch_delegate(monkeypatch)
    threading = __import__("threading")
    slow_gate = threading.Event()
    first_pair_ready = threading.Event()
    ready_lock = threading.Lock()
    ready_indices = set()

    def staggered_child(task_index, goal, child=None, parent_agent=None, **kw):
        if task_index == 2:
            slow_gate.wait(timeout=5)
        with ready_lock:
            ready_indices.add(task_index)
            if {0, 1}.issubset(ready_indices):
                first_pair_ready.set()
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": f"done: {goal}",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "completed",
        }

    monkeypatch.setattr(dt, "_run_single_child", staggered_child)
    set_session_vars(
        platform="telegram",
        chat_id="7",
        session_key="agent:main:telegram:dm:7",
        session_id="parent-sess",
        async_delivery=True,
    )
    parsed = json.loads(
        dt.delegate_task(
            tasks=[{"goal": "task number zero"}, {"goal": "task number one"}, {"goal": "task number two"}],
            background=True,
            result_delivery="after_turn",
            parent_agent=_fake_parent(),
        )
    )
    assert parsed["status"] == "dispatched", parsed
    assert first_pair_ready.wait(timeout=2)
    publish_deadline = time.time() + 2
    while process_registry.completion_queue.qsize() < 2 and time.time() < publish_deadline:
        time.sleep(0.01)
    assert process_registry.completion_queue.qsize() >= 2

    drained = process_registry.drain_notifications(
        owns_event=lambda _event: True,
        skip_poll_observed=False,
    )
    assert len(drained) == 1
    first_event, first_text = drained[0]
    assert first_event["delivery_event_keys"] == ["task:0", "task:1"]
    assert [result["task_index"] for result in first_event["results"]] == [0, 1]
    assert "done: task number zero" in first_text
    assert "done: task number one" in first_text
    assert "done: task number two" not in first_text

    import tools.async_delegation as ad

    claim = ad.claim_event_delivery(first_event, "test-after-turn")
    assert claim
    assert ad.complete_event_delivery(first_event, claim)

    slow_gate.set()
    deadline = time.time() + 2
    late = []
    while time.time() < deadline and not late:
        late = process_registry.drain_notifications(
            owns_event=lambda _event: True,
            skip_poll_observed=False,
        )
        if not late:
            time.sleep(0.02)
    assert len(late) == 1
    late_event, late_text = late[0]
    assert late_event["delivery_event_keys"] == ["task:2"]
    assert [result["task_index"] for result in late_event["results"]] == [2]
    assert "done: task number two" in late_text
    claim = ad.claim_event_delivery(late_event, "test-after-turn")
    assert claim
    assert ad.complete_event_delivery(late_event, claim)
    finalize_deadline = time.time() + 2
    while ad.active_count() and time.time() < finalize_deadline:
        time.sleep(0.01)
    assert ad.active_count() == 0


def test_apiserver_session_with_id_dispatches_background(monkeypatch):
    """async_delivery=False + a raw session id (HERMES_SESSION_ID) →
    background dispatch (the completion wakes the session via the
    api_server self-post), NOT the forced-sync fallback."""
    dt = _patch_delegate(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_ID", "raw-sid-7")
    set_session_vars(
        platform="api_server",
        chat_id="raw-sid-7",
        session_key="raw-sid-7",
        session_id="raw-sid-7",
        async_delivery=False,
    )

    out = dt.delegate_task(
        goal="bg on api_server", context="ctx",
        background=True, parent_agent=_fake_parent(),
    )
    parsed = json.loads(out)
    assert parsed["status"] == "dispatched", parsed
    assert parsed["mode"] == "background"

    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    # The raw session id is stamped so the gateway drain can self-post the
    # wake to the REAL session (session_key alone is the raw id here, which
    # carries no parseable routing metadata). Crucially this is the SPAWNER's
    # id, not the subagent-internal id the child build clobbered
    # HERMES_SESSION_ID with (see clobbering_build_child).
    assert evt["origin_session_id"] == "raw-sid-7"


# ---------------------------------------------------------------------------
# _current_origin_session_id — the clobber-proof origin capture helper
# ---------------------------------------------------------------------------


def test_apiserver_session_without_id_stays_synchronous(monkeypatch):
    """No session id to wake → keep the sync fallback (a detached result
    would never re-enter any conversation)."""
    dt = _patch_delegate(monkeypatch)
    set_session_vars(
        platform="api_server",
        chat_id="",
        session_key="",
        session_id="",
        async_delivery=False,
    )

    out = dt.delegate_task(
        goal="one-shot", context="ctx",
        background=True, parent_agent=_fake_parent(),
    )
    parsed = json.loads(out)
    assert parsed.get("status") != "dispatched", parsed
    assert "SYNCHRONOUSLY" in parsed.get("note", "")
    assert process_registry.completion_queue.empty()
