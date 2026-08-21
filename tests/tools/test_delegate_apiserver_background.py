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
import threading
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
    """No session id to wake → keep the sync fallback and parent ownership."""
    dt = _patch_delegate(monkeypatch)
    parent = _fake_parent()
    child = MagicMock()
    child._delegate_role = "leaf"
    child._subagent_id = "sync-child"

    def build_parent_owned_child(**kwargs):
        kwargs["parent_agent"]._active_children.append(child)
        return child

    def assert_parent_owned_while_running(*_args, **_kwargs):
        assert child in parent._active_children
        return {
            "task_index": 0,
            "status": "completed",
            "summary": "done synchronously",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "completed",
        }

    monkeypatch.setattr(dt, "_build_child_agent", build_parent_owned_child)
    monkeypatch.setattr(dt, "_run_single_child", assert_parent_owned_while_running)
    set_session_vars(
        platform="api_server",
        chat_id="",
        session_key="",
        session_id="",
        async_delivery=False,
    )

    out = dt.delegate_task(
        goal="one-shot", context="ctx",
        background=True, parent_agent=parent,
    )
    parsed = json.loads(out)
    assert parsed.get("status") != "dispatched", parsed
    assert "SYNCHRONOUSLY" in parsed.get("note", "")
    assert process_registry.completion_queue.empty()


def test_capacity_queue_never_falls_back_to_synchronous_execution(monkeypatch):
    dt = _patch_delegate(monkeypatch)
    set_session_vars(
        platform="api_server",
        chat_id="raw-sid-queued",
        session_key="raw-sid-queued",
        session_id="raw-sid-queued",
        async_delivery=False,
    )

    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        lambda **_kwargs: {
            "status": "queued",
            "delegation_id": "deleg_queued1",
            "queue_reason": "capacity",
        },
    )
    monkeypatch.setattr(
        dt,
        "_run_single_child",
        lambda *_args, **_kwargs: pytest.fail("queued work ran synchronously"),
    )

    parsed = json.loads(dt.delegate_task(
        goal="queue me", context="ctx", background=True,
        parent_agent=_fake_parent(),
    ))

    assert parsed["status"] == "queued"
    assert parsed["mode"] == "background"
    assert parsed["delegation_id"] == "deleg_queued1"
    assert parsed["queue_reason"] == "capacity"
    assert "queued" in parsed["note"].lower()
    assert "subagent_ids" not in parsed
    assert "action='list'" in parsed["control_hint"]


def test_capacity_queue_defers_child_construction_until_admitted(monkeypatch):
    dt = _patch_delegate(monkeypatch)
    set_session_vars(
        platform="api_server",
        chat_id="raw-sid-lazy",
        session_key="raw-sid-lazy",
        session_id="raw-sid-lazy",
        async_delivery=False,
    )
    build_calls = []

    def build_child(**kwargs):
        build_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(dt, "_build_child_agent", build_child)
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        lambda **_kwargs: {
            "status": "queued",
            "delegation_id": "deleg_lazy1",
            "queue_reason": "capacity",
        },
    )

    parsed = json.loads(dt.delegate_task(
        goal="build me later", context="ctx", background=True,
        parent_agent=_fake_parent(),
    ))

    assert parsed["status"] == "queued"
    assert build_calls == []


def test_resource_queue_builds_child_only_after_promotion(monkeypatch):
    from tools import async_delegation as ad

    ad._reset_for_tests()
    dt = _patch_delegate(monkeypatch)
    set_session_vars(
        platform="api_server",
        chat_id="raw-sid-promote",
        session_key="raw-sid-promote",
        session_id="raw-sid-promote",
        async_delivery=False,
    )
    available = {"bytes": 0}
    build_calls = []
    built_children = []
    original_build = dt._build_child_agent

    def counted_build(**kwargs):
        build_calls.append(kwargs)
        child = original_build(**kwargs)
        built_children.append(child)
        return child

    monkeypatch.setattr(dt, "_build_child_agent", counted_build)
    monkeypatch.setattr(ad, "_ADMISSION_RECHECK_SECONDS", 0.02)
    monkeypatch.setattr(
        ad, "_effective_available_memory_bytes", lambda: available["bytes"]
    )
    monkeypatch.setattr(dt, "_get_max_async_children", lambda: 1)
    monkeypatch.setattr(dt, "_get_min_available_memory_bytes", lambda: 100)
    monkeypatch.setattr(dt, "_get_resume_available_memory_bytes", lambda: 200)

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    parsed = json.loads(dt.delegate_task(
        goal="promote later", context="ctx", background=True,
        output_schema=schema,
        parent_agent=_fake_parent(),
    ))
    assert parsed["status"] == "queued"
    assert build_calls == []

    available["bytes"] = 150
    time.sleep(0.1)
    assert build_calls == []
    available["bytes"] = 250
    event = _drain_one()
    assert event is not None
    assert event["status"] == "completed"
    assert len(build_calls) == 1
    assert "OUTPUT CONTRACT" in build_calls[0]["context"]
    assert built_children[0]._delegate_output_schema == schema
    ad._reset_for_tests()


def test_cancel_during_lazy_child_build_is_nonblocking_and_cleans_up(monkeypatch):
    from tools import async_delegation as ad

    ad._reset_for_tests()
    dt = _patch_delegate(monkeypatch)
    set_session_vars(
        platform="api_server",
        chat_id="raw-sid-cancel-build",
        session_key="raw-sid-cancel-build",
        session_id="raw-sid-cancel-build",
        async_delivery=False,
    )
    build_started = threading.Event()
    release_build = threading.Event()
    child = MagicMock()
    child.run.return_value = "must not run"

    def blocking_build(**_kwargs):
        build_started.set()
        assert release_build.wait(timeout=2)
        return child

    monkeypatch.setattr(dt, "_build_child_agent", blocking_build)
    monkeypatch.setattr(dt, "_get_max_async_children", lambda: 1)
    parent = _fake_parent()
    parsed = json.loads(dt.delegate_task(
        goal="cancel during build", context=None, background=True,
        parent_agent=parent,
    ))
    assert parsed["status"] == "dispatched"
    assert build_started.wait(timeout=1)

    started = time.monotonic()
    assert ad.interrupt_for_session(
        "raw-sid-cancel-build", parent_session_id=parent.session_id,
        reason="test cancellation",
    ) == 1
    assert time.monotonic() - started < 0.5
    release_build.set()

    event = _drain_one()
    assert event is not None
    assert event["status"] == "interrupted"
    child.run.assert_not_called()
    child.close.assert_called_once()
    ad._reset_for_tests()


def test_queue_configuration_is_forwarded_to_async_admission(monkeypatch):
    dt = _patch_delegate(monkeypatch)
    set_session_vars(
        platform="api_server",
        chat_id="raw-sid-config",
        session_key="raw-sid-config",
        session_id="raw-sid-config",
        async_delivery=False,
    )
    captured = {}

    def fake_dispatch(**kwargs):
        captured.update(kwargs)
        return {"status": "dispatched", "delegation_id": "deleg_config1"}

    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch", fake_dispatch
    )
    monkeypatch.setattr(dt, "_get_max_queued_delegations", lambda: 5)
    monkeypatch.setattr(dt, "_get_min_available_memory_bytes", lambda: 4096)
    monkeypatch.setattr(dt, "_get_resume_available_memory_bytes", lambda: 8192)
    monkeypatch.setattr(dt, "_get_max_memory_psi_avg10", lambda: 12.0)
    monkeypatch.setattr(dt, "_get_resume_memory_psi_avg10", lambda: 5.0)
    monkeypatch.setattr(dt, "_get_queue_timeout_seconds", lambda: 123.0)

    parsed = json.loads(dt.delegate_task(
        goal="configured", context="ctx", background=True,
        parent_agent=_fake_parent(),
    ))

    assert parsed["status"] == "dispatched"
    assert captured["max_queued_delegations"] == 5
    assert captured["min_available_memory_bytes"] == 4096
    assert captured["resume_available_memory_bytes"] == 8192
    assert captured["max_memory_psi_avg10"] == 12.0
    assert captured["resume_memory_psi_avg10"] == 5.0
    assert captured["queue_timeout_seconds"] == 123.0
