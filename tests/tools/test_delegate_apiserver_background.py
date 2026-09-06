"""delegate_task(background=true) on stateless API-server sessions.

An OpenAI-compatible request cannot consume a detached completion after its
response ends. A bound raw session id only makes the result durable for
polling clients; it does not make delivery reachable from a plain client.
Both declared and derived API sessions therefore use the synchronous fallback.
"""

import json
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


def test_apiserver_session_with_id_returns_completion_synchronously(monkeypatch):
    """A raw session id does not make detached delivery client-reachable."""
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
    assert parsed["results"][0]["summary"] == "done: bg on api_server"
    assert "SYNCHRONOUSLY" in parsed["note"]
    assert process_registry.completion_queue.empty()


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
