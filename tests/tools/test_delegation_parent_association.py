"""Parent-session association helpers for delegated agent invocations."""

import sys
from queue import Queue
from types import ModuleType, SimpleNamespace

import tools.async_delegation as ad


class FakeSessionDB:
    def __init__(self, rows):
        self.rows = rows

    def get_session(self, session_id):
        return self.rows.get(session_id)


def test_resolve_parent_session_id_accepts_existing_session():
    db = FakeSessionDB({"parent-1": {"id": "parent-1", "ended_at": None}})

    assert ad.resolve_parent_session_id(" parent-1 ", session_db=db) == "parent-1"


def test_resolve_parent_session_id_rejects_missing_or_invalid_session():
    db = FakeSessionDB({"parent-1": {"id": "parent-1"}})

    assert ad.resolve_parent_session_id("", session_db=db) is None
    assert ad.resolve_parent_session_id(None, session_db=db) is None
    assert ad.resolve_parent_session_id("missing", session_db=db) is None


def test_build_parent_invocation_context_isolates_concurrent_parents():
    db = FakeSessionDB({"parent-a": {"id": "parent-a"}, "parent-b": {"id": "parent-b"}})
    parent_a = SimpleNamespace(session_id="parent-a", _session_db=db)
    parent_b = SimpleNamespace(session_id="parent-b", _session_db=db)

    ctx_a = ad.build_parent_invocation_context(
        parent_a,
        session_key="chat-a",
        origin_ui_session_id="tab-a",
    )
    ctx_b = ad.build_parent_invocation_context(
        parent_b,
        session_key="chat-b",
        origin_ui_session_id="tab-b",
    )

    assert ctx_a == {
        "session_key": "chat-a",
        "origin_ui_session_id": "tab-a",
        "parent_session_id": "parent-a",
    }
    assert ctx_b == {
        "session_key": "chat-b",
        "origin_ui_session_id": "tab-b",
        "parent_session_id": "parent-b",
    }


def test_build_parent_invocation_context_clears_invalid_parent_without_losing_route():
    db = FakeSessionDB({})
    parent = SimpleNamespace(session_id="stale-parent", _session_db=db)

    ctx = ad.build_parent_invocation_context(
        parent,
        session_key="chat-key",
        origin_ui_session_id="tab-1",
    )

    assert ctx == {
        "session_key": "chat-key",
        "origin_ui_session_id": "tab-1",
        "parent_session_id": None,
    }


def _install_fake_process_registry(monkeypatch, q):
    module = ModuleType("tools.process_registry")
    module.process_registry = SimpleNamespace(completion_queue=q)
    monkeypatch.setitem(sys.modules, "tools.process_registry", module)


def test_completion_event_reports_success_to_parent_session(monkeypatch):
    q = Queue()
    persisted = []
    monkeypatch.setattr(ad, "_persist_completion", lambda evt, result: persisted.append((evt, result)))
    _install_fake_process_registry(monkeypatch, q)

    record = {
        "delegation_id": "deleg-ok",
        "session_key": "chat-a",
        "origin_ui_session_id": "tab-a",
        "parent_session_id": "parent-a",
        "goal": "finish task",
        "role": "leaf",
        "model": "m",
        "dispatched_at": 10.0,
        "completed_at": 12.5,
    }
    result = {"status": "completed", "summary": "done", "api_calls": 2}

    ad._push_completion_event(record, result, "completed")

    evt = q.get_nowait()
    assert evt["status"] == "completed"
    assert evt["summary"] == "done"
    assert evt["session_key"] == "chat-a"
    assert evt["origin_ui_session_id"] == "tab-a"
    assert evt["parent_session_id"] == "parent-a"
    assert persisted == [(evt, result)]


def test_completion_event_reports_failure_to_parent_session(monkeypatch):
    q = Queue()
    monkeypatch.setattr(ad, "_persist_completion", lambda evt, result: None)
    _install_fake_process_registry(monkeypatch, q)

    record = {
        "delegation_id": "deleg-fail",
        "session_key": "chat-b",
        "origin_ui_session_id": "tab-b",
        "parent_session_id": "parent-b",
        "goal": "fail task",
        "dispatched_at": 20.0,
        "completed_at": 21.0,
    }
    result = {"status": "error", "summary": None, "error": "boom"}

    ad._push_completion_event(record, result, "error")

    evt = q.get_nowait()
    assert evt["status"] == "error"
    assert evt["error"] == "boom"
    assert evt["session_key"] == "chat-b"
    assert evt["origin_ui_session_id"] == "tab-b"
    assert evt["parent_session_id"] == "parent-b"
