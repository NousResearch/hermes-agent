"""RED-first behavioral tests for the Desktop human todo Mark done/Reopen RPC.

Backend contract under test:
  todo.snapshot {session_id}        -> {session_id, revision, generation, todos}
  todo.update_status {session_id, item_id, status, actor, expected_revision}
                                    -> same full snapshot
  todo.updated event                -> full snapshot payload

Machine-distinguishable errors:
  4001 session missing (existing _sess semantics)
  4004 invalid params
  4096 stale revision with current snapshot in error.data
  4044 item missing with current snapshot in error.data

Error codes are asserted as named constants (imported from the handler
module) so the test reads as a contract, not as magic literals — but the
VALUES are pinned too (4096/4044) because renderer code branches on them.
"""

import json

import pytest
from unittest.mock import MagicMock, patch

_original_stdout = None


@pytest.fixture()
def server():
    """Import the gateway server with sys.modules mocks (test_protocol.py pattern)."""
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test_todo_rpc")),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    real_stdout = mod._real_stdout
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._real_stdout = real_stdout
    for sid in list(mod._sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")
    mod._pending.clear()
    mod._answers.clear()


@pytest.fixture()
def todo_server(server):
    """A gateway server with one live session backed by a real TodoStore."""
    from tools.todo_tool import TodoStore

    store = TodoStore()
    store.write([
        {"id": "t1", "content": "Research competitors", "status": "in_progress"},
        {"id": "t2", "content": "Build the tray", "status": "pending"},
    ])
    server._sessions["todo-sid"] = {
        "agent": type("A", (), {"_todo_store": store})(),
        "agent_error": None,
        "agent_ready": None,
    }
    return server, store


def call(server, method, params, rid="r1"):
    handler = server._methods[method]
    return handler(rid, params)


class TestTodoSnapshot:
    def test_snapshot_returns_full_shape(self, todo_server):
        server, store = todo_server

        resp = call(server, "todo.snapshot", {"session_id": "todo-sid"})

        assert "error" not in resp
        assert resp["result"]["session_id"] == "todo-sid"
        assert resp["result"]["revision"] == store.revision
        assert resp["result"]["generation"] == store.snapshot_state()["generation"]
        assert resp["result"]["todos"] == [
            {"id": "t1", "content": "Research competitors", "status": "in_progress"},
            {"id": "t2", "content": "Build the tray", "status": "pending"},
        ]

    def test_snapshot_missing_session_is_4001(self, server):
        resp = call(server, "todo.snapshot", {"session_id": "gone"})

        assert resp["error"]["code"] == 4001

    def test_snapshot_invalid_params_is_4004(self, todo_server):
        server, _store = todo_server

        resp = call(server, "todo.snapshot", {})

        assert resp["error"]["code"] == 4004


class TestTodoUpdateStatus:
    def test_mark_done_returns_full_snapshot(self, todo_server):
        server, store = todo_server
        rev = store.revision

        resp = call(server, "todo.update_status", {
            "session_id": "todo-sid",
            "item_id": "t1",
            "status": "completed",
            "actor": "user",
            "expected_revision": rev,
        })

        assert "error" not in resp
        assert resp["result"]["revision"] == rev + 1
        assert resp["result"]["generation"] == store.snapshot_state()["generation"]
        statuses = {t["id"]: t["status"] for t in resp["result"]["todos"]}
        assert statuses == {"t1": "completed", "t2": "pending"}
        # User completion is authoritative: a stale model merge can't undo it.
        store.write([{"id": "t1", "status": "in_progress"}], merge=True)
        assert {t["id"]: t["status"] for t in store.read()}["t1"] == "completed"

    def test_reopen_returns_pending(self, todo_server):
        server, store = todo_server
        store.update_status("t2", "completed", actor="user")
        rev = store.revision

        resp = call(server, "todo.update_status", {
            "session_id": "todo-sid",
            "item_id": "t2",
            "status": "pending",
            "actor": "user",
            "expected_revision": rev,
        })

        assert "error" not in resp
        statuses = {t["id"]: t["status"] for t in resp["result"]["todos"]}
        assert statuses["t2"] == "pending"

    def test_stale_revision_is_4096_with_current_snapshot(self, todo_server):
        server, store = todo_server
        stale = store.revision
        store.write([{"id": "t3", "content": "New plan", "status": "in_progress"}])

        resp = call(server, "todo.update_status", {
            "session_id": "todo-sid",
            "item_id": "t3",
            "status": "completed",
            "actor": "user",
            "expected_revision": stale,
        })

        assert resp["error"]["code"] == 4096
        assert resp["error"]["data"]["session_id"] == "todo-sid"
        assert resp["error"]["data"]["revision"] == store.revision
        assert resp["error"]["data"]["generation"] == store.snapshot_state()["generation"]
        assert any(t["id"] == "t3" for t in resp["error"]["data"]["todos"])
        # No mutation happened.
        assert {t["id"]: t["status"] for t in store.read()}["t3"] == "in_progress"

    def test_missing_item_is_4044_with_current_snapshot(self, todo_server):
        server, store = todo_server

        resp = call(server, "todo.update_status", {
            "session_id": "todo-sid",
            "item_id": "nope",
            "status": "completed",
            "actor": "user",
            "expected_revision": store.revision,
        })

        assert resp["error"]["code"] == 4044
        assert resp["error"]["data"]["session_id"] == "todo-sid"
        assert resp["error"]["data"]["revision"] == store.revision
        assert isinstance(resp["error"]["data"]["todos"], list)

    def test_invalid_status_is_4004(self, todo_server):
        server, store = todo_server

        resp = call(server, "todo.update_status", {
            "session_id": "todo-sid",
            "item_id": "t1",
            "status": "cancelled",
            "actor": "user",
        })

        assert resp["error"]["code"] == 4004

    def test_invalid_actor_is_4004(self, todo_server):
        server, store = todo_server

        resp = call(server, "todo.update_status", {
            "session_id": "todo-sid",
            "item_id": "t1",
            "status": "completed",
            "actor": "model",
        })

        assert resp["error"]["code"] == 4004

    def test_missing_item_id_is_4004(self, todo_server):
        server, _store = todo_server

        resp = call(server, "todo.update_status", {
            "session_id": "todo-sid",
            "status": "completed",
            "actor": "user",
        })

        assert resp["error"]["code"] == 4004

    def test_missing_expected_revision_is_4004(self, todo_server):
        server, _store = todo_server

        resp = call(server, "todo.update_status", {
            "session_id": "todo-sid",
            "item_id": "t1",
            "status": "completed",
            "actor": "user",
        })

        assert resp["error"]["code"] == 4004

    def test_noop_replay_returns_snapshot_without_revision_bump(self, todo_server):
        server, store = todo_server
        store.update_status("t1", "completed", actor="user")
        rev = store.revision
        gen = store.snapshot_state()["generation"]

        resp = call(server, "todo.update_status", {
            "session_id": "todo-sid",
            "item_id": "t1",
            "status": "completed",
            "actor": "user",
            "expected_revision": rev,
        })

        assert "error" not in resp
        assert resp["result"]["revision"] == rev
        assert resp["result"]["generation"] == gen

    def test_missing_session_is_4001(self, server):
        resp = call(server, "todo.update_status", {
            "session_id": "gone",
            "item_id": "t1",
            "status": "completed",
            "actor": "user",
            "expected_revision": 1,
        })

        assert resp["error"]["code"] == 4001


class TestTodoUpdatedEvent:
    def test_success_emits_dedicated_full_snapshot_event(self, todo_server):
        server, store = todo_server
        events = []
        original_emit = server._emit

        def spy(event, sid, payload=None):
            if event == "todo.updated":
                events.append((sid, payload))
            return original_emit(event, sid, payload)

        server._emit = spy
        try:
            rev = store.revision
            call(server, "todo.update_status", {
                "session_id": "todo-sid",
                "item_id": "t1",
                "status": "completed",
                "actor": "user",
                "expected_revision": rev,
            })
        finally:
            server._emit = original_emit

        assert len(events) == 1
        sid, payload = events[0]
        assert sid == "todo-sid"
        assert payload["session_id"] == "todo-sid"
        assert payload["revision"] == rev + 1
        assert "generation" in payload
        assert {"id": "t1", "content": "Research competitors", "status": "completed"} in payload["todos"]

    def test_stale_revision_does_not_emit_event(self, todo_server):
        server, store = todo_server
        events = []
        original_emit = server._emit

        def spy(event, sid, payload=None):
            if event == "todo.updated":
                events.append(payload)
            return original_emit(event, sid, payload)

        server._emit = spy
        try:
            stale = store.revision
            store.write([{"id": "t4", "content": "Later plan", "status": "in_progress"}])
            call(server, "todo.update_status", {
                "session_id": "todo-sid",
                "item_id": "t4",
                "status": "completed",
                "actor": "user",
                "expected_revision": stale,
            })
        finally:
            server._emit = original_emit

        assert events == []


class TestTodoToolResultAndCompleEvent:
    """The todo tool result and tool.complete payload must carry revision +
    generation so model-driven events become authoritative snapshots."""

    def test_todo_tool_result_includes_revision_and_generation(self):
        from tools.todo_tool import TodoStore, todo_tool

        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "in_progress"}])

        result = json.loads(todo_tool(store=store))

        assert result["revision"] == store.revision
        assert result["generation"] == store.snapshot_state()["generation"]
        assert isinstance(result["todos"], list)

    def test_tool_complete_todo_payload_carries_authoritative_fields(self, todo_server):
        server, store = todo_server
        events = []
        original_emit = server._emit

        def spy(event, sid, payload=None):
            if event == "tool.complete":
                events.append(payload)
            return original_emit(event, sid, payload)

        server._emit = spy
        try:
            server._on_tool_complete(
                "todo-sid",
                "call-1",
                "todo",
                {},
                json.dumps({"todos": store.read(), "revision": store.revision, "generation": 7}),
            )
        finally:
            server._emit = original_emit

        assert events, "tool.complete must still be emitted for todo"
        payload = events[-1]
        assert payload["todos"] == store.read()
        assert payload["revision"] == store.revision
        assert payload["generation"] == 7