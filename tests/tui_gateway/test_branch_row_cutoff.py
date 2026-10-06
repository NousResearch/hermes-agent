"""Branch boundaries address persisted rows, not hydrated Desktop bubble counts."""

from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def branch_store(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from tui_gateway import server

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_idempotency_keys", {})
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    with SessionDB(home / "state.db") as db:
        db.create_session("parent", source="desktop", model="test-model")
        # Desktop hides notices and folds consecutive assistant text into one bubble.
        rows: list[dict] = [
            {"role": "user", "content": "first question"},
            {"role": "user", "content": "model changed", "display_kind": "notice"},
            {"role": "assistant", "content": "answer begins"},
            {"role": "assistant", "content": "answer ends"},
            {"role": "user", "content": "later question"},
            {"role": "assistant", "content": "later answer"},
        ]
        row_ids = [db.append_message("parent", **row) for row in rows]
        db.create_session("unrelated", source="desktop", model="test-model")
        foreign_id = db.append_message("unrelated", "assistant", "foreign answer")
        parent = {
            "session_key": "parent", "history": [dict(rows[-1])],
            "history_lock": threading.Lock(), "running": False, "source": "desktop",
            "profile_home": str(home), "cwd": str(tmp_path), "created_at": 1.0,
            "agent": SimpleNamespace(model="test-model"),
        }
        server._sessions["parent"] = parent
        monkeypatch.setattr(server, "_get_db", lambda: db)

        # Keep model/worker startup out of the test; the RPC, source projection,
        # profile-owned database lookup and branch persistence remain real.
        def build_child(session, sid, key, history, source):
            child = dict(session, session_key=key, history=history, parent_session_id="parent")
            server._sessions[sid] = child
            return child["agent"]

        monkeypatch.setattr(server, "_build_branch_agent", build_child)
        yield server, db, parent, rows, row_ids, foreign_id


@pytest.mark.parametrize("entrypoint", ["rpc", "direct"])
@pytest.mark.parametrize("boundary", ["exact", "exact_over_count", "legacy", "whole", "branch_whole"])
def test_branch_copies_inclusive_row_prefix_and_retries_once(branch_store, entrypoint, boundary):
    server, db, parent, rows, row_ids, _ = branch_store
    params: dict = {"session_id": "parent", "name": "fork", "idempotency_key": "fork-retry"}
    expected = rows
    if boundary.startswith("exact"):
        params["through_row_id"] = row_ids[3]
        if boundary == "exact_over_count":
            params["count"] = 2  # Two hydrated bubbles contain four persisted rows.
        expected = rows[:4]
    elif boundary == "legacy":
        params["count"] = 2
        expected = rows[:2]
    method = "session.branch_whole" if boundary == "branch_whole" else "session.branch"

    def branch():
        if entrypoint == "rpc":
            return server.handle_request({"id": "fork", "method": method, "params": dict(params)})
        return server._branch_live("fork", dict(params), parent, omit_messages=method == "session.branch_whole")

    response = branch()
    assert "result" in response, response
    result = response["result"]
    child = result["stored_session_id"]
    persisted = db.get_messages(child)
    assert [row["content"] for row in persisted] == [row["content"] for row in expected]
    assert persisted[1]["display_kind"] == "notice"
    assert db.get_session(child)["parent_session_id"] == "parent"
    assert result["message_count"] == len(expected)
    if boundary != "branch_whole":
        assert [row["text"] for row in result["messages"]] == [row["content"] for row in expected]
    else:
        assert result["messages_omitted"] is True
        assert "messages" not in result
    retry = branch()
    assert retry["result"]["stored_session_id"] == child
    assert retry["result"]["session_id"] == result["session_id"]
    assert len(db.get_messages(child)) == len(expected)
    assert {row[0] for row in db._conn.execute("SELECT id FROM sessions")} == {"parent", "unrelated", child}
    assert [row["id"] for row in db.get_messages("parent")] == row_ids


@pytest.mark.parametrize("entrypoint", ["rpc", "direct"])
@pytest.mark.parametrize("target", ["missing", "foreign", 0, -1, True, "1", 1.5])
def test_invalid_branch_row_never_creates_a_child(branch_store, entrypoint, target):
    server, db, parent, _, row_ids, foreign_id = branch_store
    target = {"missing": foreign_id + 100, "foreign": foreign_id}.get(target, target)
    params = {"session_id": "parent", "through_row_id": target, "count": 1,
              "name": "must not exist", "idempotency_key": "invalid-fork"}
    if entrypoint == "rpc":
        response = server.handle_request({"id": "fork", "method": "session.branch", "params": params})
    else:
        response = server._branch_live("fork", params, parent)
    assert "error" in response, response
    assert "through_row_id" in response["error"]["message"]
    assert {row[0] for row in db._conn.execute("SELECT id FROM sessions")} == {"parent", "unrelated"}
    assert set(server._sessions) == {"parent"}
    assert not server._idempotency_keys
    assert [row["id"] for row in db.get_messages("parent")] == row_ids
