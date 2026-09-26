"""Confirmed edit/resend on a compute-host (turn isolation) session.

The compute host writes each turn's rows, so the serving process's ``history`` stays
empty: session.history hands the client a valid persisted row id, then prompt.submit
checked that empty copy and refused the edit with 4018. Both tests use real SessionDB
rows, including a compression ancestor/child boundary.
"""

import threading

from hermes_state import SessionDB
from tui_gateway import server


def _compute_host_session(monkeypatch, db, key, sid, *, prefix=None):
    session = {
        "agent": None,
        "session_key": key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "_compute_host_active": True,
        "display_history_prefix": prefix or [],
    }
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *a, **k: None)
    # Run the (already truncated) turn in-process: dispatch to a real child is out of scope.
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)
    return session


def _store(db, key, rows):
    with db._lock:
        db._insert_message_rows(db._conn, key, rows)
        db._conn.commit()


def _submit(sid, target, ordinal, text):
    return server.handle_request({
        "id": text,
        "method": "prompt.submit",
        "params": {
            "session_id": sid,
            "text": text,
            "truncate_before_row_id": target,
            "truncate_before_user_ordinal": ordinal,
            "confirm_truncate": True,
        },
    })


def test_confirmed_edit_reloads_idle_compute_host_history(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "iso-truncate.db")
    key, sid = "iso-truncate", "iso-truncate-sid"
    db.create_session(key, "tui")
    rows = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]
    _store(db, key, rows)
    target = rows[2]["_row_id"]
    session = _compute_host_session(monkeypatch, db, key, sid)

    try:
        response = _submit(sid, target, 1, "edited second")
        assert response.get("error") is None, response
        assert [row["content"] for row in db.get_messages_as_conversation(key)] == [
            "first", "reply 1", "edited second"]
        assert [row["content"] for row in session["history"]] == ["first", "reply 1"]
    finally:
        server._release_active_session_slot(session)
        server._sessions.pop(sid, None)


def test_compute_host_edit_refuses_ancestor_without_rewriting_child(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "iso-lineage-truncate.db")
    parent, child, sid = "iso-parent", "iso-child", "iso-lineage-sid"
    db.create_session(parent, "tui")
    db.create_session(child, "tui", parent_session_id=parent)
    parent_rows = [
        {"role": "user", "content": "ancestor"},
        {"role": "assistant", "content": "ancestor reply"},
    ]
    child_rows = [
        {"role": "user", "content": "tip first"},
        {"role": "assistant", "content": "tip reply"},
        {"role": "user", "content": "tip second"},
        {"role": "assistant", "content": "tip second reply"},
    ]
    _store(db, parent, parent_rows)
    _store(db, child, child_rows)
    prefix = db.get_messages_as_conversation(parent, include_row_ids=True)
    session = _compute_host_session(monkeypatch, db, child, sid, prefix=prefix)

    try:
        ancestor = _submit(sid, parent_rows[0]["_row_id"], 0, "wrong ancestor edit")
        assert ancestor["error"]["code"] == 4018
        assert [row["content"] for row in db.get_messages_as_conversation(child)] == [
            "tip first", "tip reply", "tip second", "tip second reply"]

        edited = _submit(sid, child_rows[2]["_row_id"], 2, "edited tip second")
        assert edited.get("error") is None, edited
        assert [row["content"] for row in db.get_messages_as_conversation(parent)] == [
            "ancestor", "ancestor reply"]
        assert [row["content"] for row in db.get_messages_as_conversation(child)] == [
            "tip first", "tip reply", "edited tip second"]
    finally:
        server._release_active_session_slot(session)
        server._sessions.pop(sid, None)
