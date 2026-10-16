"""session.create {temporary: true}: a hidden temporary row that never lists, is deleted on close, survives
backend reclaim (the client still shows it), and cannot be branched into a saved chat."""

import pytest

from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def db(monkeypatch, tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: database)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    try:
        yield database
    finally:
        database.close()


def _rpc(method: str, params: dict) -> dict:
    return server.handle_request({"id": method, "method": method, "params": params})


def _open_temporary_chat(db) -> tuple[str, str]:
    result = _rpc("session.create", {"cols": 96, "source": "desktop", "temporary": True})["result"]
    sid, key = result["session_id"], result["stored_session_id"]
    assert server._persist_session_row_for_submit("rid", server._sessions[sid]) is None  # first prompt.submit
    db.append_messages_batch(key, [{"role": "user", "content": "incognito question"}])
    return sid, key


def test_temporary_row_is_hidden_unlisted_and_deleted_on_close(db):
    sid, key = _open_temporary_chat(db)
    try:
        assert db.is_session_temporary(key) and db.get_session(key)["hidden"] == 1
        listed = _rpc("session.list", {"include_hidden": True})["result"]["sessions"]
        assert key not in {s["id"] for s in listed}

        assert "result" in _rpc("session.close", {"session_id": sid})
        assert db.get_session(key) is None
        assert db.search_messages("incognito") == []
    finally:
        server._sessions.pop(sid, None)


@pytest.mark.parametrize("reason", ["ws_disconnect", "idle_timeout"])
def test_backend_reclaim_keeps_the_row_for_resume(db, reason):
    sid, key = _open_temporary_chat(db)
    session = server._pop_session_by_id(sid)
    server._teardown_popped_session(session, end_reason=reason)
    assert db.is_session_temporary(key)


def test_normal_chat_close_keeps_its_row(db):
    result = _rpc("session.create", {"cols": 96, "source": "desktop"})["result"]
    sid, key = result["session_id"], result["stored_session_id"]
    try:
        assert server._persist_session_row_for_submit("rid", server._sessions[sid]) is None
        db.append_messages_batch(key, [{"role": "user", "content": "keep me"}])
        assert "result" in _rpc("session.close", {"session_id": sid})
        assert db.get_session(key) is not None and not db.is_session_temporary(key)
    finally:
        server._sessions.pop(sid, None)


def test_temporary_chat_cannot_be_branched(db):
    sid, key = _open_temporary_chat(db)
    try:
        resp = _rpc("session.create", {"cols": 96, "source": "desktop", "parent_session_id": key,
                                       "messages": [{"role": "user", "content": "incognito question"}]})
        assert resp["error"]["code"] == 4008
    finally:
        server._sessions.pop(sid, None)


def test_live_branch_of_a_temporary_chat_is_refused(db, monkeypatch):
    """``session.branch`` (Desktop branch actions, TUI /branch) would copy the transcript into a saved,
    listed chat with memory and titling on."""
    sid, key = _open_temporary_chat(db)
    # The live-session gate waits for a built agent; the fixture builds none, so fake one.
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    session = server._sessions[sid]
    session.update(agent=object(), agent_error=None)
    if (ready := session.get("agent_ready")) is not None:
        ready.set()
    try:
        for method in ("session.branch", "session.branch_whole"):
            error = _rpc(method, {"session_id": sid})["error"]
            assert (error["code"], error["message"]) == (4008, "temporary chats can't be branched")
        assert db.is_session_temporary(key) and db.get_session(key)["end_reason"] is None  # untouched
        # Fail closed: when the stored flag cannot be read, the transcript is not copied either.
        session.pop("temporary", None)
        monkeypatch.setattr(db, "is_session_temporary", lambda _sid: (_ for _ in ()).throw(RuntimeError("locked")))
        assert _rpc("session.branch", {"session_id": sid})["error"]["code"] == 4008
        resp = _rpc("session.create", {"cols": 96, "source": "desktop", "parent_session_id": key,
                                       "messages": [{"role": "user", "content": "incognito question"}]})
        assert resp["error"]["code"] == 4008
    finally:
        server._sessions.pop(sid, None)


def test_startup_sweep_spares_live_temporary_chats(db):
    sid, key = _open_temporary_chat(db)
    try:
        db.mark_session_temporary(key, owner={"pid": 2**22 + 7, "host": "elsewhere"})  # owner unverifiable + old
        assert server._sweep_orphaned_temporary_rows() == []  # the chat is live in this backend
        server._sessions.pop(sid)
        assert server._sweep_orphaned_temporary_rows() == []  # unverifiable owner, not idle long enough
    finally:
        server._sessions.pop(sid, None)


def test_backend_shutdown_deletes_open_temporary_chats(db):
    """Quitting Desktop tears every session down as ``tui_shutdown``: a temporary chat goes with it."""
    sid, key = _open_temporary_chat(db)
    session = server._pop_session_by_id(sid)
    server._teardown_popped_session(session, end_reason="tui_shutdown")
    assert db.get_session(key) is None
