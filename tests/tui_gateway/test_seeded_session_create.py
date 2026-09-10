"""session.create with seeded messages: the seed is durable before the first prompt, and durable once."""

from hermes_state import SessionDB
from tui_gateway import server


def _quiet_create(monkeypatch, db):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)


def _create(params: dict) -> dict:
    resp = server.handle_request({"id": "create", "method": "session.create", "params": params})
    assert "result" in resp, resp
    return resp["result"]


def test_parentless_seed_survives_a_restart_and_hides_its_runbook(monkeypatch, tmp_path):
    """A client that opens a chat with its first turns already written (no parent) gets a durable row and
    transcript at create: a restart before the first prompt resumes it, the hidden runbook stays out of the
    transcript on the wire, and the seed is not written a second time by the first-submit path."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _quiet_create(monkeypatch, db)
    sids = []
    try:
        result = _create({
            "cols": 96, "source": "desktop", "title": "Welcome to Hermes",
            "messages": [
                {"role": "user", "content": "Private setup runbook", "display_kind": "hidden"},
                {"role": "assistant", "content": "Welcome to Hermes"},
                # Only "hidden" is accepted from the wire; other kinds are stamped by the gateway itself.
                {"role": "user", "content": "Second question", "display_kind": "steer"},
            ]})
        sids.append(result["session_id"])
        key = result["stored_session_id"]
        assert [m["role"] for m in result["messages"]] == ["assistant", "user"]
        assert result["message_count"] == 2  # counts what is on the wire, as session.resume does

        assert db.get_session(key)["title"] == "Welcome to Hermes"
        rows = db.get_messages_as_conversation(key)
        assert [r["content"] for r in rows] == ["Private setup runbook", "Welcome to Hermes", "Second question"]
        assert rows[0]["display_kind"] == "hidden"
        assert rows[2].get("display_kind") is None
        listed = server.handle_request({"id": "list", "method": "session.list", "params": {}})["result"]["sessions"]
        assert next(s for s in listed if s["id"] == key)["preview"].startswith("Second question")  # the hidden row is not the preview

        server._sessions.pop(sids.pop())  # the gateway restarts; only state.db remains
        resumed = server.handle_request({"id": "resume", "method": "session.resume", "params": {"session_id": key, "cols": 96}})
        assert "result" in resumed, resumed
        sids.append(resumed["result"]["session_id"])
        assert [m["role"] for m in resumed["result"]["messages"]] == ["assistant", "user"]

        server._persist_branch_seed(server._sessions[sids[-1]])  # what the first prompt.submit calls
        assert len(db.get_messages_as_conversation(key)) == 3
    finally:
        for sid in sids:
            server._sessions.pop(sid, None)
        db.close()


def test_branch_child_seed_is_written_once(monkeypatch, tmp_path):
    """A seeded branch child persists its copied transcript at create (#93959); the first prompt's seed
    persist is the fallback for a failed create-time copy, not a second copy."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _quiet_create(monkeypatch, db)
    seed = [{"role": "user", "content": "hello from parent"}, {"role": "assistant", "content": "parent reply"}]
    db.create_session("parent-1", source="desktop")
    db.append_messages_batch("parent-1", seed)
    db.set_session_title("parent-1", "Parent chat")
    sid = None
    try:
        result = _create({"cols": 96, "source": "desktop", "parent_session_id": "parent-1", "messages": seed})
        sid, key = result["session_id"], result["stored_session_id"]
        assert [r["content"] for r in db.get_messages_as_conversation(key)] == ["hello from parent", "parent reply"]

        server._persist_branch_seed(server._sessions[sid])
        assert [r["content"] for r in db.get_messages_as_conversation(key)] == ["hello from parent", "parent reply"]
    finally:
        if sid:
            server._sessions.pop(sid, None)
        db.close()
