"""A Desktop queue entry (``message_id`` + ``submitted_at``) is durable at ACCEPT, before the ack.

Desktop sends ``message_id`` and ``submitted_at``, treats the ``queued`` response as accepted and
removes its local entry. The accept-time persist used to skip source-identified envelopes (the
drained turn's persist owned the row), so a backend stop before ``_drain_queued_prompt`` left the
prompt in memory only and it disappeared after restart (PR #63298 review). The ACCEPT-time persist
is THE durable write for every envelope: prompt text AND stable identity, written synchronously
before the ack. The drain FINALIZES that row in place — never mints a second row for the same
identity — and the never-drained restart retire stays scoped to anonymous residue.
"""

import types

from hermes_state import SessionDB
from tui_gateway import server

_MSG_ID = "desktop-msg-7"
_SUBMITTED_AT = 1728000000.5


def _desktop_session(monkeypatch, db):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    resp = server.handle_request({"id": "c", "method": "session.create", "params": {"cols": 96, "source": "desktop"}})
    assert "result" in resp, resp
    sid = resp["result"]["session_id"]
    server._sessions[sid]["agent"] = types.SimpleNamespace()
    return sid, resp["result"]["stored_session_id"]


def _busy(session, in_flight="prompt A"):
    with session["history_lock"]:
        session["running"] = True
        server._start_inflight_turn(session, in_flight)


def _identity_rows(db, key, message_id):
    return [r for r in db.get_messages(key, include_inactive=True)
            if str(r.get("platform_message_id") or "") == message_id]


def _run_identity_turn(session, db, key, text, reply):
    """The turn body of a drained Desktop envelope: adopt the accept-time row, crash-persist the
    user turn with the source identity ``_invoke_agent`` stamps, flush the reply."""
    from agent.turn_context import _stage_turn_user_message
    from tests.tui_gateway.test_queued_prompt_persistence import _flush_agent
    agent = _flush_agent(db, key)
    server._adopt_submit_user_row(session, agent, text, text)
    user_msg, _pending = _stage_turn_user_message(
        agent, text, text, _SUBMITTED_AT, _MSG_ID, None, None)
    messages = [user_msg]
    agent._persist_user_message_idx = 0
    agent._flush_messages_to_session_db(messages, [])
    agent._flush_messages_to_session_db(messages + [{"role": "assistant", "content": reply}], [])


def _accept_desktop_envelope(monkeypatch, tmp_path):
    """Turn A live, a Desktop-style envelope accepted busy (message_id + submitted_at). The
    ``queued`` ack already returned; the turn has NOT run."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    server._ensure_session_db_row(session)  # the lazy row a real first submit would have written
    db.append_message(key, "user", content="prompt A")  # turn A's row
    _busy(session)
    resp = server._handle_busy_submit("r1", sid, session, "desktop prompt DESKTOP-MARKER", "ws-1",
                                      queued=True, display_kind=None,
                                      submitted_at=_SUBMITTED_AT, message_id=_MSG_ID)
    assert resp["result"]["status"] == "queued"
    return db, sid, key, session


def test_desktop_accept_is_durable_before_the_queued_ack(monkeypatch, tmp_path):
    """RED for the bug (PR #63298 review): the accept acked ``{"status": "queued"}`` and Desktop
    dropped its local entry, but a source-identified envelope's prompt existed only in memory —
    a fresh SessionDB (the shape a restarted backend opens) read nothing back. The accepted
    response must not come before durable storage of the prompt and its stable identity."""
    db, sid, key, _session = _accept_desktop_envelope(monkeypatch, tmp_path)
    try:
        fresh = SessionDB(db_path=tmp_path / "state.db")
        try:
            rows = fresh.get_messages_as_conversation(key, repair_alternation=True, include_row_ids=True)
            assert any(r["role"] == "user" and "desktop prompt DESKTOP-MARKER" in str(r["content"])
                       for r in rows), "accepted Desktop prompt missing from a fresh SessionDB"
            assert fresh.has_platform_message_id(key, _MSG_ID), \
                "the stable client identity must be durable with the prompt (retry dedup)"
        finally:
            fresh.close()
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_desktop_queued_prompt_survives_restart_and_reopen(monkeypatch, tmp_path):
    """The accepted ack transferred ownership to the backend: after a stop before the drain, the
    reopen ritual every resume path runs must keep the prompt present. Its stable source id is its
    turn boundary: never glued into turn A's user message, never retired as anonymous queue
    residue."""
    db, sid, key, session = _accept_desktop_envelope(monkeypatch, tmp_path)
    try:
        db.append_message(key, "assistant", content="reply A")  # turn A concludes; the drain never runs
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)  # the restart: the in-memory queue is gone
        fresh = SessionDB(db_path=tmp_path / "state.db")
        try:
            fresh.reopen_session(key)
            repaired = fresh.get_messages_as_conversation(key, repair_alternation=True, include_row_ids=True)
            assert [(r["role"], r["content"]) for r in repaired] == [
                ("user", "prompt A"),
                ("user", "desktop prompt DESKTOP-MARKER"),  # its own boundary row, never folded
                ("assistant", "reply A"),
            ]
        finally:
            fresh.close()
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_desktop_drain_finalizes_one_row_for_the_identity(monkeypatch, tmp_path):
    """Accept -> drain -> exactly ONE user row for the identity: the drain FINALIZES the accept-time
    row in place and the drained turn adopts it. A second row for the same identity would break
    retry dedup and split one message across two transcript entries."""
    db, sid, key, session = _accept_desktop_envelope(monkeypatch, tmp_path)
    try:
        db.append_message(key, "assistant", content="reply A")
        with session["history_lock"]:
            session["running"] = False
            server._clear_inflight_turn(session)

        def _dispatch(rid, s, sess, text, **kw):
            # Source identity rides the drained turn exactly as the real dispatch sends it.
            assert kw.get("submitted_at") == _SUBMITTED_AT and kw.get("message_id") == _MSG_ID
            _run_identity_turn(sess, db, key, text, "reply B")

        monkeypatch.setattr(server, "_run_prompt_submit", _dispatch)
        assert server._drain_queued_prompt("r2", sid, session) is True

        rows = _identity_rows(db, key, _MSG_ID)
        assert len(rows) == 1, f"the drain must not mint a second row for the identity: {rows}"
        assert rows[0]["active"] == 1
        assert "DESKTOP-MARKER" in str(rows[0]["content"])
        assert [(r["role"], r["content"]) for r in db.get_messages_as_conversation(key)] == [
            ("user", "prompt A"), ("user", "desktop prompt DESKTOP-MARKER"),
            ("assistant", "reply A"), ("assistant", "reply B")]
        # The drained row is finalized, not never-drained residue: reopen keeps it.
        fresh = SessionDB(db_path=tmp_path / "state.db")
        try:
            fresh.reopen_session(key)
            assert [(r["active"], r["content"]) for r in _identity_rows(fresh, key, _MSG_ID)] == [
                (1, "desktop prompt DESKTOP-MARKER")]
        finally:
            fresh.close()
    finally:
        server._sessions.pop(sid, None)
        db.close()
