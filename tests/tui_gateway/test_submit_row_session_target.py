"""One Desktop turn writes its user row and its tool rows into TWO sessions (#123545, evidence A + B).

``prompt.submit``'s durable user row is an OFF-turn write: it has only ``session["session_key"]``, because
the agent may not be built yet. The turn's own transcript is flushed under ``agent.session_id``. Those
two ids diverge for the whole of a turn that begins after the agent's session rotated: the compression
continuation, a lease-wait re-resolve, or an adopted tip all move ``agent.session_id`` while
``session_key`` is re-anchored only at turn end. Result: the user row lands in the parent, every tool
row and the final assistant text in the child, and the user reads one chat and sees part of the turn.

The regression test drives the REAL rotation (``publish_compression_child`` closes the parent and
mints the continuation) and asserts one turn's rows stay in ONE session.
"""

from types import SimpleNamespace

from agent.turn_context import _stage_turn_user_message
from hermes_state import SessionDB
from run_agent import AIAgent
from tui_gateway import server


def _flush_agent(db, key):
    agent = SimpleNamespace(
        _session_db=db, _session_db_created=True, _persist_disabled=False, session_id=key,
        _session_persist_lock=None, _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0, _persist_user_message_idx=None, _persist_user_message_override=None,
        _persist_user_message_timestamp=None, _pending_cli_user_message=None)
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    agent._flush_messages_to_session_db_unlocked = AIAgent._flush_messages_to_session_db_unlocked.__get__(
        agent, AIAgent)
    return agent


def _desktop_session(monkeypatch, db):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    resp = server.handle_request(
        {"id": "c", "method": "session.create", "params": {"cols": 96, "source": "desktop"}})
    assert "result" in resp, resp
    return resp["result"]["session_id"], resp["result"]["stored_session_id"]


def _rotate_to_compression_child(db, parent, agent, *, reopen_parent):
    """The production rotation: the real publisher closes the parent and mints the continuation.

    ``reopen_parent`` models a ``session.resume`` of the pre-rotation id, which reopens the parent row
    unconditionally (``methods_session.read_history`` / ``_schedule_resume_hydration``). Left closed,
    the ``_ended_by_compression`` guard refuses the submit append outright and the row is silently
    dropped; reopened, the stale key is writable and the turn's rows split across both sessions.
    """
    from hermes_state_ids import new_session_id

    child = new_session_id()
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="desktop", model="test-model",
        messages=[{"role": "user", "content": "earlier turn"}, {"role": "assistant", "content": "earlier reply"}],
        compression_lock_holder=None, require_compression_lease=False)
    agent.session_id = child  # what every adoption/rotation path does to the live agent
    if reopen_parent:
        db.reopen_session(parent)
    return child

def _rows(db, key):
    return [(r["role"], (r["content"] or "")[:48]) for r in
            db.get_messages_as_conversation(key, include_inactive=True)]


def test_submit_user_row_lands_where_the_turns_tool_rows_land(monkeypatch, tmp_path):
    """The reporter's exact shape: one typed message, a tool result and the final text — all one session."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        # An earlier turn's activity made the parent row real; the rotation below needs it to exist.
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "earlier turn")
        assert server._ensure_session_db_row(session) is not False
        agent = _flush_agent(db, key)
        session["agent"] = agent
        # The PREVIOUS turn rotated the agent onto the continuation; ``session_key`` still names the
        # parent because the re-anchor happens at turn end. This is the state every later turn sees.
        child = _rotate_to_compression_child(db, key, agent, reopen_parent=True)
        assert session["session_key"] == key and agent.session_id == child

        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "Lets make things right")
        assert server._persist_session_row_for_submit("rid", session, "Lets make things right", None) is None

        server._adopt_submit_user_row(session, agent, "Lets make things right", "Lets make things right")
        user_msg, _pending = _stage_turn_user_message(
            agent, "Lets make things right", "Lets make things right", None, None, None, None)
        messages = [user_msg]
        agent._persist_user_message_idx = 0
        agent._flush_messages_to_session_db(messages, [])            # turn-start crash persist
        agent._flush_messages_to_session_db(                          # turn end: tool row + final text
            messages + [
                {"role": "assistant", "content": "calling a tool", "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "memory", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "t1", "name": "memory", "content": "failed write"},
                {"role": "assistant", "content": "done"},
            ], [])

        parent_user_rows = [r for r in _rows(db, key) if r[0] == "user" and r[1] == "Lets make things right"]
        assert not parent_user_rows, f"user row written to the rotated-away parent {key}: {parent_user_rows}"
        child_rows = _rows(db, child)
        assert child_rows == [
            ("user", "earlier turn"), ("assistant", "earlier reply"),  # the handoff
            ("user", "Lets make things right"), ("assistant", "calling a tool"),
            ("tool", "failed write"), ("assistant", "done"),
        ], f"the turn's rows must follow agent.session_id into the continuation: {child_rows}"
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_expanded_submit_row_is_rewritten_on_the_session_that_owns_it(monkeypatch, tmp_path):
    """The @-expansion rewrite addresses the row by (session_id, row_id). When the submit row was written to
    the continuation, the rewrite must address it there — a stale session_key misses the row silently."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "earlier turn")
        assert server._ensure_session_db_row(session) is not False
        agent = _flush_agent(db, key)
        session["agent"] = agent
        child = _rotate_to_compression_child(db, key, agent, reopen_parent=True)

        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "look at @notes.md")
        assert server._persist_session_row_for_submit("rid", session, "look at @notes.md", None) is None

        expanded = "look at @notes.md\n\n<file notes.md>todo</file>"
        server._adopt_submit_user_row(session, agent, expanded, "look at @notes.md")
        assert [r[1] for r in _rows(db, child) if r[0] == "user" and "notes.md" in r[1]] == [expanded], (
            f"the rewritten row must live in the continuation {child}: {_rows(db, child)}")
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_unrotated_session_keeps_writing_to_session_key(monkeypatch, tmp_path):
    """The ordinary case is unchanged: no rotation, the row lands under session_key as before."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "plain send")
        agent = _flush_agent(db, key)
        session["agent"] = agent
        assert server._persist_session_row_for_submit("rid", session, "plain send", None) is None
        assert _rows(db, key) == [("user", "plain send")]
    finally:
        server._sessions.pop(sid, None)
        db.close()
