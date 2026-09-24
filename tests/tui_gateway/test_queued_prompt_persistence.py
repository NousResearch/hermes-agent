"""A prompt accepted while the agent is busy is durable at ACCEPT time.

``_handle_busy_submit`` used to keep the queued turn in memory only: a cold transcript read
(``session.resume``) lacked the message until its turn ran, and a backend restart lost it
permanently. The accept now writes the user row through the same #111868 machinery the idle
submit uses, keeps the envelope's row content in sync when a text-only arrival merges, and
re-places the row at the transcript end when the queued turn actually dispatches, so the
stored raw transcript stays [user, assistant, user, assistant] and the turn adopts its row
instead of writing a duplicate.
"""

import types

from agent.turn_context import _stage_turn_user_message
from hermes_state import SessionDB
from run_agent import AIAgent
from tui_gateway import server


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


def _active_rows(db, key):
    return db.get_messages_as_conversation(key, repair_alternation=True, include_row_ids=True)


def _flush_agent(db, key):
    """Agent shell owning the real flush (the crash persist at turn start runs this same code)."""
    agent = types.SimpleNamespace(
        _session_db=db, _session_db_created=True, _persist_disabled=False, session_id=key,
        _session_persist_lock=None, _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0, _persist_user_message_idx=None, _persist_user_message_override=None,
        _persist_user_message_timestamp=None, _pending_cli_user_message=None)
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    agent._flush_messages_to_session_db_unlocked = AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    return agent


def _run_turn(session, db, key, text, reply):
    """The turn body the real ``_run_prompt_submit`` runs once the agent is ready: adopt the
    staged row, crash-persist the user turn, flush the reply (mirrors test_submit_time_user_row)."""
    agent = _flush_agent(db, key)
    server._adopt_submit_user_row(session, agent, text, text)
    user_msg, _pending = _stage_turn_user_message(agent, text, text, None, None, None, None)
    messages = [user_msg]
    agent._persist_user_message_idx = 0
    agent._flush_messages_to_session_db(messages, [])
    agent._flush_messages_to_session_db(messages + [{"role": "assistant", "content": reply}], [])


def _accept_busy_then_run_both_turns(monkeypatch, tmp_path, queued_text="queued text QUEUED-MARKER"):
    """Real two-turn flow: turn A's row is in the transcript, turn A is live, B is accepted busy,
    A's reply lands, the real ``_drain_queued_prompt`` dispatches B and B's turn body (adopt +
    crash-persist + flush) writes reply B. Returns ``(db, sid, key)`` in steady state."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    server._ensure_session_db_row(session)  # the lazy row a real first submit would have written
    db.append_message(key, "user", content="prompt A")  # turn A's row, as A's turn would have written it
    _busy(session)
    resp = server._handle_busy_submit("r1", sid, session, queued_text, "ws-1", queued=True, display_kind=None)
    assert resp["result"]["status"] == "queued"
    db.append_message(key, "assistant", content="reply A")  # turn A concludes
    with session["history_lock"]:
        session["running"] = False
        server._clear_inflight_turn(session)
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda rid, s, sess, text, **kw: _run_turn(sess, db, key, text, "reply B"))
    assert server._drain_queued_prompt("r2", sid, session) is True
    return db, sid, key


def test_busy_accept_writes_the_queued_user_row_immediately(monkeypatch, tmp_path):
    """RED for the bug: accept acked {"status": "queued"} with no DB write, so a cold read
    (session.resume) or a restart saw nothing until the queued turn actually ran."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        _busy(session)
        resp = server._handle_busy_submit("r1", sid, session, "queued text QUEUED-MARKER", "ws-1",
                                          queued=True, display_kind=None)
        assert resp["result"]["status"] == "queued"
        # The turn has not run: the cold resume read must already see the accepted message.
        assert any(r["role"] == "user" and "queued text QUEUED-MARKER" in str(r["content"])
                   for r in _active_rows(db, key))
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_queued_turn_replays_as_its_own_turn_after_the_live_turn(monkeypatch, tmp_path):
    """The accept-time row lands BEFORE the in-flight turn's assistant rows (raw [uA, uB, aA]);
    the drain must re-place it at the transcript end, so the repaired projection keeps FOUR
    separate messages instead of gluing the two user turns into one."""
    db, sid, key = _accept_busy_then_run_both_turns(monkeypatch, tmp_path)
    try:
        assert [(r["role"], r["content"]) for r in _active_rows(db, key)] == [
            ("user", "prompt A"), ("assistant", "reply A"),
            ("user", "queued text QUEUED-MARKER"), ("assistant", "reply B")]
        # Exactly one ACTIVE row carries the queued text; the accept-time row survives inactive.
        active = [r for r in _active_rows(db, key) if "QUEUED-MARKER" in str(r["content"])]
        assert len(active) == 1
        every = db.get_messages_as_conversation(key, include_inactive=True, include_row_ids=True)
        superseded = [r for r in every if "QUEUED-MARKER" in str(r["content"]) and r["_row_id"] != active[0]["_row_id"]]
        assert len(superseded) == 1  # durable history, never deleted
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_queued_prompt_survives_a_backend_restart(monkeypatch, tmp_path):
    """A queued prompt is durable at accept: a brand-new SessionDB on the same file (the shape a
    restarted backend opens) reads the queued message back."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        _busy(session)
        server._handle_busy_submit("r1", sid, session, "queued text QUEUED-MARKER", "ws-1",
                                   queued=True, display_kind=None)
        fresh = SessionDB(db_path=tmp_path / "state.db")  # a restarted backend opens a new handle
        try:
            assert any(r["role"] == "user" and "queued text QUEUED-MARKER" in str(r["content"])
                       for r in fresh.get_messages_as_conversation(key, repair_alternation=True,
                                                                   include_row_ids=True))
        finally:
            fresh.close()
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_merged_queue_text_updates_the_written_row_in_place(monkeypatch, tmp_path):
    """A text-only arrival merges into the queued envelope; the already-written row must not
    lag the envelope, or cold readers see half the merged prompt and adoption stops matching."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        _busy(session)
        assert server._handle_busy_submit("r1", sid, session, "first QUEUED-MARKER-A", "ws-1",
                                          queued=True, display_kind=None)["result"]["status"] == "queued"
        assert server._handle_busy_submit("r2", sid, session, "second", "ws-1",
                                          queued=True, display_kind=None)["result"]["status"] == "queued"
        assert session["queued_prompt"]["text"] == "first QUEUED-MARKER-A\n\nsecond"
        rows = [r for r in _active_rows(db, key) if r["role"] == "user"]
        assert len(rows) == 1  # merge syncs the ONE row, never appends a second
        assert rows[0]["content"] == session["queued_prompt"]["text"]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_cleared_queue_keeps_the_queued_row_as_the_interrupted_shape(monkeypatch, tmp_path):
    """A cancelled queued prompt keeps its trailing user row: a trailing user row with no reply
    is the documented interrupted-transcript shape (never delete it on cancel)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        _busy(session)
        server._handle_busy_submit("r1", sid, session, "queued text QUEUED-MARKER", "ws-1",
                                   queued=True, display_kind=None)
        server._ac_set_queue(session, [])  # Stop / queue clear
        assert not session.get("queued_prompt") and not session.get("queued_prompts")
        rows = _active_rows(db, key)
        assert rows and rows[-1]["role"] == "user" and "queued text QUEUED-MARKER" in str(rows[-1]["content"])
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_drained_turn_adopts_the_replaced_row_and_writes_no_duplicate(monkeypatch, tmp_path):
    """Four active rows after the whole flow (uA, aA, uB, aB): the drained turn adopted the
    re-placed row instead of appending a fifth."""
    db, sid, key = _accept_busy_then_run_both_turns(monkeypatch, tmp_path)
    try:
        assert len(db.get_messages_as_conversation(key)) == 4
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_partial_drain_never_puts_a_later_prompt_before_an_earlier_one(monkeypatch, tmp_path):
    """Two non-mergeable (image-bearing) prompts accepted mid-turn; only the FIRST drains, then the
    process dies (the in-memory queue is lost). Acceptance order must still hold in the transcript:
    the second prompt must never render before the first. Without queue-wide re-placement at each
    drain, the second prompt's accept-time row stays ahead of the in-flight reply while the first's
    heals past it — permanently rendering the later prompt before the earlier one."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        server._ensure_session_db_row(session)  # the lazy row a real first submit would have written
        db.append_message(key, "user", content="prompt A")  # turn A's row
        _busy(session)
        session["attached_images"] = ["/tmp/b.png"]
        assert server._handle_busy_submit("r1", sid, session, "prompt B QUEUED-B", "ws-1",
                                          queued=True, display_kind=None)["result"]["status"] == "queued"
        session["attached_images"] = ["/tmp/c.png"]
        assert server._handle_busy_submit("r2", sid, session, "prompt C QUEUED-C", "ws-1",
                                          queued=True, display_kind=None)["result"]["status"] == "queued"
        db.append_message(key, "assistant", content="reply A")  # turn A concludes
        with session["history_lock"]:
            session["running"] = False
            server._clear_inflight_turn(session)
        monkeypatch.setattr(server, "_run_prompt_submit",
                            lambda rid, s, sess, text, **kw: _run_turn(sess, db, key, text, "reply B"))
        assert server._drain_queued_prompt("r3", sid, session) is True  # drains B only
        # The crash: the queue (and C's turn intent) is gone with the process.
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)
        rendered = "".join(str(r["content"]) for r in _active_rows(db, key))
        assert "prompt B QUEUED-B" in rendered and "prompt C QUEUED-C" in rendered  # durable, both kept
        assert rendered.index("prompt B QUEUED-B") < rendered.index("prompt C QUEUED-C"), \
            "a later-accepted prompt rendered before an earlier one after a partial drain"
    finally:
        server._sessions.pop(sid, None)
        db.close()
