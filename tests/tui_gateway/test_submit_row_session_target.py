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


def _rotated_session(monkeypatch, db):
    """A live desktop session whose agent already rotated onto a continuation while ``session_key``
    still names the (reopened) parent — the state every turn after a compression rotation sees."""
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    with session["history_lock"]:
        session["running"] = True
        server._start_inflight_turn(session, "earlier turn")
    assert server._ensure_session_db_row(session) is not False
    agent = _flush_agent(db, key)
    session["agent"] = agent
    return sid, key, session, agent, _rotate_to_compression_child(db, key, agent, reopen_parent=True)


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


def _full_rows(db, key):
    return [(r["role"], str(r["content"])) for r in db.get_messages_as_conversation(key, include_inactive=True)]


def test_model_switch_marker_lands_in_the_live_session(monkeypatch, tmp_path):
    """``_append_model_switch_marker`` writes a DURABLE ``role=user`` pivot under ``session_key`` while the
    live agent writes to ``agent.session_id``. On a rotated session the notice is filed under a parent the
    conversation no longer reads — the reporter's 130 stray ``model_switch`` rows. ``personality_switch``
    is unaffected: it only ever touches ``session["history"]``, never the DB."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, agent, child = _rotated_session(monkeypatch, db)
    try:
        server._append_model_switch_marker(session, model="test-model-2", provider="test-provider")
        prefix = server._MODEL_SWITCH_MARKER_PREFIX
        assert [r for r in _full_rows(db, child) if prefix in r[1]], f"marker not in the live session: {_full_rows(db, child)}"
        assert not [r for r in _full_rows(db, key) if prefix in r[1]], (
            f"marker filed under the rotated-away parent: {_full_rows(db, key)}")
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_model_switch_markers_do_not_accumulate_across_switches(monkeypatch, tmp_path):
    """The in-memory path is self-replacing: each switch strips the prior marker so N switches leave ONE
    marker, not N re-sent on every API call (#65891). The DURABLE write had no counterpart, so N switches
    left N active rows that all replay on resume — the invariant held in memory only. Filed in the live
    session (see the test above), those rows are exactly the reporter's 130 stray markers."""
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
        for i in range(3):
            server._append_model_switch_marker(session, model=f"model-{i}", provider="test-provider")
        prefix = server._MODEL_SWITCH_MARKER_PREFIX
        # LIVE rows only (get_messages defaults to active=1), and full content: the marker prefix is
        # longer than the _rows() helper's 48-char preview.
        live = [r for r in db.get_messages(key) if prefix in str(r.get("content") or "")]
        assert len(live) == 1, (
            f"3 switches must leave 1 live durable marker, not {len(live)}: "
            f"{[str(r.get('content'))[:60] for r in live]}")
        # The superseded rows are preserved inactive, never deleted (same contract as deactivate_message).
        kept = [r for r in db.get_messages(key, include_inactive=True)
                if prefix in str(r.get("content") or "")]
        assert len(kept) == 3, f"superseded markers must be kept inactive, not deleted: {len(kept)}"
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_message_react_targets_the_row_in_the_session_that_owns_it(monkeypatch, tmp_path):
    """``message.react`` with ``newest_role`` resolves the row via ``latest_message_row_id``, which filters
    on one session_id. On a rotated session the newest user row lives in the continuation, so a stale
    ``session_key`` reacts to the PARENT's last user row — the previous turn — or 404s when the parent
    has no text row. Same defect class as the submit row: an off-turn write addressed by a key a
    rotation invalidated (#123545)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, agent, child = _rotated_session(monkeypatch, db)
    try:
        # The continuation's newest user row is the rotated turn's own submit row.
        db.append_message(child, "user", content="the turn the user just reacted to")
        # The parent still holds the PREVIOUS turn's last user row.
        db.append_message(key, "user", content="the previous turn")

        got = server.handle_request({"id": "r1", "method": "message.react", "params": {
            "session_id": sid, "newest_role": "user", "emoji": "thumbsup"}})
        assert "result" in got, got
        want = db._read_one("SELECT id FROM messages WHERE session_id = ? AND content = ?",
                            (child, "the turn the user just reacted to"))
        assert want is not None
        row_id = int(got["result"]["row_id"])
        assert row_id == want[0], (
            f"reaction landed on row {row_id} (session "
            f"{db._read_one('SELECT session_id FROM messages WHERE id = ?', (row_id,))[0]}), "
            f"not the newest user row {want[0]} in the live session {child}")
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_session_history_reads_the_continuation_after_a_rotation(monkeypatch, tmp_path):
    """``session.history`` addresses the durable transcript by session. ``include_ancestors`` walks PARENT
    pointers, so a stale ``session_key`` materializes root..parent and never the continuation — a
    reconnect in the post-rotation window renders a transcript missing every turn since the rotation."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, agent, child = _rotated_session(monkeypatch, db)
    try:
        db.append_message(child, "user", content="sent after the rotation")
        got = server.handle_request({"id": "h1", "method": "session.history", "params": {"session_id": sid}})
        assert "result" in got, got
        texts = [m.get("text") or m.get("content") for m in got["result"]["messages"]]
        assert "sent after the rotation" in texts, (
            f"session.history served the stale parent {key}, not the live {child}: {got['result']['messages']}")
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_out_of_band_probe_reads_the_continuation_after_a_rotation(monkeypatch, tmp_path):
    """``_adopt_out_of_band_turns`` keyset-probes for foreign rows (a Telegram reply, a cron delivery)
    written since the turn started. On a rotated session the parent holds none of them, so the probe
    returns nothing and the model never sees the out-of-band turn — a regression of the contract this
    function exists for (#42962/#86588)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, agent, child = _rotated_session(monkeypatch, db)
    try:
        from tui_gateway import prompt_turn
        # _adopt_out_of_band_turns reads _message_row_id, which methods_prompt publishes onto server's
        # globals at bind_module time (prompt_turn's own module never imports it). Importing the module
        # here runs that binding — the same order server.py's own import loop produces.
        from tui_gateway import methods_prompt
        assert hasattr(server, "_message_row_id"), "the bind seam must publish _message_row_id"
        # Stamp the in-memory history with the row ids the rotation actually created, so `seen` is the
        # newest row the agent's own flush wrote and the foreign row is strictly newer.
        with session["history_lock"]:
            session["history"] = [
                dict(r, _row_id=r["_row_id"]) for r in db.get_messages_as_conversation(child, include_row_ids=True)
            ]
            session["history_version"] = 1
        stamped = [m["_row_id"] for m in session["history"]]
        assert stamped, "the rotation must have created rows to stamp"
        seen = max(stamped)
        # Another surface appends to the session the LIVE agent writes to, after those rows.
        foreign = db.append_message(child, "user", content="a Telegram reply that arrived mid-turn")
        assert foreign > seen, f"the foreign row {foreign} must sort after the in-flight rows {stamped}"

        # Call the REBOUND copy on server: bind_module re-creates each body against server's globals, and
        # that copy is what production runs. prompt_turn's original still points at its own module dict,
        # where _message_row_id (published by methods_prompt) was never bound.
        server._adopt_out_of_band_turns(session)
        texts = [str(m.get("content")) for m in session["history"]]
        assert "a Telegram reply that arrived mid-turn" in texts, (
            f"the out-of-band probe read the stale parent {key} and adopted nothing: {texts}")
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




def test_busy_queue_accept_row_lands_with_the_turn_and_is_addressed_there(monkeypatch, tmp_path):
    """The queue accept shares ``_write_submit_user_row`` and then addresses that row twice more:
    the text merge (``set_user_message_content``) and the drain deactivation
    (``deactivate_message``). Both UPDATEs are qualified by session_id, so addressing them by a
    rotated-away ``session_key`` silently updates zero rows — the merged text never lands and the
    accept-time row stays ACTIVE beside its replacement, the [uA, uB, aA] glue that
    ``_replace_queued_user_row_for_turn`` exists to prevent."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key, session, agent, child = _rotated_session(monkeypatch, db)
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "in-flight turn A")
        db.append_message(agent.session_id, "user", content="in-flight turn A")
        assert server._handle_busy_submit("r1", sid, session, "first QUEUED-A", "ws-1",
                                          queued=True, display_kind=None)["result"]["status"] == "queued"
        # The accept-time row is in the continuation, where the drained turn will write.
        assert [r for r in _rows(db, child) if "QUEUED-A" in r[1]]

        # A text-only follow-up merges into the envelope: the already-written row must follow.
        assert server._handle_busy_submit("r2", sid, session, "second", "ws-1",
                                          queued=True, display_kind=None)["result"]["status"] == "queued"
        assert session["queued_prompt"]["text"] == "first QUEUED-A\n\nsecond"
        merged = [r[1] for r in _rows(db, child) if r[0] == "user" and "QUEUED-A" in r[1]]
        assert merged == ["first QUEUED-A\n\nsecond"], f"merge update missed the row's session: {merged}"

        # Drain: the accept-time row is re-placed at the end and the early one deactivated.
        server._replace_queued_user_row_for_turn(session, session["queued_prompt"])
        assert len([r for r in _rows(db, child) if "QUEUED-A" in r[1]]) == 2
        active = [r for r in db.get_messages_as_conversation(child, include_row_ids=True)
                  if "QUEUED-A" in str(r["content"])]
        assert len(active) == 1, f"deactivation missed the row's session: {len(active)} still active"
    finally:
        server._sessions.pop(sid, None)
        db.close()
