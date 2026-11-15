"""Regression for #42962: a turn another surface (Telegram, cron) appended to a live desktop/TUI session must
reach the model on the next local prompt, not only the repainted transcript.

Second half (#81951): the same reconciliation must not hinge on the live record's in-memory ``_row_id``
stamps. A record whose history was rebuilt from provider-format messages carries none, and the gateway's
concurrent rows were then silently dropped — the UI showed the full transcript while the provider request
held only the early snapshot.
"""

import contextlib
import threading

from hermes_state import SessionDB
from tui_gateway import server


def _bind_db(monkeypatch, db):
    @contextlib.contextmanager
    def _owner_db(session):
        yield db
    monkeypatch.setattr(server, "_session_db", _owner_db)


def _seed(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    db.append_message("s1", "user", "My codeword is MANGO.", timestamp=1.0)
    db.append_message("s1", "assistant", "OK", timestamp=2.0)
    # What the agent's own flush leaves in memory: the rows stamped with their durable ids.
    history = db.get_messages_as_conversation("s1", include_row_ids=True)
    return db, {"session_key": "s1", "history": history, "history_lock": threading.Lock(), "history_version": 0}


def test_next_turn_adopts_rows_another_writer_appended(tmp_path, monkeypatch):
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    # A Telegram turn lands on the same session while the desktop is idle; its reply repeats an
    # earlier one verbatim, so a text anchor would have mis-cut here — the row id boundary must not.
    db.append_message("s1", "user", "My second codeword is KIWI.", timestamp=3.0)
    db.append_message("s1", "assistant", "OK", timestamp=4.0)
    # This turn's own prompt is already durable at submit (#111868) and must NOT be adopted as foreign.
    own = db.append_message("s1", "user", "List every codeword.", timestamp=5.0)
    session["_submit_user_row"] = {"role": "user", "content": "List every codeword.", "_row_id": own}

    server._adopt_out_of_band_turns(session)

    assert [m["content"] for m in session["history"]] == [
        "My codeword is MANGO.", "OK", "My second codeword is KIWI.", "OK"]
    assert session["history_version"] == 1
    # Adopted rows are stamped: a second pass (next turn) finds nothing new.
    server._adopt_out_of_band_turns(session)
    assert len(session["history"]) == 4 and session["history_version"] == 1


def test_compaction_epochs(tmp_path, monkeypatch):
    """A local compaction re-stamps the in-memory dicts with the re-inserted rows' ids, so nothing sits above
    the boundary. A compaction by ANOTHER surface rewrites the transcript under us: the summary shows up as a
    foreign row and the history is re-hydrated from the DB instead of appended to."""
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    local = [{"role": "assistant", "content": "summary of MANGO", "_compressed_summary": True}, session["history"][-1]]
    db.archive_and_compact("s1", local, tail_count=1)
    session["history"] = local
    server._adopt_out_of_band_turns(session)
    assert [m.get("_row_id") for m in session["history"]] == [3, 4] and session["history_version"] == 0

    remote = [{"role": "assistant", "content": "summary of MANGO", "_compressed_summary": True},
              {"role": "user", "content": "Repeat it"}, {"role": "assistant", "content": "MANGO"}]  # its own copies
    db.archive_and_compact("s1", remote, tail_count=1)
    db.append_message("s1", "user", "KIWI")
    db.append_message("s1", "assistant", "OK")
    own = db.append_message("s1", "user", "List every codeword")
    session["_submit_user_row"] = {"role": "user", "content": "List every codeword", "_row_id": own}
    server._adopt_out_of_band_turns(session)

    assert [m["content"] for m in session["history"]] == ["summary of MANGO", "Repeat it", "MANGO", "KIWI", "OK"]
    assert session["history"][0].get("_compressed_summary") and session["history_version"] == 1


# ── #81951: the live record that lost its durable stamps ─────────────────────

def _seed_exchange(db, n: int, *, session_id: str = "s1") -> None:
    """One well-formed exchange, so alternation repair is a no-op on the lineage."""
    call = [{"id": f"c{n}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]
    db.append_message(session_id, "user", f"ask{n}", timestamp=float(n * 10))
    db.append_message(session_id, "assistant", "", timestamp=float(n * 10 + 1), tool_calls=call)
    db.append_message(session_id, "tool", f"probe{n}", timestamp=float(n * 10 + 2),
                      tool_call_id=f"c{n}", tool_name="terminal")
    db.append_message(session_id, "assistant", f"answer{n}", timestamp=float(n * 10 + 3))


def _hydrated_record(db, *, stamped: bool) -> dict:
    """The desktop's live record as hydration leaves it: the first exchange, row ids or not."""
    history = db.get_messages_as_conversation("s1", repair_alternation=True, include_row_ids=True)
    if not stamped:
        for message in history:
            message.pop("_row_id", None)
    return {"session_key": "s1", "history": history, "history_lock": threading.Lock(), "history_version": 0}


def _gateway_appends(db, turns: int = 10) -> None:
    for n in range(1, turns + 1):
        _seed_exchange(db, n)


def _immediate_thread(monkeypatch) -> None:
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)


def _desktop_prompt(db, session, monkeypatch, text: str = "continue on desktop") -> list:
    """Drive ``prompt.submit`` on ``session``; returns the ``conversation_history`` the model was given."""
    seen = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            seen["history"] = list(conversation_history or [])
            return {"final_response": "ok", "messages": [
                *(conversation_history or []), {"role": "user", "content": prompt},
                {"role": "assistant", "content": "ok"}]}

    session["agent"] = _Agent()
    server._sessions["sid-81951"] = session
    _bind_db(monkeypatch, db)
    _immediate_thread(monkeypatch)
    monkeypatch.setattr(server, "_get_usage", lambda _a: {})
    monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
    monkeypatch.setattr(server, "_emit", lambda *a: None)
    try:
        resp = server.handle_request({"id": "1", "method": "prompt.submit",
                                      "params": {"session_id": "sid-81951", "text": text}})
        assert resp.get("result"), resp
    finally:
        server._sessions.pop("sid-81951", None)
    return seen["history"]


def test_desktop_prompt_carries_gateway_rows_appended_after_hydration(tmp_path, monkeypatch):
    """The reported shape: desktop hydrated the first exchange, the gateway appended ten more, and the
    next desktop prompt must send the whole conversation — not the early snapshot — to the provider."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    _seed_exchange(db, 0)
    session = _hydrated_record(db, stamped=True)
    _gateway_appends(db)

    contents = [m.get("content") for m in _desktop_prompt(db, session, monkeypatch)]

    assert contents[:4] == ["ask0", "", "probe0", "answer0"]
    assert [c for c in contents if isinstance(c, str) and c.startswith("answer")] == [
        f"answer{n}" for n in range(11)]


def test_unstamped_desktop_prompt_does_not_adopt_its_own_user_row(tmp_path, monkeypatch):
    """Through ``prompt.submit`` the store already holds THIS turn's user row when adoption runs
    (``_persist_submit_user_row``); the fallback must stop below it like the stamped path does, or the
    model sees the prompt twice and the live history keeps both copies."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    _seed_exchange(db, 0)
    session = _hydrated_record(db, stamped=False)
    _gateway_appends(db, turns=1)

    contents = [m.get("content") for m in _desktop_prompt(db, session, monkeypatch)]

    assert contents == ["ask0", "", "probe0", "answer0", "ask1", "", "probe1", "answer1"]
    assert [m["content"] for m in session["history"]].count("continue on desktop") == 1


def test_unstamped_live_record_is_caught_up_from_the_store(tmp_path, monkeypatch):
    """No durable row ids in memory: the store-ahead fallback still folds the gateway's rows in."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    _seed_exchange(db, 0)
    session = _hydrated_record(db, stamped=False)
    _bind_db(monkeypatch, db)
    _gateway_appends(db)

    server._adopt_out_of_band_turns(session)

    assert session["history"][-1]["content"] == "answer10"
    assert len(session["history"]) == 44 and session["history_version"] == 1
    # Adopted rows are stamped, so the row-id path is authoritative again on the next turn.
    server._adopt_out_of_band_turns(session)
    assert len(session["history"]) == 44 and session["history_version"] == 1


def _foreign_compaction(db) -> None:
    """Another surface compacts the session under the live record, then keeps talking on it."""
    db.archive_and_compact("s1", [
        {"role": "assistant", "content": "summary of ask0/ask1", "_compressed_summary": True},
        {"role": "user", "content": "KIWI"}, {"role": "assistant", "content": "MANGO"}], tail_count=2)


def test_unstamped_record_is_rehydrated_after_a_foreign_compaction(tmp_path, monkeypatch):
    """The store was compacted by another surface: an unstamped pre-compaction history is stale from the
    root and must be re-hydrated from the DB exactly like the stamped path does, not kept because the
    compacted store is shorter."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    _seed_exchange(db, 0)
    _seed_exchange(db, 1)
    stamped, unstamped = _hydrated_record(db, stamped=True), _hydrated_record(db, stamped=False)
    _foreign_compaction(db)
    _bind_db(monkeypatch, db)

    for session in (stamped, unstamped):
        server._adopt_out_of_band_turns(session)
        assert [m["content"] for m in session["history"]] == ["summary of ask0/ask1", "KIWI", "MANGO"]
        assert session["history"][0].get("_compressed_summary") and session["history_version"] == 1


def test_unstamped_prompt_after_a_foreign_compaction_sends_the_compacted_transcript(tmp_path, monkeypatch):
    """Through ``prompt.submit``: the model is given the compacted transcript, not the stale pre-compaction
    one, and this turn's own user row is not adopted."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    _seed_exchange(db, 0)
    _seed_exchange(db, 1)
    session = _hydrated_record(db, stamped=False)
    _foreign_compaction(db)

    contents = [m.get("content") for m in _desktop_prompt(db, session, monkeypatch)]

    assert contents == ["summary of ask0/ask1", "KIWI", "MANGO"]


def test_unstamped_record_already_holding_the_summary_only_appends(tmp_path, monkeypatch):
    """An unstamped record that already carries the compaction (its own) is not re-hydrated: rows the
    gateway appended after it are folded in by the prefix path."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    _seed_exchange(db, 0)
    _foreign_compaction(db)
    session = _hydrated_record(db, stamped=False)
    session["history"][1]["local_marker"] = True  # identity survives only if the prefix is kept
    _bind_db(monkeypatch, db)
    db.append_message("s1", "user", "PEAR")
    db.append_message("s1", "assistant", "OK")

    server._adopt_out_of_band_turns(session)

    assert [m["content"] for m in session["history"]] == ["summary of ask0/ask1", "KIWI", "MANGO", "PEAR", "OK"]
    assert session["history"][1].get("local_marker") and session["history_version"] == 1


def test_unstamped_record_that_diverged_from_the_store_is_left_alone(tmp_path, monkeypatch):
    """A local rewrite the store has not caught up with must never be clobbered by the store tail."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    _seed_exchange(db, 0)
    session = _hydrated_record(db, stamped=False)
    _bind_db(monkeypatch, db)
    _gateway_appends(db)
    session["history"][0]["content"] = "locally edited ask"  # head no longer matches the store

    server._adopt_out_of_band_turns(session)

    assert session["history"][0]["content"] == "locally edited ask"
    assert len(session["history"]) == 4 and session["history_version"] == 0


def test_unstamped_record_is_untouched_when_the_store_is_not_ahead(tmp_path, monkeypatch):
    """A not-yet-flushed local tail (store shorter) is the fresher record — adopt nothing."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    _seed_exchange(db, 0)
    session = _hydrated_record(db, stamped=False)
    _bind_db(monkeypatch, db)
    session["history"] = list(session["history"]) + [{"role": "user", "content": "typed but unflushed"}]

    server._adopt_out_of_band_turns(session)

    assert session["history"][-1]["content"] == "typed but unflushed"
    assert session["history_version"] == 0


def test_unstamped_history_equal_to_the_store_is_untouched(tmp_path, monkeypatch):
    """Heads match and lengths are equal: nothing appended out of band, no-op."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    _seed_exchange(db, 0)
    session = _hydrated_record(db, stamped=False)
    _bind_db(monkeypatch, db)
    before = list(session["history"])

    server._adopt_out_of_band_turns(session)

    assert session["history"] == before and session["history_version"] == 0
