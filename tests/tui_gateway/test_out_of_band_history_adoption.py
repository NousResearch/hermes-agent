"""Regression for #42962: a turn another surface (Telegram, cron) appended to a live desktop/TUI session must
reach the model on the next local prompt, not only the repainted transcript."""

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


def test_compaction_resequenced_rows_are_not_adopted_again(tmp_path, monkeypatch):
    """archive_and_compact re-inserts the carried messages as fresh rows while the in-memory dicts keep the
    archived originals' ids (only the persisted marker is restamped); an unflushed local row carries no id.
    Neither may be read back into history as if another surface had written it."""
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    compacted = [{"role": "assistant", "content": "summary of MANGO", "_compressed_summary": True},
                 dict(session["history"][-1])]  # carried tail keeps _row_id 2
    db.archive_and_compact("s1", compacted, tail_count=1)
    session["history"] = compacted + [{"role": "user", "content": "unflushed local turn"}]

    server._adopt_out_of_band_turns(session)

    assert [m["content"] for m in session["history"]] == ["summary of MANGO", "OK", "unflushed local turn"]
    assert session["history_version"] == 0
