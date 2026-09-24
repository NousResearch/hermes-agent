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


def test_model_switch_marker_is_not_adopted_as_a_foreign_row(tmp_path, monkeypatch):
    """The gateway persists the switch marker itself; unstamped, its row sat above the history's highest id
    and the next turn re-appended it as a foreign turn, so the model saw the marker twice."""
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: None)  # the row already exists

    server._append_model_switch_marker(session, model="model-b", provider="p")
    own = db.append_message("s1", "user", "Which model is this?", timestamp=5.0)
    session["_submit_user_row"] = {"role": "user", "content": "Which model is this?", "_row_id": own}
    server._adopt_out_of_band_turns(session)

    markers = [m for m in session["history"] if server._is_model_switch_marker(m)]
    assert len(markers) == 1 and "model-b" in markers[0]["content"]
    assert [m["content"] for m in session["history"][:2]] == ["My codeword is MANGO.", "OK"]
