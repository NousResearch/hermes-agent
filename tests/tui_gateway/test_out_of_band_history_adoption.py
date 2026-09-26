"""Regression for #42962: a turn another surface (Telegram, cron) appended to a live desktop/TUI session must
reach the model on the next local prompt, not only the repainted transcript."""

import contextlib
import threading

from agent.session_persistence import _durable_content
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


def test_next_turn_adopts_tool_result_across_live_history_boundary(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    db.append_message("s1", "user", "Read the status file.")
    db.append_message(
        "s1",
        "assistant",
        "",
        tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }],
    )
    session = {
        "session_key": "s1",
        "history": db.get_messages_as_conversation("s1", include_row_ids=True),
        "history_lock": threading.Lock(),
        "history_version": 0,
    }
    _bind_db(monkeypatch, db)
    db.append_message(
        "s1", "tool", "healthy", tool_name="read_file", tool_call_id="call-1")
    db.append_message("s1", "assistant", "The service is healthy.")
    own = db.append_message("s1", "user", "What changed?")
    session["_submit_user_row"] = {
        "role": "user", "content": "What changed?", "_row_id": own}

    server._adopt_out_of_band_turns(session)

    assert [message["role"] for message in session["history"]] == [
        "user", "assistant", "tool", "assistant"]
    assert session["history"][1]["tool_calls"][0]["id"] == "call-1"
    assert session["history"][2]["tool_call_id"] == "call-1"
    assert session["history"][3]["content"] == "The service is healthy."
    assert session["history_version"] == 1


def test_next_turn_cleans_interrupted_tool_block_across_live_history_boundary(
        tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    db.append_message("s1", "user", "Read the status file.")
    db.append_message(
        "s1",
        "assistant",
        "",
        tool_calls=[{
            "id": "call-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }],
    )
    session = {
        "session_key": "s1",
        "history": db.get_messages_as_conversation("s1", include_row_ids=True),
        "history_lock": threading.Lock(),
        "history_version": 0,
    }
    _bind_db(monkeypatch, db)
    db.append_message(
        "s1",
        "tool",
        '{"output": "[Command interrupted]", "exit_code": 130}',
        tool_name="read_file",
        tool_call_id="call-1",
    )
    own = db.append_message("s1", "user", "Try again.")
    session["_submit_user_row"] = {
        "role": "user", "content": "Try again.", "_row_id": own}

    server._adopt_out_of_band_turns(session)

    assert [message["role"] for message in session["history"]] == ["user"]
    assert session["history_version"] == 1


def test_next_turn_adopts_rows_when_live_history_lacks_row_ids(tmp_path, monkeypatch):
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    # A completed gateway turn can leave provider-format history without durable
    # row markers. Recover the boundary from the matching durable prefix.
    session["history"] = [
        {"role": message["role"], "content": message["content"]}
        for message in session["history"]
    ]
    db.append_message("s1", "user", "My second codeword is KIWI.", timestamp=3.0)
    db.append_message("s1", "assistant", "OK", timestamp=4.0)
    own = db.append_message("s1", "user", "List every codeword.", timestamp=5.0)
    session["_submit_user_row"] = {
        "role": "user", "content": "List every codeword.", "_row_id": own}

    server._adopt_out_of_band_turns(session)

    assert [message["content"] for message in session["history"]] == [
        "My codeword is MANGO.", "OK", "My second codeword is KIWI.", "OK"]
    assert [message.get("_row_id") for message in session["history"]] == [1, 2, 3, 4]
    assert session["history_version"] == 1


def test_marker_free_history_must_match_the_durable_prefix(tmp_path, monkeypatch):
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    session["history"] = [
        {"role": "user", "content": "My codeword is MANGO."},
        {"role": "assistant", "content": "Different reply"},
    ]
    db.append_message("s1", "user", "My second codeword is KIWI.")
    db.append_message("s1", "assistant", "OK")
    own = db.append_message("s1", "user", "List every codeword.")
    session["_submit_user_row"] = {
        "role": "user", "content": "List every codeword.", "_row_id": own}
    before = list(session["history"])

    server._adopt_out_of_band_turns(session)

    assert session["history"] == before
    assert all("_row_id" not in message for message in session["history"])
    assert session["history_version"] == 0


def test_marker_free_structured_user_content_must_match_durable_prefix(
    tmp_path, monkeypatch
):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    durable_native_content = [
        {"type": "text", "text": "Describe this image.\n@image:/tmp/image-a.png"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    db.append_message("s1", "user", _durable_content(durable_native_content))
    db.append_message("s1", "assistant", "It is image A.")
    session = {
        "session_key": "s1",
        "history": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image.\n@image:/tmp/image-b.png"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
                ],
            },
            {"role": "assistant", "content": "It is image A."},
        ],
        "history_lock": threading.Lock(),
        "history_version": 0,
    }
    _bind_db(monkeypatch, db)
    db.append_message("s1", "user", "Continue.")
    db.append_message("s1", "assistant", "Done.")
    own = db.append_message("s1", "user", "What changed?")
    session["_submit_user_row"] = {
        "role": "user",
        "content": "What changed?",
        "_row_id": own,
    }
    before = list(session["history"])

    server._adopt_out_of_band_turns(session)

    assert session["history"] == before
    assert all("_row_id" not in message for message in session["history"])
    assert session["history_version"] == 0


def test_marker_free_native_image_history_does_not_adopt_without_exact_identity(
    tmp_path, monkeypatch
):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    native_content = [
        {"type": "text", "text": "Describe this image.\n@image:/tmp/image-a.png"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    db.append_message("s1", "user", _durable_content(native_content))
    db.append_message("s1", "assistant", "It is image A.")
    session = {
        "session_key": "s1",
        "history": [
            {"role": "user", "content": native_content},
            {"role": "assistant", "content": "It is image A."},
        ],
        "history_lock": threading.Lock(),
        "history_version": 0,
    }
    _bind_db(monkeypatch, db)
    db.append_message("s1", "user", "Continue.")
    db.append_message("s1", "assistant", "Done.")
    own = db.append_message("s1", "user", "What changed?")
    session["_submit_user_row"] = {
        "role": "user",
        "content": "What changed?",
        "_row_id": own,
    }
    before = list(session["history"])

    server._adopt_out_of_band_turns(session)

    assert session["history"] == before
    assert all("_row_id" not in message for message in session["history"])
    assert session["history_version"] == 0


def test_next_turn_reloads_history_rewritten_without_summary(tmp_path, monkeypatch):
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    rewritten = [
        {"role": "user", "content": "My replacement codeword is PEAR."},
        {"role": "assistant", "content": "Recorded"},
    ]
    db.replace_messages("s1", rewritten, active_only=True, archive_dropped=True)
    own = db.append_message("s1", "user", "List every codeword.")
    session["_submit_user_row"] = {
        "role": "user", "content": "List every codeword.", "_row_id": own}

    server._adopt_out_of_band_turns(session)

    assert [message["content"] for message in session["history"]] == [
        "My replacement codeword is PEAR.", "Recorded"]
    assert session["history_version"] == 1


def test_rewrite_excludes_submit_row_before_repairing_alternation(tmp_path, monkeypatch):
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    db.replace_messages(
        "s1",
        [{"role": "user", "content": "External pending request."}],
        active_only=True,
        archive_dropped=True,
    )
    own = db.append_message("s1", "user", "My current prompt.")
    session["_submit_user_row"] = {
        "role": "user", "content": "My current prompt.", "_row_id": own}

    server._adopt_out_of_band_turns(session)

    assert [message["content"] for message in session["history"]] == [
        "External pending request."
    ]
    assert all(
        "My current prompt." not in str(message.get("content", ""))
        for message in session["history"]
    )
    assert session["history_version"] == 1


def test_marker_free_native_image_requires_exact_durable_identity(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="desktop")
    durable_native_content = [
        {"type": "text", "text": "Describe this image.\n@image:/tmp/image.png"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    db.append_message("s1", "user", _durable_content(durable_native_content))
    db.append_message("s1", "assistant", "It is image A.")
    session = {
        "session_key": "s1",
        "history": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image.\n@image:/tmp/image.png"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
                ],
            },
            {"role": "assistant", "content": "It is image A."},
        ],
        "history_lock": threading.Lock(),
        "history_version": 0,
    }
    _bind_db(monkeypatch, db)
    db.append_message("s1", "user", "Continue.")
    db.append_message("s1", "assistant", "Done.")
    own = db.append_message("s1", "user", "What changed?")
    session["_submit_user_row"] = {
        "role": "user", "content": "What changed?", "_row_id": own}
    before = list(session["history"])

    server._adopt_out_of_band_turns(session)

    assert session["history"] == before
    assert all("_row_id" not in message for message in session["history"])
    assert session["history_version"] == 0


def test_next_turn_reloads_history_rewritten_to_empty(tmp_path, monkeypatch):
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    db.replace_messages("s1", [], active_only=True, archive_dropped=True)
    own = db.append_message("s1", "user", "Start again.")
    session["_submit_user_row"] = {
        "role": "user", "content": "Start again.", "_row_id": own}

    server._adopt_out_of_band_turns(session)

    assert session["history"] == []
    assert session["history_version"] == 1


def test_rewrite_reload_failure_keeps_live_history(tmp_path, monkeypatch):
    db, session = _seed(tmp_path)
    _bind_db(monkeypatch, db)
    db.replace_messages("s1", [], active_only=True, archive_dropped=True)
    own = db.append_message("s1", "user", "Start again.")
    session["_submit_user_row"] = {
        "role": "user", "content": "Start again.", "_row_id": own}
    before = list(session["history"])
    monkeypatch.setattr(server, "_load_durable_truncation_history", lambda *args, **kwargs: None)

    server._adopt_out_of_band_turns(session)

    assert session["history"] == before
    assert session["history_version"] == 0


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
