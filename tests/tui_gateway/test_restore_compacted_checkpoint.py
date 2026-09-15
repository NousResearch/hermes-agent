"""An explicit Desktop rewind may expand the archived prefix, never the abandoned suffix."""
from copy import deepcopy
import threading
from types import SimpleNamespace

from agent.context_compressor import SUMMARY_PREFIX, _SUMMARY_END_MARKER

import pytest

from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def compacted_session(tmp_path, monkeypatch, request):
    db = SessionDB(tmp_path / "profile" / "state.db")
    sid = "bot-conversation"
    db.create_session(session_id=sid, source="desktop")
    original = [
        {"role": "user", "content": "first", "timestamp": 1, "api_content": "first\n[original wire context]"},
        {"role": "assistant", "content": "first reply", "timestamp": 2,
         "reasoning_content": "private reasoning", "reasoning_details": [{"type": "text", "text": "detail"}],
         "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": "file bytes", "tool_call_id": "call-1", "tool_name": "read_file", "timestamp": 2.1},
        {"role": "assistant", "content": "finished reading", "timestamp": 2.2},
        {"role": "user", "content": "checkpoint", "timestamp": 3},
        {"role": "assistant", "content": "discarded reply", "timestamp": 4},
        {"role": "user", "content": "later", "timestamp": 5},
        {"role": "assistant", "content": "later reply", "timestamp": 6},
    ]
    kind = getattr(request, "param", "plain")
    prior = {"role": "user", "content": f"{SUMMARY_PREFIX}\ncontext from before this segment\n{_SUMMARY_END_MARKER}",
             "_compressed_summary": True, "timestamp": 0}
    if kind == "standalone":
        original.insert(0, prior)
    elif kind == "carrier":
        original[0]["content"] = prior["content"] + "\n\nfirst"
        original[0]["_compressed_summary"] = True
    target_index = len(original) - 4
    if kind == "target-carrier":
        original[target_index]["content"] = prior["content"] + "\n\ncheckpoint"
        original[target_index]["_compressed_summary"] = True
    db.append_messages_batch(sid, original)
    target = db.get_messages(sid)[target_index]["id"]
    summary = {"role": "user", "content": "summary of future discarded reply", "_compressed_summary": True}
    # Protected head is copied to a newer physical row; its logical position must survive.
    db.archive_and_compact(sid, [*deepcopy(original[:target_index]), summary, *deepcopy(original[-2:])], tail_count=2)
    session = {"session_key": sid, "profile_home": str(db.db_path.parent),
               "history": db.get_messages_as_conversation(sid, include_row_ids=True, repair_alternation=True)}

    # Exercise the real profile-aware owner lookup; touching the default store is a bug.
    monkeypatch.setattr(server, "_get_db", lambda: pytest.fail("must use the session's profile DB"))
    yield db, sid, session, target, original
    db.close()


@pytest.mark.parametrize("compacted_session", ["plain", "standalone", "carrier", "target-carrier"], indirect=True)
@pytest.mark.parametrize("generations", [1, 2])
def test_restore_compacted_prefix_survives_resume_and_second_rewind(compacted_session, generations, monkeypatch):
    db, sid, session, target, original = compacted_session
    if generations == 2:
        db.archive_and_compact(sid, deepcopy(session["history"]))
        session["history"] = db.get_messages_as_conversation(sid, include_row_ids=True, repair_alternation=True)
        session["display_history_prefix"] = [{"role": "user", "content": "ancestor display only"}]
    displayed = db.get_messages_as_conversation(sid, include_compacted=True, include_row_ids=True)
    requested_ids = {m["_row_id"] for m in displayed}
    session.update(agent=SimpleNamespace(), history_lock=threading.Lock(), running=False,
                   attached_images=[], cols=80)
    delivered = threading.Event()
    received = []

    def turn_sink(_rid, _sid, candidate, text, *_args):
        received.append((text, deepcopy(candidate["history"])))
        delivered.set()

    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_run_after_agent_ready", turn_sink)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    response = server.handle_request({"id": "request", "method": "prompt.submit", "params": {
        "session_id": sid, "text": "checkpoint", "confirm_truncate": True,
        "truncate_before_row_id": target, "truncate_before_user_ordinal": generations,
        "rebind_survivor_row_ids": list(requested_ids),
    }})
    assert "error" not in response
    assert delivered.wait(5)
    session["_run_thread"].join(timeout=5)
    fields = response["result"]
    assert received[0][0] == "checkpoint"
    from agent.context_compressor import history_before_user_originated_turn
    prefix, _ = history_before_user_originated_turn(original, len(original) - 4)
    expected = [m["content"] for m in prefix]
    assert [m["content"] for m in received[0][1]] == expected
    assert [m["content"] for m in session["history"]] == expected
    model, display = db.get_resume_conversations(sid)
    assert [m["content"] for m in model] == expected
    wire = next(m for m in model if m.get("api_content"))
    assert wire["api_content"] == next(m["api_content"] for m in original if m.get("api_content"))
    call = next(m for m in model if m.get("tool_calls"))
    original_call = next(m for m in original if m.get("tool_calls"))
    assert call["reasoning_content"] == original_call["reasoning_content"]
    assert call["reasoning_details"] == original_call["reasoning_details"]
    assert [m["content"] for m in display] == expected
    assert call["tool_calls"] == original_call["tool_calls"]
    assert next(m for m in model if m.get("tool_call_id"))["tool_call_id"] == call["tool_calls"][0]["id"]
    assert db.get_session(sid)["tool_call_count"] == len(call["tool_calls"])
    assert db.get_session(sid)["message_count"] == len(model)
    assert db.get_session(sid)["rewind_count"] == 1
    assert fields["survivor_row_id_map"][str(target)] is None
    # No hard deletes, and discarded compaction generations cannot reappear after reload.
    assert any(m["id"] == target for m in db.get_messages(sid, include_inactive=True))
    first_display_id = next(m["_row_id"] for m in displayed if m.get("api_content"))
    first_live_id = next(m["_row_id"] for m in model if m.get("api_content"))
    assert fields["survivor_row_id_map"][str(first_display_id)] == first_live_id
    error, _ = server._truncate_history_for_submit("second", sid, session, {
        "confirm_truncate": True, "confirm_empty_truncate": True,
        "truncate_before_row_id": first_live_id,
    }, {first_live_id})
    assert error is None
    # An earlier handoff/scaffold remains for carrier sessions; no discarded answers return.
    assert not any(m.get("content") in {"discarded reply", "later", "later reply"}
                   for m in db.get_messages(sid, include_compacted=True))


@pytest.mark.parametrize("rejection", ["unconfirmed", "ordinal", "missing", "abandoned", "foreign", "busy", "compression", "summary", "duplicate", "empty", "write_failure"])
def test_rejected_archived_restore_leaves_database_and_memory_unchanged(compacted_session, monkeypatch, rejection):
    db, sid, session, target, _ = compacted_session
    params = {"confirm_truncate": True, "truncate_before_row_id": target, "truncate_before_user_ordinal": 1}
    if rejection == "unconfirmed":
        params["confirm_truncate"] = False
    elif rejection == "ordinal":
        params["truncate_before_user_ordinal"] = 9
    elif rejection == "missing":
        params["truncate_before_row_id"] = 999999
    elif rejection == "abandoned":
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET compacted = 0 WHERE id = ?", (target,)))
    elif rejection == "foreign":
        db.create_session(session_id="foreign", source="desktop")
        params["truncate_before_row_id"] = db.append_message("foreign", "user", "checkpoint")
    elif rejection == "summary":
        summary_row = next(m for m in db.get_messages(sid) if m.get("_compressed_summary"))
        db.archive_and_compact(sid, [{"role": "user", "content": "only summary", "_compressed_summary": True}])
        params["truncate_before_row_id"] = summary_row["id"]
        session["history"] = db.get_messages_as_conversation(sid, include_row_ids=True)
    elif rejection == "duplicate":
        params["truncate_before_row_id"] = db.get_messages(sid, include_inactive=True)[0]["id"]
    elif rejection == "compression":
        db._execute_write(lambda conn: conn.execute(
            "INSERT INTO compression_locks VALUES (?, ?, ?, ?)", (sid, "other-worker", 0, 9999999999)))
    elif rejection == "busy":
        db._execute_write(lambda conn: conn.execute(
            "INSERT INTO session_turn_leases VALUES (?, ?, ?, ?)", (sid, "other-worker", 0, 9999999999)))
    elif rejection == "empty":
        first = db.get_messages(sid, include_compacted=True)[0]["id"]
        db.archive_and_compact(sid, [{"role": "user", "content": "only summary", "_compressed_summary": True}])
        # Choose the latest display representative of the first turn, now archived.
        params.update(truncate_before_row_id=first, truncate_before_user_ordinal=0)
        session["history"] = db.get_messages_as_conversation(sid, include_row_ids=True)
    elif rejection == "write_failure":
        db._execute_write(lambda conn: conn.execute(
            "CREATE TRIGGER reject_restore BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT, 'disk failure'); END"))
    before_counts = db.get_session(sid)
    before_db = db.get_messages(sid, include_inactive=True)
    before_session = deepcopy(session)
    error, _ = server._truncate_history_for_submit("request", sid, session, params, {target})
    assert error is not None
    assert db.get_session(sid) == before_counts
    assert db.get_messages(sid, include_inactive=True) == before_db
    assert session == before_session
