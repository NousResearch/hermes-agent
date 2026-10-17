"""One-shot continuation admission, stale snapshots and unknown outcomes (#137277)."""
import threading
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("retry-db", source="desktop")
    db.append_messages_batch("retry-db", [
        {"role": "user", "content": "count once"},
        {"role": "assistant", "content": "working", "tool_calls": [
            {"id": "call1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "content": "count=1", "tool_call_id": "call1", "tool_name": "terminal",
         "display_metadata": {"tool_result_received": True}},
    ])
    history = db.get_messages_as_conversation("retry-db", include_row_ids=True)
    agent = SimpleNamespace(session_id="retry-db", api_mode="chat_completions")
    session = {"agent": agent, "session_key": "retry-db", "history": history,
               "running": False, "history_version": 0, "history_lock": threading.Lock()}
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *args: False)
    server._start_inflight_turn(session, "count once")
    server._fail_inflight_turn(session, "HTTP 429")
    state = SimpleNamespace(agent=agent, result={"messages": history})
    token = server._record_retry_checkpoint(session, state, {"layer": "provider", "code": "rate_limit"})
    assert token
    yield db, session, state, token
    db.close()


@pytest.mark.parametrize("change", ["new-user", "compaction", "tool-outcome", "missing-token"])
def test_changed_or_missing_checkpoint_never_rewinds_or_dispatches(checkpoint, change):
    db, session, _, token = checkpoint
    if change == "new-user":
        db.append_message("retry-db", "user", "new intent")
    elif change == "compaction":
        db.replace_messages("retry-db", [{"role": "user", "content": "compressed summary"}])
    elif change == "tool-outcome":
        rows = db.get_messages_as_conversation("retry-db", include_row_ids=True)
        rows[-1]["effect_disposition"] = "unknown"
        db.replace_messages("retry-db", rows)
    else:
        session["inflight_turn"] = None  # fresh backend/cold rebind cannot prove the old failure
    before = db.get_messages("retry-db")
    error, text = server._claim_prompt_retry("retry", session, token)
    assert error["error"]["code"] == 4034
    assert text is None and session["running"] is False
    assert db.get_messages("retry-db") == before


def test_live_rebind_keeps_token_and_duplicate_click_never_queues(checkpoint):
    db, session, _, token = checkpoint
    assert server._inflight_snapshot(session)["retry_token"] == token
    before = db.get_messages("retry-db")
    error, text = server._claim_prompt_retry("new-runtime", session, token)
    assert error is None and text == "count once"
    assert session["_retry_user_row_id"] == before[0]["id"]
    error, _ = server._claim_prompt_retry("duplicate", session, token)
    assert error["error"]["code"] == 4009
    assert not session.get("queued_prompt")
    assert db.get_messages("retry-db") == before


@pytest.mark.parametrize("unsafe", ["unknown", "missing", "unproven", "duplicate"])
def test_incomplete_or_unproven_tool_results_offer_no_continuation(checkpoint, unsafe):
    _, _session, state, _ = checkpoint
    rows = [dict(row) for row in state.result["messages"]]
    if unsafe == "unknown":
        rows[-1]["effect_disposition"] = "unknown"
    elif unsafe == "missing":
        rows.pop()
    elif unsafe == "unproven":
        rows[-1]["display_metadata"] = {}
    else:
        rows.append(dict(rows[-1]))
    assert server._retry_tools_complete(rows[1:]) is False


@pytest.mark.parametrize("terminal", ["failure", "cancel", "new-user"])
def test_aborted_retry_never_leaks_anchor_into_another_turn(checkpoint, terminal):
    _, session, _, token = checkpoint
    assert server._claim_prompt_retry("retry", session, token)[0] is None
    if terminal == "failure":
        server._fail_inflight_turn(session, "agent build failed")
    elif terminal == "cancel":
        server._clear_inflight_turn(session)
    else:
        server._start_inflight_turn(session, "new user")
    assert "_retry_user_row_id" not in session


def test_compute_host_isolation_is_never_bypassed(checkpoint, monkeypatch):
    _, session, state, token = checkpoint
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *args: True)
    assert server._record_retry_checkpoint(session, state, {"layer": "provider", "code": "rate_limit"}) is None
    server._sessions["isolated-retry"] = session
    try:
        response = server.handle_request({"id": "retry", "method": "prompt.retry", "params": {
            "session_id": "isolated-retry", "retry_token": token}})
        assert response["error"]["code"] == 4034
        assert session["running"] is False
    finally:
        server._sessions.pop("isolated-retry", None)


@pytest.mark.parametrize("field", ["content", "api_content"])
def test_earlier_replayed_image_context_has_no_continuation(checkpoint, field):
    db, session, state, _ = checkpoint
    rows = [{"role": "user", "content": "Earlier image", field: "[Image attached at: missing.png]\n[screenshot]"},
            {"role": "assistant", "content": "Earlier image context"}, *state.result["messages"]]
    db.replace_messages("retry-db", rows)
    history = db.get_messages_as_conversation("retry-db", include_row_ids=True, repair_alternation=False)
    state.result["messages"] = history
    before = db.get_messages("retry-db")
    assert server._record_retry_checkpoint(session, state, {"layer": "provider", "code": "rate_limit"}) is None
    assert db.get_messages("retry-db") == before
    assert not session["running"]
