"""prompt.submit writes the user's message at send time, and the turn that follows adopts that row instead of
writing a second one (#111868: a Desktop freeze during a slow first agent build left a session row with no message)."""

import threading
from types import SimpleNamespace

import pytest

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
    return resp["result"]["session_id"], resp["result"]["stored_session_id"]


def _flush_agent(db, key):
    """Agent shell owning the real flush (the crash persist at turn start runs this same code)."""
    agent = SimpleNamespace(
        _session_db=db, _session_db_created=True, _persist_disabled=False, session_id=key,
        _session_persist_lock=None, _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0, _persist_user_message_idx=None, _persist_user_message_override=None,
        _persist_user_message_timestamp=None, _pending_cli_user_message=None)
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    agent._flush_messages_to_session_db_unlocked = AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    return agent


def test_user_message_is_durable_at_submit_before_any_agent_turn(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "please refactor the login page")
        assert server._persist_session_row_for_submit("rid", session, "please refactor the login page", None) is None
        # The agent build has not even started: the transcript already resumes with the sent message.
        assert [(r["role"], r["content"]) for r in db.get_messages_as_conversation(key)] == [
            ("user", "please refactor the login page")]
        assert db.get_session(key)["message_count"] == 1
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_submit_ack_binds_the_written_row_even_if_worker_consumes_staging(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]

    class InlineThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *args: False)
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *args: session.pop("_submit_user_row", None))
    try:
        replies = []
        for _ in range(2):
            session["running"] = False
            replies.append(server.handle_request({"id": "p", "method": "prompt.submit", "params": {
                "session_id": sid, "text": "same prompt"}})["result"])
        rows = db.get_messages_as_conversation(key, include_row_ids=True)
        assert [reply.get("user_row_id") for reply in replies] == [row["_row_id"] for row in rows]
        assert replies[0]["user_row_id"] != replies[1]["user_row_id"]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_retried_idle_client_message_id_returns_original_ack_from_live_and_durable_state(
    monkeypatch, tmp_path
):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]

    class DeferredThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Thread", DeferredThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *args: False)
    monkeypatch.setattr(server.time, "time", lambda: 1_790_594_550.0)
    request = {"id": "p", "method": "prompt.submit", "params": {
        "session_id": sid,
        "text": "idempotency probe",
        "submitted_at": 1_790_594_548.125,
        "client_message_id": "desktop-message-idempotent",
    }}
    try:
        first = server.handle_request(request)
        live_retry = server.handle_request({**request, "id": "live-retry"})

        assert live_retry["result"] == first["result"]
        assert not session.get("queued_prompt")
        assert len(db.get_messages_as_conversation(key, include_row_ids=True)) == 1

        with session["history_lock"]:
            session["running"] = False
            server._clear_inflight_turn(session)
        durable_retry = server.handle_request({**request, "id": "durable-retry"})

        assert durable_retry["result"] == first["result"]
        assert len(db.get_messages_as_conversation(key, include_row_ids=True)) == 1
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_pending_live_ack_never_falls_through_to_partial_durable_reconstruction(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, _key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    client_message_id = "desktop-pending-exact-ack"
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(
                session, "pending exact ack", user_timestamp=1_790_594_548.125,
                client_message_id=client_message_id)
            session["inflight_turn"]["_submit_ack"] = {
                "status": "streaming",
                "client_message_id": client_message_id,
                "survivor_user_row_ids": [17],
            }
            session["inflight_turn"]["_submit_ack_ready"] = threading.Event()
        assert server._persist_session_row_for_submit(
            "rid", session, "pending exact ack", None,
            {"client_message_id": client_message_id}, 1_790_594_548.125,
        ) is None

        assert server._client_message_ack(session, client_message_id) is None
    finally:
        server._sessions.pop(sid, None)
        db.close()



def test_failed_submit_persistence_releases_client_ack_reservation(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, _key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    client_message_id = "client-failed-persist-retry"

    def disk_full(_session):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(server, "_ensure_session_db_row", disk_full)
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(
                session, "retry after disk full", client_message_id=client_message_id)
            ready = server._reserve_client_message_admission_locked(session, client_message_id)
            session["inflight_turn"]["_submit_ack_ready"] = ready

        response = server._persist_session_row_for_submit(
            "failed", session, "retry after disk full", None,
            {"client_message_id": client_message_id}, 1_790_594_548.125,
        )

        assert response["error"]["data"]["code"] == "disk_full"
        assert client_message_id not in session.get("_client_message_admissions", {})
        # A concurrent duplicate that was waiting on this admission learns the real cause.
        duplicate = server._await_client_message_ack("duplicate", session, client_message_id, ready)
        assert duplicate["id"] == "duplicate"
        assert duplicate["error"] == response["error"]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_exact_client_ack_with_survivor_rebind_fields_is_not_evicted():
    session = {"history_lock": threading.RLock()}
    client_message_id = "truncate-original"
    exact_ack = {
        "status": "streaming",
        "client_message_id": client_message_id,
        "user_row_id": 41,
        "user_timestamp": 1_790_594_548.125,
        "survivor_user_row_ids": [17, None, 29],
        "survivor_row_id_map": {"17": 117, "29": None},
    }
    with session["history_lock"]:
        server._remember_client_message_ack_locked(session, client_message_id, exact_ack)
        for n in range(128):
            server._remember_client_message_ack_locked(
                session,
                f"sent-{n}",
                {"status": "streaming", "client_message_id": f"sent-{n}", "user_row_id": n + 1},
            )
        replay, pending = server._live_client_message_admission_locked(session, client_message_id)

    assert pending is None
    assert replay == exact_ack


def test_concurrent_idle_retries_share_one_admission_and_original_ack(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]

    class DeferredThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            pass

    rendezvous = threading.Barrier(2)
    slot_checks = 0
    slot_checks_lock = threading.Lock()

    def synchronized_slot_check(*_args):
        nonlocal slot_checks
        with slot_checks_lock:
            slot_checks += 1
        rendezvous.wait(timeout=2)
        return None

    real_thread = threading.Thread
    monkeypatch.setattr(server.threading, "Thread", DeferredThread)
    monkeypatch.setattr(server, "_ensure_active_session_slot", synchronized_slot_check)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *args: False)
    monkeypatch.setattr(server.time, "time", lambda: 1_790_594_550.0)
    request = {"method": "prompt.submit", "params": {
        "session_id": sid,
        "text": "one concurrent send",
        "submitted_at": 1_790_594_548.125,
        "client_message_id": "desktop-concurrent-idempotent",
    }}
    try:
        replies = {}

        def submit(request_id):
            replies[request_id] = getattr(server, "handle_request")({**request, "id": request_id})

        callers = [real_thread(target=submit, args=(request_id,)) for request_id in ("first", "retry")]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(timeout=5)
        assert not any(caller.is_alive() for caller in callers)

        first, retry = replies["first"], replies["retry"]
        assert slot_checks == 2  # both requests passed the old pre-admission dedupe together
        assert first["result"] == retry["result"]
        assert first["result"]["status"] == "streaming"
        assert isinstance(first["result"].get("user_row_id"), int)
        assert session["inflight_turn"]["client_message_id"] == "desktop-concurrent-idempotent"
        assert not session.get("queued_prompt")
        assert not session.get("queued_prompts")
        assert len(db.get_messages_as_conversation(key, include_row_ids=True)) == 1
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_submit_uses_valid_client_timestamp_and_persists_client_identity(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]

    class InlineThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *args: False)
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *args: session.pop("_submit_user_row", None))
    monkeypatch.setattr(server.time, "time", lambda: 1_790_594_550.0)
    try:
        response = server.handle_request({"id": "p", "method": "prompt.submit", "params": {
            "session_id": sid,
            "text": "timestamp identity probe",
            "submitted_at": 1_790_594_548.125,
            "client_message_id": "desktop-message-abc",
        }})

        assert response["result"]["user_timestamp"] == 1_790_594_548.125
        assert response["result"]["client_message_id"] == "desktop-message-abc"
        rows = db.get_messages_as_conversation(key, include_row_ids=True)
        assert rows[0]["timestamp"] == 1_790_594_548.125
        assert rows[0]["display_metadata"]["client_message_id"] == "desktop-message-abc"
    finally:
        server._sessions.pop(sid, None)
        db.close()


@pytest.mark.parametrize(
    "client_message_id",
    ["a" * 129, "contains space", "line\nbreak", "ümlaut"],
)
def test_invalid_client_message_id_is_ignored_without_rejecting_the_prompt(client_message_id):
    received_at = 1_790_000_000.0

    assert server._prompt_send_envelope(
        {"submitted_at": received_at, "client_message_id": client_message_id}, received_at
    ) == (received_at, None)


@pytest.mark.parametrize(
    "submitted_at",
    [None, True, float("nan"), float("inf"), 946_684_799.999, 1_790_000_300.001, "1790000000"],
)
def test_invalid_or_absent_client_timestamp_falls_back_to_first_receipt(submitted_at):
    received_at = 1_790_000_000.0

    assert server._prompt_send_envelope(
        {"submitted_at": submitted_at, "client_message_id": "client-a"}, received_at
    ) == (received_at, "client-a")


def test_turn_adopts_the_submit_row_and_writes_no_duplicate(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "look at @notes.md")
        assert server._persist_session_row_for_submit("rid", session, "look at @notes.md", None) is None
        agent = _flush_agent(db, key)
        # The prologue rewrote the persisted prompt (@-expansion): the early row follows it.
        expanded = "look at @notes.md\n\n<file notes.md>todo</file>"
        server._adopt_submit_user_row(session, agent, expanded, "look at @notes.md")
        assert "_submit_user_row" not in session
        user_msg, _pending = _stage_turn_user_message(agent, expanded, expanded, None, None, None, None)
        assert user_msg is agent._pending_cli_user_message  # adopted by identity, not rebuilt
        messages = [user_msg]
        agent._persist_user_message_idx = 0
        agent._flush_messages_to_session_db(messages, [])  # the turn-start crash persist
        agent._flush_messages_to_session_db(messages + [{"role": "assistant", "content": "done"}], [])  # turn end
        rows = db.get_messages_as_conversation(key, include_inactive=True)
        assert [(r["role"], r["content"]) for r in rows] == [("user", expanded), ("assistant", "done")]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_completion_receipt_covers_only_committed_current_turn_rows(monkeypatch, tmp_path):
    from tui_gateway.prompt_turn import _TurnRun

    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    try:
        server._ensure_session_db_row(session)
        agent = _flush_agent(db, key)
        old = [{"role": "user", "content": "again"}, {"role": "assistant", "content": "same"}]
        agent._flush_messages_to_session_db(old, [])
        current = [{"role": "user", "content": "again"}, {"role": "assistant", "content": "same"}]
        messages = old + current
        agent._persist_user_message_idx = len(old)
        agent.context_compressor = SimpleNamespace(compression_count=0)
        st = _TurnRun(agent, None, None, False, history=old, compression_count=0,
                      result={"messages": messages, "final_response": "same"})
        agent._flush_messages_to_session_db(messages, old)
        payload, _, _ = server._complete_turn_payload(session, st, None, 80)
        rows = db.get_messages_as_conversation(key, include_row_ids=True)
        assert payload.get("persisted_turn") == {
            "user_row_id": rows[-2]["_row_id"],
            "row_ids": [row["_row_id"] for row in rows[-2:]],
            "final_assistant_row_id": rows[-1]["_row_id"],
            "complete": True,
        }
        # Compression can discard an already-streamed segment even while the old prefix survives.
        agent.context_compressor = SimpleNamespace(compression_count=1)
        compressed, _, _ = server._complete_turn_payload(session, st, None, 80)
        assert compressed["persisted_turn"]["complete"] is False
        # The row address alone does not assert that a subsequently mutated body was committed.
        current[-1].pop("_db_persisted")
        current[-1]["content"] = "not flushed"
        partial, _, _ = server._complete_turn_payload(session, st, None, 80)
        assert partial["persisted_turn"] == {
            "user_row_id": rows[-2]["_row_id"], "row_ids": [rows[-2]["_row_id"]], "complete": False}
        # No authoritative current-turn anchor: never infer from matching text or positions in old history.
        agent._persist_user_message_idx = None
        missing, _, _ = server._complete_turn_payload(session, st, None, 80)
        assert "persisted_turn" not in missing
        # A preflight failure may return only old history while the agent still carries its old cursor.
        agent._persist_user_message_idx = 0
        st.result["messages"] = old
        stale, _, _ = server._complete_turn_payload(session, st, None, 80)
        assert "persisted_turn" not in stale
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_failed_build_drops_the_staged_row_and_a_later_turn_never_adopts_it(monkeypatch, tmp_path):
    """The submit-time row is the durable record of THAT send only. A turn that ends before the agent runs
    (build failed / bounded wait expired) must drop the staging dict, and a later turn without a matching
    prompt.submit (wake-up, auto-continue, queued drain) must not rewrite the user's row to its own text."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt",
                        lambda _session, _rid, _sid: {"error": {"message": "agent initialization failed"}})
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "please refactor the login page")
        assert server._persist_session_row_for_submit("rid", session, "please refactor the login page", None) is None
        server._run_after_agent_ready("rid", sid, session, "please refactor the login page", None, None, None)
        assert "_submit_user_row" not in session, "staged row survived a turn that never reached the agent"

        # Even if a staged row were still around, a turn whose raw submit differs must leave the DB alone.
        with session["history_lock"]:
            server._start_inflight_turn(session, "please refactor the login page")
        server._persist_submit_user_row(session, "please refactor the login page", None)
        agent = _flush_agent(db, key)
        synthesized = "[subagent finished] result summary"
        server._adopt_submit_user_row(session, agent, synthesized, synthesized)
        assert "_submit_user_row" not in session
        assert agent._pending_cli_user_message is None
        rows = db.get_messages_as_conversation(key, include_inactive=True)
        assert [(r["role"], r["content"]) for r in rows] == [
            ("user", "please refactor the login page"), ("user", "please refactor the login page")]
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_the_staged_submit_row_carries_the_uid_its_db_row_was_written_with(monkeypatch, tmp_path):
    """The turn adopts the staged dict as its user message; if it lacked the row's uid, the next host copy
    (in-place compaction) would mint a second identity for the same message."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid, key = _desktop_session(monkeypatch, db)
    session = server._sessions[sid]
    try:
        with session["history_lock"]:
            session["running"] = True
            server._start_inflight_turn(session, "please refactor the login page")
        assert server._persist_session_row_for_submit("rid", session, "please refactor the login page", None) is None
        staged = session["_submit_user_row"]
        stored = db._conn.execute("SELECT message_uid FROM messages WHERE id = ?", (staged["_row_id"],)).fetchone()
        assert staged.get("message_uid") == stored[0]
    finally:
        server._sessions.pop(sid, None)
        db.close()
