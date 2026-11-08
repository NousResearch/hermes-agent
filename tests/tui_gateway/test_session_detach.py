"""Tab retirement detaches one viewer without abandoning work or ending stored chats."""

import threading
from types import SimpleNamespace

import pytest

from hermes_cli.active_sessions import active_session_registry_snapshot
from hermes_state import SessionDB
from tui_gateway import server


class _Peer:
    _closed = False

    def write(self, message):
        return True


class _Timer:
    def __init__(self, delay, callback):
        self.callback = callback

    def start(self):
        pass

    def cancel(self):
        pass


def _detach(peer, sid="retired", **params):
    return server.dispatch({"id": "detach", "method": "session.detach",
                            "params": {"session_id": sid, **params}}, peer)


@pytest.fixture
def detached_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_concurrent_sessions": 2})
    monkeypatch.setattr(server, "_broadcast_global_event", lambda *args: None)
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    sessions = []

    def make(sid, peer):
        db.create_session(sid, source="desktop")
        closed = []
        ready = threading.Event()
        ready.set()
        agent = SimpleNamespace(session_id=sid, close=lambda: closed.append(True))
        session = {"agent": agent, "agent_ready": ready, "session_key": sid,
                   "source": "desktop", "history": [], "history_lock": threading.RLock(),
                   "running": False, "transport": peer, "viewers": {peer: 1}}
        server._sessions[sid] = session
        assert server._ensure_active_session_slot(sid, session) is None
        sessions.append(session)
        return session, closed

    yield make, db
    for session in sessions:
        server._release_active_session_slot(session)
    db.close()


def test_detach_only_removes_callers_viewer_and_keeps_shared_socket(detached_env):
    make, _db = detached_env
    owner, peer, stranger = _Peer(), _Peer(), _Peer()
    session, closed = make("retired", owner)
    other, _ = make("other", owner)
    server._rebind_live_transport("retired", session, peer)

    assert _detach(stranger) == {"jsonrpc": "2.0", "id": "detach", "result": {"detached": False}}
    assert _detach(owner)["result"] == {"detached": True}
    assert not server._session_transport_contains(session, owner)
    assert server._session_transport_contains(session, peer)
    assert server._session_transport_contains(other, owner)
    assert _detach(owner)["result"] == {"detached": False}
    assert _detach(owner, sid="missing")["result"] == {"detached": False}
    assert server._pending_ws_reaps == {}
    assert not closed
    assert len(active_session_registry_snapshot()) == 2
    assert not owner._closed and not peer._closed


@pytest.mark.parametrize("unfinished", [None, "running", "queued", "retained", "isolated", "building", "loop",
                                        "loop-read-error", "heartbeat-read-error", "loop-corrupt", "heartbeat-corrupt"])
def test_last_viewer_detach_reclaims_only_settled_runtime(detached_env, unfinished, monkeypatch):
    make, db = detached_env
    owner = _Peer()
    session, closed = make("retired", owner)
    if unfinished == "running":
        session["running"] = True
    elif unfinished == "queued":
        session["queued_prompts"] = [{"text": "next"}]
    elif unfinished == "retained":
        session["inflight_turn"] = {"status": "error", "user": "retry me"}
    elif unfinished == "isolated":
        session["_compute_host_turn_id"] = "unfinished-child"
    elif unfinished == "building":
        session["agent_ready"].clear()
    elif unfinished == "loop":
        from hermes_cli.loops import LoopManager
        LoopManager("retired").set("synthetic loop", interval_seconds=1800)
    elif unfinished and unfinished.endswith("-read-error"):
        from hermes_cli.loops import LoopManager
        from hermes_cli.heartbeat import HeartbeatManager
        prefix = unfinished.removesuffix("-read-error") + ":"
        if prefix == "loop:":
            LoopManager("retired").set("synthetic loop", interval_seconds=1800)
        else:
            HeartbeatManager("retired").set("synthetic heartbeat", interval_seconds=1800)
        get_meta = SessionDB.get_meta

        def unavailable(self, key):
            if key.startswith(prefix):
                raise OSError("synthetic underlying store unavailable")
            return get_meta(self, key)

        monkeypatch.setattr(SessionDB, "get_meta", unavailable)
    elif unfinished and unfinished.endswith("-corrupt"):
        db.set_meta(unfinished.removesuffix("-corrupt") + ":retired", "{")

    assert _detach(owner)["result"] == {"detached": unfinished is None}
    assert len(active_session_registry_snapshot()) == 1
    assert db.get_session("retired")["ended_at"] is None
    assert not closed
    if unfinished:
        assert server._session_transport_contains(session, owner)
        assert server._pending_ws_reaps == {}
    else:
        assert session["transport"] is server._detached_ws_transport
        server._pending_ws_reaps["retired"].callback()
        assert "retired" not in server._sessions
        assert active_session_registry_snapshot() == []
        assert db.get_session("retired")["ended_at"] is None
        assert closed == [True]


def test_detach_cannot_park_a_submit_between_reattach_and_turn_claim(detached_env, monkeypatch):
    make, _db = detached_env
    owner = _Peer()
    session, _closed = make("retired", owner)
    at_claim, release_claim, detach_attempted, detach_done = (threading.Event() for _ in range(4))
    lock = threading.RLock()
    acquired = []

    class ObservedResumeLock:
        def __enter__(self):
            if threading.current_thread().name == "detacher":
                got_lock = lock.acquire(blocking=False)
                acquired.append(got_lock)
                detach_attempted.set()
                if not got_lock:
                    lock.acquire()
            else:
                lock.acquire()

        def __exit__(self, *args):
            lock.release()

    monkeypatch.setattr(server, "_session_resume_lock", ObservedResumeLock())
    claim = server._lock_in_submit_turn

    def gated_claim(*args):
        at_claim.set()
        assert release_claim.wait(10)
        return claim(*args)

    monkeypatch.setattr(server, "_lock_in_submit_turn", gated_claim)
    monkeypatch.setattr(server, "_legacy_group_fence_error", lambda *args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *args: False)
    # Stop after real turn admission; this test never builds an agent or calls a provider.
    monkeypatch.setattr(server, "_persist_session_row_for_submit",
                        lambda rid, *args: server._ok(rid, {"status": "streaming"}))
    submitted, detached = [], []

    def submit():
        token = server.bind_transport(owner)
        try:
            submitted.append(server.handle_request({"id": "submit", "method": "prompt.submit",
                                                    "params": {"session_id": "retired", "text": "synthetic"}}))
        finally:
            server.reset_transport(token)

    def detach():
        try:
            detached.append(_detach(owner))
        finally:
            detach_done.set()

    submitter = threading.Thread(target=submit)
    detacher = threading.Thread(target=detach, name="detacher")
    try:
        submitter.start()
        assert at_claim.wait(10)
        detacher.start()
        assert detach_attempted.wait(10)
        if acquired[0]:
            assert detach_done.wait(10)
    finally:
        release_claim.set()
        submitter.join(10)
        if detacher.ident is not None:
            detacher.join(10)
    assert not submitter.is_alive() and not detacher.is_alive()
    assert submitted[0]["result"]["status"] == "streaming"
    assert detached[0]["result"] == {"detached": False}
    assert session["running"] is True
    assert server._session_transport_contains(session, owner)
    assert server._pending_ws_reaps == {}


def test_final_idle_info_allows_retirement_only_after_housekeeping_and_followups(detached_env, monkeypatch, tmp_path):
    from hermes_cli.execution_budget import execution_budget_snapshot
    make, _db = detached_env
    owner = _Peer()
    session, _closed = make("retired", owner)
    session.update(history_version=0, running=True, attached_images=[], image_counter=0, cols=80,
                   slash_worker=None, show_reasoning=False, tool_progress_mode="all", inflight_turn=None)
    agent = session["agent"]
    agent.clear_interrupt = lambda: None
    agent.run_conversation = lambda *args, **kw: {"final_response": "done"}
    for name in ("_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd", "_sync_session_key_after_compress"):
        monkeypatch.setattr(server, name, lambda *args, **kw: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kw: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_goal_followup_after_turn", lambda *args: "synthetic continuation")
    tail, release_tail, followup, release_followup = (threading.Event() for _ in range(4))
    idle_info = []

    def housekeeping(*args):
        tail.set()
        assert release_tail.wait(10)

    def followups(*args):
        followup.set()
        assert release_followup.wait(10)

    monkeypatch.setattr(server, "_post_turn_housekeeping", housekeeping)
    monkeypatch.setattr(server, "_run_post_turn_followups", followups)
    monkeypatch.setattr(server, "_emit_settled_session_info",
                        lambda *args: idle_info.append(_detach(owner)["result"]))
    try:
        assert server._run_prompt_submit("turn", "retired", session, "synthetic") is True
        worker = session["_run_thread"]
        assert tail.wait(10)
        assert session["running"] is False
        assert len(execution_budget_snapshot()[1]) == 1
        assert _detach(owner)["result"] == {"detached": False}
        assert idle_info == []
        release_tail.set()
        assert followup.wait(10)
        assert len(execution_budget_snapshot()[1]) == 1
        assert _detach(owner)["result"] == {"detached": False}
    finally:
        release_tail.set()
        release_followup.set()
        if worker := session.get("_run_thread"):
            worker.join(10)
    assert not worker.is_alive()
    assert idle_info == [{"detached": True}]
    assert execution_budget_snapshot()[1] == []
