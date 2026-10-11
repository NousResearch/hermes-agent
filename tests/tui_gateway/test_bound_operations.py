"""Conditional local operations fence identity at admission and fail closed on host routing."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tui_gateway import server, server_requests
from tests.tui_gateway.test_conditional_activation import Peer, activate, rpc, setup
from tests.tui_gateway import test_running_conditional_activation as running


@pytest.fixture
def built(monkeypatch, tmp_path):
    yield from running.built.__wrapped__(monkeypatch, tmp_path)


def invoke(peer, binding, method, **operation):
    return rpc(peer, "session.invoke_bound", session_id=binding["session_id"],
               expected_binding=binding, operation={"method": method, **operation})


def request(sid, *, qids=None):
    req = server_requests.ServerRequest(sid, "clarify", {"prompt": "choose"}, qids=qids)
    with server_requests._lock:
        server_requests._open[req.id] = req
    return req


def test_host_routing_and_past_child_lifetimes_never_certify_local_identity(monkeypatch, tmp_path):
    creator, binding, record = setup(monkeypatch, tmp_path)
    from hermes_cli.profiles import get_profile_dir
    (get_profile_dir("secondary") / "config.yaml").write_text("dashboard:\n  turn_isolation: true\n")
    secondary = rpc(creator, "session.create", profile="secondary")["result"]["creation_binding"]
    assert activate(Peer(), secondary, profile="secondary")["error"]["code"] == 4007
    assert "result" in activate(Peer(), binding)  # launch policy is still local
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    peer = Peer()
    viewers = dict(record.get("viewers", {}))
    assert activate(peer, binding)["error"]["code"] == 4007  # unbuilt but next turn belongs to a child
    assert record.get("viewers", {}) == viewers and not peer.frames
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": False})
    assert "result" in activate(peer, binding)
    class FailedHost:
        def submit_turn(self, *a, **k):
            raise RuntimeError("pipe failed")
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *a: FailedHost())
    assert server._submit_prompt_to_compute_host("turn", binding["session_id"], record, "accepted")["error"]
    assert record["_compute_host_ever_owned"]
    # Real mirror adoption can return to the old key after child rotation/restart.
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *a, **k: True)
    for key in ("child-tip", binding["stored_session_id"]):
        with record["history_lock"]:
            server._compute_host_adopt_frame_meta(record, {"sid": binding["session_id"], "session_key": key})
    assert record["session_key"] == binding["stored_session_id"]
    assert activate(Peer(), binding)["error"]["code"] == 4007
    assert invoke(peer, binding, "session.interrupt")["error"]["code"] == 4007
    assert not peer.frames


def test_operations_and_raw_bypasses_are_fenced_with_request_ownership_and_rotation(built, monkeypatch):
    creator, binding, record, agent, db = built
    peer = Peer()
    assert "result" in activate(peer, binding)
    for operation in ({"method": "session.close"}, {"method": "prompt.submit", "text": 7},
                      {"method": "request.answer", "id": "x", "result": [], "owner": "alice"}):
        assert rpc(peer, "session.invoke_bound", session_id=binding["session_id"],
                   expected_binding=binding, operation=operation)["error"]["code"] == 4000
    assert invoke(Peer(), binding, "session.interrupt")["error"]["code"] == 4007
    with monkeypatch.context() as patch:
        def unreadable(*a, **k):
            raise OSError("database identity unavailable")
        patch.setattr(db, "get_compression_tip", unreadable)
        assert invoke(peer, binding, "prompt.submit", text="must not persist")["error"]["code"] == 4007
    assert not db.get_messages(binding["stored_session_id"])
    req = request(binding["session_id"], qids=["one", "two"])
    foreign = request("another-runtime")
    try:
        assert invoke(peer, binding, "request.answer", id=foreign.id, result={"answer": "spoof"})["error"]["code"] == 4007
        assert not foreign.answered and not foreign.event.is_set()
        assert rpc(peer, "request.answer", id=req.id, result={"answer": "bypass"})["error"]["code"] == 4007
        server.dispatch({"jsonrpc": "2.0", "id": req.id, "result": {"answer": "raw"}}, peer)
        assert not req.answered
        assert invoke(peer, binding, "clarify.lock", request_id=req.id, question_id="one", answer="chosen")["result"] == {
            "operation_result": {"status": "ok", "remaining": ["two"]}}
        assert invoke(peer, binding, "request.answer", id=req.id, result={"answers": {"two": "last"}})["result"] == {
            "operation_result": {"status": "ok"}}
        assert req.result == {"answers": {"one": "chosen", "two": "last"}, "outcome": "submitted"}
        assert invoke(peer, binding, "request.answer", id=req.id, result={})["error"]["code"] == 4007  # no retry receipt
        fresh = request(binding["session_id"])
        entered, release, writer_started = Event(), Event(), Event()
        resolve = server_requests.resolve_response
        def settling(frame, *a, **k):
            entered.set()
            assert release.wait(5)
            return resolve(frame, *a, **k)
        def rotate():
            writer_started.set()
            agent.session_id = "descendant"
            agent.session_id = binding["stored_session_id"]
        with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=2) as pool:
            patch.setattr(server_requests, "resolve_response", settling)
            answer = pool.submit(invoke, peer, binding, "request.answer", id=fresh.id, result={"answer": "yes"})
            try:
                assert entered.wait(5)
                writer = pool.submit(rotate)
                assert writer_started.wait(5)
                assert agent.session_id == binding["stored_session_id"]
            finally:
                release.set()
            assert answer.result(timeout=5)["result"]["operation_result"]["status"] == "ok"
            writer.result(timeout=5)
        assert fresh.answered
        pending = request(binding["session_id"])
        for method, kwargs in (("prompt.submit", {"text": "new"}), ("session.interrupt", {}),
                               ("request.answer", {"id": pending.id, "result": {"answer": "old"}}),
                               ("clarify.lock", {"request_id": pending.id, "question_id": "one"})):
            assert invoke(peer, binding, method, **kwargs)["error"]["code"] == 4007
        assert not pending.answered and not agent._interrupt_requested
        assert rpc(peer, "prompt.submit", session_id=binding["session_id"], text="legacy")["error"]["code"] == 4007
        with server._sessions_lock:
            server._sessions[binding["session_id"]] = dict(record)
        assert invoke(peer, binding, "session.interrupt")["error"]["code"] == 4007
        assert rpc(peer, "session.close", session_id=binding["session_id"])["error"]["code"] == 4007
    finally:
        with server_requests._lock:
            for item in (req, foreign, locals().get("fresh"), locals().get("pending")):
                if item is not None:
                    server_requests._open.pop(item.id, None)


@pytest.mark.parametrize("drift", ["none", "worker", "lease", "tip", "lease_abort"])
def test_prompt_worker_carries_binding_through_lease_admission_without_replay(built, monkeypatch, drift):
    creator, binding, record, agent, db = built
    peer = Peer()
    assert "result" in activate(peer, binding)
    # Real prompt.submit persistence + worker + facade lease; ancillary UI/turn
    # presentation is replaced by a minimal driver of the real engine facade.
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *a: False)
    entered, release = Event(), Event()
    errors, calls, results = [], [], []
    lease_entered, lease_release = Event(), Event()
    acquire = db.acquire_session_turn_lease
    def acquiring(*a, **k):
        lease_entered.set()
        assert lease_release.wait(5)
        return False if drift == "lease_abort" else acquire(*a, **k)
    if drift in {"lease", "tip", "lease_abort"}:
        monkeypatch.setattr(db, "acquire_session_turn_lease", acquiring)
    def run(rid, sid, current, text, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            results.append(agent.run_conversation(text))
        except ValueError as exc:
            errors.append(str(exc))
        finally:
            with current["history_lock"]:
                current["running"] = False
                current["inflight_turn"] = None
    monkeypatch.setattr(server, "_run_prompt_submit", run)
    monkeypatch.setattr("agent.conversation_loop.run_conversation", lambda *a, **k: calls.append(a[1]) or {
        "final_response": "done", "messages": []})
    monkeypatch.setattr("agent.relay_cwd.resolve_relay_scope_cwds", lambda *a: ("", ""))
    result = invoke(peer, binding, "prompt.submit", text="deliberate new input")
    assert result["result"]["operation_result"]["status"] == "streaming"
    worker = record["_run_thread"]
    try:
        assert entered.wait(5)
        rows = db.get_messages(binding["stored_session_id"])
        assert len([r for r in rows if r["role"] == "user"]) == 1
        if drift == "worker":
            agent.session_id = "changed-after-acceptance"
            agent.session_id = binding["stored_session_id"]
        if drift in {"lease", "tip", "lease_abort"}:
            release.set()
            assert lease_entered.wait(5)
            if drift == "lease":
                agent.session_id = "changed-during-wait"
                agent.session_id = binding["stored_session_id"]
            elif drift == "tip":
                monkeypatch.setattr(db, "resolve_resume_session_id", lambda sid, **kw: "compressed-by-another-holder")
            else:
                agent._interrupt_requested = True
    finally:
        release.set()
        lease_release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    if drift == "none":
        assert calls == ["deliberate new input"] and not errors
        assert agent._active_session_turn_lease_holder is None
        return
    assert not calls
    if drift == "lease_abort":
        assert results[0]["interrupted"] and results[0]["messages"] == []
    else:
        assert errors and "identity changed" in errors[0]
    assert getattr(agent, "_active_session_turn_lease_holder", None) is None
    if drift != "lease_abort":
        assert invoke(peer, binding, "prompt.submit", text="do not retry")["error"]["code"] == 4007
    assert len([r for r in db.get_messages(binding["stored_session_id"]) if r["role"] == "user"]) == 1
