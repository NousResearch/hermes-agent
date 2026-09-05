"""A live handoff source must resume the durable conversation, not its old cache."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from agent.turn_facade_lease import admit_durable_turn_lease
from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def handback(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    key, sid = "handback-stored", "handback-live"
    db.create_session(key, source="tui")
    db.append_message(key, "user", "source question")
    db.append_message(key, "assistant", "source answer")
    history = db.get_messages_as_conversation(key, include_row_ids=True)
    seen = []
    called = threading.Event()

    def run(prompt, conversation_history=None, **_kwargs):
        called.set()
        admission = admit_durable_turn_lease(
            agent, session_id=key, relay_turn_id="fake-local-turn",
            task_context={"session_id": key, "platform": "tui"},
            conversation_history=conversation_history)
        if admission.early_result is not None:
            return admission.early_result
        try:
            model_history = list(admission.conversation_history or [])
            seen.append(model_history)

            return {"final_response": "local answer", "messages": [
                *model_history, {"role": "user", "content": prompt},
                {"role": "assistant", "content": "local answer"}]}
        finally:
            if admission.lease is not None:
                admission.lease.release()

    agent = SimpleNamespace(
        run_conversation=run, _session_db=db, session_id=key,
        _session_messages=list(history), close=Mock(), _persist_session=Mock(),
        commit_memory_session=Mock(),
        _emit_status=lambda *_a: None, _emit_warning=lambda *_a: None,
        _liveness_activity_lock=threading.Lock,
    )
    session = {"agent": agent, "history": list(history), "history_lock": threading.Lock(),
               "history_version": 0, "running": False, "session_key": key,
               "source": "tui", "attached_images": [], "cwd": str(tmp_path)}
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_get_usage", lambda _a: {})
    monkeypatch.setattr(server, "render_message", lambda *_a: "")
    monkeypatch.setattr(server, "_after_complete_turn", lambda *_a: None)
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True, home_channel=HomeChannel(
        platform=Platform.DISCORD, chat_id="fake-home", name="fake destination"))
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)

    def rpc(method, **params):
        return server.handle_request({"id": "handback-test", "method": method, "params": params})

    def destination():
        db.record_gateway_session_peer(key, source="discord", session_key="fake-route", chat_id="fake-home")
        db.append_message(key, "user", "destination question")
        db.append_message(key, "assistant", "destination answer")

    yield SimpleNamespace(db=db, key=key, sid=sid, session=session, agent=agent,
                          seen=seen, called=called, rpc=rpc, destination=destination)
    thread = session.get("_run_thread")
    if thread is not None:
        thread.join(timeout=5)
    db.close()


@pytest.mark.parametrize("outcome", ["completed", "failed", "lost_ack", "lease_busy", "read_failure"])
@pytest.mark.parametrize("entry", ["session.resume", "session.activate"])
def test_explicit_handback_feeds_destination_turns_to_next_model(handback, monkeypatch, outcome, entry):
    h = handback
    queued = h.rpc("handoff.request", session_id=h.sid, platform="discord")
    assert queued["result"]["queued"] is True
    assert h.db.claim_handoff(h.key)
    if outcome == "failed":
        h.db.fail_handoff(h.key, "destination delivery failed after binding")
    else:
        h.db.complete_handoff(h.key)
    h.destination()
    if outcome != "lost_ack":
        assert h.rpc("handoff.state", session_id=h.sid)["result"]["state"] == (
            "failed" if outcome == "failed" else "completed")
    row_before = h.db.get_session(h.key)
    resumed = h.rpc(entry, session_id=h.key if entry == "session.resume" else h.sid)
    assert resumed["result"]["session_id"] == h.sid
    assert h.db.get_session(h.key) == row_before
    if outcome in {"lease_busy", "read_failure"}:
        with monkeypatch.context() as failure:
            if outcome == "lease_busy":
                assert h.db.try_acquire_session_turn_lease(h.key, "fake-gateway-turn")
                failure.setattr("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 0)
            else:
                failure.setattr(h.db, "get_messages_as_conversation", Mock(side_effect=RuntimeError("read unavailable")))
            assert h.rpc("prompt.submit", session_id=h.sid, text="blocked attempt")["result"]["status"] == "streaming"
            assert h.called.wait(5)
            h.session["_run_thread"].join(timeout=5)
            assert not h.session["_run_thread"].is_alive()
            assert not h.seen
            assert h.agent._reload_history_after_handoff is True
            assert h.db.get_session(h.key)["ended_at"] is None
        if outcome == "lease_busy":
            # Failed local admission must not release the destination's lease.
            assert not h.db.try_acquire_session_turn_lease(h.key, "other-probe")
            h.db.release_session_turn_lease(h.key, "fake-gateway-turn")
        else:
            # A read failure after admission must release the local lease.
            assert h.db.try_acquire_session_turn_lease(h.key, "other-probe")
            h.db.release_session_turn_lease(h.key, "other-probe")
        h.called.clear()
    # Another idle destination turn can finish AFTER resume, BEFORE local admission.
    h.db.append_message(h.key, "user", "late destination question")
    h.db.append_message(h.key, "assistant", "late destination answer")
    expected = [(m["role"], m["content"]) for m in h.db.get_messages_as_conversation(h.key)]
    result = h.rpc("prompt.submit", session_id=h.sid, text="continue locally")
    assert result["result"]["status"] == "streaming"
    assert h.called.wait(5), "fake model was not invoked"
    h.session["_run_thread"].join(timeout=5)
    assert not h.session["_run_thread"].is_alive()
    assert h.seen
    assert [(m["role"], m["content"]) for m in h.seen[0]] == expected
    assert h.agent._reload_history_after_handoff is False
    assert h.db.get_session(h.key)["ended_at"] is None
    h.agent.close.assert_not_called()
    h.agent.commit_memory_session.assert_not_called()


def test_handback_history_survives_real_facade_failure_and_recovery(tmp_path, monkeypatch, request):
    from agent.session_persistence import SessionPersistenceMixin
    from agent.turn_facade import TurnFacadeMixin
    from tui_gateway.session_auto_continue import _restore_agent_history_after_turn_error

    class Agent(TurnFacadeMixin, SessionPersistenceMixin, SimpleNamespace):
        pass

    db = SessionDB(db_path=tmp_path / "state.db")
    request.addfinalizer(db.close)
    key = "handback-retry"
    db.create_session(key, source="tui")
    db.append_message(key, "user", "source question")
    db.append_message(key, "assistant", "source answer")
    h = SimpleNamespace(db=db, key=key, session={
        "history": db.get_messages_as_conversation(key, include_row_ids=True),
        "history_lock": threading.Lock(), "history_version": 0,
    })
    agent = Agent()
    agent._session_db, agent.session_id, agent.platform = h.db, h.key, "tui"
    agent._session_messages = list(h.session["history"])
    agent._last_flushed_db_idx = len(agent._session_messages)
    agent._session_persist_lock = threading.RLock()
    agent._emit_status = lambda *_a: None
    agent._conversation_root_id = lambda: h.key
    # Explicit handback has already transferred its one-shot intent to this agent.
    agent._reload_history_after_handoff = True
    pending = {"role": "user", "content": "retry locally"}
    agent._pending_cli_user_message = pending
    db.append_message(key, "user", "destination question")
    db.append_message(key, "assistant", "destination answer")
    expected = h.db.get_messages_as_conversation(h.key, include_row_ids=True)
    seen = []

    # Keep the facade, durable lease bracket, recovery and persistence real. Only
    # unrelated relay/binding/timers and the fallible loop entry are substituted.
    coordinator = SimpleNamespace(
        register_session_initializer=lambda *_a, **_kw: None,
        acquire_conversation=lambda **_kw: object(),
        begin_turn=lambda *_a, **_kw: SimpleNamespace(relay_enabled=False),
        finish_logical_calls=lambda *_a, **_kw: None,
        end_turn=lambda *_a, **_kw: None,
        release_conversation=lambda *_a, **_kw: None,
    )
    monkeypatch.setattr("agent.relay_runtime.SESSION_COORDINATOR", coordinator)
    monkeypatch.setattr("agent.subagent_lifecycle.bind_subagent_parent", lambda _a: nullcontext())
    monkeypatch.setattr("agent.turn_facade_lease.DurableTurnLease.build_threads", lambda _s: None)
    monkeypatch.setattr("agent.turn_facade_lease.DurableTurnLease.start", lambda _s: None)

    def loop(current, prompt, _system, history, *_a, **_kw):
        seen.append(list(history))
        assert current._active_session_turn_lease_holder
        if len(seen) == 1:
            raise RuntimeError("turn prologue failed before persistence")
        messages = [*history, {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "local answer"}]
        current._persist_session(messages, history)
        return {"messages": messages, "final_response": "local answer", "completed": True}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    with pytest.raises(RuntimeError, match="turn prologue failed before persistence"):
        agent.run_conversation("retry locally", conversation_history=h.session["history"])
    assert seen[0] == expected
    assert agent._active_session_turn_lease_holder is None
    assert h.db.try_acquire_session_turn_lease(h.key, "retry-probe")
    h.db.release_session_turn_lease(h.key, "retry-probe")
    assert h.db.get_session(h.key)["ended_at"] is None
    assert h.db.get_messages_as_conversation(h.key, include_row_ids=True) == expected
    assert agent._pending_cli_user_message is pending
    assert _restore_agent_history_after_turn_error(h.session, agent)

    # No resume/activate or manual rearming between the failed attempt and retry.
    result = agent.run_conversation("retry locally", conversation_history=h.session["history"])
    assert seen[1] == expected
    assert result["completed"] is True
    assert agent._reload_history_after_handoff is False
    durable = h.db.get_messages_as_conversation(h.key, include_row_ids=True)
    assert durable[:len(expected)] == expected
    assert [(m["role"], m["content"]) for m in durable[len(expected):]] == [
        ("user", "retry locally"), ("assistant", "local answer")]
    assert agent._last_flushed_db_idx == len(durable)
    assert agent._active_session_turn_lease_holder is None
    assert h.db.get_session(h.key)["ended_at"] is None

    # Ordinary successful reuse must not turn handback into a permanent DB reload.
    monkeypatch.setattr(h.db, "get_messages_as_conversation", Mock(side_effect=AssertionError("unexpected reload")))
    result = agent.run_conversation("ordinary follow-up", conversation_history=result["messages"])
    assert result["completed"] is True
    assert seen[2] == agent._session_messages[:-2]
    assert agent._reload_history_after_handoff is False


@pytest.mark.parametrize("condition", [
    "pending", "running", "local_busy", "ordinary_live",
])
@pytest.mark.parametrize("entry", ["session.resume", "session.activate"])
def test_handback_preserves_ownership_when_not_safe_to_refresh(handback, monkeypatch, condition, entry):
    h = handback
    if condition != "ordinary_live":
        assert h.rpc("handoff.request", session_id=h.sid, platform="discord")["result"]["queued"]
        if condition != "pending":
            assert h.db.claim_handoff(h.key)
        if condition not in {"pending", "running"}:
            h.db.complete_handoff(h.key)
    h.destination()
    if condition == "local_busy":
        h.session["running"] = True
    before = h.session["history"]
    row = h.db.get_session(h.key)
    with h.db._read_ctx() as conn:
        leases_before = [tuple(r) for r in conn.execute("SELECT * FROM session_turn_leases")]

    resumed = h.rpc(entry, session_id=h.key if entry == "session.resume" else h.sid, omit_messages=True)
    if condition in {"pending", "running"}:
        assert resumed["error"]["code"] == 4009
    else:
        assert resumed["result"]["session_id"] == h.sid
        assert resumed["result"]["running"] == (condition == "local_busy")
    assert h.session["history"] is before
    assert h.db.get_session(h.key) == row
    with h.db._read_ctx() as conn:
        assert [tuple(r) for r in conn.execute("SELECT * FROM session_turn_leases")] == leases_before
    h.agent.close.assert_not_called()
    h.agent._persist_session.assert_not_called()
    h.agent.commit_memory_session.assert_not_called()
