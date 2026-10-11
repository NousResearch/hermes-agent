"""The real prompt pipeline keeps the conditional witness across both worker hops."""
from threading import Event

import pytest

from tui_gateway import server
from tests.tui_gateway import test_running_conditional_activation as running
from tests.tui_gateway.test_conditional_activation import Peer, activate
from tests.tui_gateway.test_bound_operations import invoke


@pytest.fixture
def built(monkeypatch, tmp_path):
    yield from running.built.__wrapped__(monkeypatch, tmp_path)


@pytest.fixture
def delegated(built):
    from run_agent import AIAgent

    db = built[4]
    db.create_session("delegated-child", source="tool")
    child = AIAgent(model="test-model", provider="openai", api_key="test-key",
                    base_url="https://example.invalid/v1", quiet_mode=True, skip_memory=True,
                    skip_context_files=True, skip_background_review=True, save_trajectories=False,
                    platform="gui", session_id="delegated-child", session_db=db)
    try:
        yield child
    finally:
        child.close()


def test_real_prompt_pipeline_admits_once_and_refuses_a_changed_worker_identity(built, delegated, monkeypatch):
    creator, binding, record, agent, db = built
    peer = Peer()
    assert "result" in activate(peer, binding)
    # Only environment-heavy presentation/config work is stubbed. The actual
    # submit row, admission, both workers, turn facade and DB lease remain live.
    for name in ("_wire_callbacks", "_sync_agent_model_with_config", "_tts_stream_begin"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *a: False)
    monkeypatch.setattr(server, "_get_usage", lambda *a: {})
    monkeypatch.setattr("agent.relay_cwd.resolve_relay_scope_cwds", lambda *a: ("", ""))
    loop_entered, loop_release = Event(), Event()
    calls, child_calls = [], []
    def loop(engine, text, *a, **k):
        if engine is delegated:
            child_calls.append(text)
            return {"final_response": "child completed", "messages": []}
        # A legitimate child turn inherits the hosting thread's context, but its
        # own engine/lease must not inherit the parent's consumed admission pin.
        assert delegated.run_conversation("delegated work")["final_response"] == "child completed"
        calls.append(text)
        loop_entered.set()
        assert loop_release.wait(5)
        return {"final_response": "done", "messages": [{"role": "user", "content": text},
                                                       {"role": "assistant", "content": "done"}]}
    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    response = invoke(peer, binding, "prompt.submit", text="new deliberate turn")
    assert response["result"]["operation_result"]["status"] == "streaming"
    first = record["_run_thread"]
    try:
        assert loop_entered.wait(5)
        active = record["_run_thread"]
        assert agent._active_session_turn_lease_holder
        assert invoke(peer, binding, "prompt.submit", text="not a queued replay")["error"]["code"] == 4009
        assert calls == ["new deliberate turn"] and child_calls == ["delegated work"]
    finally:
        loop_release.set()
        first.join(timeout=5)
        record["_run_thread"].join(timeout=5)
    assert not record["running"] and not active.is_alive()
    # Park at durable admission after both production thread hops, then rotate.
    acquire = db.acquire_session_turn_lease
    entered, release = Event(), Event()
    def acquiring(*a, **k):
        entered.set()
        assert release.wait(5)
        return acquire(*a, **k)
    monkeypatch.setattr(db, "acquire_session_turn_lease", acquiring)
    response = invoke(peer, binding, "prompt.submit", text="do not execute on another segment")
    assert response["result"]["operation_result"]["status"] == "streaming"
    first = record["_run_thread"]
    try:
        assert entered.wait(5)
        agent.session_id = "changed"
        agent.session_id = binding["stored_session_id"]
    finally:
        release.set()
        first.join(timeout=5)
        record["_run_thread"].join(timeout=5)
    assert calls == ["new deliberate turn"]
    assert not record["running"] and agent._active_session_turn_lease_holder is None
    assert record["inflight_turn"]["status"] == "error"
    assert invoke(peer, binding, "session.interrupt")["error"]["code"] == 4007


def test_bound_interrupt_stops_only_the_original_running_turn(built, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    creator, binding, record, agent, db = built
    peer = Peer()
    entered, release = Event(), Event()
    def loop(*a, **k):
        entered.set()
        assert release.wait(5)
        return {"final_response": "partial", "messages": [], "interrupted": True}
    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    monkeypatch.setattr("agent.relay_cwd.resolve_relay_scope_cwds", lambda *a: ("", ""))
    with record["history_lock"]:
        record["running"] = True
        server._start_inflight_turn(record, "existing accepted work")
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker = pool.submit(agent.run_conversation, "existing accepted work")
        try:
            assert entered.wait(5)
            assert "result" in activate(peer, binding)
            stop = pool.submit(invoke, peer, binding, "session.interrupt")
            assert stop.result(timeout=5)["result"]["operation_result"]["status"] == "interrupted"
            assert agent._interrupt_requested and agent._hard_interrupt_requested.is_set()
            assert not worker.done()  # no wait/restart of the worker for cancellation
        finally:
            release.set()
        assert worker.result(timeout=5)["interrupted"]
    record["running"] = False
