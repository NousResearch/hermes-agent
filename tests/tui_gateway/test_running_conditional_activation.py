"""Local running-engine membership qualification; provider execution stays stubbed."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tui_gateway import server
from tests.tui_gateway.test_conditional_activation import Peer, activate, setup


@pytest.fixture
def built(monkeypatch, tmp_path):
    creator, binding, record = setup(monkeypatch, tmp_path)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    for name in ("_register_session_cwd", "_session_todo_state"):
        monkeypatch.setattr(server, name, lambda *a: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("test-model", "openai"))
    db = SessionDB(server._hermes_home / "state.db")
    db.create_session(binding["stored_session_id"], source="tui")
    agent = AIAgent(model="test-model", provider="openai", api_key="test-key",
                    base_url="https://example.invalid/v1", quiet_mode=True, skip_memory=True,
                    skip_context_files=True, skip_background_review=True, save_trajectories=False,
                    platform="gui", session_id=binding["stored_session_id"], session_db=db)
    assert server._attach_built_agent(binding["session_id"], record, agent)
    record["agent_ready"].set()
    try:
        yield creator, binding, record, agent, db
    finally:
        agent.close()
        db.close()


def test_running_turn_keeps_its_execution_and_lease_while_peer_joins(built, monkeypatch):
    creator, binding, record, agent, db = built
    entered, release = Event(), Event()
    calls = []
    def loop(engine, message, *a, **k):
        calls.append((engine, message))
        entered.set()
        assert release.wait(5)
        return {"final_response": "completed", "messages": []}
    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    monkeypatch.setattr("agent.relay_cwd.resolve_relay_scope_cwds", lambda *a: ("", ""))
    with record["history_lock"]:
        record["running"] = True
        server._start_inflight_turn(record, "accepted work")
    assert activate(Peer(), binding)["error"]["code"] == 4007  # accepted but no engine lease yet
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(agent.run_conversation, "accepted work")
            try:
                assert entered.wait(5)
                holder = agent._active_session_turn_lease_holder
                assert holder
                peer = Peer()
                assert activate(peer, binding)["result"] == {"attached": True, "accepted_binding": binding}
                assert activate(peer, binding)["result"]["accepted_binding"] == binding
                server._emit("message.complete", binding["session_id"], {"text": "ongoing output"})
                assert creator.delivered.wait(5) and peer.delivered.wait(5)
                assert record["running"] and agent._active_session_turn_lease_holder == holder
                assert not agent._interrupt_requested and not agent._hard_interrupt_requested.is_set()
                assert calls == [(agent, "accepted work")]
            finally:
                release.set()
            assert pending.result(timeout=5)["final_response"] == "completed"
    finally:
        with record["history_lock"]:
            record["running"] = False
            record["inflight_turn"] = None


def test_engine_witness_settling_and_revision_races_refuse_without_side_effects(built, monkeypatch):
    creator, binding, record, agent, db = built
    assert "result" in activate(Peer(), binding)
    entered, release = Event(), Event()
    # A contended lease forces real post-admission reload. Identity must be guarded
    # throughout resolution/loading, not just the eventual segment assignment.
    acquire = db.acquire_session_turn_lease
    def contend(*a, **k):
        k["on_contended"]()
        return acquire(*a, **k)
    resolve = db.resolve_resume_session_id
    def resolving(sid):
        entered.set()
        assert release.wait(5)
        return resolve(sid)
    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as pool:
        patch.setattr(db, "acquire_session_turn_lease", contend)
        patch.setattr(db, "resolve_resume_session_id", resolving)
        from agent.turn_facade_lease import admit_durable_turn_lease
        pending = pool.submit(admit_durable_turn_lease, agent, session_id=agent.session_id,
                              relay_turn_id="qualification", task_context={"platform": "gui"},
                              conversation_history=[])
        peer = Peer()
        try:
            assert entered.wait(5)
            viewers = dict(record["viewers"])
            assert activate(peer, binding)["error"]["code"] == 4007
            assert record["viewers"] == viewers and not peer.frames
        finally:
            release.set()
        pending.result(timeout=5).lease.release()
    assert "result" in activate(Peer(), binding)
    viewers = dict(record["viewers"])
    def replace_engine():
        other = AIAgent.__new__(AIAgent)
        other.session_id = binding["stored_session_id"]
        other._session_db = db
        assert server._attach_built_agent(binding["session_id"], record, other)
    changes = (
        replace_engine,
        lambda: record.__setitem__("agent", None),
        lambda: record["agent_ready"].clear(),
        lambda: record.__setitem__("_compute_host_active", True),
        lambda: record.__setitem__("inflight_turn", {"error": "interrupted", "recoverable": True}),
        lambda: setattr(agent, "_interrupt_requested", True),
        lambda: record.__setitem__("profile_home", str(server._hermes_home / "other")),
        lambda: setattr(agent, "_session_db", object()),
    )
    for change in changes:
        old_interrupt = agent._interrupt_requested
        old_home, old_db = record["profile_home"], agent._session_db
        old_agent, old_host, old_turn = record["agent"], record.get("_compute_host_active"), record["inflight_turn"]
        change()
        refused = Peer()
        assert activate(refused, binding)["error"]["code"] == 4007
        assert record["viewers"] == viewers and not refused.frames
        record["agent"], record["_compute_host_active"], record["inflight_turn"] = old_agent, old_host, old_turn
        agent._interrupt_requested = old_interrupt
        record["agent_ready"].set()
        record["profile_home"], agent._session_db = old_home, old_db
        assert "result" in activate(Peer(), binding)
        viewers = dict(record["viewers"])
    assert "result" in activate(Peer(), binding)
    # Acceptance wins before a direct engine assignment: receipt is the guarded cut;
    # the later ABA revision still invalidates all subsequent activation attempts.
    entered.clear()
    release.clear()
    attach = server._attach_session_transport
    def attaching(session, peer):
        entered.set()
        assert release.wait(5)
        return attach(session, peer)
    started = Event()
    def rotate():
        started.set()
        agent.session_id = "different"
        agent.session_id = binding["stored_session_id"]
    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=2) as pool:
        patch.setattr(server, "_attach_session_transport", attaching)
        accepted = pool.submit(activate, Peer(), binding)
        try:
            assert entered.wait(5)
            writer = pool.submit(rotate)
            assert started.wait(5)
            assert agent.session_id == binding["stored_session_id"]
        finally:
            release.set()
        assert accepted.result(timeout=5)["result"]["accepted_binding"] == binding
        writer.result(timeout=5)
    viewers = dict(record["viewers"])
    refused = Peer()
    assert activate(refused, binding)["error"]["code"] == 4007
    assert record["viewers"] == viewers and not refused.frames
