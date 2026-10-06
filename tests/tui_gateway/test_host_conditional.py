"""Positive child authority is exercised through a real supervisor/process/JSONL pipe."""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue
import sys
import threading

import pytest

from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor
from tests.tui_gateway.test_bound_operations import invoke
from tests.tui_gateway.test_conditional_activation import Peer, activate, rpc, setup


@pytest.fixture
def hosted(monkeypatch, tmp_path):
    creator, binding, record = setup(monkeypatch, tmp_path)
    root = Path(__file__).resolve().parents[2]
    supervisor = HostSupervisor(argv=[sys.executable, "-m", "tests.tui_gateway._conditional_host_fixture"],
        cwd=root, registry_path=tmp_path / "registry.json", expected_build_sha="unknown",
        expected_hermes_home=str(server._hermes_home),
        env={"HOME": str(tmp_path), "HERMES_HOME": str(server._hermes_home), "HERMES_RUNTIME_DIR": str(tmp_path / "runtime")},
        rpc_sink=server._relay_compute_host_rpc, respawn_max=0)
    supervisor.conditional_admission_sink = getattr(server, "_host_admission_allowed", None)
    monkeypatch.setattr(server, "_compute_host_supervisor", supervisor)
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *a: supervisor)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    done = {}
    original = server._on_compute_host_turn_done
    def completed(*a):
        original(*a)
        done.setdefault(a[1], threading.Event()).set()
    monkeypatch.setattr(server, "_on_compute_host_turn_done", completed)
    try:
        assert "result" in server._submit_prompt_to_compute_host("warm", binding["session_id"], record, "warm")
        assert done.setdefault(binding["session_id"], threading.Event()).wait(15)
        yield creator, binding, record, supervisor, done
    finally:
        supervisor.shutdown()


def control(supervisor, binding, action):
    return supervisor.control(binding["session_id"], route_name="session.save",
        payload={"type": "test.control", "test_action": action}, timeout=15)["result"]


def test_positive_child_activation_and_operations_keep_one_engine_and_one_subscription(hosted, monkeypatch):
    creator, binding, record, supervisor, done = hosted
    peer = Peer()
    prepared, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    class ObservedGate:
        def __init__(self):
            self.lock = threading.Lock()
        def __enter__(self):
            if self.lock.locked():
                second_entered.set()
            self.lock.acquire()
        def __exit__(self, *args):
            self.lock.release()
    peer._host_activation_lock = ObservedGate()
    exchange = supervisor.conditional_exchange
    def hold_first(boot, payload):
        if prepared.is_set():
            second_entered.set()  # an unfenced second prepare is also observable
        reply = exchange(boot, payload)
        if not prepared.is_set():
            prepared.set()
            assert release.wait(5)
        return reply
    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=2) as pool:
        patch.setattr(supervisor, "conditional_exchange", hold_first)
        first = pool.submit(activate, peer, binding)
        assert prepared.wait(5)
        second = pool.submit(activate, peer, binding)
        assert second_entered.wait(5)
        release.set()
        assert first.result(timeout=10)["result"]["accepted_binding"] == binding
        assert second.result(timeout=10)["result"]["accepted_binding"] == binding
    token = record["host_bound_subscribers"][peer]["subscription"]
    assert activate(peer, binding)["result"]["accepted_binding"] == binding
    assert record["host_bound_subscribers"][peer]["subscription"] == token
    assert control(supervisor, binding, "inspect")["members"] == 1
    secondary = rpc(creator, "session.create", profile="secondary")["result"]["creation_binding"]
    secondary_record = server._sessions[secondary["session_id"]]
    assert "result" in server._submit_prompt_to_compute_host("secondary-warm", secondary["session_id"], secondary_record, "warm")
    assert done.setdefault(secondary["session_id"], threading.Event()).wait(15)
    scoped = Peer()
    assert activate(scoped, secondary)["error"]["code"] == 4007
    assert not scoped.frames
    assert "result" in activate(scoped, secondary, profile="secondary")
    assert rpc(scoped, "session.invoke_bound", session_id=secondary["session_id"], expected_binding=secondary,
               operation={"method": "session.interrupt"})["error"]["code"] == 4007
    control(supervisor, secondary, "replace")  # same boot/key/engine, different child runtime object
    assert rpc(scoped, "session.invoke_bound", session_id=secondary["session_id"], profile="secondary",
               expected_binding=secondary, operation={"method": "session.interrupt"})["error"]["code"] == 4007
    assert activate(Peer(), secondary, profile="secondary")["error"]["code"] == 4007
    server._detach_session_transport(secondary_record, scoped)
    request = control(supervisor, binding, "request")["id"]
    assert invoke(peer, binding, "clarify.lock", request_id=request, question_id="one", answer="selected")["result"]["operation_result"]["remaining"] == ["two"]
    assert invoke(peer, binding, "request.answer", id=request, result={"answers": {"two": "last"}})["result"]["operation_result"]["status"] == "ok"
    assert invoke(peer, binding, "request.answer", id=request, result={})["error"]["code"] == 4007
    lost = control(supervisor, binding, "request")["id"]
    receive = supervisor.conditional_receive
    def lose_commit_reply(ticket):
        reply = receive(ticket)
        if "response" in reply:
            raise queue.Empty  # child committed; caller cannot infer settlement
        return reply
    with monkeypatch.context() as patch:
        patch.setattr(supervisor, "conditional_receive", lose_commit_reply)
        assert invoke(peer, binding, "request.answer", id=lost, result={"answers": {"one": "first", "two": "last"}})["error"]["code"] == 5019
    assert invoke(peer, binding, "request.answer", id=lost, result={})["error"]["code"] == 4007
    foreign = control(supervisor, binding, "foreign_request")["id"]
    assert invoke(peer, binding, "request.answer", id=foreign, result={"answer": "must not settle"})["error"]["code"] == 4007
    assert control(supervisor, binding, "inspect")["requests"][foreign] is None
    assert activate(Peer("bob"), binding)["error"]["code"] == 4007
    assert invoke(peer, binding, "prompt.submit", text="hold")["result"]["operation_result"]["status"] == "streaming"
    # Real host turn owns a SQLite lease; activation does not wait/restart/stop it.
    holding = threading.Event()
    for _ in range(150):
        if control(supervisor, binding, "inspect")["holder"]:
            holding.set()
            break
        holding.wait(.02)
    assert holding.is_set()
    running = Peer()
    assert "result" in activate(running, binding)
    assert invoke(peer, binding, "prompt.submit", text="must not queue")["error"]["code"] == 4009
    assert invoke(peer, binding, "session.interrupt")["result"]["operation_result"]["status"] == "interrupted"
    server._detach_session_transport(record, running)
    server._detach_session_transport(record, peer)
    for _ in range(150):
        if control(supervisor, binding, "inspect")["members"] == 0:
            break
        threading.Event().wait(.02)
    assert control(supervisor, binding, "inspect")["members"] == 0


def test_child_aba_parent_replacement_expiry_and_stopped_lifetime_refuse_without_work(hosted, monkeypatch):
    creator, binding, record, supervisor, done = hosted
    peer = Peer()
    assert "result" in activate(peer, binding)
    viewers = dict(record["viewers"])
    late_dead = Peer()
    attach = server._attach_session_transport
    def close_at_final_attach(session, target):
        if target is late_dead:
            target._closed = True
        return attach(session, target)
    with monkeypatch.context() as patch:
        patch.setattr(server, "_attach_session_transport", close_at_final_attach)
        assert activate(late_dead, binding)["error"]["code"] == 4007
    assert record["viewers"] == viewers and not server._session_transport_contains(record, late_dead)
    assert not late_dead.frames
    # Losing the child receipt cannot publish tentative events or grant authority.
    receive = supervisor.conditional_receive
    def lost_activation(ticket):
        reply = receive(ticket)
        if "response" in reply:
            assert retrying not in record["viewers"]
            assert set(server._session_live_transports(record)) == {creator, peer}
            control(supervisor, binding, "notice")
            raise queue.Empty
        return reply
    retrying = Peer()
    with monkeypatch.context() as patch:
        patch.setattr(supervisor, "conditional_receive", lost_activation)
        assert activate(retrying, binding)["error"]["code"] == 4007
    assert not retrying.frames and retrying not in record["viewers"]
    assert "result" in activate(retrying, binding)
    server._detach_session_transport(record, retrying)
    before = control(supervisor, binding, "inspect")["messages"]
    exchange = supervisor.conditional_exchange
    def replaced(boot, payload):
        prepared = exchange(boot, payload)
        with server._sessions_lock:
            server._sessions[binding["session_id"]] = dict(record)
        return prepared
    with monkeypatch.context() as patch:
        patch.setattr(supervisor, "conditional_exchange", replaced)
        assert invoke(peer, binding, "prompt.submit", text="replacement must not execute")["error"]["code"] == 4007
    with server._sessions_lock:
        server._sessions[binding["session_id"]] = record
    # Rejected parent authority has not committed or mutated the child.
    assert control(supervisor, binding, "inspect")["messages"] == before
    send = supervisor.conditional_send
    # Hold commit behind a child identity writer: it can proceed only when the
    # bounded reservation expires. No sleeps guess whether the cut is still held.
    receive = supervisor.conditional_receive
    def delayed(boot, payload):
        return ("delay", boot, payload) if payload["action"] == "commit" else send(boot, payload)
    def expired(ticket):
        if ticket[0] == "delay":
            control(supervisor, binding, "notice")
            control(supervisor, binding, "rotate")
            return receive(send(ticket[1], ticket[2]))
        return receive(ticket)
    with monkeypatch.context() as patch:
        patch.setattr(supervisor, "conditional_send", delayed)
        patch.setattr(supervisor, "conditional_receive", expired)
        stranger = Peer()
        assert activate(stranger, binding)["error"]["code"] == 4007
        assert not stranger.frames and not server._session_transport_contains(record, stranger)
    assert invoke(peer, binding, "prompt.submit", text="must not execute")["error"]["code"] == 4007
    assert control(supervisor, binding, "inspect")["messages"] == before
    pid = supervisor.pid
    supervisor.shutdown()
    assert activate(Peer(), binding)["error"]["code"] == 4007
    assert supervisor.pid == pid and not supervisor.is_running()  # query never starts a child
    old_boot = record["_conditional_host_boot"]
    supervisor.start()  # simulate an explicitly initiated new host lifetime
    assert supervisor.conditional_boot() != old_boot
    assert activate(Peer(), binding)["error"]["code"] == 4007
    assert "result" in server._submit_prompt_to_compute_host("new-lifetime", binding["session_id"], record, "warm")
    # The new child's ordinary relay revokes old conditional delivery membership.
    for _ in range(150):
        if not server._session_transport_contains(record, peer):
            break
        threading.Event().wait(.02)
    assert not server._session_transport_contains(record, peer)
    assert invoke(peer, binding, "session.interrupt")["error"]["code"] == 4007
    fresh = rpc(creator, "session.create")["result"]["creation_binding"]
    fresh_record = server._sessions[fresh["session_id"]]
    assert "result" in server._submit_prompt_to_compute_host("fresh", fresh["session_id"], fresh_record, "warm")
    assert done.setdefault(fresh["session_id"], threading.Event()).wait(15)
    fresh_peer = Peer()
    assert "result" in activate(fresh_peer, fresh)
    control(supervisor, fresh, "lease_barrier")
    assert "result" in invoke(fresh_peer, fresh, "prompt.submit", text="accepted but must not execute")
    for _ in range(150):
        if control(supervisor, fresh, "inspect")["lease_wait"]:
            break
        threading.Event().wait(.02)
    assert control(supervisor, fresh, "inspect")["lease_wait"]
    with server._sessions_lock:
        server._sessions[fresh["session_id"]] = dict(fresh_record)
    control(supervisor, fresh, "lease_release")
    for _ in range(150):
        observation = control(supervisor, fresh, "inspect")
        if not observation["running"]:
            break
        threading.Event().wait(.02)
    assert not observation["running"] and observation["holder"] is None
    assert observation["calls"] == ["warm"]
    assert len([row for row in observation["messages"] if row["role"] == "user" and row["content"] == "accepted but must not execute"]) == 1
