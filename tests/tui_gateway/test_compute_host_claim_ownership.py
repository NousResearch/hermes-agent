"""A delayed compute-host Stop or completion cannot change a replacement turn."""

import threading
import types

import pytest

from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry
from tui_gateway import server
from tui_gateway import server_requests
from tests.tui_gateway.test_turn_claim_interrupt import _idle_session


@pytest.mark.parametrize("replacement_phase, old_result", [("pending", None), ("approval", "deny")])
def test_compute_host_stop_fallback_preserves_successor_requests(monkeypatch, replacement_phase, old_result):
    sid = "host-fallback-claim"
    session = _idle_session()
    session.update(_compute_host_active=True, running=True, _turn_claim=1)
    session["inflight_turn"] = {"user": "old"}
    key = session["session_key"]
    old = _ApprovalEntry({"request_id": "old-approval", "command": "old"})
    successor = _ApprovalEntry({"request_id": "new-approval", "command": "new"})
    monkeypatch.setitem(approval._gateway_queues, key, [old])
    monkeypatch.setattr(server_requests, "_answerable", lambda _sid: True)
    monkeypatch.setattr(server_requests, "_write", lambda _frame: None)
    monkeypatch.setattr(server_requests, "_emit", lambda *_a: None)
    server_requests.send_async("sudo", sid, {}, lambda _result: None)
    successor_requests = []

    def replace_claim():
        with session["history_lock"]:
            session["running"] = False
            server._claim_session_turn(session)
            session["_turn_cancel_requested"] = False
        previous_requests = {request["id"] for request in server_requests.open_requests(sid)}
        server_requests.send_async("sudo", sid, {}, lambda _result: None)
        successor_requests.extend(request["id"] for request in server_requests.open_requests(sid)
                                  if request["id"] not in previous_requests)
        approval._gateway_queues.setdefault(key, []).append(successor)

    clear_pending = server._clear_pending

    def clear_captured(*args, **kwargs):
        if replacement_phase == "pending":
            replace_claim()
        clear_pending(*args, **kwargs)

    resolve_approval = approval.resolve_gateway_approval

    def resolve_captured(*args, **kwargs):
        if replacement_phase == "approval":
            replace_claim()
        return resolve_approval(*args, **kwargs)

    class Host:
        def interrupt(self, _sid, **_kwargs):
            raise RuntimeError("host died")

        def has_pending_turn(self, _sid):
            return False

    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *_a: Host())
    monkeypatch.setattr(server, "_clear_pending", clear_captured)
    monkeypatch.setattr(approval, "resolve_gateway_approval", resolve_captured)
    server._sessions[sid] = session
    try:
        response = server.handle_request(
            {"id": "stop", "method": "session.interrupt", "params": {"session_id": sid}})
        assert response is not None
        outcome = {
            "response": response["result"], "claim": session["_turn_claim"],
            "cancelled": session["_turn_cancel_requested"],
            "requests": [request["id"] for request in server_requests.open_requests(sid)],
            "approvals": approval.list_gateway_approvals(key), "old_result": old.result,
        }
    finally:
        server._sessions.pop(sid, None)
        server_requests.cancel(sid)
    expected_approvals = [old.data, successor.data] if old_result is None else [successor.data]
    assert outcome == {
        "response": {"status": "interrupted", "turn_isolation": True},
        "claim": 2, "cancelled": False, "requests": successor_requests,
        "approvals": expected_approvals, "old_result": old_result,
    }


def test_compute_host_stop_cannot_cancel_a_successor(monkeypatch):
    session = _idle_session()
    session.update(_compute_host_active=True, running=True, _turn_claim=1)
    session["inflight_turn"] = {"user": "old"}
    next_inflight, next_queue = {"user": "new"}, {"text": "next"}

    class Host:
        def interrupt(self, _sid, **_kwargs):
            with session["history_lock"]:
                session["running"] = False
                server._claim_session_turn(session)
                session.update(inflight_turn=next_inflight, queued_prompt=next_queue, _turn_cancel_requested=False)
            raise RuntimeError("host died")

        def has_pending_turn(self, _sid):
            return False

    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *_a: Host())
    server._sessions["host-claim"] = session
    try:
        response = server.handle_request(
            {"id": "stop", "method": "session.interrupt", "params": {"session_id": "host-claim"}})
        outcome = {
            "response": response["result"], "running": session["running"],
            "claim": session["_turn_claim"], "cancelled": session.get("_turn_cancel_requested"),
            "inflight": session["inflight_turn"], "queued": session.get("queued_prompt"),
        }
    finally:
        server._sessions.pop("host-claim", None)
    assert outcome == {
        "response": {"status": "interrupted", "turn_isolation": True}, "running": True,
        "claim": 2, "cancelled": False, "inflight": next_inflight, "queued": next_queue,
    }


def test_delayed_compute_host_completion_cannot_clear_successor(monkeypatch):
    session = _idle_session()
    session.update(running=True, _turn_claim=1)
    pending, events = [], []
    admitted, release_callback = threading.Event(), threading.Event()
    original_done = server._on_compute_host_turn_done

    def delayed_done(*args, **kwargs):
        admitted.set()
        assert release_callback.wait(10)
        original_done(*args, **kwargs)

    host = types.SimpleNamespace(submit_turn=lambda _frame, *, on_complete: pending.append(on_complete))
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *_a: host)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_compute_host_turn_frame", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_on_compute_host_turn_done", delayed_done)
    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a: None)
    monkeypatch.setattr(server, "_compute_host_session_info", lambda _session: {})
    server._submit_prompt_to_compute_host("old", "host-claim", session, "old")
    worker = threading.Thread(target=pending[0], args=({"type": "turn.end", "session_key": "obsolete-key"},))
    worker.start()
    next_inflight = {"user": "new"}
    try:
        assert admitted.wait(10)
        with session["history_lock"]:
            session["running"] = False
            server._claim_session_turn(session)
            session["inflight_turn"] = next_inflight
            session["pending_model_switch"] = {"model": "new-model"}
        release_callback.set()
        worker.join(10)
        assert not worker.is_alive()
        outcome = {
            "running": session["running"], "claim": session["_turn_claim"],
            "inflight": session["inflight_turn"], "model": session.get("pending_model_switch"),
            "key": session["session_key"], "events": events,
        }
    finally:
        release_callback.set()
        worker.join(10)
    assert outcome == {
        "running": True, "claim": 2, "inflight": next_inflight, "model": {"model": "new-model"},
        "key": "turn-claim-key", "events": [],
    }
