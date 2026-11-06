"""Peer run progress belongs to the home's task generation, including recovery."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway.session_controls import AuthorityConnection
from tests.tui_gateway.test_hosted_room_peer_recovery import _crashed_turn, _UnreachablePeerClient
from tests.tui_gateway.test_hosted_room_retry_progress import _deferred_room, _worker_row
from tui_gateway.hosted_room_member_activity import forget_task_progress, task_progress
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient


def test_peer_poll_uses_the_task_generation_and_late_replaced_polls_stay_out(tmp_path, monkeypatch):
    service, task_id, lease = _deferred_room(tmp_path, monkeypatch)
    [old_task] = driver.list_tasks(service.db_path, room_id="room", status="deferred")
    client = PeerRunsHTTPClient(base_url="https://peer.example.test", api_key="")
    for generation in (1, 2):
        client._runs[(task_id, generation)] = {
            "run_id": f"run-{generation}", "session_id": "peer-session", "room_id": "room",
            "task_id": task_id, "execution_generation": generation, "target_profile": "default"}
    client._request = lambda path, **kwargs: {
        "run_id": path.rsplit("/", 1)[-1], "status": "running", "execution_generation": 91,
        "tool": "terminal" if path.endswith("1") else "web_search", "progress_at": 1234.5}

    def poll(generation):
        return client.status(room_id="room", profile="default", session_id="peer-session",
                             grant="signed.room.grant", task_id=task_id, execution_generation=generation)

    def record(task, info):
        service.runtime._report_pending_action(task, session_id="peer-session", info=info)

    old_info = poll(1)
    record(old_task, old_info)
    assert task_progress("room", task_id, 1) == {"tool": "terminal", "progress_at": 1234.5}
    service.retry_room_task("room", task_id=task_id, member_id="worker", execution_generation=1)
    driver.start_task(service.db_path, old_task["identity"], lease, expected_cancel_generation=0,
                      clock=service.runtime.clock)
    [current] = driver.list_tasks(service.db_path, room_id="room", status="running")
    assert current["execution_generation"] == 2
    record(old_task, old_info)
    record(current, old_info)  # a transport answer for a different attempt is rejected too
    assert "progress_at" not in _worker_row(service)
    record(current, poll(2))  # the peer's admission generation must not replace the home's
    forget_task_progress("room", task_id, 1)  # retirement may race the replacement's first frame
    record(old_task, old_info)
    assert {k: _worker_row(service)[k] for k in ("tool", "progress_at")} == {
        "tool": "web_search", "progress_at": 1234.5}


def _recovering_peer(tmp_path, monkeypatch):
    peer = _UnreachablePeerClient()
    turn = _crashed_turn(tmp_path, peer)
    turn.now[0] = 108.0
    turn.service.runtime._run_room_once(turn.binding)
    assert driver.get_task(turn.db, turn.identity)["status"] == "deferred"
    peer.reachable = True
    peer.history = lambda **kwargs: []
    info = {
        "active": True, "task_id": turn.identity.task_id, "execution_generation": 1,
        "tool": "terminal", "progress_at": 1234.5}
    peer.status = lambda **kwargs: dict(info)
    runtime_status = turn.service.runtime.status
    monkeypatch.setattr(turn.service.runtime, "status", lambda: {**runtime_status(), "running": True})
    turn.service.authorize_room = lambda *_args: None
    authority = SimpleNamespace(profile_id=str(tmp_path), instance_id="home", events={}, sessions={},
                                db=SimpleNamespace(db_path=turn.db), hosted_room_service=turn.service)
    connection = AuthorityConnection(authority, object(), {"user_id": "reader"})

    def state():
        answer = asyncio.run(connection.dispatch({
            "id": "state", "method": "groups.state", "params": {"room_id": "room-1"}}))
        assert "result" in answer, answer
        return answer["result"]["driver_status"]

    assert state()["member_activity"] == []
    retried = turn.service.retry_room_task("room-1", task_id=turn.identity.task_id, execution_generation=1)
    assert (retried["status"], retried["execution_generation"]) == ("indeterminate", 1)
    return turn, peer, info, state


def test_groups_state_shows_only_confirmed_active_recovered_peer_progress(tmp_path, monkeypatch):
    turn, peer, info, state = _recovering_peer(tmp_path, monkeypatch)
    [activity] = state()["member_activity"]
    assert activity == {"member_id": "member-peer", "status": "running", "thread_id": "thread-1",
                        # A retried attempt is timed from the retry (P2-2), not its first start at 100.
                        "round_index": 0, "started_at": 108.0, "tool": "terminal", "progress_at": 1234.5}
    assert state()["working"] is True and state()["counts"] == {"indeterminate": 1}
    for update in ({"active": False}, {"active": True, "execution_generation": 99}):
        info.update(update)
        turn.service.runtime._reconcile_indeterminate(turn.binding, turn.lease)
        assert state()["member_activity"] == []
    info.update(active=True, execution_generation=1, tool="web_search", progress_at=1240.0)
    turn.service.runtime._reconcile_indeterminate(turn.binding, turn.lease)
    assert state()["member_activity"][0]["tool"] == "web_search"
    peer.reachable = False
    turn.service.runtime._reconcile_indeterminate(turn.binding, turn.lease)
    assert state()["member_activity"] == []


def test_repeated_refused_retry_preserves_the_recovered_attempts_progress(tmp_path, monkeypatch):
    turn, _peer, info, state = _recovering_peer(tmp_path, monkeypatch)
    for progress_at in (1240.0, 1250.0):
        with pytest.raises(driver.InvalidTaskTransitionError, match="still active"):
            turn.service.retry_room_task("room-1", task_id=turn.identity.task_id, execution_generation=1)
        assert state()["member_activity"][0]["progress_at"] == info["progress_at"]
        info.update(tool="web_search", progress_at=progress_at)
        turn.service.runtime._reconcile_indeterminate(turn.binding, turn.lease)
        assert state()["member_activity"][0]["progress_at"] == progress_at
    assert state()["member_activity"][0]["tool"] == "web_search"
    assert {r["dispatch"]["execution_generation"] for r in _peer.recoveries} == {1}
