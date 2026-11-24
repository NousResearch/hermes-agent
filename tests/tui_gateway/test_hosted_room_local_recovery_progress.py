"""Confirmed local recovery remains visible without replaying the admitted turn."""

from gateway import hosted_room_driver as driver
from tests.tui_gateway.test_hosted_room_retry_progress import _deferred_room, _worker_row, store_fixture
from tui_gateway.hosted_room_member_activity import task_is_active

assert store_fixture  # shared isolated persistence fixture


def test_recovery_shows_confirmed_local_activity_without_replaying_the_turn(tmp_path, monkeypatch, store):
    service, task_id, lease = _deferred_room(tmp_path, monkeypatch, store)
    [task] = driver.list_tasks(service.store, placement=service.placement, room_id="room", status="deferred")
    driver.reopen_deferred_task(
        service.store, task["identity"], lease, placement=service.placement,
        expected_execution_generation=1, expected_cancel_generation=0, clock=service.runtime.clock)
    info = {"active": True, "task_id": task_id, "execution_generation": 1,
            "tool": "terminal", "progress_at": 1234.5}
    service.runtime._report_pending_action(task, session_id="worker-session", info=info)

    assert task_is_active("room", task_id, 1) is True
    status = service.status("room")
    assert status["working"] is True
    assert _worker_row(service)["tool"] == "terminal"
    assert status["counts"] == {"indeterminate": 1}
    [retained] = driver.list_tasks(service.store, placement=service.placement, room_id="room")
    assert retained["execution_generation"] == 1

    for update in ({"active": False}, {"execution_generation": 99}, {"task_id": "another-task"}):
        service.runtime._report_pending_action(task, session_id="worker-session", info={**info, **update})
        assert service.status("room")["working"] is False
        assert service.status("room")["member_activity"] == []
