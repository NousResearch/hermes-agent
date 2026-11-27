"""Every spelling of a group chat orders Stop with validated approval dispatch."""
import threading

from gateway import hosted_room_driver as driver
from tests.tui_gateway.test_hosted_room_service import _FakeRPC
from tests.tui_gateway.test_hosted_room_approval_lifetime import pending_approval as pending_approval_fixture

pending_approval = pending_approval_fixture


class _SignallingLock:
    """Signal that a contender reached the held lock before its owner proceeds."""
    def __init__(self, contended):
        self._lock = threading.RLock()
        self.contended = contended

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()


def test_allow_paused_before_dispatch_is_ordered_with_a_concurrent_stop(pending_approval, monkeypatch):
    """Stop retires the cached approval, so only a pause after validation proves the lock's ordering."""
    service, task, _lease = pending_approval
    statuses_at_dispatch = []

    class _RecordingRPC(_FakeRPC):
        def approve(self, **kwargs):
            statuses_at_dispatch.append(
                driver.get_task(service.store, task["identity"], placement=service.placement)["status"])
            return super().approve(**kwargs)

    service.rpc = service.runtime.rpc = _RecordingRPC()
    for hook in ("_interrupt_stopping_task", "_settle_stopping_completion", "_peer_stop_acknowledged"):
        monkeypatch.setattr(service.runtime, hook, lambda *_args: False)
    validated, stop_progress = threading.Event(), threading.Event()
    planted = service._approval_stop_locks["room"] = _SignallingLock(stop_progress)
    dispatch = service._dispatch_approval

    def paused_dispatch(*args, **kwargs):
        validated.set()
        assert stop_progress.wait(10), "Stop neither finished nor waited for this Allow"
        return dispatch(*args, **kwargs)

    monkeypatch.setattr(service, "_dispatch_approval", paused_dispatch)
    stopper = threading.Thread(target=lambda: (validated.wait(10), service.stop_room("room", cancel_id="stop"),
                                               stop_progress.set()))
    stopper.start()
    service.approve_room_task(
        "room", member_id="ops", task_id=task["identity"].task_id,
        execution_generation=task["execution_generation"], request_id="approval", choice="once")
    stopper.join(10)
    assert not stopper.is_alive()
    assert service._approval_stop_locks.get("room") is planted
    assert statuses_at_dispatch == ["running"]
    assert driver.get_task(service.store, task["identity"], placement=service.placement)["status"] == "stopping"


def test_every_spelling_of_a_room_shares_its_stop_lock_and_cached_approvals(pending_approval):
    """Storage strips room ids, so a padded id names the same room: same lock, same approvals."""
    service, _task, _lease = pending_approval
    stop_progress, done = threading.Event(), threading.Event()
    held = service._approval_stop_locks["room"] = _SignallingLock(stop_progress)
    try:
        with held:
            stopper = threading.Thread(target=lambda: (service.stop_room("  room ", cancel_id="stop"),
                                                       done.set(), stop_progress.set()))
            stopper.start()
            assert stop_progress.wait(10), "Stop neither finished nor waited for the room lock"
            assert not done.is_set(), "a padded room id took another lock"
    finally:
        stopper.join(10)
    assert done.is_set()
    assert ("room", "ops") not in service._pending_actions
    del held
    assert "room" not in service._approval_stop_locks, "an idle room keeps no lock"
