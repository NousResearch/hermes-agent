"""Exact target-side interruption and recovery contracts."""

from pathlib import Path

from gateway import hosted_room_driver as state
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING,
    FakeSessionRPC,
    _admit,
    _identity,
    _runtime,
    _wait_for,
    db,  # noqa: F401
)


def _make_indeterminate(db: Path, identity, clock, *, ttl_seconds: float = 1):
    old_lease = state.acquire_lease(
        db,
        room_id=BINDING.room_id,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="old-process",
        ttl_seconds=ttl_seconds,
        clock=clock,
    )
    _admit(db, identity)
    attempt = state.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    return attempt


def _recover(db: Path, clock):
    lease = state.acquire_lease(
        db,
        room_id=BINDING.room_id,
        gateway_id=BINDING.gateway_id,
        authority_epoch=BINDING.authority_epoch,
        process_generation="recovery-process",
        ttl_seconds=30,
        clock=clock,
    )
    state.recover_room(db, lease, clock=clock)
    state.release_lease(db, lease, clock=clock)


def test_retry_uses_runtime_session_id_returned_by_resume(db: Path):
    identity = _identity()
    now = [100.0]
    clock = lambda: now[0]
    _make_indeterminate(db, identity, clock)
    now[0] = 102.0
    _recover(db, clock)

    rpc = FakeSessionRPC(auto_complete=False)
    stored_id = rpc.add_session(active=False, task_id=identity.task_id)
    runtime_id = "runtime-session"
    rpc.states[runtime_id] = rpc.states.pop(stored_id)

    def resume(**kwargs):
        rpc.calls.append(("resume", dict(kwargs)))
        return {"session_id": runtime_id}

    observed: dict[str, str] = {}
    original_history = rpc.history
    original_info = rpc.info

    def history(**kwargs):
        observed["history"] = kwargs["session_id"]
        return original_history(**kwargs)

    def info(**kwargs):
        observed["info"] = kwargs["session_id"]
        return original_info(**kwargs)

    rpc.resume = resume
    rpc.history = history
    rpc.info = info
    runtime = _runtime(db, rpc, clock=clock)

    assert runtime.retry_indeterminate(identity)["status"] == "queued"
    assert observed == {"history": runtime_id, "info": runtime_id}


def test_retry_reconciles_exact_target_cancellation_as_interruption(db: Path):
    identity = _identity()
    now = [100.0]
    clock = lambda: now[0]
    attempt = _make_indeterminate(db, identity, clock)
    now[0] = 102.0
    _recover(db, clock)
    peer_rpc = FakeSessionRPC(auto_complete=False)
    session_id = peer_rpc.add_session(active=False, task_id=identity.task_id)
    peer_rpc.states[session_id]["execution_generation"] = attempt.execution_generation
    original_info = peer_rpc.info

    def cancelled_info(**kwargs):
        return {
            **original_info(**kwargs),
            "status": "cancelled",
            "execution_generation": attempt.execution_generation,
        }

    peer_rpc.info = cancelled_info
    runtime = _runtime(
        db,
        FakeSessionRPC(auto_complete=False),
        clock=clock,
        transport_resolver=lambda _binding, _task: peer_rpc,
    )
    failed = runtime.retry_indeterminate(identity)

    assert failed["status"] == "failed"
    assert failed["execution_generation"] == attempt.execution_generation
    assert failed["result"]["reason_code"] == "target_interrupted"
    assert failed["cancel_id"] is None
    assert not [call for call in peer_rpc.calls if call[0] == "submit"]


def test_ambiguous_recovery_remains_indeterminate(db: Path):
    identity = _identity()
    now = [100.0]
    clock = lambda: now[0]
    _make_indeterminate(db, identity, clock, ttl_seconds=0.2)
    rpc = FakeSessionRPC(auto_complete=False)
    rpc.add_session(active=False, task_id=identity.task_id)
    now[0] = 101.0
    runtime = _runtime(db, rpc, clock=clock)

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "indeterminate")
    assert runtime.stop(timeout=5.0)
    assert not [call for call in rpc.calls if call[0] == "submit"]


def test_terminal_peer_cancellation_settles_immediately_without_home_stop(db: Path):
    identity = _identity()
    _admit(db, identity)

    class CancelledPeerRPC(FakeSessionRPC):
        def submit(self, **kwargs):
            self.auto_complete = False
            result = super().submit(**kwargs)
            with self._lock:
                self.states[kwargs["session_id"]]["active"] = False
            kwargs["on_terminal"](
                {
                    "status": "cancelled",
                    "task_id": kwargs["task"].task_id,
                    "execution_generation": kwargs["execution_generation"],
                    "target_interrupted": True,
                }
            )
            return result

    runtime = _runtime(
        db,
        CancelledPeerRPC(auto_complete=False),
        active_poll_interval_seconds=0.01,
        turn_timeout_seconds=1830,
    )
    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "failed")
    assert runtime.stop(timeout=5.0)

    failed = state.get_task(db, identity)
    assert failed["execution_generation"] == 1
    assert failed["cancel_id"] is None
    assert failed["result"] == {
        "error": "The Group Chat member turn was interrupted on its target gateway.",
        "reason_code": "target_interrupted",
    }
