"""Durable provenance uses the producer's exact run fence, never reason text."""
from dataclasses import asdict

from gateway import hosted_room_driver as state
from tests.tui_gateway.test_hosted_room_backoff_scheduler import fixture


def test_only_proven_producer_mints_fenced_disposition_and_requeue_consumes_it(tmp_path):
    db, binding, now, rpc, runtime, publications, identity, admit = fixture(tmp_path, member=True)
    original = state.get_task(db, identity)
    runtime._run_cycle()
    saved = state.get_task(db, identity)
    proof = (saved["result"] or {}).get("nonadmission")
    assert isinstance(proof, dict), saved
    assert proof["disposition"] == "proven_nonadmission"
    assert proof["identity"] == asdict(identity)
    assert proof["execution_generation"] == saved["execution_generation"] == 1
    assert proof["cancel_generation"] == saved["cancel_generation"] == 0
    assert proof["run_gateway_id"] == saved["run_gateway_id"]
    assert proof["run_process_generation"] == saved["run_process_generation"]
    assert proof["run_lease_generation"] == saved["run_lease_generation"]
    assert state.is_proven_nonadmission(saved)
    runtime.retry_indeterminate(identity)
    queued = state.get_task(db, identity)
    assert queued["execution_generation"] == 1 and queued["result"] is None
    assert not state.is_proven_nonadmission(queued)
    assert queued["payload"] == original["payload"]


def test_unknown_producer_with_same_reason_never_mints_nonadmission(tmp_path):
    db, binding, now, rpc, runtime, publications, identity, admit = fixture(tmp_path, member=True)
    lease = runtime._ensure_lease(binding)
    state.start_task(db, identity, lease, expected_cancel_generation=0, clock=lambda: now[0])
    now[0] += runtime.lease_ttl_seconds + 1
    successor = state.acquire_lease(db, room_id=binding.room_id, gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation="successor", ttl_seconds=60,
        clock=lambda: now[0])
    state.recover_room(db, successor, clock=lambda: now[0])
    saved = state.defer_indeterminate_task(db, identity, successor, expected_execution_generation=1,
        expected_cancel_generation=0, reason="member_unavailable", clock=lambda: now[0])
    assert saved["result"] == {"reason": "member_unavailable", "retryable": True}
    assert not state.is_proven_nonadmission(saved)
