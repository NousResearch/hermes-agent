"""Host-account admission contracts, with no live authority or model requests."""

import json
import multiprocessing
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

from tools import host_policy
from tools.host_policy import HostPolicyError
from tools.worker_admission import AdmissionError, Reservation, admission_status, reserve_workers

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def authority(tmp_path, monkeypatch):
    root = (tmp_path / "authority").resolve()
    root.mkdir(mode=0o700)
    monkeypatch.setattr(host_policy, "policy_root", lambda: root)
    publish(root)
    return root


def publish(root, *, mode="enforce", limit=2, revision="test-v1"):
    policy = {"version": 1, "revision": revision, "admission": {"mode": mode, "max_workers": limit}, "approvals": {}}
    host_policy.atomic_private_json(root / "policy.json", policy)
    if not (root / "ledger.json").exists():
        host_policy.atomic_private_json(root / "ledger.json", {"version": 1, "reservations": {}})
    with host_policy.policy_lock():
        pass


def test_unmanaged_authority_does_not_create_files(tmp_path):
    assert reserve_workers(3) == []
    assert admission_status() == {"installed": False, "mode": "unmanaged"}
    assert not (tmp_path / "host-policy").exists()


def test_prospective_batch_is_all_or_none_and_cross_profile(authority):
    first = reserve_workers(1, profile="alpha", session="one")[0]
    with pytest.raises(AdmissionError, match="capacity") as error:
        reserve_workers(2, profile="beta", session="two")
    assert error.value.metadata() == {"code": "worker_capacity", "occupancy": 1, "limit": 2}
    second = reserve_workers(1, profile="beta", session="two")[0]
    assert admission_status()["reserved_workers"] == 2
    first.finish()
    first.finish()  # idempotent completion cannot release the sibling
    assert admission_status()["reserved_workers"] == 1
    second.cancel_unstarted()
    assert admission_status()["reserved_workers"] == 0


def test_concurrent_batches_cannot_over_admit(authority):
    barrier = threading.Barrier(3)

    def attempt():
        barrier.wait(timeout=20)
        try:
            return reserve_workers(2)
        except AdmissionError:
            return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt) for _ in range(2)]
        barrier.wait(timeout=20)
        admitted = [future.result(timeout=20) for future in futures]
    assert sorted(map(len, admitted)) == [0, 2]
    assert admission_status()["reserved_workers"] == 2
    for batch in admitted:
        for reservation in batch:
            reservation.finish()


def test_timeout_result_retains_capacity_until_both_execution_and_cleanup_exit(authority):
    reservation = reserve_workers(1)[0]
    reservation.claim()
    conversation, cleanup = Future(), Future()
    reservation.finish([conversation, cleanup])
    assert admission_status()["reserved_workers"] == 1
    conversation.set_result(None)
    assert admission_status()["reserved_workers"] == 1
    cleanup.set_result(None)
    assert admission_status()["reserved_workers"] == 0


def test_duplicate_claim_and_foreign_process_claim_are_rejected(authority):
    reservation = reserve_workers(1)[0]
    counterfeit = Reservation(reservation.key, reservation.generation, {**reservation.owner, "pid": os.getpid() + 1})
    with pytest.raises(AdmissionError, match="mismatch"):
        counterfeit.claim()
    reservation.claim()
    with pytest.raises(AdmissionError, match="Duplicate"):
        reservation.claim()
    assert not reservation.cancel_unstarted()
    reservation.finish()


def test_lower_limit_drains_without_erasing_existing_reservations(authority):
    batch = reserve_workers(2)
    with host_policy.policy_lock():
        publish_policy = {"version": 1, "revision": "lower", "admission": {"mode": "enforce", "max_workers": 1}, "approvals": {}}
        host_policy.atomic_private_json(authority / "policy.json", publish_policy)
    with pytest.raises(AdmissionError):
        reserve_workers(1)
    assert admission_status()["reserved_workers"] == 2
    batch[0].finish()
    with pytest.raises(AdmissionError):
        reserve_workers(1)
    batch[1].finish()
    reserve_workers(1)[0].finish()


def test_observe_counts_but_does_not_enforce(authority, caplog):
    publish(authority, mode="observe", limit=1)
    batch = reserve_workers(3)
    assert len(batch) == 3
    assert admission_status()["mode"] == "observe"
    assert "Observe-only" in caplog.text
    for reservation in batch:
        reservation.finish()


@pytest.mark.parametrize("field,value", [("version", True), ("version", 2), ("revision", ""), ("admission", None)])
def test_invalid_installed_policy_fails_closed(authority, field, value):
    policy = json.loads((authority / "policy.json").read_text())
    policy[field] = value
    host_policy.atomic_private_json(authority / "policy.json", policy)
    with pytest.raises(HostPolicyError):
        reserve_workers(1)


@pytest.mark.parametrize("limit", [True, 0, -1, "2", 2.5])
def test_invalid_limit_never_coerces(authority, limit):
    publish(authority, limit=limit)
    with pytest.raises(HostPolicyError):
        reserve_workers(1)


def test_missing_policy_or_ledger_never_resets_installed_capacity(authority):
    reservation = reserve_workers(1)[0]
    (authority / "policy.json").unlink()
    with pytest.raises(HostPolicyError):
        reserve_workers(1)
    publish(authority)
    (authority / "ledger.json").unlink()
    with pytest.raises(AdmissionError, match="missing"):
        reserve_workers(1)
    # Do not call finish against a deliberately damaged ledger: repair requires an owner.
    assert reservation.key


def test_corrupt_and_foreign_namespace_records_retain_capacity(authority):
    reservation = reserve_workers(1)[0]
    ledger = json.loads((authority / "ledger.json").read_text())
    ledger["reservations"][reservation.key]["owner"]["namespace"] = "foreign-namespace"
    host_policy.atomic_private_json(authority / "ledger.json", ledger)
    second = reserve_workers(1)[0]
    with pytest.raises(AdmissionError):
        reserve_workers(1)
    second.finish()
    ledger["reservations"][reservation.key]["owner"].pop("birth")
    host_policy.atomic_private_json(authority / "ledger.json", ledger)
    with pytest.raises(AdmissionError, match="identity"):
        reserve_workers(1)


def test_symlink_or_nonprivate_policy_is_rejected(authority, tmp_path):
    target = tmp_path / "policy-copy"
    (authority / "policy.json").rename(target)
    (authority / "policy.json").symlink_to(target)
    with pytest.raises(HostPolicyError):
        reserve_workers(1)
    (authority / "policy.json").unlink()
    target.rename(authority / "policy.json")
    (authority / "policy.json").chmod(0o644)
    with pytest.raises(HostPolicyError):
        reserve_workers(1)


def _process_attempt(root, start, results, release, crash=False):
    # Spawned processes do not inherit pytest's monkeypatches.
    from tools import host_policy as policy_module
    policy_module.policy_root = lambda: Path(root)
    start.wait(timeout=20)
    try:
        reservation = reserve_workers(1, profile=f"profile-{os.getpid()}")[0]
    except AdmissionError:
        results.put("rejected")
        return
    reservation.claim()
    results.put("admitted")
    if crash:
        return  # process exit is the only recovery evidence; no explicit release
    release.wait(timeout=20)
    reservation.finish()


def test_separate_processes_share_cap_and_dead_owner_is_recovered(authority):
    ctx = multiprocessing.get_context("spawn")
    start, release, results = ctx.Event(), ctx.Event(), ctx.Queue()
    processes = [ctx.Process(target=_process_attempt, args=(str(authority), start, results, release)) for _ in range(4)]
    try:
        for process in processes:
            process.start()
        start.set()
        outcomes = [results.get(timeout=20) for _ in processes]
        assert outcomes.count("admitted") == 2
        assert outcomes.count("rejected") == 2
        assert admission_status()["reserved_workers"] == 2
    finally:
        release.set()
        for process in processes:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    assert all(process.exitcode == 0 for process in processes)
    assert admission_status()["reserved_workers"] == 0
    process = ctx.Process(target=_process_attempt, args=(str(authority), start, results, release, True))
    process.start()
    assert results.get(timeout=20) == "admitted"
    process.join(timeout=20)
    assert process.exitcode == 0
    # The process deliberately left a record; fresh admission proves death and reaps it.
    assert admission_status()["reserved_workers"] == 1
    reservation = reserve_workers(2)
    assert admission_status()["reserved_workers"] == 2
    for item in reservation:
        item.finish()
