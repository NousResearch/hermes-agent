"""Bounded private authority reads and conservative process-death recovery."""

import json
import multiprocessing
import os
from pathlib import Path

import psutil
import pytest

from hermes_platform.host import identity
from tools import host_policy, worker_admission
from tools.host_policy import HostPolicyError
from tools.worker_admission import AdmissionError, admission_status, reserve_workers

pytestmark = pytest.mark.platforms("posix")
_canonical_policy_root = host_policy.policy_root


@pytest.fixture
def authority(tmp_path, monkeypatch):
    root = (tmp_path / "authority").resolve()
    root.mkdir(mode=0o700)
    monkeypatch.setattr(host_policy, "policy_root", lambda: root)
    host_policy.atomic_private_json(root / "policy.json", {
        "version": 1, "revision": "safety-v1",
        "admission": {"mode": "enforce", "max_workers": 1}, "approvals": {},
    })
    host_policy.atomic_private_json(root / "ledger.json", {"version": 1, "reservations": {}})
    with host_policy.policy_lock():
        pass
    return root


def _probe_admission(root, result):
    host_policy.policy_root = lambda: Path(root)
    try:
        reserve_workers(1)
    except HostPolicyError:
        result.put("refused")


@pytest.mark.parametrize("name", ["policy.json", "ledger.json"])
def test_fifo_state_refuses_without_waiting_for_a_writer(authority, name):
    path = authority / name
    path.unlink()
    os.mkfifo(path, 0o600)
    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    child = ctx.Process(target=_probe_admission, args=(str(authority), result))
    child.start()
    try:
        child.join(timeout=10)
        assert not child.is_alive(), "authority read blocked waiting for a FIFO writer"
        assert child.exitcode == 0
        assert result.get(timeout=5) == "refused"
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)


def test_hardlinked_policy_is_not_a_private_authority(authority, tmp_path):
    os.link(authority / "policy.json", tmp_path / "alias")
    with pytest.raises(HostPolicyError):
        reserve_workers(1)


@pytest.mark.parametrize("name", ["policy.json", "ledger.json"])
def test_duplicate_json_fields_are_not_an_authority(authority, name):
    path = authority / name
    path.write_bytes(b'{"version":2,' + path.read_bytes()[1:])
    with pytest.raises(HostPolicyError):
        reserve_workers(1)


def test_missing_persistent_lock_is_not_silently_recreated(authority):
    (authority / ".admission.lock").unlink()
    with pytest.raises(HostPolicyError):
        reserve_workers(1)
    assert not (authority / ".admission.lock").exists()


@pytest.mark.parametrize("method", ["finish", "cancel_unstarted"])
def test_release_never_recreates_a_missing_persistent_lock(authority, method):
    reservation = reserve_workers(1)[0]
    before = (authority / "ledger.json").read_bytes()
    lock = authority / ".admission.lock"
    lock.unlink()
    with pytest.raises(HostPolicyError):
        getattr(reservation, method)()
    assert not lock.exists()
    assert (authority / "ledger.json").read_bytes() == before


def test_giant_birth_timestamp_is_a_structured_state_refusal(authority):
    reservation = reserve_workers(1)[0]
    ledger = json.loads((authority / "ledger.json").read_text())
    ledger["reservations"][reservation.key]["owner"]["birth"] = 10 ** 1000
    host_policy.atomic_private_json(authority / "ledger.json", ledger)
    with pytest.raises(AdmissionError, match="identity"):
        reserve_workers(1)
    assert json.loads((authority / "ledger.json").read_text()) == ledger


@pytest.mark.parametrize("error", [OSError("lookup unavailable"), psutil.AccessDenied(os.getpid())])
def test_unavailable_current_identity_refuses_without_changing_capacity(authority, monkeypatch, error):
    def unavailable():
        raise error

    monkeypatch.setattr(worker_admission, "execution_owner", unavailable)
    with pytest.raises(AdmissionError, match="process identity"):
        reserve_workers(1)
    assert admission_status()["reserved_workers"] == 0


def test_account_lookup_failure_is_structured(monkeypatch):
    def unavailable():
        raise KeyError("unmapped uid")

    monkeypatch.setattr(host_policy, "policy_root", unavailable)
    with pytest.raises(HostPolicyError):
        host_policy.load_host_policy()


@pytest.mark.parametrize("probe", ["access", "namespace", "namespace_process", "giant_pid", "recycled"])
def test_existing_identity_is_reaped_only_with_proof(authority, monkeypatch, probe):
    reservation = reserve_workers(1)[0]
    owner = dict(reservation.owner)
    if probe == "access":
        def birth(_pid):
            raise psutil.AccessDenied(_pid)
        monkeypatch.setattr(identity, "process_birth", birth)
    elif probe in {"namespace", "namespace_process"}:
        def namespace():
            if probe == "namespace_process":
                raise psutil.NoSuchProcess(os.getpid())
            raise OSError("cannot inspect PID namespace")
        monkeypatch.setattr(identity, "execution_namespace", namespace)
    elif probe == "giant_pid":
        ledger = json.loads((authority / "ledger.json").read_text())
        owner["pid"] = 10 ** 1000
        ledger["reservations"][reservation.key]["owner"] = owner
        host_policy.atomic_private_json(authority / "ledger.json", ledger)
    else:
        monkeypatch.setattr(identity, "process_birth", lambda _pid: reservation.owner["birth"] + 1)
    # Admission's current identity stays known; the existing-owner probe fails.
    monkeypatch.setattr(worker_admission, "execution_owner", lambda: reservation.owner)
    if probe == "recycled":
        assert identity.owner_proven_gone(reservation.owner)
        replacement = reserve_workers(1)[0]
        replacement.finish()
    else:
        assert not identity.owner_proven_gone(owner)
        with pytest.raises(AdmissionError, match="capacity"):
            reserve_workers(1)
        assert admission_status()["reserved_workers"] == 1


def test_approval_only_authority_reports_disabled_without_creating_accounting(authority):
    policy = json.loads((authority / "policy.json").read_text())
    policy["admission"]["mode"] = "disabled"
    host_policy.atomic_private_json(authority / "policy.json", policy)
    (authority / "ledger.json").unlink()
    (authority / ".admission.lock").unlink()
    before = {path.name: path.read_bytes() for path in authority.iterdir()}
    status = admission_status()
    assert status["mode"] == "disabled"
    assert status["reserved_workers"] is None
    assert reserve_workers(1) == []
    assert {path.name: path.read_bytes() for path in authority.iterdir()} == before


def test_canonical_authority_ignores_profile_and_environment_overrides(tmp_path, monkeypatch):
    import pwd
    from types import SimpleNamespace

    account = tmp_path / "os-account"
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: SimpleNamespace(pw_dir=str(account)))
    for key in ("HOME", "HERMES_HOME", "XDG_STATE_HOME", "HERMES_GATEWAY_LOCK_DIR"):
        monkeypatch.setenv(key, str(tmp_path / key))
    assert _canonical_policy_root() == account / ".local/state/hermes/host-policy"


def _hold_authority(root, ready, release):
    host_policy.policy_root = lambda: Path(root)
    with host_policy.policy_lock(create=False):
        ready.set()
        assert release.wait(20)


def test_other_process_lock_refuses_within_transaction_budget(authority, monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_authority, args=(str(authority), ready, release))
    child.start()
    try:
        assert ready.wait(10)
        monkeypatch.setattr(host_policy, "LOCK_TIMEOUT_SECONDS", 0.05)
        with pytest.raises(HostPolicyError, match="locked"):
            reserve_workers(1)
        assert json.loads((authority / "ledger.json").read_text())["reservations"] == {}
    finally:
        release.set()
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
    assert child.exitcode == 0
