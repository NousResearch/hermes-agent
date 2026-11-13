"""Cooperative OS-account worker admission; count children, never async calls.

Lock order: authority transaction precedes no other Hermes lock. Callers must
reserve before construction and must not wait for workers under the authority.
Uncertain death retains capacity; there is no time-based lease expiry or replay.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import uuid

import psutil

from hermes_platform.host.identity import execution_owner, owner_proven_gone
from tools import host_policy
from tools.host_policy import HostPolicyError

logger = logging.getLogger(__name__)


class AdmissionError(HostPolicyError):
    def __init__(self, message: str, *, code: str = "worker_admission_unavailable", occupancy=None, limit=None):
        super().__init__(message)
        self.code, self.occupancy, self.limit = code, occupancy, limit

    def metadata(self) -> dict:
        return {"code": self.code, "occupancy": self.occupancy, "limit": self.limit}


def _finite_positive(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _current_owner() -> dict:
    try:
        owner = execution_owner()
    except (psutil.Error, OSError, KeyError) as exc:
        raise AdmissionError("Cannot establish this worker's process identity") from exc
    if not _valid_owner(owner):
        raise AdmissionError("Cannot establish this worker's process identity")
    return owner


def _valid_owner(owner) -> bool:
    return (
        isinstance(owner, dict) and set(owner) == {"pid", "birth", "namespace"}
        and type(owner["pid"]) is int and owner["pid"] > 0
        and _finite_positive(owner["birth"])
        and isinstance(owner["namespace"], str) and bool(owner["namespace"])
    )


def _ledger(*, allow_missing: bool = False) -> dict:
    path = host_policy.policy_root() / "ledger.json"
    try:
        path.lstat()
    except FileNotFoundError:
        if allow_missing:
            return {"version": 1, "reservations": {}}
        raise AdmissionError("Worker ledger is missing; qualify observe-only accounting before enforcing")
    data = host_policy.read_private_json(path)
    if (set(data) != {"version", "reservations"} or type(data.get("version")) is not int
            or data["version"] != 1 or not isinstance(data.get("reservations"), dict)):
        raise AdmissionError("Invalid worker ledger; repair it without discarding live reservations")
    for key, item in data["reservations"].items():
        if not (
            isinstance(key, str) and key and isinstance(item, dict)
            and set(item) == {"generation", "owner", "state", "profile", "session", "revision"}
            and isinstance(item.get("generation"), str) and item["generation"]
            and _valid_owner(item.get("owner")) and isinstance(item.get("state"), str)
            and item["state"] in {"reserved", "running"}
            and all(isinstance(item.get(field), str) for field in ("profile", "session", "revision"))
        ):
            raise AdmissionError("Invalid worker identity in ledger; capacity retained pending repair")
    return data


def _save(data: dict) -> None:
    host_policy.atomic_private_json(host_policy.policy_root() / "ledger.json", data)


def _reap(data: dict) -> None:
    reservations = data["reservations"]
    for key, item in list(reservations.items()):
        if owner_proven_gone(item["owner"]):
            del reservations[key]


def _scope(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def reserve_workers(count: int, *, profile: str = "", session: str = "") -> list[Reservation]:
    if type(count) is not int or count < 1:
        raise AdmissionError("Worker batch size must be a positive integer")
    policy = host_policy.load_host_policy()
    if policy is None or policy["admission"]["mode"] == "disabled":
        return []
    owner = _current_owner()
    with host_policy.policy_lock(create=False):
        policy = host_policy.load_host_policy()
        if policy is None:
            raise AdmissionError("Host policy disappeared before admission")
        if policy["admission"]["mode"] == "disabled":
            return []
        data = _ledger(allow_missing=policy["admission"]["mode"] == "observe")
        _reap(data)
        used, limit = len(data["reservations"]), policy["admission"]["max_workers"]
        if used + count > limit and policy["admission"]["mode"] == "enforce":
            raise AdmissionError(
                f"Delegated-worker capacity reached ({used}/{limit}; requested {count}). "
                "Wait for actual workers to exit; changing execution mode does not bypass this limit.",
                code="worker_capacity", occupancy=used, limit=limit,
            )
        reservations = []
        for _ in range(count):
            key, generation = uuid.uuid4().hex, uuid.uuid4().hex
            data["reservations"][key] = {
                "generation": generation, "owner": owner, "state": "reserved",
                "profile": _scope(profile), "session": _scope(session), "revision": policy["revision"],
            }
            reservations.append(Reservation(key, generation, owner))
        _save(data)
    if used + count > limit:
        logger.warning("Observe-only worker admission: %d existing + %d requested exceeds %d", used, count, limit)
    return reservations


def admission_status() -> dict:
    """Read-only allowlisted projection; does not reap or rewrite the ledger."""
    policy = host_policy.load_host_policy()
    if policy is None:
        return {"installed": False, "mode": "unmanaged"}
    if policy["admission"]["mode"] == "disabled" and not (host_policy.policy_root() / "ledger.json").exists():
        return {
            "installed": True, "revision": policy["revision"], **policy["admission"],
            "reserved_workers": None, "scope": "host_os_account", "accounting": "disabled",
        }
    with host_policy.policy_lock(create=False):
        policy = host_policy.load_host_policy()
        if policy is None:
            raise AdmissionError("Host policy disappeared during diagnostics")
        data = _ledger(allow_missing=policy["admission"]["mode"] in {"observe", "disabled"})
        return {"installed": True, "revision": policy["revision"], **policy["admission"],
                "reserved_workers": len(data["reservations"]), "scope": "host_os_account"}


class Reservation:
    """Process-bound, single-use claim retained through lifecycle AND inner Futures."""

    def __init__(self, key: str, generation: str, owner: dict):
        self.key, self.generation, self.owner = key, generation, owner
        self._lock = threading.Lock()
        self._claimed = self._finished = self._released = False
        self._pending = 0

    def _entry(self, data: dict) -> dict:
        entry = data["reservations"].get(self.key)
        if (not isinstance(entry, dict) or entry.get("generation") != self.generation
                or entry.get("owner") != self.owner or _current_owner() != self.owner):
            raise AdmissionError("Worker claim owner/generation mismatch; no work was started")
        return entry

    def claim(self) -> None:
        with self._lock:
            if self._claimed or self._finished:
                raise AdmissionError("Duplicate or cancelled worker claim")
            with host_policy.policy_lock(create=False):
                # A malformed replacement cannot authorize new execution.
                host_policy.load_host_policy()
                data = _ledger()
                entry = self._entry(data)
                if entry["state"] != "reserved":
                    raise AdmissionError("Worker reservation was already claimed")
                entry["state"] = "running"
                _save(data)
            self._claimed = True

    def finish(self, futures=()) -> None:
        """Caller has completed cleanup; Futures may still own real conversation work."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            pending = [future for future in futures if future is not None]
            self._pending = len(pending)
        # add_done_callback may execute synchronously: do not hold the local lock.
        for future in pending:
            future.add_done_callback(self._future_done)
        self._release_if_quiescent()

    def _future_done(self, _future) -> None:
        with self._lock:
            self._pending -= 1
        self._release_if_quiescent()

    def _release_if_quiescent(self) -> None:
        with self._lock:
            if not self._finished or self._pending or self._released:
                return
            with host_policy.policy_lock(create=False):
                data = _ledger()
                self._entry(data)
                del data["reservations"][self.key]
                _save(data)
            self._released = True

    def cancel_unstarted(self) -> bool:
        """A local cancellation serializes with claim; never release a running child."""
        with self._lock:
            if self._claimed or self._finished:
                return False
            self._finished = True
        self._release_if_quiescent()
        return True
