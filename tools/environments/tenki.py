"""Tenki cloud sandbox execution environment."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import re
import shlex
import tarfile
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

from hermes_constants import get_hermes_home
from tools.environments.base import (
    BaseEnvironment,
    _atomic_save_json_durable,
    _ThreadedProcessHandle,
    _windows_replace_file_write_through,
)
from tools.environments.file_sync import (
    FileSyncManager,
    _credential_host_paths,
    iter_sync_files,
    normalize_forward_env_names,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)
from tools.tenki_config import (
    resolve_tenki_api_endpoint,
    resolve_tenki_auth_token,
    resolve_tenki_workspace_id,
)

logger = logging.getLogger(__name__)
_SNAPSHOT_NAMESPACE = "direct"
_SNAPSHOT_RETIREMENT_NAMESPACE = "retire"
_SNAPSHOT_RETIRED_NAMESPACE = "retired"
_CREATE_ATTEMPT_NAMESPACE = "create-attempt"
_REMOTE_BINDING_NAMESPACE = "remote-binding"
_CREATE_ATTEMPT_EXPIRY_GRACE = 3600
_TENKI_CPU_RANGE = (1, 16)
_TENKI_MEMORY_MB_RANGE = (128, 65_536)
_TENKI_DISK_GB_RANGE = (5, 100)
_SNAPSHOT_LOCKS: dict[str, threading.RLock] = {}
_SNAPSHOT_LOCKS_GUARD = threading.Lock()
_QUARANTINED_TASK_OWNERSHIP_FILES: list[Any] = []
_QUARANTINED_TASK_OWNERSHIP_GUARD = threading.Lock()
_TERMINATE_RETRY_DELAYS = (0.1, 0.5)

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX
    _msvcrt = None


class _SnapshotPointerCommitUncertain(OSError):
    """The new pointer is visible, but its directory entry may not be durable."""

    def __init__(self, message: str):
        super().__init__(message)
        self.previous_snapshot_id: str | None = None
        self.new_snapshot_id: str | None = None


class _SnapshotPointerConflict(RuntimeError):
    """A stale writer attempted to replace newer or retired recovery state."""


@dataclass(frozen=True)
class _RemoteBinding:
    """One task's durable binding to its remote Tenki sandbox lineage.

    ``conflicted`` means a fork was positively observed and may never be
    forgotten by a later omission-prone listing; ``unresolvable`` additionally
    means at least one branch had no authoritative id, so no exact lookup can
    ever prove it terminated. The default instance is "no binding recorded".
    """

    remote_id: str | None = None
    attempt_id: str | None = None
    validated: bool = False
    conflicted: bool = False
    conflict_ids: tuple[str, ...] = ()
    unresolvable: bool = False


def _load_recovery_registry(path: Path) -> dict:
    """Load Tenki recovery state, failing closed when it is unreadable.

    Deliberately the opposite of :func:`base._load_json_store`, which returns
    ``{}`` on any read error: treating a malformed existing registry as empty
    would let the next read-modify-write erase the only snapshot pointer,
    create attempt, or exact remote binding. A missing file is the sole valid
    empty state.
    """
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Tenki recovery registry is unreadable: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            f"Tenki recovery registry must contain a JSON object: {path}"
        )
    return value


def _snapshot_store_path() -> Path:
    """Resolve the snapshot registry path for the *active* profile.

    Resolved per call (not frozen at import) so the multiplexing gateway,
    which overrides ``HERMES_HOME`` per turn, writes each profile's snapshot
    pointers into that profile's own home instead of whichever profile
    happened to import this module first.
    """
    return get_hermes_home() / "tenki_snapshots.json"


def _task_ownership_lock_path(profile_home: Path, task_id: str) -> Path:
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return profile_home / "locks" / "tenki" / f"{task_hash}.lock"


def _lock_open_file(lock_file: Any, *, blocking: bool, requirement: str) -> None:
    """Take one exclusive cross-process lock on *lock_file*'s first byte.

    *blocking* selects between waiting for the holder (the snapshot registry,
    which must serialize pointer RMW) and failing fast (task ownership, which
    must never wait on another Hermes process). Locking is mandatory on both
    paths: without a kernel lock two processes can each report success while
    silently losing one task's sole recovery pointer or forking its sandbox,
    so an unsupported platform fails closed.
    """
    if _fcntl is not None:
        flags = _fcntl.LOCK_EX if blocking else _fcntl.LOCK_EX | _fcntl.LOCK_NB
        _fcntl.flock(lock_file.fileno(), flags)
        return
    if _msvcrt is not None:
        # Windows byte-range locks require a real byte at the current file
        # position. Concurrent initializers may append more than one byte,
        # but every process locks byte zero.
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(" ")
            lock_file.flush()
        lock_file.seek(0)
        _msvcrt.locking(
            lock_file.fileno(),
            _msvcrt.LK_LOCK if blocking else _msvcrt.LK_NBLCK,
            1,
        )
        return
    raise RuntimeError(
        f"{requirement} requires fcntl or msvcrt cross-process file locking"
    )


def _unlock_open_file(lock_file: Any) -> None:
    """Best-effort kernel unlock; closing the file stays with the caller."""
    try:
        if _fcntl is not None:
            _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
        elif _msvcrt is not None:
            lock_file.seek(0)
            _msvcrt.locking(
                lock_file.fileno(),
                _msvcrt.LK_UNLCK,
                1,
            )
    except OSError:
        pass


def _acquire_task_ownership_lock(profile_home: Path, task_id: str):
    """Acquire one profile/task lifetime lock or fail without remote mutation."""
    lock_path = _task_ownership_lock_path(profile_home, task_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        _lock_open_file(
            lock_file,
            blocking=False,
            requirement="Tenki task ownership",
        )
    except (BlockingIOError, OSError) as exc:
        lock_file.close()
        raise RuntimeError(
            "Tenki sandbox task is already active in another Hermes process: "
            f"{task_id}"
        ) from exc
    except Exception:
        lock_file.close()
        raise
    return lock_file


def _release_task_ownership_lock(lock_file: Any) -> None:
    if lock_file is None:
        return
    try:
        _unlock_open_file(lock_file)
    finally:
        lock_file.close()


def _quarantine_task_ownership_lock(lock_file: Any) -> None:
    """Keep a failed task exclusively owned until this process exits."""
    if lock_file is None:
        return
    with _QUARANTINED_TASK_OWNERSHIP_GUARD:
        _QUARANTINED_TASK_OWNERSHIP_FILES.append(lock_file)


def _profile_token() -> str:
    """Canonical profile identity, stable across standalone/multiplex modes."""
    try:
        home = str(get_hermes_home().resolve())
    except Exception:
        home = str(get_hermes_home())
    return _profile_token_for_basis(f"home:{home}")


def _profile_token_for_basis(basis: str) -> str:
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]


def _legacy_profile_tokens() -> tuple[str, ...]:
    """Tokens emitted before profile identity was canonicalized to home."""
    try:
        home_path = get_hermes_home().resolve()
    except Exception:
        home_path = get_hermes_home()

    bases = [str(home_path)]  # old default-profile path basis
    if home_path.parent.name == "profiles":
        profile_name = home_path.name
    else:
        profile_name = os.getenv("HERMES_PROFILE", "").strip()
    if profile_name:
        bases.append(f"profile:{profile_name}")

    canonical = _profile_token()
    return tuple(
        token
        for token in dict.fromkeys(
            _profile_token_for_basis(basis) for basis in bases
        )
        if token != canonical
    )


def _load_snapshots(store_path: Path | None = None) -> dict:
    path = store_path or _snapshot_store_path()
    with _snapshot_store_lock(path):
        return _load_recovery_registry(path)


def _snapshot_platform() -> str:
    return os.name


@contextmanager
def _snapshot_store_lock(path: Path):
    """Serialize snapshot-pointer RMW in-process and across processes."""
    key = str(path)
    with _SNAPSHOT_LOCKS_GUARD:
        lock = _SNAPSHOT_LOCKS.setdefault(key, threading.RLock())
    with lock:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+", encoding="utf-8")
        os_locked = False
        try:
            _lock_open_file(
                lock_file,
                blocking=True,
                requirement="Tenki snapshot registry",
            )
            os_locked = True
            yield
        finally:
            if os_locked:
                _unlock_open_file(lock_file)
            lock_file.close()


def _atomic_save_snapshots(path: Path, data: dict) -> None:
    """Apply Tenki's recovery-pointer policy to the durable write primitive.

    The mechanism lives in ``base._atomic_save_json_durable``; this supplies the
    Tenki-specific policy: an uncertain commit surfaces as
    ``_SnapshotPointerCommitUncertain`` so callers keep both the old and the new
    remote snapshot alive, and the platform/Windows-replace hooks stay resolved
    from this module so they remain individually substitutable.
    """
    _atomic_save_json_durable(
        path,
        data,
        subject="snapshot-pointer",
        store_label="Tenki snapshot registry",
        commit_uncertain_error=_SnapshotPointerCommitUncertain,
        platform=_snapshot_platform(),
        replace_write_through=_windows_replace_file_write_through,
    )


@contextmanager
def _mutate_store(store_path: Path | None):
    """Read-modify-write the recovery registry under its cross-process lock.

    Resolves the active profile's store path, takes the lock, fail-closed loads
    the registry, yields it for mutation, and durably republishes it. The save
    runs on every normal exit *including an early ``return``* from the body:
    several callers deliberately republish unchanged state to re-establish
    durability after an uncertain commit. A body that raises skips the save and
    releases the lock, leaving the previous durable state visible.

    Callers whose no-op path must NOT write (or that need the save inside their
    own ``try``) cannot use this and manage the lock themselves.
    """
    path = store_path or _snapshot_store_path()
    with _snapshot_store_lock(path):
        snapshots = _load_recovery_registry(path)
        yield snapshots
        _atomic_save_snapshots(path, snapshots)


def _snapshot_key(task_id: str) -> str:
    return f"{_SNAPSHOT_NAMESPACE}:{task_id}"


def _snapshot_retirement_key(snapshot_id: str) -> str:
    return f"{_SNAPSHOT_RETIREMENT_NAMESPACE}:{snapshot_id}"


def _snapshot_retired_key(snapshot_id: str) -> str:
    return f"{_SNAPSHOT_RETIRED_NAMESPACE}:{snapshot_id}"


def _create_attempt_key(task_id: str) -> str:
    return f"{_CREATE_ATTEMPT_NAMESPACE}:{task_id}"


def _remote_binding_key(task_id: str) -> str:
    return f"{_REMOTE_BINDING_NAMESPACE}:{task_id}"


def _record_field(value: Any, field: str) -> Any:
    """Read *field* from a registry record, tolerating the bare-string shape.

    Early unpublished builds wrote a plain id string where the current format
    writes a dict, and a registry written by one of those builds must still be
    comparable rather than silently read as absent.
    """
    return value.get(field) if isinstance(value, dict) else value


def _create_attempt_state(
    task_id: str,
    store_path: Path | None = None,
) -> tuple[str | None, float | None]:
    snapshots = _load_snapshots(store_path)
    value = snapshots.get(_create_attempt_key(task_id))
    if isinstance(value, str) and value:
        return value, None
    if not isinstance(value, dict):
        return None, None
    attempt_id = value.get("attempt_id")
    expires_at = value.get("expires_at")
    return (
        attempt_id if isinstance(attempt_id, str) and attempt_id else None,
        float(expires_at) if isinstance(expires_at, (int, float)) else None,
    )


def _get_create_attempt(
    task_id: str,
    store_path: Path | None = None,
) -> str | None:
    # Test helper: no production caller. Kept as the readable projection the
    # suite asserts against.
    return _create_attempt_state(task_id, store_path)[0]


def _remote_binding_state(
    task_id: str,
    store_path: Path | None = None,
) -> _RemoteBinding:
    snapshots = _load_snapshots(store_path)
    value = snapshots.get(_remote_binding_key(task_id))
    if isinstance(value, str) and value:
        # This unversioned shape was used only by unpublished development
        # candidates. It cannot prove how ownership was established, so never
        # auto-adopt it as a supported Hermes lineage.
        return _RemoteBinding(
            remote_id=value,
            conflicted=True,
            conflict_ids=(value,),
        )
    if not isinstance(value, dict):
        return _RemoteBinding()
    remote_id = value.get("remote_id")
    attempt_id = value.get("attempt_id")
    conflict_ids = value.get("conflict_ids")
    parsed_conflict_ids = tuple(
        sorted({
            item
            for item in (
                conflict_ids if isinstance(conflict_ids, list) else []
            )
            if isinstance(item, str) and item
        })
    )
    parsed_remote_id = (
        remote_id if isinstance(remote_id, str) and remote_id else None
    )
    parsed_attempt_id = (
        attempt_id if isinstance(attempt_id, str) and attempt_id else None
    )
    conflicted = bool(value.get("conflicted", False))
    return _RemoteBinding(
        remote_id=parsed_remote_id,
        attempt_id=parsed_attempt_id,
        validated=(
            bool(value.get("validated", False))
            and parsed_attempt_id is not None
        ),
        conflicted=conflicted,
        conflict_ids=parsed_conflict_ids,
        unresolvable=bool(
            value.get("unresolvable", conflicted and not parsed_conflict_ids)
        ),
    )


def _get_remote_binding(
    task_id: str,
    store_path: Path | None = None,
) -> str | None:
    # Test helper: no production caller. Kept as the readable projection the
    # suite asserts against.
    return _remote_binding_state(task_id, store_path).remote_id


def _begin_create_attempt(
    task_id: str,
    attempt_id: str,
    expires_at: float,
    store_path: Path | None = None,
) -> None:
    """Durably journal one unique remote create before issuing its RPC."""
    with _mutate_store(store_path) as snapshots:
        key = _create_attempt_key(task_id)
        existing = snapshots.get(key)
        if existing not in (None, attempt_id):
            existing_id = _record_field(existing, "attempt_id")
            if existing_id != attempt_id:
                raise _SnapshotPointerConflict(
                    f"task {task_id} already has unresolved create attempt "
                    f"{existing_id}"
                )
        snapshots[key] = {
            "attempt_id": attempt_id,
            "expires_at": expires_at,
        }


def _clear_create_attempt(
    task_id: str,
    attempt_id: str,
    store_path: Path | None = None,
) -> None:
    """Durably clear exactly the create attempt whose remote is gone."""
    with _mutate_store(store_path) as snapshots:
        key = _create_attempt_key(task_id)
        existing = snapshots.get(key)
        if existing is None:
            # A prior removal may be visible after an uncertain directory/
            # write-through commit. Returning here still re-publishes that
            # marker-free state, so this retry establishes durability before
            # ownership can be released.
            return
        existing_id = _record_field(existing, "attempt_id")
        if existing_id != attempt_id:
            raise _SnapshotPointerConflict(
                f"create attempt advanced from {attempt_id} to {existing_id}"
            )
        snapshots.pop(key, None)


def _store_remote_binding(
    task_id: str,
    remote_id: str,
    attempt_id: str | None,
    *,
    validated: bool,
    store_path: Path | None = None,
) -> None:
    """Bind a task to one authoritative Tenki id before the remote is used."""
    if validated and attempt_id is None:
        raise ValueError(
            "a validated Tenki binding requires its durable create attempt id"
        )
    with _mutate_store(store_path) as snapshots:
        attempt_key = _create_attempt_key(task_id)
        existing_attempt = _record_field(
            snapshots.get(attempt_key),
            "attempt_id",
        )
        if attempt_id is not None and existing_attempt != attempt_id:
            raise _SnapshotPointerConflict(
                f"create attempt advanced from {attempt_id} to "
                f"{existing_attempt}"
            )
        binding_key = _remote_binding_key(task_id)
        existing_binding = _record_field(
            snapshots.get(binding_key),
            "remote_id",
        )
        if existing_binding not in (None, remote_id):
            raise _SnapshotPointerConflict(
                f"remote binding advanced from {remote_id} to "
                f"{existing_binding}"
            )
        snapshots[binding_key] = {
            "remote_id": remote_id,
            "attempt_id": attempt_id,
            "validated": validated,
            "conflicted": False,
            "unresolvable": False,
        }
        if attempt_id is not None:
            snapshots.pop(attempt_key, None)


def _replace_create_attempt_with_lineage_conflict(
    task_id: str,
    attempt_id: str,
    remote_ids: list[str],
    *,
    unresolvable: bool,
    store_path: Path | None = None,
) -> None:
    """Atomically replace an uncertain create with a durable conflict."""
    ids = sorted(set(remote_ids))
    with _mutate_store(store_path) as snapshots:
        attempt_key = _create_attempt_key(task_id)
        existing_attempt_id = _record_field(
            snapshots.get(attempt_key),
            "attempt_id",
        )
        if existing_attempt_id != attempt_id:
            raise _SnapshotPointerConflict(
                f"create attempt advanced from {attempt_id} to "
                f"{existing_attempt_id}"
            )
        binding_key = _remote_binding_key(task_id)
        if snapshots.get(binding_key) is not None:
            raise _SnapshotPointerConflict(
                f"task {task_id} acquired a remote binding during reconciliation"
            )
        snapshots[binding_key] = {
            "remote_id": ids[0] if ids else None,
            "attempt_id": attempt_id,
            "validated": False,
            "conflicted": True,
            "conflict_ids": ids,
            "unresolvable": unresolvable,
        }
        snapshots.pop(attempt_key, None)


def _mark_remote_binding_validated(
    task_id: str,
    remote_id: str,
    store_path: Path | None = None,
) -> None:
    """Durably publish positive sole-lineage validation for a bound remote."""
    with _mutate_store(store_path) as snapshots:
        key = _remote_binding_key(task_id)
        existing = snapshots.get(key)
        existing_id = _record_field(existing, "remote_id")
        if existing_id != remote_id:
            raise _SnapshotPointerConflict(
                f"remote binding advanced from {remote_id} to {existing_id}"
            )
        if isinstance(existing, dict) and existing.get("conflicted"):
            raise _SnapshotPointerConflict(
                f"remote binding {remote_id} has a durable lineage conflict"
            )
        if isinstance(existing, dict):
            existing = dict(existing)
        else:
            existing = {"remote_id": remote_id, "attempt_id": None}
        existing["validated"] = True
        snapshots[key] = existing


def _mark_remote_binding_conflicted(
    task_id: str,
    remote_id: str,
    conflict_ids: list[str],
    *,
    unresolvable: bool,
    store_path: Path | None = None,
) -> None:
    """Permanently remember an observed fork despite later list omissions."""
    with _mutate_store(store_path) as snapshots:
        key = _remote_binding_key(task_id)
        existing = snapshots.get(key)
        existing_id = _record_field(existing, "remote_id")
        if existing_id != remote_id:
            raise _SnapshotPointerConflict(
                f"remote binding advanced from {remote_id} to {existing_id}"
            )
        if isinstance(existing, dict):
            conflict_record: dict[str, Any] = dict(existing)
        else:
            conflict_record = {
                "remote_id": remote_id,
                "attempt_id": None,
            }
        conflict_record["validated"] = False
        conflict_record["conflicted"] = True
        conflict_record["conflict_ids"] = sorted(set(conflict_ids))
        conflict_record["unresolvable"] = unresolvable
        snapshots[key] = conflict_record


def _store_unmanaged_lineage_conflict(
    task_id: str,
    remote_ids: list[str],
    *,
    unresolvable: bool,
    store_path: Path | None = None,
) -> None:
    """Record visible task-name collisions without claiming their ownership."""
    ids = sorted(set(remote_ids))
    with _mutate_store(store_path) as snapshots:
        if snapshots.get(_create_attempt_key(task_id)) is not None:
            raise _SnapshotPointerConflict(
                f"task {task_id} acquired a create attempt during collision check"
            )
        binding_key = _remote_binding_key(task_id)
        if snapshots.get(binding_key) is not None:
            raise _SnapshotPointerConflict(
                f"task {task_id} acquired a remote binding during collision check"
            )
        snapshots[binding_key] = {
            # None means at least one visible collision had no authoritative
            # id. That conflict cannot be auto-cleared, because no exact
            # lookup can ever prove the unidentified branch terminated.
            "remote_id": ids[0] if ids else None,
            "attempt_id": None,
            "validated": False,
            "conflicted": True,
            "unmanaged": True,
            "conflict_ids": ids,
            "unresolvable": unresolvable,
        }


def _clear_remote_binding(
    task_id: str,
    remote_id: str,
    store_path: Path | None = None,
) -> None:
    with _mutate_store(store_path) as snapshots:
        key = _remote_binding_key(task_id)
        existing = _record_field(snapshots.get(key), "remote_id")
        if existing is None:
            # Returning still re-publishes this binding-free state, so an
            # earlier removal left visible by an uncertain commit becomes
            # durable before ownership can be released.
            return
        if existing != remote_id:
            raise _SnapshotPointerConflict(
                f"remote binding advanced from {remote_id} to {existing}"
            )
        snapshots.pop(key, None)


def _pending_snapshot_retirements(
    store_path: Path | None = None,
) -> tuple[str, ...]:
    snapshots = _load_snapshots(store_path)
    prefix = f"{_SNAPSHOT_RETIREMENT_NAMESPACE}:"
    return tuple(
        value
        for key, value in snapshots.items()
        if key.startswith(prefix) and isinstance(value, str) and value
    )


def _queue_snapshot_retirement(
    snapshot_id: str,
    store_path: Path | None = None,
) -> None:
    # Deliberately NOT _mutate_store: an already-tombstoned snapshot must exit
    # without writing at all, and that helper always republishes on return.
    path = store_path or _snapshot_store_path()
    with _snapshot_store_lock(path):
        snapshots = _load_recovery_registry(path)
        if _snapshot_retired_key(snapshot_id) in snapshots:
            return
        snapshots[_snapshot_retirement_key(snapshot_id)] = snapshot_id
        _atomic_save_snapshots(path, snapshots)


def _queue_snapshot_pointer_retirement(
    task_id: str,
    snapshot_id: str,
    store_path: Path | None = None,
) -> None:
    """Atomically detach one unusable pointer and journal its retirement."""
    with _mutate_store(store_path) as snapshots:
        for key in (_snapshot_key(task_id), task_id):
            if snapshots.get(key) == snapshot_id:
                snapshots.pop(key, None)
        if _snapshot_retired_key(snapshot_id) not in snapshots:
            snapshots[_snapshot_retirement_key(snapshot_id)] = snapshot_id


def _confirm_snapshot_store_durable(
    store_path: Path | None = None,
) -> None:
    """Re-establish local durability before acting on retirement records.

    A prior pointer replace may be visible even though its directory fsync
    failed. Retrying that fsync before any remote deletion makes the visible
    pointer+journal commit durable and preserves the uncertain-commit rule:
    neither predecessor is retired until local recovery metadata is safe.
    """
    path = store_path or _snapshot_store_path()
    with _snapshot_store_lock(path):
        if not path.exists():
            return
        with path.open("rb") as store_file:
            os.fsync(store_file.fileno())
        platform = _snapshot_platform()
        if platform == "nt":
            # Re-publish the visible state through MoveFileExW with
            # MOVEFILE_WRITE_THROUGH. This also upgrades state left by an
            # uncertain prior replacement before a retirement retry can delete
            # a remote recovery copy.
            _atomic_save_snapshots(path, _load_recovery_registry(path))
            return
        if platform != "posix":
            raise RuntimeError(
                "Tenki snapshot registry cannot confirm rename durability on "
                f"platform {platform!r}"
            )
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        dir_fd = os.open(path.parent, flags)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _get_snapshot_restore_candidate(
    task_id: str, store_path: Path | None = None
) -> tuple[str | None, bool]:
    snapshots = _load_snapshots(store_path)
    namespaced_key = _snapshot_key(task_id)
    snapshot_id = snapshots.get(namespaced_key)
    if isinstance(snapshot_id, str) and snapshot_id:
        return snapshot_id, False
    legacy_snapshot_id = snapshots.get(task_id)
    if isinstance(legacy_snapshot_id, str) and legacy_snapshot_id:
        return legacy_snapshot_id, True
    return None, False


def _store_snapshot(
    task_id: str,
    snapshot_id: str,
    store_path: Path | None = None,
) -> str | None:
    """Atomically install *snapshot_id* and return its predecessor, if any."""
    previous: Any = None
    # _mutate_store commits on exit from the ``with`` block, so the annotating
    # handler has to wrap the block itself. Only the commit can raise this
    # type, so the wider span does not widen what is caught.
    try:
        with _mutate_store(store_path) as snapshots:
            key = _snapshot_key(task_id)
            if _snapshot_retired_key(snapshot_id) in snapshots:
                raise _SnapshotPointerConflict(
                    f"snapshot {snapshot_id} was already retired"
                )
            previous = snapshots.get(key)
            if previous is None:
                previous = snapshots.get(task_id)
            snapshots[key] = snapshot_id
            snapshots.pop(task_id, None)
            if (
                isinstance(previous, str)
                and previous
                and previous != snapshot_id
            ):
                # Journal the predecessor in the same atomic commit as the new
                # pointer. A crash or transient delete failure can then retry
                # retirement without ever forgetting the old remote snapshot.
                snapshots[_snapshot_retirement_key(previous)] = previous
    except _SnapshotPointerCommitUncertain as exc:
        exc.previous_snapshot_id = (
            previous if isinstance(previous, str) and previous else None
        )
        exc.new_snapshot_id = snapshot_id
        raise
    return previous if isinstance(previous, str) and previous else None


def _migrate_snapshot_pointer(
    task_id: str,
    source_task_id: str,
    snapshot_id: str,
    store_path: Path | None = None,
) -> None:
    """CAS-migrate one legacy pointer without overwriting newer state."""
    with _mutate_store(store_path) as snapshots:
        if _snapshot_retired_key(snapshot_id) in snapshots:
            raise _SnapshotPointerConflict(
                f"legacy snapshot {snapshot_id} was already retired"
            )

        target_key = _snapshot_key(task_id)
        target_value = snapshots.get(target_key)
        source_keys = (
            _snapshot_key(source_task_id),
            source_task_id,
        )
        source_matches = any(
            snapshots.get(key) == snapshot_id for key in source_keys
        )

        if target_value not in (None, snapshot_id):
            raise _SnapshotPointerConflict(
                f"canonical pointer advanced to {target_value}"
            )
        if target_value is None and not source_matches:
            raise _SnapshotPointerConflict(
                "legacy source pointer changed before migration"
            )

        snapshots[target_key] = snapshot_id
        for source_key in source_keys:
            if source_key != target_key and snapshots.get(source_key) == snapshot_id:
                snapshots.pop(source_key, None)
        # Same-task legacy format is the plain task key.
        if snapshots.get(task_id) == snapshot_id:
            snapshots.pop(task_id, None)


def _safe_name(value: str, *, fallback: str = "default", max_len: int = 48) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value or "").strip("-._")
    return (safe or fallback)[:max_len]


def _supports_any_kwargs(sig: inspect.Signature | None) -> bool:
    if sig is None:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def _named_parameters(sig: inspect.Signature) -> set[str]:
    """Explicitly named parameters, ignoring ``self``/``*args``/``**kwargs``."""
    return {
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }


def _add_supported(
    kwargs: dict[str, Any],
    sig: inspect.Signature | None,
    names: tuple[str, ...],
    value: Any,
) -> None:
    if value in (None, "", [], {}):
        return
    if sig is not None:
        for name in names:
            if name in sig.parameters:
                kwargs[name] = value
                return
    if _supports_any_kwargs(sig):
        kwargs[names[0]] = value


def _result_attr(result: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if not hasattr(result, name):
            continue
        value = getattr(result, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                pass
        if value is not None:
            return value
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and math.isfinite(number) else None


def _tenki_resource_value(
    value: Any,
    *,
    minimum: int,
    maximum: int,
    label: str,
    alignment: int = 1,
) -> int | None:
    """Return a supported Tenki resource value, otherwise use SDK defaults."""
    try:
        number = math.ceil(float(value))
    except (TypeError, ValueError, OverflowError):
        number = None
    if (
        number is not None
        and minimum <= number <= maximum
        and number % alignment == 0
    ):
        return number
    alignment_hint = (
        f" and aligned to {alignment}" if alignment > 1 else ""
    )
    logger.warning(
        "Tenki: ignoring unsupported %s %r (supported range: %s-%s%s); "
        "using the Tenki workspace default",
        label,
        value,
        minimum,
        maximum,
        alignment_hint,
    )
    return None


class TenkiEnvironment(BaseEnvironment):
    """Tenki sandbox backend.

    Tenki's SDK exposes process handles inside a remote sandbox, so this adapts
    them to the normal Hermes ``ProcessHandle`` contract with
    ``_ThreadedProcessHandle``.
    """

    _snapshot_timeout = 60
    _terminal_states = frozenset({"TERMINATING", "TERMINATED", "DELETED", "FAILED", "ERROR"})

    def __init__(
        self,
        image: str = "",
        cwd: str = "/home/tenki",
        timeout: int = 60,
        cpu: float = 1,
        memory: int = 5120,
        disk: int = 51200,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        api_endpoint: str = "",
        workspace_id: str = "",
        name_prefix: str = "hermes",
        allow_inbound: bool = False,
        allow_outbound: bool = True,
        max_duration: int = 3600,
        idle_timeout: int = 0,
        pause_retention: int = 0,
        sync_hermes_home: bool = False,
        forward_env: list[str] | None = None,
    ):
        super().__init__(cwd=cwd, timeout=timeout)

        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            _lazy_ensure("terminal.tenki", prompt=False)
        except ImportError:
            pass
        except Exception as exc:
            raise ImportError(str(exc))

        from tenki import Client, Sandbox

        self._Client = Client
        self._Sandbox = Sandbox
        self._client = None
        self._sandbox = None
        self._create_outcome_uncertain = False
        self._create_attempt_id: str | None = None
        self._create_attempt_expires_at: float | None = None
        self._remote_binding = _RemoteBinding()
        self._create_lineage_ambiguous = False
        self._lock = threading.Lock()
        self._lifecycle_condition = threading.Condition(self._lock)
        self._persistent = persistent_filesystem
        self._sync_hermes_home = sync_hermes_home
        self._sync_manager: FileSyncManager | None = None
        self._cleanup_in_progress = False
        self._cleanup_complete = False
        self._cleanup_sandbox = None
        self._cancel_in_progress = 0
        self._cancel_generation = 0
        self._active_operations = 0
        self._task_id = task_id
        self._profile_token = _profile_token()
        canonical_prefix = f"tenki:{self._profile_token}:"
        task_suffix = (
            self._task_id[len(canonical_prefix):]
            if self._task_id.startswith(canonical_prefix)
            else None
        )
        self._profile_task_candidates = [
            (self._profile_token, self._task_id),
            *[
                (
                    token,
                    f"tenki:{token}:{task_suffix}"
                    if task_suffix is not None
                    else self._task_id,
                )
                for token in _legacy_profile_tokens()
            ],
        ]
        # Bind the profile's snapshot-store path at construction, while the
        # correct HERMES_HOME context is active. Cleanup (and the idle-reaper
        # snapshot save) can run in a background thread that does NOT inherit
        # the per-turn HERMES_HOME contextvar, so re-resolving there would write
        # the pointer into the wrong profile's home.
        self._snapshot_store = _snapshot_store_path()
        self._profile_home = get_hermes_home()
        self._profile_sync_context = copy_context()
        self._snapshot_restore_id: str | None = None
        self._snapshot_restore_from_legacy_key = False
        self._snapshot_restore_task_id = self._task_id
        self._image = image
        self._cpu = cpu
        self._memory = memory
        self._disk = disk
        self._api_endpoint = resolve_tenki_api_endpoint(api_endpoint)
        self._workspace_id = resolve_tenki_workspace_id(workspace_id)
        self._auth_token = resolve_tenki_auth_token()
        self._name_prefix = _safe_name(name_prefix, fallback="hermes", max_len=28)
        # Every name this wrapper will ever answer to is fixed here: the prefix,
        # the profile token, and the task id never change after construction.
        # Precomputing turns the per-row match in _sandbox_matches_task into a
        # dict lookup — a workspace listing is scanned up to three times per
        # create, and re-deriving (and re-sanitizing) one name per legacy
        # profile candidate per listed sandbox was hundreds of regex substs per
        # environment init. First candidate wins on a name collision, matching
        # the previous first-match-wins scan order.
        self._canonical_sandbox_name = self._sandbox_name_for(
            self._profile_token,
            self._task_id,
        )
        self._expected_sandbox_names: dict[str, tuple[str, str]] = {}
        for candidate_identity in self._profile_task_candidates:
            self._expected_sandbox_names.setdefault(
                self._sandbox_name_for(*candidate_identity),
                candidate_identity,
            )
        self._allow_inbound = allow_inbound
        self._allow_outbound = allow_outbound
        self._effective_max_duration = _positive_float(max_duration) or 3600
        self._idle_timeout = idle_timeout
        self._pause_retention = pause_retention
        self._forward_env = normalize_forward_env_names(
            forward_env,
            setting_name="tenki_forward_env",
        )
        self._remote_home = "/home/tenki"
        # Hold this profile/task lock for the wrapper's full lifetime. It is
        # acquired before pointer discovery and remote listing/creation, so two
        # Hermes processes cannot fork a task or terminate one another's live
        # sandbox. OS crash/exit releases the kernel lock automatically.
        self._task_ownership_file = _acquire_task_ownership_lock(
            self._profile_home,
            self._task_id,
        )
        try:
            (
                self._create_attempt_id,
                self._create_attempt_expires_at,
            ) = _create_attempt_state(
                self._task_id,
                self._snapshot_store,
            )
            self._remote_binding = _remote_binding_state(
                self._task_id,
                self._snapshot_store,
            )
            if self._create_attempt_id is not None and (
                self._remote_binding.remote_id is not None
                or self._remote_binding.conflicted
            ):
                raise RuntimeError(
                    "Tenki task has both an unresolved create and a remote "
                    f"binding: {self._task_id}"
                )
            self._create_outcome_uncertain = bool(self._create_attempt_id)
            if self._persistent:
                seen_task_ids: set[str] = set()
                for _token, candidate_task_id in self._profile_task_candidates:
                    if candidate_task_id in seen_task_ids:
                        continue
                    seen_task_ids.add(candidate_task_id)
                    snapshot_id, from_legacy_key = (
                        _get_snapshot_restore_candidate(
                            candidate_task_id,
                            self._snapshot_store,
                        )
                    )
                    if snapshot_id:
                        self._snapshot_restore_id = snapshot_id
                        self._snapshot_restore_from_legacy_key = from_legacy_key
                        self._snapshot_restore_task_id = candidate_task_id
                        break

            self._ensure_sandbox()
            self._resolve_remote_home()
            if self._sync_hermes_home:
                self._sync_manager = FileSyncManager(
                    get_files_fn=self._profile_sync_files,
                    upload_fn=self._tenki_upload,
                    delete_fn=self._tenki_delete,
                    bulk_upload_fn=self._tenki_bulk_upload,
                    bulk_download_fn=self._tenki_bulk_download,
                    get_upload_only_host_paths_fn=(
                        self._profile_credential_host_paths
                    ),
                )
                self._sync_manager.sync(force=True)
            self.init_session()
        except BaseException:
            self._abort_failed_initialization()
            raise

    def _sandbox_create_signature(self) -> inspect.Signature | None:
        """Signature that actually validates the sandbox create kwargs.

        ``Sandbox.create`` is a bare ``**kwargs`` passthrough: it pops the
        client-construction kwargs and forwards everything else to
        ``Client.create``. Introspecting it therefore accepts *every* name and
        filters nothing, so a kwarg the SDK has dropped (``project_id``, gone
        in tenki 0.5) sails through and only fails as a ``TypeError`` at call
        time. ``Client.create`` is the real validator in both paths, so prefer
        it and fall back to ``Sandbox.create`` only if it can't be introspected.
        """
        candidates = (
            getattr(self._Client, "create", None),
            getattr(self._Sandbox, "create", None),
        )
        fallback: inspect.Signature | None = None
        for target in candidates:
            if target is None:
                continue
            try:
                sig = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            # A pure **kwargs passthrough names nothing it accepts; keep it only
            # as a last resort so _add_supported still degrades to "send it".
            if _supports_any_kwargs(sig) and not _named_parameters(sig):
                fallback = fallback or sig
                continue
            return sig
        return fallback

    def _create_kwargs(self) -> dict[str, Any]:
        sig = self._sandbox_create_signature()
        kwargs: dict[str, Any] = {}
        sandbox_name = self._sandbox_name()

        _add_supported(kwargs, sig, ("name",), sandbox_name)
        if self._snapshot_restore_id:
            _add_supported(kwargs, sig, ("snapshot_id",), self._snapshot_restore_id)
        else:
            _add_supported(kwargs, sig, ("image", "template"), self._image)
        cpu_cores = _tenki_resource_value(
            self._cpu,
            minimum=_TENKI_CPU_RANGE[0],
            maximum=_TENKI_CPU_RANGE[1],
            label="CPU cores",
        )
        _add_supported(kwargs, sig, ("cpu_cores", "cpu"), cpu_cores)
        memory_mb = _tenki_resource_value(
            self._memory,
            minimum=_TENKI_MEMORY_MB_RANGE[0],
            maximum=_TENKI_MEMORY_MB_RANGE[1],
            label="memory (MB)",
            alignment=2,
        )
        _add_supported(kwargs, sig, ("memory_mb", "memory"), memory_mb)

        try:
            disk_gb_value = float(self._disk) / 1024
        except (TypeError, ValueError, OverflowError):
            disk_gb_value = self._disk
        disk_gb = _tenki_resource_value(
            disk_gb_value,
            minimum=_TENKI_DISK_GB_RANGE[0],
            maximum=_TENKI_DISK_GB_RANGE[1],
            label="root disk (GB)",
        )
        _add_supported(kwargs, sig, ("disk_size_gb", "disk_gb", "disk"), disk_gb)

        _add_supported(kwargs, sig, ("allow_inbound",), self._allow_inbound)
        _add_supported(kwargs, sig, ("allow_outbound",), self._allow_outbound)
        _add_supported(
            kwargs,
            sig,
            ("max_duration",),
            self._effective_max_duration,
        )
        idle_timeout = _positive_float(self._idle_timeout)
        if idle_timeout is not None:
            idle_timeout_minutes = max(1, math.ceil(idle_timeout / 60))
            _add_supported(kwargs, sig, ("idle_timeout_minutes",), idle_timeout_minutes)
        pause_retention = _positive_float(self._pause_retention)
        if pause_retention is not None:
            _add_supported(kwargs, sig, ("pause_retention",), pause_retention)
        _add_supported(kwargs, sig, ("workspace_id",), self._workspace_id)
        # Client-construction kwargs, NOT sandbox kwargs: Sandbox.create pops
        # these into the Client it builds, so they never appear in the
        # create-signature we filter against. The persistent path builds its own
        # client and strips them again in _create_sandbox_from_kwargs.
        if self._api_endpoint:
            kwargs["base_url"] = self._api_endpoint
        if self._auth_token:
            kwargs["auth_token"] = self._auth_token
        _add_supported(kwargs, sig, ("env",), self._sandbox_env())
        _add_supported(
            kwargs,
            sig,
            ("metadata",),
            {
                "hermes_task_id": self._task_id,
                "hermes_backend": "tenki",
                "hermes_profile": self._profile_token,
            },
        )
        _add_supported(kwargs, sig, ("tags",), ["hermes-agent"])
        # Keep readiness outside Client.create so every post-RPC readiness
        # failure occurs after Hermes owns the exact Sandbox handle. Otherwise
        # the SDK can construct a Sandbox, fail in its internal wait, and raise
        # without returning the only identity that is safe to clean up.
        _add_supported(kwargs, sig, ("wait",), False)
        # Do NOT emit a create-time ``timeout`` here: the SDK's Sandbox.create
        # pops ``timeout`` into the *Client* (HTTP) timeout, while Client.create
        # treats ``timeout`` as the *wait-for-ready* budget — so the same value
        # would mean two different things across the two create paths. The HTTP
        # timeout is set explicitly in _create_client(); readiness uses the
        # SDK's default wait budget.
        return kwargs

    def _sandbox_env(self) -> dict[str, str]:
        """Environment variables injected into Tenki sandbox processes.

        The supervisor's Tenki control-plane credential is used host-side to
        create and manage the sandbox (see ``_create_kwargs`` /
        ``_create_client``); it is deliberately NOT injected into the guest.
        Guest code is model-controlled and can print, exfiltrate, or reuse
        whatever is in its environment, and the sandbox is billed against the
        supervisor's account — so a leaked ``TENKI_AUTH_TOKEN`` would let guest
        code create, terminate, and bill account resources outside the
        parent's configured limits. Nested-sandbox support is still available
        as an explicit opt-in: list ``TENKI_AUTH_TOKEN`` (or ``TENKI_API_KEY``)
        in ``terminal.tenki_forward_env``.

        ``terminal.tenki_forward_env`` is the explicit allowlist for
        task-specific credentials such as GitHub tokens; the generic
        ``terminal.env_passthrough`` allowlist is also honored for skill
        variables that are not protected by Hermes' provider-secret blocklist.
        """
        env: dict[str, str] = {}
        env.update(self._resolve_forwarded_env(self._forward_env))
        env.update(self._passthrough_env())
        # If the operator explicitly opted into forwarding the control-plane
        # credential (for nested-sandbox creation), supply the already-resolved
        # token. Re-reading the env var here would miss a `tenki login`
        # credential, which lives in the Tenki CLI config, not the environment —
        # so the documented opt-in would silently forward nothing.
        if self._auth_token:
            for key in ("TENKI_AUTH_TOKEN", "TENKI_API_KEY"):
                if key in self._forward_env and not env.get(key):
                    env[key] = self._auth_token
                    logger.warning(
                        "Tenki: forwarding the control-plane credential %s into the "
                        "sandbox as requested by terminal.tenki_forward_env. Guest code "
                        "can read it and create/terminate/bill account resources. "
                        "Tenki sandboxes are profile-isolated, but every process inside "
                        "this profile's sandbox can read the forwarded value.",
                        key,
                    )
        return env

    @staticmethod
    def _resolve_forwarded_env(keys: list[str] | set[str] | tuple[str, ...]) -> dict[str, str]:
        if not keys:
            return {}
        from tools.tenki_config import _global_credential_fallback_allowed, _scoped_env

        get_env_value = None
        if _global_credential_fallback_allowed():
            try:
                from hermes_cli.config import get_env_value
            except Exception:
                get_env_value = None

        env: dict[str, str] = {}
        for key in keys:
            # Scope-aware read first: under a multiplexed profile turn this
            # resolves the active profile's value, never another profile's raw
            # os.environ. The ~/.hermes/.env fallback is consulted only when no
            # profile scope is authoritative.
            value = _scoped_env(key)
            if not value and get_env_value is not None:
                try:
                    value = get_env_value(key) or ""
                except Exception:
                    value = ""
            if value:
                env[key] = value
        return env

    @staticmethod
    def _passthrough_env() -> dict[str, str]:
        try:
            from tools.env_passthrough import get_all_passthrough

            keys = sorted(get_all_passthrough())
        except Exception:
            keys = []
        return TenkiEnvironment._resolve_forwarded_env(keys)

    def _create_client(self):
        if self._client is None:
            self._client = self._Client(
                auth_token=self._auth_token,
                base_url=self._api_endpoint,
                timeout=max(60, self.timeout),
            )
        return self._client

    def _sandbox_name(self) -> str:
        # The profile token namespaces the name so two profiles sharing one
        # Tenki account never collide on a name or reuse each other's sandbox.
        return self._canonical_sandbox_name

    def _sandbox_name_for(self, profile_token: str, task_id: str) -> str:
        return f"{self._name_prefix}-{profile_token}-{_safe_name(task_id)}"

    @staticmethod
    def _sandbox_state(sandbox: Any) -> str:
        state = getattr(sandbox, "state", "")
        if callable(state):
            try:
                state = state()
            except TypeError:
                state = ""
        return str(state or "").upper()

    def _sandbox_matches_task(self, sandbox: Any) -> bool:
        name = getattr(sandbox, "name", "")
        info = getattr(sandbox, "info", None)
        if not name and info is not None:
            name = getattr(info, "name", "")
        # ``isinstance`` guard, not a behavior change: a non-``str`` name could
        # never equal one of the generated names under the old scan either, and
        # an unhashable one would raise inside the lookup.
        matched_identity = (
            self._expected_sandbox_names.get(name)
            if isinstance(name, str)
            else None
        )
        if matched_identity is None:
            return False
        matched_token, matched_task_id = matched_identity
        metadata = getattr(info, "metadata", {}) if info is not None else {}
        # Never reuse another profile's sandbox: if the candidate carries a
        # profile token it must match ours (the name already encodes it, but
        # metadata is the authoritative, defense-in-depth check).
        if isinstance(metadata, dict) and metadata.get("hermes_profile"):
            if metadata.get("hermes_profile") != matched_token:
                return False
        if isinstance(metadata, dict) and metadata.get("hermes_task_id"):
            return metadata.get("hermes_task_id") == matched_task_id
        return True

    def _sandbox_has_owned_identity(
        self,
        sandbox: Any,
        expected_attempt_id: str,
    ) -> bool:
        """Require the canonical metadata Hermes writes before claiming use."""
        name = getattr(sandbox, "name", "")
        info = getattr(sandbox, "info", None)
        if not name and info is not None:
            name = getattr(info, "name", "")
        metadata = getattr(info, "metadata", None) if info is not None else None
        return (
            name == self._sandbox_name()
            and isinstance(metadata, dict)
            and metadata.get("hermes_backend") == "tenki"
            and metadata.get("hermes_task_id") == self._task_id
            and metadata.get("hermes_profile") == self._profile_token
            and metadata.get("hermes_create_attempt") == expected_attempt_id
        )

    def _list_kwargs(self, client: Any) -> dict[str, Any]:
        """Kwargs for the sandbox listing used to re-attach a persistent sandbox.

        tenki 0.5 folded the old ``list_project`` / ``list_workspace`` helpers
        into ``Client.list``, which takes the workspace as a keyword. Older SDK
        builds don't accept it, so scope the listing only when the installed
        signature says it will be honored.
        """
        kwargs: dict[str, Any] = {"tags": ["hermes-agent"]}
        if not self._workspace_id:
            return kwargs
        try:
            sig = inspect.signature(client.list)
        except (TypeError, ValueError):
            return kwargs
        if "workspace_id" in sig.parameters or _supports_any_kwargs(sig):
            kwargs["workspace_id"] = self._workspace_id
        return kwargs

    def _fail_ambiguous_lineage(
        self,
        message: str,
        cause: BaseException | None = None,
    ) -> NoReturn:
        """Mark this task's lineage unusable and fail closed in one step.

        The flag and the raise are inseparable: a raise that forgot the flag
        would let ``_abort_failed_initialization`` release task ownership (and
        ``cleanup`` terminate a sandbox) while two remote lineages may still be
        live. Passing *cause* preserves explicit exception chaining.
        """
        self._create_lineage_ambiguous = True
        if cause is not None:
            raise RuntimeError(message) from cause
        raise RuntimeError(message)

    def _assert_no_unmanaged_persistent_sandbox(self) -> None:
        """Refuse visible name collisions that have no local ownership state.

        Tenki support is new in this unmerged change, so supported Hermes
        sandboxes always have a create-attempt or exact-id binding in this
        profile store. A matching remote with neither is external/unpublished
        state: listing may detect it, but listing can never confer ownership.
        """
        if not self._persistent:
            return
        client = self._create_client()
        try:
            candidates = client.list(**self._list_kwargs(client))
        except Exception as exc:
            # A failed listing is not proof that no sandbox exists. Creating
            # from the last snapshot in this state can fork a live sandbox
            # containing newer, unsnapshotted state.
            raise RuntimeError(
                "Tenki could not check for unmanaged persistent sandbox "
                f"collisions for task {self._task_id}"
            ) from exc

        remote_ids: list[str] = []
        unidentified_collision = False
        for sandbox in candidates:
            if not self._sandbox_matches_task(sandbox):
                continue
            state = self._sandbox_state(sandbox)
            if state in self._terminal_states:
                continue
            remote_id = self._sandbox_identity(sandbox)
            if remote_id is None:
                unidentified_collision = True
                continue
            remote_ids.append(remote_id)
        if not remote_ids and not unidentified_collision:
            return
        try:
            _store_unmanaged_lineage_conflict(
                self._task_id,
                remote_ids,
                unresolvable=unidentified_collision,
                store_path=self._snapshot_store,
            )
        except BaseException:
            self._create_lineage_ambiguous = True
            raise
        self._remote_binding = _RemoteBinding(
            remote_id=(
                None
                if unidentified_collision
                else sorted(set(remote_ids))[0]
            ),
            conflicted=True,
            conflict_ids=tuple(sorted(set(remote_ids))),
            unresolvable=unidentified_collision,
        )
        self._fail_ambiguous_lineage(
            "Tenki found an unmanaged persistent sandbox collision for task "
            f"{self._task_id}; terminate the reported remote ids before retrying"
        )

    def _ensure_sandbox_ready(self, sandbox: Any) -> bool:
        """Return False only when *sandbox* is definitively terminal.

        All ambiguous control-plane failures propagate. Callers may create a
        replacement only after a successful refresh proves the old sandbox is
        terminal; treating a transient refresh/resume/readiness error as
        absence would fork persistent state.
        """
        refresh = getattr(sandbox, "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception as exc:
                raise RuntimeError(
                    "Tenki could not refresh the persistent sandbox for task "
                    f"{self._task_id}"
                ) from exc

        state = self._sandbox_state(sandbox)
        if state in self._terminal_states:
            return False

        try:
            if state != "RUNNING":
                resume = getattr(sandbox, "resume", None)
                if not callable(resume):
                    raise RuntimeError(
                        "persistent sandbox is not running and cannot be resumed"
                    )
                resume()
                wait_ready = getattr(sandbox, "wait_ready", None)
                if callable(wait_ready):
                    wait_ready(max(60, self.timeout))
        except Exception as exc:
            raise RuntimeError(
                "Tenki could not make the persistent sandbox ready for task "
                f"{self._task_id}"
            ) from exc

        final_state = self._sandbox_state(sandbox)
        if final_state in self._terminal_states:
            return False
        if final_state != "RUNNING":
            raise RuntimeError(
                "Tenki could not confirm that the persistent sandbox is "
                f"running for task {self._task_id} (state={final_state!r})"
            )
        return True

    def _ensure_sandbox(self) -> None:
        with self._lifecycle_condition:
            # Sandbox-level cancellation pauses/terminates outside the lock.
            # Do not inspect or replace its exact reference until cancellation
            # has restored the ownership invariant and notified waiters.
            while self._cancel_in_progress:
                self._lifecycle_condition.wait()
            if self._cleanup_in_progress or self._cleanup_complete:
                raise RuntimeError("Tenki cleanup is in progress")
            if (
                self._remote_binding.remote_id is not None
                or self._remote_binding.conflicted
            ):
                self._resolve_remote_binding()
            if self._create_outcome_uncertain:
                if not self._reconcile_uncertain_create():
                    raise RuntimeError(
                        "Tenki cannot create a replacement while a prior "
                        f"create is unresolved for task {self._task_id}"
                    )
            if self._sandbox is not None:
                sandbox = self._sandbox
                if not self._remote_binding.validated:
                    self._validate_created_lineage()
                if self._ensure_sandbox_ready(sandbox):
                    self._after_sandbox_ownership_confirmed()
                    return
                self._sandbox = None
                if not self._clear_remote_binding_marker(sandbox):
                    raise RuntimeError(
                        "Tenki cannot replace a terminal sandbox until its "
                        f"remote binding is durably cleared for {self._task_id}"
                    )
            self._assert_no_unmanaged_persistent_sandbox()
            try:
                self._sandbox = self._create_sandbox_with_snapshot_fallback()
            except BaseException:
                if self._create_outcome_uncertain:
                    self._reconcile_uncertain_create()
                raise
            self._validate_created_lineage()
            self._wait_created_sandbox_ready(self._sandbox)
            self._after_sandbox_ownership_confirmed()
            sandbox_id = self._sandbox_identity(self._sandbox)
            logger.info("Tenki: created sandbox %s for task %s", sandbox_id or "<unknown>", self._task_id)

    def _after_sandbox_ownership_confirmed(self) -> None:
        """Retry local housekeeping only after exact live ownership exists."""
        if not self._persistent:
            return
        self._migrate_loaded_snapshot_pointer()
        self._retry_pending_snapshot_retirements()

    def _require_sandbox(self) -> Any:
        # Capture the reference under the lock: cancel() may null out
        # self._sandbox between _ensure_sandbox() and the caller's use of it,
        # and the operation must run against the sandbox that ensure produced.
        self._ensure_sandbox()
        with self._lock:
            # cleanup() can claim the sandbox after _ensure_sandbox() releases
            # the lock but before this capture. Never hand a caller the same
            # reference cleanup is syncing, snapshotting, or terminating.
            if self._cleanup_in_progress or self._cleanup_complete:
                raise RuntimeError("Tenki cleanup is in progress")
            sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Tenki sandbox was torn down mid-operation")
        return sandbox

    @contextmanager
    def _sandbox_operation(self):
        """Lease one sandbox reference for the full duration of an operation."""
        while True:
            with self._lifecycle_condition:
                while self._cancel_in_progress:
                    self._lifecycle_condition.wait()
                cancel_generation = self._cancel_generation
            sandbox = self._require_sandbox()
            with self._lifecycle_condition:
                if self._cancel_in_progress:
                    # Cancellation may pause or detach the reference returned
                    # above. Wait for exact ownership to be restored, then run
                    # ensure again so a paused persistent sandbox is resumed.
                    while self._cancel_in_progress:
                        self._lifecycle_condition.wait()
                    continue
                if self._cancel_generation != cancel_generation:
                    # A cancel completed entirely between readiness resolution
                    # and lease publication. Even when persistent ownership
                    # restored the same object, it may now be PAUSED; run the
                    # readiness path again before publishing a lease.
                    continue
                if self._cleanup_in_progress or self._cleanup_complete:
                    raise RuntimeError("Tenki cleanup is in progress")
                if self._sandbox is not sandbox:
                    # An ephemeral cancel detached this candidate between
                    # require() and lease publication. Resolve a fresh one.
                    continue
                self._active_operations += 1
                break
        try:
            yield sandbox
        finally:
            with self._lifecycle_condition:
                self._active_operations -= 1
                self._lifecycle_condition.notify_all()

    @contextmanager
    def _transfer_operation(self):
        """Use cleanup's owned sandbox for sync-back, otherwise take a lease."""
        with self._lifecycle_condition:
            cleanup_sandbox = (
                self._cleanup_sandbox if self._cleanup_in_progress else None
            )
        if cleanup_sandbox is not None:
            yield cleanup_sandbox
            return
        with self._sandbox_operation() as sandbox:
            yield sandbox

    def _create_sandbox_from_kwargs(self, kwargs: dict[str, Any]):
        # Always create through the client owned by this environment. In Tenki
        # 0.5.1, Sandbox.create() constructs a hidden Client and marks the
        # sandbox as owning it, but Sandbox.close()/terminate() does not close
        # that client; only the SDK context-manager exit does. Keeping one
        # explicit client lets cleanup close the control-plane channel for
        # ephemeral sandboxes too, and reuse it safely after cancellation.
        client = self._create_client()
        create_kwargs = dict(kwargs)
        for key in ("auth_token", "api_key", "base_url", "api_endpoint"):
            create_kwargs.pop(key, None)

        # Prove Python call-shape failures before journaling or issuing the
        # non-idempotent RPC. Anything that escapes Client.create() after this
        # point is outcome-uncertain: Tenki decodes the response into
        # SandboxInfo only after create_session commits, and that post-RPC
        # conversion can itself raise TypeError or ValueError.
        try:
            inspect.signature(client.create).bind(**create_kwargs)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Tenki create arguments do not match the installed SDK"
            ) from exc

        if self._create_attempt_id is not None:
            raise RuntimeError(
                "Tenki cannot start a new create while attempt "
                f"{self._create_attempt_id} is unresolved"
            )
        attempt_id = uuid.uuid4().hex
        expires_at = (
            time.time()
            + self._effective_max_duration
            + _CREATE_ATTEMPT_EXPIRY_GRACE
        )
        _begin_create_attempt(
            self._task_id,
            attempt_id,
            expires_at,
            self._snapshot_store,
        )
        self._create_attempt_id = attempt_id
        self._create_attempt_expires_at = expires_at
        metadata = dict(create_kwargs.get("metadata") or {})
        metadata["hermes_create_attempt"] = attempt_id
        create_kwargs["metadata"] = metadata

        # The RPC may commit remotely and then time out before returning a
        # handle. The durable attempt id remains both local and remotely
        # queryable until that exact remote is gone.
        self._create_outcome_uncertain = True
        sandbox = client.create(**create_kwargs)
        if not self._bind_remote_sandbox(sandbox, attempt_id=attempt_id):
            raise RuntimeError(
                "Tenki created a sandbox but could not durably bind its "
                f"remote identity for task {self._task_id}"
            )
        self._create_outcome_uncertain = False
        return sandbox

    def _validate_created_lineage(self) -> None:
        """Validate exact ownership; use list only to detect visible conflicts."""
        sandbox = self._sandbox
        expected_attempt_id = self._remote_binding.attempt_id
        remote_id = self._sandbox_identity(sandbox)
        client = self._client
        get_sandbox = getattr(client, "get", None)
        list_sandboxes = getattr(client, "list", None)
        if (
            expected_attempt_id is None
            or remote_id is None
            or remote_id != self._remote_binding.remote_id
            or not self._sandbox_has_owned_identity(
                sandbox,
                expected_attempt_id,
            )
            or not callable(get_sandbox)
            or not callable(list_sandboxes)
        ):
            self._fail_ambiguous_lineage(
                f"Tenki cannot validate the new lineage for task {self._task_id}"
            )
        try:
            exact_sandbox = get_sandbox(remote_id)
            if (
                self._sandbox_identity(exact_sandbox) != remote_id
                or not self._sandbox_has_owned_identity(
                    exact_sandbox,
                    expected_attempt_id,
                )
            ):
                raise RuntimeError(
                    "Tenki exact-id lookup did not preserve the bound "
                    "task/profile/create-attempt identity"
                )
            if self._sandbox_state(exact_sandbox) in self._terminal_states:
                raise RuntimeError(
                    "Tenki exact-id lookup returned a terminal sandbox"
                )
            candidates = list_sandboxes(**self._list_kwargs(client))
            active = [
                candidate
                for candidate in candidates
                if self._sandbox_matches_task(candidate)
                and self._sandbox_state(candidate) not in self._terminal_states
            ]
        except BaseException as exc:
            raise RuntimeError(
                "Tenki could not validate that a new sandbox is the sole "
                f"lineage for task {self._task_id}"
            ) from exc

        other = [
            candidate
            for candidate in active
            if self._sandbox_identity(candidate) != remote_id
        ]
        if other:
            other_ids = [
                candidate_id
                for candidate in other
                if (candidate_id := self._sandbox_identity(candidate))
            ]
            # An active result without an id is an unresolvable conflict, but
            # every known id remains durable and actionable.
            unresolvable = len(other_ids) != len(other)
            conflict_ids = [remote_id, *other_ids]
            # Once a fork has been positively observed, omission-prone future
            # listings may never erase that fact or promote one branch.
            try:
                _mark_remote_binding_conflicted(
                    self._task_id,
                    remote_id,
                    conflict_ids,
                    unresolvable=unresolvable,
                    store_path=self._snapshot_store,
                )
            except BaseException as exc:
                self._fail_ambiguous_lineage(
                    "Tenki detected multiple lineages but could not durably "
                    f"record the conflict for task {self._task_id}",
                    exc,
                )
            self._remote_binding = replace(
                self._remote_binding,
                validated=False,
                conflicted=True,
                conflict_ids=tuple(sorted(set(conflict_ids))),
                unresolvable=unresolvable,
            )
            self._fail_ambiguous_lineage(
                "Tenki detected multiple active sandbox lineages for task "
                f"{self._task_id}; leaving every lineage untouched"
            )
        try:
            _mark_remote_binding_validated(
                self._task_id,
                remote_id,
                self._snapshot_store,
            )
        except BaseException as exc:
            raise RuntimeError(
                "Tenki could not durably record sole-lineage validation for "
                f"task {self._task_id}"
            ) from exc
        self._remote_binding = replace(self._remote_binding, validated=True)

    def _wait_created_sandbox_ready(self, sandbox: Any) -> None:
        """Wait only after the exact newly created Sandbox handle is owned."""
        wait_ready = getattr(sandbox, "wait_ready", None)
        if callable(wait_ready):
            wait_ready(max(60, self.timeout))
        state = self._sandbox_state(sandbox)
        if state in self._terminal_states:
            raise RuntimeError(
                f"Tenki create ended in terminal state {state!r}"
            )
        if state != "RUNNING":
            raise RuntimeError(
                f"Tenki create did not reach RUNNING state (state={state!r})"
            )

    # Snapshot errors that mean the recorded snapshot can never restore, so
    # dropping the pointer and booting the base image is the right recovery.
    # A *transient* failure (network, rate-limit, ambiguous) is NOT in this set:
    # it must propagate with the pointer intact so a later attempt can still
    # recover the persistent state instead of silently booting an empty sandbox.
    # Snapshot-specific error type names that mean the snapshot can never
    # restore. Note InvalidStateError is intentionally NOT here: the restore RPC
    # maps a generic FAILED_PRECONDITION (workspace/policy/etc.) to it, so it is
    # only unrecoverable when its message points at the snapshot itself
    # (handled by message inspection below); a bare InvalidStateError stays
    # transient so an unrelated precondition can't destroy a valid pointer.
    # ``RegistryArtifactNotFoundError`` was renamed ``RegistryImageNotFoundError``
    # in tenki 0.5; both names are listed so either SDK generation resolves.
    _UNRECOVERABLE_SNAPSHOT_ERRORS = frozenset({
        "SnapshotNotFoundError",          # snapshot is gone
        "RegistryImageNotFoundError",     # backing image is gone (0.5+)
        "RegistryArtifactNotFoundError",  # same, pre-0.5 name
        "SnapshotNotDurableError",        # explicitly never reached durability
    })

    @classmethod
    def _snapshot_unrecoverable(cls, exc: BaseException) -> bool:
        """True when the error confirms the snapshot can never restore.

        Covers "gone" (not-found), "explicitly non-durable", and a generic
        ``InvalidStateError`` whose message identifies the snapshot as the
        failing precondition (the SDK's restore RPC collapses a bad/non-durable
        snapshot into a generic FAILED_PRECONDITION → InvalidStateError). Only
        these justify discarding the recovery pointer and booting a base image;
        every other error (including a bare InvalidStateError, rate-limit,
        quota, auth blip, or network failure) is transient and re-raised with
        the pointer preserved.
        """
        def _is_snapshot_specific_invalid_state(e: BaseException, invalid_state_cls) -> bool:
            if invalid_state_cls is not None and not isinstance(e, invalid_state_cls):
                return False
            if invalid_state_cls is None and type(e).__name__ != "InvalidStateError":
                return False
            msg = str(e).lower()
            # FAILED_PRECONDITION is a broad service status: workspace policy,
            # quota, or a temporarily blocked restore can all map to the same
            # InvalidStateError. Discard the pointer only for an explicit,
            # permanent non-durability statement; ambiguous mentions of a
            # snapshot remain retryable.
            return any(
                phrase in msg
                for phrase in (
                    "snapshot is not durable",
                    "snapshot not durable",
                    "non-durable snapshot",
                    "snapshot is non-durable",
                )
            )

        # Resolve each class independently: a single `from tenki import (...)`
        # of all four fails outright when one has been renamed (as
        # RegistryArtifactNotFoundError was in 0.5), silently dropping the
        # isinstance check for the classes that *are* present.
        sdk: Any = None
        try:
            import tenki

            sdk = tenki
        except Exception:
            pass

        if sdk is not None:
            unrecoverable = tuple(
                cls_obj
                for cls_obj in (
                    getattr(sdk, name, None) for name in cls._UNRECOVERABLE_SNAPSHOT_ERRORS
                )
                if isinstance(cls_obj, type) and issubclass(cls_obj, BaseException)
            )
            if unrecoverable and isinstance(exc, unrecoverable):
                return True
            invalid_state_cls = getattr(sdk, "InvalidStateError", None)
            if (
                isinstance(invalid_state_cls, type)
                and issubclass(invalid_state_cls, BaseException)
                and _is_snapshot_specific_invalid_state(exc, invalid_state_cls)
            ):
                return True

        # Name-based fallback for SDK builds that don't export every class, and
        # for a subclass the installed SDK no longer exports under its own name.
        for typ in type(exc).__mro__:
            if typ.__name__ in cls._UNRECOVERABLE_SNAPSHOT_ERRORS:
                return True
        return _is_snapshot_specific_invalid_state(exc, None)

    def _retire_unrecoverable_restore_pointer(
        self,
        exc: BaseException,
    ) -> None:
        snapshot_id = self._snapshot_restore_id
        if snapshot_id is None:
            return
        logger.warning(
            "Tenki: snapshot %s for task %s is unrecoverable; creating from "
            "base image: %s",
            snapshot_id,
            self._task_id,
            exc,
        )
        _queue_snapshot_pointer_retirement(
            self._snapshot_restore_task_id,
            snapshot_id,
            self._snapshot_store,
        )
        self._delete_remote_snapshot(
            snapshot_id,
            reason="unrecoverable restore pointer",
        )
        self._snapshot_restore_id = None
        self._snapshot_restore_from_legacy_key = False
        self._snapshot_restore_task_id = self._task_id

    def _preflight_snapshot_restore(self) -> None:
        """Reject a known-bad snapshot before issuing a create RPC."""
        snapshot_id = self._snapshot_restore_id
        if snapshot_id is None:
            return
        client = self._create_client()
        snapshots = getattr(client, "snapshots", None)
        get_snapshot = getattr(snapshots, "get", None)
        if not callable(get_snapshot):
            return
        try:
            snapshot = get_snapshot(snapshot_id)
        except BaseException as exc:
            if not self._snapshot_unrecoverable(exc):
                raise
            self._retire_unrecoverable_restore_pointer(exc)
            return

        state = str(getattr(snapshot, "state", "") or "").upper()
        durability = str(
            getattr(snapshot, "durability_state", "") or ""
        ).upper()
        if state in {"FAILED", "DELETING", "DELETED"}:
            reason = str(getattr(snapshot, "failure_reason", "") or "")
            detail = f": {reason}" if reason else ""
            self._retire_unrecoverable_restore_pointer(
                RuntimeError(
                    f"snapshot {snapshot_id} is {state}{detail}"
                )
            )
            return
        if durability == "PROPAGATION_FAILED" or (
            state == "READY" and durability == "UNSPECIFIED"
        ):
            self._retire_unrecoverable_restore_pointer(
                RuntimeError(
                    f"snapshot {snapshot_id} has unusable durability state "
                    f"{durability or '<empty>'}"
                )
            )
            return
        if durability != "DURABLE":
            # A pending/unknown durability transition is not permanent, but
            # only Tenki's positive DURABLE state can authorize a restore
            # create. Preserve the pointer and retry on a later lifecycle.
            raise RuntimeError(
                f"snapshot {snapshot_id} is not durably restorable "
                f"(state={state!r}, durability={durability!r})"
            )

    def _create_sandbox_with_snapshot_fallback(self):
        self._preflight_snapshot_restore()
        kwargs = self._create_kwargs()
        try:
            sandbox = self._create_sandbox_from_kwargs(kwargs)
        except Exception as exc:
            if not self._snapshot_restore_id:
                raise
            if not self._snapshot_unrecoverable(exc):
                # Ambiguous/transient failure — keep the snapshot pointer so a
                # later attempt can still recover, rather than deleting it and
                # booting a blank base image (silent loss of persistent state).
                logger.warning(
                    "Tenki: snapshot restore %s for task %s failed transiently (%s); "
                    "preserving it for retry",
                    self._snapshot_restore_id,
                    self._task_id,
                    exc,
                )
                raise
            # Client.create can commit and then raise while decoding even with
            # wait=False. Reconcile the exact durable attempt before changing
            # the snapshot pointer or issuing another non-idempotent create.
            if not self._reconcile_uncertain_create():
                raise RuntimeError(
                    "Tenki could not reconcile the rejected snapshot "
                    f"create attempt for task {self._task_id}"
                ) from exc
            if self._sandbox is not None:
                # The supposedly failed RPC actually committed. Preserve and
                # validate that exact restored sandbox instead of retrying.
                return self._sandbox
            self._retire_unrecoverable_restore_pointer(exc)
            sandbox = self._create_sandbox_from_kwargs(self._create_kwargs())
        return sandbox

    def _migrate_loaded_snapshot_pointer(self) -> None:
        """Best-effort move of a loaded legacy pointer to the canonical key.

        Migration is metadata housekeeping performed *after* the remote
        sandbox has been restored. A local store/fsync/delete failure must not
        discard ownership of that known-good live sandbox; leave the source
        pointer and migration flags intact so a later call can retry.
        """
        if not self._snapshot_restore_id or not (
            self._snapshot_restore_from_legacy_key
            or self._snapshot_restore_task_id != self._task_id
        ):
            return
        source_task_id = self._snapshot_restore_task_id
        try:
            _migrate_snapshot_pointer(
                self._task_id,
                source_task_id,
                self._snapshot_restore_id,
                self._snapshot_store,
            )
        except _SnapshotPointerConflict:
            # This is not a retryable local-I/O failure: another writer has
            # advanced or retired recovery state. Never overwrite/delete that
            # newer pointer with this wrapper's stale legacy snapshot.
            raise
        except Exception as exc:
            logger.warning(
                "Tenki: could not migrate snapshot pointer for task %s from "
                "%s; keeping the restored sandbox and source pointer for a "
                "later retry: %s",
                self._task_id,
                source_task_id,
                exc,
            )
            return
        self._snapshot_restore_task_id = self._task_id
        self._snapshot_restore_from_legacy_key = False

    def _retry_pending_snapshot_retirements(self) -> None:
        """Best-effort retry of durable, profile-scoped remote deletions."""
        try:
            _confirm_snapshot_store_durable(self._snapshot_store)
            pending = _pending_snapshot_retirements(self._snapshot_store)
        except Exception as exc:
            logger.warning(
                "Tenki: could not load pending snapshot retirements for "
                "profile %s: %s",
                self._profile_token,
                exc,
            )
            return
        for snapshot_id in pending:
            self._retire_pending_snapshot_if_unreferenced(
                snapshot_id,
                reason="retrying pending retirement",
            )

    def _retire_pending_snapshot_if_unreferenced(
        self,
        snapshot_id: str,
        *,
        reason: str,
    ) -> bool:
        """Reference-check, remote-delete, and tombstone under one store lock."""
        path = self._snapshot_store
        with _snapshot_store_lock(path):
            snapshots = _load_recovery_registry(path)
            pending_key = _snapshot_retirement_key(snapshot_id)
            if snapshots.get(pending_key) != snapshot_id:
                return True
            metadata_prefixes = (
                f"{_SNAPSHOT_RETIREMENT_NAMESPACE}:",
                f"{_SNAPSHOT_RETIRED_NAMESPACE}:",
                f"{_CREATE_ATTEMPT_NAMESPACE}:",
                f"{_REMOTE_BINDING_NAMESPACE}:",
            )
            referenced = any(
                isinstance(value, str)
                and value == snapshot_id
                and not key.startswith(metadata_prefixes)
                for key, value in snapshots.items()
            )
            if referenced:
                return False

            # Publish the permanent non-resurrection claim before deleting the
            # remote, while retaining the pending retry record. If this commit
            # fails or is uncertain, leave the remote intact. A later lifecycle
            # first fsyncs visible uncertain state and then safely retries.
            retired_key = _snapshot_retired_key(snapshot_id)
            if snapshots.get(retired_key) != snapshot_id:
                snapshots[retired_key] = snapshot_id
                try:
                    _atomic_save_snapshots(path, snapshots)
                except Exception as exc:
                    logger.warning(
                        "Tenki: could not durably claim retirement of snapshot "
                        "%s (%s); leaving the remote intact: %s",
                        snapshot_id,
                        reason,
                        exc,
                    )
                    return False

            remote_snapshots = getattr(self._client, "snapshots", None)
            delete = getattr(remote_snapshots, "delete", None)
            if not callable(delete):
                logger.warning(
                    "Tenki: cannot retire snapshot %s (%s): SDK delete API "
                    "unavailable",
                    snapshot_id,
                    reason,
                )
                return False
            try:
                delete(snapshot_id)
            except Exception as exc:
                if not self._snapshot_unrecoverable(exc):
                    logger.warning(
                        "Tenki: could not retire snapshot %s (%s): %s",
                        snapshot_id,
                        reason,
                        exc,
                    )
                    return False

            # The tombstone was durable before the delete. Completion only
            # clears retry metadata; the tombstone remains indefinitely.
            snapshots.pop(pending_key, None)
            try:
                _atomic_save_snapshots(path, snapshots)
            except Exception as exc:
                # The remote is gone, but the durable tombstone and pending
                # record remain safe. A later not-found retry clears metadata.
                logger.warning(
                    "Tenki: retired remote snapshot %s but could not durably "
                    "clear its pending-retirement record: %s",
                    snapshot_id,
                    exc,
                )
                return True
            logger.info("Tenki: retired snapshot %s (%s)", snapshot_id, reason)
            return True

    def _remote_transfer_path(self, prefix: str) -> str:
        base = (self._remote_home or "/home/tenki").rstrip("/") or "/home/tenki"
        if base != "/home/tenki" and not base.startswith("/home/tenki/"):
            base = "/home/tenki"
        nonce = uuid.uuid4().hex
        return f"{base}/{prefix}.{os.getpid()}.{self._session_id}.{nonce}.tar"

    def _profile_sync_files(self) -> list[tuple[str, str]]:
        context = self._profile_sync_context.copy()
        return context.run(
            iter_sync_files,
            f"{self._remote_home}/.hermes",
        )

    def _profile_credential_host_paths(self) -> set[str]:
        context = self._profile_sync_context.copy()
        return context.run(_credential_host_paths)

    def _resolve_remote_home(self) -> None:
        try:
            result = self._exec_raw("echo \"$HOME\"", timeout=15)
            home = result[0].strip() if result[1] == 0 else ""
            if home:
                self._remote_home = home
                if self.cwd in {"~", "/home/tenki"}:
                    self.cwd = home
        except Exception:
            pass

    def _tenki_upload(self, host_path: str, remote_path: str) -> None:
        # One capture for the whole flow: cancel() nulls out self._sandbox, so
        # re-reading it per call could send the mkdir and the upload to two
        # different sandboxes (or to None).
        with self._sandbox_operation() as sandbox:
            parent = str(Path(remote_path).parent)
            sandbox.fs.mkdir(parent, recursive=True)
            sandbox.fs.upload(host_path, remote_path)

    def _tenki_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return

        # Same single-capture rule as _tenki_upload: the mkdir, the tar upload,
        # the untar, and the cleanup rm must all target one sandbox, or a
        # concurrent cancel() can leave the tar extracted into a different
        # sandbox than the one it was uploaded to.
        with self._sandbox_operation() as sandbox:
            parents = unique_parent_dirs(files)
            if parents:
                self._exec_raw_on_sandbox(
                    sandbox,
                    quoted_mkdir_command(parents),
                    timeout=30,
                )

            remote_tar = self._remote_transfer_path(".hermes_tenki_sync")
            with tempfile.NamedTemporaryFile(suffix=".tar") as tmp:
                with tarfile.open(fileobj=tmp, mode="w") as tar:
                    for host_path, remote_path in files:
                        tar.add(host_path, arcname=remote_path.lstrip("/"))
                tmp.flush()
                sandbox.fs.upload(tmp.name, remote_tar)

            try:
                output, exit_code = self._exec_raw_on_sandbox(
                    sandbox,
                    f"tar xf {shlex.quote(remote_tar)} -C /",
                    timeout=120,
                )
                if exit_code != 0:
                    raise RuntimeError(
                        f"Tenki bulk upload failed (exit {exit_code}): {output}"
                    )
            finally:
                try:
                    self._exec_raw_on_sandbox(
                        sandbox,
                        f"rm -f {shlex.quote(remote_tar)}",
                        timeout=10,
                    )
                except Exception:
                    pass

    def _tenki_bulk_download(self, dest: Path) -> None:
        with self._transfer_operation() as sandbox:
            remote_tar = self._remote_transfer_path(".hermes_tenki_sync_back")
            rel_base = f"{self._remote_home}/.hermes".lstrip("/")
            try:
                output, exit_code = self._exec_raw_on_sandbox(
                    sandbox,
                    f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}",
                    timeout=120,
                )
                if exit_code != 0:
                    raise RuntimeError(
                        f"Tenki bulk download failed (exit {exit_code}): {output}"
                    )
                sandbox.fs.download(remote_tar, str(dest))
            finally:
                try:
                    self._exec_raw_on_sandbox(
                        sandbox,
                        f"rm -f {shlex.quote(remote_tar)}",
                        timeout=10,
                    )
                except Exception:
                    pass

    def _tenki_delete(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return
        self._exec_raw(quoted_rm_command(remote_paths), timeout=30)

    def _exec_raw(self, command: str, *, login: bool = False, timeout: int = 120) -> tuple[str, int]:
        with self._sandbox_operation() as sandbox:
            return self._exec_raw_on_sandbox(
                sandbox,
                command,
                login=login,
                timeout=timeout,
            )

    def _exec_raw_on_sandbox(
        self,
        sandbox: Any,
        command: str,
        *,
        login: bool = False,
        timeout: int = 120,
    ) -> tuple[str, int]:
        flag = "-lc" if login else "-c"
        result = sandbox.exec("bash", flag, command, timeout=timeout, env=self._sandbox_env())
        return self._result_to_output(result)

    @staticmethod
    def _result_to_output(result: Any) -> tuple[str, int]:
        stdout = _text(_result_attr(result, ("stdout_text", "stdout", "output", "result", "text")))
        stderr = _text(_result_attr(result, ("stderr_text", "stderr")))
        exit_code = _result_attr(result, ("exit_code", "returncode", "status_code"))
        if exit_code is None:
            ok = _result_attr(result, ("ok", "success"))
            exit_code = 0 if ok is True else 1
        if stdout and stderr and not stdout.endswith("\n"):
            output = stdout + "\n" + stderr
        else:
            output = stdout + stderr
        return output, int(exit_code)

    def _start_process(
        self,
        cmd_string: str,
        *,
        login: bool,
        timeout: int,
        stdin_data: str | None,
        process_ref: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        with self._sandbox_operation() as sandbox:
            flag = "-lc" if login else "-c"
            start = getattr(sandbox, "start", None)
            if not callable(start):
                kwargs: dict[str, Any] = {
                    "timeout": timeout,
                    "env": self._sandbox_env(),
                }
                if stdin_data is not None:
                    kwargs["input"] = stdin_data
                result = sandbox.exec("bash", flag, cmd_string, **kwargs)
                return self._result_to_output(result)

            process = start(
                "bash",
                flag,
                cmd_string,
                timeout=timeout,
                stdin=stdin_data,
                env=self._sandbox_env(),
            )
            if process_ref is not None:
                process_ref["process"] = process
            if stdin_data is None:
                close_stdin = getattr(process, "close_stdin", None)
                if callable(close_stdin):
                    close_stdin()
            result = process.wait(timeout=timeout + 5 if timeout is not None else None)
            return self._result_to_output(result)

    def _sudo_nopasswd_works(self) -> bool:
        try:
            _output, exit_code = self._exec_raw("sudo -n true", timeout=10)
        except Exception:
            return False
        return exit_code == 0

    def _prepare_command(self, command: str | None) -> tuple[str | None, str | None]:
        if command is None:
            return None, None

        # Tenki sandboxes should rely on their own sudoers policy. Do not ask
        # the user for a host sudo password, and do not send SUDO_PASSWORD to a
        # remote cloud sandbox. The default Tenki image supports NOPASSWD sudo.
        from tools.terminal_tool import _rewrite_sudo_command_words

        transformed, sudo_count = _rewrite_sudo_command_words(command, "sudo -n")
        if sudo_count == 0:
            return command, None
        if self._sudo_nopasswd_works():
            return command, None
        return transformed, None

    def _before_execute(self) -> None:
        self._ensure_sandbox()
        if self._sync_manager:
            self._sync_manager.sync()

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        process_ref: dict[str, Any] = {}

        def cancel() -> None:
            process = process_ref.get("process")
            kill = getattr(process, "kill", None)
            if callable(kill):
                try:
                    kill()
                    return
                except Exception:
                    pass
            with self._lifecycle_condition:
                # cleanup owns the sandbox once it starts. Let that one owner
                # finish instead of racing it with a second pause/terminate.
                if self._cleanup_in_progress or self._cleanup_complete:
                    return
                sandbox = self._sandbox
                # Persistent cancellation must retain this exact reference:
                # discovery can fail transiently, and recreating from an older
                # snapshot would fork/strand unsnapshotted state. Ephemeral
                # sandboxes are deliberately detached before termination.
                if not self._persistent:
                    self._sandbox = None
                    if sandbox is not None and self._create_attempt_id is not None:
                        self._create_outcome_uncertain = True
                if sandbox is not None:
                    self._cancel_in_progress += 1
            if sandbox is None:
                return
            try:
                # For a persistent sandbox, pause (preserve the filesystem)
                # instead of terminating: an interrupted or timed-out command
                # must not destroy state the user asked to keep. The paused
                # sandbox is re-discovered and resumed on the next command.
                if self._persistent:
                    pause = getattr(sandbox, "pause", None)
                    if callable(pause):
                        try:
                            pause()
                            return
                        except Exception as exc:
                            logger.warning(
                                "Tenki: cancel could not pause persistent sandbox "
                                "for task %s; leaving it live to preserve "
                                "unsnapshotted state: %s",
                                self._task_id,
                                exc,
                            )
                    else:
                        logger.warning(
                            "Tenki: cancel found no pause support for persistent "
                            "task %s; leaving sandbox live to preserve "
                            "unsnapshotted state",
                            self._task_id,
                        )
                    return
                for method_name in ("terminate", "close"):
                    method = getattr(sandbox, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception as exc:
                            logger.warning(
                                "Tenki: cancel could not dispose ephemeral "
                                "sandbox for task %s: %s",
                                self._task_id,
                                exc,
                            )
                            continue
                        if not self._clear_remote_binding_marker(sandbox):
                            logger.error(
                                "Tenki: cancel terminated task %s but could not "
                                "clear its durable remote binding",
                                self._task_id,
                            )
                        return
            finally:
                with self._lifecycle_condition:
                    if self._persistent and not self._cleanup_complete:
                        # Reassert exact ownership before releasing ensure() or
                        # cleanup() waiters, including pause failure/missing
                        # paths.
                        self._sandbox = sandbox
                    self._cancel_generation += 1
                    self._cancel_in_progress -= 1
                    self._lifecycle_condition.notify_all()

        def exec_fn() -> tuple[str, int]:
            return self._start_process(
                cmd_string,
                login=login,
                timeout=timeout,
                stdin_data=stdin_data,
                process_ref=process_ref,
            )

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    @staticmethod
    def _sandbox_identity(sandbox: Any) -> str | None:
        if sandbox is None:
            return None
        identity = getattr(sandbox, "id", None) or getattr(
            sandbox,
            "sandbox_id",
            None,
        )
        return str(identity) if identity else None

    @staticmethod
    def _remote_identity(env: Any) -> str | None:
        return TenkiEnvironment._sandbox_identity(
            getattr(env, "_sandbox", None)
        )

    def _bind_remote_sandbox(
        self,
        sandbox: Any,
        *,
        attempt_id: str | None,
        validated: bool | None = None,
    ) -> bool:
        remote_id = self._sandbox_identity(sandbox)
        if remote_id is None:
            return False
        if validated is None:
            validated = False
        try:
            _store_remote_binding(
                self._task_id,
                remote_id,
                attempt_id,
                validated=validated,
                store_path=self._snapshot_store,
            )
        except BaseException as exc:
            logger.error(
                "Tenki: could not durably bind remote %s for task %s: %s",
                remote_id,
                self._task_id,
                exc,
            )
            return False
        self._remote_binding = _RemoteBinding(
            remote_id=remote_id,
            attempt_id=attempt_id,
            validated=validated,
        )
        if attempt_id is not None:
            self._create_attempt_id = None
            self._create_attempt_expires_at = None
        return True

    def _clear_remote_binding_marker(self, sandbox: Any) -> bool:
        remote_id = (
            self._remote_binding.remote_id or self._sandbox_identity(sandbox)
        )
        if remote_id is None:
            return True
        try:
            _clear_remote_binding(
                self._task_id,
                remote_id,
                self._snapshot_store,
            )
        except BaseException as exc:
            logger.error(
                "Tenki: could not durably clear remote binding %s for task "
                "%s: %s",
                remote_id,
                self._task_id,
                exc,
            )
            return False
        self._remote_binding = _RemoteBinding()
        return True

    @staticmethod
    def _remote_definitively_absent(exc: BaseException) -> bool:
        try:
            from tenki import SessionNotFoundError
        except (ImportError, AttributeError):
            pass
        else:
            if isinstance(exc, SessionNotFoundError):
                return True
        return any(
            typ.__name__ == "SessionNotFoundError"
            for typ in type(exc).__mro__
        )

    def _resolve_remote_binding(self) -> None:
        """Resolve a durable task binding through authoritative Client.get."""
        binding = self._remote_binding
        remote_id = binding.remote_id
        if (
            (remote_id is None and not binding.conflicted)
            or self._sandbox is not None
        ):
            return
        if binding.conflicted:
            if not binding.conflict_ids and not binding.unresolvable:
                self._fail_ambiguous_lineage(
                    "Tenki task has a malformed durable persistent-lineage "
                    f"conflict for {self._task_id}"
                )
            client = self._create_client()
            active_ids: list[str] = []
            for conflict_id in binding.conflict_ids:
                try:
                    candidate = client.get(conflict_id)
                except BaseException as exc:
                    if self._remote_definitively_absent(exc):
                        continue
                    self._fail_ambiguous_lineage(
                        "Tenki could not resolve every remote in the durable "
                        f"lineage conflict for task {self._task_id}",
                        exc,
                    )
                if self._sandbox_identity(candidate) != conflict_id:
                    self._fail_ambiguous_lineage(
                        "Tenki conflict lookup returned a different sandbox id"
                    )
                if self._sandbox_state(candidate) not in self._terminal_states:
                    active_ids.append(conflict_id)
            if active_ids:
                self._fail_ambiguous_lineage(
                    "Tenki task has a durable persistent-lineage conflict; "
                    f"active remote ids for {self._task_id}: "
                    + ", ".join(active_ids)
                )
            if binding.unresolvable:
                known_ids = ", ".join(binding.conflict_ids) or "<none>"
                self._fail_ambiguous_lineage(
                    "Tenki task has an unresolvable durable "
                    f"persistent-lineage conflict for {self._task_id}; "
                    f"known remote ids: {known_ids}; manual remote cleanup "
                    "and local conflict removal are required"
                )
            if not self._clear_remote_binding_marker(None):
                self._fail_ambiguous_lineage(
                    "Tenki could not clear a fully terminated lineage conflict "
                    f"for task {self._task_id}"
                )
            return
        client = self._create_client()
        get_sandbox = getattr(client, "get", None)
        if not callable(get_sandbox):
            raise RuntimeError(
                "Tenki SDK cannot resolve a durable sandbox binding by id"
            )
        try:
            sandbox = get_sandbox(remote_id)
        except BaseException as exc:
            if self._remote_definitively_absent(exc):
                if not self._clear_remote_binding_marker(None):
                    raise RuntimeError(
                        "Tenki remote is absent but its durable binding could "
                        f"not be cleared for task {self._task_id}"
                    ) from exc
                return
            raise RuntimeError(
                f"Tenki could not resolve bound remote {remote_id}"
            ) from exc

        if self._sandbox_identity(sandbox) != remote_id:
            raise RuntimeError(
                "Tenki by-id lookup returned a different sandbox identity"
            )
        if self._sandbox_state(sandbox) in self._terminal_states:
            if not self._clear_remote_binding_marker(sandbox):
                raise RuntimeError(
                    "Tenki terminal remote binding could not be cleared for "
                    f"task {self._task_id}"
                )
            return
        if self._persistent:
            self._sandbox = sandbox
            if not self._remote_binding.validated:
                self._validate_created_lineage()
            return

        # Not shared with _dispose_remote: the binding clear runs inside the
        # walk and gates it, so a successful terminate() whose clear failed
        # deliberately falls through to close() for a second clear attempt.
        last_exc: BaseException | None = None
        for method_name in ("terminate", "close"):
            method = getattr(sandbox, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except BaseException as exc:
                last_exc = exc
                continue
            if self._clear_remote_binding_marker(sandbox):
                return
            last_exc = RuntimeError(
                "remote terminated but its binding could not be cleared"
            )
        raise RuntimeError(
            f"Tenki could not dispose bound ephemeral remote {remote_id}: "
            f"{last_exc or 'no supported cleanup method'}"
        )

    def shares_remote_resource_with(self, other: Any) -> bool:
        """Whether two wrappers point at the same Tenki sandbox."""
        mine = self._remote_identity(self)
        theirs = self._remote_identity(other)
        return bool(mine and theirs and mine == theirs)

    @staticmethod
    def _sandbox_create_attempt(sandbox: Any) -> str | None:
        info = getattr(sandbox, "info", None)
        metadata = getattr(info, "metadata", {}) if info is not None else {}
        if not isinstance(metadata, dict):
            return None
        attempt_id = metadata.get("hermes_create_attempt")
        return (
            str(attempt_id)
            if isinstance(attempt_id, str) and attempt_id
            else None
        )

    def _clear_create_attempt_marker(self) -> bool:
        attempt_id = self._create_attempt_id
        if attempt_id is None:
            self._create_outcome_uncertain = False
            return True
        try:
            _clear_create_attempt(
                self._task_id,
                attempt_id,
                self._snapshot_store,
            )
        except BaseException as exc:
            logger.error(
                "Tenki: could not durably clear create attempt %s for task "
                "%s: %s",
                attempt_id,
                self._task_id,
                exc,
            )
            return False
        self._create_attempt_id = None
        self._create_attempt_expires_at = None
        self._create_outcome_uncertain = False
        return True

    def _persist_create_attempt_conflict(
        self,
        remote_ids: list[str],
        *,
        unresolvable: bool,
    ) -> bool:
        """Durably retain an observed fork before releasing attempt state."""
        attempt_id = self._create_attempt_id
        if attempt_id is None:
            return False
        try:
            _replace_create_attempt_with_lineage_conflict(
                self._task_id,
                attempt_id,
                remote_ids,
                unresolvable=unresolvable,
                store_path=self._snapshot_store,
            )
        except BaseException as exc:
            self._create_lineage_ambiguous = True
            logger.error(
                "Tenki: could not durably record create-attempt conflict %s "
                "for task %s: %s",
                attempt_id,
                self._task_id,
                exc,
            )
            return False
        ids = tuple(sorted(set(remote_ids)))
        self._create_attempt_id = None
        self._create_attempt_expires_at = None
        self._create_outcome_uncertain = False
        self._remote_binding = _RemoteBinding(
            remote_id=ids[0] if ids else None,
            attempt_id=attempt_id,
            conflicted=True,
            conflict_ids=ids,
            unresolvable=unresolvable,
        )
        self._create_lineage_ambiguous = True
        return True

    @staticmethod
    def _dispose_remote(
        sandbox: Any,
        method_names: tuple[str, ...],
        *,
        delays: tuple[float, ...] = _TERMINATE_RETRY_DELAYS,
    ) -> tuple[bool, BaseException | None]:
        """Try each supported disposal method until one returns cleanly.

        Walks *method_names* in order, skipping names the SDK object does not
        expose, and retries each one over *delays* before moving on. Returns
        ``(disposed, last_exception)`` instead of raising so every caller keeps
        its own failure policy — some log and continue, some quarantine
        ownership. ``BaseException`` is caught deliberately: a disposal that
        dies on ``KeyboardInterrupt`` must still let the caller decide whether
        the remote is safe, and the exception is handed back rather than
        swallowed.

        Callers that must run extra work (a durable binding clear) *inside* the
        walk cannot use this — see ``cleanup`` and ``_resolve_remote_binding``.
        """
        last_exc: BaseException | None = None
        for method_name in method_names:
            method = getattr(sandbox, method_name, None)
            if not callable(method):
                continue
            for attempt in range(len(delays) + 1):
                try:
                    method()
                    return True, last_exc
                except BaseException as exc:
                    last_exc = exc
                    if attempt < len(delays):
                        time.sleep(delays[attempt])
        return False, last_exc

    def _reconcile_uncertain_create(self) -> bool:
        """Resolve exactly one durable create attempt without touching siblings."""
        attempt_id = self._create_attempt_id
        if attempt_id is None:
            self._create_outcome_uncertain = False
            return True
        try:
            client = self._create_client()
        except BaseException as exc:
            logger.error(
                "Tenki: could not create a client to reconcile attempt %s for "
                "task %s: %s",
                attempt_id,
                self._task_id,
                exc,
            )
            return False
        list_sandboxes = getattr(client, "list", None)
        if not callable(list_sandboxes):
            return False
        try:
            candidates = list_sandboxes(**self._list_kwargs(client))
        except BaseException as exc:
            logger.error(
                "Tenki: could not reconcile uncertain create for task %s: %s",
                self._task_id,
                exc,
            )
            return False

        exact_candidates: dict[str, Any] = {}
        exact_without_id = False
        other_active: dict[str, Any] = {}
        other_without_id = False
        try:
            for candidate in candidates:
                if not self._sandbox_matches_task(candidate):
                    continue
                state = self._sandbox_state(candidate)
                exact = self._sandbox_create_attempt(candidate) == attempt_id
                candidate_id = self._sandbox_identity(candidate)
                if exact:
                    if candidate_id is None:
                        exact_without_id = True
                    else:
                        exact_candidates[candidate_id] = candidate
                    continue
                if state in self._terminal_states:
                    continue
                if candidate_id is None:
                    other_without_id = True
                else:
                    other_active[candidate_id] = candidate
        except BaseException as exc:
            logger.error(
                "Tenki: could not inspect uncertain create candidates for task "
                "%s: %s",
                self._task_id,
                exc,
            )
            return False

        expired = (
            self._create_attempt_expires_at is not None
            and time.time() >= self._create_attempt_expires_at
        )

        if (
            exact_without_id
            or other_without_id
            or other_active
            or len(exact_candidates) > 1
        ):
            # Persist the conflict before returning. If the exact create is not
            # visible and has not expired, its unknown identity must also be
            # represented, which makes the conflict permanently unresolvable
            # rather than allowing a later list omission to promote a branch.
            unresolvable = (
                exact_without_id
                or other_without_id
                or (not exact_candidates and not expired)
            )
            conflict_ids = [
                *exact_candidates,
                *other_active,
            ]
            self._persist_create_attempt_conflict(
                conflict_ids,
                unresolvable=unresolvable,
            )
            logger.error(
                "Tenki: create attempt %s for task %s conflicts with %s other "
                "active lineage(s) and %s duplicate exact attempt(s)",
                attempt_id,
                self._task_id,
                len(other_active),
                max(0, len(exact_candidates) - 1),
            )
            return False

        if not exact_candidates:
            if expired and not other_active:
                # Even if the RPC committed immediately before a crash, the
                # configured server-side max duration plus a one-hour grace has
                # elapsed. No remote from this attempt can remain live.
                return self._clear_create_attempt_marker()
            # A timeout can race an eventually consistent list. An empty result
            # is therefore not proof of absence; fail closed by quarantining the
            # task lock instead of allowing a duplicate create.
            logger.error(
                "Tenki: create attempt %s for task %s returned no exact "
                "reconcilable sandbox; absence is unproven",
                attempt_id,
                self._task_id,
            )
            return False

        remote_id = next(iter(exact_candidates))
        get_sandbox = getattr(client, "get", None)
        if not callable(get_sandbox):
            return False
        try:
            sandbox = get_sandbox(remote_id)
        except BaseException as exc:
            if self._remote_definitively_absent(exc):
                return self._clear_create_attempt_marker()
            logger.error(
                "Tenki: could not authoritatively resolve uncertain create "
                "%s for task %s: %s",
                attempt_id,
                self._task_id,
                exc,
            )
            return False
        if (
            self._sandbox_identity(sandbox) != remote_id
            or not self._sandbox_has_owned_identity(sandbox, attempt_id)
        ):
            self._persist_create_attempt_conflict(
                [],
                unresolvable=True,
            )
            logger.error(
                "Tenki: exact-id recovery did not preserve canonical ownership "
                "for attempt %s and task %s",
                attempt_id,
                self._task_id,
            )
            return False
        if self._sandbox_state(sandbox) in self._terminal_states:
            return self._clear_create_attempt_marker()

        # A terminal list row cannot authorize clearing: Client.get(id) above
        # is authoritative and may reveal that the same remote is still live.
        if self._persistent:
            # Bind the exact recovered attempt, but keep it unvalidated until
            # the normal exact-identity + conflict scan completes.
            if not self._bind_remote_sandbox(
                sandbox,
                attempt_id=attempt_id,
                validated=False,
            ):
                return False
            self._sandbox = sandbox
            self._create_outcome_uncertain = False
            return True

        disposed, last_exc = self._dispose_remote(sandbox, ("terminate", "close"))
        if not disposed:
            logger.error(
                "Tenki: could not dispose exact uncertain create %s for task "
                "%s: %s",
                attempt_id,
                self._task_id,
                last_exc or "no supported remote cleanup method",
            )
            return False
        return self._clear_create_attempt_marker()

    def _abort_failed_initialization(self) -> None:
        """Synchronously make a partially initialized wrapper inert.

        The lifetime task lock covers the whole rollback. Persistent sandboxes
        are paused so their state remains discoverable; ephemeral sandboxes are
        terminated. If the remote cannot be made safe, the kernel lock is
        quarantined for the rest of the process so no successor can attach.
        """
        sandbox = getattr(self, "_sandbox", None)
        client = getattr(self, "_client", None)
        create_uncertain = getattr(self, "_create_outcome_uncertain", False)
        remote_safe = sandbox is None and not create_uncertain
        last_exc: BaseException | None = None

        try:
            if self._create_lineage_ambiguous:
                last_exc = RuntimeError(
                    "multiple remote lineages require fail-closed ownership"
                )
            elif sandbox is None and create_uncertain:
                remote_safe = self._reconcile_uncertain_create()
                if not remote_safe:
                    last_exc = RuntimeError(
                        "uncertain remote creation could not be reconciled"
                    )
            elif sandbox is not None:
                method_names = (
                    ("pause",)
                    if self._persistent
                    else ("terminate", "close")
                )
                remote_safe, last_exc = self._dispose_remote(
                    sandbox,
                    method_names,
                )
        except BaseException as exc:
            # Rollback itself must never obscure the constructor failure or
            # release ownership in an unknown remote state.
            last_exc = exc

        if remote_safe and self._create_lineage_ambiguous:
            remote_safe = False
            last_exc = RuntimeError(
                "multiple remote lineages require fail-closed ownership"
            )
        if (
            remote_safe
            and sandbox is not None
            and not self._persistent
            and not self._clear_remote_binding_marker(sandbox)
        ):
            remote_safe = False
            last_exc = RuntimeError(
                "terminated remote binding could not be cleared"
            )

        with self._lifecycle_condition:
            self._sandbox = None
            self._client = None
            self._sync_manager = None
            self._cleanup_sandbox = None
            self._cleanup_in_progress = False
            self._cleanup_complete = True
            self._lifecycle_condition.notify_all()
        try:
            self._close_client(client)
        except BaseException as exc:
            logger.warning(
                "Tenki: client close raised during failed initialization: %s",
                exc,
            )

        if remote_safe:
            self._release_task_ownership()
            return

        lock_file = getattr(self, "_task_ownership_file", None)
        self._task_ownership_file = None
        _quarantine_task_ownership_lock(lock_file)
        logger.error(
            "Tenki: failed initialization could not safely quiesce task %s; "
            "retaining exclusive task ownership until process exit: %s",
            self._task_id,
            last_exc or "no supported remote cleanup method",
        )

    def _release_task_ownership(self) -> None:
        lock_file = getattr(self, "_task_ownership_file", None)
        self._task_ownership_file = None
        _release_task_ownership_lock(lock_file)

    def release_duplicate_wrapper(self, winner: Any) -> None:
        """Close only this wrapper/client when *winner* owns the same remote."""
        with self._lifecycle_condition:
            while (
                self._cleanup_in_progress
                or self._cancel_in_progress
                or self._active_operations
            ):
                self._lifecycle_condition.wait()
            if self._cleanup_complete:
                return
            client = self._client
            if client is getattr(winner, "_client", None):
                client = None
            self._sandbox = None
            self._client = None
            self._sync_manager = None
            self._cleanup_complete = True
            self._cleanup_in_progress = False
            self._cleanup_sandbox = None
            self._lifecycle_condition.notify_all()
        self._close_client(client)
        self._release_task_ownership()

    def discard(self) -> None:
        """Dispose an unregistered duplicate without snapshot side effects."""
        self.cleanup(discard=True)

    def cleanup(self, *, discard: bool = False):
        with self._lifecycle_condition:
            while self._cleanup_in_progress:
                self._lifecycle_condition.wait()
            if self._cleanup_complete:
                return

            self._cleanup_in_progress = True
            # cancel() may already own a detached sandbox while it performs a
            # pause/terminate RPC outside the lock. Do not close the shared
            # control-plane client until that RPC has completed.
            while self._cancel_in_progress:
                self._lifecycle_condition.wait()
            while self._active_operations:
                self._lifecycle_condition.wait()

            sandbox = self._sandbox
            sync_manager = self._sync_manager
            self._sync_manager = None
            client = self._client
            self._cleanup_sandbox = sandbox

        cleanup_succeeded = False
        try:
            if self._create_lineage_ambiguous:
                raise RuntimeError(
                    "Tenki cleanup is blocked because multiple remote lineages "
                    f"exist for task {self._task_id}"
                )
            if sandbox is None:
                if self._remote_binding.remote_id is not None:
                    self._resolve_remote_binding()
                    sandbox = self._sandbox
                    client = self._client
                    with self._lifecycle_condition:
                        self._cleanup_sandbox = sandbox
                if self._create_attempt_id is not None:
                    self._create_outcome_uncertain = True
                    if not self._reconcile_uncertain_create():
                        raise RuntimeError(
                            "Tenki cleanup cannot release task ownership while "
                            f"a create is unresolved for task {self._task_id}"
                        )
                if sandbox is None:
                    cleanup_succeeded = True
                    return

            if sync_manager and not discard:
                logger.info("Tenki: syncing files from sandbox...")
                try:
                    sync_manager.sync_back(self._profile_home)
                except Exception as exc:
                    logger.warning("Tenki: sync_back failed: %s", exc)

            snapshot_saved = False
            if self._persistent and not discard:
                snapshot_saved = self._save_persistent_snapshot(sandbox)

            if self._persistent and not discard and not snapshot_saved:
                # Persistent state was NOT durably snapshotted. Terminating now
                # would destroy the only copy, so prefer pause; and if pause
                # fails, still do NOT terminate — leave the sandbox live for a
                # later recovery attempt (the max-duration / idle reaper bounds
                # the cost). Terminating here would break the preservation
                # guarantee that the durability gate exists to uphold.
                pause = getattr(sandbox, "pause", None)
                if callable(pause):
                    try:
                        pause()
                        logger.info("Tenki: paused sandbox for task %s", self._task_id)
                    except Exception as exc:
                        logger.warning(
                            "Tenki: pause failed for task %s; leaving sandbox live to "
                            "preserve un-snapshotted state (not terminating): %s",
                            self._task_id, exc,
                        )
                else:
                    logger.warning(
                        "Tenki: no durable snapshot and no pause support for task %s; "
                        "leaving sandbox live to preserve state (not terminating)",
                        self._task_id,
                    )
                cleanup_succeeded = True
                return

            # Not shared with _dispose_remote: the durable binding clear runs
            # INSIDE the retry body, so a failed clear deliberately re-invokes
            # terminate() on the next attempt. Hoisting the clear out of the
            # loop would drop that retry.
            for method_name in ("terminate", "close"):
                method = getattr(sandbox, method_name, None)
                if not callable(method):
                    continue
                last_exc: Exception | None = None
                attempts = len(_TERMINATE_RETRY_DELAYS) + 1
                for attempt in range(attempts):
                    try:
                        method()
                        if not self._clear_remote_binding_marker(sandbox):
                            raise RuntimeError(
                                "remote terminated but its durable create "
                                "binding could not be cleared"
                            )
                        logger.info(
                            "Tenki: terminated sandbox for task %s",
                            self._task_id,
                        )
                        cleanup_succeeded = True
                        return
                    except Exception as exc:
                        last_exc = exc
                        if attempt < len(_TERMINATE_RETRY_DELAYS):
                            time.sleep(_TERMINATE_RETRY_DELAYS[attempt])
                raise RuntimeError(
                    f"Tenki cleanup failed after {attempts} attempts "
                    f"for task {self._task_id}: {last_exc}"
                ) from last_exc
            raise RuntimeError(
                f"Tenki sandbox for task {self._task_id} has no cleanup method"
            )
        finally:
            if cleanup_succeeded:
                self._close_client(client)
            with self._lifecycle_condition:
                if cleanup_succeeded:
                    if self._sandbox is sandbox:
                        self._sandbox = None
                    if self._client is client:
                        self._client = None
                    self._cleanup_complete = True
                self._cleanup_in_progress = False
                self._cleanup_sandbox = None
                self._lifecycle_condition.notify_all()
            if cleanup_succeeded:
                self._release_task_ownership()

    def _save_persistent_snapshot(self, sandbox: Any) -> bool:
        # A legacy source pointer is the predecessor of any new canonical
        # snapshot. If its migration still cannot commit, writing a new
        # canonical pointer would forget that predecessor and leak its remote
        # snapshot. Retry here (cleanup may be the first post-construction
        # lifecycle), then fail closed and pause if it remains unresolved.
        self._migrate_loaded_snapshot_pointer()
        if self._snapshot_restore_id and (
            self._snapshot_restore_from_legacy_key
            or self._snapshot_restore_task_id != self._task_id
        ):
            logger.warning(
                "Tenki: snapshot pointer migration is still pending for task "
                "%s; preserving the live sandbox instead of replacing an "
                "untracked legacy predecessor",
                self._task_id,
            )
            return False

        snapshot_id: str | None = None
        try:
            snapshot = sandbox.snapshot(name=self._sandbox_name(), wait=True)
            snapshot_id = getattr(snapshot, "id", None) or getattr(snapshot, "snapshot_id", None)
        except Exception as exc:
            logger.warning("Tenki: filesystem snapshot failed: %s", exc)
            return False
        if not snapshot_id:
            logger.warning("Tenki: snapshot completed without an id; preserving paused sandbox instead")
            return False
        # snapshot(wait=True) only waits for READY; durability is a separate,
        # required gate. If durability is not confirmed the snapshot may not be
        # a safe recovery copy, so we must NOT record it as the persistent
        # pointer or let the caller terminate the live sandbox. Return False so
        # cleanup pauses the sandbox and the prior (known-durable) snapshot
        # pointer is left intact for recovery.
        snapshots = getattr(self._client, "snapshots", None)
        wait_durable = getattr(snapshots, "wait_durable", None)
        if not callable(wait_durable):
            logger.warning(
                "Tenki: cannot confirm durability for snapshot %s for task %s; "
                "preserving paused sandbox and prior snapshot instead",
                snapshot_id,
                self._task_id,
            )
            self._delete_remote_snapshot(
                snapshot_id,
                reason="unverifiable replacement",
            )
            return False
        try:
            wait_durable(snapshot_id, timeout=300)
        except Exception as exc:
            logger.warning(
                "Tenki: snapshot %s for task %s did not reach durability (%s); "
                "preserving paused sandbox and prior snapshot instead",
                snapshot_id, self._task_id, exc,
            )
            self._delete_remote_snapshot(
                snapshot_id,
                reason="non-durable replacement",
            )
            return False
        try:
            previous_snapshot_id = _store_snapshot(
                self._task_id,
                snapshot_id,
                self._snapshot_store,
            )
        except _SnapshotPointerCommitUncertain as exc:
            # The rename is visible, but a host crash may reveal either the old
            # or new pointer. Keep both remote snapshots and the live sandbox;
            # deleting either recovery copy would make one crash outcome lossy.
            logger.warning(
                "Tenki: durable snapshot pointer commit for %s is uncertain "
                "(%s; previous=%s, new=%s); retaining both snapshots and "
                "preserving the live sandbox",
                self._task_id,
                exc,
                exc.previous_snapshot_id or "<none>",
                exc.new_snapshot_id or snapshot_id,
            )
            return False
        except Exception as exc:
            logger.warning(
                "Tenki: could not persist durable snapshot pointer %s for task %s "
                "(%s); preserving paused sandbox and prior snapshot instead",
                snapshot_id,
                self._task_id,
                exc,
            )
            self._delete_remote_snapshot(
                snapshot_id,
                reason="unrecorded replacement",
            )
            return False

        logger.info("Tenki: saved filesystem snapshot %s for task %s", snapshot_id, self._task_id)
        if previous_snapshot_id and previous_snapshot_id != snapshot_id:
            self._retire_pending_snapshot_if_unreferenced(
                previous_snapshot_id,
                reason=f"superseded by {snapshot_id}",
            )
        return True

    def _delete_remote_snapshot(
        self,
        snapshot_id: str,
        *,
        reason: str,
    ) -> bool:
        """Safely retire an unreferenced remote snapshot through the journal."""
        try:
            _queue_snapshot_retirement(snapshot_id, self._snapshot_store)
        except Exception as exc:
            logger.warning(
                "Tenki: could not durably journal snapshot %s for retirement "
                "(%s); leaving the remote intact: %s",
                snapshot_id,
                reason,
                exc,
            )
            return False
        return self._retire_pending_snapshot_if_unreferenced(
            snapshot_id,
            reason=reason,
        )

    @staticmethod
    def _close_client(client: Any) -> None:
        # Best-effort: a failed close must never propagate — cleanup() resets
        # _cleanup_in_progress after this call, and an escaping exception would
        # leave the flag stuck and brick the environment.
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning("Tenki: client close failed: %s", exc)
