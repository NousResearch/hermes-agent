"""FastVM execution environment.

Uses the FastVM Python SDK to run Hermes terminal commands in cloud VMs.
Persistent mode snapshots the VM on cleanup, deletes the live VM to stop
compute spend, and restores the task from the latest snapshot on the next use.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import logging
import math
import os
import shlex
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

from hermes_constants import get_hermes_home
from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
    _load_json_store,
    _save_json_store,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)

DEFAULT_FASTVM_CWD = "/root"
DEFAULT_FASTVM_MACHINE = "c1m2"
_DEFAULT_DISK_GIB = 50
_SNAPSHOT_STORE_NAME = "fastvm_snapshots.json"
_LIFECYCLE_LOCK_NAME = ".fastvm.lock"
_READY_SNAPSHOT_STATUSES = frozenset({"ready", "completed"})
_ERROR_SNAPSHOT_STATUSES = frozenset({"error", "failed"})
_DEAD_VM_STATUSES = frozenset({"deleted", "deleting", "error", "failed", "stopped"})
_RUNNING_VM_STATUS = "running"
_VM_READY_TIMEOUT = 45
_VM_READY_POLL_INTERVAL = 2.0
_SNAPSHOT_POLL_INTERVAL = 2.0
_DEFAULT_LEASE_TTL_SECONDS = 900


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_store_path() -> Path:
    return get_hermes_home() / _SNAPSHOT_STORE_NAME


def _lifecycle_lock_path() -> Path:
    return get_hermes_home() / _LIFECYCLE_LOCK_NAME


@contextmanager
def _fastvm_lifecycle_lock():
    """Serialize FastVM state transitions across Hermes processes."""
    path = _lifecycle_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_snapshots() -> dict:
    return _load_json_store(_snapshot_store_path())


def _save_snapshots(data: dict) -> None:
    _save_json_store(_snapshot_store_path(), data)


def _coerce_snapshot_record(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str) and value:
        return {"snapshot_id": value, "leases": {}}
    if not isinstance(value, dict):
        return None
    record = dict(value)
    snapshot_id = value.get("snapshot_id")
    active_vm_id = value.get("active_vm_id")
    has_snapshot = isinstance(snapshot_id, str) and bool(snapshot_id)
    has_active_vm = isinstance(active_vm_id, str) and bool(active_vm_id)
    has_leases = isinstance(value.get("leases"), dict)
    if not (has_snapshot or has_active_vm or has_leases):
        return None
    if not isinstance(record.get("leases"), dict):
        record["leases"] = {}
    else:
        record["leases"] = {
            str(lease_id): dict(lease)
            for lease_id, lease in record["leases"].items()
            if isinstance(lease, dict)
        }
    return record


def _get_snapshot_record(task_id: str) -> dict[str, Any] | None:
    if not task_id:
        return None
    with _fastvm_lifecycle_lock():
        return _coerce_snapshot_record(_load_snapshots().get(task_id))


def _store_snapshot_record(task_id: str, record: dict[str, Any]) -> None:
    snapshot_id = record.get("snapshot_id")
    active_vm_id = record.get("active_vm_id")
    leases = record.get("leases")
    has_snapshot = isinstance(snapshot_id, str) and bool(snapshot_id)
    has_active_vm = isinstance(active_vm_id, str) and bool(active_vm_id)
    has_leases = isinstance(leases, dict)
    if not task_id or not (has_snapshot or has_active_vm or has_leases):
        return
    normalized = _coerce_snapshot_record(record)
    if normalized is None:
        return
    with _fastvm_lifecycle_lock():
        snapshots = _load_snapshots()
        snapshots[task_id] = normalized
        _save_snapshots(snapshots)


def _delete_snapshot_record(task_id: str, snapshot_id: str | None = None) -> None:
    if not task_id:
        return
    with _fastvm_lifecycle_lock():
        snapshots = _load_snapshots()
        record = _coerce_snapshot_record(snapshots.get(task_id))
        if record is None:
            return
        existing = record.get("snapshot_id")
        if snapshot_id is not None and existing != snapshot_id:
            return
        snapshots.pop(task_id, None)
        _save_snapshots(snapshots)


def _extract_id(value: Any) -> str | None:
    for attr in ("id", "snapshot_id", "snapshotId", "vm_id", "vmId"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    if isinstance(value, dict):
        for key in ("id", "snapshot_id", "snapshotId", "vm_id", "vmId"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _extract_status(value: Any) -> str:
    status = getattr(value, "status", None)
    if status is None and isinstance(value, dict):
        status = value.get("status")
    return str(status or "").lower()


def _extract_metadata(value: Any) -> dict[str, str]:
    metadata = getattr(value, "metadata", None)
    if metadata is None and isinstance(value, dict):
        metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    return {str(k): str(v) for k, v in metadata.items() if v is not None}


def _extract_created_at(value: Any) -> str:
    created_at = getattr(value, "created_at", None)
    if created_at is None and isinstance(value, dict):
        created_at = value.get("created_at") or value.get("createdAt")
    if isinstance(created_at, datetime):
        return created_at.astimezone(timezone.utc).isoformat()
    return str(created_at or "")


def _iter_vms_response(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    for attr in ("data", "items", "vms"):
        items = getattr(value, attr, None)
        if items is not None:
            return list(items)
        if isinstance(value, dict) and attr in value:
            return list(value[attr])
    try:
        return list(value)
    except TypeError:
        return []


def _parse_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def _lease_is_stale(lease: dict[str, Any], *, now: datetime, hostname: str) -> bool:
    expires_at = _parse_utc(lease.get("expires_at"))
    if expires_at is not None and expires_at <= now:
        return True
    if str(lease.get("hostname") or "") != hostname:
        return False
    try:
        pid = int(lease.get("pid") or 0)
    except (TypeError, ValueError):
        return True
    return not _process_exists(pid)


def _prune_stale_leases(record: dict[str, Any], *, now: datetime, hostname: str) -> None:
    leases = record.setdefault("leases", {})
    if not isinstance(leases, dict):
        record["leases"] = {}
        return
    stale = [
        lease_id
        for lease_id, lease in leases.items()
        if not isinstance(lease, dict) or _lease_is_stale(lease, now=now, hostname=hostname)
    ]
    for lease_id in stale:
        leases.pop(lease_id, None)


def _safe_task_label(task_id: str) -> str:
    label = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in task_id)
    return (label or "default")[:24]


class FastVMEnvironment(BaseEnvironment):
    """FastVM cloud VM backend with snapshot-backed sleep/wake."""

    _stdin_mode = "heredoc"
    _snapshot_timeout = 60

    def __init__(
        self,
        machine_type: str = DEFAULT_FASTVM_MACHINE,
        base_snapshot_id: str | None = None,
        cwd: str = DEFAULT_FASTVM_CWD,
        timeout: int = 60,
        disk_gib: int = _DEFAULT_DISK_GIB,
        persistent_filesystem: bool = True,
        live_resume: bool = True,
        task_id: str = "default",
        launch_timeout: int = 300,
        snapshot_timeout: int = 300,
        lease_ttl_seconds: int = _DEFAULT_LEASE_TTL_SECONDS,
        _client: Any | None = None,
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        self._machine_type = machine_type or DEFAULT_FASTVM_MACHINE
        self._base_snapshot_id = base_snapshot_id or None
        self._disk_gib = max(1, int(disk_gib or _DEFAULT_DISK_GIB))
        self._persistent = persistent_filesystem
        self._live_resume = live_resume
        self._task_id = task_id or "default"
        self._launch_timeout = max(1, int(launch_timeout or 300))
        self._snapshot_timeout_seconds = max(1, int(snapshot_timeout or 300))
        self._lease_ttl_seconds = max(
            1, int(lease_ttl_seconds or _DEFAULT_LEASE_TTL_SECONDS)
        )
        self._hostname = socket.gethostname()
        self._lease_id = f"{self._hostname}:{os.getpid()}:{uuid.uuid4().hex}"
        self._requested_cwd = requested_cwd
        self._workspace_root = DEFAULT_FASTVM_CWD
        self._remote_home = DEFAULT_FASTVM_CWD
        self._lock = threading.Lock()
        self._vm: Any | None = None
        self._sync_manager: FileSyncManager | None = None

        if _client is None:
            from fastvm import FastvmClient

            _client = FastvmClient()
        self._client = _client

        self._vm = self._create_vm()
        self._configure_attached_vm(requested_cwd=requested_cwd)
        self._sync_manager.sync(force=True)
        self.init_session()

    def _vm_id(self, vm: Any | None = None) -> str:
        vm = vm or self._vm
        vm_id = _extract_id(vm)
        if not vm_id:
            raise RuntimeError("FastVM VM is not attached")
        return vm_id

    def _launch_metadata(self) -> dict[str, str]:
        return {
            "hermes_backend": "fastvm",
            "hermes_task_id": self._task_id,
            "hermes_live_resume": "true" if self._live_resume else "false",
        }

    def _launch_name(self, *, restore: bool) -> str:
        if self._persistent:
            return f"hermes-{_safe_task_label(self._task_id)}"[:64]
        prefix = "hermes-fastvm-restore" if restore else "hermes-fastvm"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{prefix}-{_safe_task_label(self._task_id)}-{stamp}"[:64]

    def _new_lease(self, *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        expires_at = now.timestamp() + self._lease_ttl_seconds
        return {
            "hostname": self._hostname,
            "pid": os.getpid(),
            "created_at": _utc_now(),
            "updated_at": now.isoformat(),
            "expires_at": datetime.fromtimestamp(expires_at, timezone.utc).isoformat(),
        }

    def _touch_lease_locked(
        self,
        snapshots: dict,
        record: dict[str, Any],
        vm: Any,
        *,
        now: datetime | None = None,
    ) -> None:
        now = now or datetime.now(timezone.utc)
        leases = record.setdefault("leases", {})
        existing = leases.get(self._lease_id)
        lease = self._new_lease(now=now)
        if isinstance(existing, dict) and existing.get("created_at"):
            lease["created_at"] = existing["created_at"]
        leases[self._lease_id] = lease

        vm_id = self._vm_id(vm)
        record["active_vm_id"] = vm_id
        name = getattr(vm, "name", None)
        if isinstance(vm, dict):
            name = vm.get("name", name)
        if isinstance(name, str) and name:
            record["active_vm_name"] = name
        created_at = _extract_created_at(vm)
        if created_at:
            record["active_vm_created_at"] = created_at
        snapshots[self._task_id] = record
        _save_snapshots(snapshots)

    def _retrieve_running_vm(self, vm_id: str | None) -> Any | None:
        if not vm_id:
            return None
        try:
            vm = self._client.vms.retrieve(vm_id)
        except Exception as exc:
            logger.debug(
                "FastVM: failed to retrieve active VM %s for task %s: %s",
                vm_id,
                self._task_id,
                exc,
            )
            return None
        status = _extract_status(vm)
        if not status or status == _RUNNING_VM_STATUS:
            return vm
        return None

    def _find_running_vm_by_metadata(self) -> Any | None:
        query = {
            "metadata.hermes_backend": "fastvm",
            "metadata.hermes_task_id": self._task_id,
        }
        try:
            response = self._client.vms.list(
                status=_RUNNING_VM_STATUS,
                extra_query=query,
            )
        except Exception as exc:
            logger.debug(
                "FastVM: metadata VM lookup failed for task %s: %s",
                self._task_id,
                exc,
            )
            return None

        candidates = []
        for vm in _iter_vms_response(response):
            metadata = _extract_metadata(vm)
            if metadata.get("hermes_backend") != "fastvm":
                continue
            if metadata.get("hermes_task_id") != self._task_id:
                continue
            status = _extract_status(vm)
            if status and status != _RUNNING_VM_STATUS:
                continue
            candidates.append(vm)
        if not candidates:
            return None
        candidates.sort(key=lambda vm: _extract_created_at(vm), reverse=True)
        return candidates[0]

    def _launch_from_snapshot(self, snapshot_id: str) -> Any:
        return self._client.vms.launch(
            snapshot_id=snapshot_id,
            name=self._launch_name(restore=True),
            metadata=self._launch_metadata(),
            wait_timeout=self._launch_timeout,
        )

    def _launch_fresh_vm(self) -> Any:
        logger.info(
            "FastVM: launching fresh VM for task %s (%s, %d GiB)",
            self._task_id,
            self._machine_type,
            self._disk_gib,
        )
        return self._client.vms.launch(
            machine_type=self._machine_type,
            disk_gi_b=self._disk_gib,
            name=self._launch_name(restore=False),
            metadata=self._launch_metadata(),
            wait_timeout=self._launch_timeout,
        )

    def _create_vm(self) -> Any:
        if not self._persistent:
            if self._base_snapshot_id:
                return self._launch_from_snapshot(self._base_snapshot_id)
            return self._launch_fresh_vm()

        with _fastvm_lifecycle_lock():
            snapshots = _load_snapshots()
            snapshot_record = _coerce_snapshot_record(snapshots.get(self._task_id)) or {}
            now = datetime.now(timezone.utc)
            _prune_stale_leases(snapshot_record, now=now, hostname=self._hostname)

            active_vm_id = snapshot_record.get("active_vm_id")
            vm = self._retrieve_running_vm(active_vm_id)
            if vm is not None:
                logger.info(
                    "FastVM: attaching task %s to active VM %s",
                    self._task_id,
                    self._vm_id(vm),
                )
                self._touch_lease_locked(snapshots, snapshot_record, vm, now=now)
                return vm

            if active_vm_id:
                snapshot_record.pop("active_vm_id", None)
                snapshot_record.pop("active_vm_name", None)
                snapshot_record.pop("active_vm_created_at", None)

            vm = self._find_running_vm_by_metadata()
            if vm is not None:
                logger.info(
                    "FastVM: discovered active VM %s for task %s",
                    self._vm_id(vm),
                    self._task_id,
                )
                self._touch_lease_locked(snapshots, snapshot_record, vm, now=now)
                return vm

            snapshot_id = snapshot_record.get("snapshot_id")
            if snapshot_id:
                try:
                    logger.info(
                        "FastVM: restoring task %s from snapshot %s",
                        self._task_id,
                        snapshot_id,
                    )
                    vm = self._launch_from_snapshot(snapshot_id)
                    self._touch_lease_locked(snapshots, snapshot_record, vm, now=now)
                    return vm
                except Exception as exc:
                    if self._live_resume:
                        raise RuntimeError(
                            "FastVM live-resume restore failed for "
                            f"snapshot {snapshot_id}; refusing to fall back to a fresh VM"
                        ) from exc
                    logger.warning(
                        "FastVM: failed to restore snapshot %s for task %s; "
                        "falling back to a fresh VM: %s",
                        snapshot_id,
                        self._task_id,
                        exc,
                    )
                    snapshot_record.pop("snapshot_id", None)
                    snapshot_record.pop("created_at", None)
                    snapshot_record.pop("source_vm_id", None)

            if self._base_snapshot_id:
                logger.info(
                    "FastVM: launching task %s from base snapshot %s",
                    self._task_id,
                    self._base_snapshot_id,
                )
                vm = self._launch_from_snapshot(self._base_snapshot_id)
            else:
                vm = self._launch_fresh_vm()
            self._touch_lease_locked(snapshots, snapshot_record, vm, now=now)
            return vm

    def _wait_for_running(self, vm: Any | None = None) -> Any:
        vm = vm or self._vm
        vm_id = self._vm_id(vm)
        deadline = time.monotonic() + _VM_READY_TIMEOUT
        current = vm
        while True:
            status = _extract_status(current)
            if not status or status == _RUNNING_VM_STATUS:
                return current
            if status in _DEAD_VM_STATUSES:
                raise RuntimeError(f"FastVM VM entered terminal state: {status}")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"FastVM VM did not reach running state (last status: {status})"
                )
            time.sleep(_VM_READY_POLL_INTERVAL)
            current = self._client.vms.retrieve(vm_id)

    def _detect_remote_home(self) -> str:
        try:
            result = self._client.vms.run(
                self._vm_id(),
                command=["bash", "-lc", 'printf %s "$HOME"'],
                timeout_sec=10,
                timeout=30,
            )
            home = str(getattr(result, "stdout", "") or "").strip()
            if home.startswith("/"):
                return home
        except Exception as exc:
            logger.debug("FastVM: home detection failed for task %s: %s", self._task_id, exc)
        return DEFAULT_FASTVM_CWD

    def _configure_attached_vm(self, *, requested_cwd: str) -> None:
        self._vm = self._wait_for_running(self._vm)
        self._remote_home = self._detect_remote_home()
        self._workspace_root = self._remote_home or DEFAULT_FASTVM_CWD

        container_base = (
            "/.hermes"
            if self._remote_home == "/"
            else f"{self._remote_home.rstrip('/')}/.hermes"
        )
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(container_base),
            upload_fn=self._fastvm_upload,
            delete_fn=self._fastvm_delete,
            bulk_upload_fn=self._fastvm_bulk_upload,
            bulk_download_fn=self._fastvm_bulk_download,
        )

        if requested_cwd == "~":
            self.cwd = self._remote_home
        elif requested_cwd in ("", DEFAULT_FASTVM_CWD):
            self.cwd = self._workspace_root
        else:
            self.cwd = requested_cwd

    def _ensure_vm_ready(self) -> None:
        requested_cwd = self.cwd or self._requested_cwd or DEFAULT_FASTVM_CWD
        if self._vm is None:
            self._vm = self._create_vm()
            self._configure_attached_vm(requested_cwd=requested_cwd)
            return

        try:
            vm = self._client.vms.retrieve(self._vm_id())
        except Exception as exc:
            logger.warning(
                "FastVM: failed to retrieve VM for task %s; recreating from snapshot: %s",
                self._task_id,
                exc,
            )
            self._vm = self._create_vm()
            self._configure_attached_vm(requested_cwd=requested_cwd)
            return

        status = _extract_status(vm)
        if status in _DEAD_VM_STATUSES:
            logger.warning(
                "FastVM: VM entered state %s for task %s; recreating from snapshot",
                status,
                self._task_id,
            )
            self._vm = self._create_vm()
            self._configure_attached_vm(requested_cwd=requested_cwd)
            return

        self._vm = self._wait_for_running(vm)

    def _renew_lease(self) -> None:
        if not self._persistent or self._vm is None:
            return
        with _fastvm_lifecycle_lock():
            snapshots = _load_snapshots()
            record = _coerce_snapshot_record(snapshots.get(self._task_id)) or {}
            now = datetime.now(timezone.utc)
            _prune_stale_leases(record, now=now, hostname=self._hostname)
            self._touch_lease_locked(snapshots, record, self._vm, now=now)

    def _fastvm_upload(self, host_path: str, remote_path: str) -> None:
        self._fastvm_bulk_upload([(host_path, remote_path)])

    def _fastvm_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return
        parents = unique_parent_dirs(files)
        if parents:
            result = self._client.vms.run(
                self._vm_id(),
                command=["bash", "-lc", quoted_mkdir_command(parents)],
                timeout_sec=30,
                timeout=60,
            )
            if int(getattr(result, "exit_code", 1)) != 0:
                raise RuntimeError(
                    f"FastVM mkdir failed: {getattr(result, 'stderr', '') or getattr(result, 'stdout', '')}"
                )

        for host_path, remote_path in files:
            self._client.upload(self._vm_id(), host_path, remote_path)

    def _fastvm_delete(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return
        result = self._client.vms.run(
            self._vm_id(),
            command=["bash", "-lc", quoted_rm_command(remote_paths)],
            timeout_sec=30,
            timeout=60,
        )
        if int(getattr(result, "exit_code", 1)) != 0:
            raise RuntimeError(
                f"FastVM delete failed: {getattr(result, 'stderr', '') or getattr(result, 'stdout', '')}"
            )

    def _fastvm_bulk_download(self, dest_tar_path: Path) -> None:
        remote_hermes = (
            "/.hermes"
            if self._remote_home == "/"
            else f"{self._remote_home.rstrip('/')}/.hermes"
        )
        archive_member = remote_hermes.lstrip("/")
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        try:
            result = self._client.vms.run(
                self._vm_id(),
                command=[
                    "bash",
                    "-lc",
                    f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(archive_member)}",
                ],
                timeout_sec=120,
                timeout=180,
            )
            if int(getattr(result, "exit_code", 1)) != 0:
                raise RuntimeError(
                    f"FastVM bulk download failed: {getattr(result, 'stderr', '') or getattr(result, 'stdout', '')}"
                )
            self._client.download(self._vm_id(), remote_tar, str(dest_tar_path))
        finally:
            try:
                self._client.vms.run(
                    self._vm_id(),
                    command=["bash", "-lc", f"rm -f {shlex.quote(remote_tar)}"],
                    timeout_sec=30,
                    timeout=60,
                )
            except Exception:
                pass

    def _before_execute(self) -> None:
        with self._lock:
            self._ensure_vm_ready()
            self._renew_lease()
            if self._sync_manager is not None:
                self._sync_manager.sync()

    def _delete_vm_for_cancel(self, vm_id: str) -> None:
        if not self._persistent:
            self._client.vms.delete(vm_id)
            return
        with _fastvm_lifecycle_lock():
            snapshots = _load_snapshots()
            record = _coerce_snapshot_record(snapshots.get(self._task_id)) or {}
            now = datetime.now(timezone.utc)
            _prune_stale_leases(record, now=now, hostname=self._hostname)
            leases = record.setdefault("leases", {})
            other_leases = {
                lease_id: lease
                for lease_id, lease in leases.items()
                if lease_id != self._lease_id
            }
            if other_leases:
                logger.warning(
                    "FastVM: refusing cancel-delete of VM %s for task %s because "
                    "%d other lease(s) are active",
                    vm_id,
                    self._task_id,
                    len(other_leases),
                )
                return
            if record.get("active_vm_id") == vm_id:
                record.pop("active_vm_id", None)
                record.pop("active_vm_name", None)
                record.pop("active_vm_created_at", None)
                snapshots[self._task_id] = record
                _save_snapshots(snapshots)
            self._client.vms.delete(vm_id)

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        del stdin_data

        vm_id = self._vm_id()
        lock = self._lock

        def cancel() -> None:
            # FastVM does not currently expose per-exec cancellation. Delete the
            # VM as a last-resort interrupt so runaway foreground commands stop.
            with lock:
                try:
                    self._delete_vm_for_cancel(vm_id)
                except Exception:
                    pass
                if self._vm is not None and _extract_id(self._vm) == vm_id:
                    self._vm = None

        def exec_fn() -> tuple[str, int]:
            result = self._client.vms.run(
                vm_id,
                command=["bash", "-lc" if login else "-c", cmd_string],
                timeout_sec=timeout,
                timeout=timeout + 120,
            )
            stdout = str(getattr(result, "stdout", "") or "")
            stderr = str(getattr(result, "stderr", "") or "")
            if stderr and stdout and not stdout.endswith("\n"):
                output = f"{stdout}\n{stderr}"
            else:
                output = stdout + stderr
            exit_code = int(getattr(result, "exit_code", 1))
            if getattr(result, "timed_out", False) and exit_code == 0:
                exit_code = 124
            return output, exit_code

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def _wait_for_snapshot_ready(self, snapshot: Any) -> str | None:
        snapshot_id = _extract_id(snapshot)
        if not snapshot_id:
            return None
        deadline = time.monotonic() + self._snapshot_timeout_seconds
        current = snapshot
        while True:
            status = _extract_status(current)
            if not status or status in _READY_SNAPSHOT_STATUSES:
                return snapshot_id
            if status in _ERROR_SNAPSHOT_STATUSES:
                logger.warning("FastVM: snapshot %s entered state %s", snapshot_id, status)
                return None
            if time.monotonic() >= deadline:
                logger.warning(
                    "FastVM: snapshot %s did not become ready before timeout "
                    "(last status: %s)",
                    snapshot_id,
                    status,
                )
                return None
            time.sleep(_SNAPSHOT_POLL_INTERVAL)
            current = self._client.snapshots.retrieve(snapshot_id)

    def _delete_remote_snapshot(self, snapshot_id: str | None) -> None:
        if not snapshot_id:
            return
        try:
            self._client.snapshots.delete(snapshot_id)
        except Exception as exc:
            logger.debug("FastVM: failed to delete old snapshot %s: %s", snapshot_id, exc)

    def _snapshot_vm(self, vm: Any) -> str | None:
        if not self._persistent or not self._task_id:
            return None
        vm_id = self._vm_id(vm)
        name = (
            f"hermes-{_safe_task_label(self._task_id)}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{vm_id[:8]}"
        )[:64]
        try:
            snapshot = self._client.snapshots.create(
                vm_id=vm_id,
                name=name,
                timeout=self._snapshot_timeout_seconds + 30,
            )
        except Exception as exc:
            logger.warning(
                "FastVM: snapshot creation failed for task %s VM %s: %s",
                self._task_id,
                vm_id,
                exc,
            )
            return None

        snapshot_id = self._wait_for_snapshot_ready(snapshot)
        if not snapshot_id:
            return None
        logger.info("FastVM: saved snapshot %s for task %s", snapshot_id, self._task_id)
        return snapshot_id

    def cleanup(self):
        with self._lock:
            vm = self._vm
            sync_manager = self._sync_manager
            if vm is None:
                return

            if self._persistent:
                self._cleanup_persistent(vm, sync_manager)
                self._vm = None
                self._sync_manager = None
                return

            if sync_manager is not None:
                try:
                    sync_manager.sync_back()
                except Exception as exc:
                    logger.warning(
                        "FastVM: sync_back failed for task %s: %s",
                        self._task_id,
                        exc,
                    )

            try:
                self._client.vms.delete(self._vm_id(vm))
            except Exception as exc:
                logger.warning(
                    "FastVM: failed to delete VM %s for task %s: %s",
                    self._vm_id(vm),
                    self._task_id,
                    exc,
                )
            finally:
                self._vm = None
                self._sync_manager = None

    def _cleanup_persistent(self, vm: Any, sync_manager: FileSyncManager | None) -> None:
        vm_id = self._vm_id(vm)
        with _fastvm_lifecycle_lock():
            snapshots = _load_snapshots()
            record = _coerce_snapshot_record(snapshots.get(self._task_id)) or {}
            now = datetime.now(timezone.utc)
            _prune_stale_leases(record, now=now, hostname=self._hostname)
            leases = record.setdefault("leases", {})
            leases.pop(self._lease_id, None)

            active_vm_id = record.get("active_vm_id")
            if active_vm_id and active_vm_id != vm_id:
                snapshots[self._task_id] = record
                _save_snapshots(snapshots)
                logger.warning(
                    "FastVM: cleanup for task %s skipped destructive actions because "
                    "active VM changed from %s to %s",
                    self._task_id,
                    vm_id,
                    active_vm_id,
                )
                return

            if leases:
                snapshots[self._task_id] = record
                _save_snapshots(snapshots)
                logger.info(
                    "FastVM: detached from VM %s for task %s; %d lease(s) remain",
                    vm_id,
                    self._task_id,
                    len(leases),
                )
                return

            if sync_manager is not None:
                try:
                    sync_manager.sync_back()
                except Exception as exc:
                    logger.warning(
                        "FastVM: sync_back failed for task %s: %s",
                        self._task_id,
                        exc,
                    )

            snapshot_id = self._snapshot_vm(vm)
            if not snapshot_id:
                record["active_vm_id"] = vm_id
                snapshots[self._task_id] = record
                _save_snapshots(snapshots)
                raise RuntimeError(
                    "FastVM persistent cleanup refused to delete VM "
                    f"{vm_id} because snapshot creation failed"
                )

            old_snapshot_id = record.get("snapshot_id")
            record.update(
                {
                    "snapshot_id": snapshot_id,
                    "created_at": _utc_now(),
                    "source_vm_id": vm_id,
                    "machine_type": self._machine_type,
                    "base_snapshot_id": self._base_snapshot_id or "",
                    "live_resume": self._live_resume,
                    "leases": {},
                }
            )
            record.pop("active_vm_id", None)
            record.pop("active_vm_name", None)
            record.pop("active_vm_created_at", None)
            snapshots[self._task_id] = record
            _save_snapshots(snapshots)

            if old_snapshot_id and old_snapshot_id != snapshot_id:
                self._delete_remote_snapshot(old_snapshot_id)

            try:
                self._client.vms.delete(vm_id)
            except Exception as exc:
                logger.warning(
                    "FastVM: failed to delete VM %s for task %s: %s",
                    vm_id,
                    self._task_id,
                    exc,
                )
