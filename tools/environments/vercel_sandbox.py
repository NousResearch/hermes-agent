"""Vercel Sandbox execution environment.

Uses the Vercel Python SDK to run commands in cloud sandboxes through Hermes'
shared ``BaseEnvironment`` shell contract. When persistence is enabled, the
backend stores task-scoped snapshot metadata under ``HERMES_HOME`` and restores
new sandboxes from those snapshots on later task reuse.
"""

from __future__ import annotations

from functools import cache
from dataclasses import dataclass
from datetime import timedelta
import logging
import math
import os
import shlex
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

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
    quoted_rm_command,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vercel.sandbox import Resources, Sandbox, SandboxStatus, WriteFile

DEFAULT_VERCEL_CWD = "/vercel/sandbox"
_DEFAULT_CONTAINER_DISK_MB = 51200


def _ensure_vercel_sdk() -> None:
    """Lazy-install vercel SDK on demand. Idempotent."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("terminal.vercel", prompt=False)
    except ImportError:
        pass
    except Exception as e:
        raise ImportError(str(e))


_CREATE_RETRY_ATTEMPTS = 3
_WRITE_RETRY_ATTEMPTS = 3
_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRY_BACKOFF_STEP = timedelta(milliseconds=100)
_MIN_SANDBOX_TIMEOUT = timedelta(minutes=5)
_MIN_RUNNING_WAIT = timedelta(seconds=1)
_RUNNING_WAIT_TIMEOUT = timedelta(seconds=30)
_RUNNING_WAIT_POLL_INTERVAL = timedelta(milliseconds=250)
_STOP_TIMEOUT = timedelta(seconds=15)
_STOP_POLL_INTERVAL = timedelta(milliseconds=500)
_SNAPSHOT_STORE_NAME = "vercel_sandbox_snapshots.json"


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _extract_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    for value in (getattr(exc, "status_code", None), getattr(response, "status_code", None)):
        if isinstance(value, int):
            return value
    return None


def _is_transient_vercel_error(exc: BaseException) -> bool:
    for error in _exception_chain(exc):
        status_code = _extract_status_code(error)
        if status_code in _TRANSIENT_STATUS_CODES:
            return True
        if isinstance(
            error,
            (httpx.NetworkError, httpx.ProtocolError, httpx.ReadError),
        ):
            return True
        error_name = type(error).__name__.lower()
        if "ratelimit" in error_name or "servererror" in error_name:
            return True
    return False


def _retry_vercel_call(
    label: str,
    callback,
    *,
    attempts: int,
):
    backoff_seconds = _RETRY_BACKOFF_STEP.total_seconds()
    for attempt in range(1, attempts + 1):
        try:
            return callback()
        except Exception as exc:
            if attempt >= attempts or not _is_transient_vercel_error(exc):
                raise
            logger.warning(
                "Vercel: %s failed (%s); retrying %d/%d",
                label,
                exc,
                attempt,
                attempts,
            )
            time.sleep(backoff_seconds * attempt)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_result_output(result: Any) -> str:
    try:
        return _coerce_text(result.output())
    except (AttributeError, TypeError):
        return _coerce_text(result)


def _extract_result_returncode(result: Any) -> int:
    try:
        exit_code = result.exit_code
    except AttributeError:
        try:
            exit_code = result.returncode
        except AttributeError:
            return 1
    return exit_code if isinstance(exit_code, int) else 1


def _snapshot_store_path() -> Path:
    return get_hermes_home() / _SNAPSHOT_STORE_NAME


def _load_snapshots() -> dict:
    return _load_json_store(_snapshot_store_path())


def _save_snapshots(data: dict) -> None:
    _save_json_store(_snapshot_store_path(), data)


def _get_snapshot_id(task_id: str) -> str | None:
    if not task_id:
        return None
    snapshot_id = _load_snapshots().get(task_id)
    return snapshot_id if isinstance(snapshot_id, str) and snapshot_id else None


def _store_snapshot(task_id: str, snapshot_id: str) -> None:
    if not task_id or not snapshot_id:
        return
    snapshots = _load_snapshots()
    snapshots[task_id] = snapshot_id
    _save_snapshots(snapshots)


def _delete_snapshot(task_id: str, snapshot_id: str | None = None) -> None:
    if not task_id:
        return
    snapshots = _load_snapshots()
    existing = snapshots.get(task_id)
    if existing is None:
        return
    if snapshot_id is not None and existing != snapshot_id:
        return
    snapshots.pop(task_id, None)
    _save_snapshots(snapshots)


def _extract_snapshot_id(snapshot: Any) -> str | None:
    for attr in ("snapshot_id", "snapshotId", "id"):
        value = getattr(snapshot, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(snapshot, dict):
        for key in ("snapshot_id", "snapshotId", "id"):
            value = snapshot.get(key)
            if isinstance(value, str) and value:
                return value
    return None


@cache
def _sandbox_status_type() -> type[SandboxStatus]:
    _ensure_vercel_sdk()
    from vercel.sandbox import SandboxStatus

    return SandboxStatus


@cache
def _terminal_sandbox_states() -> frozenset[SandboxStatus]:
    SandboxStatus = _sandbox_status_type()
    return frozenset(
        {
            SandboxStatus.ABORTED,
            SandboxStatus.FAILED,
            SandboxStatus.STOPPED,
        }
    )


@dataclass(frozen=True, slots=True)
class _SandboxCreateParams:
    timeout: timedelta
    runtime: str | None = None
    resources: Resources | None = None


class VercelSandboxEnvironment(BaseEnvironment):
    """Vercel cloud sandbox backend."""

    _stdin_mode = "heredoc"

    def __init__(
        self,
        runtime: str | None = None,
        cwd: str = DEFAULT_VERCEL_CWD,
        timeout: int = 60,
        cpu: float = 1,
        memory: int = 5120,
        disk: int = _DEFAULT_CONTAINER_DISK_MB,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        self._runtime = runtime or None
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._requested_cwd = requested_cwd
        self._lock = threading.Lock()
        self._sandbox: Sandbox | None = None
        self._workspace_root = DEFAULT_VERCEL_CWD
        self._remote_home = DEFAULT_VERCEL_CWD
        self._sync_manager: FileSyncManager | None = None
        self._create_params = self._build_create_params(cpu=cpu, memory=memory, disk=disk)

        self._sandbox = self._create_sandbox()
        self._configure_attached_sandbox(requested_cwd=requested_cwd)
        self._sync_manager.sync(force=True)
        self.init_session()

    def _build_create_params(self, *, cpu: float, memory: int, disk: int) -> _SandboxCreateParams:
        if disk not in {0, _DEFAULT_CONTAINER_DISK_MB}:
            raise ValueError(
                "Vercel Sandbox does not support configurable container_disk. "
                "Use the default shared setting."
            )

        _ensure_vercel_sdk()
        from vercel.sandbox import Resources

        sandbox_timeout = max(
            timedelta(seconds=max(self.timeout, 0)),
            _MIN_SANDBOX_TIMEOUT,
        )
        vcpus = math.floor(cpu) if cpu > 0 else None
        memory_mb = memory if memory > 0 else None
        resources = (
            Resources(vcpus=vcpus, memory=memory_mb)
            if vcpus is not None or memory_mb is not None
            else None
        )

        return _SandboxCreateParams(
            timeout=sandbox_timeout,
            runtime=self._runtime,
            resources=resources,
        )

    def _create_sandbox(self) -> Sandbox:
        _ensure_vercel_sdk()
        from vercel.sandbox import Sandbox

        snapshot_id = _get_snapshot_id(self._task_id) if self._persistent else None
        if snapshot_id:
            try:
                return _retry_vercel_call(
                    "sandbox restore",
                    lambda: Sandbox.create(
                        timeout=self._create_params.timeout,
                        runtime=self._create_params.runtime,
                        resources=self._create_params.resources,
                        source={"type": "snapshot", "snapshot_id": snapshot_id},
                    ),
                    attempts=_CREATE_RETRY_ATTEMPTS,
                )
            except Exception as exc:
                logger.warning(
                    "Vercel: failed to restore snapshot %s for task %s; "
                    "falling back to a fresh sandbox: %s",
                    snapshot_id,
                    self._task_id,
                    exc,
                )
                _delete_snapshot(self._task_id, snapshot_id)

        params = self._create_params
        return _retry_vercel_call(
            "sandbox create",
            lambda: Sandbox.create(
                timeout=params.timeout,
                runtime=params.runtime,
                resources=params.resources,
            ),
            attempts=_CREATE_RETRY_ATTEMPTS,
        )

    def _configure_attached_sandbox(self, *, requested_cwd: str) -> None:
        self._wait_for_running()
        self._workspace_root = self._detect_workspace_root()
        self._remote_home = self._detect_remote_home()

        if self._remote_home == "/":
            container_base = "/.hermes"
        else:
            container_base = f"{self._remote_home.rstrip('/')}/.hermes"
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(container_base),
            upload_fn=self._vercel_upload,
            delete_fn=self._vercel_delete,
            bulk_upload_fn=self._vercel_bulk_upload,
            bulk_download_fn=self._vercel_bulk_download,
        )

        if requested_cwd == "~":
            self.cwd = self._remote_home
        elif requested_cwd in {"", DEFAULT_VERCEL_CWD}:
            self.cwd = self._workspace_root
        else:
            self.cwd = requested_cwd

    def _detect_workspace_root(self) -> str:
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        cwd = sandbox.sandbox.cwd
        return cwd if cwd.startswith("/") else DEFAULT_VERCEL_CWD

    def _detect_remote_home(self) -> str:
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        try:
            result = sandbox.run_command(
                "sh",
                ["-lc", 'printf %s "$HOME"'],
                cwd=self._workspace_root,
            )
        except Exception as exc:
            logger.debug(
                "Vercel: home detection failed for task %s: %s",
                self._task_id,
                exc,
            )
            return self._workspace_root

        home = _extract_result_output(result).strip()
        if home.startswith("/"):
            return home
        return self._workspace_root

    def _wait_for_running(self, timeout: timedelta = _RUNNING_WAIT_TIMEOUT) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        SandboxStatus = _sandbox_status_type()
        status = sandbox.status
        if status is None or status == SandboxStatus.RUNNING:
            return
        if status in _terminal_sandbox_states():
            raise RuntimeError(f"Sandbox entered terminal state: {status}")

        try:
            sandbox.wait_for_status(
                SandboxStatus.RUNNING,
                timeout=max(timeout, _MIN_RUNNING_WAIT),
                poll_interval=_RUNNING_WAIT_POLL_INTERVAL,
            )
        except TimeoutError as exc:
            status = sandbox.status
            if status in _terminal_sandbox_states():
                raise RuntimeError(f"Sandbox entered terminal state: {status}") from exc
            raise RuntimeError(
                f"Sandbox did not reach running state (last status: {status})"
            ) from exc

    def _close_sandbox_client(self, sandbox: Sandbox | None) -> None:
        if sandbox is None:
            return
        try:
            sandbox.client.close()
        except Exception:
            pass

    def _stop_sandbox(self, sandbox: Sandbox | None) -> None:
        if sandbox is None:
            return
        try:
            sandbox.stop(
                blocking=True,
                timeout=_STOP_TIMEOUT,
                poll_interval=_STOP_POLL_INTERVAL,
            )
        except TypeError:
            try:
                sandbox.stop()
            except Exception:
                pass
        except Exception:
            pass

    def _snapshot_sandbox(self, sandbox: Sandbox) -> str | None:
        if not self._persistent or not self._task_id:
            return None
        try:
            snapshot = sandbox.snapshot()
        except Exception as exc:
            logger.warning(
                "Vercel: filesystem snapshot failed for task %s: %s",
                self._task_id,
                exc,
            )
            return None

        snapshot_id = _extract_snapshot_id(snapshot)
        if not snapshot_id:
            logger.warning(
                "Vercel: filesystem snapshot for task %s did not return a snapshot id",
                self._task_id,
            )
            return None

        _store_snapshot(self._task_id, snapshot_id)
        logger.info(
            "Vercel: saved filesystem snapshot %s for task %s",
            snapshot_id,
            self._task_id,
        )
        return snapshot_id

    def _ensure_sandbox_ready(self) -> None:
        sandbox = self._sandbox
        requested_cwd = self.cwd or self._requested_cwd or DEFAULT_VERCEL_CWD

        if sandbox is None:
            self._sandbox = self._create_sandbox()
            self._configure_attached_sandbox(requested_cwd=requested_cwd)
            return

        try:
            sandbox.refresh()
        except Exception as exc:
            logger.warning(
                "Vercel: sandbox refresh failed for task %s: %s; recreating",
                self._task_id,
                exc,
            )
            self._close_sandbox_client(sandbox)
            self._sandbox = self._create_sandbox()
            self._configure_attached_sandbox(requested_cwd=requested_cwd)
            return

        status = sandbox.status
        if status in _terminal_sandbox_states():
            logger.warning(
                "Vercel: sandbox entered state %s for task %s; recreating",
                status,
                self._task_id,
            )
            self._close_sandbox_client(sandbox)
            self._sandbox = self._create_sandbox()
            self._configure_attached_sandbox(requested_cwd=requested_cwd)
            return

        self._wait_for_running()

    def _vercel_upload(self, host_path: str, remote_path: str) -> None:
        self._vercel_bulk_upload([(host_path, remote_path)])

    def _vercel_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return

        payload: list[WriteFile] = [
            {
                "path": remote_path,
                "content": Path(host_path).read_bytes(),
            }
            for host_path, remote_path in files
        ]

        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        _retry_vercel_call(
            "write_files",
            lambda: sandbox.write_files(payload),
            attempts=_WRITE_RETRY_ATTEMPTS,
        )

    def _vercel_delete(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return

        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        result = sandbox.run_command(
            "bash",
            ["-lc", quoted_rm_command(remote_paths)],
            cwd=self._workspace_root,
        )
        if _extract_result_returncode(result) != 0:
            raise RuntimeError(
                f"Vercel delete failed: {_extract_result_output(result).strip()}"
            )

    def _vercel_bulk_download(self, dest_tar_path: Path) -> None:
        remote_hermes = (
            "/.hermes"
            if self._remote_home == "/"
            else f"{self._remote_home.rstrip('/')}/.hermes"
        )
        archive_member = remote_hermes.lstrip("/")
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")

        try:
            result = sandbox.run_command(
                "bash",
                [
                    "-lc",
                    f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(archive_member)}",
                ],
                cwd=self._workspace_root,
            )
            if _extract_result_returncode(result) != 0:
                raise RuntimeError(
                    f"Vercel bulk download failed: {_extract_result_output(result).strip()}"
                )

            sandbox.download_file(remote_tar, dest_tar_path)
        finally:
            try:
                sandbox.run_command(
                    "bash",
                    ["-lc", f"rm -f {shlex.quote(remote_tar)}"],
                    cwd=self._workspace_root,
                )
            except Exception:
                pass

    def _before_execute(self) -> None:
        with self._lock:
            self._ensure_sandbox_ready()
            if self._sync_manager is not None:
                self._sync_manager.sync()

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        """Run a bash command in the Vercel sandbox.

        ``timeout`` is not forwarded to the Vercel SDK (which does not expose
        a per-exec timeout parameter); the base class ``_wait_for_process``
        enforces timeout by killing the sandbox via ``cancel_fn``.

        ``stdin_data`` is intentionally discarded here because
        ``_stdin_mode = "heredoc"`` causes the base class ``execute()`` to
        embed any stdin payload into the command string before calling this
        method.
        """
        del timeout
        del stdin_data

        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        workspace_root = self._workspace_root
        lock = self._lock

        def cancel() -> None:
            with lock:
                self._stop_sandbox(sandbox)

        def exec_fn() -> tuple[str, int]:
            result = sandbox.run_command(
                "bash",
                ["-lc" if login else "-c", cmd_string],
                cwd=workspace_root,
            )
            return _extract_result_output(result), _extract_result_returncode(result)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            sandbox = self._sandbox
            sync_manager = self._sync_manager
            if sandbox is not None and sync_manager is not None:
                try:
                    sync_manager.sync_back()
                except Exception as exc:
                    logger.warning(
                        "Vercel: sync_back failed for task %s: %s",
                        self._task_id,
                        exc,
                    )
            self._sandbox = None
            self._sync_manager = None

        if sandbox is None:
            return

        snapshot_id = self._snapshot_sandbox(sandbox)
        # Always stop the sandbox during cleanup to avoid resource leaks,
        # matching the Modal and Daytona patterns.
        self._stop_sandbox(sandbox)
        self._close_sandbox_client(sandbox)
