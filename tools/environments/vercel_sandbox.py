"""Vercel Sandbox execution environment.

Runs commands in Vercel cloud sandboxes through the shared ``BaseEnvironment``
shell contract. With persistence enabled, task-scoped snapshot ids are stored
under ``HERMES_HOME`` and new sandboxes are restored from them on task reuse.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shlex
import threading
import time
from datetime import timedelta
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from hermes_constants import get_hermes_home
from tools.environments.base import BaseEnvironment, _load_json_store, _save_json_store
from tools.environments.base_output import _ThreadedProcessHandle
from tools.environments.file_sync import FileSyncManager, iter_sync_files, quoted_rm_command
from tools.environments.remote_common import ensure_lazy_dep

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vercel.sandbox import Sandbox, SandboxStatus, WriteFile

DEFAULT_VERCEL_CWD = "/vercel/sandbox"
_DEFAULT_CONTAINER_DISK_MB = 51200
_CREATE_RETRY_ATTEMPTS = 3
_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_RUNNING_WAIT_TIMEOUT = timedelta(seconds=30)


def _ensure_vercel_sdk() -> None:
    """Lazy-install vercel SDK on demand. Idempotent."""
    # The SDK (>=0.7) ships default-on telemetry; Hermes policy is opt-in only, so disable it
    # before the SDK is imported. setdefault: an explicit user value is never overridden.
    os.environ.setdefault("VERCEL_TELEMETRY_DISABLED", "1")
    ensure_lazy_dep("terminal.vercel")


def _is_transient_vercel_error(exc: BaseException) -> bool:
    """True when any exception in the cause/context chain looks retryable."""
    seen: set[int] = set()
    error: BaseException | None = exc
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        codes = (getattr(error, "status_code", None), getattr(getattr(error, "response", None), "status_code", None))
        status = next((c for c in codes if isinstance(c, int)), None)
        name = type(error).__name__.lower()
        if (status in _TRANSIENT_STATUS_CODES
                or isinstance(error, (httpx.NetworkError, httpx.ProtocolError, httpx.ReadError))
                or "ratelimit" in name or "servererror" in name):
            return True
        error = error.__cause__ or error.__context__
    return False


def _retry_vercel_call(label: str, callback, *, attempts: int):
    for attempt in range(1, attempts + 1):
        try:
            return callback()
        except Exception as exc:
            if attempt >= attempts or not _is_transient_vercel_error(exc):
                raise
            logger.warning("Vercel: %s failed (%s); retrying %d/%d", label, exc, attempt, attempts)
            time.sleep(0.1 * attempt)


def _result_parts(result: Any) -> tuple[str, int]:
    """(output, returncode) from an SDK command result; tolerates raw strings and older result shapes."""
    try:
        value = result.output()
    except (AttributeError, TypeError):
        value = result
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    exit_code = getattr(result, "exit_code" if hasattr(result, "exit_code") else "returncode", None)
    return ("" if value is None else str(value)), (exit_code if isinstance(exit_code, int) else 1)


def _snapshot_store() -> Path:
    return get_hermes_home() / "vercel_sandbox_snapshots.json"


def _load_snapshots() -> dict:
    return _load_json_store(_snapshot_store())


def _store_snapshot(task_id: str, snapshot_id: str) -> None:
    if task_id and snapshot_id:
        _save_json_store(_snapshot_store(), {**_load_snapshots(), task_id: snapshot_id})


def _delete_snapshot(task_id: str, snapshot_id: str) -> None:
    """Drop the stored id for ``task_id`` only if it still equals ``snapshot_id``."""
    snapshots = _load_snapshots()
    if task_id and snapshots.get(task_id) == snapshot_id:
        snapshots.pop(task_id)
        _save_json_store(_snapshot_store(), snapshots)


def _extract_snapshot_id(snapshot: Any) -> str | None:
    """Accept SDK objects or raw dicts; attribute lookup first, then dict keys."""
    getters = [lambda k: getattr(snapshot, k, None)] + ([snapshot.get] if isinstance(snapshot, dict) else [])
    candidates = (get(key) for get in getters for key in ("snapshot_id", "snapshotId", "id"))
    return next((v for v in candidates if isinstance(v, str) and v), None)


@cache
def _sandbox_status_type() -> type[SandboxStatus]:
    _ensure_vercel_sdk()
    from vercel.sandbox import SandboxStatus
    return SandboxStatus


def _is_terminal(status: Any) -> bool:
    S = _sandbox_status_type()
    return status in {S.ABORTED, S.FAILED, S.STOPPED}


class VercelSandboxEnvironment(BaseEnvironment):
    """Vercel cloud sandbox backend."""

    _stdin_mode = "heredoc"

    def __init__(self, runtime: str | None = None, cwd: str = DEFAULT_VERCEL_CWD, timeout: int = 60,
                 cpu: float = 1, memory: int = 5120, disk: int = _DEFAULT_CONTAINER_DISK_MB,
                 persistent_filesystem: bool = True, task_id: str = "default"):
        super().__init__(cwd=cwd, timeout=timeout)
        if disk not in {0, _DEFAULT_CONTAINER_DISK_MB}:
            raise ValueError(
                "Vercel Sandbox does not support configurable container_disk. "
                "Use the default shared setting.")
        self._persistent, self._task_id, self._requested_cwd = persistent_filesystem, task_id, cwd
        self._lock = threading.Lock()
        self._sandbox: Sandbox | None = None
        self._workspace_root = self._remote_home = DEFAULT_VERCEL_CWD
        self._sync_manager: FileSyncManager | None = None
        _ensure_vercel_sdk()
        from vercel.sandbox import Resources
        vcpus, memory_mb = (math.floor(cpu) if cpu > 0 else None), (memory if memory > 0 else None)
        self._create_kwargs = {
            "timeout": max(timedelta(seconds=max(self.timeout, 0)), timedelta(minutes=5)),
            "runtime": runtime or None,
            "resources": Resources(vcpus=vcpus, memory=memory_mb) if (vcpus, memory_mb) != (None, None) else None}
        self._attach_fresh_sandbox(cwd)
        self._sync_manager.sync(force=True)
        self.init_session()

    def _require_sandbox(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        return self._sandbox

    def _remote_hermes_dir(self) -> str:
        return f"{self._remote_home.rstrip('/')}/.hermes"

    def _create_sandbox(self) -> Sandbox:
        _ensure_vercel_sdk()
        from vercel.sandbox import Sandbox
        snapshot_id = _load_snapshots().get(self._task_id) if self._persistent and self._task_id else None
        if isinstance(snapshot_id, str) and snapshot_id:
            try:
                source = {"type": "snapshot", "snapshot_id": snapshot_id}
                return _retry_vercel_call(
                    "sandbox restore", lambda: Sandbox.create(**self._create_kwargs, source=source),
                    attempts=_CREATE_RETRY_ATTEMPTS)
            except Exception as exc:
                logger.warning("Vercel: failed to restore snapshot %s for task %s; falling back to a fresh sandbox: %s",
                               snapshot_id, self._task_id, exc)
                _delete_snapshot(self._task_id, snapshot_id)
        return _retry_vercel_call("sandbox create", lambda: Sandbox.create(**self._create_kwargs),
                                  attempts=_CREATE_RETRY_ATTEMPTS)

    def _attach_fresh_sandbox(self, requested_cwd: str) -> None:
        """Create a sandbox, wait until it runs, then wire cwd/home and the file sync manager."""
        self._sandbox = self._create_sandbox()
        self._wait_for_running()
        cwd = self._require_sandbox().sandbox.cwd
        self._workspace_root = cwd if cwd.startswith("/") else DEFAULT_VERCEL_CWD
        self._remote_home = self._detect_remote_home()
        container_base = self._remote_hermes_dir()
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(container_base),
            upload_fn=lambda host_path, remote_path: self._vercel_bulk_upload([(host_path, remote_path)]),
            delete_fn=self._vercel_delete,
            bulk_upload_fn=self._vercel_bulk_upload, bulk_download_fn=self._vercel_bulk_download)
        self.cwd = {"~": self._remote_home, "": self._workspace_root,
                    DEFAULT_VERCEL_CWD: self._workspace_root}.get(requested_cwd, requested_cwd)

    def _detect_remote_home(self) -> str:
        try:
            result = self._require_sandbox().run_command("sh", ["-lc", 'printf %s "$HOME"'], cwd=self._workspace_root)
        except Exception as exc:
            logger.debug("Vercel: home detection failed for task %s: %s", self._task_id, exc)
            return self._workspace_root
        home = _result_parts(result)[0].strip()
        return home if home.startswith("/") else self._workspace_root

    def _wait_for_running(self, timeout: timedelta = _RUNNING_WAIT_TIMEOUT) -> None:
        sandbox, running = self._require_sandbox(), _sandbox_status_type().RUNNING
        status = sandbox.status
        if status is None or status == running:
            return
        if _is_terminal(status):
            raise RuntimeError(f"Sandbox entered terminal state: {status}")
        try:
            sandbox.wait_for_status(running, timeout=max(timeout, timedelta(seconds=1)),
                                    poll_interval=timedelta(milliseconds=250))
        except TimeoutError as exc:
            status = sandbox.status
            if _is_terminal(status):
                raise RuntimeError(f"Sandbox entered terminal state: {status}") from exc
            raise RuntimeError(f"Sandbox did not reach running state (last status: {status})") from exc

    @staticmethod
    def _close_sandbox_client(sandbox: Sandbox | None) -> None:
        with contextlib.suppress(Exception):  # None sandbox -> AttributeError -> no-op
            sandbox.client.close()

    @staticmethod
    def _stop_sandbox(sandbox: Sandbox | None) -> None:
        with contextlib.suppress(Exception):  # None sandbox -> AttributeError -> no-op
            try:
                sandbox.stop(blocking=True, timeout=timedelta(seconds=15), poll_interval=timedelta(milliseconds=500))
            except TypeError:  # older SDKs: stop() takes no arguments
                sandbox.stop()

    def _snapshot_sandbox(self, sandbox: Sandbox) -> None:
        if not self._persistent or not self._task_id:
            return
        try:
            snapshot_id = _extract_snapshot_id(sandbox.snapshot())
        except Exception as exc:
            logger.warning("Vercel: filesystem snapshot failed for task %s: %s", self._task_id, exc)
            return
        if not snapshot_id:
            logger.warning("Vercel: filesystem snapshot for task %s did not return a snapshot id", self._task_id)
            return
        _store_snapshot(self._task_id, snapshot_id)
        logger.info("Vercel: saved filesystem snapshot %s for task %s", snapshot_id, self._task_id)

    def _ensure_sandbox_ready(self) -> None:
        """Reuse a healthy sandbox; recreate when refresh fails or it hit a terminal state."""
        sandbox = self._sandbox
        requested_cwd = self.cwd or self._requested_cwd or DEFAULT_VERCEL_CWD
        if sandbox is not None:
            try:
                sandbox.refresh()
            except Exception as exc:
                logger.warning("Vercel: sandbox refresh failed for task %s: %s; recreating", self._task_id, exc)
            else:
                status = sandbox.status
                if not _is_terminal(status):
                    self._wait_for_running()
                    return
                logger.warning("Vercel: sandbox entered state %s for task %s; recreating", status, self._task_id)
            self._close_sandbox_client(sandbox)
        self._attach_fresh_sandbox(requested_cwd)

    def _run_checked(self, script: str, label: str) -> None:
        output, returncode = _result_parts(
            self._require_sandbox().run_command("bash", ["-lc", script], cwd=self._workspace_root))
        if returncode != 0:
            raise RuntimeError(f"Vercel {label} failed: {output.strip()}")

    def _vercel_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return
        payload: list[WriteFile] = [
            {"path": remote_path, "content": Path(host_path).read_bytes()} for host_path, remote_path in files]
        sandbox = self._require_sandbox()
        _retry_vercel_call("write_files", lambda: sandbox.write_files(payload), attempts=3)

    def _vercel_delete(self, remote_paths: list[str]) -> None:
        if remote_paths:
            self._run_checked(quoted_rm_command(remote_paths), "delete")

    def _vercel_bulk_download(self, dest_tar_path: Path) -> None:
        archive_member = self._remote_hermes_dir().lstrip("/")
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        sandbox = self._require_sandbox()
        try:
            self._run_checked(f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(archive_member)}", "bulk download")
            sandbox.download_file(remote_tar, dest_tar_path)
        finally:
            with contextlib.suppress(Exception):
                sandbox.run_command("bash", ["-lc", f"rm -f {shlex.quote(remote_tar)}"], cwd=self._workspace_root)

    def _before_execute(self) -> None:
        with self._lock:
            self._ensure_sandbox_ready()
            if self._sync_manager is not None:
                self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120, stdin_data: str | None = None):
        """``timeout`` is enforced by the base ``_wait_for_process`` via ``cancel_fn`` (the SDK has no
        per-exec timeout); ``stdin_data`` is already embedded as a heredoc by the base ``execute()``."""
        del timeout, stdin_data
        sandbox, workspace_root, lock = self._require_sandbox(), self._workspace_root, self._lock

        def cancel() -> None:
            with lock:
                self._stop_sandbox(sandbox)

        def exec_fn() -> tuple[str, int]:
            return _result_parts(
                sandbox.run_command("bash", ["-lc" if login else "-c", cmd_string], cwd=workspace_root))
        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            sandbox, sync_manager = self._sandbox, self._sync_manager
            if sandbox is not None and sync_manager is not None:
                try:
                    sync_manager.sync_back()
                except Exception as exc:
                    logger.warning("Vercel: sync_back failed for task %s: %s", self._task_id, exc)
            self._sandbox = None
            self._sync_manager = None
        if sandbox is None:
            return
        self._snapshot_sandbox(sandbox)
        # Always stop the sandbox during cleanup to avoid resource leaks (matches Modal/Daytona).
        self._stop_sandbox(sandbox)
        self._close_sandbox_client(sandbox)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from dataclasses import dataclass  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
