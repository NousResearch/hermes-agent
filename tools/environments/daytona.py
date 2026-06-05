"""Daytona cloud execution environment.

Uses the Daytona Python SDK to run commands in cloud sandboxes.
Supports persistent sandboxes: when enabled, sandboxes are stopped on cleanup
and resumed on next creation, preserving the filesystem across sessions.
"""

import logging
import math
import os
import shlex
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)


def _get_active_profile_name() -> str:
    """Return the active Hermes profile name, or ``"default"`` on any error."""
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _sandbox_attr(sandbox, *names):
    for name in names:
        value = getattr(sandbox, name, None)
        if value is not None:
            return value
    return None


def _sandbox_labels(sandbox) -> dict:
    labels = _sandbox_attr(sandbox, "labels", "metadata")
    return labels if isinstance(labels, dict) else {}


def _sandbox_name(sandbox) -> str:
    return str(_sandbox_attr(sandbox, "name", "sandbox_name") or "")


def _sandbox_state(sandbox) -> str:
    state = _sandbox_attr(sandbox, "state", "status")
    value = getattr(state, "value", state)
    return str(value or "").lower()


def _sandbox_task_id(sandbox) -> str | None:
    labels = _sandbox_labels(sandbox)
    task_id = labels.get("hermes_task_id")
    return str(task_id) if task_id else None


def _parse_daytona_time(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, timezone.utc)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sandbox_age_seconds(sandbox, now: datetime) -> float | None:
    for attr in ("last_active_at", "updated_at", "created_at", "createdAt"):
        dt = _parse_daytona_time(_sandbox_attr(sandbox, attr))
        if dt is not None:
            return (now - dt).total_seconds()
    return None


def _pid_is_alive(pid: object) -> bool | None:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return None
    if pid_int <= 0:
        return None
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(pid_int))
    except Exception:
        return None


def reap_orphan_sandboxes(
    *,
    max_age_seconds: int = 600,
    current_task_id: str | None = None,
    profile_filter: str | None = None,
    daytona_client=None,
) -> int:
    """Stop stale Hermes-created Daytona sandboxes from prior processes.

    Daytona persistent cleanup normally stops the active sandbox to preserve
    its filesystem. The startup reaper uses the same non-destructive action
    for old sandboxes because Daytona metadata does not reliably distinguish
    persistent from disposable sandboxes after process death. Sandboxes without
    any usable age metadata are left alone.
    """
    try:
        if daytona_client is None:
            from daytona import Daytona

            daytona_client = Daytona()
    except Exception as e:
        logger.debug("Daytona orphan reaper could not initialize SDK: %s", e)
        return 0

    if not profile_filter:
        return 0
    list_kwargs = {"labels": {"hermes_profile": profile_filter}, "limit": 100}

    candidates = []
    try:
        try:
            results = daytona_client.list(**list_kwargs)
        except TypeError:
            results = daytona_client.list(labels=list_kwargs["labels"])
        candidates.extend(results)
    except Exception as e:
        logger.debug("Daytona orphan reaper list failed: %s", e)

    now = datetime.now(timezone.utc)
    stopped = 0
    for sandbox in candidates:
        labels = _sandbox_labels(sandbox)
        task_id = _sandbox_task_id(sandbox)
        sandbox_profile = labels.get("hermes_profile")
        owner_pid = labels.get("hermes_owner_pid")
        if not task_id or not sandbox_profile or not owner_pid:
            continue
        if sandbox_profile != profile_filter:
            continue
        if current_task_id and task_id == current_task_id:
            continue

        age = _sandbox_age_seconds(sandbox, now)
        if age is None or age < max_age_seconds:
            continue

        owner_alive = _pid_is_alive(owner_pid)
        if owner_alive is not False:
            continue

        state = _sandbox_state(sandbox)
        if state and state in {"stopped", "archived"}:
            continue

        try:
            sandbox.stop()
            stopped += 1
            logger.info(
                "Daytona orphan reaper stopped sandbox %s for task %s",
                _sandbox_attr(sandbox, "id") or _sandbox_name(sandbox) or "<unknown>",
                task_id or "<unknown>",
            )
        except Exception as e:
            logger.debug("Daytona orphan reaper stop failed: %s", e)
    return stopped


class DaytonaEnvironment(BaseEnvironment):
    """Daytona cloud sandbox execution backend.

    Spawn-per-call via _ThreadedProcessHandle wrapping blocking SDK calls.
    cancel_fn wired to sandbox.stop() for interrupt support.
    Shell timeout wrapper preserved (SDK timeout unreliable).
    """

    _stdin_mode = "heredoc"

    def __init__(
        self,
        image: str,
        cwd: str = "/home/daytona",
        timeout: int = 60,
        cpu: int = 1,
        memory: int = 5120,
        disk: int = 10240,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("terminal.daytona", prompt=False)
        except ImportError:
            pass
        except Exception as e:
            raise ImportError(str(e))
        from daytona import (
            Daytona,
            CreateSandboxFromImageParams,
            Resources,
            SandboxState,
        )

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._SandboxState = SandboxState
        self._daytona = Daytona()
        self._sandbox = None
        self._lock = threading.Lock()

        memory_gib = max(1, math.ceil(memory / 1024))
        disk_gib = max(1, math.ceil(disk / 1024))
        if disk_gib > 10:
            logger.warning(
                "Daytona: requested disk (%dGB) exceeds platform limit (10GB). "
                "Capping to 10GB.", disk_gib,
            )
            disk_gib = 10
        resources = Resources(cpu=cpu, memory=memory_gib, disk=disk_gib)

        profile = _get_active_profile_name()
        labels = {
            "hermes_task_id": task_id,
            "hermes_profile": profile,
            "hermes_owner_pid": str(os.getpid()),
        }
        sandbox_name = f"hermes-{uuid.uuid4().hex[:12]}"

        if self._persistent:
            try:
                # Daytona SDK >=0.108.0 uses cursor-based pagination and
                # list() returns an iterator. Offset-based pagination (page=1)
                # is removed on June 10, 2026.
                results = self._daytona.list(
                    labels={"hermes_task_id": task_id, "hermes_profile": profile},
                    limit=1,
                )
                self._sandbox = next(iter(results), None)
                if self._sandbox is not None and (
                    _sandbox_labels(self._sandbox).get("hermes_task_id") != task_id
                    or _sandbox_labels(self._sandbox).get("hermes_profile") != profile
                ):
                    self._sandbox = None
                if self._sandbox is not None:
                    self._sandbox.set_labels(labels)
                    self._sandbox.start()
                    logger.info("Daytona: resumed sandbox %s for task %s",
                                self._sandbox.id, task_id)
            except Exception as e:
                logger.debug("Daytona: no reusable sandbox found for task %s: %s",
                             task_id, e)
                self._sandbox = None

        if self._sandbox is None:
            self._sandbox = self._daytona.create(
                CreateSandboxFromImageParams(
                    image=image,
                    name=sandbox_name,
                    labels=labels,
                    auto_stop_interval=0,
                    resources=resources,
                )
            )
            logger.info("Daytona: created sandbox %s for task %s",
                        self._sandbox.id, task_id)

        # Detect remote home dir
        self._remote_home = "/root"
        try:
            home = self._sandbox.process.exec("echo $HOME").result.strip()
            if home:
                self._remote_home = home
                if requested_cwd in {"~", "/home/daytona"}:
                    self.cwd = home
        except Exception:
            pass
        logger.info("Daytona: resolved home to %s, cwd to %s", self._remote_home, self.cwd)

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._daytona_upload,
            delete_fn=self._daytona_delete,
            bulk_upload_fn=self._daytona_bulk_upload,
            bulk_download_fn=self._daytona_bulk_download,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    def _daytona_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via Daytona SDK."""
        parent = str(PurePosixPath(remote_path).parent)
        self._sandbox.process.exec(quoted_mkdir_command([parent]))
        self._sandbox.fs.upload_file(host_path, remote_path)

    def _daytona_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in a single HTTP call via Daytona SDK.

        Uses ``sandbox.fs.upload_files()`` which batches all files into one
        multipart POST, avoiding per-file TLS/HTTP overhead (~580 files
        goes from ~5 min to <2 s).
        """
        from daytona.common.filesystem import FileUpload

        if not files:
            return

        parents = unique_parent_dirs(files)
        if parents:
            self._sandbox.process.exec(quoted_mkdir_command(parents))

        uploads = [
            FileUpload(source=host_path, destination=remote_path)
            for host_path, remote_path in files
        ]
        self._sandbox.fs.upload_files(uploads)

    def _daytona_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        # PID-suffixed remote temp path avoids collisions if sync_back fires
        # concurrently for the same sandbox (e.g. retry after partial failure).
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        self._sandbox.process.exec(
            f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}"
        )
        self._sandbox.fs.download_file(remote_tar, str(dest))
        # Clean up remote temp file
        try:
            self._sandbox.process.exec(f"rm -f {shlex.quote(remote_tar)}")
        except Exception:
            pass  # best-effort cleanup

    def _daytona_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via SDK exec."""
        self._sandbox.process.exec(quoted_rm_command(remote_paths))

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def _ensure_sandbox_ready(self) -> None:
        """Restart sandbox if it was stopped (e.g., by a previous interrupt)."""
        self._sandbox.refresh_data()
        if self._sandbox.state in {self._SandboxState.STOPPED, self._SandboxState.ARCHIVED}:
            self._sandbox.start()
            logger.info("Daytona: restarted sandbox %s", self._sandbox.id)

    def _before_execute(self) -> None:
        """Ensure sandbox is ready, then sync files via FileSyncManager."""
        with self._lock:
            self._ensure_sandbox_ready()
        self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping a blocking Daytona SDK call."""
        sandbox = self._sandbox
        lock = self._lock

        def cancel():
            with lock:
                try:
                    sandbox.stop()
                except Exception:
                    pass

        if login:
            shell_cmd = f"bash -l -c {shlex.quote(cmd_string)}"
        else:
            shell_cmd = f"bash -c {shlex.quote(cmd_string)}"

        def exec_fn() -> tuple[str, int]:
            response = sandbox.process.exec(shell_cmd, timeout=timeout)
            return (response.result or "", response.exit_code)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            if self._sandbox is None:
                return

            # Sync remote changes back to host before teardown. Running
            # inside the lock (and after the _sandbox is None guard) avoids
            # firing sync_back on an already-cleaned-up env, which would
            # trigger a 3-attempt retry storm against a nil sandbox.
            if self._sync_manager:
                logger.info("Daytona: syncing files from sandbox...")
                try:
                    self._sync_manager.sync_back()
                except Exception as e:
                    logger.warning("Daytona: sync_back failed: %s", e)

            try:
                if self._persistent:
                    self._sandbox.stop()
                    logger.info("Daytona: stopped sandbox %s (filesystem preserved)",
                                self._sandbox.id)
                else:
                    self._daytona.delete(self._sandbox)
                    logger.info("Daytona: deleted sandbox %s", self._sandbox.id)
            except Exception as e:
                logger.warning("Daytona: cleanup failed: %s", e)
            self._sandbox = None
