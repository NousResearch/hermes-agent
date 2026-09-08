"""Daytona cloud execution environment.

Runs commands in Daytona cloud sandboxes via the Python SDK. Persistent mode stops
the sandbox on cleanup and resumes it next time, preserving the filesystem.
"""

import contextlib
import logging
import math
import os
import shlex
import threading
from pathlib import Path

from tools.environments.base import BaseEnvironment
from tools.environments.base_output import _ThreadedProcessHandle
from tools.environments.file_sync import (
    FileSyncManager, iter_sync_files, quoted_mkdir_command, quoted_rm_command, unique_parent_dirs)
from tools.environments.remote_common import ensure_lazy_dep

logger = logging.getLogger(__name__)


class DaytonaEnvironment(BaseEnvironment):
    """Daytona cloud sandbox execution backend.

    Spawn-per-call via _ThreadedProcessHandle wrapping blocking SDK calls; cancel_fn
    is wired to sandbox.stop() for interrupts. Shell timeout wrapper kept (SDK timeout unreliable).
    """

    _stdin_mode = "heredoc"

    def __init__(self, image: str, cwd: str = "/home/daytona", timeout: int = 60, cpu: int = 1,
                 memory: int = 5120, disk: int = 10240, persistent_filesystem: bool = True,
                 task_id: str = "default"):
        super().__init__(cwd=cwd, timeout=timeout)
        ensure_lazy_dep("terminal.daytona")
        from daytona import Daytona, CreateSandboxFromImageParams, DaytonaError, Resources, SandboxState

        self._persistent, self._task_id, self._SandboxState = persistent_filesystem, task_id, SandboxState
        self._daytona = Daytona()
        self._sandbox = None
        self._lock = threading.Lock()

        memory_gib, disk_gib = max(1, math.ceil(memory / 1024)), max(1, math.ceil(disk / 1024))
        if disk_gib > 10:
            logger.warning("Daytona: requested disk (%dGB) exceeds platform limit (10GB). "
                           "Capping to 10GB.", disk_gib)
            disk_gib = 10
        resources = Resources(cpu=cpu, memory=memory_gib, disk=disk_gib)
        labels, sandbox_name = {"hermes_task_id": task_id}, f"hermes-{task_id}"

        if self._persistent:
            try:
                self._sandbox = self._daytona.get(sandbox_name)
                self._sandbox.start()
                logger.info("Daytona: resumed sandbox %s for task %s", self._sandbox.id, task_id)
            except Exception as e:
                if not isinstance(e, DaytonaError):  # DaytonaError == not found: silently fall through
                    logger.warning("Daytona: failed to resume sandbox for task %s: %s", task_id, e)
                self._sandbox = None
            if self._sandbox is None:
                try:
                    # SDK list() is a cursor-paginated iterator (offset pagination is gone).
                    self._sandbox = next(iter(self._daytona.list(labels=labels, limit=1)), None)
                    if self._sandbox is not None:
                        self._sandbox.start()
                        logger.info("Daytona: resumed legacy sandbox %s for task %s", self._sandbox.id, task_id)
                except Exception as e:
                    logger.debug("Daytona: no legacy sandbox found for task %s: %s", task_id, e)
                    self._sandbox = None
        if self._sandbox is None:
            self._sandbox = self._daytona.create(CreateSandboxFromImageParams(
                image=image, name=sandbox_name, labels=labels, auto_stop_interval=0, resources=resources))
            logger.info("Daytona: created sandbox %s for task %s", self._sandbox.id, task_id)

        self._remote_home = "/root"
        with contextlib.suppress(Exception):
            home = self._sandbox.process.exec("echo $HOME").result.strip()
            if home:
                self._remote_home = home
                if cwd in {"~", "/home/daytona"}:
                    self.cwd = home
        logger.info("Daytona: resolved home to %s, cwd to %s", self._remote_home, self.cwd)

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._daytona_upload, delete_fn=self._daytona_delete,
            bulk_upload_fn=self._daytona_bulk_upload, bulk_download_fn=self._daytona_bulk_download)
        self._sync_manager.sync(force=True)
        self.init_session()

    def _daytona_upload(self, host_path: str, remote_path: str) -> None:
        self._sandbox.process.exec(quoted_mkdir_command([str(Path(remote_path).parent)]))
        self._sandbox.fs.upload_file(host_path, remote_path)

    def _daytona_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in one multipart POST via ``sandbox.fs.upload_files()``."""
        from daytona.common.filesystem import FileUpload

        if not files:
            return
        parents = unique_parent_dirs(files)
        if parents:
            self._sandbox.process.exec(quoted_mkdir_command(parents))
        self._sandbox.fs.upload_files(
            [FileUpload(source=host_path, destination=remote_path) for host_path, remote_path in files])

    def _daytona_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        # PID-suffixed remote temp path avoids collisions if sync_back runs concurrently.
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        self._sandbox.process.exec(f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}")
        self._sandbox.fs.download_file(remote_tar, str(dest))
        with contextlib.suppress(Exception):  # best-effort cleanup
            self._sandbox.process.exec(f"rm -f {shlex.quote(remote_tar)}")

    def _daytona_delete(self, remote_paths: list[str]) -> None:
        self._sandbox.process.exec(quoted_rm_command(remote_paths))

    def _ensure_sandbox_ready(self) -> None:
        """Restart sandbox if it was stopped (e.g., by a previous interrupt)."""
        self._sandbox.refresh_data()
        if self._sandbox.state in {self._SandboxState.STOPPED, self._SandboxState.ARCHIVED}:
            self._sandbox.start()
            logger.info("Daytona: restarted sandbox %s", self._sandbox.id)

    def _before_execute(self) -> None:
        with self._lock:
            self._ensure_sandbox_ready()
        self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120,
                  stdin_data: str | None = None):
        sandbox, lock = self._sandbox, self._lock

        def cancel():
            with lock, contextlib.suppress(Exception):
                sandbox.stop()

        shell_cmd = f"bash {'-l ' if login else ''}-c {shlex.quote(cmd_string)}"

        def exec_fn() -> tuple[str, int]:
            response = sandbox.process.exec(shell_cmd, timeout=timeout)
            return (response.result or "", response.exit_code)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            if self._sandbox is None:
                return
            # sync_back runs inside the lock and after the None guard so an
            # already-cleaned-up env can't trigger a 3-attempt retry storm on a nil sandbox.
            if self._sync_manager:
                logger.info("Daytona: syncing files from sandbox...")
                try:
                    self._sync_manager.sync_back()
                except Exception as e:
                    logger.warning("Daytona: sync_back failed: %s", e)
            try:
                if self._persistent:
                    self._sandbox.stop()
                    logger.info("Daytona: stopped sandbox %s (filesystem preserved)", self._sandbox.id)
                else:
                    self._daytona.delete(self._sandbox)
                    logger.info("Daytona: deleted sandbox %s", self._sandbox.id)
            except Exception as e:
                logger.warning("Daytona: cleanup failed: %s", e)
            self._sandbox = None
