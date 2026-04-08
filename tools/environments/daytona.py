"""Daytona cloud execution environment.

Uses the Daytona Python SDK to run commands in cloud sandboxes.
Supports persistent sandboxes: when enabled, sandboxes are stopped on cleanup
and resumed on next creation, preserving the filesystem across sessions.
"""

import logging
import math
import shlex
import threading
import warnings
from pathlib import Path
from typing import Dict, Optional

from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
    _file_mtime_key,
)

logger = logging.getLogger(__name__)


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

        from daytona import (
            Daytona,
            CreateSandboxFromImageParams,
            DaytonaError,
            Resources,
            SandboxState,
        )

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._SandboxState = SandboxState
        self._DaytonaError = DaytonaError
        self._daytona = Daytona()
        self._sandbox = None
        self._lock = threading.Lock()
        self._last_sync_time: float = 0

        memory_gib = max(1, math.ceil(memory / 1024))
        disk_gib = max(1, math.ceil(disk / 1024))
        if disk_gib > 10:
            warnings.warn(
                f"Daytona: requested disk ({disk_gib}GB) exceeds platform limit (10GB). "
                f"Capping to 10GB.",
                stacklevel=2,
            )
            disk_gib = 10
        resources = Resources(cpu=cpu, memory=memory_gib, disk=disk_gib)

        labels = {"hermes_task_id": task_id}
        sandbox_name = f"hermes-{task_id}"

        if self._persistent:
            try:
                self._sandbox = self._daytona.get(sandbox_name)
                self._sandbox.start()
                logger.info("Daytona: resumed sandbox %s for task %s",
                            self._sandbox.id, task_id)
            except DaytonaError:
                self._sandbox = None
            except Exception as e:
                logger.warning("Daytona: failed to resume sandbox for task %s: %s",
                               task_id, e)
                self._sandbox = None

            if self._sandbox is None:
                try:
                    page = self._daytona.list(labels=labels, page=1, limit=1)
                    if page.items:
                        self._sandbox = page.items[0]
                        self._sandbox.start()
                        logger.info("Daytona: resumed legacy sandbox %s for task %s",
                                    self._sandbox.id, task_id)
                except Exception as e:
                    logger.debug("Daytona: no legacy sandbox found for task %s: %s",
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
                if requested_cwd in ("~", "/home/daytona"):
                    self.cwd = home
        except Exception:
            pass
        logger.info("Daytona: resolved home to %s, cwd to %s", self._remote_home, self.cwd)

        self._synced_files: Dict[str, tuple] = {}
        self._sync_files()
        self.init_session()

    def _upload_if_changed(self, host_path: str, remote_path: str) -> bool:
        file_key = _file_mtime_key(host_path)
        if file_key is None:
            return False
        if self._synced_files.get(remote_path) == file_key:
            return False
        try:
            parent = str(Path(remote_path).parent)
            self._sandbox.process.exec(f"mkdir -p {parent}")
            self._sandbox.fs.upload_file(host_path, remote_path)
            self._synced_files[remote_path] = file_key
            return True
        except Exception as e:
            logger.debug("Daytona: upload failed %s: %s", host_path, e)
            return False

    def _sync_files(self) -> None:
        container_base = f"{self._remote_home}/.hermes"
        try:
            from tools.credential_files import get_credential_file_mounts, iter_skills_files
            for mount_entry in get_credential_file_mounts():
                remote_path = mount_entry["container_path"].replace("/root/.hermes", container_base, 1)
                self._upload_if_changed(mount_entry["host_path"], remote_path)
            for entry in iter_skills_files(container_base=container_base):
                self._upload_if_changed(entry["host_path"], entry["container_path"])
        except Exception as e:
            logger.debug("Daytona: could not sync skills/credentials: %s", e)

    def _ensure_sandbox_ready(self):
        """Restart sandbox if it was stopped (e.g., by a previous interrupt)."""
        self._sandbox.refresh_data()
        if self._sandbox.state in (self._SandboxState.STOPPED, self._SandboxState.ARCHIVED):
            self._sandbox.start()
            logger.info("Daytona: restarted sandbox %s", self._sandbox.id)

    def _before_execute(self):
        """Ensure sandbox is ready, then rate-limited file sync via base class."""
        with self._lock:
            self._ensure_sandbox_ready()
        super()._before_execute()

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
