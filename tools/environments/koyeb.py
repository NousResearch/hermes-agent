"""Koyeb cloud execution environment.

Uses the Koyeb Python SDK to run commands in cloud sandboxes.
Supports persistent sandboxes: when enabled, sandboxes are stopped on cleanup
and resumed on next creation, preserving the filesystem across sessions.
"""

import logging
import math
import os
import shlex
import threading
from pathlib import Path
from typing import Any

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


class KoyebEnvironment(BaseEnvironment):
    """Koyeb cloud sandbox execution backend.

    Spawn-per-call via _ThreadedProcessHandle wrapping blocking SDK calls.
    cancel_fn wired to sandbox.delete() for interrupt support.
    Shell timeout wrapper preserved (SDK timeout unreliable).
    """

    _stdin_mode = "heredoc"

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        cpu: int = 1,
        memory: int = 5120,
        disk: int = 10240,
        persistent_filesystem: bool = True,
        task_id: str = "default",
        instance_type: str = "micro",
        region: str = None,
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        from koyeb import Sandbox

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._sandbox = None
        self._lock = threading.Lock()
        self._instance_type = instance_type
        self._region = region or os.getenv("KOYEB_REGION", "na")
        self._api_token = os.getenv("KOYEB_API_TOKEN")

        # Convert memory from MB to GB (Koyeb uses GB)
        memory_gib = max(1, math.ceil(memory / 1024))
        
        # Koyeb instance types: micro, small, medium, large, xlarge, 2xlarge, etc.
        # For now, we'll use the instance_type parameter directly
        # cpu and memory parameters are kept for compatibility but may be overridden by instance_type

        sandbox_name = f"hermes-{task_id}"
        labels = {"hermes_task_id": task_id}

        # Try to reuse existing sandbox if persistent
        if self._persistent:
            try:
                # List existing sandboxes with our label
                existing = Sandbox.list(api_token=self._api_token, labels=labels)
                if existing:
                    self._sandbox = existing[0]
                    logger.info("Koyeb: resumed sandbox %s for task %s",
                                self._sandbox.id, task_id)
            except Exception as e:
                logger.debug("Koyeb: could not resume sandbox for task %s: %s",
                             task_id, e)
                self._sandbox = None

        # Create new sandbox if needed
        if self._sandbox is None:
            try:
                self._sandbox = Sandbox.create(
                    image=image,
                    name=sandbox_name,
                    wait_ready=True,
                    instance_type=self._instance_type,
                    region=self._region,
                    api_token=self._api_token,
                    timeout=300,
                    idle_timeout=0,  # Disable auto-sleep for persistent sandboxes
                    delete_after_delay=0,
                    delete_after_inactivity_delay=0,
                )
                logger.info("Koyeb: created sandbox %s for task %s",
                            self._sandbox.id, task_id)
            except Exception as e:
                logger.error("Koyeb: failed to create sandbox: %s", e)
                raise

        # Detect remote home dir
        self._remote_home = "/root"
        try:
            home = self._sandbox.exec("echo $HOME").stdout.strip()
            if home:
                self._remote_home = home
                if requested_cwd in ("~", "/root"):
                    self.cwd = home
        except Exception:
            pass
        logger.info("Koyeb: resolved home to %s, cwd to %s", self._remote_home, self.cwd)

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._koyeb_upload,
            delete_fn=self._koyeb_delete,
            bulk_upload_fn=self._koyeb_bulk_upload,
            bulk_download_fn=self._koyeb_bulk_download,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    def _koyeb_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via Koyeb SDK."""
        parent = str(Path(remote_path).parent)
        self._sandbox.exec(f"mkdir -p {shlex.quote(parent)}")
        self._sandbox.filesystem.upload_file(host_path, remote_path)

    def _koyeb_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files via Koyeb SDK."""
        if not files:
            return

        parents = unique_parent_dirs(files)
        if parents:
            self._sandbox.exec(quoted_mkdir_command(parents))

        # Upload files one by one (Koyeb SDK doesn't have bulk upload for files)
        for host_path, remote_path in files:
            self._sandbox.filesystem.upload_file(host_path, remote_path)

    def _koyeb_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        # PID-suffixed remote temp path avoids collisions if sync_back fires
        # concurrently for the same sandbox (e.g. retry after partial failure).
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        self._sandbox.exec(
            f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}"
        )
        self._sandbox.filesystem.download_file(remote_tar, str(dest))
        # Clean up remote temp file
        try:
            self._sandbox.exec(f"rm -f {shlex.quote(remote_tar)}")
        except Exception:
            pass  # best-effort cleanup

    def _koyeb_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via SDK exec."""
        self._sandbox.exec(quoted_rm_command(remote_paths))

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def _ensure_sandbox_ready(self) -> None:
        """Restart sandbox if it was stopped (e.g., by a previous interrupt)."""
        # Koyeb sandboxes don't have a stopped state like Daytona
        # They're either running or need to be recreated
        pass

    def _before_execute(self) -> None:
        """Ensure sandbox is ready, then sync files via FileSyncManager."""
        with self._lock:
            self._ensure_sandbox_ready()
        self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping a blocking Koyeb SDK call."""
        sandbox = self._sandbox
        lock = self._lock

        def cancel():
            with lock:
                try:
                    sandbox.delete()
                except Exception:
                    pass

        if login:
            shell_cmd = f"bash -l -c {shlex.quote(cmd_string)}"
        else:
            shell_cmd = f"bash -c {shlex.quote(cmd_string)}"

        def exec_fn() -> tuple[str, int]:
            result = sandbox.exec(shell_cmd, timeout=timeout)
            output = result.stdout or ""
            if result.stderr:
                output = f"{output}\n{result.stderr}" if output else result.stderr
            return (output, result.exit_code)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            if self._sandbox is None:
                return

            # Sync remote changes back to host before teardown
            if self._sync_manager:
                logger.info("Koyeb: syncing files from sandbox...")
                try:
                    self._sync_manager.sync_back()
                except Exception as e:
                    logger.warning("Koyeb: sync_back failed: %s", e)

            try:
                if self._persistent:
                    # For persistent sandboxes, we don't delete them
                    # They'll be reused on next creation
                    logger.info("Koyeb: keeping sandbox %s (filesystem preserved)",
                                self._sandbox.id)
                else:
                    self._sandbox.delete()
                    logger.info("Koyeb: deleted sandbox %s", self._sandbox.id)
            except Exception as e:
                logger.warning("Koyeb: cleanup failed: %s", e)
            self._sandbox = None
