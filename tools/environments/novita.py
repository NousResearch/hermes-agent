"""Novita AI cloud execution environment.

Uses the Novita Sandbox Python SDK to run commands in cloud sandboxes.
Supports persistent sandboxes: when enabled, sandboxes are paused on cleanup
and resumed on next creation, preserving the filesystem across sessions.
"""

import logging
import os
import shlex
import tarfile
import tempfile
import threading
import time
from pathlib import Path

from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_rm_command,
)

logger = logging.getLogger(__name__)


class NovitaEnvironment(BaseEnvironment):
    """Novita AI cloud sandbox execution backend.

    Spawn-per-call via _ThreadedProcessHandle wrapping blocking SDK calls.
    cancel_fn wired to sandbox.kill() for interrupt support.
    Shell timeout wrapper preserved (SDK timeout unreliable).
    """

    _stdin_mode = "heredoc"
    _sandbox_timeout = 30 * 60
    _snapshot_timeout = 60  # Novita cold-starts may be slower than local
    _cleanup_sync_timeout = 60
    _cleanup_lifecycle_timeout = 15

    def __init__(
        self,
        template: str = "",
        cwd: str = "/home/user",
        timeout: int = 60,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("terminal.novita", prompt=False)
        except ImportError:
            pass
        except Exception as e:
            raise ImportError(str(e))

        from novita_sandbox.core import Sandbox, SandboxQuery, CommandExitException

        self._persistent = persistent_filesystem
        self._CommandExitException = CommandExitException
        self._task_id = task_id
        self._sandbox = None
        self._lock = threading.Lock()
        self._invalidated = False

        metadata = {"hermes_task_id": task_id}
        template_id = template if template else None

        if self._persistent:
            try:
                paginator = Sandbox.list(
                    query=SandboxQuery(metadata=metadata),
                    limit=1,
                )
                if paginator.has_next:
                    items = paginator.next_items()
                    if items:
                        sandbox_info = items[0]
                        self._sandbox = Sandbox.connect(sandbox_info.sandbox_id)
                        logger.info(
                            "Novita: resumed sandbox %s for task %s",
                            sandbox_info.sandbox_id, task_id,
                        )
            except Exception as e:
                logger.warning(
                    "Novita: failed to find/resume sandbox for task %s: %s",
                    task_id, e,
                )
                self._sandbox = None

        if self._sandbox is None:
            self._sandbox = Sandbox.create(
                template=template_id,
                timeout=self._sandbox_timeout,
                metadata=metadata,
                secure=True,
            )
            logger.info(
                "Novita: created sandbox %s for task %s",
                self._sandbox.sandbox_id, task_id,
            )

        # Detect remote home dir
        self._remote_home = "/home/user"
        try:
            try:
                result = self._sandbox.commands.run("echo $HOME", timeout=15)
            except CommandExitException as e:
                result = e
            home = result.stdout.strip()
            if home:
                self._remote_home = home
                if requested_cwd in ("~", "/home/user", "/home/daytona", "/root"):
                    self.cwd = home
        except Exception:
            pass
        logger.info(
            "Novita: resolved home to %s, cwd to %s",
            self._remote_home, self.cwd,
        )

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._novita_upload,
            delete_fn=self._novita_delete,
            bulk_upload_fn=self._novita_bulk_upload,
            bulk_download_fn=self._novita_bulk_download,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    def _novita_run(self, cmd: str, timeout: int = 15) -> None:
        """Run a fire-and-forget command, tolerating non-zero exit codes."""
        try:
            self._sandbox.commands.run(cmd, timeout=timeout)
        except self._CommandExitException:
            pass  # non-zero exit is acceptable for maintenance commands

    def _novita_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via Novita SDK."""
        parent = str(Path(remote_path).parent)
        self._novita_run(f"mkdir -p {parent}")
        with open(host_path, "rb") as f:
            self._sandbox.files.write(remote_path, f.read())

    def _novita_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files as one compressed tarball and extract remotely."""
        if not files:
            return

        remote_tar = f"/tmp/.hermes_upload.{os.getpid()}.{time.time_ns()}.tar.gz"
        local_tar = None
        try:
            with tempfile.NamedTemporaryFile(prefix="hermes-novita-upload-", suffix=".tar.gz", delete=False) as tmp:
                local_tar = Path(tmp.name)

            with tarfile.open(local_tar, "w:gz") as tar:
                for host_path, remote_path in files:
                    tar.add(host_path, arcname=remote_path.lstrip("/"), recursive=False)

            self._sandbox.files.write(remote_tar, local_tar.read_bytes())
            self._sandbox.commands.run(
                f"tar -xzf {shlex.quote(remote_tar)} -C /",
                timeout=120,
            )
        finally:
            if local_tar is not None:
                local_tar.unlink(missing_ok=True)
            try:
                self._novita_run(f"rm -f {shlex.quote(remote_tar)}")
            except Exception:
                pass  # best-effort cleanup

    def _novita_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via SDK exec."""
        self._novita_run(quoted_rm_command(remote_paths))

    def _run_with_timeout(self, label: str, fn, timeout: float) -> bool:
        """Run a blocking SDK cleanup call with a bounded wait."""
        done = threading.Event()
        error: list[BaseException] = []

        def _worker():
            try:
                fn()
            except BaseException as exc:
                error.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=_worker, daemon=True, name=f"novita-{label}")
        thread.start()
        if not done.wait(timeout):
            logger.warning("Novita: %s timed out after %.1fs", label, timeout)
            return False
        if error:
            raise error[0]
        return True

    def _novita_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a compressed tar archive."""
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar.gz"
        self._sandbox.commands.run(
            f"tar -czf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}",
            timeout=120,
        )
        data = self._sandbox.files.read(
            remote_tar,
            format="bytes",
            request_timeout=120,
        )
        dest.write_bytes(bytes(data))
        try:
            self._novita_run(f"rm -f {shlex.quote(remote_tar)}")
        except Exception:
            pass  # best-effort cleanup

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def _before_execute(self) -> None:
        """Sync files via FileSyncManager before each command."""
        if self._invalidated:
            raise RuntimeError(
                "Novita sandbox was invalidated by a previous interrupt or timeout"
            )
        try:
            if hasattr(self._sandbox, "is_running") and not self._sandbox.is_running():
                self._invalidated = True
                raise RuntimeError("Novita sandbox is no longer running")
        except RuntimeError:
            raise
        except Exception as e:
            logger.debug("Novita: could not verify sandbox state: %s", e)
        self._sync_manager.sync()

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        """Return a _ThreadedProcessHandle wrapping a blocking Novita SDK call.

        Novita commands.run() runs the command via ``bash -l -c {cmd}``,
        so cmd_string is passed directly without an extra shell wrapper.
        The ``login`` flag has no effect since Novita always uses a login shell.
        """
        sandbox = self._sandbox
        lock = self._lock

        def cancel():
            with lock:
                try:
                    sandbox.kill()
                    self._invalidated = True
                except Exception:
                    pass

        CommandExitException = self._CommandExitException

        def exec_fn() -> tuple[str, int]:
            try:
                result = sandbox.commands.run(cmd_string, timeout=int(timeout))
            except CommandExitException as e:
                # Non-zero exit is normal — not a fatal error.
                # CommandExitException IS a CommandResult, so it has stdout/stderr/exit_code.
                combined = e.stdout
                if e.stderr:
                    combined = combined + e.stderr
                return (combined, e.exit_code)
            combined = result.stdout
            if result.stderr:
                combined = combined + result.stderr
            return (combined, result.exit_code)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            if self._sandbox is None:
                return
            started = time.monotonic()
            if self._sync_manager:
                logger.info("Novita: syncing files from sandbox...")
                try:
                    self._run_with_timeout(
                        "sync_back",
                        self._sync_manager.sync_back,
                        self._cleanup_sync_timeout,
                    )
                    logger.info(
                        "Novita: sync_back completed in %.1fs",
                        time.monotonic() - started,
                    )
                except Exception as e:
                    logger.warning("Novita: sync_back failed: %s", e)
            try:
                if self._persistent:
                    pause_started = time.monotonic()
                    self._run_with_timeout(
                        "beta_pause",
                        self._sandbox.beta_pause,
                        self._cleanup_lifecycle_timeout,
                    )
                    logger.info(
                        "Novita: paused sandbox %s (filesystem preserved) in %.1fs",
                        self._sandbox.sandbox_id,
                        time.monotonic() - pause_started,
                    )
                else:
                    kill_started = time.monotonic()
                    self._run_with_timeout(
                        "kill",
                        self._sandbox.kill,
                        self._cleanup_lifecycle_timeout,
                    )
                    logger.info(
                        "Novita: killed sandbox %s in %.1fs",
                        self._sandbox.sandbox_id,
                        time.monotonic() - kill_started,
                    )
            except Exception as e:
                logger.warning("Novita: cleanup failed: %s", e)
            self._sandbox = None
