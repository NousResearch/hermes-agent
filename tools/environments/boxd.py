"""boxd cloud execution environment.

Uses the boxd Python SDK (sync API) to run commands inside a boxd VM.
Supports persistent VMs: when enabled, the VM is suspended on cleanup
(sub-millisecond resume) and rehydrated on next session, preserving the
filesystem and warm process state across runs. When disabled, VMs are
destroyed on cleanup.

Filesystem sync uses tar + ``write_file`` / ``read_file`` for batched
transfer. Single-file uploads go through ``box.write_file`` directly.
"""

import io
import logging
import os
import shlex
import tarfile
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
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)


def _mb_to_size_str(mb: int) -> str:
    """Convert MB integer to a boxd size string ("NG" or "NM").

    boxd's BoxConfig accepts strings like "8G" or "512M". Hermes' container
    config tracks memory/disk in MB. Round up to the nearest GB when >=1GB,
    otherwise emit MB so callers can request small footprints in tests.
    """
    if mb <= 0:
        return ""
    if mb >= 1024:
        gib = (mb + 1023) // 1024
        return f"{gib}G"
    return f"{mb}M"


class BoxdEnvironment(BaseEnvironment):
    """boxd cloud VM execution backend.

    Spawn-per-call via _ThreadedProcessHandle wrapping blocking SDK calls.
    cancel_fn is wired to ``box.suspend()`` so an interrupted command halts
    the whole VM (matching Daytona's sandbox.stop() behavior).
    """

    _stdin_mode = "heredoc"
    _snapshot_timeout = 60  # boxd cold starts can include image fetch

    def __init__(
        self,
        image: str = "",
        cwd: str = "/root",
        timeout: int = 60,
        cpu: int = 2,
        memory: int = 8192,
        disk: int = 102400,
        persistent_filesystem: bool = True,
        task_id: str = "default",
        auto_suspend_timeout: int = 300,
        compute=None,
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("terminal.boxd", prompt=False)
        except ImportError:
            pass
        except Exception as e:
            raise ImportError(str(e))
        from boxd import (
            BoxConfig,
            BoxdError,
            Compute,
            LifecycleConfig,
            NotFoundError,
        )

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._BoxdError = BoxdError
        self._NotFoundError = NotFoundError
        # If the caller injects a Compute we don't own its lifecycle —
        # closing it on cleanup would surprise them. Tests that spin up
        # many BoxdEnvironments in series share one Compute to dodge an
        # SDK-side auth-state issue that bites after ~6 close cycles
        # in a single process; production keeps the default (own it).
        self._owns_compute = compute is None
        self._compute = compute if compute is not None else Compute()
        self._box = None
        self._lock = threading.Lock()

        vm_name = f"hermes-{task_id}"

        config_kwargs: dict = {
            "vcpu": int(cpu) if cpu else 0,
            "memory": _mb_to_size_str(int(memory)),
            "disk": _mb_to_size_str(int(disk)),
            "lifecycle": LifecycleConfig(auto_suspend_timeout=int(auto_suspend_timeout)),
        }
        # Drop empty memory/disk so the server default kicks in instead of
        # being forced to "" (which the gRPC layer may reject).
        config_kwargs = {k: v for k, v in config_kwargs.items() if v not in ("", 0)}
        # vcpu==0 is fine — the server treats it as "default", but we want
        # callers to be able to request small footprints in tests.
        if int(cpu) > 0:
            config_kwargs["vcpu"] = int(cpu)
        box_config = BoxConfig(**config_kwargs)

        if self._persistent:
            try:
                self._box = self._compute.box.get(vm_name)
                logger.info("boxd: found existing VM %s for task %s",
                            self._box.id, task_id)
                self._wake_box(self._box)
            except NotFoundError:
                self._box = None
            except Exception as e:
                logger.warning("boxd: failed to resume VM for task %s: %s",
                               task_id, e)
                self._box = None

        if self._box is None:
            create_kwargs = {"name": vm_name, "config": box_config}
            if image:
                create_kwargs["image"] = image
            self._box = self._compute.box.create(**create_kwargs)
            logger.info("boxd: created VM %s for task %s",
                        self._box.id, task_id)

        # Detect remote $HOME so file sync targets the right path.
        # The first few exec calls after create() can fail with
        # ConnectionError while the in-VM exec endpoint comes up — retry
        # briefly so we don't fall back to the wrong default and corrupt
        # every subsequent file sync.
        self._remote_home = "/root"
        deadline = time.monotonic() + 10.0
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                home_result = self._box.exec("bash", "-c", "echo $HOME")
                home = (home_result.stdout or "").strip()
                if home:
                    self._remote_home = home
                    if requested_cwd in ("~", "/root"):
                        self.cwd = home
                    break
            except Exception as e:
                logger.debug(
                    "boxd: $HOME detect attempt %d failed (%s); retrying",
                    attempt, type(e).__name__,
                )
                time.sleep(0.5)
        else:
            logger.warning(
                "boxd: $HOME detection never succeeded — falling back to %s. "
                "File sync may target the wrong directory.",
                self._remote_home,
            )
        logger.info("boxd: resolved home to %s, cwd to %s",
                    self._remote_home, self.cwd)

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._boxd_upload,
            delete_fn=self._boxd_delete,
            bulk_upload_fn=self._boxd_bulk_upload,
            bulk_download_fn=self._boxd_bulk_download,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    # ------------------------------------------------------------------
    # File sync
    # ------------------------------------------------------------------

    def _boxd_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via boxd's write_file."""
        parent = str(Path(remote_path).parent)
        self._box.exec("bash", "-c", f"mkdir -p {shlex.quote(parent)}")
        content = Path(host_path).read_bytes()
        self._box.write_file(content, remote_path)

    def _boxd_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in one round trip via a tar archive.

        Builds a gzipped tar in memory, ships it to the VM with a single
        ``write_file`` call, then extracts it via ``tar xzf`` and deletes
        the staging file. This avoids the per-file gRPC overhead that
        kills startup time when ~hundreds of skill / memory files need to
        sync at once.
        """
        if not files:
            return

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for host_path, remote_path in files:
                tar.add(host_path, arcname=remote_path.lstrip("/"))

        # PID-suffixed remote path avoids collisions if sync fires
        # concurrently for the same VM (e.g. retry after partial failure).
        remote_tar = f"/tmp/.hermes-bulk-up.{os.getpid()}.tar.gz"
        self._box.write_file(buf.getvalue(), remote_tar)

        parents = unique_parent_dirs(files)
        mkdir_part = quoted_mkdir_command(parents)
        cmd = (
            f"{mkdir_part} && "
            f"tar xzf {shlex.quote(remote_tar)} -C / && "
            f"rm -f {shlex.quote(remote_tar)}"
        )
        result = self._box.exec("bash", "-c", cmd)
        if result.exit_code != 0:
            raise RuntimeError(
                f"boxd bulk upload extract failed (exit {result.exit_code}): "
                f"{result.stderr or result.stdout}"
            )

    def _boxd_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        remote_tar = f"/tmp/.hermes-bulk-dl.{os.getpid()}.tar"
        result = self._box.exec(
            "bash", "-c",
            f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}",
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"boxd bulk download tar failed (exit {result.exit_code}): "
                f"{result.stderr or result.stdout}"
            )
        data = self._box.read_file(remote_tar)
        try:
            self._box.exec("bash", "-c", f"rm -f {shlex.quote(remote_tar)}")
        except Exception:
            pass  # best-effort cleanup
        dest.write_bytes(data)

    def _boxd_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via exec."""
        self._box.exec("bash", "-c", quoted_rm_command(remote_paths))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _wake_box(self, box) -> None:
        """Resume a suspended VM. No-op (best effort) if already running."""
        status = (getattr(box, "status", "") or "").lower()
        if status in ("running", "started", ""):
            # "" (Box from get() may not populate status uniformly) — try
            # a no-op exec to surface real state. Skip the resume RPC.
            return
        try:
            box.resume()
            logger.info("boxd: resumed VM %s", box.id)
        except Exception as e:
            # Some servers no-op resume on already-running VMs and return
            # a benign error; others reject. Either way, swallow — the
            # next exec will surface a real failure if the VM is broken.
            logger.debug("boxd: resume on %s returned %s — continuing", box.id, e)

    def _ensure_box_ready(self) -> None:
        """Wake the VM if it suspended between commands."""
        status = (getattr(self._box, "status", "") or "").lower()
        if status == "suspended":
            self._wake_box(self._box)

    def _before_execute(self) -> None:
        with self._lock:
            self._ensure_box_ready()
        self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping a blocking boxd exec."""
        box = self._box
        lock = self._lock

        def cancel():
            with lock:
                try:
                    # suspend() halts every running process inside the VM
                    # while preserving disk state — exactly what we want
                    # for an interrupted command.
                    box.suspend()
                except Exception:
                    pass

        if login:
            shell_args = ("bash", "-l", "-c", cmd_string)
        else:
            shell_args = ("bash", "-c", cmd_string)

        def exec_fn() -> tuple[str, int]:
            result = box.exec(*shell_args, timeout=timeout)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            if stderr:
                output = f"{stdout}\n{stderr}" if stdout else stderr
            else:
                output = stdout
            return (output, result.exit_code)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            if self._box is None:
                return

            if self._sync_manager:
                logger.info("boxd: syncing files from VM...")
                try:
                    self._sync_manager.sync_back()
                except Exception as e:
                    logger.warning("boxd: sync_back failed: %s", e)

            try:
                if self._persistent:
                    self._box.suspend()
                    logger.info("boxd: suspended VM %s (filesystem + memory preserved)",
                                self._box.id)
                else:
                    self._box.destroy()
                    logger.info("boxd: destroyed VM %s", self._box.id)
            except Exception as e:
                logger.warning("boxd: cleanup failed: %s", e)
            self._box = None

            # Close the SDK transport only if we created it. When the
            # caller injected one, they're responsible for its lifecycle.
            if self._owns_compute:
                try:
                    self._compute.close()
                except Exception:
                    pass
