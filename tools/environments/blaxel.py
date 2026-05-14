"""Blaxel cloud sandbox execution environment.

Uses the Blaxel Python SDK (``blaxel.core.SyncSandboxInstance``) to run
commands in cloud sandboxes. Supports persistent sandboxes: when enabled,
a task-scoped Blaxel volume is created (or reused)
and mounted at ``/blaxel/persistent`` on the sandbox. The volume survives
sandbox deletion / TTL expiry, so the agent's working files remain
durable across sessions even if the sandbox itself goes away.
"""

import logging
import os
import hashlib
import re
import shlex
import threading
import time
import uuid
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


# Blaxel platform caps a single blocking ``process.exec`` at 60s. Anything
# beyond that is run with ``wait_for_completion=False`` and polled via
# ``process.get()`` / ``process.logs()`` so longer Hermes timeouts work.
_BLAXEL_BLOCKING_EXEC_CAP_SECONDS = 50
_BLAXEL_POLL_INTERVAL_SECONDS = 1.0

# Mount point inside the sandbox where the persistent Blaxel volume lives.
# When ``persistent_filesystem=True``, this directory is durable across
# sandbox recreation; everything else in the sandbox is ephemeral.
_BLAXEL_VOLUME_MOUNT_PATH = "/blaxel/persistent"
_BLAXEL_RESOURCE_MAX_LENGTH = 40
_BLAXEL_LABEL_MAX_LENGTH = 63
_BLAXEL_HASH_LENGTH = 8


def _safe_blaxel_component(value: str, max_length: int) -> str:
    """Return a DNS-label-safe, deterministic Blaxel name component."""
    raw = str(value or "default")
    slug = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug) or "default"
    if len(slug) <= max_length:
        return slug

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_BLAXEL_HASH_LENGTH]
    keep = max(1, max_length - _BLAXEL_HASH_LENGTH - 1)
    prefix = slug[:keep].rstrip("-") or slug[:keep]
    return f"{prefix}-{digest}"


def _blaxel_resource_name(prefix: str, task_id: str, suffix: str = "") -> str:
    """Build a Blaxel resource name capped for downstream platform labels."""
    component_max = _BLAXEL_RESOURCE_MAX_LENGTH - len(prefix) - len(suffix) - 1
    if component_max < _BLAXEL_HASH_LENGTH + 2:
        raise ValueError("Blaxel resource prefix/suffix leaves no room for task id")
    component = _safe_blaxel_component(task_id, component_max)
    return f"{prefix}-{component}{suffix}"


def _blaxel_label_value(task_id: str) -> str:
    return _safe_blaxel_component(task_id, _BLAXEL_LABEL_MAX_LENGTH)


class BlaxelEnvironment(BaseEnvironment):
    """Blaxel cloud sandbox execution backend.

    Spawn-per-call via _ThreadedProcessHandle wrapping blocking SDK calls.
    cancel_fn wired to ``process.kill(name)`` (or sandbox.delete() as a last
    resort) for interrupt support.
    """

    _stdin_mode = "heredoc"

    def __init__(
        self,
        image: str,
        cwd: str = "/blaxel",
        timeout: int = 60,
        cpu: int = 1,
        memory: int = 4096,
        disk: int = 10240,
        persistent_filesystem: bool = True,
        task_id: str = "default",
        ttl: str = "24h",
    ):
        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        # Lazy import: keep blaxel SDK optional until backend is selected.
        from blaxel.core import SyncSandboxInstance
        try:
            from blaxel.core.sandbox import SandboxAPIError
        except Exception:
            SandboxAPIError = Exception
        try:
            from blaxel.core import SyncVolumeInstance
            from blaxel.core.volume import VolumeAPIError
        except Exception:
            SyncVolumeInstance = None
            VolumeAPIError = Exception

        self._SyncSandboxInstance = SyncSandboxInstance
        self._SandboxAPIError = SandboxAPIError
        self._SyncVolumeInstance = SyncVolumeInstance
        self._VolumeAPIError = VolumeAPIError
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._lock = threading.Lock()
        self._sandbox = None
        self._volume_name: str | None = None

        sandbox_name = _blaxel_resource_name("hermes", task_id)
        self._sandbox_name = sandbox_name
        task_label = _blaxel_label_value(task_id)

        bl_region = os.getenv("BL_REGION", "").strip() or "us-pdx-1"

        # Blaxel sandbox config — see blaxel.core.SandboxCreateConfiguration.
        # Resources: memory in MB. CPU/disk are not first-class on Blaxel
        # sandboxes today (the platform allocates them per image profile),
        # so we pass only ``memory`` and surface the others in logs.
        # Region is read by the SDK from the BL_REGION env var; we don't
        # surface it as a Hermes-level knob.
        sandbox_config: dict[str, object] = {
            "name": sandbox_name,
            "image": image,
            "memory": int(memory),
            "ttl": ttl,
            "region": bl_region,
            "labels": {"hermes_task_id": task_label},
        }

        if cpu and int(cpu) != 1:
            logger.info("Blaxel: ignoring cpu=%s (allocated by image profile)", cpu)
        if disk and int(disk) != 10240:
            logger.info("Blaxel: ignoring disk=%s (allocated by image profile)", disk)

        # When persistent, ensure a Blaxel volume exists and mount it.
        # Blaxel sandboxes can disappear (TTL, platform churn); only volumes
        # are truly durable, so all real persistence lives there.
        if self._persistent:
            if SyncVolumeInstance is None:
                raise RuntimeError(
                    "Blaxel persistent filesystem requested, but the installed "
                    "blaxel SDK does not expose volume support. Set "
                    "terminal.container_persistent=false or install a newer SDK."
                )
            volume_name = _blaxel_resource_name("hermes", task_id, "-data")
            self._volume_name = volume_name
            volume_size_mb = max(1024, int(memory))
            try:
                SyncVolumeInstance.create_if_not_exists({
                    "name": volume_name,
                    "size": volume_size_mb,
                    "region": bl_region,
                    "labels": {"hermes_task_id": task_label},
                })
                logger.info("Blaxel: ensured volume %s (%d MB, region=%s)",
                            volume_name, volume_size_mb, bl_region)
                sandbox_config["volumes"] = [{
                    "name": volume_name,
                    "mount_path": _BLAXEL_VOLUME_MOUNT_PATH,
                    "read_only": False,
                }]
            except Exception as e:
                logger.warning(
                    "Blaxel: could not ensure volume %s (%s)",
                    volume_name, e,
                )
                self._volume_name = None
                raise RuntimeError(
                    "Blaxel persistent filesystem requested, but volume "
                    f"{volume_name!r} could not be created or reused: {e}. "
                    "Set terminal.container_persistent=false to use an "
                    "ephemeral sandbox."
                ) from e

        self._sandbox = self._create_or_recreate_sandbox(sandbox_config)
        logger.info(
            "Blaxel: sandbox %s ready for task %s (volume %s)",
            sandbox_name, task_id, self._volume_name or "<none>",
        )

        self._wait_for_sandbox_ready()

        # When the persistent volume is mounted, point cwd at it so the
        # agent's working files end up on durable storage. Otherwise fall
        # back to $HOME.
        self._remote_home = "/root"
        try:
            response = self._sandbox.process.exec({
                "command": "echo $HOME",
                "wait_for_completion": True,
                "timeout": 10_000,
            })
            home = (
                getattr(response, "stdout", None)
                or getattr(response, "logs", None)
                or ""
            ).strip()
            if home:
                self._remote_home = home
        except Exception:
            pass

        if requested_cwd in ("~", "/blaxel"):
            if self._volume_name:
                self.cwd = _BLAXEL_VOLUME_MOUNT_PATH
                # Make sure the mount point exists and is the cwd target.
                try:
                    self._sandbox.process.exec({
                        "command": f"mkdir -p {_BLAXEL_VOLUME_MOUNT_PATH}",
                        "wait_for_completion": True,
                        "timeout": 10_000,
                    })
                except Exception:
                    pass
            else:
                self.cwd = self._remote_home
        logger.info("Blaxel: resolved home to %s, cwd to %s",
                    self._remote_home, self.cwd)

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._blaxel_upload,
            delete_fn=self._blaxel_delete,
            bulk_upload_fn=self._blaxel_bulk_upload,
            bulk_download_fn=self._blaxel_bulk_download,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    # ------------------------------------------------------------------
    # File sync transport
    # ------------------------------------------------------------------

    def _create_or_recreate_sandbox(self, sandbox_config: dict):
        """Create a sandbox by name; if the name conflicts with a stale
        (terminated) sandbox, delete it and create fresh.

        This replaces ``create_if_not_exists``, which can reattach to a
        sandbox the platform is already tearing down — leading to all
        subsequent ``process.exec`` calls returning ``WORKLOAD_UNAVAILABLE``.
        """
        SyncSandboxInstance = self._SyncSandboxInstance
        try:
            return SyncSandboxInstance.create(sandbox_config)
        except self._SandboxAPIError as e:
            status = getattr(e, "status_code", None)
            # 409 from the SDK or controlplane means the sandbox name is taken.
            # The HTTP error envelope sometimes embeds a 409 in the message
            # body even when ``status_code`` is None on the SDK error.
            is_conflict = status == 409 or "409" in str(e) or "ALREADY_EXISTS" in str(e)
            if not is_conflict:
                raise

        # Name conflict: probe the existing one, reuse if healthy, recreate
        # otherwise.
        existing = None
        try:
            existing = SyncSandboxInstance.get(sandbox_config["name"])
        except Exception:
            existing = None

        if existing is not None and self._sandbox_is_responsive(existing):
            logger.info(
                "Blaxel: reusing existing healthy sandbox %s",
                sandbox_config["name"],
            )
            return existing

        # Stale handle — delete it and try again. Best-effort delete.
        logger.info(
            "Blaxel: existing sandbox %s is unresponsive, recreating",
            sandbox_config["name"],
        )
        try:
            SyncSandboxInstance.delete(sandbox_config["name"])
        except Exception as e:
            logger.debug("Blaxel: stale sandbox delete failed: %s", e)
        # Wait briefly for the volume to detach before retrying.
        time.sleep(3.0)
        return SyncSandboxInstance.create(sandbox_config)

    def _sandbox_is_responsive(self, sandbox, max_wait_seconds: float = 15.0) -> bool:
        """Return True if a no-op ``process.exec`` returns within the budget."""
        delay = 0.5
        deadline = time.monotonic() + max_wait_seconds
        while time.monotonic() < deadline:
            try:
                sandbox.process.exec({
                    "command": ":",
                    "wait_for_completion": True,
                    "timeout": 5_000,
                })
                return True
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 4.0)
        return False

    def _wait_for_sandbox_ready(self, max_wait_seconds: float = 60.0) -> None:
        """Poll the sandbox until ``process.exec`` succeeds.

        Right after ``create``/``create_if_not_exists`` the sandbox can
        return ``WORKLOAD_UNAVAILABLE`` (HTTP 404) for a few seconds while
        the workload is still scheduling. The platform docs explicitly say
        "Retry with exponential backoff: 500ms → 30s" — so we do.
        """
        delay = 0.5
        deadline = time.monotonic() + max_wait_seconds
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._sandbox.process.exec({
                    "command": ":",  # no-op
                    "wait_for_completion": True,
                    "timeout": 5_000,
                })
                return
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
        raise RuntimeError(
            "Blaxel sandbox "
            f"{self._sandbox_name} did not become ready within "
            f"{max_wait_seconds:.0f}s (last error: {last_err})"
        )

    def _blaxel_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via Blaxel SDK (auto-chunks files >5MB)."""
        parent = str(Path(remote_path).parent)
        self._sandbox.process.exec({
            "command": f"mkdir -p {shlex.quote(parent)}",
            "wait_for_completion": True,
            "timeout": 10_000,
        })
        with open(host_path, "rb") as f:
            self._sandbox.fs.write_binary(remote_path, f.read())

    def _blaxel_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files. Blaxel SDK has no batched upload endpoint,
        so we mkdir parents in one shell call and then write files one by
        one — the SDK still parallelizes large-file part uploads internally.
        """
        if not files:
            return

        parents = unique_parent_dirs(files)
        if parents:
            self._sandbox.process.exec({
                "command": quoted_mkdir_command(parents),
                "wait_for_completion": True,
                "timeout": 30_000,
            })

        for host_path, remote_path in files:
            with open(host_path, "rb") as f:
                self._sandbox.fs.write_binary(remote_path, f.read())

    def _blaxel_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        self._sandbox.process.exec({
            "command": (
                f"tar cf {shlex.quote(remote_tar)} -C / "
                f"{shlex.quote(rel_base)}"
            ),
            "wait_for_completion": True,
            "timeout": 60_000,
        })
        data = self._sandbox.fs.read_binary(remote_tar)
        with open(dest, "wb") as f:
            f.write(data)
        try:
            self._sandbox.process.exec({
                "command": f"rm -f {shlex.quote(remote_tar)}",
                "wait_for_completion": True,
                "timeout": 10_000,
            })
        except Exception:
            pass

    def _blaxel_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via SDK exec."""
        self._sandbox.process.exec({
            "command": quoted_rm_command(remote_paths),
            "wait_for_completion": True,
            "timeout": 30_000,
        })

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def _before_execute(self) -> None:
        """Sync files before each command. Blaxel sandboxes auto-resume
        from standby on the first request, so we don't need an explicit
        start/refresh step here."""
        self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping a blocking Blaxel SDK call."""
        sandbox = self._sandbox

        if login:
            shell_cmd = f"bash -l -c {shlex.quote(cmd_string)}"
        else:
            shell_cmd = f"bash -c {shlex.quote(cmd_string)}"

        process_name = f"hermes-{uuid.uuid4().hex[:12]}"

        def cancel():
            try:
                sandbox.process.kill(process_name)
            except Exception:
                pass

        def exec_fn() -> tuple[str, int]:
            return self._exec_with_timeout(shell_cmd, process_name, timeout)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def _exec_with_timeout(
        self, shell_cmd: str, process_name: str, timeout: int,
    ) -> tuple[str, int]:
        """Run *shell_cmd* with a Hermes-style timeout.

        For short timeouts (<= 50s) we use the SDK's blocking ``wait_for_completion``
        path. For longer timeouts we run async and poll, since Blaxel caps a
        single blocking exec at 60s.
        """
        sandbox = self._sandbox
        if timeout <= _BLAXEL_BLOCKING_EXEC_CAP_SECONDS:
            response = sandbox.process.exec({
                "command": shell_cmd,
                "name": process_name,
                "wait_for_completion": True,
                "timeout": int(timeout * 1000),
            })
            return self._collect_output(response, process_name)

        sandbox.process.exec({
            "command": shell_cmd,
            "name": process_name,
            "wait_for_completion": False,
        })

        deadline = time.monotonic() + timeout
        last_status = None
        while time.monotonic() < deadline:
            try:
                proc_info = sandbox.process.get(process_name)
            except Exception as e:
                logger.debug("Blaxel: process.get(%s) failed: %s",
                             process_name, e)
                time.sleep(_BLAXEL_POLL_INTERVAL_SECONDS)
                continue
            last_status = getattr(proc_info, "status", None)
            if last_status and str(last_status) not in ("running", "ProcessResponseStatus.RUNNING"):
                exit_code = getattr(proc_info, "exit_code", None)
                if exit_code is None:
                    exit_code = 0
                return self._collect_output(proc_info, process_name, exit_code=exit_code)
            time.sleep(_BLAXEL_POLL_INTERVAL_SECONDS)

        try:
            sandbox.process.kill(process_name)
        except Exception:
            pass
        return ("", 124)

    def _collect_output(self, response, process_name: str,
                        exit_code: int | None = None) -> tuple[str, int]:
        """Pull stdout/stderr from a ProcessResponse, falling back to
        ``process.logs()`` when the response object only has a name."""
        stdout = getattr(response, "stdout", "") or ""
        stderr = getattr(response, "stderr", "") or ""
        logs = getattr(response, "logs", "") or ""
        rc = exit_code if exit_code is not None else (
            getattr(response, "exit_code", None) or 0
        )

        output = ""
        if stdout or stderr:
            output = stdout
            if stderr:
                output = (output + "\n" + stderr) if output else stderr
        elif logs:
            output = logs
        else:
            try:
                output = self._sandbox.process.logs(process_name, "all") or ""
            except Exception:
                output = ""
        return (output, int(rc))

    def cleanup(self):
        with self._lock:
            if self._sandbox is None:
                return

            if self._sync_manager:
                logger.info("Blaxel: syncing files from sandbox...")
                try:
                    self._sync_manager.sync_back()
                except Exception as e:
                    logger.warning("Blaxel: sync_back failed: %s", e)

            try:
                if self._persistent:
                    self._sandbox.delete()
                    logger.info(
                        "Blaxel: deleted sandbox %s%s",
                        self._sandbox_name,
                        f" (volume {self._volume_name} preserved)"
                        if self._volume_name else
                        " (no persistent volume mounted)",
                    )
                else:
                    self._sandbox.delete()
                    logger.info("Blaxel: deleted sandbox %s", self._sandbox_name)
            except self._SandboxAPIError as e:
                if getattr(e, "status_code", None) == 404:
                    logger.info("Blaxel: sandbox %s already gone", self._sandbox_name)
                else:
                    logger.warning("Blaxel: cleanup failed: %s", e)
            except Exception as e:
                logger.warning("Blaxel: cleanup failed: %s", e)
            self._sandbox = None
