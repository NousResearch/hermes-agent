"""Modal cloud execution environment using the native Modal SDK directly.

Uses ``Sandbox.create()`` + ``Sandbox.exec()`` instead of the older runtime
wrapper, while preserving Hermes' persistent snapshot behavior across sessions.
"""

import asyncio
import json
import logging
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from tools.environments.modal_common import (
    BaseModalExecutionEnvironment,
    ModalExecStart,
    PreparedModalExec,
)

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = get_hermes_home() / "modal_snapshots.json"
_DIRECT_SNAPSHOT_NAMESPACE = "direct"


def _load_snapshots() -> Dict[str, str]:
    """Load snapshot ID mapping from disk."""
    if _SNAPSHOT_STORE.exists():
        try:
            return json.loads(_SNAPSHOT_STORE.read_text())
        except Exception:
            pass
    return {}


def _save_snapshots(data: Dict[str, str]) -> None:
    """Persist snapshot ID mapping to disk."""
    _SNAPSHOT_STORE.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_STORE.write_text(json.dumps(data, indent=2))


def _direct_snapshot_key(task_id: str) -> str:
    return f"{_DIRECT_SNAPSHOT_NAMESPACE}:{task_id}"


def _get_snapshot_restore_candidate(task_id: str) -> tuple[str | None, bool]:
    """Return a snapshot id and whether it came from the legacy key format."""
    snapshots = _load_snapshots()

    namespaced_key = _direct_snapshot_key(task_id)
    snapshot_id = snapshots.get(namespaced_key)
    if isinstance(snapshot_id, str) and snapshot_id:
        return snapshot_id, False

    legacy_snapshot_id = snapshots.get(task_id)
    if isinstance(legacy_snapshot_id, str) and legacy_snapshot_id:
        return legacy_snapshot_id, True

    return None, False


def _store_direct_snapshot(task_id: str, snapshot_id: str) -> None:
    """Persist the direct Modal snapshot id under the direct namespace."""
    snapshots = _load_snapshots()
    snapshots[_direct_snapshot_key(task_id)] = snapshot_id
    snapshots.pop(task_id, None)
    _save_snapshots(snapshots)


def _delete_direct_snapshot(task_id: str, snapshot_id: str | None = None) -> None:
    """Remove direct Modal snapshot entries for a task, including legacy keys."""
    snapshots = _load_snapshots()
    updated = False

    for key in (_direct_snapshot_key(task_id), task_id):
        value = snapshots.get(key)
        if value is None:
            continue
        if snapshot_id is None or value == snapshot_id:
            snapshots.pop(key, None)
            updated = True

    if updated:
        _save_snapshots(snapshots)


def _resolve_modal_image(image_spec: Any) -> Any:
    """Convert registry references or snapshot ids into Modal image objects."""
    import modal as _modal

    if not isinstance(image_spec, str):
        return image_spec

    if image_spec.startswith("im-"):
        return _modal.Image.from_id(image_spec)

    return _modal.Image.from_registry(
        image_spec,
        setup_dockerfile_commands=[
            "RUN rm -rf /usr/local/lib/python*/site-packages/pip* 2>/dev/null; "
            "python -m ensurepip --upgrade --default-pip 2>/dev/null || true",
        ],
    )


class _AsyncWorker:
    """Background thread with its own event loop for async-safe Modal calls."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=30)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run_coroutine(self, coro, timeout=600):
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("AsyncWorker loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10)


@dataclass
class _DirectModalExecHandle:
    thread: threading.Thread
    result_holder: Dict[str, Any]


class ModalEnvironment(BaseModalExecutionEnvironment):
    """Modal cloud execution via native Modal sandboxes."""

    _stdin_mode = "heredoc"
    _poll_interval_seconds = 0.2
    _interrupt_output = "[Command interrupted - Modal sandbox terminated]"
    _unexpected_error_prefix = "Modal execution error"

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        modal_sandbox_kwargs: Optional[Dict[str, Any]] = None,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._base_image = image
        self._sandbox = None
        self._app = None
        self._worker = _AsyncWorker()
        self._synced_files: Dict[str, tuple] = {}

        sandbox_kwargs = dict(modal_sandbox_kwargs or {})

        restored_snapshot_id = None
        restored_from_legacy_key = False
        if self._persistent:
            restored_snapshot_id, restored_from_legacy_key = _get_snapshot_restore_candidate(
                self._task_id
            )
            if restored_snapshot_id:
                logger.info("Modal: restoring from snapshot %s", restored_snapshot_id[:20])

        import modal as _modal

        cred_mounts = []
        try:
            from tools.credential_files import get_credential_file_mounts, iter_skills_files

            for mount_entry in get_credential_file_mounts():
                cred_mounts.append(
                    _modal.Mount.from_local_file(
                        mount_entry["host_path"],
                        remote_path=mount_entry["container_path"],
                    )
                )
                logger.info(
                    "Modal: mounting credential %s -> %s",
                    mount_entry["host_path"],
                    mount_entry["container_path"],
                )

            # Mount individual skill files (symlinks filtered out).
            skills_files = iter_skills_files()
            for entry in skills_files:
                cred_mounts.append(
                    _modal.Mount.from_local_file(
                        entry["host_path"],
                        remote_path=entry["container_path"],
                    )
                )
            if skills_files:
                logger.info("Modal: mounting %d skill files", len(skills_files))
        except Exception as e:
            logger.debug("Modal: could not load credential file mounts: %s", e)

        self._worker.start()

        async def _create_sandbox(image_spec: Any):
            app = await _modal.App.lookup.aio("hermes-agent", create_if_missing=True)
            create_kwargs = dict(sandbox_kwargs)
            if cred_mounts:
                existing_mounts = list(create_kwargs.pop("mounts", []))
                existing_mounts.extend(cred_mounts)
                create_kwargs["mounts"] = existing_mounts
            sandbox = await _modal.Sandbox.create.aio(
                "sleep",
                "infinity",
                image=image_spec,
                app=app,
                timeout=int(create_kwargs.pop("timeout", 3600)),
                **create_kwargs,
            )
            return app, sandbox

        try:
            target_image_spec = restored_snapshot_id or image
            try:
                # _resolve_modal_image keeps the Modal bootstrap fix together:
                # it applies setup_dockerfile_commands with ensurepip before
                # Modal builds registry images, while snapshot ids restore via
                # modal.Image.from_id() without rebuilding.
                effective_image = _resolve_modal_image(target_image_spec)
                self._app, self._sandbox = self._worker.run_coroutine(
                    _create_sandbox(effective_image),
                    timeout=300,
                )
            except Exception as exc:
                if not restored_snapshot_id:
                    raise

                logger.warning(
                    "Modal: failed to restore snapshot %s, retrying with base image: %s",
                    restored_snapshot_id[:20],
                    exc,
                )
                _delete_direct_snapshot(self._task_id, restored_snapshot_id)
                base_image = _resolve_modal_image(image)
                self._app, self._sandbox = self._worker.run_coroutine(
                    _create_sandbox(base_image),
                    timeout=300,
                )
            else:
                if restored_snapshot_id and restored_from_legacy_key:
                    _store_direct_snapshot(self._task_id, restored_snapshot_id)
                    logger.info(
                        "Modal: migrated legacy snapshot entry for task %s",
                        self._task_id,
                    )
        except Exception:
            self._worker.stop()
            raise

        logger.info("Modal: sandbox created (task=%s)", self._task_id)

    def _push_file_to_sandbox(self, host_path: str, container_path: str) -> bool:
        """Push a single file into the sandbox if changed. Returns True if synced."""
        hp = Path(host_path)
        try:
            stat = hp.stat()
            file_key = (stat.st_mtime, stat.st_size)
        except OSError:
            return False

        if self._synced_files.get(container_path) == file_key:
            return False

        try:
            content = hp.read_bytes()
        except Exception:
            return False

        import base64
        b64 = base64.b64encode(content).decode("ascii")
        container_dir = str(Path(container_path).parent)
        cmd = (
            f"mkdir -p {shlex.quote(container_dir)} && "
            f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(container_path)}"
        )

        async def _write():
            proc = await self._sandbox.exec.aio("bash", "-c", cmd)
            await proc.wait.aio()

        self._worker.run_coroutine(_write(), timeout=15)
        self._synced_files[container_path] = file_key
        return True

    def _sync_files(self) -> None:
        """Push credential files and skill files into the running sandbox.

        Runs before each command. Uses mtime+size caching so only changed
        files are pushed (~13μs overhead in the no-op case).
        """
        try:
            from tools.credential_files import get_credential_file_mounts, iter_skills_files

            for entry in get_credential_file_mounts():
                if self._push_file_to_sandbox(entry["host_path"], entry["container_path"]):
                    logger.debug("Modal: synced credential %s", entry["container_path"])

            for entry in iter_skills_files():
                if self._push_file_to_sandbox(entry["host_path"], entry["container_path"]):
                    logger.debug("Modal: synced skill file %s", entry["container_path"])
        except Exception as e:
            logger.debug("Modal: file sync failed: %s", e)

    def _before_execute(self) -> None:
        self._sync_files()

    def _start_modal_exec(self, prepared: PreparedModalExec) -> ModalExecStart:
        full_command = f"cd {shlex.quote(prepared.cwd)} && {prepared.command}"
        result_holder = {"value": None, "error": None}

        def _run():
            try:
                async def _do_execute():
                    process = await self._sandbox.exec.aio(
                        "bash",
                        "-c",
                        full_command,
                        timeout=prepared.timeout,
                    )
                    stdout = await process.stdout.read.aio()
                    stderr = await process.stderr.read.aio()
                    exit_code = await process.wait.aio()
                    if isinstance(stdout, bytes):
                        stdout = stdout.decode("utf-8", errors="replace")
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode("utf-8", errors="replace")
                    output = stdout
                    if stderr:
                        output = f"{stdout}\n{stderr}" if stdout else stderr
                    return self._result(output, exit_code)

                result_holder["value"] = self._worker.run_coroutine(
                    _do_execute(),
                    timeout=prepared.timeout + 30,
                )
            except Exception as e:
                result_holder["error"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return ModalExecStart(handle=_DirectModalExecHandle(thread=t, result_holder=result_holder))

    def _poll_modal_exec(self, handle: _DirectModalExecHandle) -> dict | None:
        if handle.thread.is_alive():
            return None
        if handle.result_holder["error"]:
            return self._error_result(f"Modal execution error: {handle.result_holder['error']}")
        return handle.result_holder["value"]

    def _cancel_modal_exec(self, handle: _DirectModalExecHandle) -> None:
        self._worker.run_coroutine(
            self._sandbox.terminate.aio(),
            timeout=15,
        )

    def cleanup(self):
        """Snapshot the filesystem (if persistent) then stop the sandbox."""
        if self._sandbox is None:
            return

        if self._persistent:
            try:
                async def _snapshot():
                    img = await self._sandbox.snapshot_filesystem.aio()
                    return img.object_id

                try:
                    snapshot_id = self._worker.run_coroutine(_snapshot(), timeout=60)
                except Exception:
                    snapshot_id = None

                if snapshot_id:
                    _store_direct_snapshot(self._task_id, snapshot_id)
                    logger.info(
                        "Modal: saved filesystem snapshot %s for task %s",
                        snapshot_id[:20],
                        self._task_id,
                    )
            except Exception as e:
                logger.warning("Modal: filesystem snapshot failed: %s", e)

        try:
            self._worker.run_coroutine(
                self._sandbox.terminate.aio(),
                timeout=15,
            )
        except Exception:
            pass
        finally:
            self._worker.stop()
            self._sandbox = None
            self._app = None
