"""Modal cloud execution environment using SWE-ReX directly.

Supports persistent filesystem snapshots: when enabled, the sandbox's filesystem
is snapshotted on cleanup and restored on next creation, so installed packages,
project files, and config changes survive across sessions.
"""

import asyncio
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli.config import get_hermes_home
from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

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
    """Return a snapshot id for direct Modal restore and whether the key is legacy."""
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
    """Background thread with its own event loop for async-safe swe-rex calls."""

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


class ModalEnvironment(BaseEnvironment):
    """Modal cloud execution via SWE-ReX.

    Uses swe-rex's ModalDeployment directly for sandbox management.
    Adds sudo -S support, configurable resources (CPU, memory, disk),
    and optional filesystem persistence via Modal's snapshot API.
    """

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
        self._deployment = None
        self._worker = _AsyncWorker()

        sandbox_kwargs = dict(modal_sandbox_kwargs or {})

        restored_snapshot_id = None
        restored_from_legacy_key = False
        if self._persistent:
            restored_snapshot_id, restored_from_legacy_key = _get_snapshot_restore_candidate(self._task_id)
            if restored_snapshot_id:
                logger.info("Modal: restoring from snapshot %s", restored_snapshot_id[:20])

        self._worker.start()

        from swerex.deployment.modal import ModalDeployment

        async def _create_and_start(image_spec: Any):
            deployment = ModalDeployment(
                image=image_spec,
                startup_timeout=180.0,
                runtime_timeout=3600.0,
                deployment_timeout=3600.0,
                install_pipx=True,
                modal_sandbox_kwargs=sandbox_kwargs,
            )
            await deployment.start()
            return deployment

        try:
            target_image_spec = restored_snapshot_id or image
            try:
                effective_image = _resolve_modal_image(target_image_spec)
                self._deployment = self._worker.run_coroutine(_create_and_start(effective_image))
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
                self._deployment = self._worker.run_coroutine(_create_and_start(base_image))
            else:
                if restored_snapshot_id and restored_from_legacy_key:
                    _store_direct_snapshot(self._task_id, restored_snapshot_id)
                    logger.info("Modal: migrated legacy snapshot entry for task %s", self._task_id)
        except Exception:
            self._worker.stop()
            raise

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            command = f"{command} << '{marker}'\n{stdin_data}\n{marker}"

        exec_command, sudo_stdin = self._prepare_command(command)

        # Modal sandboxes execute commands via the Modal SDK and cannot pipe
        # subprocess stdin directly the way a local Popen can. When a sudo
        # password is present, use a shell-level pipe from printf so that the
        # password feeds sudo -S without appearing as an echo argument embedded
        # in the shell string.
        if sudo_stdin is not None:
            import shlex
            exec_command = (
                f"printf '%s\\n' {shlex.quote(sudo_stdin.rstrip())} | {exec_command}"
            )

        from swerex.runtime.abstract import Command as RexCommand

        effective_cwd = cwd or self.cwd
        effective_timeout = timeout or self.timeout

        result_holder = {"value": None, "error": None}

        def _run():
            try:
                async def _do_execute():
                    return await self._deployment.runtime.execute(
                        RexCommand(
                            command=exec_command,
                            shell=True,
                            check=False,
                            cwd=effective_cwd,
                            timeout=effective_timeout,
                            merge_output_streams=True,
                        )
                    )

                output = self._worker.run_coroutine(_do_execute())
                result_holder["value"] = {
                    "output": output.stdout,
                    "returncode": output.exit_code,
                }
            except Exception as e:
                result_holder["error"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.2)
            if is_interrupted():
                try:
                    self._worker.run_coroutine(
                        asyncio.wait_for(self._deployment.stop(), timeout=10),
                        timeout=15,
                    )
                except Exception:
                    pass
                return {
                    "output": "[Command interrupted - Modal sandbox terminated]",
                    "returncode": 130,
                }

        if result_holder["error"]:
            return {"output": f"Modal execution error: {result_holder['error']}", "returncode": 1}
        return result_holder["value"]

    def cleanup(self):
        """Snapshot the filesystem (if persistent) then stop the sandbox."""
        if self._deployment is None:
            return

        if self._persistent:
            try:
                sandbox = getattr(self._deployment, "_sandbox", None)
                if sandbox:
                    async def _snapshot():
                        img = await sandbox.snapshot_filesystem.aio()
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
                asyncio.wait_for(self._deployment.stop(), timeout=10),
                timeout=15,
            )
        except Exception:
            pass
        finally:
            self._worker.stop()
            self._deployment = None
