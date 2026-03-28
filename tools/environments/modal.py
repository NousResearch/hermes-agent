"""Modal cloud execution environment using the Modal SDK directly.

Replaces the previous swe-rex ModalDeployment wrapper with native Modal
Sandbox.create() + Sandbox.exec() calls.  This eliminates the need for
swe-rex's HTTP runtime server and unencrypted tunnel, fixing:
  - AsyncUsageWarning from synchronous App.lookup in async context
  - DeprecationError from unencrypted_ports / .url on unencrypted tunnels

Supports persistent filesystem snapshots: when enabled, the sandbox's
filesystem is snapshotted on cleanup and restored on next creation, so
installed packages, project files, and config changes survive across sessions.
"""

import asyncio
import json
import logging
import shlex
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli.config import get_hermes_home
from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = get_hermes_home() / "modal_snapshots.json"


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


class _AsyncWorker:
    """Background thread with its own event loop for async-safe Modal calls.

    Allows sync code to submit async coroutines and block for results,
    even when called from inside another running event loop (e.g. Atropos).
    """

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
    """Modal cloud execution via native Modal SDK.

    Uses Modal's Sandbox.create() for container lifecycle and Sandbox.exec()
    for command execution — no intermediate HTTP server or tunnel required.
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
        self._sandbox = None
        self._app = None
        self._worker = _AsyncWorker()

        sandbox_kwargs = dict(modal_sandbox_kwargs or {})

        # If persistent, try to restore from a previous snapshot
        restored_image = None
        if self._persistent:
            snapshot_id = _load_snapshots().get(self._task_id)
            if snapshot_id:
                try:
                    import modal
                    restored_image = modal.Image.from_id(snapshot_id)
                    logger.info("Modal: restoring from snapshot %s", snapshot_id[:20])
                except Exception as e:
                    logger.warning("Modal: failed to restore snapshot, using base image: %s", e)
                    restored_image = None

        effective_image = restored_image if restored_image else image

        # Pre-build a modal.Image with pip fix for Modal's legacy image builder.
        # Some task images have broken pip; fix via ensurepip before Modal uses it.
        import modal as _modal
        if isinstance(effective_image, str):
            effective_image = _modal.Image.from_registry(
                effective_image,
                setup_dockerfile_commands=[
                    "RUN rm -rf /usr/local/lib/python*/site-packages/pip* 2>/dev/null; "
                    "python -m ensurepip --upgrade --default-pip 2>/dev/null || true",
                ],
            )

        # Start the async worker thread and create sandbox on it
        # so all gRPC channels are bound to the worker's event loop.
        self._worker.start()

        async def _create_sandbox():
            app = await _modal.App.lookup.aio(
                "hermes-agent", create_if_missing=True
            )
            sandbox = await _modal.Sandbox.create.aio(
                "sleep", "infinity",
                image=effective_image,
                app=app,
                timeout=int(sandbox_kwargs.pop("timeout", 3600)),
                **sandbox_kwargs,
            )
            return app, sandbox

        self._app, self._sandbox = self._worker.run_coroutine(
            _create_sandbox(), timeout=300
        )
        logger.info("Modal: sandbox created (task=%s)", self._task_id)

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            command = f"{command} << '{marker}'\n{stdin_data}\n{marker}"

        exec_command, sudo_stdin = self._prepare_command(command)

        # Modal sandboxes execute commands via exec() and cannot pipe
        # subprocess stdin directly.  When a sudo password is present,
        # use a shell-level pipe from printf.
        if sudo_stdin is not None:
            exec_command = (
                f"printf '%s\\n' {shlex.quote(sudo_stdin.rstrip())} | {exec_command}"
            )

        effective_cwd = cwd or self.cwd
        effective_timeout = timeout or self.timeout

        # Wrap command with cd + stderr merge
        full_command = f"cd {shlex.quote(effective_cwd)} && {exec_command}"

        # Run in a background thread so we can poll for interrupts
        result_holder = {"value": None, "error": None}

        def _run():
            try:
                async def _do_execute():
                    process = await self._sandbox.exec.aio(
                        "bash", "-c", full_command,
                        timeout=effective_timeout,
                    )
                    # Read stdout; redirect stderr to stdout in the shell
                    # command so we get merged output
                    stdout = await process.stdout.read.aio()
                    stderr = await process.stderr.read.aio()
                    exit_code = await process.wait.aio()
                    # Merge stdout + stderr (stderr after stdout)
                    output = stdout
                    if stderr:
                        output = f"{stdout}\n{stderr}" if stdout else stderr
                    return output, exit_code

                output, exit_code = self._worker.run_coroutine(
                    _do_execute(), timeout=effective_timeout + 30
                )
                result_holder["value"] = {
                    "output": output,
                    "returncode": exit_code,
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
                        self._sandbox.terminate.aio(),
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
                    snapshots = _load_snapshots()
                    snapshots[self._task_id] = snapshot_id
                    _save_snapshots(snapshots)
                    logger.info("Modal: saved filesystem snapshot %s for task %s",
                                snapshot_id[:20], self._task_id)
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
