"""Modal cloud execution environment wrapping mini-swe-agent's SwerexModalEnvironment.

Supports persistent filesystem snapshots: when enabled, the sandbox's filesystem
is snapshotted on cleanup and restored on next creation, so installed packages,
project files, and config changes survive across sessions.
"""

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = Path.home() / ".hermes" / "modal_snapshots.json"


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


class ModalEnvironment(BaseEnvironment):
    """Modal cloud execution via mini-swe-agent.

    Wraps SwerexModalEnvironment and adds sudo -S support, configurable
    resources (CPU, memory, disk), and optional filesystem persistence
    via Modal's snapshot_filesystem() API.
    """

    _patches_applied = False

    def __init__(
        self,
        image: str,
        cwd: str = "~",
        timeout: int = 60,
        modal_sandbox_kwargs: Optional[Dict[str, Any]] = None,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)

        if not ModalEnvironment._patches_applied:
            try:
                from environments.patches import apply_patches

                apply_patches()
            except ImportError:
                pass
            ModalEnvironment._patches_applied = True

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._base_image = image

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

        from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment

        self._inner = SwerexModalEnvironment(
            image=effective_image,
            cwd=cwd,
            timeout=timeout,
            startup_timeout=180.0,
            runtime_timeout=3600.0,
            modal_sandbox_kwargs=sandbox_kwargs,
        )

    def execute(
        self, command: str, cwd: str = "", *, timeout: int | None = None, stdin_data: str | None = None
    ) -> dict:
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            command = f"{command} << '{marker}'\n{stdin_data}\n{marker}"

        exec_command = self._prepare_command(command)

        # Run in a background thread so we can poll for interrupts
        result_holder = {"value": None, "error": None}

        def _run():
            try:
                result_holder["value"] = self._inner.execute(exec_command, cwd=cwd, timeout=timeout)
            except Exception as e:
                result_holder["error"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.2)
            if is_interrupted():
                try:
                    self._inner.stop()
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
        if self._persistent:
            try:
                sandbox = getattr(self._inner, "deployment", None)
                sandbox = getattr(sandbox, "_sandbox", None) if sandbox else None
                if sandbox:
                    import asyncio

                    async def _snapshot():
                        img = await sandbox.snapshot_filesystem.aio()
                        return img.object_id

                    try:
                        snapshot_id = asyncio.run(_snapshot())
                    except RuntimeError:
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            snapshot_id = pool.submit(asyncio.run, _snapshot()).result(timeout=60)

                    snapshots = _load_snapshots()
                    snapshots[self._task_id] = snapshot_id
                    _save_snapshots(snapshots)
                    logger.info("Modal: saved filesystem snapshot %s for task %s", snapshot_id[:20], self._task_id)
            except Exception as e:
                logger.warning("Modal: filesystem snapshot failed: %s", e)

        if hasattr(self._inner, "stop"):
            self._inner.stop()
