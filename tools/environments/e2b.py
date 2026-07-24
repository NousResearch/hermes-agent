"""E2B cloud execution environment.

Uses the E2B Python SDK (``e2b``) to run commands in secure cloud sandboxes
through Hermes' shared ``BaseEnvironment`` shell contract. When persistence
is enabled, the backend pauses the sandbox on cleanup and resumes it (via
``Sandbox.connect``) on the next session, preserving the filesystem across
reuses.
"""

from __future__ import annotations

import logging
import os
import shlex
import threading
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
    _load_json_store,
    _save_json_store,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)


DEFAULT_E2B_CWD = "/home/user"
# E2B's own SDK default. Matches ``SandboxBase.default_template`` — a minimal
# Ubuntu image with bash. Users can swap to a richer template via
# ``TERMINAL_E2B_TEMPLATE`` (e.g. ``code-interpreter-v1`` for Jupyter-ready
# sandboxes, or a custom template built with ``e2b template build``).
DEFAULT_E2B_TEMPLATE = "base"
KNOWN_E2B_TEMPLATES = (
    "base",
    "code-interpreter-v1",
    "desktop",
)
_SANDBOX_STORE_NAME = "e2b_sandboxes.json"
_DEFAULT_SANDBOX_TIMEOUT_SECONDS = 300


def _sandbox_store_path() -> Path:
    return get_hermes_home() / _SANDBOX_STORE_NAME


def _load_sandbox_store() -> dict:
    return _load_json_store(_sandbox_store_path())


def _save_sandbox_store(data: dict) -> None:
    _save_json_store(_sandbox_store_path(), data)


def _get_saved_sandbox_id(task_id: str) -> str | None:
    if not task_id:
        return None
    value = _load_sandbox_store().get(task_id)
    return value if isinstance(value, str) and value else None


def _store_sandbox_id(task_id: str, sandbox_id: str) -> None:
    if not task_id or not sandbox_id:
        return
    store = _load_sandbox_store()
    store[task_id] = sandbox_id
    _save_sandbox_store(store)


def _delete_sandbox_id(task_id: str, sandbox_id: str | None = None) -> None:
    if not task_id:
        return
    store = _load_sandbox_store()
    existing = store.get(task_id)
    if existing is None:
        return
    if sandbox_id is not None and existing != sandbox_id:
        return
    store.pop(task_id, None)
    _save_sandbox_store(store)


def _extract_sandbox_id(sandbox: Any) -> str | None:
    for attr in ("sandbox_id", "sandboxId", "id"):
        value = getattr(sandbox, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


class E2BEnvironment(BaseEnvironment):
    """E2B cloud sandbox execution backend.

    Spawn-per-call via ``_ThreadedProcessHandle`` wrapping the blocking
    ``sandbox.commands.run`` SDK call. ``cancel_fn`` is wired to
    ``sandbox.kill`` for interrupt support.

    When ``persistent_filesystem=True`` the sandbox is paused on cleanup and
    reconnected on next construction (via the saved sandbox id under
    ``HERMES_HOME``), so files written during the previous session survive.
    """

    _stdin_mode = "heredoc"

    def __init__(
        self,
        template: str | None = None,
        cwd: str = DEFAULT_E2B_CWD,
        timeout: int = 60,
        cpu: int = 1,
        memory: int = 5120,
        disk: int = 10240,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        del cpu, memory, disk  # E2B resources are defined by the template, not per-run.

        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("terminal.e2b", prompt=False)
        except ImportError:
            pass
        except Exception as e:
            raise ImportError(str(e))
        from e2b import Sandbox

        self._Sandbox = Sandbox
        self._template = template or None
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._requested_cwd = requested_cwd
        self._lock = threading.Lock()
        self._sandbox: Any = None
        # E2B inactivity timeout. Hermes enforces per-command timeouts in
        # ``_wait_for_process`` via ``cancel_fn``; this is a floor so the
        # sandbox itself isn't reaped mid-command by the E2B platform.
        self._sandbox_timeout = max(int(timeout), _DEFAULT_SANDBOX_TIMEOUT_SECONDS)

        self._sandbox = self._create_or_resume_sandbox()
        self._remote_home = self._detect_remote_home(requested_cwd)

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._e2b_upload,
            delete_fn=self._e2b_delete,
            bulk_upload_fn=self._e2b_bulk_upload,
            bulk_download_fn=self._e2b_bulk_download,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def _create_fresh_sandbox(self) -> Any:
        kwargs: dict[str, Any] = {"timeout": self._sandbox_timeout}
        if self._template:
            kwargs["template"] = self._template
        sandbox = self._Sandbox.create(**kwargs)
        logger.info(
            "E2B: created sandbox %s for task %s",
            _extract_sandbox_id(sandbox) or "?",
            self._task_id,
        )
        return sandbox

    def _create_or_resume_sandbox(self) -> Any:
        saved = _get_saved_sandbox_id(self._task_id) if self._persistent else None
        if saved:
            try:
                sandbox = self._Sandbox.connect(saved)
                logger.info(
                    "E2B: resumed sandbox %s for task %s", saved, self._task_id
                )
                return sandbox
            except Exception as exc:
                logger.warning(
                    "E2B: failed to resume sandbox %s for task %s (%s); creating fresh",
                    saved,
                    self._task_id,
                    exc,
                )
                _delete_sandbox_id(self._task_id, saved)

        return self._create_fresh_sandbox()

    def _detect_remote_home(self, requested_cwd: str) -> str:
        """Resolve the remote ``$HOME``, falling back to the E2B default."""
        try:
            result = self._sandbox.commands.run('printf %s "$HOME"')
            home = (getattr(result, "stdout", "") or "").strip()
        except Exception as exc:
            logger.debug(
                "E2B: home detection failed for task %s: %s", self._task_id, exc
            )
            home = ""

        if not home.startswith("/"):
            home = DEFAULT_E2B_CWD

        # "~" and the wizard default both resolve to the detected home.
        if requested_cwd in ("~", "", DEFAULT_E2B_CWD):
            self.cwd = home
        return home

    def _ensure_sandbox_ready(self) -> None:
        """Reconnect (or recreate) if the sandbox was reaped between calls."""
        if self._sandbox is None:
            self._sandbox = self._create_or_resume_sandbox()
            return

        sandbox_id = _extract_sandbox_id(self._sandbox)
        try:
            is_running = self._sandbox.is_running()
        except AttributeError:
            # Older SDKs: fall back to a trivial probe.
            try:
                self._sandbox.commands.run("true", timeout=5)
                is_running = True
            except Exception:
                is_running = False
        except Exception:
            is_running = False

        if is_running:
            return

        if sandbox_id:
            try:
                self._sandbox = self._Sandbox.connect(sandbox_id)
                logger.info("E2B: reconnected sandbox %s", sandbox_id)
                return
            except Exception as exc:
                logger.warning(
                    "E2B: reconnect to %s failed (%s); recreating",
                    sandbox_id,
                    exc,
                )
                _delete_sandbox_id(self._task_id, sandbox_id)

        self._sandbox = self._create_fresh_sandbox()

    # ------------------------------------------------------------------
    # File sync adapters
    # ------------------------------------------------------------------

    def _e2b_upload(self, host_path: str, remote_path: str) -> None:
        parent = str(Path(remote_path).parent)
        self._sandbox.files.make_dir(parent)
        self._sandbox.files.write(remote_path, Path(host_path).read_bytes())

    def _e2b_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return

        parents = unique_parent_dirs(files)
        if parents:
            # E2B exposes make_dir per path; batch via a single mkdir -p.
            self._sandbox.commands.run(quoted_mkdir_command(parents))

        payload = [
            {"path": remote_path, "data": Path(host_path).read_bytes()}
            for host_path, remote_path in files
        ]
        # Prefer the bulk API when available; fall back to per-file writes.
        write_files = getattr(self._sandbox.files, "write_files", None)
        if callable(write_files):
            write_files(payload)
            return
        for entry in payload:
            self._sandbox.files.write(entry["path"], entry["data"])

    def _e2b_delete(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return
        self._sandbox.commands.run(quoted_rm_command(remote_paths))

    def _e2b_bulk_download(self, dest: Path) -> None:
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        result = self._sandbox.commands.run(
            f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}"
        )
        exit_code = getattr(result, "exit_code", 0)
        if exit_code not in (0, None):
            raise RuntimeError(
                f"E2B bulk download failed: {getattr(result, 'stderr', '') or getattr(result, 'stdout', '')}"
            )

        try:
            data = self._sandbox.files.read(remote_tar, format="bytes")
            if isinstance(data, str):
                data = data.encode("utf-8", errors="replace")
            Path(dest).write_bytes(bytes(data))
        finally:
            try:
                self._sandbox.commands.run(f"rm -f {shlex.quote(remote_tar)}")
            except Exception:
                pass  # best-effort cleanup

    # ------------------------------------------------------------------
    # BaseEnvironment contract
    # ------------------------------------------------------------------

    def _before_execute(self) -> None:
        with self._lock:
            self._ensure_sandbox_ready()
        self._sync_manager.sync()

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        """Run a bash command in the E2B sandbox.

        ``stdin_data`` is intentionally discarded: ``_stdin_mode="heredoc"``
        embeds any stdin payload into ``cmd_string`` before this runs.
        """
        del stdin_data

        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("E2B sandbox is not attached")
        shell_cmd = (
            f"bash -l -c {shlex.quote(cmd_string)}"
            if login
            else f"bash -c {shlex.quote(cmd_string)}"
        )
        lock = self._lock

        def cancel() -> None:
            with lock:
                try:
                    sandbox.kill()
                except Exception:
                    pass

        def exec_fn() -> tuple[str, int]:
            # E2B merges stdout+stderr into separate fields; join them to
            # match BaseEnvironment's single-stream output contract. Pass
            # a generous per-exec timeout so the Hermes-level poll loop,
            # not the SDK, enforces cancellation.
            result = sandbox.commands.run(shell_cmd, timeout=max(timeout, 1))
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            output = stdout if not stderr else (stdout + stderr)
            return output, int(getattr(result, "exit_code", 0) or 0)

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            sandbox = self._sandbox
            sync_manager = self._sync_manager
            if sandbox is not None and sync_manager is not None:
                try:
                    sync_manager.sync_back()
                except Exception as exc:
                    logger.warning(
                        "E2B: sync_back failed for task %s: %s", self._task_id, exc
                    )
            self._sandbox = None
            self._sync_manager = None

        if sandbox is None:
            return

        sandbox_id = _extract_sandbox_id(sandbox)
        if self._persistent and self._task_id and sandbox_id:
            try:
                sandbox.beta_pause()
                _store_sandbox_id(self._task_id, sandbox_id)
                logger.info(
                    "E2B: paused sandbox %s for task %s (filesystem preserved)",
                    sandbox_id,
                    self._task_id,
                )
                return
            except Exception as exc:
                logger.warning(
                    "E2B: pause failed for task %s: %s; falling back to kill",
                    self._task_id,
                    exc,
                )
                _delete_sandbox_id(self._task_id, sandbox_id)

        try:
            sandbox.kill()
            logger.info("E2B: killed sandbox %s", sandbox_id or "?")
        except Exception as exc:
            logger.warning("E2B: kill failed for task %s: %s", self._task_id, exc)
