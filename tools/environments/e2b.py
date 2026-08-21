"""E2B cloud sandbox execution environment.

Hermes runs outside E2B and delegates terminal, file, and code execution to a
task-scoped sandbox. Persistent mode preserves only the filesystem: cleanup
pauses with ``keep_memory=False`` and later construction reconnects by the
saved sandbox ID. Ephemeral mode kills the sandbox on cleanup.
"""

from __future__ import annotations

import logging
import os
import posixpath
import shlex
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from e2b import SandboxLifecycle

from hermes_constants import get_hermes_home
from tools.environments.base import (
    BaseEnvironment,
    EnvironmentConnectionError,
    _ThreadedProcessHandle,
    _load_json_store,
    _save_json_store,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)

DEFAULT_E2B_CWD = "/home/user"
DEFAULT_E2B_TEMPLATE = "base"
_E2B_HERMES_HOME = f"{DEFAULT_E2B_CWD}/.hermes"
_SANDBOX_STORE_NAME = "e2b_sandboxes.json"
_COMMAND_TIMEOUT_GRACE_SECONDS = 5
_STORE_LOCK = threading.Lock()


def _ensure_e2b_sdk() -> None:
    """Lazy-install the supported E2B SDK on demand."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("terminal.e2b", prompt=False)
    except ImportError:
        pass
    except Exception as exc:
        raise ImportError(str(exc)) from exc


def _sandbox_store_path() -> Path:
    return get_hermes_home() / _SANDBOX_STORE_NAME


def _record_key(task_id: str, template: str) -> str:
    # Length-prefix the task ID so the key stays readable while remaining
    # unambiguous when either value contains separators.
    return f"{len(task_id)}:{task_id}{template}"


def _load_sandbox_record(task_id: str, template: str) -> dict[str, str] | None:
    if not task_id:
        return None
    with _STORE_LOCK:
        record = _load_json_store(_sandbox_store_path()).get(_record_key(task_id, template))
    if not isinstance(record, dict):
        return None
    sandbox_id = record.get("sandbox_id")
    stored_template = record.get("template")
    if not isinstance(sandbox_id, str) or not sandbox_id:
        return None
    if not isinstance(stored_template, str) or not stored_template:
        return None
    return {"sandbox_id": sandbox_id, "template": stored_template}


def _store_sandbox_record(task_id: str, sandbox_id: str, template: str) -> None:
    if not task_id or not sandbox_id:
        return
    with _STORE_LOCK:
        data = _load_json_store(_sandbox_store_path())
        data[_record_key(task_id, template)] = {
            "sandbox_id": sandbox_id,
            "template": template,
        }
        _save_json_store(_sandbox_store_path(), data)


def _delete_sandbox_record(
    task_id: str,
    template: str,
    sandbox_id: str | None = None,
) -> None:
    if not task_id:
        return
    with _STORE_LOCK:
        data = _load_json_store(_sandbox_store_path())
        key = _record_key(task_id, template)
        record = data.get(key)
        if record is None:
            return
        if sandbox_id is not None and (
            not isinstance(record, dict) or record.get("sandbox_id") != sandbox_id
        ):
            return
        data.pop(key, None)
        _save_json_store(_sandbox_store_path(), data)


def _sandbox_id(sandbox: Any) -> str:
    value = getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "id", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError("E2B create/connect did not return a sandbox ID")
    return value


def _connection_error(action: str, exc: BaseException) -> EnvironmentConnectionError:
    return EnvironmentConnectionError(
        f"E2B {action} failed: {exc}",
        retry_hint=(
            "Verify E2B connectivity and the E2B_API_KEY configured for the active "
            "Hermes profile, then retry. Existing persistent sandboxes are preserved "
            "when recovery is possible."
        ),
    )


class E2BEnvironment(BaseEnvironment):
    """E2B sandbox backend with filesystem-only persistence."""

    _stdin_mode = "heredoc"
    _profile_scoped_passthrough = True

    def __init__(
        self,
        *,
        api_key: str,
        template: str = DEFAULT_E2B_TEMPLATE,
        cwd: str = DEFAULT_E2B_CWD,
        timeout: int = 60,
        lifetime_seconds: int = 300,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                "E2B_API_KEY is required for the E2B terminal backend in the active profile"
            )

        requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)
        self._api_key = api_key
        self._template = template.strip() or DEFAULT_E2B_TEMPLATE
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._requested_cwd = requested_cwd
        self._sandbox_timeout = max(int(lifetime_seconds), int(timeout) + 5, 60)
        self._lock = threading.RLock()
        self._sandbox: Any | None = None
        self._sandbox_id: str | None = None
        self._created_for_initialization = False
        hermes_home = get_hermes_home()
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(_E2B_HERMES_HOME),
            upload_fn=self._e2b_upload,
            delete_fn=self._e2b_delete,
            bulk_upload_fn=self._e2b_bulk_upload,
            bulk_download_fn=self._e2b_bulk_download,
            sync_back_roots=[
                (str(hermes_home / "skills"), f"{_E2B_HERMES_HOME}/skills"),
                (str(hermes_home / "memories"), f"{_E2B_HERMES_HOME}/memories"),
            ],
        )

        try:
            self._sandbox = self._connect_or_create()
            self._sandbox_id = _sandbox_id(self._sandbox)
            if requested_cwd in {"", "~", "/root"}:
                self.cwd = DEFAULT_E2B_CWD
            self._initialize_remote_state()
            self.init_session()
        except Exception:
            self._abort_initialization()
            raise

    def _create_sandbox(self):
        _ensure_e2b_sdk()
        from e2b import Sandbox

        lifecycle: SandboxLifecycle = (
            {
                "on_timeout": {"action": "pause", "keep_memory": False},
                "auto_resume": False,
            }
            if self._persistent
            else {"on_timeout": "kill", "auto_resume": False}
        )
        try:
            sandbox = Sandbox.create(
                template=self._template,
                timeout=self._sandbox_timeout,
                lifecycle=lifecycle,
                api_key=self._api_key,
            )
        except Exception as exc:
            raise _connection_error("sandbox creation", exc) from exc

        sandbox_id = _sandbox_id(sandbox)
        self._created_for_initialization = True
        if self._persistent:
            _store_sandbox_record(self._task_id, sandbox_id, self._template)
        logger.info("E2B: created sandbox %s for task %s", sandbox_id, self._task_id)
        return sandbox

    def _connect_or_create(self):
        _ensure_e2b_sdk()
        from e2b import Sandbox
        from e2b.exceptions import SandboxNotFoundException

        self._created_for_initialization = False

        record = (
            _load_sandbox_record(self._task_id, self._template)
            if self._persistent
            else None
        )

        if record:
            sandbox_id = record["sandbox_id"]
            try:
                sandbox = Sandbox.connect(
                    sandbox_id,
                    timeout=self._sandbox_timeout,
                    api_key=self._api_key,
                )
                logger.info("E2B: resumed sandbox %s for task %s", sandbox_id, self._task_id)
                return sandbox
            except SandboxNotFoundException:
                logger.info(
                    "E2B: stored sandbox %s for task %s no longer exists; creating fresh",
                    sandbox_id,
                    self._task_id,
                )
                _delete_sandbox_record(self._task_id, self._template, sandbox_id)
            except Exception as exc:
                raise _connection_error(f"sandbox resume ({sandbox_id})", exc) from exc

        return self._create_sandbox()

    def _initialize_remote_state(self) -> None:
        """Create the state root and require one successful sync cycle."""
        self._ensure_remote_hermes_dir()
        try:
            sync_succeeded = self._sync_manager.sync(force=True)
        except Exception as exc:
            raise _connection_error("initial state sync", exc) from exc
        if not sync_succeeded:
            raise _connection_error(
                "initial state sync",
                RuntimeError("Hermes state could not be uploaded to the sandbox"),
            )

    def _initialize_replacement_sandbox(self) -> None:
        """Bootstrap a newly-created replacement before exposing it to tools."""
        self._sync_manager.reset_remote_state()
        try:
            self._initialize_remote_state()
            self._snapshot_ready = False
            self.init_session()
        except Exception:
            self._abort_initialization()
            raise

    def _abort_initialization(self) -> None:
        """Best-effort rollback for a sandbox that never became usable."""
        sandbox = self._sandbox
        sandbox_id = self._sandbox_id
        if sandbox is None:
            return

        try:
            if self._persistent and not self._created_for_initialization:
                # A resumed sandbox may contain state from an earlier session;
                # preserve its filesystem and pointer when this startup fails.
                sandbox.pause(keep_memory=False, api_key=self._api_key)
            else:
                killed = sandbox.kill(api_key=self._api_key)
                if killed is False:
                    logger.info(
                        "E2B: sandbox %s was already gone during initialization rollback",
                        sandbox_id,
                    )
                if self._persistent and sandbox_id:
                    _delete_sandbox_record(
                        self._task_id,
                        self._template,
                        sandbox_id,
                    )
        except Exception as exc:
            logger.warning(
                "E2B: failed to roll back sandbox %s after initialization error: %s",
                sandbox_id,
                exc,
            )
        finally:
            self._sandbox = None
            self._sandbox_id = None

    def _ensure_sandbox_ready(self, *, create_if_missing: bool = True) -> bool:
        from e2b.exceptions import SandboxNotFoundException

        if self._sandbox is None or not self._sandbox_id:
            if not create_if_missing:
                return False
            self._sandbox = self._connect_or_create()
            self._sandbox_id = _sandbox_id(self._sandbox)
            if self._created_for_initialization:
                self._initialize_replacement_sandbox()
            return True

        try:
            # Explicitly reconnect so an E2B lifecycle timeout that paused the
            # sandbox is handled without enabling cross-profile auto-resume.
            self._sandbox.connect(
                timeout=self._sandbox_timeout,
                api_key=self._api_key,
            )
        except SandboxNotFoundException:
            stale_id = self._sandbox_id
            _delete_sandbox_record(self._task_id, self._template, stale_id)
            if not create_if_missing:
                self._sandbox = None
                self._sandbox_id = None
                return False
            self._sandbox = self._create_sandbox()
            self._sandbox_id = _sandbox_id(self._sandbox)
            self._initialize_replacement_sandbox()
        except Exception as exc:
            raise _connection_error(f"sandbox reconnect ({self._sandbox_id})", exc) from exc
        return True

    def _require_sandbox(self):
        sandbox = self._sandbox
        if sandbox is None:
            raise EnvironmentConnectionError("E2B sandbox is not attached")
        return sandbox

    def _ensure_remote_hermes_dir(self) -> None:
        """Create the synced Hermes state directory in a fresh template."""
        self._run_remote_mkdir([_E2B_HERMES_HOME])

    def _run_remote_mkdir(self, directories: list[str]) -> None:
        """Create remote directories and surface command failures uniformly."""
        if not directories:
            return
        try:
            result = self._require_sandbox().commands.run(
                quoted_mkdir_command(directories),
                cwd=DEFAULT_E2B_CWD,
                timeout=max(self.timeout, 60),
            )
        except Exception as exc:
            raise _connection_error("sandbox home setup", exc) from exc
        if getattr(result, "exit_code", 0) != 0:
            detail = getattr(result, "stderr", "") or getattr(result, "stdout", "")
            raise _connection_error(
                "sandbox home setup",
                RuntimeError(detail or f"mkdir exited with {result.exit_code}"),
            )

    def _e2b_upload(self, host_path: str, remote_path: str) -> None:
        parent = posixpath.dirname(remote_path)
        if parent:
            self._run_remote_mkdir([parent])
        self._require_sandbox().files.write(remote_path, Path(host_path).read_bytes())

    def _e2b_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return
        parents = unique_parent_dirs(files)
        if parents:
            self._run_remote_mkdir(parents)
        payload = [
            {"path": remote_path, "data": Path(host_path).read_bytes()}
            for host_path, remote_path in files
        ]
        self._require_sandbox().files.write_files(payload)

    def _e2b_delete(self, remote_paths: list[str]) -> None:
        sandbox = self._require_sandbox()
        for remote_path in remote_paths:
            try:
                sandbox.files.remove(remote_path)
            except Exception as exc:
                # FileSyncManager deletion is idempotent. E2B exposes a
                # dedicated file-not-found exception, so suppress only that.
                from e2b.exceptions import FileNotFoundException

                if not isinstance(exc, FileNotFoundException):
                    raise

    def _e2b_bulk_download(self, dest_tar_path: Path) -> None:
        sandbox = self._require_sandbox()
        remote_tar = f"/tmp/.hermes-sync-{os.getpid()}-{threading.get_ident()}.tar"
        try:
            result = sandbox.commands.run(
                f"tar cf {shlex.quote(remote_tar)} -C / home/user/.hermes",
                cwd=DEFAULT_E2B_CWD,
                timeout=max(self.timeout, 60),
            )
            if getattr(result, "exit_code", 1) != 0:
                raise RuntimeError(
                    f"E2B bulk download failed: {getattr(result, 'stderr', '')}"
                )
            dest_tar_path.write_bytes(bytes(sandbox.files.read(remote_tar, format="bytes")))
        finally:
            try:
                sandbox.files.remove(remote_tar)
            except Exception:
                pass

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
        """Run through E2B and cancel only the command PID on timeout/interrupt."""
        del stdin_data
        sandbox = self._require_sandbox()
        shell_cmd = f"bash {'-l ' if login else ''}-c {shlex.quote(cmd_string)}"
        sdk_timeout = max(int(timeout) + _COMMAND_TIMEOUT_GRACE_SECONDS, 5)
        state_lock = threading.Lock()
        state: dict[str, Any] = {"handle": None, "cancel_requested": False}

        def cancel() -> None:
            with state_lock:
                state["cancel_requested"] = True
                handle = state["handle"]
            if handle is not None:
                handle.kill()

        def exec_fn() -> tuple[str, int]:
            from e2b.sandbox.commands.command_handle import CommandExitException
            from e2b.exceptions import AuthenticationException, SandboxException

            try:
                handle = sandbox.commands.run(
                    shell_cmd,
                    background=True,
                    cwd=DEFAULT_E2B_CWD,
                    timeout=sdk_timeout,
                )
                with state_lock:
                    state["handle"] = handle
                    cancel_requested = state["cancel_requested"]
                if cancel_requested:
                    handle.kill()

                chunks: list[str] = []
                try:
                    result = handle.wait(
                        on_stdout=lambda value: chunks.append(value),
                        on_stderr=lambda value: chunks.append(value),
                    )
                    if not chunks:
                        chunks.extend([result.stdout or "", result.stderr or ""])
                    return "".join(chunks), result.exit_code
                except CommandExitException as exc:
                    if not chunks:
                        chunks.extend([exc.stdout or "", exc.stderr or ""])
                    return "".join(chunks), exc.exit_code
            except (AuthenticationException, SandboxException) as exc:
                raise _connection_error("command execution", exc) from exc

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def _wait_for_process(
        self,
        proc,
        timeout: int = 120,
        *,
        bounded_capture: bool = False,
    ) -> dict:
        result = super()._wait_for_process(
            proc,
            timeout=timeout,
            bounded_capture=bounded_capture,
        )
        process_error = getattr(proc, "_error", None)
        if process_error is not None:
            raise process_error
        return result

    def cleanup(self) -> None:
        with self._lock:
            sandbox = self._sandbox
            sandbox_id = self._sandbox_id
            if sandbox is None:
                return

            if self._persistent:
                try:
                    # E2B may have auto-paused the sandbox after its timeout
                    # while Hermes was idle. Reconnect before pulling remote
                    # state so sync_back can read a filesystem-only snapshot.
                    if not self._ensure_sandbox_ready(create_if_missing=False):
                        logger.info(
                            "E2B: sandbox %s for task %s no longer exists",
                            sandbox_id,
                            self._task_id,
                        )
                        return
                    sandbox = self._require_sandbox()
                    sandbox_id = self._sandbox_id
                except Exception as exc:
                    logger.warning(
                        "E2B: reconnect before sync_back failed for task %s: %s",
                        self._task_id,
                        exc,
                    )
                    self._sandbox = None
                    self._sandbox_id = None
                    return

            try:
                self._sync_manager.sync_back()
            except Exception as exc:
                logger.warning("E2B: sync_back failed for task %s: %s", self._task_id, exc)

            cleanup_succeeded = False
            try:
                if self._persistent:
                    paused = sandbox.pause(keep_memory=False, api_key=self._api_key)
                    if paused:
                        logger.info(
                            "E2B: paused sandbox %s for task %s (filesystem preserved)",
                            sandbox_id,
                            self._task_id,
                        )
                    else:
                        logger.info(
                            "E2B: sandbox %s for task %s was already paused",
                            sandbox_id,
                            self._task_id,
                        )
                else:
                    killed = sandbox.kill(api_key=self._api_key)
                    if killed:
                        logger.info("E2B: killed ephemeral sandbox %s", sandbox_id)
                    else:
                        logger.info("E2B: ephemeral sandbox %s was already gone", sandbox_id)
                cleanup_succeeded = True
            except Exception as exc:
                # Preserve the pointer after a failed pause: the configured
                # on-timeout lifecycle still pauses it, and a later retry can
                # reconnect. Keep an ephemeral sandbox attached after a failed
                # kill so explicit cleanup or the destructor can retry it.
                logger.warning("E2B: cleanup failed for sandbox %s: %s", sandbox_id, exc)
            finally:
                if self._persistent or cleanup_succeeded:
                    self._sandbox = None
                    self._sandbox_id = None
