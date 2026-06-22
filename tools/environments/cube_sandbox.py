"""CubeSandbox microVM execution environment (E2B-compatible API).

Uses ``e2b_code_interpreter.Sandbox`` for KVM-isolated bash. The Hermes
process must reach both the control plane (``CUBE_API_URL``, typically port
3000) and the data plane (sandbox hostnames, typically port 443).
"""

from __future__ import annotations

import logging
import os
import shlex
import threading
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.cube_split import remote_workspace_root, workspace_root
from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
    _load_json_store,
    _save_json_store,
)
from tools.environments.file_sync import (
    FileSyncManager,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)
from tools.workspace_sync import (
    SYNC_BACK_SCOPE_WORKSPACE,
    WorkspaceSyncError,
    check_workspace_sync_ready,
    get_touched_paths,
    host_path_from_remote_workspace,
    iter_workspace_sync_files,
    remap_pod_path_to_vm,
    rewrite_terminal_command_for_workspace_sync,
    workspace_sync_back_after_terminal,
    workspace_sync_back_scope,
    workspace_sync_enabled,
    workspace_sync_max_bytes,
)

logger = logging.getLogger(__name__)

_SNAPSHOT_DIR = get_hermes_home() / ".cube-snapshots"


def _snapshot_path(task_id: str) -> Path:
    safe = task_id.replace("/", "_").replace("\\", "_")
    return _SNAPSHOT_DIR / f"{safe}.json"


def _load_task_snapshot(task_id: str) -> str | None:
    data = _load_json_store(_snapshot_path(task_id))
    snapshot_id = data.get("snapshot_id")
    if isinstance(snapshot_id, str) and snapshot_id.strip():
        return snapshot_id.strip()
    return None


def _save_task_snapshot(task_id: str, snapshot_id: str) -> None:
    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _save_json_store(_snapshot_path(task_id), {"snapshot_id": snapshot_id})


def _delete_task_snapshot(task_id: str) -> None:
    path = _snapshot_path(task_id)
    if path.exists():
        path.unlink()


def _resolve_api_url(api_url: str | None) -> str:
    return (
        (api_url or "").strip()
        or os.environ.get("CUBE_API_URL", "").strip()
        or os.environ.get("E2B_API_URL", "").strip()
    )


def _resolve_api_key(api_key: str | None) -> str:
    return (
        (api_key or "").strip()
        or os.environ.get("CUBE_API_KEY", "").strip()
        or os.environ.get("E2B_API_KEY", "").strip()
    )


def _resolve_template_id(template_id: str | None) -> str:
    return (
        (template_id or "").strip()
        or os.environ.get("CUBE_TEMPLATE_ID", "").strip()
    )


def _apply_e2b_env(api_url: str, api_key: str) -> None:
    os.environ["E2B_API_URL"] = api_url
    os.environ["CUBE_API_URL"] = api_url
    if api_key:
        os.environ["E2B_API_KEY"] = api_key
        os.environ["CUBE_API_KEY"] = api_key


def _ensure_cube_sdk() -> None:
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("terminal.cube_sandbox", prompt=False)
    except ImportError:
        pass
    except Exception as exc:
        raise ImportError(str(exc)) from exc


def _token_api_url() -> str:
    return (
        os.environ.get("SANDBOX_TOKEN_API_URL", "").strip()
        or os.environ.get("CUBE_TOKEN_API_URL", "").strip()
    )


def check_cube_sandbox_requirements() -> bool:
    """Verify Cube control plane, template, SDK, and credentials are configured."""
    if not _resolve_api_url(None):
        logger.error("CubeSandbox requires CUBE_API_URL or E2B_API_URL")
        return False
    if not _resolve_template_id(None):
        logger.error("CubeSandbox requires CUBE_TEMPLATE_ID")
        return False
    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("terminal.cube_sandbox", prompt=False)
    except Exception as exc:
        logger.error("CubeSandbox SDK unavailable: %s", exc)
        return False
    if not _token_api_url() and not _resolve_api_key(None):
        logger.error("CubeSandbox requires SANDBOX_TOKEN_API_URL or CUBE_API_KEY")
        return False
    return True


def _destroy_sandbox(sb: Any) -> None:
    for method in ("kill", "close"):
        fn = getattr(sb, method, None)
        if callable(fn):
            fn()
            return


class CubeSandboxEnvironment(BaseEnvironment):
    """CubeSandbox microVM backend via E2B-compatible SDK."""

    _stdin_mode = "heredoc"

    def __init__(
        self,
        cwd: str = "/home/user",
        timeout: int = 60,
        api_url: str | None = None,
        api_key: str | None = None,
        template_id: str | None = None,
        cpu: int = 1,
        memory: int = 5120,
        disk: int = 51200,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd or "/home/user", timeout=timeout)

        _ensure_cube_sdk()
        from e2b_code_interpreter import Sandbox

        self._Sandbox = Sandbox
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._template_id = _resolve_template_id(template_id)
        # Lifecycle only (cleanup). Do not wrap commands.run — execute_code RPC
        # polls the sandbox concurrently while python3 script.py is running.
        self._lifecycle_lock = threading.Lock()
        self._sandbox = None

        resolved_url = _resolve_api_url(api_url)
        if not resolved_url:
            raise RuntimeError(
                "CubeSandbox requires CUBE_API_URL or E2B_API_URL "
                "(control plane, typically http://cube-sandbox:3000)"
            )
        if not self._template_id:
            raise RuntimeError("CubeSandbox requires CUBE_TEMPLATE_ID (tpl-...)")

        resolved_key = _resolve_api_key(api_key)
        _apply_e2b_env(resolved_url, resolved_key)

        if cpu or memory or disk:
            logger.debug(
                "Cube: resource hints cpu=%s memory_mb=%s disk_mb=%s (SDK may ignore)",
                cpu, memory, disk,
            )

        create_template = self._template_id
        if self._persistent:
            snapshot_id = _load_task_snapshot(task_id)
            if snapshot_id:
                create_template = snapshot_id
                logger.info(
                    "Cube: restoring task %s from snapshot %s",
                    task_id, snapshot_id,
                )

        self._sandbox = Sandbox.create(template=create_template)
        logger.info(
            "Cube: sandbox %s ready for task %s (template=%s)",
            getattr(self._sandbox, "sandbox_id", "?"),
            task_id,
            create_template,
        )

        self._remote_workspace = remote_workspace_root()
        self._workspace_sync: FileSyncManager | None = None
        self._post_terminal_sync_back_failed = False
        if workspace_sync_enabled():
            pod_ws = workspace_root()
            self.cwd = self._remote_workspace
            sandbox = self._sandbox
            if sandbox is not None:
                sandbox.commands.run(
                    f"mkdir -p {shlex.quote(self._remote_workspace)}",
                    timeout=60.0,
                )
            workspace_scope = workspace_sync_back_scope() == SYNC_BACK_SCOPE_WORKSPACE
            pod_ws_resolved = pod_ws.resolve()

            def _resolve_workspace_remote(remote_path: str) -> str | None:
                return host_path_from_remote_workspace(
                    remote_path,
                    pod_workspace=pod_ws_resolved,
                    remote_root=self._remote_workspace,
                )

            self._workspace_sync = FileSyncManager(
                get_files_fn=lambda: iter_workspace_sync_files(
                    pod_ws,
                    paths=get_touched_paths(task_id),
                    remote_root=self._remote_workspace,
                    max_bytes=workspace_sync_max_bytes(),
                ),
                upload_fn=self._cube_upload_file,
                delete_fn=self._cube_delete_files,
                bulk_upload_fn=self._cube_bulk_upload,
                bulk_download_fn=self._cube_bulk_download_tar,
                resolve_unmapped_remote_path=(
                    _resolve_workspace_remote if workspace_scope else None
                ),
                sync_back_without_prior_push=workspace_scope,
                sync_back_fail_on_oversized_tar=workspace_scope,
            )
            logger.info(
                "Cube: workspace sync enabled (pod=%s → vm=%s)",
                pod_ws,
                self._remote_workspace,
            )

        self.init_session()

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
        rewrite_compound_background: bool = True,
    ) -> dict:
        if workspace_sync_enabled():
            command = rewrite_terminal_command_for_workspace_sync(command)
            if cwd:
                cwd = remap_pod_path_to_vm(cwd)
        try:
            return super().execute(
                command,
                cwd=cwd,
                timeout=timeout,
                stdin_data=stdin_data,
                rewrite_compound_background=rewrite_compound_background,
            )
        except WorkspaceSyncError as exc:
            return {"output": "", "returncode": -1, "error": str(exc)}

    def _after_execute(self, result: dict) -> None:
        if self._workspace_sync is None:
            return
        if not workspace_sync_back_after_terminal():
            return
        if result.get("returncode", 0) != 0:
            return
        if not self._workspace_sync.sync_back(hermes_home=workspace_root()):
            self._post_terminal_sync_back_failed = True
            logger.warning("Cube: post-terminal sync_back failed")
        else:
            self._post_terminal_sync_back_failed = False

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        sandbox = self._sandbox

        def cancel():
            logger.info("Cube: cancel requested for task %s (sandbox kept alive)", self._task_id)

        # e2b SDK already runs: /bin/bash -l -c <cmd>. Wrapping again in bash -c
        # breaks multi-line Hermes scripts (nested quoting) and hides failures
        # because non-zero exits raise CommandExitException.
        shell_cmd = cmd_string
        if stdin_data:
            shell_cmd = f"{shell_cmd} <<'HERMES_STDIN'\n{stdin_data}\nHERMES_STDIN"
        if login:
            logger.debug(
                "Cube: login=True ignored for task %s (E2B SDK always uses bash -l)",
                self._task_id,
            )

        def exec_fn() -> tuple[str, int]:
            from e2b.sandbox.commands.command_handle import CommandExitException

            def _format_result(stdout: str, stderr: str, error: Any, exit_code: int) -> tuple[str, int]:
                parts: list[str] = []
                if stdout:
                    parts.append(stdout.rstrip("\n"))
                if stderr:
                    parts.append(stderr.rstrip("\n"))
                if error:
                    parts.append(str(error))
                combined = "\n".join(p for p in parts if p)
                return combined, int(exit_code)

            if sandbox is None:
                raise RuntimeError("Cube sandbox is not available")
            try:
                result = sandbox.commands.run(
                    shell_cmd,
                    timeout=float(timeout),
                )
            except CommandExitException as exc:
                return _format_result(
                    exc.stdout or "",
                    exc.stderr or "",
                    exc.error,
                    exc.exit_code,
                )
            except Exception as exc:
                return f"[Cube commands.run error: {exc}]", 1

            return _format_result(
                getattr(result, "stdout", "") or "",
                getattr(result, "stderr", "") or "",
                getattr(result, "error", None),
                int(getattr(result, "exit_code", 0) or 0),
            )

        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def _cube_upload_file(self, host_path: str, remote_path: str) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Cube sandbox is not available")
        parent = str(Path(remote_path).parent)
        sandbox.commands.run(f"mkdir -p {shlex.quote(parent)}", timeout=60.0)
        sandbox.files.write(remote_path, Path(host_path).read_bytes())

    def _cube_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        from e2b.sandbox.filesystem.filesystem import WriteEntry

        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Cube sandbox is not available")
        if not files:
            return
        parents = unique_parent_dirs(files)
        if parents:
            sandbox.commands.run(quoted_mkdir_command(parents), timeout=120.0)
        entries = [
            WriteEntry(path=remote, data=Path(host).read_bytes())
            for host, remote in files
        ]
        sandbox.files.write_files(entries)

    def _cube_delete_files(self, remote_paths: list[str]) -> None:
        sandbox = self._sandbox
        if sandbox is None or not remote_paths:
            return
        try:
            sandbox.commands.run(quoted_rm_command(remote_paths), timeout=60.0)
        except Exception as exc:
            logger.debug("Cube: remote delete failed (best-effort): %s", exc)

    def _cube_bulk_download_tar(self, dest: Path) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Cube sandbox is not available")
        rel_base = self._remote_workspace.lstrip("/")
        remote_tar = f"/tmp/.workspace_sync.{os.getpid()}.tar"
        sandbox.commands.run(
            f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}",
            timeout=180.0,
        )
        payload = sandbox.files.read(remote_tar, format="bytes")
        dest.write_bytes(payload)
        try:
            sandbox.commands.run(f"rm -f {shlex.quote(remote_tar)}", timeout=30.0)
        except Exception:
            pass

    def _before_execute(self) -> None:
        if self._workspace_sync is None:
            return
        pod_ws = workspace_root()
        check_workspace_sync_ready(
            get_touched_paths(self._task_id),
            pod_workspace=pod_ws,
            remote_root=self._remote_workspace,
            max_bytes=workspace_sync_max_bytes(),
        )
        self._workspace_sync.sync(force=True, strict=True)

    def cleanup(self) -> None:
        with self._lifecycle_lock:
            if self._sandbox is None:
                return
            sb = self._sandbox

        try:
            if self._workspace_sync is not None:
                if not workspace_sync_back_after_terminal():
                    logger.info("Cube: syncing workspace from sandbox to Pod...")
                    if not self._workspace_sync.sync_back(hermes_home=workspace_root()):
                        logger.error("Cube: workspace sync_back failed")
                elif self._post_terminal_sync_back_failed:
                    logger.info(
                        "Cube: retrying workspace sync_back after post-terminal failure..."
                    )
                    if self._workspace_sync.sync_back(hermes_home=workspace_root()):
                        self._post_terminal_sync_back_failed = False
                    else:
                        logger.error("Cube: workspace sync_back retry failed")

            with self._lifecycle_lock:
                self._sandbox = None

            if self._persistent:
                snap = sb.create_snapshot()
                snapshot_id = getattr(snap, "snapshot_id", None)
                if isinstance(snapshot_id, str) and snapshot_id:
                    _save_task_snapshot(self._task_id, snapshot_id)
                    logger.info(
                        "Cube: saved snapshot %s for task %s",
                        snapshot_id, self._task_id,
                    )
                else:
                    logger.warning(
                        "Cube: create_snapshot returned no snapshot_id for task %s",
                        self._task_id,
                    )
            _destroy_sandbox(sb)
            logger.info("Cube: destroyed sandbox for task %s", self._task_id)
        except Exception as exc:
            logger.warning("Cube: cleanup failed for task %s: %s", self._task_id, exc)
            try:
                _destroy_sandbox(sb)
            except Exception:
                pass

        if not self._persistent:
            _delete_task_snapshot(self._task_id)
