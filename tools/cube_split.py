"""Cube split helpers — keep core touch points small for upstream rebases.

Low-risk file I/O routing lives here. High-risk ``terminal`` / ``execute_code``
routing is owned by ``plugins/cube_sandbox/``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CUBE_SANDBOX_TYPE = "cube"


def is_cube_split_enabled() -> bool:
    """Return True when file tools must stay on the Pod workspace, not Cube."""
    return os.getenv("SANDBOX_TYPE", "").strip().lower() == _CUBE_SANDBOX_TYPE


def cube_session_key_enabled() -> bool:
    """Return True when terminal/code should use instance-session sandbox keys."""
    if is_cube_split_enabled():
        return True
    return os.getenv("TERMINAL_ENV", "").strip() == "cube_sandbox"


def sandbox_task_id_for_session(session_id: str) -> str | None:
    """Map a session id to the terminal env cache key used under Cube isolation.

    ``AIAgent.close()`` and ``cleanup_vm()`` receive the bare ``session_id``,
    while ``terminal_tool`` stores environments under
    ``{HERMES_INSTANCE_NAME}-{session_id}``.  Pass an already-composite key
    (starts with ``{instance}-``) through unchanged.
    """
    if not cube_session_key_enabled():
        return None
    sid = (session_id or "").strip()
    if not sid:
        return None
    instance = os.getenv("HERMES_INSTANCE_NAME", "local")
    prefix = f"{instance}-"
    if sid.startswith(prefix):
        return sid
    return f"{prefix}{sid}"


def resolve_sandbox_task_id() -> str | None:
    """Return ``{HERMES_INSTANCE_NAME}-{HERMES_SESSION_ID}`` when session isolation applies."""
    if not cube_session_key_enabled():
        return None
    try:
        from gateway.session_context import get_session_env
    except ImportError:
        return None
    session_id = get_session_env("HERMES_SESSION_ID", "").strip()
    if not session_id:
        return None
    return sandbox_task_id_for_session(session_id)


def env_cache_keys_for_cleanup(task_id: str) -> list[str]:
    """Return candidate keys for ``cleanup_vm()`` and ``process_registry.kill_all()``.

    Cube session isolation stores envs and background processes under
    ``{instance}-{session_id}`` while ``AIAgent.close()`` passes the bare
    ``session_id`` — include both forms.
    When Cube is disabled, returns only *task_id* (unchanged upstream behavior).
    """
    keys: list[str] = []

    def _add(key: str) -> None:
        normalized = (key or "").strip()
        if normalized and normalized not in keys:
            keys.append(normalized)

    _add(task_id)
    cube_key = sandbox_task_id_for_session(task_id)
    if cube_key:
        _add(cube_key)
    return keys


def live_tracking_cwd_if_split() -> str | None:
    """Pod workspace root for path resolution when cube split is active."""
    if not is_cube_split_enabled():
        return None
    return str(workspace_root())


def workspace_root() -> Path:
    """Absolute Pod workspace root for low-risk file I/O."""
    raw = os.getenv("HERMES_WORKSPACE_ROOT", "").strip()
    if raw:
        base = Path(os.path.expanduser(raw))
    else:
        terminal_cwd = os.getenv("TERMINAL_CWD", "").strip()
        if terminal_cwd:
            base = Path(os.path.expanduser(terminal_cwd))
        else:
            base = Path("/workspace")
    if not base.is_absolute():
        base = Path(os.getcwd()) / base
    return base.resolve()


def workspace_file_ops_cache_key(task_id: str) -> str:
    return f"{task_id}:workspace"


def remote_workspace_root() -> str:
    """Absolute path inside the Cube microVM for synced workspace files."""
    raw = os.getenv("CUBE_REMOTE_WORKSPACE_MOUNT", "").strip()
    if raw:
        return raw.rstrip("/")
    return "/home/user/workspace"


def get_split_file_ops(task_id: str = "default") -> Any | None:
    """Pod-local ``ShellFileOperations`` for cube split; ``None`` when not active."""
    if not is_cube_split_enabled():
        return None

    from tools.environments.local import LocalEnvironment
    from tools.file_tools import ShellFileOperations, _file_ops_cache, _file_ops_lock
    from tools.terminal_tool import _resolve_container_task_id

    container_key = _resolve_container_task_id(task_id)
    cache_key = workspace_file_ops_cache_key(container_key)

    with _file_ops_lock:
        cached = _file_ops_cache.get(cache_key)
        if cached is not None:
            return cached

    ws = workspace_root()
    ws.mkdir(parents=True, exist_ok=True)
    timeout_raw = os.getenv("TERMINAL_TIMEOUT", "120")
    try:
        timeout = int(timeout_raw)
    except ValueError:
        timeout = 120

    local_env = LocalEnvironment(cwd=str(ws), timeout=timeout)
    file_ops = ShellFileOperations(local_env)
    with _file_ops_lock:
        _file_ops_cache[cache_key] = file_ops
    logger.info(
        "Cube split: workspace file ops ready at %s (key=%s)",
        ws,
        cache_key,
    )
    return file_ops


def file_tool_hook(
    path: str,
    task_id: str,
    *,
    resolved: Path | str | None = None,
    register: bool = False,
) -> str | None:
    """Cube-split path guard / touched-path registration (no-op when overlay absent)."""
    try:
        from tools.workspace_sync import file_tool_register_touched, file_tool_validate_path
    except ImportError:
        return None

    if register:
        if resolved is not None:
            file_tool_register_touched(resolved, task_id)
        return None
    return file_tool_validate_path(path, task_id, resolved=resolved)


def clear_split_state(task_id: str) -> None:
    """Clear cube split registry state after env cleanup (touched paths + workspace cache)."""
    try:
        from tools.workspace_sync import clear_touched
    except ImportError:
        clear_touched = None  # type: ignore[assignment]

    try:
        from tools.file_tools import _file_ops_cache, _file_ops_lock
    except ImportError:
        return

    with _file_ops_lock:
        _file_ops_cache.pop(workspace_file_ops_cache_key(task_id), None)

    if clear_touched is not None:
        clear_touched(task_id)


def deferred_file_ops_cache_clear() -> bool:
    """True when ``file_ops`` cache must survive ``env.cleanup()`` (cube sync_back)."""
    return is_cube_split_enabled()


def env_teardown_begin() -> bool:
    """Return True when ``file_ops`` cache must survive ``env.cleanup()`` (cube sync_back)."""
    return deferred_file_ops_cache_clear()


def env_teardown_end(task_id: str) -> None:
    """Finalize cube split state after ``env.cleanup()`` (touched paths + file_ops cache)."""
    clear_split_state(task_id)
    try:
        from tools.file_tools import clear_file_ops_cache

        clear_file_ops_cache(task_id)
    except ImportError:
        pass
