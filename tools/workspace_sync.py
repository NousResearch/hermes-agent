"""Pod workspace ↔ Cube microVM file sync registry and path mapping.

Used with ``FileSyncManager`` in ``cube_sandbox.py`` when ``SANDBOX_TYPE=cube``.
Sync-in uses session-touched paths; sync-back defaults to the full remote workspace subtree.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from tools.cube_split import is_cube_split_enabled, remote_workspace_root, workspace_root

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MB = 100
_BLOCKED_DIR_NAMES = frozenset({".git", "node_modules"})

SYNC_BACK_MODE_AFTER_TERMINAL = "after_terminal"
SYNC_BACK_MODE_CLEANUP_ONLY = "cleanup_only"

SYNC_BACK_SCOPE_TOUCHED = "touched"
SYNC_BACK_SCOPE_WORKSPACE = "workspace"


class WorkspaceSyncError(RuntimeError):
    """Raised when registered workspace files cannot be synced (fail-closed)."""

_registry_lock = threading.Lock()
_touched: dict[str, set[str]] = {}


def workspace_sync_mode() -> str:
    """Return ``touched``, ``explicit``, or ``off``."""
    raw = os.getenv("CUBE_WORKSPACE_SYNC_MODE", "touched").strip().lower()
    if raw in {"off", "false", "0"}:
        return "off"
    if raw == "explicit":
        return "explicit"
    return "touched"


def workspace_sync_enabled() -> bool:
    """True when Pod→VM workspace sync should run (split mode only)."""
    if not is_cube_split_enabled():
        return False
    if workspace_sync_mode() == "off":
        return False
    env = os.getenv("CUBE_WORKSPACE_SYNC_ENABLED", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def workspace_sync_back_mode() -> str:
    """When to pull VM workspace changes back to the Pod.

    ``after_terminal`` (default): after each successful terminal/execute in Cube.
    ``cleanup_only``: only on sandbox ``cleanup()`` (idle teardown).
    """
    raw = os.getenv("CUBE_WORKSPACE_SYNC_BACK", SYNC_BACK_MODE_AFTER_TERMINAL).strip().lower()
    if raw in {SYNC_BACK_MODE_CLEANUP_ONLY, "cleanup", "idle"}:
        return SYNC_BACK_MODE_CLEANUP_ONLY
    if raw in {"off", "false", "0", "no"}:
        return SYNC_BACK_MODE_CLEANUP_ONLY
    return SYNC_BACK_MODE_AFTER_TERMINAL


def workspace_sync_back_after_terminal() -> bool:
    """True when VM→Pod sync should run after each successful command."""
    if not workspace_sync_enabled():
        return False
    return workspace_sync_back_mode() == SYNC_BACK_MODE_AFTER_TERMINAL


def workspace_sync_back_scope() -> str:
    """Egress scope for VM→Pod sync_back.

    ``workspace`` (default): mirror the entire remote workspace subtree.
    ``touched``: only files with a known host↔remote mapping from sync-in.
    """
    raw = os.getenv("CUBE_WORKSPACE_SYNC_BACK_SCOPE", SYNC_BACK_SCOPE_WORKSPACE).strip().lower()
    if raw in {SYNC_BACK_SCOPE_TOUCHED, "files", "mapped"}:
        return SYNC_BACK_SCOPE_TOUCHED
    return SYNC_BACK_SCOPE_WORKSPACE


def _workspace_prefix_pair() -> tuple[str, str] | None:
    """Return ``(pod_root, remote_root)`` when path rewriting applies."""
    if not workspace_sync_enabled():
        return None
    pod = str(pod_workspace_resolved())
    remote = remote_workspace_root().rstrip("/")
    if pod == remote:
        return None
    return pod, remote


def _map_pod_workspace_prefix(text: str, *, path_mode: bool = False) -> str:
    """Map Pod workspace path prefixes to VM workspace paths."""
    if not text:
        return text
    pair = _workspace_prefix_pair()
    if pair is None:
        return text
    pod, remote = pair
    if path_mode:
        norm = text.replace("\\", "/")
        if norm == pod:
            return remote
        prefix = pod + "/"
        if norm.startswith(prefix):
            return remote + norm[len(pod):]
        return text
    return text.replace(pod, remote)


def pod_workspace_resolved() -> Path:
    """Resolved Pod workspace root (from ``HERMES_WORKSPACE_ROOT``)."""
    return workspace_root().resolve()


def is_path_under_pod_workspace(
    path: str | Path,
    *,
    pod_workspace: Path | None = None,
) -> bool:
    """Return True when *path* resolves inside the Pod workspace root."""
    ws = (pod_workspace or pod_workspace_resolved())
    try:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = (ws / candidate).resolve()
        else:
            candidate = candidate.resolve()
        candidate.relative_to(ws)
        return True
    except (ValueError, OSError):
        return False


def looks_like_vm_only_path(path: str) -> bool:
    """True when *path* targets the microVM filesystem, not the Pod workspace.

    Used to reject ``write_file``/``read_file`` paths like ``/home/user/...``
    when Pod and VM roots differ. When both roots resolve to the same string
    (local dev), paths under that root are allowed.
    """
    if not is_cube_split_enabled():
        return False
    norm = path.replace("\\", "/")
    pod = str(pod_workspace_resolved())
    remote = remote_workspace_root().rstrip("/")
    if norm == pod or norm.startswith(pod + "/"):
        return False
    if remote == pod:
        return False
    if norm == remote or norm.startswith(remote + "/"):
        return True
    remote_parent = str(Path(remote).parent)
    if norm == remote_parent or norm.startswith(remote_parent + "/"):
        return True
    return False


def validate_file_tool_path(
    path: str,
    *,
    resolved: Path | str | None = None,
    pod_workspace: Path | None = None,
) -> str | None:
    """Return an error message when *path* is invalid for cube-split file tools."""
    if not is_cube_split_enabled():
        return None
    if looks_like_vm_only_path(path):
        return (
            f"Path {path!r} is a microVM path. File tools run on the Pod workspace — "
            "use a workspace-relative path (e.g. the filename only) instead."
        )
    ws = pod_workspace or pod_workspace_resolved()
    target = Path(resolved).expanduser() if resolved is not None else None
    if target is None:
        try:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                target = (ws / candidate).resolve()
            else:
                target = candidate.resolve()
        except OSError as exc:
            return f"Cannot resolve path {path!r}: {exc}"
    if not is_path_under_pod_workspace(target, pod_workspace=ws):
        return (
            f"Path resolves to {str(target)!r}, outside the Pod workspace ({str(ws)!r}). "
            "Use a workspace-relative path (e.g. 'tangshi.txt')."
        )
    return None


def remap_pod_path_to_vm(path: str) -> str:
    """Map a Pod workspace path string to its VM workspace counterpart."""
    return _map_pod_workspace_prefix(path, path_mode=True)


def rewrite_terminal_command_for_workspace_sync(command: str) -> str:
    """Replace Pod workspace path prefixes in a shell command with VM paths."""
    pair = _workspace_prefix_pair()
    if pair is None or not command:
        return command
    pod, _remote = pair
    if pod not in command:
        return command
    return _map_pod_workspace_prefix(command)


def host_path_from_remote_workspace(
    remote_path: str,
    *,
    pod_workspace: Path | None = None,
    remote_root: str | None = None,
) -> str | None:
    """Map a VM workspace path to its Pod host path (workspace-scope sync_back)."""
    remote_base = (remote_root or remote_workspace_root()).rstrip("/")
    norm = remote_path.replace("\\", "/")
    if norm != remote_base and not norm.startswith(remote_base + "/"):
        return None
    rel = norm[len(remote_base):].lstrip("/")
    if not rel or _is_blocked_relative(rel):
        return None
    if ".." in rel.split("/"):
        return None
    ws = (pod_workspace or workspace_root()).resolve()
    try:
        host = (ws / rel).resolve()
        host.relative_to(ws)
    except (ValueError, OSError):
        return None
    return str(host)


def workspace_sync_max_bytes() -> int:
    """Per-file byte cap for sync (default 100 MiB)."""
    raw = (
        os.getenv("CUBE_WORKSPACE_SYNC_MAX_MB", "").strip()
        or os.getenv("TERMINAL_FILE_SYNC_MAX_MB", "").strip()
        or str(_DEFAULT_MAX_MB)
    )
    try:
        mb = int(raw)
    except ValueError:
        mb = _DEFAULT_MAX_MB
    return max(1, mb) * 1024 * 1024


def get_explicit_paths() -> set[str]:
    """Relative paths from ``HERMES_WORKSPACE_SYNC_PATHS`` (comma-separated)."""
    raw = os.getenv("HERMES_WORKSPACE_SYNC_PATHS", "").strip()
    if not raw:
        return set()
    return {p.strip().replace("\\", "/").lstrip("/") for p in raw.split(",") if p.strip()}


def _normalize_relative(rel: str) -> str:
    return rel.replace("\\", "/").lstrip("/")


def _is_blocked_relative(rel: str) -> bool:
    parts = _normalize_relative(rel).split("/")
    return any(part in _BLOCKED_DIR_NAMES for part in parts)


def host_to_remote(
    host_path: Path | str,
    pod_workspace: Path | None = None,
    remote_root: str | None = None,
) -> str | None:
    """Map a Pod workspace file to its VM path."""
    ws = (pod_workspace or workspace_root()).resolve()
    host = Path(host_path).expanduser().resolve()
    try:
        rel = host.relative_to(ws)
    except ValueError:
        return None
    root = (remote_root or remote_workspace_root()).rstrip("/")
    return f"{root}/{rel.as_posix()}"


def file_tool_validate_path(
    path: str,
    task_id: str,
    *,
    resolved: Path | str | None = None,
) -> str | None:
    """Return a tool-error message when *path* violates cube-split rules."""
    if not is_cube_split_enabled():
        return None
    return validate_file_tool_path(path, resolved=resolved)


def file_tool_register_touched(resolved: str | Path, task_id: str) -> None:
    """Register a Pod workspace file for VM sync before high-risk tools run."""
    if not is_cube_split_enabled():
        return
    try:
        from tools.terminal_tool import _resolve_container_task_id

        register_touched(_resolve_container_task_id(task_id), resolved)
    except Exception:
        pass


def register_touched(task_id: str, host_path: str | Path) -> None:
    """Record a workspace file that may need sync into the microVM."""
    if not workspace_sync_enabled():
        return
    ws = workspace_root()
    host = Path(host_path).expanduser()
    if not host.is_absolute():
        host = (ws / host).resolve()
    else:
        host = host.resolve()
    try:
        rel = _normalize_relative(str(host.relative_to(ws.resolve())))
    except ValueError:
        return
    if not rel or _is_blocked_relative(rel):
        return
    with _registry_lock:
        _touched.setdefault(task_id, set()).add(rel)


def get_touched_paths(task_id: str) -> set[str]:
    """Return relative paths registered for *task_id* (plus explicit paths)."""
    with _registry_lock:
        paths = set(_touched.get(task_id, set()))
    if workspace_sync_mode() == "explicit":
        paths |= get_explicit_paths()
    return paths


def clear_touched(task_id: str | None = None) -> None:
    """Drop touched-path registry (mirrors file_ops cache lifecycle)."""
    with _registry_lock:
        if task_id is None:
            _touched.clear()
        else:
            _touched.pop(task_id, None)


def _resolve_sync_pair(
    rel_norm: str,
    ws: Path,
    cap: int,
    remote_base: str,
) -> tuple[str, str]:
    """Resolve one relative path to ``(host_path, remote_path)`` or raise."""
    if not rel_norm:
        raise WorkspaceSyncError("empty relative path")
    if _is_blocked_relative(rel_norm):
        raise WorkspaceSyncError(
            f"{rel_norm}: blocked path (.git/ or node_modules/ not synced)"
        )
    host = ws / rel_norm
    if not host.is_file():
        raise WorkspaceSyncError(f"{rel_norm}: file not found on Pod workspace")
    try:
        size = host.stat().st_size
    except OSError as exc:
        raise WorkspaceSyncError(f"{rel_norm}: cannot stat file ({exc})") from exc
    if size > cap:
        raise WorkspaceSyncError(
            f"{rel_norm}: {size} bytes exceeds sync cap ({cap} bytes); "
            "increase CUBE_WORKSPACE_SYNC_MAX_MB or use a smaller file"
        )
    remote = host_to_remote(host, ws, remote_base)
    if remote is None:
        raise WorkspaceSyncError(f"{rel_norm}: path is outside Pod workspace root")
    return str(host), remote


def check_workspace_sync_ready(
    paths: set[str],
    *,
    pod_workspace: Path | None = None,
    remote_root: str | None = None,
    max_bytes: int | None = None,
) -> None:
    """Fail-closed preflight: every registered path must be syncable."""
    if not paths:
        return
    ws = (pod_workspace or workspace_root()).resolve()
    cap = max_bytes if max_bytes is not None else workspace_sync_max_bytes()
    remote_base = remote_root or remote_workspace_root()
    errors: list[str] = []
    for rel in sorted(paths):
        rel_norm = _normalize_relative(rel)
        try:
            _resolve_sync_pair(rel_norm, ws, cap, remote_base)
        except WorkspaceSyncError as exc:
            errors.append(str(exc))
    if errors:
        detail = "\n".join(f"  - {msg}" for msg in errors)
        raise WorkspaceSyncError(
            "Workspace sync cannot proceed for registered file(s):\n" + detail
        )


def iter_workspace_sync_files(
    pod_workspace: Path | None = None,
    *,
    paths: set[str] | None = None,
    remote_root: str | None = None,
    max_bytes: int | None = None,
) -> list[tuple[str, str]]:
    """Build ``(host_path, remote_path)`` pairs for ``FileSyncManager``.

    Skips paths that cannot be resolved (used for sync_back mapping). Call
    :func:`check_workspace_sync_ready` before sync-in when fail-closed behavior
    is required.
    """
    rel_paths = paths or set()
    if not rel_paths:
        return []

    ws = (pod_workspace or workspace_root()).resolve()
    cap = max_bytes if max_bytes is not None else workspace_sync_max_bytes()
    remote_base = remote_root or remote_workspace_root()
    out: list[tuple[str, str]] = []

    for rel in sorted(rel_paths):
        rel_norm = _normalize_relative(rel)
        try:
            out.append(_resolve_sync_pair(rel_norm, ws, cap, remote_base))
        except WorkspaceSyncError as exc:
            logger.debug("workspace_sync: omitting %s from file list: %s", rel_norm, exc)
    return out
