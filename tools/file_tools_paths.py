"""Path resolution for the file tools: task-aware base dir, ``~`` expansion, workspace-divergence warning.

Core invariant: the base directory anchoring relative paths is ALWAYS absolute
and derived from the task's terminal cwd, never from the process cwd unless no
other anchor exists (a relative/sentinel ``TERMINAL_CWD`` would silently anchor
edits to the agent process cwd, e.g. the main repo during a worktree session).
"""

import os
import posixpath
import sys
import time
from pathlib import Path, PurePosixPath

# ``TERMINAL_CWD`` values that mean "not configured" ("." from a stale config;
# "auto"/"cwd" are wizard placeholders). gateway/run.py sanitizes the same set.
_TERMINAL_CWD_SENTINELS = frozenset({"", ".", "./", "auto", "cwd"})
_CONTAINER_PATH_BACKENDS_FALLBACK = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})
# Backend name inferred from the live environment's class name (first match wins).
_ENV_CLASS_NAME_HINTS = ("local", "ssh", "docker", "singularity", "modal", "daytona")
# Container task id -> monotonic time of the last failed SSH bring-up in path resolution.
_SSH_HOME_RETRY_AFTER = 30.0
_ssh_home_failed_at: dict[str, float] = {}


def _expand_tilde(path: str) -> str:
    """Expand ``~`` using the effective profile home (``get_subprocess_home``) so
    gateway/cron runs, whose process HOME may differ, agree with interactive CLI sessions.

    This mirrors ``hermes_constants.get_subprocess_home()`` so that ``~`` resolves consistently regardless
    of whether the tool runs interactively or inside a gateway-driven cron job (#48552).
    """
    if not path or "~" not in path:
        return path
    try:
        from hermes_constants import get_subprocess_home

        home = get_subprocess_home()
    except Exception:
        home = None
    if home and (path == "~" or path.startswith("~/")):
        return home if path == "~" else os.path.join(home, path[2:])
    return os.path.expanduser(path)


def _terminal_env_type_for_task(task_id: str = "default") -> str:
    """Best-effort terminal backend type for path-resolution decisions."""
    try:
        from tools.terminal_tool import (
            _active_environments, _env_lock, _get_env_config, _resolve_container_task_id)

        try:
            container_key = _resolve_container_task_id(task_id)
        except Exception:
            container_key = task_id
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)
        if env is not None:
            name = env.__class__.__name__.lower()
            hint = next((h for h in _ENV_CLASS_NAME_HINTS if h in name), None)
            stamped = getattr(env, "_hermes_backend_name", None)
            if hint or (isinstance(stamped, str) and stamped):
                return hint or stamped
        return str(_get_env_config().get("env_type") or os.getenv("TERMINAL_ENV") or "local").lower()
    except Exception:
        return str(os.getenv("TERMINAL_ENV") or "local").lower()


def _uses_container_paths(task_id: str = "default") -> bool:
    env_type = _terminal_env_type_for_task(task_id)
    try:
        from tools.terminal_tool import _is_container_backend

        return _is_container_backend(env_type)
    except Exception:
        return env_type in _CONTAINER_PATH_BACKENDS_FALLBACK


def container_backend_for_task(task_id: str = "default") -> str | None:
    """The task's backend name when its file paths belong to a container, else None."""
    return _terminal_env_type_for_task(task_id) if _uses_container_paths(task_id) else None


def _normalize_without_host_deref(path: str | Path | PurePosixPath) -> PurePosixPath:
    """Normalize path syntax without following host symlinks: container paths are
    meaningful inside the sandbox, and a host-side ``/workspace`` symlink must not rewrite them."""
    return PurePosixPath(posixpath.normpath(str(path)))


def _sentinel_free_abs_cwd(raw: str | None) -> str | None:
    """Return *raw* expanded when it is a non-sentinel ABSOLUTE anchor, else ``None``
    (a relative anchor is exactly the ambiguity that misroutes worktree edits)."""
    raw = str(raw or "").strip()
    if raw.lower() in _TERMINAL_CWD_SENTINELS:
        return None
    expanded = _expand_tilde(raw)
    return expanded if os.path.isabs(expanded) else None


def _configured_terminal_cwd() -> str | None:
    """Return ``$TERMINAL_CWD`` only when it names a real (absolute, non-sentinel) anchor.
    Scope-aware: under gateway multiplexing the routed profile's cwd lives in the per-turn scope."""
    # See #68559.
    from agent.runtime_cwd import scope_terminal_cwd

    return _sentinel_free_abs_cwd(scope_terminal_cwd() or None)


def _registered_task_cwd_override(task_id: str = "default") -> str | None:
    """Return a registered cwd override keyed by the RAW task id, when available.

    ``terminal_tool`` collapses CWD-only overrides to the shared ``"default"``
    env, but the cwd value stays keyed by the raw session id.
    """
    try:
        from tools.terminal_tool import resolve_task_overrides

        overrides = resolve_task_overrides(task_id)
    except Exception:
        return None

    return _sentinel_free_abs_cwd(overrides.get("cwd"))


def _authoritative_workspace_root(task_id: str = "default") -> str | None:
    """Best-effort absolute workspace root, or ``None`` when no reliable anchor exists.

    Order: (1) the session's own cwd record (per-session, so one session's
    ``cd`` never leaks into another); (2) a registered raw-keyed cwd override
    (TUI/Desktop/ACP); (3) a sentinel-free absolute ``$TERMINAL_CWD``.
    """
    try:
        from tools.terminal_tool import get_session_cwd

        recorded = get_session_cwd(task_id)
    except Exception:
        recorded = None
    return recorded or _registered_task_cwd_override(task_id) or _configured_terminal_cwd()


def _host_text(text: str, container_paths: bool) -> str:
    """Expand ``~``; on host backends also translate Git Bash ``/c/Users/...`` drive
    paths before Path sees them. Container/WSL Linux paths are never rewritten."""
    if not container_paths:
        from tools.environments.local import _msys_to_windows_path

        text = _msys_to_windows_path(text)
    return _expand_tilde(text)


def _anchor(text: str, base, container_paths: bool) -> Path | PurePosixPath:
    """Return *text* as an absolute, normalized path, joining it onto ``base()`` when
    relative. Container: pure-posix, no host deref. Host: resolve() (win32: ntpath normpath)."""
    if container_paths:
        if not posixpath.isabs(text):
            text = posixpath.join(str(base()), text)
        return _normalize_without_host_deref(text)
    if sys.platform == "win32":
        import ntpath

        if not ntpath.isabs(text):
            text = ntpath.join(str(base()), text)
        return Path(ntpath.normpath(text))
    p = Path(text)
    if not p.is_absolute():
        p = Path(base()) / p
    return p.resolve()


def _ssh_remote_anchor(task_id: str) -> str:
    """Working directory on the SSH target, read RAW: ``_authoritative_workspace_root``
    expands ``~`` on the Hermes host, which names a directory the remote does not have.

    Same precedence (session record, registered override, ``$TERMINAL_CWD``); a
    value that is neither ``~``-prefixed nor POSIX-absolute (a Windows or relative
    host path) is skipped. ``coerce_ssh_remote_cwd`` maps the host subprocess home back to ``~``.
    """
    from agent.runtime_cwd import scope_terminal_cwd
    from tools.terminal_tool import get_session_cwd, resolve_task_overrides
    from tools.terminal_tool_config import coerce_ssh_remote_cwd

    for raw in (get_session_cwd(task_id), resolve_task_overrides(task_id).get("cwd"), scope_terminal_cwd()):
        text = str(raw or "").strip()
        if text.lower() not in _TERMINAL_CWD_SENTINELS and (text.startswith("~") or posixpath.isabs(text)):
            return coerce_ssh_remote_cwd(text, "ssh")
    return "~"


def _ssh_remote_home(task_id: str) -> str | None:
    """The SSH user's home as detected at connect (``echo $HOME``), else None.

    Brings the environment up through the file tools' own creator (same cwd
    and cache as the call that follows) when none is live: they resolve before
    touching the backend, and a first call keyed ``~/x`` while later ones key
    ``/home/u/x`` splits read tracking and staleness checks for one file. A
    failed bring-up is remembered briefly so one tool call's several
    resolutions don't each wait out the SSH connect timeout; the tool's own
    backend call reports the error.
    """
    from tools.file_tools import _get_file_ops
    from tools.terminal_tool import _resolve_container_task_id
    from tools.terminal_tool_lifecycle import get_active_env

    key = _resolve_container_task_id(task_id)
    env = get_active_env(task_id)
    if env is None:
        if time.monotonic() - _ssh_home_failed_at.get(key, float("-inf")) < _SSH_HOME_RETRY_AFTER:
            return None
        try:
            env = _get_file_ops(task_id).env
        except Exception:  # noqa: BLE001 — connect failure
            _ssh_home_failed_at[key] = time.monotonic()
            return None
    _ssh_home_failed_at.pop(key, None)
    # A guessed /home/<user> (``echo $HOME`` failed) must not stand in for the real home.
    home = getattr(env, "_remote_home", None) if getattr(env, "_remote_home_detected", False) else None
    return home if isinstance(home, str) and posixpath.isabs(home) else None


def _resolve_ssh_path(filepath: str, task_id: str) -> PurePosixPath:
    """Resolve *filepath* in the SSH target's namespace, never via the Hermes host
    (``Path.resolve()`` and ``get_subprocess_home()`` both name host directories).

    ``~`` becomes the remote home once the live environment has detected it, so the
    result is absolute and the write guards see the real target. Before that it stays
    ``~``-prefixed for the remote shell; ``~user`` always does.
    """
    text = str(filepath or "").strip()
    if not text.startswith("~") and not posixpath.isabs(text):
        text = posixpath.join(_ssh_remote_anchor(task_id), text)
    if text == "~" or text.startswith("~/"):
        home = _ssh_remote_home(task_id)
        text = home + text[1:] if home else text
    if not text.startswith("~"):
        return _normalize_without_host_deref(text)
    head, _, tail = text.partition("/")
    tail = posixpath.normpath(tail) if tail else "."
    return PurePosixPath(head if tail == "." else f"{head}/{tail}")


def _ssh_path_escapes_home(resolved: str) -> bool:
    """True for a still-``~``-relative SSH path that climbs above ``~``: its absolute
    target is unknown until the remote home is, so no path guard can classify it."""
    tail = resolved.partition("/")[2] if resolved.startswith("~") else ""
    return tail == ".." or tail.startswith("../")


def _resolve_base_dir(
    task_id: str = "default", *, container_paths: bool | None = None) -> Path | PurePosixPath:
    """Return the ABSOLUTE base directory for resolving relative paths:
    ``_authoritative_workspace_root``, else the process cwd as a last resort.

    SSH resolves in the remote namespace (callers passing *container_paths* have
    already ruled SSH out).
    """
    if container_paths is None:
        if _terminal_env_type_for_task(task_id) == "ssh":
            return _resolve_ssh_path(".", task_id)
        container_paths = _uses_container_paths(task_id)
    root = _authoritative_workspace_root(task_id)
    # A backend's relative cwd is anchored to the process cwd once, here.
    return _anchor(_host_text(root or os.getcwd(), container_paths), os.getcwd, container_paths)


def _resolve_path_for_task(filepath: str, task_id: str = "default") -> Path | PurePosixPath:
    """Resolve *filepath* against the task's absolute base directory
    (absolute inputs are returned resolved-but-unanchored)."""
    if _terminal_env_type_for_task(task_id) == "ssh":
        return _resolve_ssh_path(filepath, task_id)
    container_paths = _uses_container_paths(task_id)
    return _anchor(_host_text(filepath, container_paths),
                   lambda: _resolve_base_dir(task_id, container_paths=container_paths), container_paths)



def _path_resolution_warning(filepath: str, resolved: Path | PurePosixPath, task_id: str = "default") -> str | None:
    """Warn when a RELATIVE path resolved OUTSIDE the task's workspace root (the
    edit is about to land in a different checkout than the terminal's cwd).
    ``None`` for absolute paths, an unknown root, or a path under the root.
    SSH compares in the remote namespace, as ``_resolve_path_for_task`` resolved it."""
    try:
        if _terminal_env_type_for_task(task_id) == "ssh":
            if filepath.startswith("~") or posixpath.isabs(filepath):
                return None
            root = _resolve_ssh_path(".", task_id)
        else:
            if Path(_expand_tilde(filepath)).is_absolute():
                return None
            workspace_root = _authoritative_workspace_root(task_id)
            if not workspace_root:
                return None
            if _uses_container_paths(task_id):
                root = _normalize_without_host_deref(Path(_expand_tilde(workspace_root)))
            else:
                root = Path(_expand_tilde(workspace_root)).resolve()
        if resolved.is_relative_to(root):
            return None
        return (
            f"Relative path {filepath!r} resolved to {str(resolved)!r}, which is "
            f"OUTSIDE the active workspace ({str(root)!r}). The edit will land in "
            f"a different directory than the terminal's cwd. If this is not "
            f"intended (e.g. a git-worktree session writing into the main "
            f"checkout), pass an absolute path under the workspace instead.")
    except Exception:
        return None
