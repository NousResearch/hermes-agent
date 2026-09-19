"""Path resolution for the file tools: task-aware base dir, ``~`` expansion, workspace-divergence warning.

Core invariant: the base directory anchoring relative paths is ALWAYS absolute
and derived from the task's terminal cwd, never from the process cwd unless no
other anchor exists (a relative/sentinel ``TERMINAL_CWD`` would silently anchor
edits to the agent process cwd, e.g. the main repo during a worktree session).
"""

import os
import posixpath
import sys
from pathlib import Path, PurePosixPath

# ``TERMINAL_CWD`` values that mean "not configured" ("." from a stale config;
# "auto"/"cwd" are wizard placeholders). gateway/run.py sanitizes the same set.
_TERMINAL_CWD_SENTINELS = frozenset({"", ".", "./", "auto", "cwd"})
_CONTAINER_PATH_BACKENDS_FALLBACK = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})
# Backend name inferred from the live environment's class name (first match wins).
_ENV_CLASS_NAME_HINTS = ("local", "ssh", "docker", "singularity", "modal", "daytona")


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


def _resolve_base_dir(
    task_id: str = "default", *, container_paths: bool | None = None) -> Path | PurePosixPath:
    """Return the ABSOLUTE base directory for resolving relative paths:
    ``_authoritative_workspace_root``, else the process cwd as a last resort."""
    root = _authoritative_workspace_root(task_id)
    if container_paths is None:
        container_paths = _uses_container_paths(task_id)
    # A backend's relative cwd is anchored to the process cwd once, here.
    return _anchor(_host_text(root or os.getcwd(), container_paths), os.getcwd, container_paths)


def _resolve_path_for_task(filepath: str, task_id: str = "default") -> Path | PurePosixPath:
    """Resolve *filepath* against the task's absolute base directory
    (absolute inputs are returned resolved-but-unanchored)."""
    container_paths = _uses_container_paths(task_id)
    return _anchor(_host_text(filepath, container_paths),
                   lambda: _resolve_base_dir(task_id, container_paths=container_paths), container_paths)



# Message prefix distinguishing the doubled-path warning (#67185) from the
# plain out-of-workspace warning, so callers can prioritize it over staleness
# (matched with str.startswith — path contents can't spoof it).
_DOUBLED_PATH_MARKER = "Doubled path:"


def _cwd_echo_warning(
    filepath: str,
    resolved: Path | PurePosixPath,
    root: Path | PurePosixPath,
) -> str | None:
    """Warn when a relative input replays the workspace root's own directories.

    Models sometimes echo an absolute path without its leading separator
    (``home/user/dev/notes/x.md`` for ``/home/user/dev/notes/x.md``). Joined
    onto the base, that resolves to a silently doubled tree
    (``/home/user/dev/home/user/dev/notes/x.md``). The write succeeds (parents
    are created) and the doubled path lands INSIDE the workspace, so the
    out-of-workspace warning never fires. Detect the replayed prefix and
    name the absolute path the model most likely intended.
    """
    # Lexically collapse '.'/'..' first so 'junk/../home/user/dev/x' is still
    # detected and 'home/user/dev/../../..' (which un-doubles itself) is not.
    if isinstance(root, Path):
        input_parts = Path(os.path.normpath(filepath)).parts
    else:  # container mode — sandbox-local POSIX semantics
        input_parts = PurePosixPath(posixpath.normpath(filepath)).parts
    drive = getattr(root, "drive", "")
    is_unc = drive.startswith("\\\\")
    if is_unc:
        # A UNC anchor holds server+share; an echoed UNC path replays them as
        # ordinary segments, so include them in the comparison.
        root_parts = tuple(s for s in drive.strip("\\").split("\\") if s) + root.parts[1:]
    else:
        root_parts = root.parts[1:]  # root is always absolute; drop the anchor
    # A single-segment root (e.g. the container ``/workspace``) is not enough
    # evidence: ``workspace/x.md`` is usually a real subdirectory, not an echo.
    if len(root_parts) < 2 or len(input_parts) < len(root_parts):
        return None
    echo = input_parts[: len(root_parts)]
    if isinstance(root, Path) and os.name == "nt":
        # Match the filesystem's case-insensitivity rules, not casefold().
        matches = tuple(os.path.normcase(p) for p in echo) == tuple(
            os.path.normcase(p) for p in root_parts
        )
    else:
        matches = echo == root_parts
    if not matches:
        return None
    if not isinstance(root, Path):
        suggested = str(PurePosixPath("/").joinpath(*input_parts))
    elif is_unc:
        suggested = "\\\\" + "\\".join(input_parts)
    else:
        suggested = str(Path(root.anchor).joinpath(*input_parts))
    return (
        f"{_DOUBLED_PATH_MARKER} relative input {filepath!r} repeats the "
        f"active workspace's own directories ({str(root)!r}), so it resolved "
        f"to {str(resolved)!r}. This usually means an absolute path missing "
        f"its leading separator. If you meant {suggested!r}, pass that "
        f"absolute path instead."
    )


def _path_resolution_warning(filepath: str, resolved: Path, task_id: str = "default") -> str | None:
    """Warn when a RELATIVE path resolved OUTSIDE the task's workspace root (the
    edit is about to land in a different checkout than the terminal's cwd).
    ``None`` for absolute paths, an unknown root, or a path under the root.

    Also covers the inverse failure mode (#67185): a relative input that
    replays the workspace root's own directories resolves to a doubled path
    INSIDE the workspace — see :func:`_cwd_echo_warning`."""
    try:
        if Path(_expand_tilde(filepath)).is_absolute():
            return None
        workspace_root = _authoritative_workspace_root(task_id)
        if not workspace_root:
            return None
        if _uses_container_paths(task_id):
            root = _normalize_without_host_deref(Path(_expand_tilde(workspace_root)))
            display_root: Path | PurePosixPath = root
        else:
            display_root = Path(_expand_tilde(workspace_root))
            root = display_root.resolve()
        # Doubled-path guard first: a cwd-shaped relative input (absolute path
        # missing its leading separator) resolves INSIDE the workspace, so the
        # containment check below would wrongly stay silent (#67185). Check
        # the pre-resolve() root too: when the workspace cwd is reached via a
        # symlink, the model echoes the symlinked (display) form, whose parts
        # don't match the resolved root's.
        echo_warning = _cwd_echo_warning(filepath, resolved, root)
        if echo_warning is None and display_root != root:
            echo_warning = _cwd_echo_warning(filepath, resolved, display_root)
        if echo_warning:
            return echo_warning
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
