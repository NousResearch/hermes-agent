"""MCP server config loading and stdio launch environment: ${VAR}/Cursor-style
interpolation, hidden-whitespace and suspicious-entry filtering, the filtered
subprocess env, command resolution, the cached-npx binary shortcut and the shared stderr log."""

import json
import logging
import os
import re
import shutil
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from tools.mcp_tool_common import _env_ref_name, _prepend_path

logger = logging.getLogger("tools.mcp_tool")

_mcp_stderr_log_fh: Dict[str, Any] = {}  # profile home key -> handle
_mcp_stderr_log_lock = threading.Lock()


def _get_mcp_stderr_log() -> Any:
    """Shared append-mode handle for MCP subprocess stderr, cached until shutdown PER PROFILE HOME (a
    multiplexed gateway's secondary profile must log under ITS ``logs/``, not the launch profile's). Must
    expose a real fd (asyncio wires the child's stderr to it); falls back to ``/dev/null``, then real stderr."""
    from hermes_constants import get_hermes_home, hermes_home_key, mkdir_under_hermes_home
    home_key = hermes_home_key()
    with _mcp_stderr_log_lock:
        fh = _mcp_stderr_log_fh.get(home_key)
        if fh is None or fh.closed:
            try:
                log_dir = get_hermes_home() / "logs"
                mkdir_under_hermes_home(log_dir)
                # Line-buffered so output lands promptly; errors="replace" tolerates garbled binary.
                fh = open(log_dir / "mcp-stderr.log", "a", encoding="utf-8", errors="replace", buffering=1)
                fh.fileno()  # confirm a real fd before committing
            except Exception as exc:  # pragma: no cover — best-effort fallback
                logger.debug("Failed to open MCP stderr log, using devnull: %s", exc)
                try:
                    fh = open(os.devnull, "w", encoding="utf-8")
                except Exception:
                    fh = sys.stderr
            _mcp_stderr_log_fh[home_key] = fh
        return fh


def _close_mcp_stderr_logs(*, scope: Optional[str] = None) -> None:
    """Release cached parent handles after the selected MCP transports have stopped."""
    with _mcp_stderr_log_lock:
        keys = list(_mcp_stderr_log_fh) if scope is None else [scope]
        for key in keys:
            fh = _mcp_stderr_log_fh.pop(key, None)
            # The last-resort fallback is borrowed, not owned by MCP.
            if fh is None or fh is sys.stderr or fh is sys.__stderr__:
                continue
            try:
                fh.close()
            except OSError:
                logger.warning("Could not close MCP stderr log for %s", key, exc_info=True)


def _write_stderr_log_header(server_name: str) -> None:
    """Session marker so operators can find each server's output in the shared log
    (per-line prefixes would need a pipe + reader thread)."""
    fh = _get_mcp_stderr_log()
    try:
        fh.write(f"\n===== [{datetime.now():%Y-%m-%d %H:%M:%S}] starting MCP server '{server_name}' =====\n")
        fh.flush()
    except Exception:
        pass


# Env vars safe to pass to stdio subprocesses (no secrets).
_SAFE_ENV_KEYS = frozenset({"PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR"})

# Windows process/location vars needed by launcher-style tools (e.g. Docker Desktop's MCP plugin discovery).
_SAFE_ENV_KEYS_CASE_INSENSITIVE = frozenset({
    "ALLUSERSPROFILE", "APPDATA", "COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)",
    "COMMONPROGRAMW6432", "COMPUTERNAME", "COMSPEC", "HOMEDRIVE", "HOMEPATH",
    "LOCALAPPDATA", "NUMBER_OF_PROCESSORS", "OS", "PATHEXT", "PROCESSOR_ARCHITECTURE",
    "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "PUBLIC",
    "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP", "USERDOMAIN", "USERNAME",
    "USERPROFILE", "WINDIR"})

# ${VAR_NAME} interpolation; any non-} chars allowed so MY-VAR / my.var work.
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _workspace_folder() -> str:
    """Absolute workspace root for ``${workspaceFolder}``: the session's authoritative root
    (terminal cwd / task override / $TERMINAL_CWD), else cwd."""
    try:
        from tools.file_tools_paths import _authoritative_workspace_root
        root = _authoritative_workspace_root()
    except Exception:
        root = None
    return root or os.getcwd()


def _workspace_basename() -> str:
    root = _workspace_folder()
    return os.path.basename(root.rstrip("/\\")) or root


# Cursor's case-sensitive context vars -> resolver.
_CONTEXT_VAR_RESOLVERS = {
    "userHome": lambda: os.path.expanduser("~"), "workspaceFolder": lambda: _workspace_folder(),
    "workspaceFolderBasename": _workspace_basename, "pathSeparator": lambda: os.sep, "/": lambda: os.sep}


def _build_safe_env(user_env: Optional[dict]) -> dict:
    """Filtered env for stdio subprocesses so API keys/tokens don't leak: the safe baseline
    keys, ``XDG_*``, vars injected by an external secret source (users configured that backend
    precisely so subprocesses can consume them), plus the server config's own ``env``."""
    from agent.secret_scope import get_secret
    from hermes_cli.env_loader import secret_source_names
    env = {
        key: value for key, value in os.environ.items()
        if key in _SAFE_ENV_KEYS or key.upper() in _SAFE_ENV_KEYS_CASE_INSENSITIVE or key.startswith("XDG_")}
    # Source-tagged names are process-wide (any profile's hydration tags them) while os.environ
    # holds only the LAUNCH profile's values, so the value must come from the active profile's
    # secret scope; a profile that lacks the name gets nothing, never another profile's token.
    for key in secret_source_names():
        value = get_secret(key)
        if value is not None:
            env[key] = value
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        if key in os.environ:
            env[key] = os.environ[key]
    if user_env:
        env.update(user_env)
    from agent.delegation_context import delegated_child_subprocess_env
    return delegated_child_subprocess_env(env)


def _which_with_config_pathext(command: str, path_arg, env: dict):
    """Resolve *command* under the config env's PATHEXT (Windows only; ``shutil.which`` uses the PARENT's).

    The extension walk mirrors ``which`` itself (existing-suffix short-circuit, configured
    extensions in order) but reads nothing from and writes nothing to ``os.environ``: swapping
    the parent's PATHEXT around a ``which`` call would publish this server's per-profile value
    to every other thread for the duration, and a ``finally``-restore cannot undo that window."""
    cfg_pathext = next((v for k, v in env.items() if k.upper() == "PATHEXT" and isinstance(v, str) and v.strip()), None)
    if not cfg_pathext or cfg_pathext == os.environ.get("PATHEXT"):
        return None
    # PATHEXT is Windows-defined: ";"-separated even when resolved off-Windows
    exts = [ext for ext in cfg_pathext.split(";") if ext]
    candidates = [command + ext for ext in exts]
    if not candidates or any(command.lower().endswith(ext.lower()) for ext in exts):
        candidates = [command]
    directories = str(path_arg or "").split(os.pathsep)
    if sys.platform == "win32" and os.curdir not in directories:
        directories.insert(0, os.curdir)  # Windows resolves from the cwd first
    for raw in directories:
        directory = raw or os.curdir  # POSIX: an empty PATH component means the cwd
        if not os.path.isdir(directory):
            continue
        for candidate in candidates:
            resolved = os.path.join(directory, candidate)
            if os.path.isfile(resolved) and os.access(resolved, os.F_OK | os.X_OK):
                return resolved
    return None


def _launcher_fallback(command: str, *, windows: Optional[bool] = None) -> str:
    """Well-known install locations for bare launcher commands; *command* unchanged when none exists.

    One resolver for two launcher families. ``npx``/``npm``/``node``: the managed tree comes from
    ``iter_hermes_node_dirs`` (Windows unpacks into ``<home>\\node``, POSIX into ``<home>/node/bin``)
    under the active profile's ``get_hermes_home()``; on Windows the real files are
    ``npx.cmd``/``node.exe`` (``windows`` injectable, as for ``_npx_bin_candidates``).
    ``uv``/``uvx``: GUI launches (the Electron desktop app, macOS LaunchAgents) inherit the bare
    ``/usr/bin:/bin:/usr/sbin:/sbin`` PATH, which carries none of uv's install locations, so a bare
    ``command: uvx`` MCP server fails with ENOENT at ``execvp`` from Desktop even though it works
    from an interactive terminal (#37589). The directory table lives in
    ``hermes_platform.resolver.known_dirs.uv_tool_dirs`` (probed in the order uv's own docs install
    it: the per-user installer first, then Homebrew); the Hermes-managed ``<home>/bin`` is probed
    before it."""
    from hermes_constants import get_hermes_home
    from hermes_platform.resolver.known_dirs import uv_tool_dirs
    home = os.path.expanduser("~")
    if command in {"uv", "uvx"}:
        # expanduser: the table carries the ``~`` form so both this walk and
        # locate_command's expandvars+expanduser agree on one spelling.
        directories = [os.path.join(str(get_hermes_home()), "bin"),
                       *(os.path.expanduser(d) for d in uv_tool_dirs())]
    else:
        from hermes_constants import iter_hermes_node_dirs
        # /usr/local/bin: canonical Node location (from-source Linux, Hermes Docker image, Intel
        # Homebrew), needed when a hand-authored env.PATH omits it — npx's shebang re-execs
        # /usr/bin/env node.
        directories = [*map(str, iter_hermes_node_dirs(get_hermes_home())),
                      os.path.join(home, ".local", "bin"), os.path.join(os.sep, "usr", "local", "bin")]
    candidates = (c for d in directories for c in _npx_bin_candidates(d, command, windows=windows))
    return next((c for c in candidates if os.path.isfile(c) and os.access(c, os.X_OK)), command)


# Historical name (tests and external callers import _node_fallback).
_node_fallback = _launcher_fallback


def _resolve_stdio_command(command: str, env: dict) -> tuple[str, dict]:
    """Resolve a stdio command against the exact subprocess env (bare launchers under a filtered PATH).

    A ``PATH`` lookup only runs when the child env actually carries one: ``shutil.which`` with
    ``path=None`` silently falls back to the PARENT's ``os.environ["PATH"]``, letting a command
    "resolve" against an env the child will never be spawned with. An absent child PATH is a
    miss; an explicitly empty one keeps its cwd-only meaning (same distinction the child's
    ``execvp`` will see). Bare ``npx``/``npm``/``node``/``uv``/``uvx`` still fall through to
    their explicit well-known install directories, everything else stays as-written for an
    honest spawn failure."""
    resolved_command = os.path.expanduser(str(command).strip())
    resolved_env = dict(env or {})
    if os.sep not in resolved_command:
        path_arg = resolved_env.get("PATH")
        which_hit = shutil.which(resolved_command, path=path_arg) if path_arg is not None else None
        if which_hit is None and sys.platform == "win32" and resolved_env:
            which_hit = _which_with_config_pathext(resolved_command, path_arg, resolved_env)
        if which_hit:
            resolved_command = which_hit
        elif resolved_command in {"npx", "npm", "node", "uv", "uvx"}:
            resolved_command = _launcher_fallback(resolved_command)
    command_dir = os.path.dirname(resolved_command)
    if command_dir:
        resolved_env = _prepend_path(resolved_env, command_dir)
    return resolved_command, resolved_env


def _npx_bin_candidates(bin_dir: str, name: str, *, windows: Optional[bool] = None) -> list:
    """Launcher paths to try for *name* inside an npx cache's ``.bin``, in order. On Windows that
    directory holds the extensionless sh script plus ``<name>.cmd``/``<name>.ps1``; the sh one
    cannot be spawned there and ``os.access(X_OK)`` is only an existence check, so select by
    extension (same precedence as ``hermes_constants._candidate_node_command_names``). ``windows``
    is injectable so the branch is testable without patching ``os.name`` process-wide."""
    is_windows = os.name == "nt" if windows is None else windows
    if is_windows:
        return [os.path.join(bin_dir, name + ext) for ext in (".cmd", ".exe")]
    return [os.path.join(bin_dir, name)]


def _npx_cached_bin(args: list) -> Optional[tuple]:
    """Resolve ``npx -y <pkg>`` to the already-installed binary, or None.

    ``npx`` resolves the package and then FORKS, staying resident as the real server's parent
    for nothing (~48 MB private memory per MCP server, measured); Hermes already supervises the
    child (shared death supervisor). When the package is in npx's cache we spawn its binary
    directly. Deliberately conservative — None (caller keeps plain ``npx``, so a cold machine
    still installs) for a cache miss, a version pin (``pkg@1.2.3``), extra npx flags, a manifest
    without one obvious bin, or any unreadable cache entry. Returns ``(binary_path, remaining_args)``."""
    if not isinstance(args, list) or not args:
        return None

    rest = list(args)
    while rest and rest[0] in ("-y", "--yes"):
        rest.pop(0)
    if not rest:
        return None
    # `npx pkg -y` (flag AFTER the spec) would hand the server a flag npx would have eaten.
    if any(str(a) in ("-y", "--yes") for a in rest[1:]):
        return None

    spec = str(rest[0])
    # Scoped names keep their leading '@', so only an '@' AFTER the scope is a version separator.
    if "@" in (spec[1:] if spec.startswith("@") else spec):
        return None
    if not spec or spec.startswith("-"):
        return None

    cache_root = os.environ.get("npm_config_cache") or os.path.join(os.path.expanduser("~"), ".npm")
    npx_root = os.path.join(cache_root, "_npx")
    if not os.path.isdir(npx_root):
        return None
    try:
        entries = os.listdir(npx_root)
    except OSError:
        return None

    for entry in entries:
        manifest = os.path.join(npx_root, entry, "package.json")
        try:
            with open(manifest, "r", encoding="utf-8-sig") as fh:
                deps = (json.load(fh) or {}).get("dependencies") or {}
        except (OSError, ValueError, TypeError):
            continue
        if spec not in deps:
            continue
        pkg_json = os.path.join(npx_root, entry, "node_modules", spec, "package.json")
        try:
            with open(pkg_json, "r", encoding="utf-8-sig") as fh:
                bin_field = (json.load(fh) or {}).get("bin")
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(bin_field, str):
            names = [os.path.basename(spec)]
        elif isinstance(bin_field, dict) and len(bin_field) == 1:
            names = list(bin_field.keys())
        else:
            continue  # zero or several bins: which one npx would pick is not ours to guess
        bin_dir = os.path.join(npx_root, entry, "node_modules", ".bin")
        for candidate in _npx_bin_candidates(bin_dir, names[0]):
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate, rest[1:]
    return None


def _interpolate_env_vars(value):
    """Recursively resolve ``${VAR}`` / Cursor ``${env:VAR}`` placeholders and context vars. Env
    refs resolve from the active profile's secret scope when multiplexing (the routed profile's
    value, not another profile's in ``os.environ``). Unset vars keep the literal placeholder."""
    from agent.secret_scope import get_secret as _get_secret
    if isinstance(value, str):
        def _replace(m):
            resolver = _CONTEXT_VAR_RESOLVERS.get(m.group(1).strip())
            return resolver() if resolver is not None else (_get_secret(_env_ref_name(m.group(1)), m.group(0)) or m.group(0))
        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


# (server_name, dotted key path) pairs already warned about: config loads repeat per discovery pass.
_whitespace_warned: Set[Tuple[str, str]] = set()


def _warn_hidden_whitespace(server_name: str, config: dict) -> List[str]:
    """Warn once per (server, key path) about string values with leading/trailing whitespace (a
    pasted newline causes opaque auth failures). Advisory only: values are never mutated (could be
    intentional) nor logged (often secrets). Returns flagged paths."""
    flagged: List[str] = []

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, str) and value != value.strip():
            flagged.append(path)
        elif isinstance(value, dict):
            for k, v in value.items():
                _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _walk(v, f"{path}[{i}]")
    _walk(config, "")
    for key_path in flagged:
        if (server_name, key_path) not in _whitespace_warned:
            _whitespace_warned.add((server_name, key_path))
            logger.warning(
                "MCP server '%s': config value '%s' has hidden leading or trailing whitespace — this often "
                "causes authentication or connection failures. Check for stray spaces/newlines in config.yaml "
                "(or the referenced env var).", server_name, key_path)
    return flagged


def _filter_suspicious_mcp_servers(servers: Dict[str, dict]) -> Dict[str, dict]:
    """Drop exfiltration-shaped MCP configs before any stdio spawn path."""
    try:
        from hermes_cli.mcp_security import validate_mcp_server_entry
    except Exception:
        return servers
    safe_servers = {}
    for name, cfg in servers.items():
        issues = validate_mcp_server_entry(name, cfg) if isinstance(cfg, dict) else None
        if issues:
            logger.warning("Skipping suspicious MCP server '%s': %s", name, "; ".join(issues))
        else:
            safe_servers[name] = cfg
    return safe_servers


def _portable_mcp_servers(safe_servers: Dict[str, dict]) -> None:
    """Merge plugin-provided (portable) MCP servers into *safe_servers*; native config wins on a clash. Never raises."""
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_manager
        discover_plugins()
        portable = get_plugin_manager().get_portable_mcp_servers()
        for name, cfg in _filter_suspicious_mcp_servers(portable).items():
            if name in safe_servers:
                logger.warning("Portable MCP server '%s' conflicts with native config; skipping", name)
            else:
                safe_servers[name] = dict(cfg)
    except Exception:
        logger.debug("Failed to load portable MCP servers", exc_info=True)


def _load_mcp_config() -> Dict[str, dict]:
    """``mcp_servers`` from config.yaml as ``{name: config}`` (empty on error / safe mode), ``${VAR}`` interpolated."""
    try:
        from hermes_cli.config import load_config
        from utils import env_var_enabled as _env_enabled
        if _env_enabled("HERMES_SAFE_MODE"):
            return {}
        servers = load_config().get("mcp_servers")
        try:  # ensure .env vars are available for interpolation
            from hermes_cli.env_loader import load_hermes_dotenv
            load_hermes_dotenv()
        except Exception:
            pass
        safe_servers: Dict[str, dict] = {}
        for name, cfg in _filter_suspicious_mcp_servers(servers if isinstance(servers, dict) else {}).items():
            interpolated = _interpolate_env_vars(cfg)
            if isinstance(interpolated, dict):
                _warn_hidden_whitespace(name, interpolated)
                safe_servers[name] = interpolated
        _portable_mcp_servers(safe_servers)
        return safe_servers
    except Exception as exc:
        logger.debug("Failed to load MCP config: %s", exc)
        return {}
