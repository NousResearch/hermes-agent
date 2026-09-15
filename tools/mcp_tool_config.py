"""MCP server config loading and stdio launch environment: ${VAR}/Cursor-style
interpolation, hidden-whitespace and suspicious-entry filtering, the filtered
subprocess env, command resolution, the cached-npx binary shortcut and the shared stderr log."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple
from hermes_constants import find_node_executable
from tools.mcp_tool_common import _env_ref_name, _prepend_path

logger = logging.getLogger("tools.mcp_tool")

_mcp_stderr_log_fh: Dict[str, Any] = {}  # profile home key -> handle
_mcp_stderr_log_lock = threading.Lock()
_NPM_CONFIG_CACHE_TIMEOUT_S = 2.0


def _get_mcp_stderr_log() -> Any:
    """Shared append-mode handle for MCP subprocess stderr, opened once per process PER PROFILE HOME (a
    multiplexed gateway's secondary profile must log under ITS ``logs/``, not the launch profile's). Must
    expose a real fd (asyncio wires the child's stderr to it); falls back to ``/dev/null``, then real stderr."""
    from hermes_constants import get_hermes_home, hermes_home_key
    home_key = hermes_home_key()
    with _mcp_stderr_log_lock:
        fh = _mcp_stderr_log_fh.get(home_key)
        if fh is None:
            try:
                log_dir = get_hermes_home() / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
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
    """``shutil.which`` retried under the config env's PATHEXT (Windows only; ``which`` uses the PARENT's)."""
    cfg_pathext = next((v for k, v in env.items() if k.upper() == "PATHEXT" and isinstance(v, str) and v.strip()), None)
    if not cfg_pathext or cfg_pathext == os.environ.get("PATHEXT"):
        return None
    saved = os.environ.get("PATHEXT")
    try:
        os.environ["PATHEXT"] = cfg_pathext
        return shutil.which(command, path=path_arg)
    finally:
        if saved is None:
            os.environ.pop("PATHEXT", None)
        else:
            os.environ["PATHEXT"] = saved


def _node_fallback(command: str) -> str:
    """Well-known Node install locations for bare ``npx``/``npm``/``node``; *command* unchanged when none exists."""
    home = os.path.expanduser("~")
    hermes_home = os.path.expanduser(os.getenv("HERMES_HOME", os.path.join(home, ".hermes")))
    # /usr/local/bin: canonical Node location (from-source Linux, Hermes Docker image, Intel Homebrew),
    # needed when a hand-authored env.PATH omits it — npx's shebang re-execs /usr/bin/env node.
    candidates = (os.path.join(hermes_home, "node", "bin", command), os.path.join(home, ".local", "bin", command),
                  os.path.join(os.sep, "usr", "local", "bin", command))
    return next((c for c in candidates if os.path.isfile(c) and os.access(c, os.X_OK)), command)


def _resolve_stdio_command(command: str, env: dict) -> tuple[str, dict]:
    """Resolve a stdio command against the exact subprocess env (bare ``npx``/``npm``/``node`` under a filtered PATH)."""
    resolved_command = os.path.expanduser(str(command).strip())
    resolved_env = dict(env or {})
    if os.sep not in resolved_command:
        path_arg = resolved_env.get("PATH")
        which_hit = shutil.which(resolved_command, path=path_arg)
        if which_hit is None and sys.platform == "win32" and resolved_env:
            which_hit = _which_with_config_pathext(resolved_command, path_arg, resolved_env)
        if which_hit:
            resolved_command = which_hit
        elif resolved_command in {"npx", "npm", "node"}:
            resolved_command = _node_fallback(resolved_command)
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


def _env_value(
    env: Mapping[str, str], name: str, *, case_insensitive: bool = False,
) -> Optional[str]:
    """Read one child-environment value using the platform or explicit key rules.

    ``_build_safe_env`` starts from the current process environment and then appends the
    server's explicit ``env`` mapping. A server can therefore legitimately override a
    baseline ``npm_config_cache`` using the conventional uppercase spelling. npm treats its
    ``npm_config_*`` variables case-insensitively on every host, while ordinary child
    environment lookups retain the platform's native behavior. Iterating all matching keys
    deliberately keeps the last mapping entry, matching that override.
    """
    if os.name != "nt" and not case_insensitive:
        return env.get(name)
    value = None
    folded_name = name.casefold()
    for key, candidate in env.items():
        if isinstance(key, str) and key.casefold() == folded_name:
            value = candidate
    return value


def _npx_sibling_npm_command(npx_command: str, env: dict) -> Optional[list[str]]:
    """Return a local npm CLI command paired with the resolved ``npx`` command.

    The cached-binary shortcut must use the same config source that ``npx`` would use.  On
    Windows, npm/npx are normally ``.cmd`` shims, which ``subprocess`` cannot execute directly;
    invoking the paired ``npm-cli.js`` through Node avoids a shell and console flash.  A layout we
    cannot identify is deliberately left to the original npx command.
    """
    npx_path = os.path.abspath(os.path.expanduser(str(npx_command)))
    npx_dir = os.path.dirname(npx_path)
    if not npx_dir or not os.path.isdir(npx_dir):
        return None

    if os.name == "nt":
        if os.path.splitext(npx_path)[1].lower() == ".exe":
            npm_exe = os.path.join(npx_dir, "npm.exe")
            return [npm_exe] if os.path.isfile(npm_exe) else None

        npm_cli = os.path.join(npx_dir, "node_modules", "npm", "bin", "npm-cli.js")
        if not os.path.isfile(npm_cli):
            return None
        # Prefer a Node launcher installed alongside npx.  If that layout omits
        # one, use Hermes's managed resolver rather than probing an arbitrary
        # PATH: the latter can select an unrelated system Node even while
        # Hermes owns the runtime that launches its MCP servers.
        for node_name in ("node.exe", "node.cmd", "node.bat", "node.com"):
            node = os.path.join(npx_dir, node_name)
            if os.path.isfile(node):
                return [node, npm_cli]

        managed_node = find_node_executable("node")
        if managed_node and os.path.isfile(managed_node):
            return [managed_node, npm_cli]
        return None

    npm = os.path.join(npx_dir, "npm")
    return [npm] if os.path.isfile(npm) else None


def _effective_npx_cache_env(
    npx_command: str, env: Optional[dict] = None, cwd: Optional[str] = None,
) -> Optional[dict]:
    """Return the child env with npm's effective cache pinned, or ``None`` to keep npx.

    An explicit ``npm_config_cache`` already wins.  Otherwise ask the npm installation paired
    with the resolved npx command for its local, user, and project ``.npmrc`` result.  This is an
    offline config read, not an install or registry operation.  If its result cannot be established
    within the small bound, the caller must retain npx instead of guessing a potentially stale
    platform-default cache.
    """
    cache_env = dict(os.environ if env is None else env)
    if _env_value(cache_env, "npm_config_cache", case_insensitive=True):
        return cache_env

    npm_command = _npx_sibling_npm_command(npx_command, cache_env)
    if npm_command is None:
        logger.debug("Unable to resolve npm paired with npx command %s; keeping npx", npx_command)
        return None

    try:
        from hermes_cli._subprocess_compat import windows_hide_flags

        result = subprocess.run(
            [*npm_command, "--silent", "--no-update-notifier", "--offline", "config", "get", "cache"],
            cwd=cwd,
            env=cache_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_NPM_CONFIG_CACHE_TIMEOUT_S,
            check=False,
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Unable to read npm cache configuration for %s: %s", npx_command, exc)
        return None

    configured_cache = result.stdout.strip()
    if result.returncode != 0 or not configured_cache or configured_cache.lower() in {"null", "undefined"}:
        logger.debug("npm cache configuration was unavailable for %s (exit %s)", npx_command, result.returncode)
        return None
    configured_cache = _resolve_npm_cache_path(configured_cache, cache_env, cwd)

    for key in list(cache_env):
        if isinstance(key, str) and key.casefold() == "npm_config_cache":
            del cache_env[key]
    cache_env["npm_config_cache"] = configured_cache
    return cache_env


def _resolve_npm_cache_path(configured_cache: str, env: Mapping[str, str], cwd: Optional[str]) -> str:
    """Resolve npm's cache path with the same child home and working directory.

    npm accepts ``~``-relative cache settings. ``os.path.expanduser`` reads Hermes's own process
    environment, which can differ from the stdio child's HOME/USERPROFILE, so expand a plain home
    alias from the exact child environment before handling genuinely relative paths.
    """
    cache_path = str(configured_cache)
    if cache_path == "~" or cache_path.startswith(("~/", "~\\")):
        home = _env_value(env, "HOME") or _env_value(env, "USERPROFILE")
        if home:
            cache_path = os.path.join(str(home), cache_path[2:])
        else:
            cache_path = os.path.expanduser(cache_path)
    else:
        cache_path = os.path.expanduser(cache_path)
    if not os.path.isabs(cache_path):
        cache_path = os.path.join(cwd or os.getcwd(), cache_path)
    return os.path.abspath(cache_path)


def _npx_cached_invocation(args: list) -> Optional[tuple[str, list]]:
    """Return a cache-safe npx package spec and server args, or ``None``.

    The direct launcher only implements ``npx -y <package> [server args]``. Any npx-specific
    flag, version pin, or ambiguous invocation stays with npx, including before configuration
    probing so a caller's ``--userconfig`` choice cannot be overwritten.
    """
    if not isinstance(args, list) or not args:
        return None

    rest = list(args)
    while rest and rest[0] in ("-y", "--yes"):
        rest.pop(0)
    if not rest:
        return None
    # `npx pkg -y` (flag AFTER the spec) would hand the server a flag npx would have eaten.
    if any(str(arg) in ("-y", "--yes") for arg in rest[1:]):
        return None

    spec = str(rest[0])
    # Scoped names keep their leading '@', so only an '@' AFTER the scope is a version separator.
    if not spec or spec.startswith("-") or "@" in (spec[1:] if spec.startswith("@") else spec):
        return None
    return spec, rest[1:]


def _npx_cached_bin(
    args: list,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> Optional[tuple]:
    """Resolve ``npx -y <pkg>`` to the already-installed binary, or None.

    ``npx`` resolves the package and then FORKS, staying resident as the real server's parent
    for nothing (~48 MB private memory per MCP server, measured); Hermes already supervises the
    child (shared death supervisor). When the package is in npx's cache we spawn its binary
    directly. Deliberately conservative — None (caller keeps plain ``npx``, so a cold machine
    still installs) for a cache miss, a version pin (``pkg@1.2.3``), extra npx flags, a manifest
    without one obvious bin, or any unreadable cache entry. ``env`` and ``cwd`` are
    the exact stdio-child settings when supplied, so a per-server
    ``npm_config_cache`` selects the same package cache that the original ``npx``
    command would use, including a relative configured cache path.
    Returns ``(binary_path, remaining_args)``."""
    invocation = _npx_cached_invocation(args)
    if invocation is None:
        return None
    spec, server_args = invocation

    cache_env = os.environ if env is None else env
    configured_cache = _env_value(cache_env, "npm_config_cache", case_insensitive=True)
    if configured_cache:
        cache_roots = [_resolve_npm_cache_path(configured_cache, cache_env, cwd)]
    elif os.name == "nt":
        # npm's Windows default is %LOCALAPPDATA%\npm-cache, not ~/.npm.
        # Falling back to ~/.npm keeps old/portable layouts working without
        # overriding an explicit npm_config_cache selected by the user.
        local_app_data = _env_value(cache_env, "LOCALAPPDATA")
        cache_roots = [os.path.join(local_app_data, "npm-cache")] if local_app_data else []
        cache_roots.append(os.path.join(os.path.expanduser("~"), ".npm"))
    else:
        cache_roots = [os.path.join(os.path.expanduser("~"), ".npm")]

    for cache_root in cache_roots:
        npx_root = os.path.join(cache_root, "_npx")
        if not os.path.isdir(npx_root):
            continue
        try:
            entries = os.listdir(npx_root)
        except OSError:
            continue

        for entry in entries:
            manifest = os.path.join(npx_root, entry, "package.json")
            try:
                with open(manifest, "r", encoding="utf-8") as fh:
                    deps = (json.load(fh) or {}).get("dependencies") or {}
            except (OSError, ValueError, TypeError):
                continue
            if spec not in deps:
                continue
            pkg_json = os.path.join(npx_root, entry, "node_modules", spec, "package.json")
            try:
                with open(pkg_json, "r", encoding="utf-8") as fh:
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
                    return candidate, server_args
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
