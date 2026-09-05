"""Terminal backend configuration: scope-aware TERMINAL_* reads, env-var parsing,
container-cwd sanity checks, plugin-backend classification, and the ``_quiet``
best-effort block shared by the terminal_tool_* modules.

Split out of ``tools/terminal_tool.py``; every public/patched name is re-imported there,
so ``tools.terminal_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

import logging
import json
import os
from contextlib import contextmanager
from typing import Any

# Log-record parity with the origin module.
logger = logging.getLogger("tools.terminal_tool")


@contextmanager
def _quiet(label: str, *args, exc=Exception, level: int = logging.DEBUG):
    """Best-effort block: swallow *exc*, log *label* (``%``-formatted with *args*)
    at *level* with the traceback. For janitor/advisory work that must never
    take the terminal tool down."""
    try:
        yield
    except exc:
        logger.log(level, label, *args, exc_info=True)


def _parse_env_var(name: str, default: str, converter: Any = int, type_label: str = "integer"):
    """Parse an env var with *converter*, raising a clear ValueError on bad
    values (e.g. TERMINAL_TIMEOUT=5m) instead of an opaque crash. TERMINAL_*
    names are read scope-aware via :func:`_tenv`."""
    raw = _tenv(name, default) if name.startswith("TERMINAL_") else os.getenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            f"Check ~/.hermes/.env or environment variables."
        )


def _safe_getcwd() -> str:
    """``os.getcwd()`` tolerant of a deleted cwd (FileNotFoundError) or a macOS
    TCC-protected one without Full Disk Access (PermissionError); falls back
    to TERMINAL_CWD, then the home directory."""
    try:
        return os.getcwd()
    except (FileNotFoundError, PermissionError):
        return _tenv("TERMINAL_CWD") or os.path.expanduser("~")


# Host-cwd prefixes that cannot exist inside a container sandbox (POSIX user
# dirs and Windows drive paths as they leak toward a Linux ``-w`` flag).
_HOST_CWD_PREFIXES = ("/Users/", "/home/", "C:\\", "C:/")

_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})
_BUILTIN_BACKENDS = _CONTAINER_BACKENDS | {"local", "ssh", "managed_modal"}


def _plugin_registry_lookup(env_type: str, fn_name: str, default, *args):
    """Call ``agent.terminal_env_registry.<fn_name>(env_type, *args)`` for a
    plugin backend. Fail-soft: *default* for built-in/empty backends, when the
    registry is unavailable, or when the provider raises — a misbehaving plugin
    must never take the terminal tool down."""
    if not env_type or env_type in _BUILTIN_BACKENDS:
        return default
    try:
        import agent.terminal_env_registry as reg

        return getattr(reg, fn_name)(env_type, *args)
    except Exception:
        return default


def _plugin_env_flag(env_type: str, attr: str, default=False):
    """Classification flag of a plugin-registered backend (fail-soft, see above)."""
    return _plugin_registry_lookup(env_type, "provider_flag", default, attr, default)


def _is_container_backend(env_type: str) -> bool:
    """True for built-in container backends and plugins declaring ``is_container``."""
    return env_type in _CONTAINER_BACKENDS or _plugin_env_flag(env_type, "is_container")


def _get_plugin_env_provider(env_type: str):
    """Return the registered plugin provider for *env_type*, or None."""
    return _plugin_registry_lookup(env_type, "get_provider", None)


def _is_unusable_container_cwd(cwd: str) -> bool:
    """True if *cwd* is a host or relative path that can't be a container
    workdir: ``docker run -w`` needs an absolute in-sandbox path, otherwise the
    container fails to start (exit 125). Windows drive paths aren't ``isabs``
    on POSIX, so they're caught by the prefix check."""
    return bool(cwd) and (cwd.startswith(_HOST_CWD_PREFIXES) or not os.path.isabs(cwd))


def _tenv(name: str, default: str = "") -> str:
    """Scope-aware read of a ``TERMINAL_*`` variable. Every terminal setting
    must go through this: under gateway multiplexing the active profile's
    config arrives via a per-turn scope, and a raw ``os.getenv`` would read
    whatever a previous turn pinned into the process env (cross-profile leak)."""
    from tools.terminal_scope import terminal_env

    return terminal_env(name, default)


def _tenv_bool(name: str, default: str) -> bool:
    """Scope-aware boolean ``TERMINAL_*`` read: true/1/yes (case-insensitive)."""
    return _tenv(name, default).lower() in {"true", "1", "yes"}
