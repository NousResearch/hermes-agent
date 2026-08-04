#!/usr/bin/env python3
"""
Terminal Tool Module

A terminal tool that executes commands in local, Docker, Modal, SSH,
Singularity, Daytona, and Vercel Sandbox environments. Supports local
execution, containerized backends, and cloud sandboxes, including managed
Modal mode.

Environment Selection (via TERMINAL_ENV environment variable):
- "local": Execute directly on the host machine (default, fastest)
- "docker": Execute in Docker containers (isolated, requires Docker)
- "modal": Execute in Modal cloud sandboxes (direct Modal or managed gateway)
- "vercel_sandbox": Execute in Vercel Sandbox cloud sandboxes

Features:
- Multiple execution backends (local, docker, modal, vercel_sandbox)
- Background task support
- VM/container lifecycle management
- Automatic cleanup after inactivity

Cloud sandbox note:
- Persistent filesystems preserve working state across sandbox recreation
- Persistent filesystems do NOT guarantee the same live sandbox or long-running processes survive cleanup, idle reaping, or Hermes exit

Usage:
    from terminal_tool import terminal_tool

    # Execute a simple command
    result = terminal_tool("ls -la")

    # Execute in background
    result = terminal_tool("python server.py", background=True)
"""

import importlib.util
import json
import logging
import os
import platform
import re
import shlex
import stat
import time
import threading
import atexit
import shutil
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Optional, Dict, Any, List, Hashable, Mapping

from utils import env_var_enabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global interrupt event: set by the agent when a user interrupt arrives.
# The terminal tool polls this during command execution so it can kill
# long-running subprocesses immediately instead of blocking until timeout.
# ---------------------------------------------------------------------------
from tools.interrupt import is_interrupted, _interrupt_event  # noqa: F401 — re-exported
from tools.registry import tool_error
# display_hermes_home imported lazily at call site (stale-module safety during hermes update)




# =============================================================================
# Custom Singularity Environment with more space
# =============================================================================

# Singularity helpers (scratch dir, SIF cache) now live in tools/environments/singularity.py
from tools.environments.singularity import _get_scratch_dir
from tools.tool_backend_helpers import (
    coerce_modal_mode,
    has_direct_modal_credentials,
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    resolve_modal_backend_state,
)


def _safe_parse_import_env(
    name: str,
    default: Any,
    converter,
    type_label: str,
):
    """Parse module-level numeric env vars without breaking import.

    Terminal tool is imported by CLI, ACP, tests, and tool discovery. A single
    malformed env var must not make the whole module unloadable at import time.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s: %r (expected %s). Falling back to %r.",
            name,
            raw,
            type_label,
            default,
        )
        return default


# Hard cap on foreground timeout; override via TERMINAL_MAX_FOREGROUND_TIMEOUT env var.
FOREGROUND_MAX_TIMEOUT = _safe_parse_import_env(
    "TERMINAL_MAX_FOREGROUND_TIMEOUT",
    600,
    int,
    "integer",
)

# Disk usage warning threshold (in GB)
DISK_USAGE_WARNING_THRESHOLD_GB = _safe_parse_import_env(
    "TERMINAL_DISK_WARNING_GB",
    500.0,
    float,
    "number",
)
_VERCEL_SANDBOX_DEFAULT_CWD = "/vercel/sandbox"
_SUPPORTED_VERCEL_RUNTIMES = ("node24", "node22", "python3.13")


def _is_supported_vercel_runtime(runtime: str) -> bool:
    return not runtime or runtime in _SUPPORTED_VERCEL_RUNTIMES


def _check_vercel_sandbox_requirements(config: dict[str, Any]) -> bool:
    """Validate Vercel Sandbox terminal backend requirements."""
    runtime = (config.get("vercel_runtime") or "").strip()
    if not _is_supported_vercel_runtime(runtime):
        supported = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
        logger.error(
            "Vercel Sandbox runtime %r is not supported. "
            "Set TERMINAL_VERCEL_RUNTIME to one of: %s.",
            runtime,
            supported,
        )
        return False

    disk = config.get("container_disk", 51200)
    if disk not in {0, 51200}:
        logger.error(
            "Vercel Sandbox does not support custom TERMINAL_CONTAINER_DISK=%s. "
            "Use the default shared setting (51200 MB).",
            disk,
        )
        return False

    if importlib.util.find_spec("vercel") is None:
        logger.error(
            "vercel is required for the Vercel Sandbox terminal backend: pip install vercel"
        )
        return False

    from agent.secret_scope import get_secret

    has_oidc = bool(get_secret("VERCEL_OIDC_TOKEN"))
    has_token = bool(get_secret("VERCEL_TOKEN"))
    has_project = bool(get_secret("VERCEL_PROJECT_ID"))
    has_team = bool(get_secret("VERCEL_TEAM_ID"))

    if has_oidc:
        return True

    if has_token or has_project or has_team:
        if has_token and has_project and has_team:
            return True
        logger.error(
            "Vercel Sandbox backend selected with token auth, but "
            "VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID must all "
            "be set together. VERCEL_OIDC_TOKEN is supported for one-off "
            "local development only."
        )
        return False

    logger.error(
        "Vercel Sandbox backend selected but no supported auth configuration "
        "was found. Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID "
        "for normal use. VERCEL_OIDC_TOKEN is supported for one-off local "
        "development only."
    )
    return False


# Cache for disk usage warning to avoid full rglob scan on every call.
# The check is advisory-only — staleness for up to 5 minutes is acceptable.
_disk_usage_cache: dict = {"timestamp": 0.0, "result": False}
_DISK_USAGE_CACHE_TTL = 300.0  # seconds


def _check_disk_usage_warning():
    """Check if total disk usage exceeds warning threshold.

    Result is cached for :data:`_DISK_USAGE_CACHE_TTL` seconds (default:
    5 minutes) to avoid an expensive recursive filesystem scan on every
    terminal command.  The check is advisory-only so a stale result is
    harmless.
    """
    import time as _time_mod
    now = _time_mod.monotonic()
    if now - _disk_usage_cache["timestamp"] < _DISK_USAGE_CACHE_TTL:
        return _disk_usage_cache["result"]
    try:
        scratch_dir = _get_scratch_dir()

        # Get total size of hermes directories
        total_bytes = 0
        import glob
        for path in glob.glob(str(scratch_dir / "hermes-*")):
            for f in Path(path).rglob('*'):
                if f.is_file():
                    try:
                        total_bytes += f.stat().st_size
                    except OSError as e:
                        logger.debug("Could not stat file %s: %s", f, e)
        
        total_gb = total_bytes / (1024 ** 3)
        
        exceeded = total_gb > DISK_USAGE_WARNING_THRESHOLD_GB
        if exceeded:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
        _disk_usage_cache["timestamp"] = _time_mod.monotonic()
        _disk_usage_cache["result"] = exceeded
        return exceeded
    except Exception as e:
        logger.debug("Disk usage warning check failed: %s", e, exc_info=True)
        # Don't update cache on error so the next call retries.
        return False


# Interactive sudo password cache.
#
# Scope the cache to the active session when a session key is available, then
# fall back to callback identity (ACP / CLI interactive callbacks), then the
# current thread. This prevents one interactive session from reusing another
# session's cached sudo password inside the same long-lived process.
_sudo_password_cache: dict[str, str] = {}
_sudo_password_cache_lock = threading.Lock()

# Optional UI callbacks for interactive prompts. When set, these are called
# instead of the default /dev/tty or input() readers. The CLI registers these
# so prompts route through prompt_toolkit's event loop.
# Callback slots used by the approval prompt and sudo password prompt
# routines. Stored in thread-local state so overlapping ACP sessions —
# each running in its own ThreadPoolExecutor thread — don't stomp on
# each other's callbacks. See GHSA-qg5c-hvr5-hjgr.
#
# CLI mode is single-threaded, so each thread (the only one) holds its
# own callback exactly like before. Gateway mode resolves approvals via
# the per-session queue in tools.approval, not through these callbacks,
# so it's unaffected.
_callback_tls = threading.local()


def _get_sudo_password_callback():
    return getattr(_callback_tls, "sudo_password", None)


def _get_approval_callback():
    return getattr(_callback_tls, "approval", None)


def set_sudo_password_callback(cb):
    """Register a callback for sudo password prompts (used by CLI).

    Per-thread scope — ACP sessions that run concurrently in a
    ThreadPoolExecutor each have their own callback slot.
    """
    _callback_tls.sudo_password = cb


def set_approval_callback(cb):
    """Register a callback for dangerous command approval prompts.

    Per-thread scope — ACP sessions that run concurrently in a
    ThreadPoolExecutor each have their own callback slot. See
    GHSA-qg5c-hvr5-hjgr.
    """
    _callback_tls.approval = cb


_sudo_execution_context: ContextVar[
    tuple[str, str, bool, Optional[str], str] | None
] = ContextVar(
    "hermes_sudo_execution_context", default=None,
)


@contextmanager
def _scoped_sudo_execution(
    target: str,
    backend: str,
    *,
    named: bool = False,
    sudo_password: Optional[str] = None,
    target_scope: str = "",
):
    token = _sudo_execution_context.set(
        (target, backend, named, sudo_password, target_scope),
    )
    try:
        yield
    finally:
        _sudo_execution_context.reset(token)


def _get_sudo_password_cache_scope(
    execution_target: str | None = None,
    execution_backend: str | None = None,
    execution_target_scope: str | None = None,
) -> str:
    """Return the session + target scope for interactive sudo passwords."""
    try:
        from gateway.session_context import get_session_env

        session_key = get_session_env("HERMES_SESSION_KEY", "")
    except Exception:
        session_key = os.getenv("HERMES_SESSION_KEY", "")
    if session_key:
        base_scope = f"session:{session_key}"
    else:
        callback = _get_sudo_password_callback()
        if callback is not None:
            owner = getattr(callback, "__self__", None)
            func = getattr(callback, "__func__", None)
            if owner is not None and func is not None:
                base_scope = f"callback-owner:{id(owner)}:{id(func)}"
            else:
                base_scope = f"callback:{id(callback)}"
        else:
            base_scope = f"thread:{threading.get_ident()}"

    active_context = _sudo_execution_context.get()
    if execution_target is None and active_context is not None:
        execution_target = active_context[0]
    if execution_backend is None and active_context is not None:
        execution_backend = active_context[1]
    if execution_target is None and execution_backend is None:
        return base_scope
    target_scope = execution_target_scope or ""
    if (
        execution_target_scope is None
        and active_context is not None
        and execution_target == active_context[0]
        and execution_backend == active_context[1]
    ):
        target_scope = active_context[4]
    return (
        f"{base_scope}|target:{execution_target!r}"
        f"|backend:{str(execution_backend or '').lower()}"
        f"|scope:{target_scope}"
    )


def _get_cached_sudo_password(
    execution_target: str | None = None,
    execution_backend: str | None = None,
    execution_target_scope: str | None = None,
) -> str:
    """Return the cached sudo password for the current target scope."""
    scope = _get_sudo_password_cache_scope(
        execution_target, execution_backend, execution_target_scope,
    )
    with _sudo_password_cache_lock:
        return _sudo_password_cache.get(scope, "")


def _set_cached_sudo_password(
    password: str,
    execution_target: str | None = None,
    execution_backend: str | None = None,
    execution_target_scope: str | None = None,
) -> None:
    """Persist a sudo password for the current target scope."""
    scope = _get_sudo_password_cache_scope(
        execution_target, execution_backend, execution_target_scope,
    )
    with _sudo_password_cache_lock:
        if password:
            _sudo_password_cache[scope] = password
        else:
            _sudo_password_cache.pop(scope, None)


def _reset_cached_sudo_passwords() -> None:
    """Clear all cached sudo passwords.

    Internal helper for tests and process teardown paths.
    """
    with _sudo_password_cache_lock:
        _sudo_password_cache.clear()

# =============================================================================
# Dangerous Command Approval System
# =============================================================================

# Dangerous command detection + approval now consolidated in tools/approval.py
from tools.approval import (
    check_all_command_guards as _check_all_guards_impl,
)


def _docker_volume_uses_host_path(volume_spec: str) -> bool:
    """Return True when a docker volume spec bind-mounts a host path."""
    if not isinstance(volume_spec, str):
        return False

    vol = volume_spec.strip()
    return bool(vol) and (
        vol.startswith(("/", "~", "./", "../")) or
        (len(vol) >= 3 and vol[1] == ":" and vol[2] in ("/", "\\"))
    )


def _docker_has_host_access(config: Dict[str, Any]) -> bool:
    """Return True when a Docker sandbox exposes host paths through bind mounts."""
    if config.get("env_type") != "docker":
        return False
    if config.get("host_cwd") and config.get("docker_mount_cwd_to_workspace"):
        return True
    return any(_docker_volume_uses_host_path(vol) for vol in config.get("docker_volumes", []))


def _check_all_guards(command: str, env_type: str,
                      has_host_access: bool = False,
                      execution_target: str = "default",
                      execution_backend: Optional[str] = None,
                      execution_target_named: bool = False,
                      execution_target_scope: str = "") -> dict:
    """Delegate to consolidated guard (tirith + dangerous cmd) with CLI callback."""
    return _check_all_guards_impl(command, env_type,
                                  approval_callback=_get_approval_callback(),
                                  has_host_access=has_host_access,
                                  execution_target=execution_target,
                                  execution_backend=execution_backend or env_type,
                                  execution_target_named=execution_target_named,
                                  execution_target_scope=execution_target_scope)


# Allowlist: characters that can legitimately appear in directory paths.
# Covers Unicode letters/digits, path separators, Windows drive/UNC separators,
# tilde, dot, hyphen, underscore, space, plus, at, equals, and comma.  Shell
# metacharacters remain rejected.  This intentionally fixes the old ASCII-only
# guard that blocked perfectly normal workdirs such as Chinese Obsidian vault
# paths while preserving the injection boundary around command execution
# (the cwd is additionally shlex-quoted before it reaches the shell; this
# allowlist is defense-in-depth).
_WORKDIR_SAFE_ASCII_CHARS = frozenset('/\\:_-.~ +@=,')


def _is_safe_workdir_char(ch: str) -> bool:
    if not ch:
        return False
    # Reject control characters (including newlines/tabs) and NUL bytes before
    # considering Unicode categories.
    if ord(ch) < 32 or ord(ch) == 127:
        return False
    return ch.isalnum() or ch in _WORKDIR_SAFE_ASCII_CHARS


def _validate_workdir(workdir: str) -> str | None:
    """Reject workdir values that don't look like a filesystem path.

    Uses an allowlist of safe characters rather than a deny-list, so novel
    shell metacharacters can't slip through.

    Returns None if safe, or an error message string if dangerous.
    """
    if not workdir:
        return None
    for ch in workdir:
        if not _is_safe_workdir_char(ch):
            return (
                f"Blocked: workdir contains disallowed character {repr(ch)}. "
                "Use a simple filesystem path without shell metacharacters."
            )
    return None


def _handle_sudo_failure(output: str, env_type: str) -> str:
    """
    Check for sudo failure and add helpful message for messaging contexts.
    
    Returns enhanced output if sudo failed in messaging context, else original.
    """
    is_gateway = env_var_enabled("HERMES_GATEWAY_SESSION")
    
    if not is_gateway:
        return output
    
    # Check for sudo failure indicators
    sudo_failures = [
        "sudo: a password is required",
        "sudo: no tty present",
        "sudo: a terminal is required",
    ]
    
    for failure in sudo_failures:
        if failure in output:
            from hermes_constants import display_hermes_home as _dhh
            return output + f"\n\n💡 Tip: To enable sudo over messaging, add SUDO_PASSWORD to {_dhh()}/.env on the agent machine."
    
    return output


# sudo -S rejects a bad cached/interactive password with these messages.
_SUDO_WRONG_PASSWORD_MARKERS = (
    "sudo: authentication failed",
    "sudo: incorrect password attempt",
    "sudo: maximum 3 incorrect authentication attempts",
    "sudo: 3 incorrect password attempts",
)


def _sudo_wrong_password_failure(output: str) -> bool:
    """Return True when sudo rejected a piped password."""
    if not output:
        return False
    lowered = output.lower()
    return any(marker in lowered for marker in _SUDO_WRONG_PASSWORD_MARKERS)


def _invalidate_cached_sudo_on_auth_failure(
    command: str | None,
    output: str,
    execution_target: str | None = None,
    execution_backend: str | None = None,
    execution_target_scope: str | None = None,
) -> bool:
    """Drop a session-cached sudo password after sudo rejects it.

    Env-configured ``SUDO_PASSWORD`` is left alone — that is an explicit
    operator choice, not an interactive cache entry.
    """
    if "SUDO_PASSWORD" in os.environ:
        return False
    if not _sudo_wrong_password_failure(output):
        return False
    if _count_real_sudo_invocations(command or "") == 0:
        return False
    if not _get_cached_sudo_password(
        execution_target, execution_backend, execution_target_scope,
    ):
        return False
    _set_cached_sudo_password(
        "", execution_target, execution_backend, execution_target_scope,
    )
    return True


def _prompt_for_sudo_password(timeout_seconds: int = 45) -> str:
    """
    Prompt user for sudo password with timeout.
    
    Returns the password if entered, or empty string if:
    - User presses Enter without input (skip)
    - Timeout expires (45s default)
    - Any error occurs
    
    Only works in interactive mode (HERMES_INTERACTIVE=1).
    If a _sudo_password_callback is registered (by the CLI), delegates to it
    so the prompt integrates with prompt_toolkit's UI.  Otherwise reads
    directly from /dev/tty with echo disabled.
    """
    import sys
    
    # Use the registered callback when available (prompt_toolkit-compatible)
    _sudo_cb = _get_sudo_password_callback()
    if _sudo_cb is not None:
        try:
            return _sudo_cb() or ""
        except Exception:
            return ""

    result = {"password": None, "done": False}
    
    def read_password_thread():
        """Read password with echo disabled. Uses msvcrt on Windows, /dev/tty on Unix."""
        tty_fd = None
        old_attrs = None
        try:
            if platform.system() == "Windows":
                import msvcrt
                chars = []
                while True:
                    c = msvcrt.getwch()
                    if c in {"\r", "\n"}:
                        break
                    if c == "\x03":
                        raise KeyboardInterrupt
                    chars.append(c)
                result["password"] = "".join(chars)
            else:
                import termios
                tty_fd = os.open("/dev/tty", os.O_RDONLY)
                old_attrs = termios.tcgetattr(tty_fd)
                new_attrs = termios.tcgetattr(tty_fd)
                new_attrs[3] = new_attrs[3] & ~termios.ECHO
                termios.tcsetattr(tty_fd, termios.TCSAFLUSH, new_attrs)
                chars = []
                while True:
                    b = os.read(tty_fd, 1)
                    if not b or b in {b"\n", b"\r"}:
                        break
                    chars.append(b)
                result["password"] = b"".join(chars).decode("utf-8", errors="replace")
        except (EOFError, KeyboardInterrupt, OSError):
            result["password"] = ""
        except Exception:
            result["password"] = ""
        finally:
            if tty_fd is not None and old_attrs is not None:
                try:
                    import termios as _termios
                    _termios.tcsetattr(tty_fd, _termios.TCSAFLUSH, old_attrs)
                except Exception as e:
                    logger.debug("Failed to restore terminal attributes: %s", e)
            if tty_fd is not None:
                try:
                    os.close(tty_fd)
                except Exception as e:
                    logger.debug("Failed to close tty fd: %s", e)
            result["done"] = True
    
    try:
        os.environ["HERMES_SPINNER_PAUSE"] = "1"
        time.sleep(0.2)
        
        print()
        print("┌" + "─" * 58 + "┐")
        print("│  🔐 SUDO PASSWORD REQUIRED" + " " * 30 + "│")
        print("├" + "─" * 58 + "┤")
        print("│  Enter password below (input is hidden), or:            │")
        print("│    • Press Enter to skip (command fails gracefully)     │")
        print(f"│    • Wait {timeout_seconds}s to auto-skip" + " " * 27 + "│")
        print("└" + "─" * 58 + "┘")
        print()
        print("  Password (hidden): ", end="", flush=True)
        
        password_thread = threading.Thread(target=read_password_thread, daemon=True)
        password_thread.start()
        password_thread.join(timeout=timeout_seconds)
        
        if result["done"]:
            password = result["password"] or ""
            print()  # newline after hidden input
            if password:
                print("  ✓ Password received (cached for this session)")
            else:
                print("  ⏭ Skipped - continuing without sudo")
            print()
            sys.stdout.flush()
            return password
        else:
            print("\n  ⏱ Timeout - continuing without sudo")
            print("    (Press Enter to dismiss)")
            print()
            sys.stdout.flush()
            return ""
            
    except (EOFError, KeyboardInterrupt):
        print()
        print("  ⏭ Cancelled - continuing without sudo")
        print()
        sys.stdout.flush()
        return ""
    except Exception as e:
        print(f"\n  [sudo prompt error: {e}] - continuing without sudo\n")
        sys.stdout.flush()
        return ""
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]

def _safe_command_preview(command: Any, limit: int = 200) -> str:
    """Return a log-safe preview for possibly-invalid command values."""
    if command is None:
        return "<None>"
    if isinstance(command, str):
        return command[:limit]
    try:
        return repr(command)[:limit]
    except Exception:
        return f"<{type(command).__name__}>"

def _looks_like_env_assignment(token: str) -> bool:
    """Return True when *token* is a leading shell environment assignment."""
    if "=" not in token or token.startswith("="):
        return False
    name, _value = token.split("=", 1)
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def _read_shell_token(command: str, start: int) -> tuple[str, int]:
    """Read one shell token, preserving quotes/escapes, starting at *start*."""
    i = start
    n = len(command)

    while i < n:
        ch = command[i]
        if ch.isspace() or ch in ";|&()":
            break
        if ch == "'":
            i += 1
            while i < n and command[i] != "'":
                i += 1
            if i < n:
                i += 1
            continue
        if ch == '"':
            i += 1
            while i < n:
                inner = command[i]
                if inner == "\\" and i + 1 < n:
                    i += 2
                    continue
                if inner == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        i += 1

    return command[start:i], i


def _rewrite_real_sudo_invocations(command: str) -> tuple[str, int]:
    """Rewrite only real unquoted sudo command words, not plain text mentions.

    Returns the rewritten command and the number of sudo invocations rewritten.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    command_start = True
    sudo_count = 0

    while i < n:
        ch = command[i]

        if ch.isspace():
            out.append(ch)
            if ch == "\n":
                command_start = True
            i += 1
            continue

        if ch == "#" and command_start:
            comment_end = command.find("\n", i)
            if comment_end == -1:
                out.append(command[i:])
                break
            out.append(command[i:comment_end])
            i = comment_end
            continue

        if command.startswith("&&", i) or command.startswith("||", i) or command.startswith(";;", i):
            out.append(command[i:i + 2])
            i += 2
            command_start = True
            continue

        if ch in ";|&(":
            out.append(ch)
            i += 1
            command_start = True
            continue

        if ch == ")":
            out.append(ch)
            i += 1
            command_start = False
            continue

        token, next_i = _read_shell_token(command, i)
        if command_start and token == "sudo":
            out.append("sudo -S -p ''")
            sudo_count += 1
        else:
            out.append(token)

        if command_start and _looks_like_env_assignment(token):
            command_start = True
        else:
            command_start = False
        i = next_i

    return "".join(out), sudo_count


def _count_real_sudo_invocations(command: str) -> int:
    """Return how many real sudo command words appear in *command*.

    Lightweight scan that reuses the same tokeniser as
    ``_rewrite_real_sudo_invocations`` but skips the string-building, so it
    is cheap to call from the result-processing path.
    """
    count = 0
    i = 0
    n = len(command)
    command_start = True

    while i < n:
        ch = command[i]

        if ch.isspace():
            if ch == "\n":
                command_start = True
            i += 1
            continue

        if ch == "#" and command_start:
            comment_end = command.find("\n", i)
            if comment_end == -1:
                break
            i = comment_end
            continue

        if command.startswith("&&", i) or command.startswith("||", i) or command.startswith(";;", i):
            i += 2
            command_start = True
            continue

        if ch in ";|&(":
            i += 1
            command_start = True
            continue

        if ch == ")":
            i += 1
            command_start = False
            continue

        token, next_i = _read_shell_token(command, i)
        if command_start and token == "sudo":
            count += 1

        if command_start and _looks_like_env_assignment(token):
            command_start = True
        else:
            command_start = False
        i = next_i

    return count


def _sudo_nopasswd_works() -> bool:
    """Return True when local sudo currently works without prompting.

    Only probes for the `local` terminal backend; Docker/SSH/Modal/etc. must
    not inherit the host's sudo state. Re-probes every call (no process-level
    cache) so an expired sudo timestamp cannot make a later command silently
    block waiting for a password.
    """
    active_context = _sudo_execution_context.get()
    if active_context is not None:
        terminal_env = active_context[1].strip().lower() or "local"
    else:
        terminal_env = os.getenv("TERMINAL_ENV", "local").strip().lower() or "local"
    if terminal_env != "local":
        return False

    try:
        probe = subprocess.run(
            ["sudo", "-n", "true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        return probe.returncode == 0
    except Exception:
        return False


def _rewrite_compound_background(command: str) -> str:
    """Wrap `A && B &` (or `A || B &`) to `A && { B & }` at depth 0.

    Bash parses ``A && B &`` with `&&` tighter than `&`, so it forks a
    subshell for the whole `A && B` compound and backgrounds it. Inside
    the subshell, `B` runs foreground, so the subshell waits for `B` to
    finish. When `B` is a long-running process (`python3 -m http.server`,
    `yes > /dev/null`, anything that doesn't naturally exit), the subshell
    never exits. It leaks as a process stuck in ``wait4`` forever — and
    on the way, its open stdout pipe can prevent the terminal tool from
    returning promptly.

    Rewriting the tail to `A && { B & }` preserves `&&`'s error semantics
    (skip B if A fails) while replacing the subshell with a brace group.
    The brace group runs in the current shell (no fork), backgrounds B as
    a simple command (bash doesn't wait for it in non-interactive mode),
    and exits immediately. B runs as a normal backgrounded child, orphaned
    when the parent shell exits.

    Handles redirects (``&>``, ``2>&1``) and skips content inside quoted
    strings and parenthesised subshells. Leaves simple ``cmd &`` alone —
    that construct doesn't have the subshell-wait bug.
    """
    n = len(command)
    i = 0
    paren_depth = 0
    brace_depth = 0
    # Position in *command* just after the most recent `&&` / `||` at depth 0
    # in the current statement; -1 when no chain operator is active.
    last_chain_op_end = -1
    rewrites: list[tuple[int, int]] = []  # (chain_op_end, amp_pos)

    while i < n:
        ch = command[i]

        # Newline terminates a statement at depth 0 — reset chain state.
        # Checked before the whitespace skip so we don't miss it.
        if ch == "\n" and paren_depth == 0 and brace_depth == 0:
            last_chain_op_end = -1
            i += 1
            continue

        if ch.isspace():
            i += 1
            continue

        # Comments (only at statement start — conservative: any `#` not inside
        # a token ends the line). `_read_shell_token` handles quoted strings
        # below so `#` inside quotes is safe.
        if ch == "#":
            nl = command.find("\n", i)
            if nl == -1:
                break
            i = nl
            continue

        if ch == "\\" and i + 1 < n:
            i += 2
            continue

        # Quoted tokens — consume whole string via the shared tokenizer.
        if ch in {"'", '"'}:
            _, next_i = _read_shell_token(command, i)
            i = max(next_i, i + 1)
            continue

        if ch == "(":
            paren_depth += 1
            i += 1
            continue

        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue

        # Brace groups: `{ ... }` is a group (no subshell fork), and bash
        # requires whitespace after `{`. We track depth so already-rewritten
        # output (`A && { B & }`) is idempotent — the inner `&` is part of
        # the group, not a new compound to rewrite. Also skip content inside
        # the group since `A && B &` there is separately well-formed.
        if ch == "{" and i + 1 < n and (command[i + 1].isspace() or command[i + 1] == "\n"):
            brace_depth += 1
            i += 1
            continue
        if ch == "}" and brace_depth > 0:
            brace_depth -= 1
            # Closing a group completes a compound statement; reset chain.
            last_chain_op_end = -1
            i += 1
            continue

        # Inside parens or brace groups, skip operators — they parse in their
        # own scope. `(...)` subshells have the same bug class but are not the
        # common agent pattern; leave for a follow-up.
        if paren_depth > 0 or brace_depth > 0:
            i += 1
            continue

        # Chain operators at depth 0
        if command.startswith("&&", i) or command.startswith("||", i):
            last_chain_op_end = i + 2
            i += 2
            continue

        # Statement terminators reset the chain state
        if ch == ";":
            last_chain_op_end = -1
            i += 1
            continue

        # Single `|` (pipe) starts a new pipeline stage; don't rewrite
        # across it. `||` handled above.
        if ch == "|":
            last_chain_op_end = -1
            i += 1
            continue

        # `&` handling: distinguish `&&`, `&>`, fd redirect (`>&`, `<&`),
        # and a true backgrounding `&`.
        if ch == "&":
            # `&&` handled above; won't reach here
            if i + 1 < n and command[i + 1] == ">":
                # `&>` redirect — consume
                i += 2
                continue
            # `>&` / `<&` fd target — look back past whitespace
            j = i - 1
            while j >= 0 and command[j].isspace():
                j -= 1
            if j >= 0 and command[j] in "<>":
                i += 1
                continue
            # Real background operator
            if last_chain_op_end >= 0:
                rewrites.append((last_chain_op_end, i))
            last_chain_op_end = -1
            i += 1
            continue

        # Regular unquoted token — advance past it via the shared tokenizer
        _, next_i = _read_shell_token(command, i)
        i = max(next_i, i + 1)

    if not rewrites:
        return command

    # Apply rewrites back-to-front so earlier indices remain valid.
    result = command
    for chain_end, amp_pos in reversed(rewrites):
        # Skip whitespace right after the `&&`/`||` so the brace group
        # opens flush against the inner command.
        insert_pos = chain_end
        while insert_pos < amp_pos and result[insert_pos].isspace():
            insert_pos += 1
        prefix = result[:insert_pos]
        middle = result[insert_pos:amp_pos]  # inner command + trailing space
        suffix = result[amp_pos + 1 :]
        # `{` needs a trailing space in bash; the closing `}` needs to be
        # preceded by `;` or `&` — we're providing `&` from the backgrounding.
        result = prefix + "{ " + middle + "& }" + suffix

    return result


def _transform_sudo_command(command: str | None) -> tuple[str | None, str | None]:
    """
    Transform sudo commands to use -S flag if SUDO_PASSWORD is available.

    This is a shared helper used by all execution environments to provide
    consistent sudo handling across local, SSH, and container environments.

    Returns:
        (transformed_command, sudo_stdin) where:
        - transformed_command has every bare ``sudo`` replaced with
          ``sudo -S -p ''`` so sudo reads its password from stdin.
        - sudo_stdin is the password string with a trailing newline that the
          caller must prepend to the process's stdin stream.  sudo -S reads
          exactly one line (the password) and passes the rest of stdin to the
          child command, so prepending is safe even when the caller also has
          its own stdin_data to pipe.
        - If no password is available, sudo_stdin is None and the command is
          returned unchanged so it fails gracefully with
          "sudo: a password is required".

    Callers that drive a subprocess directly (local, ssh, docker, singularity)
    should prepend sudo_stdin to their stdin_data and pass the merged bytes to
    Popen's stdin pipe.

    Callers that cannot pipe subprocess stdin (modal, daytona,
    vercel_sandbox) must embed the password in the command string
    themselves; see their execute() methods for how they handle the
    non-None sudo_stdin case.

    If SUDO_PASSWORD is not set and an interactive UI is available
    (HERMES_INTERACTIVE=1 or a registered sudo password callback):
      Prompts user for password with 45s timeout, caches for session.

    If SUDO_PASSWORD is not set and NOT interactive:
      Command runs as-is (fails gracefully with "sudo: a password is required").
    """
    if command is None:
        return None, None
    transformed, sudo_count = _rewrite_real_sudo_invocations(command)
    if sudo_count == 0:
        return command, None

    active_context = _sudo_execution_context.get()
    if active_context is not None and active_context[2]:
        configured_password = active_context[3]
        has_configured_password = configured_password is not None
        sudo_password = str(configured_password) if has_configured_password else _get_cached_sudo_password()
    else:
        try:
            from agent.secret_scope import UnscopedSecretError, get_secret
            try:
                configured_password = get_secret("SUDO_PASSWORD")
            except UnscopedSecretError:
                configured_password = os.environ.get("SUDO_PASSWORD")
        except Exception:
            configured_password = os.environ.get("SUDO_PASSWORD")
        has_configured_password = configured_password is not None
        sudo_password = configured_password if has_configured_password else _get_cached_sudo_password()

    # Local hosts with sudoers NOPASSWD should not be forced through the
    # interactive Hermes password prompt or the sudo -S password-pipe path.
    # Scoped to the local terminal backend so Docker/SSH/Modal/etc. can't
    # inherit host sudo state. Re-probes every call (no process-lifetime
    # cache) so an expired sudo timestamp doesn't make a later command block
    # silently without Hermes prompting.
    if not has_configured_password and not sudo_password and _sudo_nopasswd_works():
        return command, None

    has_sudo_prompt_callback = _get_sudo_password_callback() is not None
    should_prompt_for_sudo = (
        env_var_enabled("HERMES_INTERACTIVE") or has_sudo_prompt_callback
    )
    if not has_configured_password and not sudo_password and should_prompt_for_sudo:
        sudo_password = _prompt_for_sudo_password(timeout_seconds=45)
        if sudo_password:
            _set_cached_sudo_password(sudo_password)

    if has_configured_password or sudo_password:
        # Trailing newline is required: sudo -S reads one line per invocation.
        # Compound commands (`sudo a && sudo b`) need one password line each.
        password_line = sudo_password + "\n"
        return transformed, password_line * sudo_count

    return command, None


# Environment classes now live in tools/environments/
from tools.environments.local import LocalEnvironment as _LocalEnvironment
from tools.environments.singularity import SingularityEnvironment as _SingularityEnvironment
from tools.environments.ssh import SSHEnvironment as _SSHEnvironment
from tools.environments.docker import DockerEnvironment as _DockerEnvironment
from tools.environments.modal import ModalEnvironment as _ModalEnvironment
from tools.environments.managed_modal import ManagedModalEnvironment as _ManagedModalEnvironment
from tools.managed_tool_gateway import is_managed_tool_gateway_ready
import sys


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands on the selected execution target (using bash/Linux shell semantics). Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail (use read_file), grep/rg/find/ls (use search_files), sed/awk (use patch), or echo/heredoc file creation (use write_file). Reserve terminal for: builds, installs, git, processes, scripts, network, package managers, and anything that needs a shell.
Environment state persists: activate a virtualenv or export variables once per session, not before every command.

Foreground (default): returns INSTANTLY when the command finishes, even with a high timeout — set timeout generously for long builds.
Background: set background=true (returns a session_id). Pair with notify_on_complete=true for bounded tasks; leave silent only for servers/daemons that never exit. Never use nohup/setsid/trailing '&' — use background=true so Hermes tracks the process. After starting a server, verify readiness with a health check, then act in a separate call; no blind sleep loops. Manage with process(action="poll"/"wait").
Working directory: use 'workdir' for per-command cwd. When a command changes the session cwd (cd, pushd), the result includes a "cwd" field — trust it instead of prefixing every command with 'cd'.
PTY: set pty=true for interactive CLIs (they hang without it). Pipe git output to cat if it might page.
"""

# Global state for environment lifecycle management
_active_environments: Dict[Hashable, Any] = {}
_last_activity: Dict[Hashable, float] = {}
_env_lock = threading.Lock()
_retired_environments: list[tuple[Hashable, Any, float]] = []
_retired_environments_lock = threading.Lock()
_creation_locks: Dict[Hashable, threading.Lock] = {}  # Per-target locks for sandbox creation
_creation_locks_lock = threading.Lock()  # Protects _creation_locks dict itself
_cleanup_thread = None
_cleanup_running = False

# Once-per-process guard for the docker orphan reaper (issue #20561).
# Set when _maybe_reap_docker_orphans first runs; concurrent _create_environment
# calls for parallel subagents won't re-trigger the sweep.
_docker_orphan_reaper_ran = False
_docker_orphan_reaper_lock = threading.Lock()


def _maybe_reap_docker_orphans(container_config: Dict[str, Any]) -> None:
    """Run the docker orphan reaper once per process, if enabled.

    Sweeps long-Exited containers labeled ``hermes-agent=1`` for the current
    profile that match the issue #20561 leak class — containers left behind
    by Hermes processes that exited without firing ``atexit`` (SIGKILL,
    OOM, terminal-window-close). The reaper is conservative by default:
    only Exited containers older than ``2 × lifetime_seconds`` and scoped to
    the current profile.

    Gates:

    * ``terminal.docker_orphan_reaper: false`` disables it entirely (the
      operator opted out — usually because they're running multiple
      Hermes processes in the same profile and don't trust the
      conservative defaults).
    * ``_docker_orphan_reaper_ran`` flag — sweep runs once per Python
      interpreter, not on every subagent / RL-rollout / parallel
      ``terminal()`` call.
    """
    global _docker_orphan_reaper_ran
    if not container_config.get("docker_orphan_reaper", True):
        return
    # Cheap double-checked-locking: read without the lock, take the lock
    # only on first run, recheck inside.
    if _docker_orphan_reaper_ran:
        return
    with _docker_orphan_reaper_lock:
        if _docker_orphan_reaper_ran:
            return
        _docker_orphan_reaper_ran = True

    # 2 × the longest configured Docker-target lifetime gives every named
    # sibling a conservative grace window. A process-wide reaper runs only
    # once, so using the first-created target's value could reap a longer-lived
    # target prematurely.
    try:
        lifetime = int(container_config.get(
            "lifetime_seconds", os.getenv("TERMINAL_LIFETIME_SECONDS", "300"),
        ))
    except (TypeError, ValueError):
        lifetime = 300
    try:
        from tools.execution_targets import list_execution_targets

        targets = list_execution_targets()
        if targets and targets[0].named:
            for target in targets:
                if target.backend != "docker":
                    continue
                try:
                    lifetime = max(
                        lifetime, int(target.config.get("lifetime_seconds", 300)),
                    )
                except (TypeError, ValueError):
                    continue
    except Exception:
        logger.debug("Could not resolve Docker target lifetimes", exc_info=True)
    lifetime = max(60, lifetime)
    max_age = lifetime * 2

    try:
        from tools.environments.docker import (
            reap_orphan_containers, _get_active_profile_name,
        )
    except ImportError:
        return
    try:
        profile = _get_active_profile_name()
        removed = reap_orphan_containers(
            max_age_seconds=max_age, profile_filter=profile,
        )
        if removed:
            logger.info(
                "Docker orphan reaper removed %d stale container(s) for profile %s",
                removed, profile,
            )
    except Exception as e:
        # Never fail the env-creation path because of a janitor problem.
        logger.debug("Docker orphan reaper raised: %s", e)


# Per-task environment overrides registry.
# Allows environments (e.g., TerminalBench2Env) to specify a custom Docker/Modal
# image for a specific task_id BEFORE the agent loop starts. When the terminal or
# file tools create a new sandbox for that task_id, they check this registry first
# and fall back to the TERMINAL_MODAL_IMAGE (etc.) env var if no override is set.
#
# This is never exposed to the model -- only infrastructure code calls it.
# Thread-safe because each task_id is unique per rollout.
_task_env_overrides: Dict[Hashable, Dict[str, Any]] = {}
_task_env_overrides_lock = threading.Lock()
_active_turn_counts: Dict[Hashable, int] = {}
_active_turn_counts_lock = threading.RLock()
_deferred_environment_cleanups: Dict[Hashable, Hashable] = {}


# ── Per-session cwd records (cwd rearchitecture, step 1) ────────────────────
#
# The durable source of truth for "which directory is THIS session working
# in". Keyed by the raw session/task key (NOT the collapsed container id):
# the terminal env is shared across sessions, so any cwd state stored on the
# env is a global mutable timeshared between sessions — the root cause of the
# wrong-worktree bug class (env.cwd_owner stamping, _last_known_cwd, and the
# ownership ladder in file_tools are all patches over that misplacement).
#
# Step 1 (this change): dual-write only. Every site that learns a session's
# live cwd (post-command tracking, cwd-override registration) also records it
# here. Readers still use the legacy env.cwd ladder. Later steps flip
# file_tools and _resolve_command_cwd to read this store, then delete the
# env-side tracking + ownership guards.
_session_cwd: Dict[Hashable, str] = {}
_session_cwd_specs: Dict[Hashable, str] = {}
_session_cwd_lock = threading.Lock()


def _target_resolution(target=None):
    from tools.execution_targets import resolve_execution_target

    return resolve_execution_target(target)


def _environment_scope_key(task_key: Hashable, resolution) -> Hashable:
    return resolution.environment_key(task_key)


def _profile_scoped_task_key(task_key: Hashable) -> Hashable:
    try:
        return _target_resolution(None).scope_task_key(task_key)
    except Exception:
        return task_key


def _turn_scope_key(task_id: Hashable) -> Hashable:
    collapsed = _resolve_container_task_id(str(task_id))
    return _profile_scoped_task_key(collapsed)


def _run_deferred_environment_cleanup(task_id: Hashable) -> None:
    try:
        cleanup_vm(
            task_id,
            preserve_persistent=True,
            include_collapsed=True,
        )
    except Exception:
        logger.warning(
            "Deferred environment cleanup failed for task %s",
            task_id,
            exc_info=True,
        )


def _turn_keys_overlap(left: Hashable, right: Hashable) -> bool:
    if left == right:
        return True
    if isinstance(left, tuple) and left and left[0] == right:
        return True
    if isinstance(right, tuple) and right and right[0] == left:
        return True
    return False


def _related_active_turns_unlocked(environment_key: Hashable) -> int:
    return sum(
        count for key, count in _active_turn_counts.items()
        if _turn_keys_overlap(key, environment_key)
    )


def _register_environment_turn_key(key: Hashable) -> Hashable:
    with _active_turn_counts_lock:
        _active_turn_counts[key] = _active_turn_counts.get(key, 0) + 1
    return key


def register_environment_turn(task_id: Hashable) -> Hashable:
    return _register_environment_turn_key(_turn_scope_key(task_id))


def _release_environment_turn_key(key: Hashable) -> int:
    deferred_task_ids = []
    with _active_turn_counts_lock:
        current = _active_turn_counts.get(key, 0)
        if current <= 1:
            _active_turn_counts.pop(key, None)
        else:
            _active_turn_counts[key] = current - 1
        remaining = _related_active_turns_unlocked(key)
        for deferred_key, deferred_task_id in list(
            _deferred_environment_cleanups.items()
        ):
            if _related_active_turns_unlocked(deferred_key) == 0:
                deferred_task_ids.append(deferred_task_id)
                _deferred_environment_cleanups.pop(deferred_key, None)
    for deferred_task_id in deferred_task_ids:
        _run_deferred_environment_cleanup(deferred_task_id)
    return remaining


def release_environment_turn(task_id: Hashable) -> int:
    return _release_environment_turn_key(_turn_scope_key(task_id))


def defer_environment_turn_cleanup(task_id: Hashable) -> None:
    # Run collapsed cleanup when the final overlapping lease releases.
    key = _turn_scope_key(task_id)
    run_now = False
    with _active_turn_counts_lock:
        if _related_active_turns_unlocked(key) > 0:
            _deferred_environment_cleanups.setdefault(key, task_id)
        else:
            run_now = True
    if run_now:
        _run_deferred_environment_cleanup(task_id)


def active_environment_turns(task_id: Hashable) -> int:
    return _active_turns_for_environment_key(_turn_scope_key(task_id))


def _active_turns_for_environment_key(environment_key: Hashable) -> int:
    with _active_turn_counts_lock:
        return _related_active_turns_unlocked(environment_key)


class _EnvironmentTurnLease:
    def __init__(
        self,
        task_id: Hashable,
        *,
        environment_key: Hashable | None = None,
    ):
        self._key = (
            _register_environment_turn_key(environment_key)
            if environment_key is not None
            else register_environment_turn(task_id)
        )
        self._released = False
        self._lock = threading.Lock()

    @property
    def key(self) -> Hashable:
        return self._key

    @property
    def active(self) -> bool:
        with self._lock:
            return not self._released

    def release(self) -> int:
        with self._lock:
            if self._released:
                return _active_turns_for_environment_key(self._key)
            self._released = True
        return _release_environment_turn_key(self._key)


_logical_environment_lease: ContextVar[Optional[_EnvironmentTurnLease]] = ContextVar(
    "logical_environment_lease", default=None,
)
_tool_environment_lease: ContextVar[Optional[_EnvironmentTurnLease]] = ContextVar(
    "tool_environment_lease", default=None,
)


@contextmanager
def logical_environment_turn(task_id: Hashable):
    # Hold the shared environment for one complete logical conversation turn.
    lease = _EnvironmentTurnLease(task_id)
    token = _logical_environment_lease.set(lease)
    try:
        yield lease
    finally:
        lease.release()
        _logical_environment_lease.reset(token)


def release_logical_environment_turn(task_id: Hashable) -> int:
    # Release this context's logical lease before final cleanup.
    lease = _logical_environment_lease.get()
    key = _turn_scope_key(task_id)
    if lease is not None and lease.key == key:
        return lease.release()
    return active_environment_turns(task_id)


def release_logical_environment_turn_for_cleanup(task_id: Hashable) -> bool:
    # Preserve the established boolean cleanup-hook contract.
    lease = _logical_environment_lease.get()
    if lease is None or lease.key != _turn_scope_key(task_id):
        return False
    lease.release()
    return True


def execution_environment_turn_key(
    function_name: str,
    arguments: Mapping[str, Any],
    *,
    task_id: Hashable | None = None,
) -> Hashable | None:
    if function_name not in {
        "terminal", "read_file", "write_file", "patch", "search_files",
        "execute_code", "process",
    }:
        return None
    task_id = arguments.get("task_id") or task_id
    if not task_id:
        return None
    if function_name == "process":
        # Follow-up calls select a persisted session_id rather than a target.
        # A parent-scope lease safely covers whichever named runtime owns it.
        return _turn_scope_key(task_id)
    target = arguments.get("target")
    if function_name == "search_files":
        target = arguments.get("execution_target")
    try:
        from tools.execution_targets import resolve_execution_target

        resolution = resolve_execution_target(target)
        base_task_id = _resolve_container_task_id(str(task_id))
        return resolution.session_key(base_task_id)
    except Exception:
        # Invalid-target tools still execute to return their normal user-visible
        # validation error; the raw logical lease remains the safe fallback.
        return None


@contextmanager
def environment_turn_usage(
    task_id: Hashable,
    *,
    environment_key: Hashable | None = None,
):
    # Protect one terminal, file, or code invocation from idle cleanup.
    lease = _EnvironmentTurnLease(task_id, environment_key=environment_key)
    token = _tool_environment_lease.set(lease)
    try:
        yield
    finally:
        lease.release()
        _tool_environment_lease.reset(token)


def _current_owned_environment_turns(environment_key: Hashable) -> int:
    # Count this call's own logical/tool leases for replacement checks.
    owned = 0
    for lease in (
        _logical_environment_lease.get(),
        _tool_environment_lease.get(),
    ):
        if (
            lease is not None
            and lease.active
            and _turn_keys_overlap(lease.key, environment_key)
        ):
            owned += 1
    return owned


def record_session_cwd(
    session_key: Optional[str],
    cwd: Optional[str],
    target: Optional[str] = None,
    *,
    _resolution=None,
) -> None:
    """Record *cwd* as the working directory of *session_key*.

    Called wherever a session's live cwd becomes known: after a terminal
    command completes (the env's post-command tracking has just parsed the
    resulting cwd) and when a surface registers a workspace cwd override.
    Empty/None session keys collapse to ``"default"`` (single-session CLI).
    Non-string / empty cwds are ignored.
    """
    if not isinstance(cwd, str) or not cwd.strip():
        return
    resolution = _resolution or _target_resolution(target)
    key = resolution.session_key(session_key)
    with _session_cwd_lock:
        if _session_cwd.get(key) != cwd:
            _session_cwd[key] = cwd
        if resolution.named:
            _session_cwd_specs[key] = resolution.spec_fingerprint
        else:
            _session_cwd_specs.pop(key, None)


def get_session_cwd(
    session_key: Optional[str],
    target: Optional[str] = None,
    *,
    _resolution=None,
) -> Optional[str]:
    """Return the recorded working directory for *session_key*, if any.

    No fallback chain here on purpose: callers decide what an absent record
    means (config default, TERMINAL_CWD seed, process cwd). ``None``/empty
    keys read the ``"default"`` record.
    """
    resolution = _resolution or _target_resolution(target)
    key = resolution.session_key(session_key)
    with _session_cwd_lock:
        recorded_spec = _session_cwd_specs.get(key)
        if (
            resolution.named
            and recorded_spec is not None
            and recorded_spec != resolution.spec_fingerprint
        ):
            return None
        return _session_cwd.get(key)


def inherit_session_cwds(parent_task_id: str, child_task_id: str) -> int:
    """Seed a child with every cwd scope currently owned by its parent."""
    if not parent_task_id or not child_task_id:
        return 0
    parent_key = _profile_scoped_task_key(parent_task_id)
    child_key = _profile_scoped_task_key(child_task_id)
    inherited: Dict[Hashable, str] = {}
    inherited_specs: Dict[Hashable, str] = {}
    with _session_cwd_lock:
        for key, cwd in _session_cwd.items():
            if key == parent_key:
                inherited[child_key] = cwd
                if key in _session_cwd_specs:
                    inherited_specs[child_key] = _session_cwd_specs[key]
            elif (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] == parent_key
            ):
                child_target_key = (child_key, key[1])
                inherited[child_target_key] = cwd
                if key in _session_cwd_specs:
                    inherited_specs[child_target_key] = _session_cwd_specs[key]
        _session_cwd.update(inherited)
        _session_cwd_specs.update(inherited_specs)
    return len(inherited)


def clear_session_cwd(session_key: str) -> None:
    """Drop all legacy and named-target cwd records for a raw session."""
    raw = str(session_key or "default")
    scoped = _profile_scoped_task_key(raw)
    with _session_cwd_lock:
        _session_cwd.pop(raw, None)
        _session_cwd.pop(scoped, None)
        _session_cwd_specs.pop(raw, None)
        _session_cwd_specs.pop(scoped, None)
        for key in list(_session_cwd):
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] in {raw, scoped}
            ):
                _session_cwd.pop(key, None)
                _session_cwd_specs.pop(key, None)


def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]):
    """
    Register environment overrides for a specific task/rollout.

    Called by Atropos environments before the agent loop to configure
    per-task sandbox settings (e.g., a custom Dockerfile for the Modal image).

    Supported override keys:
        - modal_image: str -- Path to Dockerfile or Docker Hub image name
        - docker_image: str -- Docker image name
        - cwd: str -- Working directory inside the sandbox

    Args:
        task_id: The rollout's unique task identifier
        overrides: Dict of config keys to override
    """
    _task_env_overrides[_profile_scoped_task_key(task_id)] = overrides

    # If a live environment already exists for this task, a freshly registered
    # ``cwd`` override (e.g. the ACP client switching the editor's project root
    # mid-session via ``session/load`` / ``session/resume``) must take effect
    # immediately. The session record is what commands resolve against;
    # the live env's cwd is also updated so env-side seeding stays consistent.
    new_cwd = overrides.get("cwd")
    if isinstance(new_cwd, str) and new_cwd.strip():
        # A registered workspace cwd IS the session's working directory until
        # a `cd` changes it. With named targets this host/workspace override
        # belongs to the configured default target only; applying it to every
        # explicit remote/container target would replace that target's own cwd.
        try:
            default_resolution = _target_resolution(None)
        except Exception:
            default_resolution = None
        if (
            default_resolution is not None
            and not (
                default_resolution.named
                and default_resolution.backend == "ssh"
            )
        ):
            record_session_cwd(
                task_id, new_cwd, _resolution=default_resolution,
            )
        # The live env is cached under the raw task_id for per-session surfaces
        # (ACP/gateway/dashboard) and under the collapsed container id for
        # isolation-keyed rollouts. Try the raw id first, then the container id,
        # so a CWD-only override (which collapses to "default") still finds and
        # updates the originating session's env.
        container_id = _resolve_container_task_id(task_id)
        with _env_lock:
            if (
                default_resolution is not None
                and default_resolution.named
                and default_resolution.backend != "ssh"
            ):
                candidate_keys = {
                    default_resolution.environment_key(task_id),
                    default_resolution.environment_key(container_id),
                }
            else:
                candidate_keys = {task_id, container_id}
            envs = [
                env for key, env in _active_environments.items()
                if key in candidate_keys
            ]
        for env in envs:
            if getattr(env, "cwd", None) is not None:
                env.cwd = new_cwd


def clear_task_env_overrides(task_id: str):
    """
    Clear environment overrides for a task after rollout completes.

    Called during cleanup to avoid stale entries accumulating.
    """
    _task_env_overrides.pop(_profile_scoped_task_key(task_id), None)
    clear_session_cwd(task_id)


def _resolve_container_task_id(task_id: Optional[str]) -> str:
    """
    Map a tool-call ``task_id`` to the container/sandbox key used by
    ``_active_environments``.

    The top-level agent passes ``task_id=None`` and lands on ``"default"``.
    ``delegate_task`` children pass their own subagent ID so that
    file-state tracking, the active-subagents registry, and TUI events stay
    distinct per child -- but we deliberately collapse that ID back to
    ``"default"`` here so subagents share the parent's long-lived container
    (one bash, one /workspace, one set of installed packages).

    Exception: RL / benchmark environments (TerminalBench2, HermesSweEnv, ...)
    call ``register_task_env_overrides(task_id, {...})`` to request a
    per-task Docker/Modal image. When an override is registered for a
    task_id, we honour it by returning the task_id unchanged -- those
    rollouts need their own isolated sandbox, which is the whole point of
    the override.

    CWD-only overrides (registered by the ACP adapter for workspace
    tracking) are *not* isolation signals — they should not cause each
    session to spin up its own container.  Only overrides containing
    backend-specific image keys or ``env_type`` trigger isolation.
    """
    _ISOLATION_KEYS = frozenset({
        "docker_image", "modal_image", "singularity_image",
        "daytona_image", "env_type",
    })
    scoped_task_id = _profile_scoped_task_key(task_id) if task_id else None
    if scoped_task_id and scoped_task_id in _task_env_overrides:
        overrides = _task_env_overrides[scoped_task_id]
        if set(overrides.keys()) & _ISOLATION_KEYS:
            return str(task_id)
    return "default"


def resolve_task_overrides(task_id: Optional[str]) -> Dict[str, Any]:
    """Return the env overrides for *task_id*, raw key first then collapsed.

    ``register_task_env_overrides`` writes under the *raw* task/session id, but
    a CWD-only override collapses (:func:`_resolve_container_task_id`) to the
    shared ``"default"`` container so per-session surfaces (ACP/gateway/
    dashboard) don't each spin up their own sandbox. Callers that need the
    override (terminal command setup, file-tool cwd resolution) must therefore
    read the raw id FIRST and only fall back to the collapsed container id, or
    the originating session's override is silently dropped. This is the single
    source of that lookup so the terminal and file layers can't drift apart.
    """
    raw = task_id or "default"
    scoped_raw = _profile_scoped_task_key(raw)
    scoped_collapsed = _profile_scoped_task_key(_resolve_container_task_id(raw))
    return (
        _task_env_overrides.get(scoped_raw)
        or _task_env_overrides.get(scoped_collapsed)
        or {}
    )


# Configuration from environment variables

def _parse_env_var(name: str, default: str, converter: Any = int, type_label: str = "integer"):
    """Parse an environment variable with *converter*, raising a clear error on bad values.

    Without this wrapper, a single malformed env var (e.g. TERMINAL_TIMEOUT=5m)
    causes an unhandled ValueError that kills every terminal command.
    """
    raw = os.getenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            f"Check ~/.hermes/.env or environment variables."
        )


def _safe_getcwd() -> str:
    """Return the current working directory, tolerating a deleted CWD.

    ``os.getcwd()`` raises FileNotFoundError when the process's working
    directory has been removed out from under it (e.g. a scratch workspace
    that was cleaned up mid-session). Fall back to TERMINAL_CWD, then the
    user's home directory, so terminal setup never crashes on a stale CWD.
    """
    try:
        return os.getcwd()
    except FileNotFoundError:
        return os.getenv("TERMINAL_CWD") or os.path.expanduser("~")


# Path prefixes that identify a *host* working directory which cannot exist
# inside a container sandbox. Covers POSIX user dirs and Windows drive paths
# (``C:\Users\...`` / ``C:/Users/...``) — the latter is how a Windows host's
# cwd looks when it leaks toward a Linux container's ``-w`` flag.
_HOST_CWD_PREFIXES = ("/Users/", "/home/", "C:\\", "C:/")

_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})


def _is_ssh_remote_tilde_cwd(backend: str, cwd: str) -> bool:
    """Return True when *cwd* is a tilde path that the remote SSH shell must
    expand itself, so the Hermes host/container must NOT ``expanduser`` it.

    SSH ``cwd`` is interpreted by the *remote* shell (``cd ~`` / ``cd ~/x``
    over ``ssh ... bash -c``). Expanding ``~`` locally would rewrite it to the
    Hermes host HOME (often ``/opt/data`` under Docker) and inject a
    nonexistent path into the remote session. Only ``~`` / ``~/...`` on the
    ``ssh`` backend qualify; absolute remote paths still pass through
    unchanged, and every other backend keeps expanding locally.
    """
    if (backend or "").strip().lower() != "ssh":
        return False
    return cwd == "~" or cwd.startswith("~/")


def _is_unusable_container_cwd(cwd: str) -> bool:
    """Return True if *cwd* is a host/relative path that won't work as the
    working directory inside a container sandbox.

    A container's cwd must be an absolute path that exists *inside* the
    sandbox (e.g. ``/workspace`` or ``/root``). A host path (``/home/user``,
    ``C:\\Users\\me``) or a relative path (``.``, ``src/``) is meaningless to
    ``docker run -w`` and makes the container fail to start (exit 125).
    """
    if not cwd:
        return False
    if any(cwd.startswith(p) for p in _HOST_CWD_PREFIXES):
        return True
    # Relative paths (".", "src/") can't be a container workdir either. Windows
    # drive paths are absolute on Windows but os.path.isabs() is False on a
    # POSIX host, so they're already caught by the prefix check above.
    if not os.path.isabs(cwd):
        return True
    return False


def _apply_task_cwd_override(
    config: Dict[str, Any], cwd: str, cwd_override: Optional[str],
) -> str:
    """Apply a task workspace cwd without leaking host paths into containers.

    Docker's explicit mount-cwd mode is the exception: a registered host
    workspace should become the bind source and commands should run in
    ``/workspace``. Other container backends fall back to the target's already
    sanitized configured cwd.
    """
    env_type = config.get("env_type")
    if (
        env_type == "docker"
        and config.get("docker_mount_cwd_to_workspace")
        and isinstance(cwd_override, str)
        and cwd_override.strip()
    ):
        candidate = os.path.abspath(os.path.expanduser(cwd_override))
        is_host_path = (
            any(candidate.startswith(prefix) for prefix in _HOST_CWD_PREFIXES)
            or (
                os.path.isabs(candidate)
                and os.path.isdir(candidate)
            )
        )
        if is_host_path:
            config["host_cwd"] = candidate
            return "/workspace"
    if env_type in _CONTAINER_BACKENDS and _is_unusable_container_cwd(cwd):
        return config["cwd"]
    return cwd


# One-shot guard for the config-fallback bridge below.  Purely an
# optimization: after the first attempt either TERMINAL_ENV is set (bridge
# succeeded — merged config always carries terminal.backend) or the import
# failed and retrying every call would be wasted work.
_terminal_config_bridge_attempted = False


def _ensure_terminal_env_bridged() -> None:
    """Backfill TERMINAL_* env vars from config.yaml when no launcher did.

    terminal_tool reads ALL terminal settings from os.environ (TERMINAL_*).
    The CLI (cli.py ``env_mappings``), the gateway (gateway/run.py
    ``_terminal_env_map``), and TUI/dashboard PTY launches
    (``apply_terminal_config_to_env``) bridge ``terminal.*`` config into env
    vars at startup — but processes that skip all of those paths (``hermes
    serve`` / the Desktop app backend's in-process agents, the desktop cron
    ticker, ACP) used to silently fall back to the local backend even when
    config.yaml selects ``terminal.backend: docker``, running commands on the
    host the user intended to sandbox (#63141, #54449, #61115, #65696).

    Explicit terminal config keys win: when config.yaml has a ``terminal``
    section, each key present there overrides its matching env value (which may
    be stale from ``hermes setup``). Environment values for omitted terminal
    keys are preserved. When no terminal section exists, exported/.env values
    keep working unchanged.
    """
    global _terminal_config_bridge_attempted
    if _terminal_config_bridge_attempted:
        return
    _terminal_config_bridge_attempted = True
    try:
        from hermes_cli.config import apply_terminal_config_to_env, read_raw_config

        # If config.yaml has an explicit terminal section, bridge with
        # override enabled. The helper only overrides env vars for keys present
        # in that raw section; merged defaults remain backfill-only. Without a
        # terminal section, preserve an existing TERMINAL_ENV selection or
        # backfill defaults when no selection exists.
        raw_config = read_raw_config()
        has_terminal_section = isinstance(raw_config.get("terminal"), dict)

        if has_terminal_section:
            # Explicit terminal keys in config.yaml win over matching env values.
            apply_terminal_config_to_env(env=None, override=True)
        elif "TERMINAL_ENV" not in os.environ:
            # No terminal section in config.yaml, TERMINAL_ENV not set —
            # backfill from config defaults
            apply_terminal_config_to_env(env=None, override=False)
    except Exception:
        # Never let a config problem take the terminal tool down — the
        # historical local default still applies.
        logger.debug("terminal config → env fallback bridge failed", exc_info=True)


def _get_env_config(terminal_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return canonical terminal config for legacy env vars or a target mapping.

    ``terminal_config is None`` is the historical flat/env-driven path.  A
    selected named target passes its inherited mapping here directly; values
    are parsed without mutating ``os.environ`` so concurrent target calls
    cannot affect each other.
    """
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    if terminal_config is None:
        _ensure_terminal_env_bridged()

    def _get(key: str, env_name: str, default: Any) -> Any:
        if terminal_config is None:
            return os.getenv(env_name, str(default) if not isinstance(default, (list, dict)) else json.dumps(default))
        return terminal_config.get(key, default)

    def _coerce(value: Any, converter: Any, label: str, key: str) -> Any:
        if converter is json.loads and isinstance(value, (list, dict)):
            return value
        try:
            return converter(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            source = (
                f"terminal target setting {key}"
                if terminal_config is not None
                else f"TERMINAL_{key.upper()}"
            )
            raise ValueError(f"Invalid value for {source}: {value!r} (expected {label}).")

    def _bool(key: str, env_name: str, default: bool) -> bool:
        value = _get(key, env_name, default)
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        if terminal_config is not None:
            raise ValueError(
                f"Invalid value for terminal target setting {key}: {value!r} "
                "(expected boolean)."
            )
        # Preserve the legacy env-var behavior for unknown strings.
        return False

    def _json_shape(
        key: str,
        env_name: str,
        default: Any,
        expected_type: type,
        label: str,
    ) -> Any:
        value = _coerce(_get(key, env_name, default), json.loads, "valid JSON", key)
        if not isinstance(value, expected_type):
            source = (
                f"terminal target setting {key}"
                if terminal_config is not None
                else env_name
            )
            raise ValueError(
                f"Invalid value for {source}: {value!r} (expected {label})."
            )
        return value

    env_type = str(
        _get(
            "backend", "TERMINAL_ENV",
            terminal_config.get("env_type", "local") if terminal_config else "local",
        )
    ).strip().lower() or "local"
    
    mount_docker_cwd = _bool(
        "docker_mount_cwd_to_workspace", "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", False,
    )
    container_backend = env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}
    docker_backend = env_type == "docker"

    # Docker/container-only env vars may be bridged from config.yaml even when
    # the active backend is local/ssh.  Do not parse their JSON/numeric payloads
    # until a backend that can consume them is selected; a stale or invalid
    # Docker value should not make local terminal/execute_code unusable.
    if container_backend:
        container_cpu = _coerce(_get("container_cpu", "TERMINAL_CONTAINER_CPU", 1), float, "number", "container_cpu")
        container_memory = _coerce(_get("container_memory", "TERMINAL_CONTAINER_MEMORY", 5120), int, "integer", "container_memory")
        container_disk = _coerce(_get("container_disk", "TERMINAL_CONTAINER_DISK", 51200), int, "integer", "container_disk")
    else:
        container_cpu = 1.0
        container_memory = 5120
        container_disk = 51200

    if docker_backend:
        docker_forward_env = _json_shape(
            "docker_forward_env", "TERMINAL_DOCKER_FORWARD_ENV", [], list, "list",
        )
        docker_volumes = _json_shape(
            "docker_volumes", "TERMINAL_DOCKER_VOLUMES", [], list, "list",
        )
        docker_env = _json_shape(
            "docker_env", "TERMINAL_DOCKER_ENV", {}, dict, "mapping",
        )
        docker_extra_args = _json_shape(
            "docker_extra_args", "TERMINAL_DOCKER_EXTRA_ARGS", [], list, "list",
        )
        docker_shm_size = str(_get("docker_shm_size", "TERMINAL_DOCKER_SHM_SIZE", "1g") or "")
    else:
        docker_forward_env = []
        docker_volumes = []
        docker_env = {}
        docker_extra_args = []
        docker_shm_size = "1g"

    # Default cwd: local uses the host's current directory, ssh uses the
    # remote home, Vercel uses its documented workspace root, and everything
    # else starts in the backend's default root-like cwd.
    if env_type == "local":
        default_cwd = _safe_getcwd()
    elif env_type == "ssh":
        default_cwd = "~"
    elif env_type == "vercel_sandbox":
        default_cwd = _VERCEL_SANDBOX_DEFAULT_CWD
    else:
        default_cwd = "/root"

    # Read TERMINAL_CWD but sanity-check it for container backends.
    # If Docker cwd passthrough is explicitly enabled, remap the host path to
    # /workspace and track the original host path separately. Otherwise keep the
    # normal sandbox behavior and discard host paths.
    cwd = str(_get("cwd", "TERMINAL_CWD", default_cwd) or default_cwd)
    if env_type == "local" and cwd in {".", "./", "auto", "cwd"}:
        cwd = _safe_getcwd()
    if cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        docker_cwd_source = (
            (os.getenv("TERMINAL_CWD") or _safe_getcwd())
            if terminal_config is None
            else (cwd or _safe_getcwd())
        )
        candidate = os.path.abspath(os.path.expanduser(docker_cwd_source))
        if (
            any(candidate.startswith(p) for p in _HOST_CWD_PREFIXES)
            or (os.path.isabs(candidate) and os.path.isdir(candidate) and not candidate.startswith(("/workspace", "/root")))
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif env_type in _CONTAINER_BACKENDS and cwd:
        # Host paths and relative paths that won't work inside containers
        if _is_unusable_container_cwd(cwd) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                        "(host/relative path won't work in sandbox). Using %r instead.",
                        cwd, env_type, default_cwd)
            cwd = default_cwd

    return {
        "env_type": env_type,
        "modal_mode": coerce_modal_mode(_get("modal_mode", "TERMINAL_MODAL_MODE", "auto")),
        "docker_image": str(_get("docker_image", "TERMINAL_DOCKER_IMAGE", default_image)),
        "docker_forward_env": docker_forward_env,
        "singularity_image": str(_get("singularity_image", "TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}")),
        "modal_image": str(_get("modal_image", "TERMINAL_MODAL_IMAGE", default_image)),
        "daytona_image": str(_get("daytona_image", "TERMINAL_DAYTONA_IMAGE", default_image)),
        "vercel_runtime": str(_get("vercel_runtime", "TERMINAL_VERCEL_RUNTIME", "")).strip(),
        "cwd": cwd,
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": _coerce(_get("timeout", "TERMINAL_TIMEOUT", 180), int, "integer", "timeout"),
        "lifetime_seconds": _coerce(_get("lifetime_seconds", "TERMINAL_LIFETIME_SECONDS", 300), int, "integer", "lifetime_seconds"),
        # SSH-specific config
        "ssh_host": str(_get("ssh_host", "TERMINAL_SSH_HOST", "")),
        "ssh_user": str(_get("ssh_user", "TERMINAL_SSH_USER", "")),
        "ssh_port": _coerce(_get("ssh_port", "TERMINAL_SSH_PORT", 22), int, "integer", "ssh_port"),
        "ssh_key": str(_get("ssh_key", "TERMINAL_SSH_KEY", "")),
        # Persistent shell: SSH defaults to the config-level persistent_shell
        # setting (true by default for non-local backends); local is always opt-in.
        # Per-backend env vars override if explicitly set.
        "ssh_persistent": _bool(
            "ssh_persistent", "TERMINAL_SSH_PERSISTENT",
            _bool("persistent_shell", "TERMINAL_PERSISTENT_SHELL", True),
        ),
        "local_persistent": _bool("local_persistent", "TERMINAL_LOCAL_PERSISTENT", False),
        # Container resource config (applies to docker, singularity, modal,
        # daytona, and vercel_sandbox -- ignored for local/ssh)
        "container_cpu": container_cpu,
        "container_memory": container_memory,     # MB (default 5GB)
        "container_disk": container_disk,        # MB (default 50GB)
        "container_persistent": _bool("container_persistent", "TERMINAL_CONTAINER_PERSISTENT", True),
        "docker_volumes": docker_volumes,
        "docker_env": docker_env,
        "docker_run_as_host_user": _bool("docker_run_as_host_user", "TERMINAL_DOCKER_RUN_AS_HOST_USER", False),
        "docker_network": _bool("docker_network", "TERMINAL_DOCKER_NETWORK", True),
        "docker_extra_args": docker_extra_args,
        "docker_shm_size": docker_shm_size,
        # Cross-process container reuse (issue #20561).  The docs claim
        # "ONE long-lived container shared across sessions" — this toggle
        # makes that real by probing for a labeled container at startup and
        # attaching to it instead of always starting a fresh one.  Set to
        # ``false`` for hard per-process isolation (no reuse, container is
        # removed on exit).
        "docker_persist_across_processes": _bool(
            "docker_persist_across_processes", "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", True,
        ),
        # Startup orphan reaper for hermes-tagged containers left behind by
        # crashed / SIGKILL'd previous processes that bypassed atexit.
        # Conservative: only sweeps Exited containers older than 2× the
        # idle-reap window AND scoped to the current profile. Issue #20561.
        "docker_orphan_reaper": _bool(
            "docker_orphan_reaper", "TERMINAL_DOCKER_ORPHAN_REAPER", True,
        ),
    }



def _build_environment_constructor_configs(
    config: Dict[str, Any],
    resolution: 'ExecutionTargetResolution',
    base_task_id: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Build backend constructor inputs from one canonical normalized config."""
    env_type = config["env_type"]
    container_config: Optional[Dict[str, Any]] = None
    if env_type in _CONTAINER_BACKENDS:
        container_config = {
            "container_cpu": config.get("container_cpu", 1),
            "container_memory": config.get("container_memory", 5120),
            "container_disk": config.get("container_disk", 51200),
            "container_persistent": config.get("container_persistent", True),
            "vercel_runtime": config.get("vercel_runtime", ""),
            "modal_mode": config.get("modal_mode", "auto"),
            "docker_volumes": config.get("docker_volumes", []),
            "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
            "docker_forward_env": config.get("docker_forward_env", []),
            "docker_env": config.get("docker_env", {}),
            "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
            "docker_extra_args": config.get("docker_extra_args", []),
            "docker_network": config.get("docker_network", True),
            "docker_shm_size": config.get("docker_shm_size", "1g"),
            "docker_persist_across_processes": config.get("docker_persist_across_processes", True),
            "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
            "lifetime_seconds": config.get("lifetime_seconds", 300),
            "storage_task_id": resolution.storage_task_id(base_task_id),
            "legacy_storage_task_id": resolution.legacy_backend_task_id(base_task_id),
        }

    ssh_config: Optional[Dict[str, Any]] = None
    if env_type == "ssh":
        ssh_config = {
            "host": config.get("ssh_host", ""),
            "user": config.get("ssh_user", ""),
            "port": config.get("ssh_port", 22),
            "key": config.get("ssh_key", ""),
            "persistent": config.get("ssh_persistent", False),
            "runtime_scope": resolution.security_scope if resolution.named else "",
        }

    local_config: Optional[Dict[str, Any]] = None
    if env_type == "local":
        local_config = {"persistent": config.get("local_persistent", False)}
    return container_config, ssh_config, local_config


def _record_environment_lifetime(env: Any, config: Dict[str, Any]) -> None:
    """Attach the resolved target's idle lifetime to its environment."""
    try:
        env._hermes_lifetime_seconds = int(config["lifetime_seconds"])
    except (AttributeError, KeyError, TypeError, ValueError):
        pass


def _record_environment_target(env: Any, resolution: Any) -> None:
    """Bind a created environment to the exact resolved named-target spec."""
    try:
        setattr(env, "_hermes_target_name", resolution.target)
        setattr(
            env, "_hermes_target_fingerprint",
            resolution.spec_fingerprint if resolution.named else None,
        )
        setattr(env, "_hermes_target_backend", resolution.backend)
        setattr(
            env, "_hermes_target_scope",
            resolution.security_scope if resolution.named else None,
        )
        setattr(env, "_hermes_target_resolution", resolution)
        persistent = resolution.config.get("container_persistent", True)
        if isinstance(persistent, str):
            persistent = persistent.strip().lower() in {"1", "true", "yes", "on"}
        setattr(
            env,
            "_hermes_stable_storage",
            resolution.backend == "docker" and bool(persistent),
        )
    except (AttributeError, TypeError):
        pass


def _environment_matches_target(env: Any, resolution: Any) -> bool:
    """Reject cache reuse after a named target's effective config changes."""
    if env is None or not resolution.named:
        return env is not None
    fingerprint = getattr(env, "_hermes_target_fingerprint", None)
    # Third-party/test-provided environments predating named targets have no
    # binding metadata. Preserve their registration contract; every environment
    # created by core Hermes is stamped before entering the cache.
    if fingerprint is None:
        return True
    return (
        fingerprint == resolution.spec_fingerprint
        and getattr(env, "_hermes_target_name", resolution.target) == resolution.target
        and getattr(env, "_hermes_target_backend", resolution.backend) == resolution.backend
    )


def _environment_has_stable_storage(env: Any) -> bool:
    return bool(getattr(env, "_hermes_stable_storage", False))


def _environment_replacement_is_busy(env: Any, environment_key: Hashable) -> bool:
    """Protect shared persistent storage while the old runtime is still active."""
    if not _environment_has_stable_storage(env):
        return False
    active = _active_turns_for_environment_key(environment_key)
    owned = _current_owned_environment_turns(environment_key)
    if active > owned:
        return True
    try:
        from tools.process_registry import process_registry

        return process_registry.has_active_environment(env)
    except Exception:
        return False


def _cleanup_environment_resource(
    env: Any,
    *,
    force_remove: bool = False,
    preserve_storage: bool = False,
) -> None:
    """Stop one environment, optionally preserving its persistent storage."""
    import inspect

    ownership_attrs = {}
    if force_remove:
        # A replaced environment is unreachable by configuration and must not
        # retain persist-mode lifecycle semantics. Persistent storage can remain
        # owned by the stable profile/target storage identity while the obsolete
        # runtime is removed.
        attrs = ["_persist_across_processes"]
        if not preserve_storage:
            attrs.extend(["_persistent", "persistent_filesystem"])
        for attr in attrs:
            if hasattr(env, attr):
                try:
                    ownership_attrs[attr] = getattr(env, attr)
                    setattr(env, attr, False)
                except (AttributeError, TypeError):
                    pass

    try:
        if hasattr(env, "cleanup"):
            cleanup = env.cleanup
            kwargs = {}
            if force_remove:
                try:
                    if "force_remove" in inspect.signature(cleanup).parameters:
                        kwargs["force_remove"] = True
                except (TypeError, ValueError):
                    pass
            result = cleanup(**kwargs)
        elif hasattr(env, "stop"):
            result = env.stop()
        elif hasattr(env, "terminate"):
            result = env.terminate()
        else:
            return

        if inspect.isawaitable(result):
            import asyncio

            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(result)
                loop.close()
            except Exception:
                try:
                    close = getattr(result, "close", None)
                    if close is not None:
                        close()
                except Exception:
                    pass
                raise

        wait_fn = getattr(env, "wait_for_cleanup", None)
        if wait_fn is not None and not wait_fn(timeout=60.0):
            raise RuntimeError("environment cleanup did not finish within 60 seconds")
    except BaseException:
        # Cleanup can fail before the obsolete runtime is actually removed. In
        # that case the caller restores the handle to the active cache, so also
        # restore its persistence ownership flags rather than leaving a live
        # runtime reachable but disowned by this process.
        for attr, value in ownership_attrs.items():
            try:
                setattr(env, attr, value)
            except (AttributeError, TypeError):
                pass
        raise


class _EnvironmentReplacementError(RuntimeError):
    """Base error for a fail-closed named-target runtime replacement."""


class _EnvironmentReplacementBusyError(_EnvironmentReplacementError):
    """The previous stable-storage runtime still has active users."""


class _EnvironmentReplacementCleanupError(_EnvironmentReplacementError):
    """The previous stable-storage runtime could not be retired safely."""


def _prepare_environment_replacement(
    env: Any,
    environment_key: Hashable,
    *,
    target_name: str,
) -> bool:
    """Retire an idle stable-storage runtime before creating its replacement.

    Persistent Docker generations share one storage identity, so the obsolete
    runtime must be gone before the replacement container is created. The
    caller holds the per-environment creation lock; this helper owns the shared
    detach/cleanup/restore handoff used by terminal, file, and execute_code.
    """
    if env is None:
        return False
    if _environment_replacement_is_busy(env, environment_key):
        raise _EnvironmentReplacementBusyError(
            f"Execution target {target_name!r} changed while its persistent "
            "Docker runtime is still active. Wait for its commands/background "
            "processes to finish, then retry."
        )
    if not _environment_has_stable_storage(env):
        return False

    with _env_lock:
        if _active_environments.get(environment_key) is not env:
            raise _EnvironmentReplacementBusyError(
                f"Execution target {target_name!r} changed again while its "
                "previous runtime was being retired. Retry the operation."
            )
        owned_keys = [
            (key, key in _last_activity, _last_activity.get(key, 0.0))
            for key, candidate in list(_active_environments.items())
            if candidate is env
        ]
        for key, _, _ in owned_keys:
            _active_environments.pop(key, None)
            _last_activity.pop(key, None)

    try:
        _cleanup_environment_resource(
            env,
            force_remove=True,
            preserve_storage=True,
        )
    except BaseException as exc:
        with _env_lock:
            for key, had_activity, activity in owned_keys:
                if key not in _active_environments:
                    _active_environments[key] = env
                    if had_activity:
                        _last_activity[key] = activity
        if isinstance(exc, Exception):
            raise _EnvironmentReplacementCleanupError(
                "Could not retire the previous persistent Docker runtime for "
                f"execution target {target_name!r}: {exc}"
            ) from exc
        raise
    return True


def _retire_replaced_environment(env: Any, task_key: Hashable) -> None:
    """Defer teardown until no operation/process can still reference *env*."""
    if env is None:
        return
    with _retired_environments_lock:
        if all(existing_env is not env for _, existing_env, _ in _retired_environments):
            _retired_environments.append((task_key, env, time.time()))


def _collect_retired_environments(
    *,
    task_key: Hashable | None = None,
    min_age_seconds: float = 60.0,
    require_idle: bool = True,
) -> list[tuple[Hashable, Any]]:
    """Detach retired resources that are old enough and no longer in use."""
    now = time.time()
    ready: list[tuple[Hashable, Any]] = []
    keep: list[tuple[Hashable, Any, float]] = []
    try:
        from tools.process_registry import process_registry
    except ImportError:
        process_registry = None

    with _retired_environments_lock:
        candidates = list(_retired_environments)
        _retired_environments.clear()

    for retired_key, env, retired_at in candidates:
        if task_key is not None and retired_key != task_key:
            keep.append((retired_key, env, retired_at))
            continue
        busy = False
        if require_idle:
            busy = _active_turns_for_environment_key(retired_key) > 0
            if not busy and process_registry is not None:
                busy = process_registry.has_active_environment(env)
        if busy or now - retired_at < min_age_seconds:
            keep.append((retired_key, env, retired_at))
        else:
            ready.append((retired_key, env))

    # Merge records retired concurrently while we performed potentially slow
    # process liveness checks. Avoid duplicate records by environment identity.
    ready_ids = {id(env) for _, env in ready}
    with _retired_environments_lock:
        concurrent = [
            record for record in _retired_environments
            if id(record[1]) not in ready_ids
        ]
        seen = {id(record[1]) for record in concurrent}
        concurrent.extend(
            record for record in keep
            if id(record[1]) not in seen
        )
        _retired_environments[:] = concurrent
    return ready


def _cleanup_retired_environments(
    *,
    task_key: Hashable | None = None,
    min_age_seconds: float = 60.0,
    require_idle: bool = True,
) -> int:
    """Force-remove retired environments selected by lifecycle policy."""
    ready = _collect_retired_environments(
        task_key=task_key,
        min_age_seconds=min_age_seconds,
        require_idle=require_idle,
    )
    cleaned = 0
    for retired_key, env in ready:
        try:
            _cleanup_environment_resource(
                env,
                force_remove=True,
                preserve_storage=_environment_has_stable_storage(env),
            )
            cleaned += 1
            logger.info("Cleaned retired environment for task: %s", retired_key)
        except Exception as exc:
            error_str = str(exc)
            if "404" in error_str or "not found" in error_str.lower():
                cleaned += 1
                logger.info("Retired environment for task %s was already gone", retired_key)
            else:
                logger.warning(
                    "Error cleaning retired environment for task %s: %s",
                    retired_key, exc,
                )
                _retire_replaced_environment(env, retired_key)
    return cleaned


def _get_modal_backend_state(modal_mode: object | None) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection."""
    return resolve_modal_backend_state(
        modal_mode,
        has_direct=has_direct_modal_credentials(),
        managed_ready=is_managed_tool_gateway_ready("modal"),
    )


def _create_environment(
    env_type: str,
    image: str,
    cwd: str,
    timeout: int,
    ssh_config: Optional[dict] = None,
    container_config: Optional[dict] = None,
    local_config: Optional[dict] = None,
    task_id: str = "default",
    host_cwd: Optional[str] = None,
):
    """
    Create an execution environment for sandboxed command execution.
    
    Args:
        env_type: One of "local", "docker", "singularity", "modal",
            "daytona", "vercel_sandbox", "ssh"
        image: Docker/Singularity/Modal image name (ignored for local/ssh/vercel)
        cwd: Working directory
        timeout: Default command timeout
        ssh_config: SSH connection config (for env_type="ssh")
        container_config: Resource config for container backends (cpu, memory, disk, persistent)
        task_id: Task identifier for environment reuse and snapshot keying
        host_cwd: Optional host working directory to bind into Docker when explicitly enabled
        
    Returns:
        Environment instance with execute() method
    """
    cc = container_config or {}
    cpu = cc.get("container_cpu", 1)
    memory = cc.get("container_memory", 5120)
    disk = cc.get("container_disk", 51200)
    persistent = cc.get("container_persistent", True)
    volumes = cc.get("docker_volumes", [])
    docker_forward_env = cc.get("docker_forward_env", [])
    docker_env = cc.get("docker_env", {})
    docker_extra_args = cc.get("docker_extra_args", [])
    docker_network = cc.get("docker_network", True)

    if env_type == "local":
        env = _LocalEnvironment(cwd=cwd, timeout=timeout)
        setattr(
            env, "_persistent",
            bool((local_config or {}).get("persistent", False)),
        )
        return env
    
    elif env_type == "docker":
        # One-shot orphan reaper: clean up labeled containers left behind by
        # prior Hermes processes that hit SIGKILL / OOM / a closed terminal
        # before the atexit cleanup hook could run.  Gated to once per
        # process so concurrent _create_environment calls (parallel
        # subagents, RL benchmarks) don't run the reaper N times.
        # Disable via ``terminal.docker_orphan_reaper: false`` (issue #20561).
        _maybe_reap_docker_orphans(cc)
        return _DockerEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=cpu, memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
            storage_task_id=cc.get("storage_task_id"),
            legacy_storage_task_id=cc.get("legacy_storage_task_id"),
            volumes=volumes,
            host_cwd=host_cwd,
            auto_mount_cwd=cc.get("docker_mount_cwd_to_workspace", False),
            forward_env=docker_forward_env,
            env=docker_env,
            run_as_host_user=cc.get("docker_run_as_host_user", False),
            network=docker_network,
            extra_args=docker_extra_args,
            persist_across_processes=cc.get("docker_persist_across_processes", True),
            shm_size=cc.get("docker_shm_size", "1g"),
        )
    
    elif env_type == "singularity":
        return _SingularityEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=cpu, memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
        )
    
    elif env_type == "modal":
        sandbox_kwargs = {}
        if cpu > 0:
            sandbox_kwargs["cpu"] = cpu
        if memory > 0:
            sandbox_kwargs["memory"] = memory
        if disk > 0:
            try:
                import inspect, modal
                if "ephemeral_disk" in inspect.signature(modal.Sandbox.create).parameters:
                    sandbox_kwargs["ephemeral_disk"] = disk
            except Exception:
                pass

        modal_state = _get_modal_backend_state(cc.get("modal_mode"))

        if modal_state["selected_backend"] == "managed":
            return _ManagedModalEnvironment(
                image=image, cwd=cwd, timeout=timeout,
                modal_sandbox_kwargs=sandbox_kwargs,
                persistent_filesystem=persistent, task_id=task_id,
            )

        if modal_state["selected_backend"] != "direct":
            if modal_state["managed_mode_blocked"]:
                raise ValueError(
                    "Modal backend is configured for managed mode, but "
                    "Nous Tool Gateway access is not currently available and no direct "
                    "Modal credentials/config were found. "
                    + nous_tool_gateway_unavailable_message(
                        "managed Modal execution",
                    )
                    + " Choose TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials."
                )
            if modal_state["mode"] == "managed":
                raise ValueError(
                    "Modal backend is configured for managed mode, but the managed tool gateway is unavailable. "
                    + nous_tool_gateway_unavailable_message(
                        "managed Modal execution",
                    )
                )
            if modal_state["mode"] == "direct":
                raise ValueError(
                    "Modal backend is configured for direct mode, but no direct Modal credentials/config were found."
                )
            message = "Modal backend selected but no direct Modal credentials/config was found."
            if managed_nous_tools_enabled():
                message = (
                    "Modal backend selected but no direct Modal credentials/config or managed tool gateway was found."
                )
            raise ValueError(message)

        return _ModalEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=persistent, task_id=task_id,
        )
    
    elif env_type == "daytona":
        # Lazy import so daytona SDK is only required when backend is selected.
        from tools.environments.daytona import DaytonaEnvironment as _DaytonaEnvironment
        return _DaytonaEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=int(cpu), memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
        )

    elif env_type == "vercel_sandbox":
        from tools.environments.vercel_sandbox import (
            VercelSandboxEnvironment as _VercelSandboxEnvironment,
        )
        return _VercelSandboxEnvironment(
            runtime=cc.get("vercel_runtime") or None,
            cwd=cwd,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            disk=disk,
            persistent_filesystem=persistent,
            task_id=task_id,
        )

    elif env_type == "ssh":
        if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
            raise ValueError("SSH environment requires ssh_host and ssh_user to be configured")
        return _SSHEnvironment(
            host=ssh_config["host"],
            user=ssh_config["user"],
            port=ssh_config.get("port", 22),
            key_path=ssh_config.get("key", ""),
            cwd=cwd,
            timeout=timeout,
            runtime_scope=ssh_config.get("runtime_scope", ""),
        )

    else:
        raise ValueError(
            f"Unknown environment type: {env_type}. Use 'local', 'docker', "
            f"'singularity', 'modal', 'daytona', 'vercel_sandbox', or 'ssh'"
        )


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    current_time = time.time()

    # Check the process registry -- skip cleanup for sandboxes with active
    # background processes (their _last_activity gets refreshed to keep them alive).
    try:
        from tools.process_registry import process_registry
        for task_id in list(_last_activity.keys()):
            if process_registry.has_active_processes(task_id):
                _last_activity[task_id] = current_time  # Keep sandbox alive
    except ImportError:
        pass

    # Phase 1: collect stale entries and remove them from tracking dicts while
    # holding the lock.  Do NOT call env.cleanup() inside the lock -- Modal and
    # Docker teardown can block for 10-15s, which would stall every concurrent
    # terminal/file tool call waiting on _env_lock.
    envs_to_stop = []  # list of (task_id, env) pairs

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if _active_turns_for_environment_key(task_id) > 0:
                # An active tool or overlapping logical turn owns this runtime.
                # Refresh activity so it gets a complete idle window afterward.
                _last_activity[task_id] = current_time
                continue
            tracked_env = _active_environments.get(task_id)
            effective_lifetime = getattr(
                tracked_env, "_hermes_lifetime_seconds", lifetime_seconds,
            )
            if current_time - last_time > effective_lifetime:
                env = _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
                if env is not None:
                    envs_to_stop.append((task_id, env))

        # Also purge per-task creation locks for cleaned-up tasks
        with _creation_locks_lock:
            for task_id, _ in envs_to_stop:
                _creation_locks.pop(task_id, None)

    # Phase 2: stop the actual sandboxes OUTSIDE the lock so other tool calls
    # are not blocked while Modal/Docker sandboxes shut down.
    for task_id, env in envs_to_stop:
        # Invalidate stale file_ops cache entry (Bug fix: prevents
        # ShellFileOperations from referencing a dead sandbox)
        try:
            from tools.file_tools import clear_file_ops_cache
            clear_file_ops_cache(task_id)
        except ImportError:
            pass

        try:
            if hasattr(env, 'cleanup'):
                env.cleanup()
            elif hasattr(env, 'stop'):
                env.stop()
            elif hasattr(env, 'terminate'):
                env.terminate()

            logger.info("Cleaned up inactive environment for task: %s", task_id)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", task_id)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", task_id, e)

    # Replaced environments are no longer selectable. Give concurrent foreground
    # calls a one-minute grace period, then force-remove them once no tool turn or
    # background process still references their task scope.
    _cleanup_retired_environments(min_age_seconds=60.0, require_idle=True)


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    while _cleanup_running:
        try:
            config = _get_env_config()
            _cleanup_inactive_envs(config["lifetime_seconds"])
        except Exception as e:
            logger.warning("Error in cleanup thread: %s", e, exc_info=True)

        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _env_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
            _cleanup_thread.start()


def _stop_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        try:
            _cleanup_thread.join(timeout=5)
        except (SystemExit, KeyboardInterrupt):
            pass


def get_active_env(task_id: str, target: Optional[str] = None):
    """Return the active BaseEnvironment for *task_id*, or None."""
    resolution = _target_resolution(target)
    lookup = _environment_scope_key(_resolve_container_task_id(task_id), resolution)
    raw_lookup = _environment_scope_key(task_id, resolution)
    with _env_lock:
        return _active_environments.get(lookup) or _active_environments.get(raw_lookup)


def get_environment_for_target_scope(
    task_id: str, target: str, runtime_scope: str,
):
    """Find the active/retired environment that produced a scoped result."""
    raw = task_id or "default"
    collapsed = _resolve_container_task_id(raw)
    bases = {
        _profile_scoped_task_key(raw),
        _profile_scoped_task_key(collapsed),
    }

    def _matches(key: Hashable, env: Any) -> bool:
        belongs = key in bases or (
            isinstance(key, tuple) and len(key) == 2 and key[0] in bases
        )
        return bool(
            belongs
            and getattr(env, "_hermes_target_name", None) == target
            and getattr(env, "_hermes_target_scope", None) == runtime_scope
        )

    with _env_lock:
        for key, env in _active_environments.items():
            if _matches(key, env):
                return env
    with _retired_environments_lock:
        for key, env, _retired_at in _retired_environments:
            if _matches(key, env):
                return env
    return None


def _environment_is_persistent(env: Any) -> bool:
    return bool(
        getattr(env, "_persistent", False)
        or getattr(env, "persistent_filesystem", False)
    )


def is_persistent_env(task_id: str, target: Optional[str] = None) -> bool:
    """Return True if the active environment for task_id is configured for
    cross-turn persistence (``persistent_filesystem=True``).

    Used by the agent loop to skip per-turn teardown for backends whose whole
    point is to survive between turns (docker with ``container_persistent``,
    daytona, modal, etc.). Non-persistent backends (e.g. Morph) still get torn
    down at end-of-turn to prevent leakage. The idle reaper
    (``_cleanup_inactive_envs``) handles persistent envs once they exceed
    ``terminal.lifetime_seconds``.
    """
    env = get_active_env(task_id, target=target)
    if env is None:
        return False
    return _environment_is_persistent(env)




def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    task_ids = list(_active_environments.keys())
    cleaned = 0
    
    for task_id in task_ids:
        try:
            cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e, exc_info=True)

    cleaned += _cleanup_retired_environments(
        min_age_seconds=0.0, require_idle=False,
    )

    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)
        except OSError as e:
            logger.debug("Failed to remove orphaned path %s: %s", path, e)
    
    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def cleanup_vm(
    task_id: Hashable,
    *,
    force_remove: bool = False,
    preserve_persistent: bool = False,
    target: Optional[str] = None,
    include_collapsed: bool = False,
):
    """Manually clean up a specific environment by task_id.

    *force_remove* (default False) is forwarded to backends that accept it
    — currently only ``DockerEnvironment``. ``preserve_persistent`` is used
    by per-turn cleanup to keep each persistent named sibling live while
    removing only non-persistent targets. The default of False matches
    session-lifecycle semantics: this function is called from
    ``AIAgent.close()`` (TUI session close, gateway session teardown) and the
    per-turn cleanup branch for non-persistent envs, both of which should
    honor the user's persist-mode preference. Stopping the container here
    would defeat the "ONE long-lived container shared across sessions"
    contract — exactly the bug Ben reported when the container was killed
    on every TUI session close.

    Pass ``force_remove=True`` for actual user-initiated teardown
    (e.g. ``/reset``-style flows that haven't been wired yet, or future
    "destroy my sandbox" commands).

    The idle reaper passes the env through ``env.cleanup()`` directly (not
    via this function), so persist-mode idle envs are similarly no-op'd —
    only the orphan reaper at next startup reclaims them.
    """
    # Direct tuple keys are used by global/idle cleanup. For a raw task,
    # omitted target cleans every target scope owned by that exact raw key;
    # an explicit target cleans exactly that scope. Do not collapse arbitrary
    # subagent ids to "default" here: legacy cleanup_vm(child_id) never tore
    # down the parent's shared environment, and doing so in named mode would
    # let a delegate's close race/disrupt its parent.
    if isinstance(task_id, tuple):
        keys = [task_id]
    elif target is None:
        try:
            resolution = _target_resolution(None)
            scoped_task_id = resolution.scope_task_key(task_id)
            collapsed_task_id = _resolve_container_task_id(str(task_id))
            scoped_collapsed_task_id = resolution.scope_task_key(collapsed_task_id)
        except Exception:
            scoped_task_id = task_id
            collapsed_task_id = task_id
            scoped_collapsed_task_id = task_id
        matching_task_ids = {task_id, scoped_task_id}
        if include_collapsed:
            matching_task_ids.update({
                collapsed_task_id, scoped_collapsed_task_id,
            })
        with _env_lock:
            keys = [
                key for key in _active_environments
                if key in matching_task_ids
                or (
                    isinstance(key, tuple) and len(key) == 2
                    and key[0] in matching_task_ids
                )
            ]
        if not keys:
            keys = [task_id]
    else:
        resolution = _target_resolution(target)
        if resolution.named:
            keys = [resolution.environment_key(task_id)]
        else:
            keys = [task_id]

    active_process_keys = set()
    if preserve_persistent:
        try:
            from tools.process_registry import process_registry

            active_process_keys = {
                key for key in keys
                if process_registry.has_active_processes(key)
            }
        except Exception:
            logger.debug(
                "Failed to inspect active processes before cleanup",
                exc_info=True,
            )

    envs = []
    removed_keys = []
    with _env_lock:
        for key in keys:
            existing = _active_environments.get(key)
            if key in active_process_keys:
                continue
            if (
                preserve_persistent
                and existing is not None
                and _environment_is_persistent(existing)
            ):
                continue
            env = _active_environments.pop(key, None)
            _last_activity.pop(key, None)
            removed_keys.append(key)
            if env is not None:
                envs.append((key, env))

    # Clean up per-task creation lock
    with _creation_locks_lock:
        for key in removed_keys:
            _creation_locks.pop(key, None)

    # Invalidate stale file_ops cache entry
    try:
        from tools.file_tools import clear_file_ops_cache
        for key in removed_keys:
            clear_file_ops_cache(key)
    except ImportError:
        pass

    for key in keys:
        _cleanup_retired_environments(
            task_key=key,
            min_age_seconds=0.0,
            require_idle=preserve_persistent,
        )

    if not envs:
        return

    for key, env in envs:
        try:
            if hasattr(env, 'cleanup'):
                # Pass force_remove only if the env's cleanup() accepts it
                # (DockerEnvironment after issue #20561; other backends don't).
                import inspect
                sig = inspect.signature(env.cleanup)
                if "force_remove" in sig.parameters:
                    env.cleanup(force_remove=force_remove)
                else:
                    env.cleanup()
            elif hasattr(env, 'stop'):
                env.stop()
            elif hasattr(env, 'terminate'):
                env.terminate()

            logger.info("Manually cleaned up environment for task: %s", key)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", key)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", key, e)


def _atexit_cleanup():
    """Stop cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    with _retired_environments_lock:
        retired = list(_retired_environments)
        _retired_environments.clear()
    envs_to_wait = list(_active_environments.values())
    if envs_to_wait:
        logger.info("Shutting down %d remaining sandbox(es)...", len(envs_to_wait))
        cleanup_all_environments()
    seen = {id(env) for env in envs_to_wait}
    for _, env, _ in retired:
        if id(env) in seen:
            continue
        seen.add(id(env))
        try:
            _cleanup_environment_resource(
                env,
                force_remove=True,
                preserve_storage=_environment_has_stable_storage(env),
            )
        except Exception as exc:
            logger.debug("retired environment cleanup raised on exit: %s", exc)
        envs_to_wait.append(env)

    # Block briefly so docker stop/rm actually completes before the interpreter
    # exits. Issue #20561 — without this join, daemon cleanup threads can be
    # torn down mid-`docker stop`, leaving exited containers on the host.
    for env in envs_to_wait:
        wait_fn = getattr(env, "wait_for_cleanup", None)
        if wait_fn is None:
            continue
        try:
            wait_fn(timeout=15.0)
        except Exception as exc:  # never block shutdown on a bad backend
            logger.debug("wait_for_cleanup raised on exit: %s", exc)

atexit.register(_atexit_cleanup)


# =============================================================================
# Exit Code Context for Common CLI Tools
# =============================================================================
# Many Unix commands use non-zero exit codes for informational purposes, not
# to indicate failure.  The model sees a raw exit_code=1 from `grep` and
# wastes a turn investigating something that just means "no matches".
# This lookup adds a human-readable note so the agent can move on.

def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """Return a human-readable note when a non-zero exit code is non-erroneous.

    Returns None when the exit code is 0 or genuinely signals an error.
    The note is appended to the tool result so the model doesn't waste
    turns investigating expected exit codes.
    """
    if exit_code == 0:
        return None

    # Extract the last command in a pipeline/chain — that determines the
    # exit code.  Handles  `cmd1 && cmd2`, `cmd1 | cmd2`, `cmd1; cmd2`.
    # Deliberately simple: split on shell operators and take the last piece.
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()

    # Get base command name (first word), stripping env var assignments
    # like  VAR=val cmd ...
    words = last_segment.split()
    base_cmd = ""
    for w in words:
        if "=" in w and not w.startswith("-"):
            continue  # skip VAR=val
        base_cmd = w.split("/")[-1]  # handle /usr/bin/grep -> grep
        break

    if not base_cmd:
        return None

    # Command-specific semantics
    semantics: dict[str, dict[int, str]] = {
        # grep/rg/ag/ack: 1=no matches found (normal), 2+=real error
        "grep":  {1: "No matches found (not an error)"},
        "egrep": {1: "No matches found (not an error)"},
        "fgrep": {1: "No matches found (not an error)"},
        "rg":    {1: "No matches found (not an error)"},
        "ag":    {1: "No matches found (not an error)"},
        "ack":   {1: "No matches found (not an error)"},
        # diff: 1=files differ (expected), 2+=real error
        "diff":  {1: "Files differ (expected, not an error)"},
        "colordiff": {1: "Files differ (expected, not an error)"},
        # find: 1=some dirs inaccessible but results may still be valid
        "find":  {1: "Some directories were inaccessible (partial results may still be valid)"},
        # test/[: 1=condition is false (expected)
        "test":  {1: "Condition evaluated to false (expected, not an error)"},
        "[":     {1: "Condition evaluated to false (expected, not an error)"},
        # curl: common non-error codes
        "curl":  {
            6: "Could not resolve host",
            7: "Failed to connect to host",
            22: "HTTP response code indicated error (e.g. 404, 500)",
            28: "Operation timed out",
        },
        # git: 1 is context-dependent but often normal (e.g. git diff with changes)
        "git":   {1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"},
    }

    cmd_semantics = semantics.get(base_cmd)
    if cmd_semantics and exit_code in cmd_semantics:
        return cmd_semantics[exit_code]

    return None


def _command_requires_pipe_stdin(command: str) -> bool:
    """Return True when PTY mode would break stdin-driven commands.

    Some CLIs change behavior when stdin is a TTY. In particular,
    `gh auth login --with-token` expects the token to arrive via piped stdin and
    waits for EOF; when we launch it under a PTY, `process.submit()` only sends a
    newline, so the command appears to hang forever with no visible progress.
    """
    normalized = " ".join(command.lower().split())
    return (
        normalized.startswith("gh auth login")
        and "--with-token" in normalized
    )


_SHELL_LEVEL_BACKGROUND_RE = re.compile(
    r"(?:^|[;&|]\s*|&&\s*|\|\|\s*|\$\(\s*)(?:nohup|disown|setsid)\b", re.IGNORECASE | re.MULTILINE
)
_INLINE_BACKGROUND_AMP_RE = re.compile(r"\s&\s")
_TRAILING_BACKGROUND_AMP_RE = re.compile(r"\s&\s*(?:#.*)?$")


def _strip_quotes(command: str) -> str:
    """Remove single- and double-quoted content so regex checks don't match inside strings.

    This prevents false positives when keywords like 'nohup' or 'setsid' appear
    in commit messages, Python -c code, echo arguments, or PR body text.
    Also strips backtick-quoted content and heredoc-style inline text.
    """
    # Remove single-quoted strings (no escaping inside single quotes in shell)
    result = re.sub(r"'[^']*'", "''", command)
    # Remove double-quoted strings (handle escaped quotes)
    result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', result)
    # Remove backtick-quoted strings
    result = re.sub(r"`[^`]*`", "``", result)
    return result


_LONG_LIVED_FOREGROUND_PATTERNS = (
    re.compile(r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|watch)\b", re.IGNORECASE),
    re.compile(r"\bdocker\s+compose\s+up\b", re.IGNORECASE),
    re.compile(r"\bnext\s+dev\b", re.IGNORECASE),
    re.compile(r"\bvite(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bnodemon\b", re.IGNORECASE),
    re.compile(r"\buvicorn\b", re.IGNORECASE),
    re.compile(r"\bgunicorn\b", re.IGNORECASE),
    re.compile(r"\bpython(?:3)?\s+-m\s+http\.server\b", re.IGNORECASE),
)


def _looks_like_help_or_version_command(command: str) -> bool:
    """Return True for informational invocations that should never be blocked."""
    normalized = " ".join(command.lower().split())
    return (
        " --help" in normalized
        or normalized.endswith(" -h")
        or " --version" in normalized
        or normalized.endswith(" -v")
    )


def _foreground_background_guidance(command: str) -> str | None:
    """Suggest background mode when a foreground command looks long-lived.

    Prevents workflows that start a server/watch process and then stall before
    follow-up checks or test commands run.
    """
    if _looks_like_help_or_version_command(command):
        return None

    # Strip quoted content so keywords inside strings/arguments don't trigger
    # false positives (e.g., git commit -m "... setsid ...", python3 -c "os.setsid").
    unquoted = _strip_quotes(command)

    if _SHELL_LEVEL_BACKGROUND_RE.search(unquoted):
        return (
            "Foreground command uses shell-level background wrappers (nohup/disown/setsid). "
            "Re-send WITHOUT the wrapper as terminal(command=\"<cmd>\", background=true, "
            "notify_on_complete=true) so Hermes tracks the process, then run readiness "
            "checks and tests in separate commands."
        )

    if _INLINE_BACKGROUND_AMP_RE.search(unquoted) or _TRAILING_BACKGROUND_AMP_RE.search(unquoted):
        return (
            "Foreground command uses '&' backgrounding. Re-send WITHOUT the '&' as "
            "terminal(command=\"<cmd>\", background=true) — add notify_on_complete=true "
            "for bounded jobs — then run health checks and tests in follow-up terminal calls."
        )

    for pattern in _LONG_LIVED_FOREGROUND_PATTERNS:
        if pattern.search(unquoted):
            return (
                "This foreground command appears to start a long-lived server/watch process. "
                "Run it with background=true, verify readiness (health endpoint/log signal), "
                "then execute tests in a separate command."
            )

    return None


def _resolve_notification_flag_conflict(
    *,
    notify_on_complete: bool,
    watch_patterns,
    background: bool,
) -> tuple:
    """Decide what to do when both notify_on_complete and watch_patterns are set.

    These flags produce duplicate, delayed notifications when combined — one
    notification per watch-pattern match AND one on process exit, with async
    delivery that can spam the user long after the process ends. When both are
    set, we drop watch_patterns in favor of notify_on_complete (the more useful
    "let me know when it's done" signal) and return a human-readable note.

    Returns:
        (watch_patterns_to_use, conflict_note). conflict_note is "" when there
        is no conflict.
    """
    if background and notify_on_complete and watch_patterns:
        note = (
            "watch_patterns ignored because notify_on_complete=True; "
            "these two flags produce duplicate notifications when combined"
        )
        return None, note
    return watch_patterns, ""


def _resolve_command_cwd(
    *,
    workdir: Optional[str],
    default_cwd: str,
    session_key: Optional[str] = None,
    target: Optional[str] = None,
    _resolution=None,
) -> str:
    """Return the cwd for a command. Explicit ``workdir=`` overrides everything.

    Otherwise the session's own cwd RECORD (``get_session_cwd``) wins — it is
    written after every completed command for this session, so it IS the
    session's ``cd`` state, with no shared-env ambiguity: another session's
    ``cd`` lands in another record and can't affect us. A session with no
    record yet (first command) runs in ``default_cwd`` (config/override cwd),
    which is also what seeds a fresh environment.
    """
    if workdir:
        return workdir
    return get_session_cwd(
        session_key, target=target, _resolution=_resolution,
    ) or default_cwd


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: Optional[List[str]] = None,
    target: Optional[str] = None,
) -> str:
    """
    Execute a command in the configured terminal environment.

    Args:
        command: The command to execute
        background: Whether to run in background (default: False)
        timeout: Command timeout in seconds (default: from config)
        task_id: Unique identifier for environment isolation (optional)
        session_id: Conversation/session identifier for durable observability
        force: If True, skip dangerous command check (use after user confirms)
        workdir: Working directory for this command (optional, uses session cwd if not set)
        pty: If True, use pseudo-terminal for interactive CLI tools (local backend only)
        notify_on_complete: If True and background=True, you'll be notified exactly once when the process exits. The right choice for almost every long task. MUTUALLY EXCLUSIVE with watch_patterns.
        watch_patterns: List of strings to watch for in background output. HARD rate limit: 1 notification per 15s per process. After 3 strike windows in a row, watch_patterns is disabled and the session is auto-promoted to notify_on_complete. Use ONLY for rare, one-shot mid-process signals on long-lived processes (server readiness, migration-done markers). NEVER use in loops/batch jobs — error patterns there will hit the strike limit and get disabled. MUTUALLY EXCLUSIVE with notify_on_complete — set one, not both.
        target: Named execution target. Omit to use the configured default.

    Returns:
        str: JSON string with output, exit_code, and error fields

    Examples:
        # Execute a simple command
        >>> result = terminal_tool(command="ls -la /tmp")

        # Run a background task
        >>> result = terminal_tool(command="python server.py", background=True)

        # With custom timeout
        >>> result = terminal_tool(command="long_task.sh", timeout=300)
        
        # Force run after user confirmation
        # Note: force parameter is internal only, not exposed to model API
    """
    try:
        if not isinstance(command, str):
            logger.warning(
                "Rejected invalid terminal command value: %s",
                type(command).__name__,
            )
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": f"Invalid command: expected string, got {type(command).__name__}",
                "status": "error",
            }, ensure_ascii=False)

        # Resolve configuration per call. Named targets read merged config
        # directly; legacy flat config keeps the existing env-driven path.
        from tools.execution_targets import ExecutionTargetError
        try:
            target_resolution = _target_resolution(target)
        except ExecutionTargetError as exc:
            return json.dumps({
                "output": "", "exit_code": -1, "error": str(exc), "status": "error",
            }, ensure_ascii=False)
        config = (
            _get_env_config(dict(target_resolution.config))
            if target_resolution.named else _get_env_config()
        )
        env_type = config["env_type"]

        # Use task_id for environment isolation. By default all subagent
        # task_ids collapse back to "default" so the top-level agent and
        # every delegate_task child share one container; only task_ids with
        # a registered env override (RL benchmarks) get isolated sandboxes.
        effective_base_task_id = _resolve_container_task_id(task_id)
        effective_task_id = _environment_scope_key(
            effective_base_task_id, target_resolution,
        )
        raw_environment_key = _environment_scope_key(
            task_id, target_resolution,
        ) if task_id else None
        backend_task_id = target_resolution.backend_task_id(effective_base_task_id)

        # Check per-task overrides (set by environments like TerminalBench2Env)
        # before falling back to global env var config. ``resolve_task_overrides``
        # reads the raw task id first then the collapsed container id, so a
        # CWD-only override (which collapses ``effective_task_id`` to
        # ``"default"``) is still found under its originating session id while
        # isolation-keyed RL/benchmark overrides keep resolving as before.
        overrides = resolve_task_overrides(task_id)
        
        # Select image based on env type, with per-task override support
        if env_type == "docker":
            image = overrides.get("docker_image") or config["docker_image"]
        elif env_type == "singularity":
            image = overrides.get("singularity_image") or config["singularity_image"]
        elif env_type == "modal":
            image = overrides.get("modal_image") or config["modal_image"]
        elif env_type == "daytona":
            image = overrides.get("daytona_image") or config["daytona_image"]
        else:
            image = ""

        cwd_override = (
            overrides.get("cwd")
            if (
                not target_resolution.named
                or (
                    target_resolution.is_default
                    and target_resolution.backend != "ssh"
                )
            )
            else None
        )
        cwd = cwd_override or get_session_cwd(
            task_id, _resolution=target_resolution,
        ) or config["cwd"]
        cwd = _apply_task_cwd_override(config, cwd, cwd_override)
        default_timeout = config["timeout"]

        # Validate an explicit timeout before it flows into deadline math.
        # ``timeout or default`` silently turns 0 into the default (0 can't mean
        # "no timeout" here), and a negative value is truthy so it would sail
        # through to ``deadline = now + timeout`` and fire an immediate,
        # nonsensical "-Ns" timeout. Reject non-positive values outright.
        if timeout is not None and timeout <= 0:
            return tool_error(
                f"timeout must be a positive number of seconds (got {timeout})."
            )
        effective_timeout = timeout or default_timeout

        # Reject foreground commands where the model explicitly requests
        # a timeout above FOREGROUND_MAX_TIMEOUT — nudge it toward background.
        if not background and timeout and timeout > FOREGROUND_MAX_TIMEOUT:
            return tool_error(
                f"Foreground timeout {timeout}s exceeds the maximum of "
                f"{FOREGROUND_MAX_TIMEOUT}s. Use background=true with "
                f"notify_on_complete=true for long-running commands."
            )

        # Guardrail: long-lived server/watch commands should run as managed
        # background sessions, not foreground shell hacks.
        if not background:
            guidance = _foreground_background_guidance(command)
            if guidance:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": guidance,
                    "status": "error",
                }, ensure_ascii=False)

        # Start cleanup thread
        _start_cleanup_thread()

        # Get or create environment.
        # Use a per-task creation lock so concurrent tool calls for the same
        # task_id wait for the first one to finish creating the sandbox,
        # instead of each creating their own (wasting Modal resources).
        env: Any = None
        with _env_lock:
            # Prefer the collapsed container id, but fall back to an env cached
            # under the raw task_id. Per-session surfaces (ACP/gateway/dashboard)
            # with a CWD-only override collapse to "default" for container
            # sharing, yet an env may already be cached under the originating
            # task_id; honor it instead of spawning a duplicate.
            _existing_key = (
                effective_task_id if effective_task_id in _active_environments
                else (raw_environment_key if raw_environment_key in _active_environments else None)
            )
            if (
                _existing_key is not None
                and _environment_matches_target(
                    _active_environments[_existing_key], target_resolution,
                )
            ):
                _last_activity[_existing_key] = time.time()
                env = _active_environments[_existing_key]
                needs_creation = False
            else:
                needs_creation = True

        if needs_creation:
            # Per-task lock: only one thread creates the sandbox, others wait
            with _creation_locks_lock:
                if effective_task_id not in _creation_locks:
                    _creation_locks[effective_task_id] = threading.Lock()
                task_lock = _creation_locks[effective_task_id]

            with task_lock:
                # Double-check after acquiring the per-task lock
                existing_env = None
                existing_key = effective_task_id
                with _env_lock:
                    _existing_key = (
                        effective_task_id if effective_task_id in _active_environments
                        else (raw_environment_key if raw_environment_key in _active_environments else None)
                    )
                    if (
                        _existing_key is not None
                        and _environment_matches_target(
                            _active_environments[_existing_key], target_resolution,
                        )
                    ):
                        _last_activity[_existing_key] = time.time()
                        env = _active_environments[_existing_key]
                        needs_creation = False
                    elif _existing_key is not None:
                        existing_env = _active_environments[_existing_key]
                        existing_key = _existing_key

                try:
                    _prepare_environment_replacement(
                        existing_env,
                        existing_key,
                        target_name=target_resolution.target,
                    )
                except _EnvironmentReplacementError as exc:
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": str(exc),
                        "status": "error",
                    }, ensure_ascii=False)

                if needs_creation:
                    if env_type == "singularity":
                        _check_disk_usage_warning()
                    logger.info("Creating new %s environment for task %s...", env_type, effective_task_id)
                    try:
                        container_config, ssh_config, local_config = (
                            _build_environment_constructor_configs(
                                config, target_resolution, effective_base_task_id,
                            )
                        )

                        new_env = _create_environment(
                            env_type=env_type,
                            image=image,
                            cwd=cwd,
                            timeout=effective_timeout,
                            ssh_config=ssh_config,
                            container_config=container_config,
                            local_config=local_config,
                            task_id=backend_task_id,
                            host_cwd=config.get("host_cwd"),
                        )
                        _record_environment_lifetime(new_env, config)
                        _record_environment_target(new_env, target_resolution)
                    except ImportError as e:
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": f"Terminal tool disabled: environment creation failed ({e})",
                            "status": "disabled"
                        }, ensure_ascii=False)

                    publish_error = None
                    with _env_lock:
                        if target_resolution.named:
                            try:
                                from tools.execution_targets import (
                                    execution_target_config_is_frozen,
                                    resolve_live_execution_target,
                                )

                                live_resolution = (
                                    target_resolution
                                    if execution_target_config_is_frozen()
                                    else resolve_live_execution_target(target)
                                )
                            except Exception as exc:
                                publish_error = str(exc)
                            else:
                                if (
                                    live_resolution.security_scope
                                    != target_resolution.security_scope
                                ):
                                    publish_error = (
                                        f"Execution target {target_resolution.target!r} "
                                        "changed while its environment was being created."
                                    )
                        replaced_envs = []
                        if publish_error is None:
                            current = _active_environments.get(effective_task_id)
                            if current is not None and current is not new_env:
                                replaced_envs.append((effective_task_id, current))
                            if (
                                raw_environment_key is not None
                                and raw_environment_key != effective_task_id
                            ):
                                raw_env = _active_environments.get(raw_environment_key)
                                if (
                                    raw_env is not None
                                    and not _environment_matches_target(
                                        raw_env, target_resolution,
                                    )
                                ):
                                    _active_environments.pop(raw_environment_key, None)
                                    _last_activity.pop(raw_environment_key, None)
                                    replaced_envs.append((raw_environment_key, raw_env))
                            _active_environments[effective_task_id] = new_env
                            _last_activity[effective_task_id] = time.time()
                            env = new_env
                    if publish_error is not None:
                        _cleanup_environment_resource(
                            new_env,
                            force_remove=True,
                            preserve_storage=_environment_has_stable_storage(new_env),
                        )
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": publish_error + " Retry the command.",
                            "status": "error",
                        }, ensure_ascii=False)
                    for replaced_key, replaced_env in replaced_envs:
                        if replaced_env is not new_env:
                            _retire_replaced_environment(replaced_env, replaced_key)
                    logger.info("%s environment ready for task %s", env_type, effective_task_id)

        assert env is not None  # all creation failure paths return above

        # The session key that drives cwd records: get_current_session_key()'s
        # contextvar doesn't cross tool-worker threads, so fall back to the raw
        # task_id (which IS the session_key for the top-level agent) — a
        # stable, thread-safe anchor.
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="") or (task_id or "")

        # Hard-block: gateway lifecycle commands (systemctl/launchctl/hermes
        # restart|stop targeting hermes-gateway) must never run inside the
        # gateway process itself. The restart would SIGTERM the gateway, which
        # kills this very subprocess before it can complete — the service may
        # never restart. This mirrors the `hermes gateway restart` guard in
        # hermes_cli/gateway.py and the cron-path guard in hermes_cli/cron.py,
        # but applies unconditionally (force=True cannot help here).
        if os.environ.get("_HERMES_GATEWAY") == "1":
            from cron.lifecycle_guard import (
                contains_gateway_lifecycle_command_or_referenced_script,
                contains_launchctl_submit_command,
            )
            if contains_launchctl_submit_command(command):
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": (
                        "Blocked: launchctl submit/bootstrap registers a persistent "
                        "KeepAlive job and is unsafe from inside the gateway process. "
                        "Use Hermes cron for one-shot delayed work, or install an "
                        "explicit LaunchAgent from a separate shell."
                    ),
                    "status": "error",
                }, ensure_ascii=False)
            selected_target = (
                target_resolution.target if target_resolution.named else None
            )
            guard_cwd_base = get_session_cwd(
                session_key, selected_target, _resolution=target_resolution,
            )
            if guard_cwd_base is None:
                guard_cwd_base = getattr(env, "cwd", None) or cwd
            guard_cwd = _resolve_command_cwd(
                workdir=workdir,
                default_cwd=guard_cwd_base,
                session_key=session_key,
                _resolution=target_resolution,
            )

            def _read_script_in_env(script_path: str) -> Optional[str]:
                # Read only through the selected target and selected cwd.
                if env is None:
                    return None
                if target_resolution.backend == "local":
                    try:
                        local_path = Path(script_path).expanduser()
                        if not local_path.is_absolute():
                            local_path = Path(guard_cwd) / local_path
                        if local_path.is_file():
                            metadata = local_path.stat()
                            if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= 1024 * 1024:
                                data = local_path.read_bytes()
                                if len(data) <= 1024 * 1024:
                                    return data.decode("utf-8", errors="replace")
                    except Exception:
                        pass
                try:
                    result = env.execute(
                        f"cat {shlex.quote(script_path)}", cwd=guard_cwd,
                    )
                    returncode = (
                        result.get("returncode", result.get("exit_code", -1))
                        if isinstance(result, dict)
                        else getattr(result, "exit_code", -1)
                    )
                    if returncode == 0:
                        return (
                            result.get("output", "")
                            if isinstance(result, dict)
                            else getattr(result, "output", "")
                        )
                except Exception:
                    pass
                return None

            if contains_gateway_lifecycle_command_or_referenced_script(
                command,
                cwd=guard_cwd,
                read_remote_script=_read_script_in_env,
            ):
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": (
                        "Blocked: command or referenced script cannot restart or stop "
                        "the gateway from inside the gateway process. The gateway would "
                        "kill this command before it could complete (SIGTERM propagates "
                        "to child processes). Run `hermes gateway restart` from a "
                        "separate shell outside the running gateway."
                    ),
                    "status": "error",
                }, ensure_ascii=False)

        # Pre-exec security checks (tirith + dangerous command detection)
        # Skip check if force=True (user has confirmed they want to run it)
        approval_note = None
        # True when the user explicitly approved this run (or pre-confirmed via
        # force).  Drives the clean-interrupt-slate clear before env.execute so
        # an approved command can't be SIGINT-killed by a bit that landed during
        # the approval-wait (see clear_current_thread_interrupt).
        _approved_run = bool(force)
        if not force:
            approval = _check_all_guards(
                command, env_type,
                has_host_access=_docker_has_host_access(config),
                execution_target=target_resolution.target,
                execution_backend=target_resolution.backend,
                execution_target_named=target_resolution.named,
                execution_target_scope=(
                    target_resolution.security_scope
                    if target_resolution.named else ""
                ),
            )
            if not approval["approved"]:
                # Check if this is an approval_required (gateway ask mode)
                if approval.get("status") == "pending_approval":
                    pending_result = {
                        "output": "",
                        "exit_code": -1,
                        "error": "",
                        "status": "pending_approval",
                        "approval_pending": True,
                        "command": approval.get("command", command),
                        "description": approval.get("description", "command flagged"),
                        "pattern_key": approval.get("pattern_key", ""),
                        "smart_denied": approval.get("smart_denied", False),
                        "allow_permanent": approval.get("allow_permanent", True),
                    }
                    pending_result.update(target_resolution.metadata(
                        cwd=cwd if target_resolution.named else None,
                    ))
                    return json.dumps(pending_result, ensure_ascii=False)
                # Command was blocked
                desc = approval.get("description", "command flagged")
                fallback_msg = (
                    f"Command denied: {desc}. "
                    "Use the approval prompt to allow it, or rephrase the command."
                )
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": approval.get("message", fallback_msg),
                    "status": "blocked"
                }, ensure_ascii=False)
            # Track whether approval was explicitly granted by the user
            if approval.get("user_approved"):
                desc = approval.get("description", "flagged as dangerous")
                approval_note = f"Command required approval ({desc}) and was approved by the user."
                _approved_run = True
            elif approval.get("smart_approved"):
                desc = approval.get("description", "flagged as dangerous")
                approval_note = f"Command was flagged ({desc}) and auto-approved by smart approval."

        # Validate workdir against shell injection
        if workdir:
            workdir_error = _validate_workdir(workdir)
            if workdir_error:
                logger.warning("Blocked dangerous workdir: %s (command: %s)",
                               workdir[:200], _safe_command_preview(command))
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": workdir_error,
                    "status": "blocked"
                }, ensure_ascii=False)

        # Prepare command for execution
        pty_disabled_reason = None
        effective_pty = pty
        if pty and _command_requires_pipe_stdin(command):
            effective_pty = False
            pty_disabled_reason = (
                "PTY disabled for this command because it expects piped stdin/EOF "
                "(for example gh auth login --with-token). For local background "
                "processes, call process(action='close') after writing so it receives "
                "EOF."
            )

        # The session key is already computed above the gateway guard.
        if background:
            # Spawn a tracked background process via the process registry.
            # For local backends: uses subprocess.Popen with output buffering.
            # For non-local backends: runs inside the sandbox via env.execute().
            from tools.process_registry import process_registry

            effective_cwd = _resolve_command_cwd(
                workdir=workdir,
                default_cwd=cwd,
                session_key=session_key,
                _resolution=target_resolution,
            )
            try:
                spawn_metadata = {}
                environment_task_key = str(
                    target_resolution.scope_task_key(effective_base_task_id)
                )
                if target_resolution.named:
                    spawn_metadata = {
                        "target": target_resolution.target,
                        "backend": target_resolution.backend,
                        "timeout_seconds": effective_timeout,
                        "environment_task_key": environment_task_key,
                        "runtime_scope": target_resolution.security_scope,
                    }
                    if env_type == "local":
                        spawn_metadata["env_ref"] = env
                with _scoped_sudo_execution(
                    target_resolution.target,
                    target_resolution.backend,
                    named=target_resolution.named,
                    sudo_password=target_resolution.config.get("sudo_password"),
                    target_scope=(
                        target_resolution.security_scope
                        if target_resolution.named else ""
                    ),
                ):
                    if env_type == "local":
                        proc_session = process_registry.spawn_local(
                            command=command,
                            cwd=effective_cwd,
                            task_id=effective_base_task_id,
                            session_key=session_key,
                            env_vars=env.env if hasattr(env, 'env') else None,
                            use_pty=effective_pty,
                            **spawn_metadata,
                        )
                    else:
                        proc_session = process_registry.spawn_via_env(
                            env=env,
                            command=command,
                            cwd=effective_cwd,
                            task_id=effective_base_task_id,
                            session_key=session_key,
                            timeout=effective_timeout,
                            **spawn_metadata,
                        )

                # Preserve the exact legacy spawn call signature while still
                # attaching additive metadata to subsequent process results.
                if not target_resolution.named:
                    proc_session.target = target_resolution.target
                    proc_session.backend = target_resolution.backend
                    proc_session.timeout_seconds = effective_timeout
                    proc_session.environment_task_key = environment_task_key
                    checkpoint = getattr(process_registry, "_write_checkpoint", None)
                    if callable(checkpoint):
                        checkpoint()

                result_data = {
                    "output": "Background process started",
                    "session_id": proc_session.id,
                    "pid": proc_session.pid,
                    "exit_code": 0,
                    "error": None,
                }
                result_data.update(target_resolution.metadata(
                    cwd=effective_cwd if target_resolution.named else None,
                ))
                # Background spawns detached and returns exit_code 0 immediately;
                # it never inline-polls is_interrupted(), so the stale-bit kill
                # cannot occur here and this note never co-occurs with rc=130.
                if approval_note:
                    result_data["approval"] = approval_note
                if pty_disabled_reason:
                    result_data["pty_note"] = pty_disabled_reason

                # Nudge: background=True without notify_on_complete=True OR
                # watch_patterns is a silent process. The agent has NO way to
                # learn it finished short of calling process(action="poll"/"wait")
                # explicitly. That's correct only for genuine long-lived
                # processes that never exit (servers, watchers). For every
                # bounded task (tests, builds, CI pollers, deploys, batch
                # jobs) the agent almost certainly wanted notification and
                # forgot the flag. May 2026 PR #31231 incident: bg CI poller
                # ran fine, exited green, agent never noticed — user had to
                # surface the result. Cheap nudge here costs ~one read for
                # server cases (false positive) and prevents silent
                # blindness for bounded-task cases (false negative).
                if background and not notify_on_complete and not watch_patterns:
                    result_data["hint"] = (
                        "background=true without notify_on_complete=true means "
                        "this process runs SILENTLY — you will not be told when "
                        "it exits. If this is a bounded task (test suite, build, "
                        "CI poller, deploy, anything with a defined end), you "
                        "almost certainly wanted notify_on_complete=true so the "
                        "system pings you on exit. Re-launch with "
                        "notify_on_complete=true, or call process(action='poll') "
                        "/ process(action='wait') yourself to learn the outcome. "
                        "Only ignore this hint for genuine long-lived processes "
                        "that never exit (servers, watchers, daemons)."
                    )

                # Nudge: homebrewed CI watcher built from `gh pr view`
                # `--json statusCheckRollup` or `gh pr checks` piped through
                # `jq` is the #1 cause of silent CI-watcher failures in
                # hermes-agent dev work. May 2026 PRs that surfaced this
                # exact failure mode: #31329, #31448, #31695, #31709, #31745,
                # #32264, #33131. Failure modes seen:
                #   * `gh pr view --json statusCheckRollup --jq ...` with
                #     `from_entries` choking on null `conclusion` keys, loop
                #     silently exits with empty status, never terminates.
                #   * `for i in $(seq 1 60); do ... 2>&1` block-buffered stdout
                #     never flushed to background-process capture; SIGTERM
                #     cuts the buffer before flush; `process(action='log')`
                #     returns total_lines=0 forever.
                #   * conclusion vs. status field confusion: filtering for
                #     `PENDING` in `.conclusion` while in-progress checks have
                #     empty conclusion → poller declares all-green while 18/23
                #     checks still IN_PROGRESS.
                #   * grepping for TTY-only banners ("All checks were
                #     successful") that never appear when stdout is piped.
                # The canonical patterns in the green-ci-policy skill avoid
                # every one of these — drive the loop off exit codes or on
                # tab-separated `awk -F"\t" "$2==\"pending\""` (column 2).
                # The detector here is deliberately narrow: it flags the
                # statusCheckRollup JSON-API path and the `gh pr checks` +
                # jq combination, but NOT the canonical column-2 awk
                # poller (which uses awk on tabs, not as a generic
                # stdout parser). When we detect the homebrew shape, point
                # the agent at the canonical snippet rather than letting
                # it ship another broken poller.
                if background and command:
                    _gh = ("gh pr view" in command or "gh pr checks" in command)
                    _has_jq = (
                        " jq " in command or "| jq" in command or "$(jq" in command
                    )
                    _bad_shape = (
                        # The JSON-API anti-pattern. Even without jq, going
                        # through `--json statusCheckRollup` + parsing puts
                        # you in conclusion-vs-status field hell.
                        "statusCheckRollup" in command
                        # gh pr checks piped to jq is also wrong — `gh pr
                        # checks` doesn't emit JSON, so any `| jq` here is
                        # confused intent. The canonical column-2 poller
                        # uses awk-on-tabs, not jq.
                        or (_gh and _has_jq)
                    )
                    if _bad_shape:
                        existing = result_data.get("hint", "")
                        canonical_hint = (
                            "This looks like a homebrewed CI poller built from "
                            "`gh pr view --json statusCheckRollup` and/or "
                            "`gh pr checks | jq`. That shape has burned us "
                            "repeatedly in hermes-agent dev work (PRs #31329, "
                            "#31448, #31695, #31709, #31745, #32264, #33131) — "
                            "stdout buffering kills output capture, jq null-key "
                            "edge cases silently exit the loop, conclusion-vs-"
                            "status field confusion exits early with bogus "
                            "all-green verdicts, TTY-only summary banners "
                            "never appear when piped. Use the canonical "
                            "snippets in the green-ci-policy skill instead: "
                            "the exit-code-driven `gh pr checks $PR >/dev/null` "
                            "(rc 0 = green, 8 = pending, else fail) for "
                            "exit-on-first-fail behavior, or the column-2 "
                            "awk-on-tabs poller "
                            "(`awk -F\"\\t\" \"$2==\\\"pending\\\"\"`) for "
                            "sharded matrices. Load skill_view("
                            "name='github/hermes-agent-dev', "
                            "file_path='references/green-ci-policy.md') for "
                            "the verbatim snippets. If you must roll a custom "
                            "loop with rich structured output, write each tick "
                            "to a known file (`tee -a /tmp/ci.log`) and rely "
                            "on `process(action='log')` to read THAT file — "
                            "do not rely on background-process stdout capture "
                            "for line-buffered shell loops."
                        )
                        result_data["hint"] = (
                            existing + "\n\n" + canonical_hint if existing
                            else canonical_hint
                        )

                # Populate routing metadata on the session so that
                # watch-pattern and completion notifications can be
                # routed back to the correct chat/thread.
                if background and (notify_on_complete or watch_patterns):
                    from gateway.session_context import (
                        async_delivery_supported as _async_ok,
                        get_session_env as _gse,
                    )

                    # Finite sessions (stateless HTTP requests and one-shot
                    # Kanban workers) cannot route a completion back to the
                    # agent after the turn/process ends. Refuse the promise:
                    # drop the flags and tell the agent to poll.
                    if not _async_ok():
                        notify_on_complete = False
                        watch_patterns = None
                        result_data["notify_on_complete"] = False
                        result_data["notify_unsupported"] = (
                            "notify_on_complete / watch_patterns are not available in "
                            "this session — it cannot receive an async completion after "
                            "the turn ends (a one-shot runner such as `hermes -z`, a "
                            "cron job, a Kanban worker, or a stateless HTTP endpoint). "
                            "The process is "
                            "running in the background; retrieve its result with "
                            "process(action='poll') or process(action='wait')."
                        )
                        logger.info(
                            "background proc %s: async delivery unsupported on this "
                            "session; notify_on_complete/watch_patterns disabled",
                            proc_session.id,
                        )
                    else:
                        _gw_platform = _gse("HERMES_SESSION_PLATFORM", "")
                        if _gw_platform:
                            _gw_chat_id = _gse("HERMES_SESSION_CHAT_ID", "")
                            _gw_thread_id = _gse("HERMES_SESSION_THREAD_ID", "")
                            _gw_user_id = _gse("HERMES_SESSION_USER_ID", "")
                            _gw_user_name = _gse("HERMES_SESSION_USER_NAME", "")
                            _gw_message_id = _gse("HERMES_SESSION_MESSAGE_ID", "")
                            proc_session.watcher_platform = _gw_platform
                            proc_session.watcher_chat_id = _gw_chat_id
                            proc_session.watcher_user_id = _gw_user_id
                            proc_session.watcher_user_name = _gw_user_name
                            proc_session.watcher_thread_id = _gw_thread_id
                            proc_session.watcher_message_id = _gw_message_id

                # Mutual exclusion: if both notify_on_complete and watch_patterns
                # are set, drop watch_patterns. The combination produces duplicate
                # notifications (one per match + one on exit) that deliver
                # asynchronously and can spam the user long after the process ends.
                # notify_on_complete is the more useful signal for "let me know
                # when the task finishes"; watch_patterns should be reserved for
                # standalone mid-process signals on long-lived processes.
                watch_patterns, conflict_note = _resolve_notification_flag_conflict(
                    notify_on_complete=bool(notify_on_complete),
                    watch_patterns=watch_patterns,
                    background=bool(background),
                )
                if conflict_note:
                    logger.warning("background proc %s: %s", proc_session.id, conflict_note)
                    result_data["watch_patterns_ignored"] = conflict_note

                # Mark for agent notification on completion
                if notify_on_complete and background:
                    proc_session.notify_on_complete = True
                    result_data["notify_on_complete"] = True

                    # In gateway mode, auto-register a fast watcher so the
                    # gateway can detect completion and trigger a new agent
                    # turn.  CLI mode uses the completion_queue directly.
                    if proc_session.watcher_platform:
                        proc_session.watcher_interval = 5
                        process_registry.pending_watchers.append({
                            "session_id": proc_session.id,
                            "check_interval": 5,
                            "session_key": session_key,
                            "platform": proc_session.watcher_platform,
                            "chat_id": proc_session.watcher_chat_id,
                            "user_id": proc_session.watcher_user_id,
                            "user_name": proc_session.watcher_user_name,
                            "thread_id": proc_session.watcher_thread_id,
                            "message_id": proc_session.watcher_message_id,
                            "notify_on_complete": True,
                        })

                # Set watch patterns for output monitoring
                if watch_patterns and background:
                    proc_session.watch_patterns = list(watch_patterns)
                    result_data["watch_patterns"] = proc_session.watch_patterns

                return json.dumps(result_data, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": f"Failed to start background process: {str(e)}"
                }, ensure_ascii=False)
        else:
            # Run foreground command with retry logic
            max_retries = 3
            retry_count = 0
            result = None
            command_cwd = None

            # Clean interrupt slate for an approved command, ONCE before the
            # retry loop: drop a stale bit that landed on this thread during the
            # approval-wait so it can't SIGINT the just-approved run.  Do NOT
            # re-clear inside the loop -- a genuine interrupt arriving during the
            # backoff sleep between retries must survive and abort the command
            # (caught by the next attempt's _wait_for_process poll loop -> 130).
            if _approved_run:
                from tools.interrupt import clear_current_thread_interrupt
                clear_current_thread_interrupt()

            while retry_count <= max_retries:
                try:
                    command_cwd = _resolve_command_cwd(
                        workdir=workdir,
                        default_cwd=cwd,
                        session_key=session_key,
                        _resolution=target_resolution,
                    )
                    execute_kwargs = {
                        "timeout": effective_timeout,
                        "cwd": command_cwd,
                        # Foreground model-facing output: cap retention while
                        # streaming (head/tail window) so a verbose command
                        # can't OOM the gateway before truncation (#64435).
                        # Internal env.execute() consumers (file ops cat
                        # reads, RPC reads) intentionally stay unbounded.
                        "bounded_capture": True,
                    }
                    with _scoped_sudo_execution(
                        target_resolution.target,
                        target_resolution.backend,
                        named=target_resolution.named,
                        sudo_password=target_resolution.config.get("sudo_password"),
                        target_scope=(
                            target_resolution.security_scope
                            if target_resolution.named else ""
                        ),
                    ):
                        result = env.execute(command, **execute_kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str:
                        return json.dumps({
                            "output": "",
                            "exit_code": 124,
                            "error": f"Command timed out after {effective_timeout} seconds"
                        }, ensure_ascii=False)
                    
                    # Retry on transient errors
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                       wait_time, retry_count, max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                        time.sleep(wait_time)
                        continue
                    
                    logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                 max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": f"Command execution failed: {type(e).__name__}: {str(e)}"
                    }, ensure_ascii=False)
                
                # Got a result
                break

            # Dual-write (cwd rearch step 1): the env's post-command tracking
            # (marker parse / local sync) has just updated env.cwd with the
            # directory this command finished in. That cwd belongs to THIS
            # session — record it under the session key so the durable record
            # never depends on the shared env surviving or on who drives the
            # env next.
            if not workdir:
                record_session_cwd(
                    session_key, getattr(env, "cwd", None),
                    _resolution=target_resolution,
                )

            # Extract output
            output = result.get("output", "")
            returncode = result.get("returncode", 0)
            # Spill metadata from the bounded collector: present only when
            # output overflowed the capture window (see _wait_for_process).
            spill_total_chars = result.get("output_total_chars")
            spill_file_path = result.get("full_output_path")

            # Add helpful message for sudo failures in messaging context
            output = _handle_sudo_failure(output, env_type)

            sudo_auth_failed = _sudo_wrong_password_failure(output)
            sudo_cache_cleared = _invalidate_cached_sudo_on_auth_failure(
                command,
                output,
                target_resolution.target,
                target_resolution.backend,
                (
                    target_resolution.security_scope
                    if target_resolution.named else ""
                ),
            )
            if sudo_cache_cleared:
                has_sudo_prompt_callback = _get_sudo_password_callback() is not None
                if has_sudo_prompt_callback or env_var_enabled("HERMES_INTERACTIVE"):
                    output += (
                        "\n\n⚠️ Sudo authentication failed — cached password "
                        "cleared. You will be prompted again on the next sudo "
                        "command."
                    )

            # Foreground terminal output canonicalization seam: process capture
            # is already bounded by BaseEnvironment before sudo checks and hooks
            # run. Plugins may replace that bounded string; replacements are
            # still subject to the final output limit below.
            # The hook is fail-open, and the first valid string return wins.
            try:
                from hermes_cli.lifecycle import invoke_hook
                hook_kwargs = {
                    "command": command,
                    "output": output,
                    "returncode": returncode,
                    "task_id": effective_base_task_id or "",
                    "env_type": env_type,
                }
                if target_resolution.named:
                    hook_kwargs.update({
                        "execution_target": target_resolution.target,
                        "execution_backend": target_resolution.backend,
                    })
                hook_results = invoke_hook(
                    "transform_terminal_output",
                    **hook_kwargs,
                )
                for hook_result in hook_results:
                    if isinstance(hook_result, str):
                        output = hook_result
                        break
            except Exception:
                pass
            
            # Truncate output if too long, keeping both head and tail
            from tools.tool_output_limits import get_max_bytes
            MAX_OUTPUT_CHARS = get_max_bytes()
            if len(output) > MAX_OUTPUT_CHARS:
                head_chars = int(MAX_OUTPUT_CHARS * 0.4)  # 40% head (error messages often appear early)
                tail_chars = MAX_OUTPUT_CHARS - head_chars  # 60% tail (most recent/relevant output)
                omitted = len(output) - head_chars - tail_chars
                truncated_notice = (
                    f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted "
                    f"out of {len(output)} total] ...\n\n"
                )
                output = output[:head_chars] + truncated_notice + output[-tail_chars:]

            # Strip ANSI escape sequences so the model never sees terminal
            # formatting — prevents it from copying escapes into file writes.
            from tools.ansi_strip import strip_ansi
            output = strip_ansi(output)

            # Redact secrets from command output. For source/config dumps
            # (MAX_TOKENS=100, "apiKey": "x" fixtures, postgresql:// f-string
            # templates) the ENV/JSON/template passes are skipped to avoid
            # false positives (code_file=True). But for env-dump commands
            # (env/printenv/set/export/declare) the output IS a KEY=value
            # credential dump, so redact_terminal_output runs the ENV pass
            # (code_file=False) to mask opaque tokens with no vendor prefix.
            # Real prefixes, auth headers, JWTs, private keys are masked in
            # both modes. See issue #43025.
            from agent.redact import redact_terminal_output
            output = redact_terminal_output(output.strip(), command) if output else ""

            # Interpret non-zero exit codes that aren't real errors
            # (e.g. grep=1 means "no matches", diff=1 means "files differ")
            exit_note = _interpret_exit_code(command, returncode)

            # Output-pattern failure hints: map well-known error shapes
            # (command-not-found, ModuleNotFoundError, gh field drift,
            # merge conflicts, ...) to one short recovery hint so the model
            # fixes the root cause on the next call instead of spending
            # turns on re-diagnosis. See tools/terminal_hints.py.
            failure_hint = None
            if returncode != 0 and not exit_note:
                try:
                    from tools.terminal_hints import annotate_failure
                    failure_hint = annotate_failure(command, returncode, output)
                except Exception:
                    failure_hint = None

            result_dict = {
                "output": output,
                "exit_code": returncode,
                "error": None,
            }
            result_dict.update(target_resolution.metadata(
                cwd=command_cwd if target_resolution.named else None,
            ))
            try:
                post_cwd = getattr(env, "cwd", None)
                if post_cwd and command_cwd and os.path.realpath(str(post_cwd)) != os.path.realpath(str(command_cwd)):
                    result_dict["cwd"] = str(post_cwd)
            except Exception:
                pass
            if spill_file_path:
                try:
                    _sp = Path(spill_file_path)
                    raw_spill = _sp.read_text(encoding="utf-8", errors="replace")
                    _sp.write_text(
                        redact_terminal_output(strip_ansi(raw_spill), command),
                        encoding="utf-8", errors="replace",
                    )
                    result_dict["output_total_chars"] = spill_total_chars
                    result_dict["full_output_path"] = spill_file_path
                    result_dict["truncation_note"] = (
                        "Output exceeded the capture window (head+tail shown). "
                        f"Full output ({spill_total_chars:,} chars) saved to "
                        f"{spill_file_path} — search it with search_files or page it "
                        "with read_file instead of re-running the command."
                    )
                except Exception:
                    logger.debug("spill redaction failed; dropping spill handle", exc_info=True)
                    try:
                        Path(spill_file_path).unlink()
                    except OSError:
                        pass
            if target_resolution.backend == "local":
                try:
                    from agent.verification_evidence import record_terminal_result

                    evidence = record_terminal_result(
                        command=command,
                        cwd=command_cwd,
                        session_id=(
                            session_id or task_id or backend_task_id or "default"
                        ),
                        exit_code=returncode,
                        output=output,
                    )
                    if evidence:
                        result_dict["verification_evidence"] = {
                            "status": evidence.get("status"),
                            "kind": evidence.get("kind"),
                            "scope": evidence.get("scope"),
                            "canonical_command": evidence.get("canonical_command"),
                        }
                except Exception:
                    logger.debug(
                        "verification evidence recording failed", exc_info=True,
                    )
            if approval_note:
                # Treat rc=130 as an interrupt only when the executor's marker is
                # present.  A command can legitimately exit 130 on its own
                # (e.g. `bash -c 'exit 130'`); _wait_for_process returns the
                # child's natural returncode there with no marker, and that must
                # NOT be relabelled as a user interrupt in the audit note.
                if returncode == 130 and "[Command interrupted]" in output:
                    # Approved command was interrupted mid-run by a genuine Stop.
                    # Keep the audit trail but never imply success: the bare
                    # "...approved by the user." note must not co-occur with the
                    # interrupt exit code (satisfies the 3-part-signature DONE).
                    result_dict["approval"] = approval_note.rstrip(".") + ", then interrupted."
                else:
                    result_dict["approval"] = approval_note
            if exit_note:
                result_dict["exit_code_meaning"] = exit_note
            if failure_hint:
                result_dict["hint"] = failure_hint
            if sudo_auth_failed:
                result_dict["sudo_auth_failed"] = True
            if sudo_cache_cleared:
                result_dict["sudo_cache_cleared"] = True

            return json.dumps(result_dict, ensure_ascii=False)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error("terminal_tool exception:\n%s", tb_str)
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Failed to execute command: {str(e)}",
            "traceback": tb_str,
            "status": "error"
        }, ensure_ascii=False)


def _check_terminal_config_requirements(config: Dict[str, Any]) -> bool:
    """Check one already-resolved backend configuration."""
    try:
        env_type = config["env_type"]

        if env_type == "local":
            return True

        elif env_type == "docker":
            from tools.environments.docker import find_docker
            docker = find_docker()
            if not docker:
                logger.error("Docker executable not found in PATH or common install locations")
                return False
            result = subprocess.run([docker, "version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
            return result.returncode == 0

        elif env_type == "singularity":
            executable = shutil.which("apptainer") or shutil.which("singularity")
            if executable:
                result = subprocess.run([executable, "--version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
                return result.returncode == 0
            return False

        elif env_type == "ssh":
            if not config.get("ssh_host") or not config.get("ssh_user"):
                logger.error(
                    "SSH backend selected but TERMINAL_SSH_HOST and TERMINAL_SSH_USER "
                    "are not both set. Configure both or switch TERMINAL_ENV to 'local'."
                )
                return False
            return True

        elif env_type == "modal":
            modal_state = _get_modal_backend_state(config.get("modal_mode"))
            if modal_state["selected_backend"] == "managed":
                return True

            if modal_state["selected_backend"] != "direct":
                if modal_state["managed_mode_blocked"]:
                    logger.error(
                        "Modal backend selected with TERMINAL_MODAL_MODE=managed, but "
                        "Nous Tool Gateway access is not currently available and no direct "
                        "Modal credentials/config were found. %s Choose "
                        "TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials.",
                        nous_tool_gateway_unavailable_message(
                            "managed Modal execution",
                        ),
                    )
                    return False
                if modal_state["mode"] == "managed":
                    logger.error(
                        "Modal backend selected with TERMINAL_MODAL_MODE=managed, but the managed "
                        "tool gateway is unavailable. %s",
                        nous_tool_gateway_unavailable_message(
                            "managed Modal execution",
                        ),
                    )
                    return False
                elif modal_state["mode"] == "direct":
                    if managed_nous_tools_enabled():
                        logger.error(
                            "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                            "Modal credentials/config were found. Configure Modal or choose "
                            "TERMINAL_MODAL_MODE=managed/auto."
                        )
                    else:
                        logger.error(
                            "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                            "Modal credentials/config were found. Configure Modal or choose "
                            "TERMINAL_MODAL_MODE=auto."
                        )
                    return False
                else:
                    if managed_nous_tools_enabled():
                        logger.error(
                            "Modal backend selected but no direct Modal credentials/config or managed "
                            "tool gateway was found. Configure Modal, set up the managed gateway, "
                            "or choose a different TERMINAL_ENV."
                        )
                    else:
                        logger.error(
                            "Modal backend selected but no direct Modal credentials/config was found. "
                            "Configure Modal or choose a different TERMINAL_ENV."
                        )
                    return False

            if importlib.util.find_spec("modal") is None:
                logger.error("modal is required for direct modal terminal backend: pip install modal")
                return False

            return True

        elif env_type == "vercel_sandbox":
            return _check_vercel_sandbox_requirements(config)

        elif env_type == "daytona":
            from daytona import Daytona  # noqa: F401 — SDK presence check
            from agent.secret_scope import get_secret
            return get_secret("DAYTONA_API_KEY") is not None

        else:
            logger.error(
                "Unknown TERMINAL_ENV '%s'. Use one of: local, docker, singularity, "
                "modal, daytona, vercel_sandbox, ssh.",
                env_type,
            )
            return False
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e, exc_info=True)
        return False


def check_terminal_requirements() -> bool:
    """Keep tools available when any configured execution target is usable."""
    try:
        from tools.execution_targets import list_execution_targets

        inventory = list_execution_targets()
    except Exception as exc:
        # Registration checks have always failed closed for invalid terminal
        # configuration. Keep that contract; individual tool handlers still
        # produce actionable target errors when invoked directly.
        logger.error("Invalid execution target config: %s", exc)
        return False

    if inventory and inventory[0].named:
        # Cheap/always-available local targets first, so an unavailable Docker
        # daemon or cloud credential on another target does not emit a scary
        # startup error when at least one target is healthy.
        ordered = sorted(
            inventory,
            key=lambda item: (item.backend != "local", not item.is_default, item.target),
        )
        for resolution in ordered:
            try:
                config = _get_env_config(dict(resolution.config))
            except Exception:
                continue
            if _check_terminal_config_requirements(config):
                return True
        return False

    try:
        return _check_terminal_config_requirements(_get_env_config())
    except Exception as exc:
        logger.error("Invalid terminal configuration: %s", exc)
        return False


if __name__ == "__main__":
    # Simple test when run directly
    print("Terminal Tool Module")
    print("=" * 50)
    
    config = _get_env_config()
    print("\nCurrent Configuration:")
    print(f"  Environment type: {config['env_type']}")
    print(f"  Docker image: {config['docker_image']}")
    print(f"  Modal image: {config['modal_image']}")
    print(f"  Working directory: {config['cwd']}")
    print(f"  Default timeout: {config['timeout']}s")
    print(f"  Lifetime: {config['lifetime_seconds']}s")

    if not check_terminal_requirements():
        print("\n❌ Requirements not met. Please check the messages above.")
        sys.exit(1)

    print("\n✅ All requirements met!")
    print("\nAvailable Tool:")
    print("  - terminal_tool: Execute commands in sandboxed environments")

    print("\nUsage Examples:")
    print("  # Execute a command")
    print("  result = terminal_tool(command='ls -la')")
    print("  ")
    print("  # Run a background task")
    print("  result = terminal_tool(command='python server.py', background=True)")

    print("\nEnvironment Variables:")
    default_img = "nikolaik/python-nodejs:python3.11-nodejs20"
    print(
        "  TERMINAL_ENV: "
        f"{os.getenv('TERMINAL_ENV', 'local')} "
        "(local/docker/singularity/modal/daytona/vercel_sandbox/ssh)"
    )
    print(f"  TERMINAL_DOCKER_IMAGE: {os.getenv('TERMINAL_DOCKER_IMAGE', default_img)}")
    print(f"  TERMINAL_SINGULARITY_IMAGE: {os.getenv('TERMINAL_SINGULARITY_IMAGE', f'docker://{default_img}')}")
    print(f"  TERMINAL_MODAL_IMAGE: {os.getenv('TERMINAL_MODAL_IMAGE', default_img)}")
    print(f"  TERMINAL_DAYTONA_IMAGE: {os.getenv('TERMINAL_DAYTONA_IMAGE', default_img)}")
    print(f"  TERMINAL_CWD: {os.getenv('TERMINAL_CWD', _safe_getcwd())}")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  TERMINAL_SANDBOX_DIR: {os.getenv('TERMINAL_SANDBOX_DIR', f'{_dhh()}/sandboxes')}")
    print(f"  TERMINAL_TIMEOUT: {os.getenv('TERMINAL_TIMEOUT', '60')}")
    print(f"  TERMINAL_LIFETIME_SECONDS: {os.getenv('TERMINAL_LIFETIME_SECONDS', '300')}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute on the VM"
            },
            "background": {
                "type": "boolean",
                "description": "Run in the background, returning a session_id. Pair with notify_on_complete=true for anything with a defined end (tests, builds, deploys) — without it the process runs silently. Only servers/watchers/daemons that never exit should stay silent. Short commands: prefer foreground with a generous timeout.",
                "default": False
            },
            "timeout": {
                "type": "integer",
                "description": f"Max seconds to wait (default: 180, foreground max: {FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command finishes — set high for long tasks, you won't wait unnecessarily. Foreground timeout above {FOREGROUND_MAX_TIMEOUT}s is rejected; use background=true for longer commands.",
                "minimum": 1
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for this command (absolute path). Defaults to the session working directory."
            },
            "target": {
                "type": "string",
                "description": "Named execution target from terminal.targets (for example 'local' or 'devbox'). Omit to use terminal.default_target; legacy flat config accepts only 'default'.",
            },
            "pty": {
                "type": "boolean",
                "description": "Run in pseudo-terminal (PTY) mode for interactive CLI tools like Codex, Claude Code, or Python REPL. Only works with local and SSH backends. Default: false.",
                "default": False
            },
            "notify_on_complete": {
                "type": "boolean",
                "description": "With background=true: get exactly one notification when the process exits. The right choice for nearly every bounded long task — set it and keep working. MUTUALLY EXCLUSIVE with watch_patterns (watch_patterns is dropped when both are set).",
                "default": False
            },
            "watch_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Strings to watch for in background output. ONLY for rare one-shot mid-process signals on processes that never exit (e.g. ['Application startup complete'] on a server). NOT for end-of-run markers (use notify_on_complete) and NOT for per-iteration patterns like 'ERROR' in loops — rate-limited to 1 notification/15s; repeated over-firing auto-disables it and falls back to notify-on-exit. When in doubt, use notify_on_complete. MUTUALLY EXCLUSIVE with notify_on_complete."
            }
        },
        "required": ["command"]
    }
}


def _handle_terminal(args, **kw):
    return terminal_tool(
        command=args.get("command"),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
        workdir=args.get("workdir"),
        pty=args.get("pty", False),
        notify_on_complete=args.get("notify_on_complete", False),
        watch_patterns=args.get("watch_patterns"),
        target=args.get("target"),
    )


registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)
