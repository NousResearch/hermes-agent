#!/usr/bin/env python3
"""
Terminal Tool Module

A terminal tool that executes commands in local, Docker, Modal, SSH,
Singularity, and Daytona environments. Supports local execution,
containerized backends, and cloud sandboxes, including managed Modal mode.

Supported environments:
- "local": Execute directly on the host machine (default, fastest)
- "docker": Execute in Docker containers (isolated, requires Docker)
- "modal": Execute in Modal cloud sandboxes (direct Modal or managed gateway)

Features:
- Multiple execution backends (local, docker, modal)
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
import time
import threading
import atexit
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

from utils import env_var_enabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global interrupt event: set by the agent when a user interrupt arrives.
# The terminal tool polls this during command execution so it can kill
# long-running subprocesses immediately instead of blocking until timeout.
# ---------------------------------------------------------------------------
from tools.interrupt import is_interrupted, _interrupt_event  # noqa: F401 — re-exported
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


def _check_disk_usage_warning():
    """Check if total disk usage exceeds warning threshold."""
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
        
        if total_gb > DISK_USAGE_WARNING_THRESHOLD_GB:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
            return True
        
        return False
    except Exception as e:
        logger.debug("Disk usage warning check failed: %s", e, exc_info=True)
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


def _get_sudo_password_cache_scope() -> str:
    """Return the cache scope for interactive sudo passwords."""
    try:
        from gateway.session_context import get_session_env

        session_key = get_session_env("HERMES_SESSION_KEY", "")
    except Exception:
        session_key = os.getenv("HERMES_SESSION_KEY", "")
    if session_key:
        return f"session:{session_key}"

    callback = _get_sudo_password_callback()
    if callback is not None:
        owner = getattr(callback, "__self__", None)
        func = getattr(callback, "__func__", None)
        if owner is not None and func is not None:
            return f"callback-owner:{id(owner)}:{id(func)}"
        return f"callback:{id(callback)}"

    return f"thread:{threading.get_ident()}"


def _get_cached_sudo_password() -> str:
    """Return the cached sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        return _sudo_password_cache.get(scope, "")


def _set_cached_sudo_password(password: str) -> None:
    """Persist a sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
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
                      has_host_access: bool = False) -> dict:
    """Delegate to consolidated guard (tirith + dangerous cmd) with CLI callback."""
    return _check_all_guards_impl(command, env_type,
                                  approval_callback=_get_approval_callback(),
                                  has_host_access=has_host_access)


# Allowlist: characters that can legitimately appear in directory paths.
# Covers alphanumeric, path separators, Windows drive/UNC separators, tilde,
# dot, hyphen, underscore, space, plus, at, equals, and comma.  Everything
# else is rejected.
_WORKDIR_SAFE_RE = re.compile(r'^[A-Za-z0-9/\\:_\-.~ +@=,]+$')


def _validate_workdir(workdir: str) -> str | None:
    """Reject workdir values that don't look like a filesystem path.

    Uses an allowlist of safe characters rather than a deny-list, so novel
    shell metacharacters can't slip through.

    Returns None if safe, or an error message string if dangerous.
    """
    if not workdir:
        return None
    if not _WORKDIR_SAFE_RE.match(workdir):
        # Find the first offending character for a helpful message.
        for ch in workdir:
            if not _WORKDIR_SAFE_RE.match(ch):
                return (
                    f"Blocked: workdir contains disallowed character {repr(ch)}. "
                    "Use a simple filesystem path without shell metacharacters."
                )
        return "Blocked: workdir contains disallowed characters."
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
    command: str | None, output: str
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
    if not _get_cached_sudo_password():
        return False
    _set_cached_sudo_password("")
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

    Callers that cannot pipe subprocess stdin (modal, daytona) must embed
    the password in the command string themselves; see their execute()
    methods for how they handle the non-None sudo_stdin case.

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

    has_configured_password = "SUDO_PASSWORD" in os.environ
    sudo_password = (
        os.environ.get("SUDO_PASSWORD", "")
        if has_configured_password
        else _get_cached_sudo_password()
    )

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
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands on a Linux environment. Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail to read files — use read_file instead.
Do NOT use grep/rg/find to search — use search_files instead.
Do NOT use ls to list directories — use search_files(target='files') instead.
Do NOT use sed/awk to edit files — use patch instead.
Do NOT use echo/cat heredoc to create files — use write_file instead.
Reserve terminal for: builds, installs, git, processes, scripts, network, package managers, and anything that needs a shell.
Because exported environment state persists, activate a virtualenv or export setup variables once per session; do not re-source the same environment before every command unless a command proves the shell state was reset.

Foreground (default): Commands return INSTANTLY when done, even if the timeout is high. Set timeout=300 for long builds/scripts — you'll still get the result in seconds if it's fast. Prefer foreground for short commands.
Background: Set background=true to get a session_id. Almost always pair with notify_on_complete=true — bg without notify runs SILENTLY and you have no way to learn it finished short of calling process(action='poll') yourself. Two legitimate uses:
  (1) Long-lived processes that never exit (servers, watchers, daemons) — silent is correct, there's no exit to notify on.
  (2) Long-running bounded tasks (tests, builds, deploys, CI pollers, batch jobs) — MUST set notify_on_complete=true. Without it you'll either forget to poll or sit blocked waiting for the user to surface the result.
For servers/watchers, do NOT use shell-level background wrappers (nohup/disown/setsid/trailing '&') in foreground mode. Use background=true so Hermes can track lifecycle and output.
After starting a server, verify readiness with a health check or log signal, then run tests in a separate terminal() call. Avoid blind sleep loops.
Use process(action="poll") for progress checks, process(action="wait") to block until done.
Working directory: Use 'workdir' for per-command cwd.
PTY mode: Set pty=true for interactive CLI tools (Codex, Claude Code, Python REPL).

Do NOT use vim/nano/interactive tools without pty=true — they hang without a pseudo-terminal. Pipe git output to cat if it might page.
"""

# Global state for environment lifecycle management
_active_environments: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_creation_locks: Dict[str, threading.Lock] = {}  # Per-task locks for sandbox creation
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

    # 2 × lifetime_seconds gives sibling Hermes processes a generous grace
    # window. Floor at 60s so an operator with TERMINAL_LIFETIME_SECONDS=0
    # doesn't get an instant-reap that races their own setup.
    # ``container_config`` only carries container_* keys, so read
    # lifetime_seconds from the env var the rest of the module uses.
    try:
        lifetime = int(os.getenv("TERMINAL_LIFETIME_SECONDS", "300"))
    except (TypeError, ValueError):
        lifetime = 300
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
_task_env_overrides: Dict[str, Dict[str, Any]] = {}


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
    _task_env_overrides[task_id] = overrides

    # If a live environment already exists for this task, a freshly registered
    # ``cwd`` override (e.g. the ACP client switching the editor's project root
    # mid-session via ``session/load`` / ``session/resume``) must take effect on
    # the cached env too. ``terminal_tool`` resolves the per-command cwd as
    # ``workdir > env.cwd > config/override cwd`` so that ordinary in-session
    # ``cd`` state is preserved; without syncing here the override would sit
    # below the (already-set) ``env.cwd`` and be silently ignored once any
    # command has run. Pushing it onto the live env keeps ``cd`` tracking intact
    # while letting an explicit ACP cwd change win, as the client expects.
    new_cwd = overrides.get("cwd")
    if isinstance(new_cwd, str) and new_cwd.strip():
        # The live env is cached under the raw task_id for per-session surfaces
        # (ACP/gateway/dashboard) and under the collapsed container id for
        # isolation-keyed rollouts. Try the raw id first, then the container id,
        # so a CWD-only override (which collapses to "default") still finds and
        # updates the originating session's env.
        container_id = _resolve_container_task_id(task_id)
        with _env_lock:
            env = _active_environments.get(task_id) or _active_environments.get(container_id)
        if env is not None and getattr(env, "cwd", None) is not None:
            env.cwd = new_cwd


def clear_task_env_overrides(task_id: str):
    """
    Clear environment overrides for a task after rollout completes.

    Called during cleanup to avoid stale entries accumulating.
    """
    _task_env_overrides.pop(task_id, None)


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
    if task_id and task_id in _task_env_overrides:
        overrides = _task_env_overrides[task_id]
        if set(overrides.keys()) & _ISOLATION_KEYS:
            return task_id
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
    return (
        _task_env_overrides.get(raw)
        or _task_env_overrides.get(_resolve_container_task_id(raw))
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

_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona"})


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


def _get_env_config() -> Dict[str, Any]:
    """Get terminal environment configuration from environment variables."""
    # Default image with Python and Node.js for maximum compatibility
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    env_type = os.getenv("TERMINAL_ENV", "local")
    
    mount_docker_cwd = os.getenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false").lower() in {"true", "1", "yes"}
    container_backend = env_type in {"docker", "singularity", "modal", "daytona"}
    docker_backend = env_type == "docker"

    # Docker/container-only env vars may be bridged from config.yaml even when
    # the active backend is local/ssh.  Do not parse their JSON/numeric payloads
    # until a backend that can consume them is selected; a stale or invalid
    # Docker value should not make local terminal/execute_code unusable.
    if container_backend:
        container_cpu = _parse_env_var("TERMINAL_CONTAINER_CPU", "1", float, "number")
        container_memory = _parse_env_var("TERMINAL_CONTAINER_MEMORY", "5120")
        container_disk = _parse_env_var("TERMINAL_CONTAINER_DISK", "51200")
    else:
        container_cpu = 1.0
        container_memory = 5120
        container_disk = 51200

    if docker_backend:
        docker_forward_env = _parse_env_var("TERMINAL_DOCKER_FORWARD_ENV", "[]", json.loads, "valid JSON")
        docker_volumes = _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")
        docker_env = _parse_env_var("TERMINAL_DOCKER_ENV", "{}", json.loads, "valid JSON")
        docker_extra_args = _parse_env_var("TERMINAL_DOCKER_EXTRA_ARGS", "[]", json.loads, "valid JSON")
    else:
        docker_forward_env = []
        docker_volumes = []
        docker_env = {}
        docker_extra_args = []

    # Default cwd: local uses the host's current directory, ssh uses the
    # remote home, and everything else starts in the backend's default
    # root-like cwd.
    if env_type == "local":
        default_cwd = _safe_getcwd()
    elif env_type == "ssh":
        default_cwd = "~"
    else:
        default_cwd = "/root"

    # Read TERMINAL_CWD but sanity-check it for container backends.
    # If Docker cwd passthrough is explicitly enabled, remap the host path to
    # /workspace and track the original host path separately. Otherwise keep the
    # normal sandbox behavior and discard host paths.
    cwd = os.getenv("TERMINAL_CWD", default_cwd)
    if cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        docker_cwd_source = os.getenv("TERMINAL_CWD") or _safe_getcwd()
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
        "modal_mode": coerce_modal_mode(os.getenv("TERMINAL_MODAL_MODE", "auto")),
        "docker_image": os.getenv("TERMINAL_DOCKER_IMAGE", default_image),
        "docker_forward_env": docker_forward_env,
        "singularity_image": os.getenv("TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}"),
        "modal_image": os.getenv("TERMINAL_MODAL_IMAGE", default_image),
        "daytona_image": os.getenv("TERMINAL_DAYTONA_IMAGE", default_image),
        "cwd": cwd,
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": _parse_env_var("TERMINAL_TIMEOUT", "180"),
        "lifetime_seconds": _parse_env_var("TERMINAL_LIFETIME_SECONDS", "300"),
        # SSH-specific config
        "ssh_host": os.getenv("TERMINAL_SSH_HOST", ""),
        "ssh_user": os.getenv("TERMINAL_SSH_USER", ""),
        "ssh_port": _parse_env_var("TERMINAL_SSH_PORT", "22"),
        "ssh_key": os.getenv("TERMINAL_SSH_KEY", ""),
        # Persistent shell: SSH defaults to the config-level persistent_shell
        # setting (true by default for non-local backends); local is always opt-in.
        # Per-backend env vars override if explicitly set.
        "ssh_persistent": os.getenv(
            "TERMINAL_SSH_PERSISTENT",
            os.getenv("TERMINAL_PERSISTENT_SHELL", "true"),
        ).lower() in {"true", "1", "yes"},
        "local_persistent": os.getenv("TERMINAL_LOCAL_PERSISTENT", "false").lower() in {"true", "1", "yes"},
        # Container resource config (applies to docker, singularity, modal,
        # daytona -- ignored for local/ssh)
        "container_cpu": container_cpu,
        "container_memory": container_memory,     # MB (default 5GB)
        "container_disk": container_disk,        # MB (default 50GB)
        "container_persistent": os.getenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"true", "1", "yes"},
        "docker_volumes": docker_volumes,
        "docker_env": docker_env,
        "docker_run_as_host_user": os.getenv("TERMINAL_DOCKER_RUN_AS_HOST_USER", "false").lower() in {"true", "1", "yes"},
        "docker_network": os.getenv("TERMINAL_DOCKER_NETWORK", "true").lower() in {"true", "1", "yes"},
        "docker_extra_args": docker_extra_args,
        # Cross-process container reuse (issue #20561).  The docs claim
        # "ONE long-lived container shared across sessions" — this toggle
        # makes that real by probing for a labeled container at startup and
        # attaching to it instead of always starting a fresh one.  Set to
        # ``false`` for hard per-process isolation (no reuse, container is
        # removed on exit).
        "docker_persist_across_processes": os.getenv(
            "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "true"
        ).lower() in {"true", "1", "yes"},
        # Startup orphan reaper for hermes-tagged containers left behind by
        # crashed / SIGKILL'd previous processes that bypassed atexit.
        # Conservative: only sweeps Exited containers older than 2× the
        # idle-reap window AND scoped to the current profile. Issue #20561.
        "docker_orphan_reaper": os.getenv(
            "TERMINAL_DOCKER_ORPHAN_REAPER", "true"
        ).lower() in {"true", "1", "yes"},
    }


def _get_modal_backend_state(modal_mode: object | None) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection."""
    return resolve_modal_backend_state(
        modal_mode,
        has_direct=has_direct_modal_credentials(),
        managed_ready=is_managed_tool_gateway_ready("modal"),
    )


def _create_environment(env_type: str, image: str, cwd: str, timeout: int,
                        ssh_config: dict = None, container_config: dict = None,
                        local_config: dict = None,
                        task_id: str = "default",
                        host_cwd: str = None):
    """
    Create an execution environment for sandboxed command execution.
    
    Args:
        env_type: One of "local", "docker", "singularity", "modal",
            "daytona", "ssh"
        image: Docker/Singularity/Modal image name (ignored for local/ssh)
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
        return _LocalEnvironment(cwd=cwd, timeout=timeout)
    
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
            volumes=volumes,
            host_cwd=host_cwd,
            auto_mount_cwd=cc.get("docker_mount_cwd_to_workspace", False),
            forward_env=docker_forward_env,
            env=docker_env,
            run_as_host_user=cc.get("docker_run_as_host_user", False),
            network=docker_network,
            extra_args=docker_extra_args,
            persist_across_processes=cc.get("docker_persist_across_processes", True),
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
        )

    else:
        raise ValueError(
            f"Unknown environment type: {env_type}. Use 'local', 'docker', "
            f"'singularity', 'modal', 'daytona', or 'ssh'"
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
            if current_time - last_time > lifetime_seconds:
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


def get_active_env(task_id: str):
    """Return the active BaseEnvironment for *task_id*, or None."""
    lookup = _resolve_container_task_id(task_id)
    with _env_lock:
        return _active_environments.get(lookup) or _active_environments.get(task_id)


def is_persistent_env(task_id: str) -> bool:
    """Return True if the active environment for task_id is configured for
    cross-turn persistence (``persistent_filesystem=True``).

    Used by the agent loop to skip per-turn teardown for backends whose whole
    point is to survive between turns (docker with ``container_persistent``,
    daytona, modal, etc.). Non-persistent backends (e.g. Morph) still get torn
    down at end-of-turn to prevent leakage. The idle reaper
    (``_cleanup_inactive_envs``) handles persistent envs once they exceed
    ``terminal.lifetime_seconds``.
    """
    env = get_active_env(task_id)
    if env is None:
        return False
    return bool(getattr(env, "_persistent", False))




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


def cleanup_vm(task_id: str, *, force_remove: bool = False):
    """Manually clean up a specific environment by task_id.

    *force_remove* (default False) is forwarded to backends that accept it
    — currently only ``DockerEnvironment``. The default of False matches
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
    # Remove from tracking dicts while holding the lock, but defer the
    # actual (potentially slow) env.cleanup() call to outside the lock
    # so other tool calls aren't blocked.
    env = None
    with _env_lock:
        env = _active_environments.pop(task_id, None)
        _last_activity.pop(task_id, None)

    # Clean up per-task creation lock
    with _creation_locks_lock:
        _creation_locks.pop(task_id, None)

    # Invalidate stale file_ops cache entry
    try:
        from tools.file_tools import clear_file_ops_cache
        clear_file_ops_cache(task_id)
    except ImportError:
        pass

    if env is None:
        return

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

        logger.info("Manually cleaned up environment for task: %s", task_id)

    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", task_id)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _atexit_cleanup():
    """Stop cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    if _active_environments:
        count = len(_active_environments)
        logger.info("Shutting down %d remaining sandbox(es)...", count)
        # Snapshot the env objects BEFORE cleanup_all_environments empties
        # the dict; we need them to wait on docker cleanup threads after the
        # registry has been cleared.
        envs_to_wait = list(_active_environments.values())
        cleanup_all_environments()
        # Block briefly so docker stop/rm actually completes before the
        # interpreter exits. Issue #20561 — without this join, the daemon
        # cleanup threads were getting torn down mid-`docker stop`, leaving
        # Exited containers piled up on the host.
        for env in envs_to_wait:
            wait_fn = getattr(env, "wait_for_cleanup", None)
            if wait_fn is None:
                continue
            try:
                wait_fn(timeout=15.0)
            except Exception as e:  # never block shutdown on a bad backend
                logger.debug("wait_for_cleanup raised on exit: %s", e)

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
            "Use terminal(background=true) so Hermes can track the process, then run "
            "readiness checks and tests in separate commands."
        )

    if _INLINE_BACKGROUND_AMP_RE.search(unquoted) or _TRAILING_BACKGROUND_AMP_RE.search(unquoted):
        return (
            "Foreground command uses '&' backgrounding. Use terminal(background=true) for long-lived "
            "processes, then run health checks and tests in follow-up terminal calls."
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
    env: Any,
    default_cwd: str,
) -> str:
    """Return the cwd for a command, preferring the live session cwd.

    ``terminal_tool`` historically re-sent the init-time/config cwd on every
    call. That broke session-local ``cd`` state: the environment tracked the
    new directory in ``env.cwd``, but foreground/background calls kept forcing
    the old cwd back through ``env.execute(..., cwd=...)``. Explicit
    ``workdir=`` must still override everything.
    """
    if workdir:
        return workdir

    live_cwd = getattr(env, "cwd", None)
    if isinstance(live_cwd, str) and live_cwd.strip():
        return live_cwd

    return default_cwd


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

        # Get configuration
        config = _get_env_config()
        env_type = config["env_type"]

        # Use task_id for environment isolation. By default all subagent
        # task_ids collapse back to "default" so the top-level agent and
        # every delegate_task child share one container; only task_ids with
        # a registered env override (RL benchmarks) get isolated sandboxes.
        effective_task_id = _resolve_container_task_id(task_id)

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

        cwd = overrides.get("cwd") or config["cwd"]
        # A per-task cwd override (registered by the gateway/TUI for workspace
        # tracking, or by RL/benchmark envs) wins over config["cwd"] — but
        # config["cwd"] was already sanitized for container backends in
        # _get_env_config() while the override is raw. On a container backend a
        # raw host path (e.g. a Windows desktop session's C:\Users\<user>, or a
        # POSIX /home/<user>) reaches `docker run -w <host-path>` and the
        # container fails to start (exit 125). Re-apply the same host/relative
        # path guard to the *resolved* cwd so the override can't bypass it.
        # Valid in-container override paths (RL/benchmark sandboxes that set
        # cwd to /workspace, /root, etc.) are absolute non-host paths and pass
        # through untouched.
        if env_type in _CONTAINER_BACKENDS and _is_unusable_container_cwd(cwd):
            if cwd != config["cwd"]:
                logger.info(
                    "Ignoring host/relative cwd override %r for %s backend "
                    "(won't exist in sandbox). Using %r instead.",
                    cwd, env_type, config["cwd"],
                )
            cwd = config["cwd"]
        default_timeout = config["timeout"]
        effective_timeout = timeout or default_timeout

        # Reject foreground commands where the model explicitly requests
        # a timeout above FOREGROUND_MAX_TIMEOUT — nudge it toward background.
        if not background and timeout and timeout > FOREGROUND_MAX_TIMEOUT:
            return json.dumps({
                "error": (
                    f"Foreground timeout {timeout}s exceeds the maximum of "
                    f"{FOREGROUND_MAX_TIMEOUT}s. Use background=true with "
                    f"notify_on_complete=true for long-running commands."
                ),
            }, ensure_ascii=False)

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
        with _env_lock:
            # Prefer the collapsed container id, but fall back to an env cached
            # under the raw task_id. Per-session surfaces (ACP/gateway/dashboard)
            # with a CWD-only override collapse to "default" for container
            # sharing, yet an env may already be cached under the originating
            # task_id; honor it instead of spawning a duplicate.
            _existing_key = (
                effective_task_id if effective_task_id in _active_environments
                else (task_id if task_id and task_id in _active_environments else None)
            )
            if _existing_key is not None:
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
                with _env_lock:
                    _existing_key = (
                        effective_task_id if effective_task_id in _active_environments
                        else (task_id if task_id and task_id in _active_environments else None)
                    )
                    if _existing_key is not None:
                        _last_activity[_existing_key] = time.time()
                        env = _active_environments[_existing_key]
                        needs_creation = False

                if needs_creation:
                    if env_type == "singularity":
                        _check_disk_usage_warning()
                    logger.info("Creating new %s environment for task %s...", env_type, effective_task_id[:8])
                    try:
                        ssh_config = None
                        if env_type == "ssh":
                            ssh_config = {
                                "host": config.get("ssh_host", ""),
                                "user": config.get("ssh_user", ""),
                                "port": config.get("ssh_port", 22),
                                "key": config.get("ssh_key", ""),
                                "persistent": config.get("ssh_persistent", False),
                            }

                        container_config = None
                        if env_type in {"docker", "singularity", "modal", "daytona"}:
                            container_config = {
                                "container_cpu": config.get("container_cpu", 1),
                                "container_memory": config.get("container_memory", 5120),
                                "container_disk": config.get("container_disk", 51200),
                                "container_persistent": config.get("container_persistent", True),
                                "modal_mode": config.get("modal_mode", "auto"),
                                "docker_volumes": config.get("docker_volumes", []),
                                "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
                                "docker_forward_env": config.get("docker_forward_env", []),
                                "docker_env": config.get("docker_env", {}),
                                "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
                                "docker_extra_args": config.get("docker_extra_args", []),
                                "docker_network": config.get("docker_network", True),
                                "docker_persist_across_processes": config.get("docker_persist_across_processes", True),
                                "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
                            }

                        local_config = None
                        if env_type == "local":
                            local_config = {
                                "persistent": config.get("local_persistent", False),
                            }

                        new_env = _create_environment(
                            env_type=env_type,
                            image=image,
                            cwd=cwd,
                            timeout=effective_timeout,
                            ssh_config=ssh_config,
                            container_config=container_config,
                            local_config=local_config,
                            task_id=effective_task_id,
                            host_cwd=config.get("host_cwd"),
                        )
                    except ImportError as e:
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": f"Terminal tool disabled: environment creation failed ({e})",
                            "status": "disabled"
                        }, ensure_ascii=False)

                    with _env_lock:
                        _active_environments[effective_task_id] = new_env
                        _last_activity[effective_task_id] = time.time()
                        env = new_env
                    logger.info("%s environment ready for task %s", env_type, effective_task_id[:8])

        # Hard-block: gateway lifecycle commands (systemctl/launchctl/hermes
        # restart|stop targeting hermes-gateway) must never run inside the
        # gateway process itself. The restart would SIGTERM the gateway, which
        # kills this very subprocess before it can complete — the service may
        # never restart. This mirrors the `hermes gateway restart` guard in
        # hermes_cli/gateway.py and the cron-path guard in hermes_cli/cron.py,
        # but applies unconditionally (force=True cannot help here).
        if os.environ.get("_HERMES_GATEWAY") == "1":
            from hermes_cli.cron import _contains_gateway_lifecycle_command
            if _contains_gateway_lifecycle_command(command):
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": (
                        "Blocked: cannot restart or stop the gateway from inside the "
                        "gateway process. The gateway would kill this command before "
                        "it could complete (SIGTERM propagates to child processes). "
                        "Run `hermes gateway restart` from a separate shell outside "
                        "the running gateway."
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
            )
            if not approval["approved"]:
                # Check if this is an approval_required (gateway ask mode)
                if approval.get("status") == "pending_approval":
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": "",
                        "status": "pending_approval",
                        "approval_pending": True,
                        "command": approval.get("command", command),
                        "description": approval.get("description", "command flagged"),
                        "pattern_key": approval.get("pattern_key", ""),
                    }, ensure_ascii=False)
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

        # Claim the (shared "default") terminal env for the session driving this
        # command. File tools read env.cwd_owner to decide whether the env's live
        # cwd is THIS session's `cd` or a different worktree session's — without
        # it, two open worktree sessions sharing the env route each other's edits
        # to the wrong checkout. get_current_session_key()'s contextvar doesn't
        # cross tool-worker threads, so fall back to the raw task_id (which IS the
        # session_key for the top-level agent) — a stable, thread-safe anchor.
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="") or (task_id or "")
        try:
            env.cwd_owner = session_key
        except Exception:
            pass

        if background:
            # Spawn a tracked background process via the process registry.
            # For local backends: uses subprocess.Popen with output buffering.
            # For non-local backends: runs inside the sandbox via env.execute().
            from tools.process_registry import process_registry

            effective_cwd = _resolve_command_cwd(
                workdir=workdir,
                env=env,
                default_cwd=cwd,
            )
            try:
                if env_type == "local":
                    proc_session = process_registry.spawn_local(
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                        env_vars=env.env if hasattr(env, 'env') else None,
                        use_pty=effective_pty,
                    )
                else:
                    proc_session = process_registry.spawn_via_env(
                        env=env,
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                    )

                result_data = {
                    "output": "Background process started",
                    "session_id": proc_session.id,
                    "pid": proc_session.pid,
                    "exit_code": 0,
                    "error": None,
                }
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

                    # Stateless request/response sessions (the API server /
                    # WebUI path) cannot route a completion back to the agent
                    # after the turn ends — there is no persistent channel and
                    # send() is a no-op. Registering a watcher there silently
                    # no-ops (issue #10760). Refuse the promise instead: drop
                    # the flags and tell the agent to poll.
                    if not _async_ok():
                        notify_on_complete = False
                        watch_patterns = None
                        result_data["notify_on_complete"] = False
                        result_data["notify_unsupported"] = (
                            "notify_on_complete / watch_patterns are not available on "
                            "this endpoint (stateless HTTP API — no channel to deliver "
                            "an async completion after the turn ends). The process is "
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
                            _gw_session_id = _gse("HERMES_SESSION_ID", "")
                            proc_session.watcher_platform = _gw_platform
                            proc_session.watcher_chat_id = _gw_chat_id
                            proc_session.watcher_user_id = _gw_user_id
                            proc_session.watcher_user_name = _gw_user_name
                            proc_session.watcher_thread_id = _gw_thread_id
                            proc_session.watcher_message_id = _gw_message_id
                            proc_session.conversation_session_id = _gw_session_id

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
                            "conversation_session_id": proc_session.conversation_session_id,
                            "notify_on_complete": True,
                        })

                # Set watch patterns for output monitoring
                if watch_patterns and background:
                    proc_session.watch_patterns = list(watch_patterns)
                    result_data["watch_patterns"] = proc_session.watch_patterns

                # spawn_local()/spawn_via_env() checkpoint before terminal_tool
                # can stamp gateway routing and notification metadata. Persist the
                # completed notification configuration so crash recovery keeps the
                # physical conversation session boundary used by stale-event guards.
                if background and (notify_on_complete or watch_patterns):
                    process_registry._write_checkpoint()

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
                        env=env,
                        default_cwd=cwd,
                    )
                    execute_kwargs = {
                        "timeout": effective_timeout,
                        "cwd": command_cwd,
                    }
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
            
            # Extract output
            output = result.get("output", "")
            returncode = result.get("returncode", 0)

            # Add helpful message for sudo failures in messaging context
            output = _handle_sudo_failure(output, env_type)

            sudo_auth_failed = _sudo_wrong_password_failure(output)
            sudo_cache_cleared = _invalidate_cached_sudo_on_auth_failure(
                command, output
            )
            if sudo_cache_cleared:
                has_sudo_prompt_callback = _get_sudo_password_callback() is not None
                if has_sudo_prompt_callback or env_var_enabled("HERMES_INTERACTIVE"):
                    output += (
                        "\n\n⚠️ Sudo authentication failed — cached password "
                        "cleared. You will be prompted again on the next sudo "
                        "command."
                    )

            # Foreground terminal output canonicalization seam: plugins receive
            # the full output string before default truncation and may only
            # replace it by returning a string from transform_terminal_output.
            # The hook is fail-open, and the first valid string return wins.
            try:
                from hermes_cli.plugins import invoke_hook
                hook_results = invoke_hook(
                    "transform_terminal_output",
                    command=command,
                    output=output,
                    returncode=returncode,
                    task_id=effective_task_id or "",
                    env_type=env_type,
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

            result_dict = {
                "output": output,
                "exit_code": returncode,
                "error": None,
            }
            try:
                from agent.verification_evidence import record_terminal_result

                evidence = record_terminal_result(
                    command=command,
                    cwd=command_cwd,
                    session_id=session_id or task_id or effective_task_id or "default",
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
                logger.debug("verification evidence recording failed", exc_info=True)
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


def check_terminal_requirements() -> bool:
    """Check if all requirements for the terminal tool are met."""
    try:
        config = _get_env_config()
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

        elif env_type == "daytona":
            from daytona import Daytona  # noqa: F401 — SDK presence check
            return os.getenv("DAYTONA_API_KEY") is not None

        else:
            logger.error(
                "Unknown TERMINAL_ENV '%s'. Use one of: local, docker, singularity, "
                "modal, daytona, ssh.",
                env_type,
            )
            return False
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e, exc_info=True)
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
        "(local/docker/singularity/modal/daytona/ssh)"
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
                "description": "Run the command in the background. Almost always pair with notify_on_complete=true — without it, the process runs silently and you'll have no way to learn it finished short of calling process(action='poll') yourself (easy to forget, leading to silent blindness on long jobs). Two legitimate patterns: (1) Long-lived processes that never exit (servers, watchers, daemons) — these stay silent because there's no exit to notify on. (2) Long-running bounded tasks (tests, builds, deploys, CI pollers, batch jobs) — these MUST set notify_on_complete=true. For short commands, prefer foreground with a generous timeout instead.",
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
            "pty": {
                "type": "boolean",
                "description": "Run in pseudo-terminal (PTY) mode for interactive CLI tools like Codex, Claude Code, or Python REPL. Only works with local and SSH backends. Default: false.",
                "default": False
            },
            "notify_on_complete": {
                "type": "boolean",
                "description": "When true (and background=true), you'll be automatically notified exactly once when the process finishes. **This is the right choice for almost every long-running task** — tests, builds, deployments, multi-item batch jobs, anything that takes over a minute and has a defined end. Use this and keep working on other things; the system notifies you on exit. MUTUALLY EXCLUSIVE with watch_patterns — when both are set, watch_patterns is dropped.",
                "default": False
            },
            "watch_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Strings to watch for in background process output. HARD RATE LIMIT: at most 1 notification per 15 seconds per process — matches arriving inside the cooldown are dropped. After 3 consecutive 15-second windows with dropped matches, watch_patterns is automatically disabled for that process and promoted to notify_on_complete behavior (one notification on exit, no more mid-process spam). USE ONLY for truly rare, one-shot mid-process signals on LONG-LIVED processes that will never exit on their own — e.g. ['Application startup complete'] on a server so you know when to hit its endpoint, or ['migration done'] on a daemon. DO NOT use for: (1) end-of-run markers like 'DONE'/'PASS' — use notify_on_complete instead; (2) error patterns like 'ERROR'/'Traceback' in loops or multi-item batch jobs — they fire on every iteration and you'll hit the strike limit fast; (3) anything you'd ever combine with notify_on_complete. When in doubt, choose notify_on_complete. MUTUALLY EXCLUSIVE with notify_on_complete — set one, not both."
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
