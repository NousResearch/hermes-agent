#!/usr/bin/env python3
"""Terminal tool: run shell commands in the configured backend.

Backends (``TERMINAL_ENV``): local (default), docker, singularity, modal
(direct or managed gateway), daytona, vercel_sandbox, ssh, plus
plugin-registered backends. Handles background processes, sandbox lifecycle
(per-task cache, idle reaper, atexit teardown) and sudo password plumbing.
Cloud-sandbox persistent filesystems preserve working state across sandbox
recreation but do NOT guarantee the same live sandbox or long-running
processes survive cleanup, idle reaping, or Hermes exit.

Companion modules (re-exported here, so ``tools.terminal_tool.<name>`` stays the
import/patch target): ``terminal_tool_config`` (TERMINAL_* reads, ``_quiet``),
``terminal_tool_backends`` (env builders + requirement checkers),
``terminal_tool_lifecycle`` (reaper/teardown/ensure_task_env),
``terminal_tool_sudo`` (sudo password + shell rewrites), ``terminal_tool_guards``
(pre-exec blocks), ``terminal_tool_background`` (background spawn),
``terminal_tool_result`` (foreground result post-processing).
"""

import json
import logging
import os
import sys
import time
import threading
import atexit
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def _redact_terminal_error_text(value: Any) -> str:
    """Force-redact text before serializing a terminal error envelope."""
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text("" if value is None else str(value), force=True)


from tools.registry import tool_error
from tools.terminal_tool_lifecycle import (
    _check_disk_usage_warning, _cleanup_inactive_envs, _create_configured_env,
    _evict_environment_for_task, cleanup_all_environments, ensure_task_env,
)
from tools.terminal_tool_config import (
    _is_container_backend, _is_host_cwd, _is_unusable_container_cwd, _parse_env_var,
    _plugin_env_flag, _quiet, _safe_getcwd, _tenv, _tenv_bool,
)
from tools.terminal_tool_backends import (
    _REQUIREMENT_CHECKERS, _VERCEL_SANDBOX_DEFAULT_CWD, _check_plugin_requirements,
    _record_unavailable_reason, terminal_backend_unavailable_reason,  # noqa: F401 — re-exported
)
# display_hermes_home imported lazily at call site (stale-module safety during hermes update)
from tools.tool_backend_helpers import coerce_modal_mode, managed_nous_tools_enabled


def _safe_parse_import_env(name: str, default: Any, converter, type_label: str):
    """Parse a module-level numeric env var; a malformed value must never make
    the module unloadable at import time (CLI, ACP, tests, tool discovery)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s: %r (expected %s). Falling back to %r.",
            name, raw, type_label, default,
        )
        return default


# Hard cap on foreground timeout; override via TERMINAL_MAX_FOREGROUND_TIMEOUT env var.
FOREGROUND_MAX_TIMEOUT = _safe_parse_import_env("TERMINAL_MAX_FOREGROUND_TIMEOUT", 600, int, "integer")

# Disk usage warning threshold (in GB)
DISK_USAGE_WARNING_THRESHOLD_GB = _safe_parse_import_env("TERMINAL_DISK_WARNING_GB", 500.0, float, "number")


# Approval / sudo-prompt UI callbacks (CLI registers prompt_toolkit-aware
# ones). Thread-local so overlapping ACP sessions, each on its own executor
# thread, can't stomp on each other (GHSA-qg5c-hvr5-hjgr). Gateway mode
# resolves approvals via the per-session queue in tools.approval instead.
_callback_tls = threading.local()


def _get_sudo_password_callback():
    return getattr(_callback_tls, "sudo_password", None)


def _current_session_key() -> str:
    """Return the active gateway/WebUI session key, or "" outside sessions.

    Single lookup point for the ``HERMES_SESSION_KEY`` ContextVar with the
    os.environ fallback that ``get_session_env()`` applies for CLI, cron, and
    test processes. Callers scope per-session caches by prefixing the value
    with ``"session:"`` so two sessions never share a cache slot.
    """
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_KEY", "")


def _get_approval_callback():
    return getattr(_callback_tls, "approval", None)


def set_sudo_password_callback(cb):
    """Register the CLI's sudo password prompt callback (per-thread slot)."""
    _callback_tls.sudo_password = cb


def set_approval_callback(cb):
    """Register the dangerous-command approval prompt callback (per-thread slot)."""
    _callback_tls.approval = cb


def _get_sudo_password_cache_scope() -> str:
    """Return the cache scope for interactive sudo passwords."""
    session_key = _current_session_key()
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


def _current_session_profile() -> str:
    """Active session's Hermes profile name, or "" (same lookup discipline as
    :func:`_current_session_key`)."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_PROFILE", "")


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


def _in_delegated_child_context() -> bool:
    """Return True while running inside a delegate_task child.

    Subagents execute on worker threads of the parent process, so they
    inherit process-wide interactivity signals (``HERMES_INTERACTIVE=1`` set
    by the CLI at startup) that do NOT mean *this* execution context can
    reach the user. A child that passes the interactive gate with no sudo
    callback falls through to the raw ``/dev/tty`` prompt — printed mid-TUI
    from a background thread, racing siblings for the tty, and blocking the
    child for the full timeout. Children must always behave as headless for
    sudo prompting. The ContextVar is set by ``delegated_child_context()``
    around every child run and propagates through ``contextvars.copy_context``
    onto the executor thread.
    """
    try:
        from agent.delegation_context import is_delegated_child_context

        return is_delegated_child_context()
    except Exception:
        return False


def _handle_sudo_failure(output: str, env_type: str) -> str:
    """
    Check for sudo failure and add helpful message for headless contexts
    (messaging gateway sessions and delegate_task subagents).

    Returns enhanced output if sudo failed in such a context, else original.
    """
    is_gateway = env_var_enabled("HERMES_GATEWAY_SESSION")
    is_delegated_child = _in_delegated_child_context()

    if not is_gateway and not is_delegated_child:
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
            if is_delegated_child:
                return output + (
                    "\n\n💡 Tip: Subagents cannot prompt for a sudo password. "
                    f"Add SUDO_PASSWORD to {_dhh()}/.env on the agent machine, "
                    "or run the command without sudo."
                )
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
            # Blocked on a human typing their password: exclude from tool
            # deadlines (#85125 2e). Local import avoids any import-layering
            # surprises; tools.terminal_tool already imports tools.approval.
            from tools.approval import human_wait_window
            with human_wait_window():
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
        # Blocked on a human typing their password: exclude from tool
        # deadlines on both executor paths (#85125 2e). Local import avoids
        # any import-layering surprises.
        from tools.approval import human_wait_window
        with human_wait_window():
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
    terminal_env = _tenv("TERMINAL_ENV", "local").strip().lower() or "local"
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
        #
        # The consumed `&` also separated the compound from any statement
        # that followed on the same line (`A && B & C`); `{ B & } C` is a
        # syntax error, so restore a `;` when the suffix resumes with command
        # text. No separator when the suffix already starts with a
        # terminator (`;` `&` `|` newline `)` `}`) — except `&>`, which is a
        # redirect prefix for the NEXT command, not a terminator.
        tail = suffix.lstrip(" \t")
        needs_separator = bool(tail) and (
            tail[0] not in ";\n&|)}" or tail.startswith("&>")
        )
        separator = " ;" if needs_separator else ""
        result = prefix + "{ " + middle + "& }" + separator + suffix

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

    # Scope-aware read (Slack pattern): under multiplex the process env may
    # hold another profile's SUDO_PASSWORD, so honor the installed scope's
    # verdict; unscoped callers keep the legacy os.environ read.
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            _configured_password = get_secret("SUDO_PASSWORD")
        except UnscopedSecretError:
            _configured_password = os.environ.get("SUDO_PASSWORD")
    except Exception:
        _configured_password = os.environ.get("SUDO_PASSWORD")
    has_configured_password = _configured_password is not None
    sudo_password = (
        _configured_password
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
    # delegate_task children inherit the parent's process-wide
    # HERMES_INTERACTIVE=1 (and, on a recycled worker thread, potentially a
    # stale thread-local callback), but there is no user on the other side of
    # this execution context: prompting from a subagent thread fights the
    # parent's TUI for /dev/tty and blocks the child for the full timeout.
    # Children always behave as headless — configured SUDO_PASSWORD, the
    # session cache, and the NOPASSWD probe above all still work.
    should_prompt_for_sudo = (
        env_var_enabled("HERMES_INTERACTIVE") or has_sudo_prompt_callback
    ) and not _in_delegated_child_context()
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
from tools.environments.base import EnvironmentConnectionError


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands. The host OS, shell, and terminal backend are stated in your environment section — write commands for THAT platform. Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail (use read_file), grep/rg/find/ls (use search_files), sed/awk (use patch), or echo/heredoc file creation (use write_file). Reserve terminal for: builds, installs, git, processes, scripts, network, package managers — anything that needs a shell. Output is auto-truncated with the full text saved to a file — never pipe through tail/head to shorten it.
Environment state persists: activate a virtualenv or export variables once per session, not before every command.

Foreground (default): returns INSTANTLY when the command finishes, even with a high timeout — set timeout generously for long builds.
Background: set background=true (returns a session_id); add notify=true for bounded tasks, leave silent only for servers/daemons that never exit. After starting a server, verify readiness with a health check in a separate call (no blind sleep loops); manage with process(action="poll"/"wait").
Working directory: use 'workdir' for per-command cwd; when a command changes the session cwd (cd, pushd), trust the result's "cwd" field instead of prefixing every command with 'cd'.
PTY: pty=true + background=true for interactive CLIs (they hang without a terminal); drive them with process(action="write"/"submit"). Local backend only.
"""

# Environment lifecycle state.
_active_environments: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_creation_locks: Dict[str, threading.Lock] = {}  # Per-task locks for sandbox creation
_creation_locks_lock = threading.Lock()  # Protects _creation_locks dict itself
_cleanup_thread = None
_cleanup_running = False

# Once-per-process guard for the docker orphan reaper.
_docker_orphan_reaper_ran = False
_docker_orphan_reaper_lock = threading.Lock()


def _maybe_reap_docker_orphans(container_config: Dict[str, Any]) -> None:
    """Run the docker orphan reaper once per process, if enabled.

    Sweeps Exited containers labeled ``hermes-agent=1`` for the current
    profile — leftovers of Hermes processes that died without firing
    ``atexit`` (SIGKILL, OOM, closed terminal). Conservative: only containers
    older than ``2 × lifetime_seconds``, profile-scoped. Gates:
    ``terminal.docker_orphan_reaper: false`` (operator opt-out, e.g. several
    Hermes processes sharing a profile) and the once-per-interpreter flag so
    parallel subagent / RL-rollout calls don't re-sweep.
    """
    global _docker_orphan_reaper_ran
    if not container_config.get("docker_orphan_reaper", True):
        return
    if _docker_orphan_reaper_ran:  # double-checked locking
        return
    with _docker_orphan_reaper_lock:
        if _docker_orphan_reaper_ran:
            return
        _docker_orphan_reaper_ran = True

    # 2 × lifetime gives sibling processes a grace window; floor at 60s so
    # TERMINAL_LIFETIME_SECONDS=0 can't instant-reap a sibling's own setup.
    # container_config only carries container_* keys, so read the env var.
    try:
        lifetime = int(_tenv("TERMINAL_LIFETIME_SECONDS", "300"))
    except (TypeError, ValueError):
        lifetime = 300
    max_age = max(60, lifetime) * 2

    try:
        from tools.environments.docker import reap_orphan_containers, _container_identity
    except ImportError:
        return
    try:
        profile = _container_identity(container_config.get("docker_shared_container_key", ""))
        removed = reap_orphan_containers(
            max_age_seconds=max_age, profile_filter=profile,
        )
        if removed:
            logger.info(
                "Docker orphan reaper removed %d stale container(s) for profile %s",
                removed, profile,
            )


# Per-task environment overrides (never exposed to the model). RL/benchmark
# envs and ACP register a custom image / cwd for a task_id BEFORE the agent
# loop; sandbox creation consults this first, then the TERMINAL_* env vars.
_task_env_overrides: Dict[str, Dict[str, Any]] = {}

# Per-session cwd records: the durable source of truth for "which directory
# is THIS session in". Keyed by the raw session/task key, NOT the collapsed
# container id — the env is shared across sessions, so cwd state stored on
# it is a global mutable timeshared between sessions (the wrong-worktree bug
# class). Written after every completed command and on cwd-override
# registration; readers resolve against it before any env-side cwd.
_session_cwd: Dict[str, str] = {}
_session_cwd_lock = threading.Lock()

# Subagent → parent container aliasing. delegate_task children have their own
# task_id but must share the PARENT's container; under per-session isolation
# the collapse-to-"default" shortcut no longer provides that, so the spawn
# site registers an explicit alias.
_container_aliases: Dict[str, str] = {}
_container_alias_lock = threading.Lock()


def record_session_cwd(session_key: Optional[str], cwd: Optional[str]) -> None:
    """Record *cwd* as *session_key*'s working directory (after a completed
    command, or on workspace-override registration). None/empty keys collapse
    to ``"default"``; non-string / empty cwds are ignored."""
    if not isinstance(cwd, str) or not cwd.strip():
        return
    key = str(session_key or "default")
    with _session_cwd_lock:
        if _session_cwd.get(key) != cwd:
            _session_cwd[key] = cwd


def get_session_cwd(session_key: Optional[str]) -> Optional[str]:
    """Recorded cwd for *session_key*, or None. No fallback chain on purpose:
    callers decide what an absent record means. None/empty keys read ``"default"``."""
    with _session_cwd_lock:
        return _session_cwd.get(str(session_key or "default"))


def clear_session_cwd(session_key: str) -> None:
    """Drop a session's cwd record (session teardown)."""
    with _session_cwd_lock:
        _session_cwd.pop(session_key, None)


def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]):
    """Register per-task sandbox overrides (``docker_image``/``modal_image``/
    ``singularity_image``/``daytona_image``, ``env_type``, ``cwd``) before the
    agent loop runs.

    A ``cwd`` override takes effect immediately: it becomes the session's
    recorded cwd (until a ``cd`` changes it) and any live env's cwd is updated
    too, so env-side seeding stays consistent (ACP switching project root
    mid-session via ``session/load``).
    """
    _task_env_overrides[task_id] = overrides

    new_cwd = overrides.get("cwd")
    if isinstance(new_cwd, str) and new_cwd.strip():
        record_session_cwd(task_id, new_cwd)
        # Live env may be cached under the raw task_id (per-session surfaces)
        # or the collapsed container id (isolation-keyed rollouts); try both so
        # a CWD-only override (which collapses to "default") still finds it.
        container_id = _resolve_container_task_id(task_id)
        with _env_lock:
            env = _active_environments.get(task_id) or _active_environments.get(container_id)
        if env is not None and getattr(env, "cwd", None) is not None:
            env.cwd = new_cwd


def clear_task_env_overrides(task_id: str):
    """Drop a task's overrides, cwd record and container alias (rollout cleanup)."""
    _task_env_overrides.pop(task_id, None)
    clear_session_cwd(task_id)
    with _container_alias_lock:
        _container_aliases.pop(task_id, None)


def register_container_alias(child_task_id: str, parent_task_id: Optional[str]) -> None:
    """Make *child_task_id* resolve to *parent_task_id*'s container (called at
    delegate_task spawn). A missing parent id aliases to ``"default"``."""
    if not child_task_id:
        return
    with _container_alias_lock:
        _container_aliases[child_task_id] = str(parent_task_id or "default")


def _resolve_container_alias(task_id: str) -> str:
    """Follow the child→parent alias chain (cycle-safe) for *task_id*."""
    seen = set()
    key = task_id
    with _container_alias_lock:
        while key in _container_aliases and key not in seen:
            seen.add(key)
            key = _container_aliases[key]
    return key


def _session_isolation_enabled() -> bool:
    """True when non-persistent sandboxes get per-session identities.

    ``container_persistent: false`` is a statement that state must not
    survive or be shared across sessions, so sharing one sandbox across
    sessions contradicts it (#82731). Backends whose non-persistent mode is
    session-scoped:

    - ``docker`` — per-session containers (the original fix).
    - plugin backends that declare ``session_isolated_when_nonpersistent``
      (e.g. sandboxes resumed *by name*, where a shared deterministic name
      under non-persistent mode would let two independent ephemeral runs
      attach one live VM and delete it out from under each other).
    """
    _ensure_terminal_env_bridged()
    env_type = _tenv("TERMINAL_ENV", "local")
    if env_type != "docker" and not _plugin_env_flag(
        env_type, "session_isolated_when_nonpersistent"
    ):
        return False
    return _tenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() not in {"true", "1", "yes"}


def _docker_session_isolation_enabled() -> bool:
    """Docker-specific view of :func:`_session_isolation_enabled`.

    Kept separate because several docker-only paths (workspace mount
    selection, session-scoped container teardown) key off it; those must
    not fire for other backends.
    """
    if _tenv("TERMINAL_ENV", "local") != "docker":
        return False
    return _session_isolation_enabled()


def _docker_persistent_profile_scoped() -> bool:
    """True when the persistent Docker container is shared per PROFILE.

    The product contract for ``TERMINAL_ENV=docker`` +
    ``container_persistent: true`` is ONE long-lived container per Hermes
    profile, shared by every session of that profile (CLI, gateway chats,
    WebUI). Commit a270c4ade added a session-key fallback to
    :func:`_resolve_container_task_id` to stop cross-profile SSH environment
    reuse, but the fallback wasn't backend-gated, so persistent Docker
    silently fragmented into one container per gateway session (#93950 was
    downstream damage from that). This predicate gates the resolver back to
    profile scoping for exactly this backend/mode; SSH and other backends
    keep the session-scoped cache key that fixed the original leak.
    """
    _ensure_terminal_env_bridged()
    if _tenv("TERMINAL_ENV", "local") != "docker":
        return False
    return _tenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"true", "1", "yes"}


def _current_session_profile() -> str:
    """Return the active session's Hermes profile name, or "" when unset.

    Same lookup discipline as :func:`_current_session_key`: the ContextVar
    (bound per message by the gateway, per session by the WebUI streaming
    layer) with the ``get_session_env`` os.environ fallback for CLI, cron,
    and test processes.
    """
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_PROFILE", "")


_ISOLATION_OVERRIDE_KEYS = frozenset({
    "docker_image", "modal_image", "singularity_image",
    "daytona_image", "env_type",
})


def _has_isolation_overrides(task_id: Optional[str]) -> bool:
    """True when *task_id* registered image/env_type overrides — the single
    "isolated RL/benchmark rollout" predicate shared by key resolution and
    container creation so the two can't drift."""
    if not task_id or task_id not in _task_env_overrides:
        return False
    return bool(set(_task_env_overrides[task_id].keys()) & _ISOLATION_OVERRIDE_KEYS)


@dataclass(frozen=True)
class _SessionScope:
    """Backend identity + scoping predicates for one call, read once.

    ``env_type`` is the scope-aware TERMINAL_ENV; ``persistent`` is
    ``TERMINAL_CONTAINER_PERSISTENT``. Derived predicates:

    * ``session_isolated`` — non-persistent sandboxes get per-session identities:
      ``container_persistent: false`` means state must not survive or be shared
      across sessions, so one shared sandbox contradicts it. Docker, plus plugin
      backends declaring ``session_isolated_when_nonpersistent`` (sandboxes resumed
      by name, where a shared deterministic name would let two ephemeral runs
      attach one VM and delete it under each other).
    * ``docker_session_isolated`` — docker-only view: the workspace mount and
      session-scoped teardown paths must not fire for other backends.
    * ``docker_profile_scoped`` — docker + ``container_persistent: true``: ONE
      long-lived container per profile shared by every session (CLI, gateway,
      WebUI). The session-key fallback in :func:`_resolve_container_task_id` stops
      cross-profile SSH reuse; ungated it fragmented persistent Docker into one
      container per gateway session, so this restores profile scoping for exactly
      this backend/mode.
    """
    env_type: str
    persistent: bool

    @property
    def session_isolated(self) -> bool:
        if self.env_type != "docker" and not _plugin_env_flag(
            self.env_type, "session_isolated_when_nonpersistent"
        ):
            return False
        return not self.persistent

    @property
    def docker_session_isolated(self) -> bool:
        return self.env_type == "docker" and self.session_isolated

    @property
    def docker_profile_scoped(self) -> bool:
        return self.env_type == "docker" and self.persistent


def _session_scope() -> _SessionScope:
    """Bridge config → env once, then snapshot the backend scope for this call."""
    _ensure_terminal_env_bridged()
    return _SessionScope(
        env_type=_tenv("TERMINAL_ENV", "local"),
        persistent=_tenv_bool("TERMINAL_CONTAINER_PERSISTENT", "true"),
    )


def _docker_session_isolation_enabled() -> bool:
    """See :attr:`_SessionScope.docker_session_isolated` (used by the docker builder)."""
    return _session_scope().docker_session_isolated


def _resolve_container_task_id(task_id: Optional[str]) -> str:
    """Map a tool-call ``task_id`` to the ``_active_environments`` key. Order matters —
    earlier branches are authoritative where they apply:

    1. Image/``env_type`` overrides (RL/benchmark rollouts) key their own sandbox;
       CWD-only overrides (ACP workspace tracking) are NOT isolation signals.
    2. Per-session isolation (docker + ``container_persistent: false``): each
       session's task_id is its own key (a fresh chat gets a fresh sandbox with only
       ITS mounts); delegate_task children follow the alias registry to the parent.
    3. Session key present (WebUI per-session, gateway per-message): persistent
       Docker is PROFILE-scoped — ``shared:<key>`` opt-in, else ``profile:<name>``,
       with the default profile staying literally ``"default"`` so CLI and
       default-profile gateway sessions share ONE container; other backends key
       ``session:<key>`` so switching profiles can't reuse another profile's
       SSHEnvironment on the wrong host.
    4. No session key (CLI): ``shared:<key>`` when opted in (else a CLI run of a
       keyed profile would split from its gateway sessions), else ``"default"``,
       which subagent ids collapse onto to share the parent's container.
    """
    if task_id and _has_isolation_overrides(task_id):
        return task_id
    if task_id and _session_isolation_enabled():
        return _resolve_container_alias(task_id)
    # Per-session isolation: when a session key is present (the WebUI streaming
    # layer sets it per-session, the gateway per-message via contextvars), scope
    # the container to it so switching profiles can't reuse a previous profile's
    # SSHEnvironment and silently run commands on the wrong remote host. Subagents
    # inherit the same session key, so they still collapse onto the parent's
    # container (the #16177 shared-container intent). CLI mode has no session key
    # and falls through to "default", behaviour unchanged. See commit e00f940a9.
    #
    # This runs *after* the isolation-override and docker/container_persistent
    # branches above: those paths already key containers per task_id, so they
    # stay authoritative where they apply and this only covers the cases that
    # would otherwise collapse to the shared "default" key (notably SSH).
    session_key = _current_session_key()
    if session_key:
        # Persistent Docker is PROFILE-scoped by contract: one long-lived
        # container shared by every session of the profile. Key it by profile
        # (not session) so gateway chats, CLI, and WebUI all land in the same
        # container and sandbox. The bare "profile:default" key stays literally
        # "default" so CLI mode (no session key at all) and gateway sessions of
        # the default profile share the SAME container — CLI's historical key
        # IS the default profile's container.
        if _docker_persistent_profile_scoped():
            # Explicit opt-in: trusted profiles configuring the same
            # terminal.docker_shared_container_key share ONE container/cache
            # slot (and sandbox dir) regardless of profile name (#84671).
            shared = _tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip()
            if shared:
                return f"shared:{shared}"
            profile = _current_session_profile() or "default"
            if profile == "default":
                return "default"
            return f"profile:{profile}"
        return f"session:{session_key}"
    # CLI/no-session path: honour the shared-container opt-in here too, or a
    # CLI run of a keyed profile would land in "default" while its gateway
    # sessions land in "shared:<key>" — splitting the very container the
    # setting exists to unify.
    if _docker_persistent_profile_scoped():
        shared = _tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip()
        if shared:
            return f"shared:{shared}"
    return "default"


def resolve_task_overrides(task_id: Optional[str]) -> Dict[str, Any]:
    """Return the env overrides for *task_id*, raw key first then collapsed.

    ``register_task_env_overrides`` writes under the *raw* task/session id, but
    a CWD-only override collapses (:func:`_resolve_container_task_id`) to the
    shared ``"default"`` container. Callers must therefore read the raw id
    FIRST and only fall back to the collapsed container id, or the originating
    session's override is silently dropped. Single source of that lookup so
    the terminal and file layers can't drift apart.
    """
    raw = task_id or "default"
    return (
        _task_env_overrides.get(raw)
        or _task_env_overrides.get(_resolve_container_task_id(raw))
        or {}
    )


# Backends that take an image, keyed to the override/config key carrying it.
_IMAGE_KEY_BY_BACKEND = {
    "docker": "docker_image",
    "singularity": "singularity_image",
    "modal": "modal_image",
    "daytona": "daytona_image",
}


def _select_image(env_type: str, overrides: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Image for *env_type*: per-task override first, then config; "" for imageless backends."""
    key = _IMAGE_KEY_BY_BACKEND.get(env_type)
    if key is None:
        return ""
    return overrides.get(key) or config[key]


def _lookup_active_env(effective_task_id: str, task_id: Optional[str]):
    """Return the cached env for the collapsed id, else for the raw task_id, else None.

    Caller holds ``_env_lock``. Per-session surfaces (ACP/gateway/dashboard)
    with a CWD-only override collapse to ``"default"`` for container sharing,
    yet an env may already be cached under the originating task_id; honor it
    instead of spawning a duplicate. Refreshes ``_last_activity`` on a hit.
    """
    for key in (effective_task_id, task_id):
        if key and key in _active_environments:
            _last_activity[key] = time.time()
            return _active_environments[key]
    return None


def _resolve_task_host_cwd(config: Dict[str, Any], task_id: Optional[str]) -> Optional[str]:
    """Host directory to bind-mount at ``/workspace`` for *task_id*'s container.

    Single owner of the cwd-mount policy for every creation site. Shared-
    container mode: the ``TERMINAL_CWD``-derived ``config["host_cwd"]``.
    Per-session isolation (docker + ``container_persistent: false``): only
    the SESSION's own registered workspace may mount — the process env var is
    a launch artifact that outlives the session that set it, so deriving a
    fresh session's mount from it would leak the previous session's directory.
    Overrides tagged ``cwd_source: "process"`` are refused for the same reason;
    ``cwd_source: "session"`` or untagged (ACP/RL) overrides mount.
    """
    if config.get("env_type") != "docker" or not config.get("docker_mount_cwd_to_workspace"):
        return None
    # Top-level CLI parent ("default") is a single-session process — legacy behavior.
    if not _docker_session_isolation_enabled() or _resolve_container_task_id(task_id) == "default":
        return config.get("host_cwd")
    overrides = resolve_task_overrides(task_id)
    candidate = overrides.get("cwd")
    if overrides.get("cwd_source") == "process" or not isinstance(candidate, str) or not candidate.strip():
        return None
    candidate = os.path.abspath(os.path.expanduser(candidate))
    # Must exist on the host and not already be an in-container path.
    if not os.path.isdir(candidate) or candidate.startswith(("/workspace", "/root")):
        return None
    return candidate


# Configuration from environment variables

def _parse_env_var(name: str, default: str, converter: Any = int, type_label: str = "integer"):
    """Parse an environment variable with *converter*, raising a clear error on bad values.

    Without this wrapper, a single malformed env var (e.g. TERMINAL_TIMEOUT=5m)
    causes an unhandled ValueError that kills every terminal command.
    """
    raw = os.getenv(name, default)
    if name.startswith("TERMINAL_"):
        # Scope-aware: under gateway multiplexing the active profile's
        # per-turn scope overrides the process env.
        raw = _tenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            f"Check ~/.hermes/.env or environment variables."
        )


def _safe_getcwd() -> str:
    """Return the current working directory, tolerating a deleted or
    permission-restricted CWD.

    ``os.getcwd()`` raises FileNotFoundError when the process's working
    directory has been removed out from under it (e.g. a scratch workspace
    that was cleaned up mid-session). On macOS with TCC (Transparency,
    Consent, and Control), it raises PermissionError (EPERM) when the CWD
    is under a protected location (~/Documents, ~/Desktop, ~/Downloads)
    and the calling process lacks Full Disk Access. Fall back to
    TERMINAL_CWD, then the user's home directory, so terminal setup never
    crashes on a stale or TCC-blocked CWD.
    """
    try:
        return os.getcwd()
    except (FileNotFoundError, PermissionError):
        return _tenv("TERMINAL_CWD") or os.path.expanduser("~")


# Path prefixes that identify a *host* working directory which cannot exist
# inside a container sandbox. Covers POSIX user dirs and Windows drive paths
# (``C:\Users\...`` / ``C:/Users/...``) — the latter is how a Windows host's
# cwd looks when it leaks toward a Linux container's ``-w`` flag.
_HOST_CWD_PREFIXES = ("/Users/", "/home/", "C:\\", "C:/")

_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})


def _plugin_env_flag(env_type: str, attr: str, default=False):
    """Classification attribute for a plugin-registered terminal backend.

    Fail-soft: returns *default* when the registry is unavailable, the
    backend is unknown, or the provider attribute raises — a misbehaving
    plugin must degrade, never take the terminal tool down.
    """
    if not env_type or env_type in _CONTAINER_BACKENDS or env_type in {"local", "ssh", "managed_modal"}:
        return default
    try:
        from agent.terminal_env_registry import provider_flag

        return provider_flag(env_type, attr, default)
    except Exception:
        return default


def _is_container_backend(env_type: str) -> bool:
    """True when *env_type* behaves like a container/sandbox backend.

    Built-in container backends via ``_CONTAINER_BACKENDS``; plugin-registered
    backends via their declarative ``is_container`` flag.
    """
    return env_type in _CONTAINER_BACKENDS or _plugin_env_flag(env_type, "is_container")


def _get_plugin_env_provider(env_type: str):
    """Return the registered plugin provider for *env_type*, or None."""
    if not env_type or env_type in _CONTAINER_BACKENDS or env_type in {"local", "ssh", "managed_modal"}:
        return None
    try:
        from agent.terminal_env_registry import get_provider

        return get_provider(env_type)
    except Exception:
        return None


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


def _tenv(name: str, default: str = "") -> str:
    """Scope-aware read of a ``TERMINAL_*`` variable.

    Every terminal setting read in this module must go through this helper:
    under gateway multiplexing the active profile's terminal config arrives
    via a per-turn scope (``tools.terminal_scope``), and a raw ``os.getenv``
    would read whatever profile's config a previous turn pinned into the
    process env (the cross-profile backend leak fixed here).
    """
    from tools.terminal_scope import terminal_env

    return terminal_env(name, default)


# One-shot guard for the config-fallback bridge below.  Purely an
# optimization: after the first attempt either TERMINAL_ENV is set (bridge
# succeeded — merged config always carries terminal.backend) or the import
# failed and retrying every call would be wasted work.
_terminal_config_bridge_attempted = False


def _ensure_terminal_env_bridged() -> None:
    """Backfill TERMINAL_* env vars from config.yaml when no launcher did.

    CLI, gateway and TUI/dashboard PTY launches bridge ``terminal.*`` into env vars
    at startup; processes that skip those paths (``hermes serve``, Desktop
    in-process agents, desktop cron ticker, ACP) would otherwise fall back to the
    local backend even when config selects docker — running on the host the user
    meant to sandbox. Explicit keys in the ``terminal`` section override matching
    env values (possibly stale from ``hermes setup``); env values for omitted keys
    are preserved. Without a terminal section an existing TERMINAL_ENV is kept and
    defaults are backfilled only when none is set. A per-turn terminal scope
    suppresses the bridge entirely: writing scope values into the process-global
    env would re-create the first-writer-wins cross-profile leak the scope fixes.

    Ambient ``os.environ`` is the *launch* profile's authority only. Under a
    context-local ``HERMES_HOME`` override (multiplexed dashboard / gateway
    secondary profile), this bridge is a no-op — otherwise the first unscoped
    call under that override would latch the secondary profile's ``terminal.*``
    into process-global env and poison later unscoped launch-profile turns
    (#107422 residual of #68559). Routed profiles must bind a terminal scope
    instead (same rule as ``env_loader._reapply_terminal_config_bridge``).

    terminal_tool reads ALL terminal settings from os.environ (TERMINAL_*). See #61115, #65696.
    """
    from tools.terminal_scope import get_terminal_scope

    if get_terminal_scope() is not None:
        return
    # Never write a secondary profile's terminal.* into process-global env.
    from hermes_constants import get_hermes_home_override

    if get_hermes_home_override() is not None:
        return
    global _terminal_config_bridge_attempted
    if _terminal_config_bridge_attempted:
        return
    _terminal_config_bridge_attempted = True
    # Never let a config problem take the terminal tool down.
    with _quiet("terminal config → env fallback bridge failed"):
        from hermes_cli.config import apply_terminal_config_to_env, read_raw_config

        raw_config = read_raw_config()
        if isinstance(raw_config.get("terminal"), dict):
            apply_terminal_config_to_env(env=None, override=True)
        elif "TERMINAL_ENV" not in os.environ:
            apply_terminal_config_to_env(env=None, override=False)


def _get_env_config() -> Dict[str, Any]:
    """Get terminal environment configuration from environment variables."""
    # Default image with Python and Node.js for maximum compatibility
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    _ensure_terminal_env_bridged()
    env_type = _tenv("TERMINAL_ENV", "local")
    
    mount_docker_cwd = _tenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false").lower() in {"true", "1", "yes"}
    container_backend = _is_container_backend(env_type)
    docker_backend = env_type == "docker"


    if docker_backend:
        docker_forward_env = _parse_env_var("TERMINAL_DOCKER_FORWARD_ENV", "[]", json.loads, "valid JSON")
        docker_volumes = _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")
        docker_env = _parse_env_var("TERMINAL_DOCKER_ENV", "{}", json.loads, "valid JSON")
        docker_extra_args = _parse_env_var("TERMINAL_DOCKER_EXTRA_ARGS", "[]", json.loads, "valid JSON")
        docker_shm_size = _tenv("TERMINAL_DOCKER_SHM_SIZE", "1g")
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
    cwd = _tenv("TERMINAL_CWD", default_cwd)
    from hermes_cli.config import _is_ssh_remote_tilde_cwd
    if cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        docker_cwd_source = _tenv("TERMINAL_CWD") or _safe_getcwd()
        candidate = os.path.abspath(os.path.expanduser(docker_cwd_source))
        if (
            _is_host_cwd(candidate)
            or (os.path.isabs(candidate) and os.path.isdir(candidate) and not candidate.startswith(("/workspace", "/root")))
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif _is_container_backend(env_type) and cwd:
        # Host paths and relative paths that won't work inside containers
        if _is_unusable_container_cwd(cwd) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                        "(host/relative path won't work in sandbox). Using %r instead.",
                        cwd, env_type, default_cwd)
            cwd = default_cwd

    return {
        "env_type": env_type,
        "modal_mode": coerce_modal_mode(_tenv("TERMINAL_MODAL_MODE", "auto")),
        "docker_image": _tenv("TERMINAL_DOCKER_IMAGE", default_image),
        "docker_forward_env": docker_forward_env,
        "singularity_image": _tenv("TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}"),
        "modal_image": _tenv("TERMINAL_MODAL_IMAGE", default_image),
        "daytona_image": _tenv("TERMINAL_DAYTONA_IMAGE", default_image),
        "vercel_runtime": _tenv("TERMINAL_VERCEL_RUNTIME", "").strip(),
        "cwd": cwd,
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": _parse_env_var("TERMINAL_TIMEOUT", "180"),
        "lifetime_seconds": _parse_env_var("TERMINAL_LIFETIME_SECONDS", "300"),
        # SSH-specific config
        "ssh_host": _tenv("TERMINAL_SSH_HOST", ""),
        "ssh_user": _tenv("TERMINAL_SSH_USER", ""),
        "ssh_port": _parse_env_var("TERMINAL_SSH_PORT", "22"),
        "ssh_key": _tenv("TERMINAL_SSH_KEY", ""),
        # Persistent shell: SSH defaults to the config-level persistent_shell
        # setting (true by default for non-local backends); local is always opt-in.
        # Per-backend env vars override if explicitly set.
        "ssh_persistent": _tenv(
            "TERMINAL_SSH_PERSISTENT",
            _tenv("TERMINAL_PERSISTENT_SHELL", "true"),
        ).lower() in {"true", "1", "yes"},
        "local_persistent": _tenv("TERMINAL_LOCAL_PERSISTENT", "false").lower() in {"true", "1", "yes"},
        # Container resource config (applies to docker, singularity, modal,
        # daytona, and vercel_sandbox -- ignored for local/ssh)
        "container_cpu": container_cpu,
        "container_memory": container_memory,     # MB (default 5GB)
        "container_disk": container_disk,        # MB (default 50GB)
        "container_persistent": _tenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"true", "1", "yes"},
        "docker_volumes": docker_volumes,
        "docker_env": docker_env,
        "docker_run_as_host_user": _tenv("TERMINAL_DOCKER_RUN_AS_HOST_USER", "false").lower() in {"true", "1", "yes"},
        "docker_network": _tenv("TERMINAL_DOCKER_NETWORK", "true").lower() in {"true", "1", "yes"},
        "docker_extra_args": docker_extra_args,
        "docker_shm_size": docker_shm_size,
        # Cross-process container reuse (issue #20561).  The docs claim
        # "ONE long-lived container shared across sessions" — this toggle
        # makes that real by probing for a labeled container at startup and
        # attaching to it instead of always starting a fresh one.  Set to
        # ``false`` for hard per-process isolation (no reuse, container is
        # removed on exit).
        "docker_persist_across_processes": _tenv(
            "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "true"
        ).lower() in {"true", "1", "yes"},
        "docker_shared_container_key": _tenv(
            "TERMINAL_DOCKER_SHARED_CONTAINER_KEY", ""
        ).strip(),
        # Startup orphan reaper for hermes-tagged containers left behind by
        # crashed / SIGKILL'd previous processes that bypassed atexit.
        # Conservative: only sweeps Exited containers older than 2× the
        # idle-reap window AND scoped to the current profile. Issue #20561.
        "docker_orphan_reaper": _tenv(
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


def _ssh_config_from_config(config: Dict[str, Any]) -> dict:
    """Build the ``ssh_config`` dict passed to :func:`_create_environment`.

    Shared by the terminal tool's own get-or-create path and the lazy
    :func:`ensure_task_env` bring-up so both derive SSH connection settings
    from the resolved config identically.
    """
    return {
        "host": config.get("ssh_host", ""),
        "user": config.get("ssh_user", ""),
        "port": config.get("ssh_port", 22),
        "key": config.get("ssh_key", ""),
        "persistent": config.get("ssh_persistent", False),
    }


def _container_config_from_config(config: Dict[str, Any]) -> dict:
    """Build the ``container_config`` dict passed to :func:`_create_environment`.

    Shared by the terminal tool's own get-or-create path and the lazy
    :func:`ensure_task_env` bring-up (see :func:`_ssh_config_from_config`).
    """
    return {
        "container_cpu": config.get("container_cpu", 1),
        "container_memory": config.get("container_memory", 5120),
        "container_disk": config.get("container_disk", 51200),
        "container_persistent": config.get("container_persistent", True),
        "modal_mode": config.get("modal_mode", "auto"),
        "vercel_runtime": config.get("vercel_runtime", ""),
        "docker_volumes": config.get("docker_volumes", []),
        "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
        "docker_forward_env": config.get("docker_forward_env", []),
        "docker_env": config.get("docker_env", {}),
        "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
        "docker_extra_args": config.get("docker_extra_args", []),
        "docker_shm_size": config.get("docker_shm_size", "1g"),
        "docker_network": config.get("docker_network", True),
        "docker_persist_across_processes": config.get("docker_persist_across_processes", True),
        "docker_shared_container_key": config.get("docker_shared_container_key", ""),
        "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
    }


def _create_environment(env_type: str, image: str, cwd: str, timeout: int,
                        ssh_config: dict = None, container_config: dict = None,
                        local_config: dict = None,
                        task_id: str = "default",
                        host_cwd: Optional[str] = None):
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
        return _LocalEnvironment(cwd=cwd, timeout=timeout)
    
    elif env_type == "docker":
        # One-shot orphan reaper: clean up labeled containers left behind by
        # prior Hermes processes that hit SIGKILL / OOM / a closed terminal
        # before the atexit cleanup hook could run.  Gated to once per
        # process so concurrent _create_environment calls (parallel
        # subagents, RL benchmarks) don't run the reaper N times.
        # Disable via ``terminal.docker_orphan_reaper: false`` (issue #20561).
        _maybe_reap_docker_orphans(cc)
        # Per-session container isolation: a session-keyed container must not
        # outlive its session, so cross-process reuse/persist is disabled for
        # it — cleanup_vm()/the idle reaper stop+rm it instead of leaving a
        # running container behind for every chat ever opened. The shared
        # "default" container and RL/benchmark override sandboxes keep their
        # existing lifecycle.
        session_scoped = (
            _docker_session_isolation_enabled()
            and task_id != "default"
            and not _has_isolation_overrides(task_id)
        )
        docker_env_obj = _DockerEnvironment(
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
            persist_across_processes=(
                False if session_scoped
                else cc.get("docker_persist_across_processes", True)
            ),
            shared_container_key=cc.get("docker_shared_container_key", ""),
            shm_size=cc.get("docker_shm_size", "1g"),
        )
        # Marker read by is_persistent_env(): a session-scoped container
        # survives BETWEEN turns (skip per-turn teardown) but is removed at
        # session close / idle timeout. Guarded setattr: test doubles for
        # _DockerEnvironment may not accept attributes.
        if session_scoped:
            try:
                docker_env_obj._session_scoped = True
            except AttributeError:
                pass
        return docker_env_obj
    
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
        )

    else:
        provider = _get_plugin_env_provider(env_type)
        if provider is not None:
            env_obj = provider.create_environment(
                cwd=cwd, timeout=timeout, task_id=task_id,
                image=image, container_config=cc,
            )
            # Stamp the backend name so path-resolution and progress surfaces
            # can identify plugin backends without class-name sniffing.
            try:
                env_obj._hermes_backend_name = provider.name.strip().lower()
            except AttributeError:
                pass  # test doubles may reject attributes
            return env_obj
        try:
            from agent.terminal_env_registry import plugin_backend_names

            plugin_names = plugin_backend_names()
        except Exception:
            plugin_names = []
        extra = (
            ", " + ", ".join(f"'{n}'" for n in plugin_names) if plugin_names else ""
        )
        raise ValueError(
            f"Unknown environment type: {env_type}. Use 'local', 'docker', "
            f"'singularity', 'modal', 'daytona', 'vercel_sandbox', 'ssh'{extra}"
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
        with _quiet("Error in cleanup thread", level=logging.WARNING):
            _cleanup_inactive_envs(_get_env_config()["lifetime_seconds"])
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


def ensure_task_env(task_id: Optional[str] = None):
    """Lazily create and cache the sandbox env for *task_id* if none is active.

    :func:`terminal_tool` creates the environment on the first terminal command,
    but nothing else did — so under a non-local backend (ssh, docker, …) a
    session whose first action is ``vision_analyze`` on a container-only path hit
    "no active sandbox session" because the SSH/Docker handshake never ran
    (issue #62825). vision reads such paths inside the sandbox (see
    ``tools.image_source``), so it calls this to bring the env up on demand,
    reusing the same creation machinery as the terminal tool.

    No-op on the local backend (images are read host-side). Returns the env
    instance, or ``None`` when local or when creation fails (best-effort: a
    failure leaves the caller's fail-closed error path intact).
    """
    config = _get_env_config()
    env_type = config["env_type"]
    if env_type == "local":
        return None

    effective_task_id = _resolve_container_task_id(task_id)

    # Fast path: already active — mirror terminal_tool and refresh activity.
    existing = get_active_env(effective_task_id)
    if existing is not None:
        with _env_lock:
            _last_activity[effective_task_id] = time.time()
        return existing

    overrides = resolve_task_overrides(task_id)
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

    _start_cleanup_thread()

    # Per-task creation lock so a concurrent terminal_tool call and this helper
    # don't each spawn a sandbox for the same task.
    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(effective_task_id, threading.Lock())

    with task_lock:
        existing = get_active_env(effective_task_id)
        if existing is not None:
            return existing
        try:
            new_env = _create_environment(
                env_type=env_type,
                image=image,
                cwd=config["cwd"],
                timeout=config["timeout"],
                ssh_config=_ssh_config_from_config(config) if env_type == "ssh" else None,
                container_config=(
                    _container_config_from_config(config)
                    if _is_container_backend(env_type) else None
                ),
                local_config=None,
                task_id=effective_task_id,
                host_cwd=_resolve_task_host_cwd(config, task_id),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort bring-up
            logger.warning(
                "Lazy %s environment init failed for task %s: %s",
                env_type, effective_task_id[:8], exc,
            )
            return None

        with _env_lock:
            _active_environments[effective_task_id] = new_env
            _last_activity[effective_task_id] = time.time()
        logger.info(
            "%s environment lazily initialized for task %s",
            env_type, effective_task_id[:8],
        )
        return new_env


def is_persistent_env(task_id: str) -> bool:
    """Return True if the active environment for task_id is configured for
    cross-turn persistence (``persistent_filesystem=True``).

    Used by the agent loop to skip per-turn teardown for backends whose whole
    point is to survive between turns (docker with ``container_persistent``,
    daytona, modal, etc.). Non-persistent backends (e.g. Morph) still get torn
    down at end-of-turn to prevent leakage. The idle reaper
    (``_cleanup_inactive_envs``) handles persistent envs once they exceed
    ``terminal.lifetime_seconds``.

    Session-scoped docker containers (per-session isolation mode) also count
    as persistent HERE: their lifetime is the SESSION, not the turn — they
    are removed by ``AIAgent.close()`` → ``cleanup_vm`` at session teardown
    and by the idle reaper, not per-turn.
    """
    env = get_active_env(task_id)
    if env is None:
        return False
    if getattr(env, "_session_scoped", False):
        return True
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


def _cleanup_env(env, *, force_remove: bool = False) -> None:
    """Tear down one environment, passing ``force_remove`` only when accepted.

    ``DockerEnvironment.cleanup(force_remove=...)`` (issue #20561) diverges
    from the base ``cleanup(self)``; other backends expose ``stop`` /
    ``terminate`` instead. Shared by ``cleanup_vm`` and the prompt-time
    backend probe so the signature check lives in one place.
    """
    if hasattr(env, 'cleanup'):
        import inspect
        if "force_remove" in inspect.signature(env.cleanup).parameters:
            env.cleanup(force_remove=force_remove)
        else:
            env.cleanup()
    elif hasattr(env, 'stop'):
        env.stop()
    elif hasattr(env, 'terminate'):
        env.terminate()


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
        _cleanup_env(env, force_remove=force_remove)

        logger.info("Manually cleaned up environment for task: %s", task_id)

    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", task_id)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _atexit_cleanup():
    """Stop the cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    if _active_environments:
        logger.info("Shutting down %d remaining sandbox(es)...", len(_active_environments))
        # Snapshot BEFORE cleanup_all_environments empties the dict, then
        # block briefly so docker stop/rm completes before the interpreter
        # exits — otherwise daemon cleanup threads die mid-`docker stop` and
        # Exited containers pile up on the host.
        envs_to_wait = list(_active_environments.values())
        cleanup_all_environments()
        for env in envs_to_wait:
            wait_fn = getattr(env, "wait_for_cleanup", None)
            if wait_fn is not None:
                with _quiet("wait_for_cleanup raised on exit"):  # never block shutdown on a bad backend
                    wait_fn(timeout=15.0)
    # Workers of envs the idle reaper already detached are not in the registry (#86317).
    if "tools.environments.docker" in sys.modules:
        with _quiet("teardown drain raised on exit"):
            sys.modules["tools.environments.docker"].DockerEnvironment.wait_for_all_teardowns(timeout=15.0)

atexit.register(_atexit_cleanup)


def _command_requires_pipe_stdin(command: str) -> bool:
    """True when PTY mode would break a stdin-driven command: `gh auth login
    --with-token` waits for EOF on piped stdin, and under a PTY
    `process.submit()` only sends a newline, so it hangs forever."""
    normalized = " ".join(command.lower().split())
    return normalized.startswith("gh auth login") and "--with-token" in normalized


from tools.terminal_tool_guards import (
    _foreground_background_guidance, _safe_command_preview, _validate_workdir,
    gateway_lifecycle_block, self_repo_block,
)
from tools.terminal_tool_background import _YIELDED_NOTE, spawn_background_process, yield_to_background_handler
from tools.terminal_tool_result import finalize_foreground_result


def _resolve_notification_flag_conflict(*, notify_on_complete: bool, watch_patterns, background: bool) -> tuple:
    """Resolve notify_on_complete + watch_patterns both set: drop watch_patterns
    (combined they produce duplicate async notifications — one per match plus
    one on exit — that can spam the user long after the process ends).
    Returns ``(watch_patterns_to_use, conflict_note)``; note is "" without conflict."""
    if background and notify_on_complete and watch_patterns:
        return None, (
            "watch_patterns ignored because notify_on_complete=True; "
            "these two flags produce duplicate notifications when combined"
        )
    return watch_patterns, ""


def _resolve_command_cwd(
    *,
    workdir: Optional[str],
    default_cwd: str,
    session_key: Optional[str] = None,
    env_type: Optional[str] = None,
) -> str:
    """cwd for a command: explicit ``workdir`` > the session's own cwd record >
    ``default_cwd``.

    The record is written after every completed command of THIS session, so
    it is the session's ``cd`` state with no shared-env ambiguity. On
    container backends a recorded HOST path (a desktop/TUI surface registering
    its workspace) is unusable in the sandbox — ``cd <host path>`` fails with
    exit 126 — so it is discarded in favor of ``default_cwd``.

    Same guard class as the env-creation sanitizers (#50636, #54447); this is the per-command sibling site.
    """
    if workdir:
        return workdir
    recorded = get_session_cwd(session_key)
    if (
        recorded
        and _is_container_backend(env_type)
        and _is_unusable_container_cwd(recorded)
    ):
        logger.info(
            "Ignoring recorded session cwd %r for %s backend "
            "(host/relative path won't work in sandbox). Using %r instead.",
            recorded, env_type, default_cwd,
        )
        return default_cwd
    return recorded or default_cwd


def _error_json(error: str, *, exit_code: int = -1, status: Optional[str] = None, **extra) -> str:
    """The terminal error envelope: ``output``/``exit_code``/``error`` (+ ``status``, extras)."""
    body: Dict[str, Any] = {"output": "", "exit_code": exit_code, "error": error}
    if status is not None:
        body["status"] = status
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


def _fatal_error_json(e: BaseException) -> str:
    """Log the traceback and return the redacted error+traceback envelope.

    Exception text can embed the failing command line (and any secrets inline
    in it), so both fields are force-redacted before reaching the model.
    """
    import traceback
    tb_str = traceback.format_exc()
    logger.error("terminal_tool exception:\n%s", tb_str)
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "error": _redact_terminal_error_text(f"Failed to execute command: {e}"),
        "traceback": _redact_terminal_error_text(tb_str),
        "status": "error"
    }, ensure_ascii=False)


class _Rejected(Exception):
    """Carries a finished tool-result JSON out of the planning/guard helpers, so
    each early-return site is one ``raise`` instead of an isinstance-checked
    ``str | plan`` union at the caller."""

    def __init__(self, result_json: str):
        super().__init__(result_json)
        self.result_json = result_json


@dataclass
class _ApprovalVerdict:
    """Outcome of the pre-exec guard pass.

    ``note`` is the audit note attached to the result. ``approved_run`` is True
    when the user explicitly approved (or pre-confirmed via ``force``); it drives
    the clean-interrupt-slate clear before ``env.execute`` so an approved command
    can't be SIGINT-killed by a bit that landed during the approval-wait.
    """
    note: Optional[str] = None
    approved_run: bool = False


def _run_approval_guards(command: str, env_type: str, config: Dict[str, Any], *, force: bool) -> _ApprovalVerdict:
    """Run tirith + dangerous-command guards; ``force`` skips them entirely.
    Raises :class:`_Rejected` when the command may not run (denied, or pending
    gateway approval)."""
    if force:
        return _ApprovalVerdict(approved_run=True)
    approval = _check_all_guards(command, env_type, has_host_access=_docker_has_host_access(config))
    if not approval["approved"]:
        if approval.get("status") == "pending_approval":  # gateway ask mode
            raise _Rejected(_error_json(
                "", status="pending_approval",
                approval_pending=True,
                command=approval.get("command", command),
                description=approval.get("description", "command flagged"),
                pattern_key=approval.get("pattern_key", ""),
                smart_denied=approval.get("smart_denied", False),
                allow_permanent=approval.get("allow_permanent", True),
            ))
        desc = approval.get("description", "command flagged")
        fallback_msg = (
            f"Command denied: {desc}. "
            "Use the approval prompt to allow it, or rephrase the command."
        )
        raise _Rejected(_error_json(approval.get("message", fallback_msg), status="blocked",
                                    **({"user_summary": approval["user_summary"]} if approval.get("user_summary") else {})))
    desc = approval.get("description", "flagged as dangerous")
    if approval.get("user_approved"):
        return _ApprovalVerdict(
            note=f"Command required approval ({desc}) and was approved by the user.",
            approved_run=True,
        )
    if approval.get("smart_approved"):
        return _ApprovalVerdict(note=f"Command was flagged ({desc}) and auto-approved by smart approval.")
    return _ApprovalVerdict()


@dataclass
class _ExecPlan:
    """Per-call execution parameters resolved before any environment is touched."""
    config: Dict[str, Any]
    env_type: str
    effective_task_id: str
    image: str
    cwd: str
    host_cwd: Optional[str]
    effective_timeout: int
    # Set when a foreground call asked for more than FOREGROUND_MAX_TIMEOUT and was promoted to a
    # tracked background process instead of being refused (the requested seconds, for the note).
    promoted_from_foreground_timeout: Optional[int] = None


_PROMOTED_NOTE = (
    "Requested foreground timeout {requested}s exceeds the {cap}s cap, so this command was started as a "
    "tracked background process with notify_on_complete=true instead of being refused. Do NOT re-run it. "
    "Its completion (exit code + output tail) arrives as a notification; poll with "
    "process(action=\"poll\", session_id=...) if you need it sooner."
)


def _plan_execution(
    command: Any, *, task_id: Optional[str], timeout: Optional[int],
    background: bool, _host_local: bool,
) -> _ExecPlan:
    """Resolve backend, env-cache key, image, cwd and timeout for one call.

    Raises :class:`_Rejected` when the call is rejected up front (non-string
    command, non-positive or over-cap timeout, a foreground command that must
    run in the background).
    """
    if not isinstance(command, str):
        logger.warning("Rejected invalid terminal command value: %s", type(command).__name__)
        raise _Rejected(_error_json(
            f"Invalid command: expected string, got {type(command).__name__}", status="error",
        ))

    config = _get_env_config()
    env_type = "local" if _host_local else config["env_type"]

    # Fail closed under a refusal scope: the routed profile's terminal
    # policy could not be resolved, so running with the launch process's
    # ambient policy is forbidden.
    # See #68559.
    if not _host_local:
        from tools.terminal_scope import enforce_no_refusal

        enforce_no_refusal()

    effective_task_id = _resolve_container_task_id(task_id)
    if _host_local:
        # Control-plane children run beside this interpreter, never inside
        # the configured Docker/SSH backend; keep their env cache separate.
        effective_task_id = f"host-local-{effective_task_id}"

    # Per-task overrides (RL/benchmark envs, ACP workspace cwd) win over
    # the global env-var config; ``resolve_task_overrides`` reads the raw
    # task id first, then the collapsed container id.
    overrides = resolve_task_overrides(task_id)
    image = _select_image(env_type, overrides, config)

    cwd = overrides.get("cwd") or get_session_cwd(task_id) or config["cwd"]
    host_cwd = _resolve_task_host_cwd(config, task_id)
    # config["cwd"] was sanitized for container backends in _get_env_config
    # but an override / session record is raw: a host path would reach
    # `docker run -w` and fail with exit 125. Re-apply the guard to the
    # resolved cwd; when the host path IS this session's mounted workspace,
    # remap to /workspace instead of discarding it.
    if _is_container_backend(env_type) and _is_unusable_container_cwd(cwd):
        remapped = "/workspace" if host_cwd else config["cwd"]
        if cwd != remapped:
            logger.info(
                "Remapping host/relative cwd override %r for %s backend "
                "(won't exist in sandbox). Using %r instead.",
                cwd, env_type, remapped,
            )
        cwd = remapped
    # Reject non-positive timeouts before deadline math: ``timeout or
    # default`` would silently turn 0 into the default, and a negative
    # value is truthy and would fire an immediate "-Ns" timeout.
    if timeout is not None and timeout <= 0:
        raise _Rejected(tool_error(f"timeout must be a positive number of seconds (got {timeout})."))
    promoted = None
    if not background:
        # An over-cap foreground timeout is a bounded job the caller wants to wait for (test suites,
        # builds). Refusing it only bought a mechanical retry: 454 refusals in one run, every one
        # re-sent lower/split/background. Promote to a tracked background process instead; the
        # caller is told in the result. The `&`/nohup/server guidance below stays a refusal: those
        # need the command itself rewritten, which the tool cannot do safely.
        # The detachment guidance applies whether or not the call is promoted: a promoted `cmd &`
        # would start a tracked shell that exits at once while its payload runs untracked.
        guidance = _foreground_background_guidance(command)
        if guidance:
            raise _Rejected(_error_json(guidance, status="error"))
        if timeout and timeout > FOREGROUND_MAX_TIMEOUT:
            promoted = timeout

    return _ExecPlan(
        config=config, env_type=env_type, effective_task_id=effective_task_id,
        image=image, cwd=cwd, host_cwd=host_cwd, effective_timeout=timeout or config["timeout"],
        promoted_from_foreground_timeout=promoted,
    )


_PROMOTED_NOTE_POLL_ONLY = (
    "Requested foreground timeout {requested}s exceeds the {cap}s cap, so this command was started as a "
    "tracked background process instead of being refused. Do NOT re-run it. This session cannot receive "
    "completion notifications, so poll it with process(action=\"poll\", session_id=...) until it exits."
)


def _with_promoted_note(result_json: str, requested_timeout: int) -> str:
    """Attach the foreground->background promotion note to a spawn result (unchanged on error). The
    note only promises a notification when the spawn actually kept notify_on_complete (finite sessions
    such as one-shot runners cannot route one back; the spawn already said so and cleared the flag)."""
    try:
        data = json.loads(result_json)
    except (TypeError, ValueError):
        return result_json
    if not isinstance(data, dict) or data.get("error"):
        return result_json
    template = _PROMOTED_NOTE if data.get("notify_on_complete") else _PROMOTED_NOTE_POLL_ONLY
    data["promoted_from_foreground"] = template.format(requested=requested_timeout, cap=FOREGROUND_MAX_TIMEOUT)
    return json.dumps(data, ensure_ascii=False)


def _acquire_env(plan: _ExecPlan, task_id: Optional[str]) -> Any:
    """Cached env for the task, else create it under the per-task creation lock.

    Concurrent calls for the same task_id wait for the first sandbox instead
    of each creating their own; the cache is re-checked under that lock.
    Raises :class:`_Rejected` with the ``"disabled"`` envelope when creation
    raises ImportError.
    """
    _start_cleanup_thread()
    env_type, eff = plan.env_type, plan.effective_task_id

    with _env_lock:
        env: Any = _lookup_active_env(eff, task_id)
    if env is not None:
        return env

    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(eff, threading.Lock())

    with task_lock:
        with _env_lock:
            env = _lookup_active_env(eff, task_id)
        if env is not None:
            return env

        if env_type == "singularity":
            _check_disk_usage_warning()
        logger.info("Creating new %s environment for task %s...", env_type, eff[:8])
        try:
            new_env = _create_configured_env(
                plan.config, env_type, image=plan.image, cwd=plan.cwd,
                timeout=plan.effective_timeout, task_id=eff, host_cwd=plan.host_cwd,
                local_config=(
                    {"persistent": plan.config.get("local_persistent", False)}
                    if env_type == "local" else None
                ),
            )
        except ImportError as e:
            raise _Rejected(_error_json(
                _redact_terminal_error_text(f"Terminal tool disabled: environment creation failed ({e})"),
                status="disabled",
            ))

        with _env_lock:
            _active_environments[eff] = new_env
            _last_activity[eff] = time.time()
        logger.info("%s environment ready for task %s", env_type, eff[:8])
        return new_env


def _yield_kwargs(command: str, **ctx) -> dict:
    """``env.execute`` kwargs enabling yield-to-background (local backend only)."""
    handler = yield_to_background_handler(command=command, **ctx)
    return {"yield_handler": handler} if handler is not None else {}


def _run_foreground(
    command: str, env: Any, plan: _ExecPlan, *,
    task_id: Optional[str], session_id: Optional[str], session_key: str,
    workdir: Optional[str], approval_note: Optional[str], clear_interrupt: bool,
) -> str:
    """Execute in the foreground with retry on transient errors, then finalize."""
    max_retries = 3
    env_type, eff, effective_timeout = plan.env_type, plan.effective_task_id, plan.effective_timeout

    # Clean interrupt slate for an approved command, ONCE before the retry
    # loop: drop a stale bit that landed during the approval-wait so it
    # can't SIGINT the just-approved run. Do NOT re-clear inside the loop —
    # a genuine interrupt during the backoff sleep must survive and abort
    # the next attempt (rc 130).
    if clear_interrupt:
        from tools.interrupt import clear_current_thread_interrupt
        clear_current_thread_interrupt()

    for retry_count in range(max_retries + 1):
        try:
            command_cwd = _resolve_command_cwd(
                workdir=workdir, default_cwd=plan.cwd, session_key=session_key, env_type=env_type,
            )
            # bounded_capture: model-facing output keeps a head/tail window
            # while streaming so a verbose command can't OOM the gateway;
            # internal env.execute() consumers stay unbounded.
            result = env.execute(
                command, timeout=effective_timeout, cwd=command_cwd, bounded_capture=True,
                **_yield_kwargs(command, env_type=env_type, cwd=command_cwd, effective_task_id=eff,
                                task_id=task_id, session_key=session_key),
            )
            break
        except Exception as e:
            if "timeout" in str(e).lower():
                return _error_json(f"Command timed out after {effective_timeout} seconds", exit_code=124)
            # Retry on transient errors
            if retry_count < max_retries:
                wait_time = 2 ** (retry_count + 1)
                logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                               wait_time, retry_count + 1, max_retries, _safe_command_preview(command), type(e).__name__, e, eff, env_type)
                time.sleep(wait_time)
                continue
            logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                         max_retries, _safe_command_preview(command), type(e).__name__, e, eff, env_type)
            return _error_json(_redact_terminal_error_text(f"Command execution failed: {type(e).__name__}: {e}"))

    if result.get("yielded_session_id"):
        return json.dumps({
            "output": result.get("output", ""), "exit_code": None, "error": None,
            "status": "yielded_to_background", "session_id": result["yielded_session_id"],
            "pid": result.get("pid"), "notify_on_complete": True, "note": _YIELDED_NOTE,
        }, ensure_ascii=False)
    return finalize_foreground_result(
        command=command, result=result, env=env, env_type=env_type, effective_task_id=eff,
        task_id=task_id, session_id=session_id, session_key=session_key, workdir=workdir,
        command_cwd=command_cwd, approval_note=approval_note,
    )


def _pre_exec_block(
    command: str, *, env: Any, env_type: str, cwd: str,
    workdir: Optional[str], session_key: str,
) -> None:
    """Raise :class:`_Rejected` with the blocked-result JSON when the command must not run.

    Order matters: gateway lifecycle first (protects the running gateway),
    then the dangerous-workdir check, then the self-repo guard (local only).
    """
    blocked = gateway_lifecycle_block(
        command=command, env=env, env_type=env_type, cwd=cwd, workdir=workdir, session_key=session_key,
    )
    if blocked:
        raise _Rejected(blocked)
    if workdir:
        workdir_error = _validate_workdir(workdir)
        if workdir_error:
            logger.warning("Blocked dangerous workdir: %s (command: %s)",
                           workdir[:200], _safe_command_preview(command))
            raise _Rejected(_error_json(workdir_error, status="blocked"))
    if env_type == "local":
        blocked = self_repo_block(command=command, cwd=cwd, workdir=workdir, session_key=session_key)
        if blocked:
            raise _Rejected(blocked)


_PTY_DISABLED_REASON = (
    "PTY disabled for this command because it expects piped stdin/EOF "
    "(for example gh auth login --with-token). For local background "
    "processes, call process(action='close') after writing so it receives "
    "EOF."
)


def _degraded_result(e: EnvironmentConnectionError, task_id: Optional[str]) -> str:
    """Infrastructure failure (SSH host down, Docker daemon unreachable), distinct
    from a nonzero exit. ``terminal.degraded_mode``: warn (default) returns a
    structured degraded result with a retry hint; fail preserves the historical
    error+traceback result."""
    if _tenv("TERMINAL_DEGRADED_MODE", "warn").strip().lower() == "fail":
        return _fatal_error_json(e)
    logger.warning("terminal backend degraded: %s", e.reason)
    # Evict the possibly-broken backend so the next call re-creates it.
    with _quiet("degraded-env eviction failed"):
        _evict_environment_for_task(task_id)
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "status": "degraded",
        "reason": e.reason,
        "retry_hint": e.retry_hint,
        "error": f"Terminal backend degraded: {e.reason}",
    }, ensure_ascii=False)


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
    _host_local: bool = False,
) -> str:
    """Execute *command* in the configured terminal environment; returns a JSON string.

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
        watch_patterns: List of strings to watch for in background output. HARD rate limit: 1 notification per 15s per process. After 3 strike windows in a row — or after a small lifetime cap of delivered matches, however cleanly spaced — watch_patterns is disabled and the session is auto-promoted to notify_on_complete. Use ONLY for rare, one-shot mid-process signals on long-lived processes (server readiness, migration-done markers). NEVER use in loops/batch jobs — error patterns there will hit the strike limit and get disabled. MUTUALLY EXCLUSIVE with notify_on_complete — set one, not both.

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
        plan = _plan_execution(
            command, task_id=task_id, timeout=timeout, background=background, _host_local=_host_local,
        )
        env = _acquire_env(plan, task_id)
        env_type, cwd, effective_task_id = plan.env_type, plan.cwd, plan.effective_task_id

        # Get configuration
        config = _get_env_config()
        env_type = "local" if _host_local else config["env_type"]

        # Fail closed under a refusal scope (#68559): the routed profile's
        # terminal policy could not be resolved, so executing with the launch
        # process's ambient policy is forbidden — refuse with a typed,
        # model-actionable error instead.
        if not _host_local:
            from tools.terminal_scope import enforce_no_refusal

            enforce_no_refusal()

        # Use task_id for environment isolation. By default all subagent
        # task_ids collapse back to "default" so the top-level agent and
        # every delegate_task child share one container; only task_ids with
        # a registered env override (RL benchmarks) get isolated sandboxes.
        effective_task_id = _resolve_container_task_id(task_id)
        if _host_local:
            # Hermes-owned control-plane children must run beside the current
            # interpreter, never inside the model's configured Docker/SSH/etc.
            # Keep their environment cache separate from the configured backend.
            effective_task_id = f"host-local-{effective_task_id}"

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

        cwd = overrides.get("cwd") or get_session_cwd(task_id) or config["cwd"]
        # Session-scoped mount resolution (single owner: _resolve_task_host_cwd).
        # Under per-session isolation a fresh session must not inherit the
        # process-global TERMINAL_CWD mount left behind by a previous session.
        host_cwd = _resolve_task_host_cwd(config, task_id)
        # A per-task cwd override (registered by the gateway/TUI for workspace
        # tracking, or by RL/benchmark envs) wins over config["cwd"] — but
        # config["cwd"] was already sanitized for container backends in
        # _get_env_config() while the override is raw. On a container backend a
        # raw host path (e.g. a Windows desktop session's C:\Users\<user>, or a
        # POSIX /home/<user>) reaches `docker run -w <host-path>` and the
        # container fails to start (exit 125). Re-apply the same host/relative
        # path guard to the *resolved* cwd so the override can't bypass it.
        # When the host path IS this session's mounted workspace, remap it to
        # /workspace (where the mount lands) instead of discarding it.
        # Valid in-container override paths (RL/benchmark sandboxes that set
        # cwd to /workspace, /root, etc.) are absolute non-host paths and pass
        # through untouched.
        if _is_container_backend(env_type) and _is_unusable_container_cwd(cwd):
            remapped = "/workspace" if host_cwd else config["cwd"]
            if cwd != remapped:
                logger.info(
                    "Remapping host/relative cwd override %r for %s backend "
                    "(won't exist in sandbox). Using %r instead.",
                    cwd, env_type, remapped,
                )
            cwd = remapped
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
                        ssh_config = _ssh_config_from_config(config) if env_type == "ssh" else None
                        container_config = (
                            _container_config_from_config(config)
                            if _is_container_backend(env_type) else None
                        )

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
                            host_cwd=host_cwd,
                        )
                    except ImportError as e:
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": _redact_terminal_error_text(
                                f"Terminal tool disabled: environment creation failed ({e})"
                            ),
                            "status": "disabled"
                        }, ensure_ascii=False)

                    with _env_lock:
                        _active_environments[effective_task_id] = new_env
                        _last_activity[effective_task_id] = time.time()
                        env = new_env
                    logger.info("%s environment ready for task %s", env_type, effective_task_id[:8])

        assert env is not None  # all creation failure paths return above

        # The session key that drives cwd records: get_current_session_key()'s
        # contextvar doesn't cross tool-worker threads, so fall back to the raw
        # task_id (which IS the session_key for the top-level agent) — a
        # stable, thread-safe anchor.
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="") or (task_id or "")

        # Hard-block: gateway lifecycle commands (systemctl/launchctl/hermes
        # restart|stop|uninstall targeting hermes-gateway) must never run inside the
        # gateway process itself. The restart would SIGTERM the gateway, which
        # kills this very subprocess before it can complete — the service may
        # never restart. This mirrors the `hermes gateway restart` guard in
        # hermes_cli/gateway.py and the cron-path guard in hermes_cli/cron.py,
        # but applies unconditionally (force=True cannot help here).
        # Gate on the SUPERVISED-gateway probe, not the raw _HERMES_GATEWAY
        # marker: gateway.run sets it at import time, so it leaks into every
        # process that merely imports gateway.run (hermes serve --isolated,
        # CLI, web server) which are NOT the gateway and must be able to
        # restart it. A plain foreground `hermes gateway run` (env set, PID
        # owned, no supervisor) now also PASSES this guard: intentional and
        # harmless, since without a supervisor there is no KeepAlive to turn a
        # self-restart into a respawn loop.
        from tools.process_registry import _is_supervised_gateway_process

        if _is_supervised_gateway_process():
            from cron.lifecycle_guard import (
                _MAX_REFERENCED_SCRIPT_BYTES,
                contains_gateway_lifecycle_command_or_referenced_script,
                contains_launchctl_submit_command,
                lifecycle_scan_root_within_budget,
            )
            # Keep the specific launchctl diagnostic when this optional
            # pre-scan fits the budget.  The full fail-closed guard below still
            # runs when it does not, so oversized roots never reach shlex here.
            if (
                lifecycle_scan_root_within_budget(command)
                and contains_launchctl_submit_command(command)
            ):
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
            guard_cwd_base = get_session_cwd(session_key)
            if guard_cwd_base is None:
                guard_cwd_base = getattr(env, "cwd", None) or cwd
            guard_cwd = _resolve_command_cwd(
                workdir=workdir,
                default_cwd=guard_cwd_base,
                session_key=session_key,
                env_type=env_type,
            )

            def _read_script_in_env(script_path: str) -> Optional[str]:
                """Best-effort script read; uses env.execute only when local read fails.

                For local backends the script path is on the host filesystem. For
                SSH/Modal/Daytona the same path is remote; the local read misses, so we
                fall back to a bounded ``env.execute('head -c ... < path')`` read.
                """
                if env is None:
                    return None
                try:
                    local_path = Path(script_path).expanduser()
                    if not local_path.is_absolute():
                        local_path = Path(guard_cwd) / local_path
                    if local_path.is_file():
                        metadata = local_path.stat()
                        if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= _MAX_REFERENCED_SCRIPT_BYTES:
                            data = local_path.read_bytes()
                            if len(data) <= _MAX_REFERENCED_SCRIPT_BYTES:
                                if b"\x00" in data:
                                    # Binary (ELF/Mach-O/PE), not a shell script:
                                    # feeding its decoded bytes back into the guard
                                    # tokenizes machine code into bogus NUL-bearing
                                    # paths and crashes the scanner (#77703). Mirror
                                    # lifecycle_guard._read_referenced_script and
                                    # treat it as nothing to scan.
                                    return None
                                return data.decode("utf-8", errors="replace")
                except Exception:
                    pass
                # Remote / sandboxed backend: read via the environment's shell.
                # Bound the read at the source with `head -c` so an oversized
                # file (e.g. a 166MB ELF invoked by absolute path) never
                # crosses the wire — `cat` of such a binary previously pinned
                # the gateway's tool thread on a superlinear shlex scan for
                # 30+ minutes. One byte over the guard's budget is enough for
                # lifecycle_guard's sanitizer to fail the oversized case
                # closed, mirroring the local-read semantics. The `< path`
                # redirect keeps leading-dash paths out of argv (same form as
                # tools/image_source.py).
                try:
                    result = env.execute(
                        f"head -c {_MAX_REFERENCED_SCRIPT_BYTES + 1} "
                        f"< {shlex.quote(script_path)}"
                    )
                    if result.get("returncode", -1) == 0:
                        output = result.get("output", "")
                        if output and "\x00" in output:
                            # Binary content from a remote read: skip for the
                            # same reason as the local branch above (#77703).
                            return None
                        return output
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
                        "Blocked: command or referenced script cannot restart, stop, or "
                        "uninstall the gateway from inside the gateway process. The gateway would "
                        "kill this command before it could complete (SIGTERM propagates "
                        "to child processes). Run `hermes gateway restart` from a "
                        "separate shell outside the running gateway."
                    ),
                    "status": "error",
                }, ensure_ascii=False)

        # Validate before the source guard resolves an explicit workdir.
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

        # Windows-only: NTFS locks loaded module files, so rewriting the local
        # checkout backing this interpreter can corrupt the running process.
        # POSIX keeps old inodes alive for open handles, so the guard is off
        # there. Remote backends cannot reach that checkout.
        if env_type == "local":
            from tools.self_repo_guard import (
                detect_self_repo_git_mutation,
                guard_active,
            )

            guard_cwd = _resolve_command_cwd(
                workdir=workdir,
                default_cwd=cwd,
                session_key=session_key,
            )
            _self_repo_hit, _self_repo_msg = (
                detect_self_repo_git_mutation(command, guard_cwd)
                if guard_active()
                else (False, None)
            )
            if _self_repo_hit:
                logger.warning(
                    "Blocked self-repo git mutation (command: %s)",
                    _safe_command_preview(command),
                )
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": _self_repo_msg,
                    "status": "blocked",
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
                        "smart_denied": approval.get("smart_denied", False),
                        "allow_permanent": approval.get("allow_permanent", True),
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
            result = spawn_background_process(
                command=command, env=env, env_type=env_type, effective_task_id=effective_task_id,
                task_id=task_id, session_key=session_key, workdir=workdir, cwd=cwd,
                effective_pty=pty and not pty_disabled, notify_on_complete=notify_on_complete,
                watch_patterns=watch_patterns, approval_note=verdict.note,
                pty_disabled_reason=_PTY_DISABLED_REASON if pty_disabled else None,
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
                            # Stamp the spawning conversation's session-db id
                            # so the gateway's completion pre-flight
                            # (_classify_completion_target) can drop the
                            # notification when the user closes this session
                            # (/new) before the process finishes, instead of
                            # injecting it into the chat's NEW session.
                            proc_session.parent_session_id = _gse(
                                "HERMES_SESSION_ID", ""
                            )

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
                            "parent_session_id": proc_session.parent_session_id,
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
                    "error": _redact_terminal_error_text(
                        f"Failed to start background process: {e}"
                    )
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
                        env_type=env_type,
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
                        "error": _redact_terminal_error_text(
                            f"Command execution failed: {type(e).__name__}: {e}"
                        )
                    }, ensure_ascii=False)
                
                # Got a result
                break

            # Dual-write (cwd rearch step 1): the env's post-command tracking
            # (marker parse / local sync) has just updated env.cwd with the
            # directory this command finished in. That cwd belongs to THIS
            # session — record it under the session key so the durable record
            # never depends on the shared env surviving or on who drives the
            # env next.
            #
            # BUT: a per-command ``workdir`` override is transient by contract
            # (docstring: "Working directory for this command"). Recording it
            # would hijack the session's durable cwd for every later command
            # that doesn't pass ``workdir``. Skip the dual-write in that case.
            #
            # AND only when the command actually reported its cwd. The marker
            # is printed after the command returns, so an interrupted / killed
            # / timed-out command emits none and env.cwd still holds whatever
            # the last command to FINISH left there — on a shared env, that is
            # another session's directory. Recording it silently re-homes this
            # session into a directory the user never opened.
            if not workdir and (result or {}).get("cwd_observed"):
                record_session_cwd(session_key, getattr(env, "cwd", None))

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
                command, output
            )
            if sudo_cache_cleared:
                has_sudo_prompt_callback = _get_sudo_password_callback() is not None
                can_reprompt = (
                    has_sudo_prompt_callback or env_var_enabled("HERMES_INTERACTIVE")
                ) and not _in_delegated_child_context()
                if can_reprompt:
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
            elif returncode == 0:
                # Masked-success backstop: `cargo build | tail -20` returns
                # tail's exit 0 even when the build failed (bash reports the
                # last pipeline command's status; same for `cmd || echo ...`).
                # When the command shape can mask an upstream failure AND the
                # output carries strong failure indicators, warn the model so
                # exit_code 0 isn't read as a success signal. Advisory only —
                # the exit code itself is never modified.
                try:
                    from tools.terminal_hints import annotate_masked_success
                    failure_hint = annotate_masked_success(command, output)
                except Exception:
                    failure_hint = None

            result_dict = {
                "output": output,
                "exit_code": returncode,
                "error": None,
            }
            # cwd echo: when the command changed the session's working
            # directory (cd, pushd, ...), tell the model where it ended up.
            # Production mining shows 60% of terminal calls carry a
            # defensive 'cd X && ' prefix because the model can't see cwd
            # state; echoing it on change removes the guesswork (pattern
            # borrowed from crush's <cwd> injection).
            #
            # Gated on the same observation flag as the record above: without
            # it, an interrupted command echoes the shared env's leftover cwd
            # and tells the model it moved to a directory another session
            # opened.
            try:
                post_cwd = getattr(env, "cwd", None) if (result or {}).get("cwd_observed") else None
                if post_cwd and command_cwd and os.path.realpath(str(post_cwd)) != os.path.realpath(str(command_cwd)):
                    result_dict["cwd"] = str(post_cwd)
            except Exception:
                pass
            # Truncation metadata (codex/opencode/goose pattern): report the
            # pre-truncation size and a spill-file handle so the model can
            # retrieve the omitted middle with read_file/search_files instead
            # of re-running the command. The spill was written raw by the
            # collector; redact it here with the same pass as the visible
            # output so no secret persists unmasked on disk.
            if spill_file_path:
                try:
                    _sp = Path(spill_file_path)
                    raw_spill = _sp.read_text(encoding="utf-8", errors="replace")
                    from tools.spill_safety import write_text_exclusive

                    # Rewrite in place via lstat-checked unlink + exclusive
                    # create so the redacted copy can't be diverted through a
                    # symlink planted between the collector's write and now.
                    write_text_exclusive(
                        _sp,
                        redact_terminal_output(strip_ansi(raw_spill), command),
                        private=True,
                        overwrite=True,
                        errors="replace",
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
            if failure_hint:
                result_dict["hint"] = failure_hint
            if sudo_auth_failed:
                result_dict["sudo_auth_failed"] = True
            if sudo_cache_cleared:
                result_dict["sudo_cache_cleared"] = True

            return json.dumps(result_dict, ensure_ascii=False)

    except EnvironmentConnectionError as e:
        # Infrastructure/connection-class failure (SSH host down, Docker
        # daemon unreachable) — distinct from a command failing with a
        # nonzero exit code.  Config gate ``terminal.degraded_mode``:
        #   warn (default) — return a structured degraded result the model
        #                    can act on (reason + retry hint, no traceback).
        #   fail           — preserve the historical error+traceback result.
        degraded_mode = _tenv("TERMINAL_DEGRADED_MODE", "warn").strip().lower()
        if degraded_mode == "fail":
            import traceback
            tb_str = traceback.format_exc()
            logger.error("terminal_tool exception:\n%s", tb_str)
            # Exception text can embed the failing command line (and any
            # secrets inline in it) — redact before returning to the model.
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": _redact_terminal_error_text(f"Failed to execute command: {e}"),
                "traceback": _redact_terminal_error_text(tb_str),
                "status": "error"
            }, ensure_ascii=False)

        logger.warning("terminal backend degraded: %s", e.reason)
        # Never keep a possibly-broken backend cached: evict it so the next
        # call re-creates the environment from scratch and simply works once
        # the backend is reachable again.
        try:
            _evict_environment_for_task(task_id)
        except Exception:
            logger.debug("degraded-env eviction failed", exc_info=True)
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "status": "degraded",
            "reason": e.reason,
            "retry_hint": e.retry_hint,
            "error": f"Terminal backend degraded: {e.reason}",
        }, ensure_ascii=False)

    except Exception as e:
        return _fatal_error_json(e)


def check_terminal_requirements() -> bool:
    """Check if all requirements for the terminal tool are met. The reason for a failure is kept for
    :func:`terminal_backend_unavailable_reason` (CLI startup notice / doctor)."""
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

        elif env_type == "vercel_sandbox":
            return _check_vercel_sandbox_requirements(config)

        elif env_type == "daytona":
            from daytona import Daytona  # noqa: F401 — SDK presence check
            from agent.secret_scope import get_secret
            return get_secret("DAYTONA_API_KEY") is not None

        else:
            provider = _get_plugin_env_provider(env_type)
            if provider is not None:
                return bool(provider.check_requirements(config))
            logger.error(
                "Unknown TERMINAL_ENV '%s'. Use one of: local, docker, singularity, "
                "modal, daytona, vercel_sandbox, ssh, or a plugin-registered backend.",
                env_type,
            )
            return False
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e, exc_info=True)
        _record_unavailable_reason(f"the requirements check failed: {e}")
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
        f"{_tenv('TERMINAL_ENV', 'local')} "
        "(local/docker/singularity/modal/daytona/vercel_sandbox/ssh)"
    )
    print(f"  TERMINAL_DOCKER_IMAGE: {_tenv('TERMINAL_DOCKER_IMAGE', default_img)}")
    print(f"  TERMINAL_SINGULARITY_IMAGE: {_tenv('TERMINAL_SINGULARITY_IMAGE', f'docker://{default_img}')}")
    print(f"  TERMINAL_MODAL_IMAGE: {_tenv('TERMINAL_MODAL_IMAGE', default_img)}")
    print(f"  TERMINAL_DAYTONA_IMAGE: {_tenv('TERMINAL_DAYTONA_IMAGE', default_img)}")
    print(f"  TERMINAL_CWD: {_tenv('TERMINAL_CWD', _safe_getcwd())}")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  TERMINAL_SANDBOX_DIR: {_tenv('TERMINAL_SANDBOX_DIR', f'{_dhh()}/sandboxes')}")
    print(f"  TERMINAL_TIMEOUT: {_tenv('TERMINAL_TIMEOUT', '60')}")
    print(f"  TERMINAL_LIFETIME_SECONDS: {_tenv('TERMINAL_LIFETIME_SECONDS', '300')}")


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
                "description": "The shell command to execute"
            },
            "background": {
                "type": "boolean",
                "description": "Run in the background, returning a session_id. Pair with notify=true for anything with a defined end (tests, builds, deploys) — without it the process runs silently. Only servers/watchers/daemons that never exit should stay silent. Short commands: prefer foreground with a generous timeout.",
                "default": False
            },
            "timeout": {
                "type": "integer",
                "description": f"Max seconds to wait (default: 180, foreground max: {FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command finishes — set high for long tasks, you won't wait unnecessarily. A foreground timeout above {FOREGROUND_MAX_TIMEOUT}s runs the command as a tracked background process with notify_on_complete=true instead (the result says so; do not re-run it).",
                "minimum": 1
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for this command (absolute path). Defaults to the session working directory."
            },
            "pty": {
                "type": "boolean",
                "description": "With background=true: run in a pseudo-terminal for interactive CLI tools (Codex, Claude Code, Python REPL). Local backend only. Default: false.",
                "default": False
            },
            "notify": {
                "description": "With background=true: notify=true fires exactly one notification when the process exits (the right choice for nearly every bounded task — builds, tests, deploys). notify=['pattern', ...] instead notifies when a line matches a pattern — ONLY for one-shot readiness signals on processes that never exit (e.g. ['Application startup complete']); rate-limited and auto-disabled if it over-fires. Omit for silent daemons.",
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "string"}}
                ]
            }
            # Legacy aliases (unadvertised, still accepted): notify_on_complete
            # (bool) and watch_patterns (list). notify=true|[...] maps onto
            # them in the dispatch wrapper; explicit notify wins on conflict.
        },
        "required": ["command"]
    }
}


def _handle_terminal(args, **kw):
    # Models sometimes send execute_code's ``code`` here; name the stray
    # argument and the right tool instead of failing on command=None.
    if "command" not in args and "code" in args:
        return tool_error(
            "terminal received a 'code' parameter, but it requires a shell "
            "command in 'command'. Use execute_code(code=...) for Python; "
            "for shell, retry as terminal(command=...)."
        )
    # `notify` is the advertised interface: true → notify_on_complete,
    # ['pat', ...] → watch_patterns. The legacy args remain accepted
    # (old transcripts, internal callers); explicit `notify` wins.
    notify = args.get("notify")
    notify_on_complete = args.get("notify_on_complete", False)
    watch_patterns = args.get("watch_patterns")
    # Background-only modifiers on a foreground call were silently ignored;
    # fail with the corrected call instead (poka-yoke, no schema cost).
    if not args.get("background", False):
        if notify or watch_patterns or notify_on_complete:
            return tool_error(
                "notify only applies to background commands (foreground "
                "results return directly). Either drop notify, or run as "
                "terminal(command=..., background=true, notify=...)."
            )
        if args.get("pty", False):
            return tool_error(
                "pty requires background=true (a PTY session is interacted "
                "with via process(action='write'/'submit'), which needs a "
                "tracked background process). Retry as terminal(command=..., "
                "background=true, pty=true)."
            )
    if notify is not None:
        if isinstance(notify, bool):
            notify_on_complete = notify
            watch_patterns = None
        elif isinstance(notify, list):
            watch_patterns = notify
            notify_on_complete = False
        else:
            return tool_error(
                "notify must be true/false (notify on exit) or a list of "
                "strings (notify on output pattern match)."
            )
    return terminal_tool(
        command=args.get("command"),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
        workdir=args.get("workdir"),
        pty=args.get("pty", False),
        notify_on_complete=notify_on_complete,
        watch_patterns=watch_patterns,
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
import importlib.util  # noqa: F401,E402
import platform  # noqa: F401,E402
import re  # noqa: F401,E402
import shlex  # noqa: F401,E402
import shutil  # noqa: F401,E402
import stat  # noqa: F401,E402
import subprocess  # noqa: F401,E402
import sys  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'cleanup_vm': ('tools.terminal_tool_lifecycle', 'cleanup_vm'),
    'env_var_enabled': ('utils', 'env_var_enabled'),
    'get_active_env': ('tools.terminal_tool_lifecycle', 'get_active_env'),
    'has_direct_modal_credentials': ('tools.tool_backend_helpers', 'has_direct_modal_credentials'),
    'is_interrupted': ('tools.interrupt', 'is_interrupted'),
    'is_managed_tool_gateway_ready': ('tools.managed_tool_gateway', 'is_managed_tool_gateway_ready'),
    'is_persistent_env': ('tools.terminal_tool_lifecycle', 'is_persistent_env'),
    'nous_tool_gateway_unavailable_message': ('tools.tool_backend_helpers', 'nous_tool_gateway_unavailable_message'),
    'resolve_modal_backend_state': ('tools.tool_backend_helpers', 'resolve_modal_backend_state'),
    'strip_inert_heredoc_bodies': ('tools.shell_heredoc', 'strip_inert_heredoc_bodies'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
