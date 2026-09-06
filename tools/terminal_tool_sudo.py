"""Sudo password plumbing and shell-command rewrites for the terminal tool: per-scope
password cache, /dev/tty prompt, the quote-aware shell scanner behind the real-sudo rewrite
(``sudo -S -p ''``) and the compound-background brace-group rewrite, and the NOPASSWD probe.
Split out of ``tools/terminal_tool.py``; every public/patched name is re-imported there so
``tools.terminal_tool.<name>`` keeps resolving (and monkeypatching) as before."""

import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

from utils import env_var_enabled

# Log-record parity with the origin module.
logger = logging.getLogger("tools.terminal_tool")

# Interactive sudo password cache, scoped to the session key when present, else callback
# identity (ACP / CLI), else the current thread — so one session can never reuse another's
# cached password in a long-lived process.
_sudo_password_cache: dict[str, str] = {}
_sudo_password_cache_lock = threading.Lock()


def _get_sudo_password_cache_scope() -> str:
    """Return the cache scope for interactive sudo passwords."""
    from tools.terminal_tool import _current_session_key, _get_sudo_password_callback
    session_key = _current_session_key()
    if session_key:
        return f"session:{session_key}"
    callback = _get_sudo_password_callback()
    if callback is None:
        return f"thread:{threading.get_ident()}"
    owner = getattr(callback, "__self__", None)
    func = getattr(callback, "__func__", None)
    if owner is not None and func is not None:
        return f"callback-owner:{id(owner)}:{id(func)}"
    return f"callback:{id(callback)}"


def _get_cached_sudo_password() -> str:
    """Return the cached sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        return _sudo_password_cache.get(scope, "")


def _set_cached_sudo_password(password: str) -> None:
    """Persist a sudo password for the current scope ("" drops the entry)."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        if password:
            _sudo_password_cache[scope] = password
        else:
            _sudo_password_cache.pop(scope, None)


def _reset_cached_sudo_passwords() -> None:
    """Clear all cached sudo passwords (tests / process teardown)."""
    with _sudo_password_cache_lock:
        _sudo_password_cache.clear()


def _in_delegated_child_context() -> bool:
    """True while running inside a delegate_task child. Subagents run on parent-process worker
    threads and inherit process-wide ``HERMES_INTERACTIVE=1``, which does NOT mean they can reach
    the user: a raw ``/dev/tty`` sudo prompt from a child races the TUI for the tty and blocks the
    child for the full timeout, so children are always headless for sudo prompting. The ContextVar
    is set by ``delegated_child_context()`` and propagates via ``copy_context``."""
    try:
        from agent.delegation_context import is_delegated_child_context
        return is_delegated_child_context()
    except Exception:
        return False


_SUDO_HEADLESS_FAILURES = ("sudo: a password is required", "sudo: no tty present", "sudo: a terminal is required")


def _handle_sudo_failure(output: str, env_type: str) -> str:
    """Append a SUDO_PASSWORD tip when sudo failed in a headless context
    (gateway session or delegate_task child); otherwise return *output* as is."""
    is_gateway = env_var_enabled("HERMES_GATEWAY_SESSION")
    is_delegated_child = _in_delegated_child_context()
    if not (is_gateway or is_delegated_child) or not any(f in output for f in _SUDO_HEADLESS_FAILURES):
        return output
    from hermes_constants import display_hermes_home as _dhh
    if is_delegated_child:
        return output + (
            "\n\n💡 Tip: Subagents cannot prompt for a sudo password. "
            f"Add SUDO_PASSWORD to {_dhh()}/.env on the agent machine, "
            "or run the command without sudo."
        )
    return output + f"\n\n💡 Tip: To enable sudo over messaging, add SUDO_PASSWORD to {_dhh()}/.env on the agent machine."


# sudo -S rejects a bad cached/interactive password with these messages.
_SUDO_WRONG_PASSWORD_MARKERS = (
    "sudo: authentication failed",
    "sudo: incorrect password attempt",
    "sudo: maximum 3 incorrect authentication attempts",
    "sudo: 3 incorrect password attempts",
)


def _sudo_wrong_password_failure(output: str) -> bool:
    """Return True when sudo rejected a piped password."""
    lowered = (output or "").lower()
    return any(marker in lowered for marker in _SUDO_WRONG_PASSWORD_MARKERS)


def _invalidate_cached_sudo_on_auth_failure(command: str | None, output: str) -> bool:
    """Drop a session-cached sudo password after sudo rejects it. Env-configured
    ``SUDO_PASSWORD`` is left alone — an explicit operator choice, not a cache entry."""
    if (
        "SUDO_PASSWORD" in os.environ
        or not _sudo_wrong_password_failure(output)
        or _count_real_sudo_invocations(command or "") == 0
        or not _get_cached_sudo_password()
    ):
        return False
    _set_cached_sudo_password("")
    return True


def _release_tty(tty_fd, old_attrs) -> None:
    """Restore echo and close the /dev/tty fd opened by the password reader (best effort)."""
    if tty_fd is None:
        return
    if old_attrs is not None:
        try:
            import termios as _termios
            _termios.tcsetattr(tty_fd, _termios.TCSAFLUSH, old_attrs)
        except Exception as e:
            logger.debug("Failed to restore terminal attributes: %s", e)
    try:
        os.close(tty_fd)
    except Exception as e:
        logger.debug("Failed to close tty fd: %s", e)


def _read_hidden_password(result: dict) -> None:
    """Read one line with echo disabled into ``result``. Uses msvcrt on Windows, /dev/tty on Unix."""
    tty_fd = old_attrs = None
    try:
        chars = []
        if platform.system() == "Windows":
            import msvcrt
            while (c := msvcrt.getwch()) not in {"\r", "\n"}:
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
            while (b := os.read(tty_fd, 1)) and b not in {b"\n", b"\r"}:
                chars.append(b)
            result["password"] = b"".join(chars).decode("utf-8", errors="replace")
    except (KeyboardInterrupt, Exception):
        result["password"] = ""
    finally:
        _release_tty(tty_fd, old_attrs)
        result["done"] = True


def _prompt_for_sudo_password(timeout_seconds: int = 45) -> str:
    """Prompt for a sudo password; "" on skip (empty Enter), timeout, or error. Prefers the
    CLI-registered callback (prompt_toolkit-integrated); otherwise reads /dev/tty (msvcrt on
    Windows) with echo disabled. Human wait time is excluded from tool deadlines (``human_wait_window``)."""
    from tools.terminal_tool import _get_sudo_password_callback
    _sudo_cb = _get_sudo_password_callback()
    if _sudo_cb is not None:
        try:
            from tools.approval_human_wait import human_wait_window
            with human_wait_window():
                return _sudo_cb() or ""
        except Exception:
            return ""

    result = {"password": None, "done": False}
    try:
        os.environ["HERMES_SPINNER_PAUSE"] = "1"
        time.sleep(0.2)
        print("\n".join((
            "",
            "┌" + "─" * 58 + "┐",
            "│  🔐 SUDO PASSWORD REQUIRED" + " " * 30 + "│",
            "├" + "─" * 58 + "┤",
            "│  Enter password below (input is hidden), or:            │",
            "│    • Press Enter to skip (command fails gracefully)     │",
            f"│    • Wait {timeout_seconds}s to auto-skip" + " " * 27 + "│",
            "└" + "─" * 58 + "┘",
            "",
        )))
        print("  Password (hidden): ", end="", flush=True)
        password_thread = threading.Thread(target=_read_hidden_password, args=(result,), daemon=True)
        password_thread.start()
        from tools.approval_human_wait import human_wait_window
        with human_wait_window():
            password_thread.join(timeout=timeout_seconds)
        if not result["done"]:
            print("\n  ⏱ Timeout - continuing without sudo\n    (Press Enter to dismiss)\n")
            sys.stdout.flush()
            return ""
        password = result["password"] or ""
        # Newline after the hidden input, then the outcome line.
        if password:
            print("\n  ✓ Password received (cached for this session)\n")
        else:
            print("\n  ⏭ Skipped - continuing without sudo\n")
        sys.stdout.flush()
        return password
    except (EOFError, KeyboardInterrupt):
        print("\n  ⏭ Cancelled - continuing without sudo\n")
        sys.stdout.flush()
        return ""
    except Exception as e:
        print(f"\n  [sudo prompt error: {e}] - continuing without sudo\n")
        sys.stdout.flush()
        return ""
    finally:
        os.environ.pop("HERMES_SPINNER_PAUSE", None)


def _looks_like_env_assignment(token: str) -> bool:
    """Return True when *token* is a leading shell environment assignment."""
    if "=" not in token or token.startswith("="):
        return False
    name, _value = token.split("=", 1)
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


# One shell word: single-/double-quoted runs (unterminated ok; backslash escapes inside double
# quotes), a backslash escape (a trailing lone `\` is a plain char), or any non-`;|&()`/space char.
_SHELL_WORD_RE = re.compile(r"""(?:'[^']*'?|"(?:\\.|[^"])*"?|\\.|[^\s;|&()])*""", re.DOTALL)


def _read_shell_token(command: str, start: int) -> tuple[str, int]:
    """Read one shell token, preserving quotes/escapes, starting at *start*."""
    end = _SHELL_WORD_RE.match(command, start).end()  # type: ignore[union-attr]  (`*` always matches)
    return command[start:end], end


def _scan_shell(command: str, background: bool = False) -> Iterator[tuple[str, int, int, bool]]:
    """Quote-aware shell scanner shared by the sudo and compound-background rewriters.

    Yields ``(kind, start, end, at_command_start)`` events that tile *command* exactly:
    ``ws`` (one whitespace char), ``comment`` (``#`` up to, not including, the newline),
    ``op`` (``&& || ;; ; | & ( )``), ``word`` (one ``_read_shell_token`` token). Comments open
    only at command start (after newline, an operator, or a leading ``VAR=val``) in sudo mode.
    Background mode (compound-background semantics) additionally opens comments anywhere
    outside a token, emits ``escape`` for a bare ``\\x``, ops ``&>``, ``{ `` and a closing
    ``}``, and tracks ``(...)``/``{ ... }`` depth: inside a group nothing is an operator, so
    every non-structural char (whitespace included) surfaces as a single ``skip`` event.
    """
    i, n = 0, len(command)
    at_start = True
    parens = braces = 0
    two_char_ops = ("&&", "||", "&>") if background else ("&&", "||", ";;")
    while i < n:
        ch = command[i]
        grouped = parens or braces
        was_start = at_start
        if ch.isspace():
            kind, end = ("skip" if grouped else "ws"), i + 1
            at_start = at_start or ch == "\n"
        elif ch == "#" and (background or at_start):
            end = command.find("\n", i)
            kind, end = "comment", (n if end == -1 else end)
        elif background and ch == "\\" and i + 1 < n:
            kind, end = "escape", i + 2
        elif background and ch in "'\"":
            kind, end = "word", _read_shell_token(command, i)[1]
        elif background and (
            ch in "()" or (ch == "{" and i + 1 < n and command[i + 1].isspace()) or (ch == "}" and braces)
        ):
            parens = max(0, parens + (ch == "(") - (ch == ")"))
            braces += (ch == "{") - (ch == "}")
            kind, end = "op", i + 1
        elif background and grouped:
            kind, end = "skip", i + 1
        elif command.startswith(two_char_ops, i):
            kind, end, at_start = "op", i + 2, True
        elif ch in ";|&()":
            kind, end, at_start = "op", i + 1, ch != ")"
        else:
            token, end = _read_shell_token(command, i)
            kind, at_start = "word", bool(at_start and _looks_like_env_assignment(token))
        yield kind, i, end, was_start
        i = end


def _rewrite_real_sudo_invocations(command: str) -> tuple[str, int]:
    """Rewrite only real unquoted sudo command words (at a command-start position, see
    ``_scan_shell``), not plain text mentions; comments are copied through verbatim.
    Returns the rewritten command and the number of sudo invocations rewritten."""
    out: list[str] = []
    sudo_count = 0
    for kind, start, end, at_start in _scan_shell(command):
        text = command[start:end]
        if kind == "word" and at_start and text == "sudo":
            text = "sudo -S -p ''"
            sudo_count += 1
        out.append(text)
    return "".join(out), sudo_count


def _count_real_sudo_invocations(command: str) -> int:
    """Return how many real sudo command words appear in *command*."""
    return _rewrite_real_sudo_invocations(command)[1]


def _sudo_nopasswd_works() -> bool:
    """True when local sudo currently works without prompting. Local backend only — Docker/SSH/
    Modal must not inherit host sudo state. Re-probes every call (no cache) so an expired sudo
    timestamp can't make a later command silently block waiting for a password."""
    from tools.terminal_tool import _tenv
    if (_tenv("TERMINAL_ENV", "local").strip().lower() or "local") != "local":
        return False
    try:
        probe = subprocess.run(
            ["sudo", "-n", "true"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=3, check=False,
        )
        return probe.returncode == 0
    except Exception:
        return False


def _rewrite_compound_background(command: str) -> str:
    """Wrap `A && B &` (or `A || B &`) to `A && { B & }` at depth 0. Bash binds `&&` tighter
    than `&`, so `A && B &` backgrounds a subshell that runs B in the foreground and waits for
    it; a long-running B leaves that subshell stuck in ``wait4`` forever, and its open stdout
    pipe can keep the terminal tool from returning. The brace group keeps `&&`'s
    skip-B-on-failure semantics without a fork: bash backgrounds B as a simple command and
    exits immediately, orphaning B normally. Redirects (``&>``, ``2>&1``), quoted strings,
    comments and ``(...)``/``{ ... }`` bodies never count as the backgrounding ``&`` (see
    ``_scan_shell``); tracking brace depth also makes the rewrite idempotent. `(...)` subshells
    have the same bug class but are not the common agent pattern; left for a follow-up.
    Simple ``cmd &`` is left alone — it doesn't have the subshell-wait bug."""
    chain_end = -1  # just after the last depth-0 `&&`/`||` of this statement; -1 = none active
    rewrites: list[tuple[int, int]] = []  # (chain_op_end, amp_pos)
    for kind, start, end, _ in _scan_shell(command, background=True):
        text = command[start:end]
        if kind == "op" and text in ("&&", "||"):
            chain_end = end
        elif kind == "ws" and text == "\n" or kind == "op" and text in (";", "|", "}"):
            # Newline / `;` end a statement, `|` starts a pipeline stage, `}` closes a group.
            chain_end = -1
        elif kind == "op" and text == "&":
            # `&&` and `&>` never reach here; a `>&` / `<&` fd target (look back past
            # whitespace) is a redirect, anything else is the real background operator.
            j = start - 1
            while j >= 0 and command[j].isspace():
                j -= 1
            if j >= 0 and command[j] in "<>":
                continue
            if chain_end >= 0:
                rewrites.append((chain_end, start))
            chain_end = -1

    # Apply rewrites back-to-front so earlier indices remain valid.
    result = command
    for chain_end, amp_pos in reversed(rewrites):
        # Skip whitespace right after the `&&`/`||` so the brace group opens flush against
        # the inner command. `{` needs a trailing space in bash; the closing `}` needs to be
        # preceded by `;` or `&` — we're providing `&` from the backgrounding.
        insert_pos = chain_end
        while insert_pos < amp_pos and result[insert_pos].isspace():
            insert_pos += 1
        # The consumed `&` also separated the compound from any statement that followed
        # on the same line (`A && B & C`); `{ B & } C` is a syntax error, so restore a `;`
        # when the suffix resumes with command text. No separator when the suffix already
        # starts with a terminator (`;` `&` `|` newline `)` `}`) — except `&>`, which is a
        # redirect prefix for the NEXT command, not a terminator. Strip only spaces/tabs:
        # a newline already terminates the group.
        suffix = result[amp_pos + 1 :]
        tail = suffix.lstrip(" \t")
        needs_separator = bool(tail) and (tail[0] not in ";\n&|)}" or tail.startswith("&>"))
        separator = " ;" if needs_separator else ""
        result = result[:insert_pos] + "{ " + result[insert_pos:amp_pos] + "& }" + separator + suffix
    return result


def _transform_sudo_command(command: str | None) -> tuple[str | None, str | None]:
    """Rewrite bare ``sudo`` to ``sudo -S -p ''`` when a password is available (shared by every
    execution environment). Returns ``(command, sudo_stdin)``: ``sudo_stdin`` is one password
    line per sudo invocation that the caller must PREPEND to the process stdin (sudo -S consumes
    exactly one line and passes the rest through, so it's safe alongside the caller's own
    stdin_data). Backends that can't pipe stdin (modal, daytona, vercel_sandbox) embed the
    password in the command string themselves. With no password available the command is
    returned unchanged and ``sudo_stdin`` is None, so it fails gracefully with "sudo: a password
    is required". Password sources, in order: configured SUDO_PASSWORD, the session cache, then
    an interactive prompt (45s timeout, cached on success) when a UI is reachable."""
    from tools.terminal_tool import _get_sudo_password_callback
    if command is None:
        return None, None
    transformed, sudo_count = _rewrite_real_sudo_invocations(command)
    if sudo_count == 0:
        return command, None

    # Scope-aware read: under multiplex the process env may hold another profile's SUDO_PASSWORD;
    # unscoped callers (UnscopedSecretError) keep the os.environ read.
    try:
        from agent.secret_scope import get_secret
        _configured_password = get_secret("SUDO_PASSWORD")
    except Exception:
        _configured_password = os.environ.get("SUDO_PASSWORD")
    has_configured_password = _configured_password is not None
    sudo_password = _configured_password if has_configured_password else _get_cached_sudo_password()

    # sudoers NOPASSWD hosts must not be forced through the prompt or the -S pipe (local only).
    if not has_configured_password and not sudo_password and _sudo_nopasswd_works():
        return command, None

    # delegate_task children inherit HERMES_INTERACTIVE=1 (and possibly a stale thread-local
    # callback on a recycled worker) but have no user on the other side — always headless;
    # configured password, session cache and the NOPASSWD probe still apply.
    should_prompt_for_sudo = (
        env_var_enabled("HERMES_INTERACTIVE") or _get_sudo_password_callback() is not None
    ) and not _in_delegated_child_context()
    if not has_configured_password and not sudo_password and should_prompt_for_sudo:
        sudo_password = _prompt_for_sudo_password(timeout_seconds=45)
        if sudo_password:
            _set_cached_sudo_password(sudo_password)

    if has_configured_password or sudo_password:
        # sudo -S reads one line per invocation: compound `sudo a && sudo b` needs one line each.
        return transformed, (sudo_password + "\n") * sudo_count
    return command, None
