"""Pre-execution guards for the terminal tool.

Pure functions that decide whether a command may run at all: workdir
validation, the foreground long-lived/background-operator guidance, the
supervised-gateway lifecycle block, and the Windows self-repo git guard.
Each ``*_block`` helper returns a finished JSON error string, or None when
the command may proceed. Split out of tools/terminal_tool.py; the origin
module re-imports every public helper so ``tools.terminal_tool.<name>``
keeps resolving.
"""

import json
import logging
import re
import shlex
import stat
from pathlib import Path
from typing import Any, Optional

from tools.shell_heredoc import strip_inert_heredoc_bodies

logger = logging.getLogger("tools.terminal_tool")


# Workdir allowlist: Unicode alnum plus path/drive/UNC separators and common
# punctuation; shell metacharacters stay rejected. Unicode is allowed on
# purpose (e.g. CJK vault paths). Defense-in-depth — the cwd is also
# shlex-quoted before reaching the shell.
_WORKDIR_SAFE_ASCII_CHARS = frozenset('/\\:_-.~ +@=,')


def _is_safe_workdir_char(ch: str) -> bool:
    if not ch or ord(ch) < 32 or ord(ch) == 127:  # control chars / NUL
        return False
    return ch.isalnum() or ch in _WORKDIR_SAFE_ASCII_CHARS


def _validate_workdir(workdir: str) -> str | None:
    """Error message if *workdir* has a disallowed character, else None.
    Allowlist rather than deny-list so novel metacharacters can't slip through."""
    for ch in workdir or "":
        if not _is_safe_workdir_char(ch):
            return (
                f"Blocked: workdir contains disallowed character {repr(ch)}. "
                "Use a simple filesystem path without shell metacharacters."
            )
    return None


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


def _blocked_json(error: str, status: str) -> str:
    """The guard result envelope: exit_code 1 + *error* + *status*."""
    return json.dumps({"output": "", "exit_code": 1, "error": error, "status": status}, ensure_ascii=False)


_SHELL_LEVEL_BACKGROUND_RE = re.compile(
    r"(?:^|[;&|]\s*|&&\s*|\|\|\s*|\$\(\s*)(?:nohup|disown|setsid)\b", re.IGNORECASE | re.MULTILINE
)
_INLINE_BACKGROUND_AMP_RE = re.compile(r"\s&\s")
_TRAILING_BACKGROUND_AMP_RE = re.compile(r"\s&\s*(?:#.*)?$")


def _strip_quotes(command: str) -> str:
    """Blank quoted / backtick content and provably-inert heredoc bodies so
    regex checks can't match keywords (nohup, setsid, '&') inside strings.

    Heredocs are masked FIRST: their delimiter may itself be quoted
    (``<<'EOF'``). ``strip_inert_heredoc_bodies`` is conservative — only a
    quoted, terminated delimiter on a simple opener fed to a known non-shell
    consumer is masked, so a real background operator can't hide in one.
    """
    result = strip_inert_heredoc_bodies(command)
    result = re.sub(r"'[^']*'", "''", result)
    result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', result)
    return re.sub(r"`[^`]*`", "``", result)


_LONG_LIVED_FOREGROUND_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|watch)\b",
    r"\bdocker\s+compose\s+up\b",
    r"\bnext\s+dev\b",
    r"\bvite(?:\.(?:js|ts|mjs|cjs))?(?:\s+(?!build\b)|$)",
    r"\bnodemon\b",
    r"\buvicorn\b",
    r"\bgunicorn\b",
    r"\bpython(?:3)?\s+-m\s+http\.server\b",
))

# Commands that START one of the services above, keyed by the pattern that
# must then match their word sequence. Scoping long-lived detection to the
# command's own word sequence means `ps aux | grep uvicorn` (inspection) is
# never mistaken for `python -m uvicorn` (server start).
_LONG_LIVED_START_WORDS = {
    "npx",
    "npm",
    "pnpm",
    "yarn",
    "bun",
    "docker",
    "next",
    "vite",
    "uvicorn",
    "gunicorn",
    "nodemon",
}

# A segment wrapped in `timeout <N>` is bounded: it cannot occupy the
# foreground forever, so it is not a long-lived process even when the wrapped
# executable is a server (`timeout 15 .venv/bin/python -m uvicorn ...`).
# Optional `VAR=...` assignments, flags (e.g. `-s KILL`) and unit suffixes
# (`15s`/`1m`) are accepted.
_TIMEOUT_BOUNDED_SEGMENT_RE = re.compile(
    r"^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*timeout\s+(?:[^\d\s][^\s]*\s+)*\d+(?:\.\d+)?(?:s|m|h|d)?\b",
    re.IGNORECASE,
)


def _shell_words(command: str) -> list[str]:
    """Tokenize a command into shell words (quotes handled by shlex)."""
    try:
        return shlex.split(command)
    except ValueError:
        # Malformed quoting; fall back to whitespace split so detection still
        # runs on whatever is present rather than silently skipping.
        return command.split()


def _logical_command_segments(command: str) -> list[str]:
    """Split a command into logical units on shell list/pipeline operators.

    ``&&``/``||`` are consumed as one separator. The split is quote-naive on
    purpose: the caller strips quoted spans first, and the only job here is
    to find where each command starts so server detection can be scoped.
    """
    parts: list[str] = []
    i = 0
    n = len(command)
    while i < n:
        while i < n and command[i].isspace():
            i += 1
        start = i
        while i < n and command[i] not in ";|&":
            i += 1
        if i > start:
            parts.append(command[start:i])
        if i < n and command[i] in "&|" and i + 1 < n and command[i + 1] == command[i]:
            i += 2
        else:
            i += 1
    return parts


def _command_start_words(segment: str) -> list[str]:
    """Return the words that actually START the command in a segment.

    Walks past env assignments, `cd <path> &&` links, and the common wrappers
    (``sudo``/``env``/``exec``/``timeout N``), so the returned list begins
    with the program being started (e.g. ``["python", "-m", "uvicorn", ...]``
    or ``["ps", "aux", "|", "grep", ...]``).
    """
    words = _shell_words(segment)
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        # env assignment: FOO=1 (no spaces around `=`)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w):
            i += 1
            continue
        if w in ("cd",):
            # consume `cd <path>` and a following `&&` / `;`
            i += 2
            while i < len(words) and words[i] in ("&&", ";"):
                i += 1
            continue
        if w in ("sudo", "env", "exec", "timeout"):
            i += 1
            while i < len(words):
                tok = words[i]
                if tok in ("&&", ";", "|"):
                    break
                # consume a numeric arg (`timeout 5`)
                if re.match(r"^-?\d", tok):
                    i += 1
                    continue
                # consume short-option clusters and their value (`-s TERM`,
                # `-u root`, `-k 30`); stop at the first bare word.
                if re.match(r"^-[a-zA-Z]", tok):
                    i += 1
                    # a short option with a value?
                    if i < len(words) and not words[i].startswith("-"):
                        i += 1
                    continue
                break
            continue
        out.extend(words[i:])
        break
    return out


def _is_detached_docker_compose_up(words: list[str]) -> bool:
    """Return True when the command is ``docker compose up -d|--detach``.

    ``docker compose up -d`` (or ``--detach``) hands the containers over to the
    Docker daemon and the command line itself returns immediately — it is NOT a
    long-lived foreground process, so it must not be nudged to background=true
    (and must not be hard-blocked as a server start). Only bare ``docker
    compose up`` (no detach) keeps the foreground attached and legitimately
    trips the long-lived guard.
    """
    if not words:
        return False
    first = words[0].rsplit("/", 1)[-1]
    if first != "docker":
        return False
    try:
        up = words.index("up")
    except ValueError:
        return False
    # `compose` must precede the `up` subcommand (so `docker ps | grep up`
    # inspection stays untouched), and the detach flag must come after it.
    if "compose" not in words[:up]:
        return False
    return any(tok in ("-d", "--detach") for tok in words[up + 1:])


def _long_lived_foreground_hit(unquoted: str) -> bool:
    """Scoped long-lived detection: a real server/service START only.

    ``ps aux | grep uvicorn`` / ``journalctl -u X | grep uvicorn`` are
    inspections, not starts — the server word only appears as a grep operand
    there. ``docker compose up -d`` returns immediately (daemon-owned) and
    ``timeout <N> ...`` wrappers bound the process, so neither is long-lived.
    """
    for segment in _logical_command_segments(unquoted):
        if _TIMEOUT_BOUNDED_SEGMENT_RE.match(segment):
            continue
        words = _command_start_words(segment)
        if not words:
            continue
        first = words[0].rsplit("/", 1)[-1]
        if first not in _LONG_LIVED_START_WORDS and not first.startswith("python"):
            # `python -m uvicorn` / `python -m http.server` start a service;
            # any other first word (ps, grep, curl, ...) is not a start.
            continue
        if _is_detached_docker_compose_up(words):
            continue
        if any(p.search(" ".join(words)) for p in _LONG_LIVED_FOREGROUND_PATTERNS):
            return True
    return False

# Ordered (predicate on the unquoted command, guidance) — first hit wins.
_FOREGROUND_GUIDANCE = (
    (
        _SHELL_LEVEL_BACKGROUND_RE.search,
        "Foreground command uses shell-level background wrappers (nohup/disown/setsid). "
        "Re-send WITHOUT the wrapper as terminal(command=\"<cmd>\", background=true, "
        "notify_on_complete=true) so Hermes tracks the process, then run readiness "
        "checks and tests in separate commands.",
    ),
    (
        lambda s: _INLINE_BACKGROUND_AMP_RE.search(s) or _TRAILING_BACKGROUND_AMP_RE.search(s),
        "Foreground command uses '&' backgrounding. Re-send WITHOUT the '&' as "
        "terminal(command=\"<cmd>\", background=true) — add notify_on_complete=true "
        "for bounded jobs — then run health checks and tests in follow-up terminal calls.",
    ),
    (
        _long_lived_foreground_hit,
        "This foreground command appears to start a long-lived server/watch process. "
        "Run it with background=true, verify readiness (health endpoint/log signal), "
        "then execute tests in a separate command.",
    ),
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
    """Guidance text when a foreground command looks long-lived or uses shell
    backgrounding (it should be a managed background session), else None."""
    if _looks_like_help_or_version_command(command):
        return None
    unquoted = _strip_quotes(command)
    return next((msg for hit, msg in _FOREGROUND_GUIDANCE if hit(unquoted)), None)


def _read_script_for_guard(env: Any, guard_cwd: str, script_path: str, max_bytes: int) -> Optional[str]:
    """Best-effort script read: host filesystem first, then a bounded
    ``env.execute('head -c ... < path')`` for remote backends. Binary content
    (NUL byte) is not a script: feeding it to the guard tokenizes machine code
    into bogus paths and crashes the scanner, so it yields None."""
    if env is None:
        return None
    try:
        local_path = Path(script_path).expanduser()
        if not local_path.is_absolute():
            local_path = Path(guard_cwd) / local_path
        if local_path.is_file():
            metadata = local_path.stat()
            if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= max_bytes:
                data = local_path.read_bytes()
                if len(data) <= max_bytes:
                    return None if b"\x00" in data else data.decode("utf-8", errors="replace")
    except Exception:
        pass
    # Remote backend: bound the read at the source with `head -c` so an
    # oversized binary never crosses the wire (an unbounded `cat` once
    # pinned the gateway's tool thread for 30+ min on a shlex scan). One
    # byte over budget is enough for lifecycle_guard to fail closed. The
    # `< path` redirect keeps leading-dash paths out of argv.
    try:
        result = env.execute(f"head -c {max_bytes + 1} < {shlex.quote(script_path)}")
        if result.get("returncode", -1) == 0:
            output = result.get("output", "")
            return None if output and "\x00" in output else output
    except Exception:
        pass
    return None


def gateway_lifecycle_block(
    *,
    command: str,
    env: Any,
    env_type: str,
    cwd: str,
    workdir: Optional[str],
    session_key: str,
) -> Optional[str]:
    """Refuse gateway lifecycle commands issued from inside the supervised gateway.

    ``systemctl``/``launchctl``/``hermes gateway restart|stop|uninstall``
    targeting hermes-gateway would SIGTERM the gateway — and this very
    subprocess — before completing, so the service may never come back.
    Applies unconditionally (``force=True`` cannot bypass it). Gated on the
    SUPERVISED-gateway probe, not the raw ``_HERMES_GATEWAY`` marker: that
    marker leaks into every process that merely imports gateway.run (hermes
    serve, CLI, web server), which must still be able to restart the gateway;
    an unsupervised foreground ``hermes gateway run`` has no KeepAlive to turn
    a self-restart into a respawn loop, so it passes too.
    Returns the JSON error string when blocked, else None.
    """
    from tools.process_registry import _is_supervised_gateway_process
    from tools.terminal_tool import _resolve_command_cwd, get_session_cwd

    if not _is_supervised_gateway_process():
        return None
    from cron.lifecycle_guard import (
        _MAX_REFERENCED_SCRIPT_BYTES,
        contains_gateway_lifecycle_command_or_referenced_script,
        contains_launchctl_submit_command,
        lifecycle_scan_root_within_budget,
    )
    # Keep the specific launchctl diagnostic when this optional pre-scan fits the
    # budget. The full fail-closed guard below still runs when it does not, so
    # oversized roots never reach shlex here.
    if lifecycle_scan_root_within_budget(command) and contains_launchctl_submit_command(command):
        return _blocked_json(
            "Blocked: launchctl submit/bootstrap registers a persistent "
            "KeepAlive job and is unsafe from inside the gateway process. "
            "Use Hermes cron for one-shot delayed work, or install an "
            "explicit LaunchAgent from a separate shell.",
            "error",
        )
    guard_cwd_base = get_session_cwd(session_key)
    if guard_cwd_base is None:
        guard_cwd_base = getattr(env, "cwd", None) or cwd
    guard_cwd = _resolve_command_cwd(
        workdir=workdir, default_cwd=guard_cwd_base, session_key=session_key, env_type=env_type,
    )
    if contains_gateway_lifecycle_command_or_referenced_script(
        command,
        cwd=guard_cwd,
        read_remote_script=lambda p: _read_script_for_guard(env, guard_cwd, p, _MAX_REFERENCED_SCRIPT_BYTES),
    ):
        return _blocked_json(
            "Blocked: command or referenced script cannot restart, stop, or "
            "uninstall the gateway from inside the gateway process. The gateway would "
            "kill this command before it could complete (SIGTERM propagates "
            "to child processes). Run `hermes gateway restart` from a "
            "separate shell outside the running gateway.",
            "error",
        )
    return None


def self_repo_block(
    *,
    command: str,
    cwd: str,
    workdir: Optional[str],
    session_key: str,
) -> Optional[str]:
    """Windows-only guard against git-mutating the checkout backing this interpreter.

    NTFS locks loaded module files, so rewriting the live checkout can corrupt
    the running process; POSIX keeps old inodes alive for open handles, so the
    guard is off there (``guard_active``). Local backend only — remote
    backends cannot reach that checkout. Returns the JSON error string when
    blocked, else None.
    """
    from tools.self_repo_guard import detect_self_repo_git_mutation, guard_active
    from tools.terminal_tool import _resolve_command_cwd

    if not guard_active():
        return None
    guard_cwd = _resolve_command_cwd(workdir=workdir, default_cwd=cwd, session_key=session_key)
    hit, msg = detect_self_repo_git_mutation(command, guard_cwd)
    if not hit:
        return None
    logger.warning("Blocked self-repo git mutation (command: %s)", _safe_command_preview(command))
    return _blocked_json(msg, "blocked")
