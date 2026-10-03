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
import threading
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


def _uses_background_amp(unquoted: str) -> bool:
    return bool(_INLINE_BACKGROUND_AMP_RE.search(unquoted) or _TRAILING_BACKGROUND_AMP_RE.search(unquoted))


_LONG_LIVED_FOREGROUND_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|watch)\b",
    r"\bdocker\s+compose\s+up\b",
    r"\bnext\s+dev\b",
    r"\bvite(?:\s|$)",
    r"\bnodemon\b",
    r"\buvicorn\b",
    r"\bgunicorn\b",
    r"\bpython(?:3)?\s+-m\s+http\.server\b",
))

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
        _uses_background_amp,
        "Foreground command uses '&' backgrounding. Re-send WITHOUT the '&' as "
        "terminal(command=\"<cmd>\", background=true) — add notify_on_complete=true "
        "for bounded jobs — then run health checks and tests in follow-up terminal calls.",
    ),
    (
        lambda s: any(p.search(s) for p in _LONG_LIVED_FOREGROUND_PATTERNS),
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


# This rejection is fully deterministic -- a command matching one of the
# _FOREGROUND_GUIDANCE predicates will ALWAYS be rejected, identically, no
# matter how many times it is retried. Hermes's general loop guardrail
# (agent/tool_guardrails.py) deliberately exempts "terminal" from its
# same-tool-failure hard-stop, since most terminal failures are genuine varied
# diagnosis, not a loop -- but that exemption also covers this one
# non-diagnostic, 100%-reproducible category, and retried commands usually
# differ, so no single exact signature repeats enough to trip anything else.
# Observed: 13 identical rejections in one task. Track it locally instead and,
# from the 3rd rejection, hand the model the exact call to resend.
#
# Counted per agent (the code-kernel owner: approval session key, qualified per
# delegated child, task_id as fallback) and per user turn. Only a
# background=true call resets it: resetting on unrelated successful foreground
# commands let a model retrying `&` between other work never reach 3 (live:
# counts went 1,1,2,1,1,1,1). The turn scope keeps one early slip from
# escalating a mistake many turns later in a long gateway conversation; a
# batch-eval rollout is a single turn, so its retry loop still escalates.
_FG_BG_REJECTION_LOCK = threading.Lock()
_FG_BG_REJECTIONS: dict[str, tuple[str, int]] = {}  # owner -> (turn id, rejections)
_FG_BG_ESCALATE_AFTER = 3
_CORRECTED_CALL_MAX_CHARS = 200
_LEADING_DETACH_WRAPPER_RE = re.compile(r"^(?:nohup|setsid)\s+(?=[^\s-])")


def _fg_bg_owner(task_id: str) -> str:
    # Same identity session kernels use, so delegated children get their own
    # counter instead of adding to the parent's (they inherit its session key).
    from tools.code_kernel import _resolve_owner

    return _resolve_owner(task_id)


def _backgrounded_command(command: str) -> str | None:
    """*command* as it should be resent with background=true: minus a trailing
    '&' and one leading nohup/setsid. None when fixing it would take more than
    that (inline '&', disown, wrapper flags, chained wrappers) or the result is
    too long to quote back -- the caller then shows the placeholder form rather
    than a guessed rewrite."""
    fixed = command.strip()
    tail = _TRAILING_BACKGROUND_AMP_RE.search(fixed)
    # Only strip an '&' the quote-masked command also ends with, so a quoted or
    # heredoc '&' is never cut.
    if tail and (masked := _TRAILING_BACKGROUND_AMP_RE.search(_strip_quotes(fixed))):
        if masked.group(0) == tail.group(0):
            fixed = fixed[:tail.start()].rstrip()
    if wrapper := _LEADING_DETACH_WRAPPER_RE.match(fixed):
        fixed = fixed[wrapper.end():]
    rest = _strip_quotes(fixed)
    if not fixed or len(fixed) > _CORRECTED_CALL_MAX_CHARS or (
        _SHELL_LEVEL_BACKGROUND_RE.search(rest) or _uses_background_amp(rest)
    ):
        return None
    return fixed


def _escalated_guidance(command: str, msg: str, count: int) -> str:
    fixed = _backgrounded_command(command)
    call = 'terminal(command={}, background=true, notify_on_complete=true)'.format(
        json.dumps(fixed, ensure_ascii=False) if fixed else '"<cmd>"'
    )
    how = (
        f"Resend exactly this: {call}" if fixed else
        f"Resend it as {call}, where <cmd> is your command without the '&' / "
        "nohup / setsid / disown."
    )
    return (
        f"REPEATED REJECTION (#{count}): you have already received this guidance. "
        "A foreground command that backgrounds itself or starts a long-lived process "
        "is NEVER accepted as written, and rewording it will not help. "
        f"{msg} {how}"
    )


def _foreground_background_guidance(command: str, task_id: str = "") -> str | None:
    """Guidance text when a foreground command looks long-lived or uses shell
    backgrounding (it should be a managed background session), else None.

    Only the terminal tool's own call site passes ``task_id``; without it the
    guidance never escalates.
    """
    if _looks_like_help_or_version_command(command):
        return None
    unquoted = _strip_quotes(command)
    msg = next((m for hit, m in _FOREGROUND_GUIDANCE if hit(unquoted)), None)
    if msg is None or not task_id:
        return msg
    from tools.approval_context import get_current_turn_id

    owner, turn = _fg_bg_owner(task_id), get_current_turn_id()
    with _FG_BG_REJECTION_LOCK:
        last_turn, count = _FG_BG_REJECTIONS.get(owner, (turn, 0))
        count = (count if last_turn == turn else 0) + 1
        _FG_BG_REJECTIONS[owner] = (turn, count)
    if count < _FG_BG_ESCALATE_AFTER:
        return msg
    return _escalated_guidance(command, msg, count)


def reset_foreground_background_guard(task_id: str) -> None:
    """Clear the caller's rejection count; called only for background=true calls,
    the corrected usage this guard steers toward."""
    if task_id:
        with _FG_BG_REJECTION_LOCK:
            _FG_BG_REJECTIONS.pop(_fg_bg_owner(task_id), None)


def clear_foreground_background_guard(session_key: str) -> None:
    """Drop a session's counters, its delegated children's included. Wired into
    ``tools.approval.clear_session`` so a long-lived gateway does not keep one
    entry per finished conversation."""
    if not session_key:
        return
    child_prefix = f"{session_key}::child::"
    with _FG_BG_REJECTION_LOCK:
        for owner in [o for o in _FG_BG_REJECTIONS if o == session_key or o.startswith(child_prefix)]:
            del _FG_BG_REJECTIONS[owner]


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
        HOST_INTERPRETER_KILL_REJECTION,
        contains_host_interpreter_kill,
        contains_launchctl_submit_command,
        lifecycle_scan_root_within_budget,
        scan_gateway_lifecycle,
    )
    # Keep the specific launchctl diagnostic when this optional pre-scan fits the
    # budget. The full fail-closed guard below still runs when it does not, so
    # oversized roots never reach shlex here.
    if lifecycle_scan_root_within_budget(command) and contains_launchctl_submit_command(command):
        return _blocked_json(
            "Blocked: launchctl submit/bootstrap is restricted inside a supervised "
            "gateway regardless of the job label, to prevent indirect gateway "
            "restart loops. This guard does not inspect the job's KeepAlive settings "
            "or determine whether it is independent of Hermes. Perform authorized "
            "LaunchAgent maintenance from a separate shell outside the gateway, "
            "not by switching launchctl verbs to bypass this rejection.",
            "error",
        )
    guard_cwd_base = get_session_cwd(session_key)
    if guard_cwd_base is None:
        guard_cwd_base = getattr(env, "cwd", None) or cwd
    guard_cwd = _resolve_command_cwd(
        workdir=workdir, default_cwd=guard_cwd_base, session_key=session_key, env_type=env_type,
        mounted_host=getattr(env, "host_cwd", None),
        env=env,
    )
    unsafe, refusal = scan_gateway_lifecycle(
        command,
        cwd=guard_cwd,
        read_remote_script=lambda p: _read_script_for_guard(env, guard_cwd, p, _MAX_REFERENCED_SCRIPT_BYTES),
    )
    if unsafe and refusal:
        # Not a lifecycle command: a script the command EXECUTES could not be scanned (budget,
        # size, device, live SQLite, cloud placeholder). Say so, or the model rewords and retries
        # the same command in a loop (#113944).
        return _blocked_json(
            f"Blocked: the lifecycle guard could not scan this command or referenced script: {refusal}. "
            "Nothing in the command is known to contain a gateway lifecycle command, but a "
            "script the command executes must be scannable (a regular text file under 1 MiB) "
            "before it can run inside the gateway process.",
            "error",
        )
    if unsafe:
        # Name the ownership-scoped route for image-name kills: the intent is almost always "stop
        # MY background job", and re-rolling the same over-broad spelling is what takes the gateway down.
        if lifecycle_scan_root_within_budget(command) and contains_host_interpreter_kill(command):
            return _blocked_json(HOST_INTERPRETER_KILL_REJECTION, "error")
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
