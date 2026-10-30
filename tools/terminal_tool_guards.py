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
import os
import re
import shlex
import stat
from datetime import datetime
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
                f"Blocked: workdir contains disallowed character {ch!r}. "
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
        lambda s: _INLINE_BACKGROUND_AMP_RE.search(s) or _TRAILING_BACKGROUND_AMP_RE.search(s),
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


def _foreground_background_guidance(command: str) -> str | None:
    """Guidance text when a foreground command looks long-lived or uses shell
    backgrounding (it should be a managed background session), else None."""
    if _looks_like_help_or_version_command(command):
        return None
    unquoted = _strip_quotes(command)
    return next((msg for hit, msg in _FOREGROUND_GUIDANCE if hit(unquoted)), None)


# `&` spellings that are not backgrounding: and-lists, fd redirects (`2>&1`, `>&2`, `&>`, `&>>`,
# `<&3`) and the stderr pipe `|&`.
_NON_BACKGROUND_AMP_RE = re.compile(r"&&|\d*[<>]&(?:\d+|-)?|&>>?|\|&")
_TRAILING_AMP_ONLY_RE = re.compile(r"\s&\s*$")
# Executable substitutions run code the quote-masking cannot see into (`"$(sleep 9 & echo $!)"`
# hides a second, untracked background job), so any spelling of one refuses the conversion. Raw
# substring match on purpose: a false refusal is today's behaviour, a false conversion is not.
_EXECUTABLE_SUBSTITUTION_MARKERS = ("$(", "`", "<(", ">(")


def _strip_trailing_amp(command: str) -> str:
    """*command* minus its final `&` and the UNESCAPED whitespace right before it.

    A backslash-escaped space is part of the last argument (`trailing\\ ` is the word `trailing `),
    so trimming stops at the first whitespace byte preceded by an odd run of backslashes.
    """
    body = command.rstrip()[:-1]
    end = len(body)
    while end and body[end - 1].isspace():
        backslashes = len(body[:end - 1]) - len(body[:end - 1].rstrip("\\"))
        if backslashes % 2:
            break
        end -= 1
    return body[:end]


def _approval_identity(command: str) -> tuple:
    """Everything the approval path keys a decision on for *command*: hardline verdict, the
    dangerous pattern key/description (what session/permanent approvals are stored under), the
    sudo-stdin floor, runtime self-delete, the user's deny rules and the permanent allowlist."""
    from agent.runtime_self_protection import command_deletes_runtime
    from tools.approval import _command_matches_permanent_allowlist, _match_user_deny_rule
    from tools.approval_detection import _check_sudo_stdin_guard, detect_dangerous_command, detect_hardline_command
    return (
        tuple(detect_hardline_command(command)),
        tuple(detect_dangerous_command(command)),
        tuple(_check_sudo_stdin_guard(command)),
        command_deletes_runtime(command),
        _match_user_deny_rule(command),
        bool(_command_matches_permanent_allowlist(command)),
    )


def trailing_amp_background_command(command: str) -> str | None:
    """The command without its trailing `&` when that `&` is its ONLY backgrounding, else None.

    Only the unambiguous shape converts: one line, one and-or list or pipeline (`a && b | c &`
    backgrounds the whole list in bash, exactly what background=true runs), nothing after the
    `&`. Inline `a & b`, `;` sequences, multi-line scripts/heredocs, nohup/disown/setsid and any
    executable substitution (`$(...)`, backticks, `<(...)`/`>(...)`) keep the refusal: the tool
    cannot prove there is no other background job or that later commands don't depend on a
    readiness check. Quoted `&` (and inert heredoc bodies) never count. Conversion must leave
    the approval identity untouched, otherwise the refusal stands.
    """
    if not isinstance(command, str) or _looks_like_help_or_version_command(command):
        return None
    if any(marker in command for marker in _EXECUTABLE_SUBSTITUTION_MARKERS):
        return None
    unquoted = _strip_quotes(command)
    if not _TRAILING_AMP_ONLY_RE.search(unquoted) or not command.rstrip().endswith("&"):
        return None
    body = unquoted.rstrip()[:-1]
    if "\n" in body or "\r" in body or ";" in body or "#" in body or _SHELL_LEVEL_BACKGROUND_RE.search(body):
        return None
    if "&" in _NON_BACKGROUND_AMP_RE.sub("", body):
        return None
    stripped = _strip_trailing_amp(command)
    if not stripped.strip():
        return None
    # Approval runs on what executes (the stripped command). Some patterns key on the `&` itself
    # (`hermes gateway run &` = gateway outside its service manager), and approvals are stored per
    # pattern key, so a different key would let a pre-approval for another category pass.
    if _approval_identity(command) != _approval_identity(stripped):
        return None
    return stripped


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


# --- FIX-2068-e: unsupervised (CLI/interactive) raw-kick defer gate ---------------------------
#
# The supervised branch of ``gateway_lifecycle_block`` below hard-blocks gateway lifecycle
# commands issued from INSIDE the gateway (self-kill). By design it lets the CLI/interactive
# scope through ("CLI ... must still be able to restart the gateway") — but that pass-through
# carried ZERO in-flight awareness, which is exactly the 2026-10-25 FIX-2068 storm: a CLI
# session issued 6 bare ``launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`` terminal calls
# (12:24/14:27/14:39/14:49/15:50/16:25), each SIGTERM killing every in-flight kanban worker
# (FIX-2066 died 5x, FIX-2068 died 2x, both hit dispatcher gave_up). The CLI-side
# ``hermes gateway`` guard (refuse_while_sessions_live) was deliberately bypassed ("不走 hermes CLI").
#
# This section adds the missing in-flight awareness + rate limiting to the UNSUPERVISED scope:
# a destructive gateway-lifecycle command issued from a CLI/interactive process is deferred
# (blocked with a defer message) while kanban workers are in flight, with a 2h log-dedup
# cooldown (the drift-check pattern). The legitimate manual-restart channel is preserved:
#   * 0 workers in flight -> allowed (log only, do not block), and
#   * HERMES_GATEWAY_RESTART_FORCE=1 -> explicit bypass (allowed, log only).
# Non-lifecycle commands are never touched. The supervised branch is unchanged (regression-safe).

_KANBAN_WORKER_MARKER = "hermes_cli.kanban_worker_client"
_KICK_FORCE_ENV = "HERMES_GATEWAY_RESTART_FORCE"
# Same cooldown file the L3 drift-check uses (one file, many keys) — a distinct key so this
# guard's defer dedup never races drift-check's gateway-code-drift window.
_KICK_COOLDOWN_FILE = Path.home() / ".hermes" / "data" / "l3-cooldown.json"
_KICK_COOLDOWN_KEY = "cli-raw-kick-defer"
_KICK_COOLDOWN_HOURS = 2
# FIX-946: the out-of-process deferred-restart channel (consumed by gateway-drift-check.py
# within 30 min, itself guarded against in-flight kanban workers). Recommended in the defer msg.
_FIX946_SIGNAL_FILE = Path.home() / ".hermes" / "data" / "gateway-restart-pending"
# ps command-line signatures of the live gateway — the plain `-m hermes_cli.main gateway run`
# form and the `--external-supervisor` runpy re-exec form (drift-check FIX-2068e patterns).
_GW_CMD_PATTERNS = (
    "hermes_cli.main gateway run",
    "hermes_cli/main.py', 'gateway', 'run",
)
# Fast pre-filter: a gateway-lifecycle command must mention at least one of these, so the 99% of
# terminal commands that never touch the gateway skip the (shlex-costly) full detector below.
_LIFECYCLE_PREFILTER_TOKENS = (
    "launchctl", "systemctl", "hermes", "pkill", "kill", "taskkill", "stop-process",
)


def _mentions_gateway_lifecycle_token(command: str) -> bool:
    low = command.lower()
    return any(tok in low for tok in _LIFECYCLE_PREFILTER_TOKENS)


def _live_gateway_pid() -> Optional[int]:
    """PID of the live gateway a kick would target, or None (no gateway running).

    Official pid-file lookup first, then a ps command-line fallback (both gateway forms) so a
    profile-scoped pid-file miss still finds the launchd job's gateway. Any failure -> None
    (no gateway found -> nothing in flight to kill -> the caller allows)."""
    try:
        from gateway.status import get_running_pid

        pid = get_running_pid(cleanup_stale=False)
        if pid:
            return int(pid)
    except Exception as exc:
        logger.debug("FIX-2068-e get_running_pid failed: %s", exc)
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True, timeout=10, check=False
        ).stdout
        for line in out.splitlines():
            if "stderr_timestamp" in line:
                continue  # the stderr wrapper's argv contains the child's, not the gateway itself
            if any(p in line for p in _GW_CMD_PATTERNS):
                pid_s = line.strip().split()[0]
                if pid_s.isdigit():
                    return int(pid_s)
    except Exception as exc:
        logger.debug("FIX-2068-e ps gateway-pid fallback failed: %s", exc)
    return None


def kanban_workers_in_flight(gw_pid: Optional[int]) -> int:
    """Count live kanban runs parented to gateway process *gw_pid* (drift-check FIX-2068e pattern).

    A ``launchctl kickstart -k`` SIGTERMs the gateway and its process tree, so workers whose ppid is
    the live gateway pid lose their in-flight task cards on a kick. Orphans from a previous gateway
    (ppid=1) do NOT lose work in a new-gateway kick, so they are not counted. ps failure or no
    gateway pid -> 0 (degrade: never let the guard itself break the restart machinery)."""
    if not gw_pid:
        return 0
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-ax", "-o", "pid=,ppid=,command="], capture_output=True, text=True, timeout=10, check=False
        ).stdout
        n = 0
        for line in out.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) < 3:
                continue
            _pid, ppid_s, cmd = parts
            if ppid_s == str(gw_pid) and _KANBAN_WORKER_MARKER in cmd:
                n += 1
        return n
    except Exception as exc:
        logger.warning("FIX-2068-e worker count failed (degrading to 0): %s", exc)
        return 0


def _in_kick_cooldown() -> bool:
    """True while a previous in-flight defer is inside the 2h log-dedup window."""
    if not _KICK_COOLDOWN_FILE.exists():
        return False
    try:
        state = json.loads(_KICK_COOLDOWN_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    last = state.get(_KICK_COOLDOWN_KEY)
    if not last:
        return False
    try:
        age = (datetime.now() - datetime.fromisoformat(last)).total_seconds()
        return age < _KICK_COOLDOWN_HOURS * 3600
    except ValueError:
        return False


def _mark_kick_cooldown() -> None:
    state: dict = {}
    if _KICK_COOLDOWN_FILE.exists():
        try:
            state = json.loads(_KICK_COOLDOWN_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}
    state[_KICK_COOLDOWN_KEY] = datetime.now().isoformat()
    try:
        _KICK_COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KICK_COOLDOWN_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    except OSError as exc:
        logger.warning("FIX-2068-e cooldown write failed: %s", exc)


def _unsupervised_kick_defer_block(
    command: str, *, _gw_pid: Optional[int] = None, _workers_in_flight: Optional[int] = None,
) -> Optional[str]:
    """FIX-2068-e: defer a destructive gateway-lifecycle command issued from an UNSUPERVISED
    (CLI/interactive) process while kanban workers are in flight. Returns the JSON defer string
    when blocked, else None (the command may proceed).

    ``_gw_pid`` / ``_workers_in_flight`` are injection seams for tests; when None they are
    computed live (gateway pid -> ps worker count).
    """
    # 1. Fast pre-filter: skip the full detector for the 99% of commands that never touch the
    #    gateway (no lifecycle token present).
    if not _mentions_gateway_lifecycle_token(command):
        return None
    # 2. Full detection — the SAME data-sink-exempted choke point the supervised path uses, so
    #    splice bypasses and data-sink false positives behave identically in both scopes, and
    #    unrelated labels (com.example.*, other hermes services) stay allowed.
    from cron.lifecycle_guard import _lifecycle_command_scan_with_data_exemption

    if not _lifecycle_command_scan_with_data_exemption(command):
        return None
    # Compute live state once (injection seams above short-circuit this for tests).
    if _gw_pid is None:
        _gw_pid = _live_gateway_pid()
    if _workers_in_flight is None:
        _workers_in_flight = kanban_workers_in_flight(_gw_pid)
    # 3. Explicit force bypass: a deliberate manual restart that accepts killing in-flight work.
    if os.environ.get(_KICK_FORCE_ENV) == "1":
        logger.info(
            "FIX-2068-e unsupervised gateway kick ALLOWED via force bypass %s=1 "
            "(%d worker(s) in flight; command: %s)",
            _KICK_FORCE_ENV, _workers_in_flight, _safe_command_preview(command),
        )
        return None
    # 4. Legitimate manual-restart channel: nothing in flight -> allow (log only, do not block).
    if _workers_in_flight <= 0:
        logger.info(
            "FIX-2068-e unsupervised gateway kick ALLOWED (0 kanban workers in flight; command: %s)",
            _safe_command_preview(command),
        )
        return None
    # 5. In flight > 0 -> defer (block) + log one line + 2h log-dedup cooldown.
    first_defer = not _in_kick_cooldown()
    if first_defer:
        _mark_kick_cooldown()
        logger.info(
            "FIX-2068-e unsupervised gateway kick DEFERRED: %d kanban worker(s) in flight on "
            "pid=%s; recommended channel = FIX-946 signal file %s; force bypass = %s=1 "
            "(command: %s)",
            _workers_in_flight, _gw_pid, _FIX946_SIGNAL_FILE, _KICK_FORCE_ENV,
            _safe_command_preview(command),
        )
    else:
        logger.info(
            "FIX-2068-e unsupervised gateway kick still DEFERRED (2h cooldown active): "
            "%d worker(s) in flight (command: %s)",
            _workers_in_flight, _safe_command_preview(command),
        )
    defer_msg = (
        f"Deferred: this command would restart/stop the gateway, but {_workers_in_flight} "
        f"kanban worker(s) are in flight on gateway pid={_gw_pid} — a raw `launchctl kickstart` "
        f"SIGTERMs the gateway process tree and kills their in-flight task cards (the 2026-10-25 "
        f"FIX-2068 storm shape). It was issued from an unsupervised (CLI/interactive) process, so "
        f"it is deferred rather than self-approved. Choose one:\n"
        f"  1. Preferred channel — write the FIX-946 restart signal file; the drift-check consumes "
        f"it out-of-process within 30 min and itself defers while workers are in flight:\n"
        f"       printf '%s' '{{\"reason\":\"manual\"}}' > {_FIX946_SIGNAL_FILE}\n"
        f"  2. Wait for the {_workers_in_flight} worker(s) to finish, then re-run this command.\n"
        f"  3. Force this kick now (you accept killing the {_workers_in_flight} in-flight worker(s)):\n"
        f"       {_KICK_FORCE_ENV}=1 <this command>"
    )
    return _blocked_json(defer_msg, "deferred")


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

    UNSUPERVISED (CLI/interactive) scope — FIX-2068-e dual mode: a CLI must still
    restart the gateway, but the raw pass-through previously had ZERO in-flight
    awareness (the 2026-10-25 storm: 6 bare ``launchctl kickstart`` terminal calls
    killed every in-flight kanban worker). Destructive lifecycle commands from here
    now route through ``_unsupervised_kick_defer_block``: 0 kanban workers in flight
    -> allow (log only), >0 -> defer (block) + one log line pointing at the FIX-946
    out-of-process restart channel, ``HERMES_GATEWAY_RESTART_FORCE=1`` -> explicit
    bypass. Non-lifecycle commands and unrelated launchd labels are never touched.

    Returns the JSON error string when blocked, else None.
    """
    from tools.process_registry import _is_supervised_gateway_process
    from tools.terminal_tool import _resolve_command_cwd, get_session_cwd

    if not _is_supervised_gateway_process():
        # FIX-2068-e: UNSUPERVISED (CLI/interactive) scope. The supervised hard block above does
        # not apply here by design — a CLI must still be able to restart the gateway. But that
        # pass-through previously carried ZERO in-flight awareness (the 2026-10-25 FIX-2068 storm:
        # 6 bare `launchctl kickstart` terminal calls killed every in-flight kanban worker). Route
        # destructive lifecycle commands through the in-flight defer gate: 0 workers -> allow,
        # >0 workers -> defer (block) + log, force env -> allow. Non-lifecycle commands pass.
        return _unsupervised_kick_defer_block(command)
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
