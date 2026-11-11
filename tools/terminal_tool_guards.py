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
    r"\bdocker\s+compose\s+logs\s+(?:-f|--follow)\b",   # what usually follows a detached `up`
    r"\bnext\s+dev\b",
    r"\bvite(?:\s|$)",
    r"\bnodemon\b",
    r"\buvicorn\b",
    r"\bgunicorn\b",
    r"\bpython(?:3)?\s+-m\s+http\.server\b",
))

# The patterns above match a server name anywhere, so a command that only MENTIONS one
# (`grep -rn uvicorn .`, `ls node_modules/vite`, `npm install nodemon`) was refused too. A
# command segment is excused when the word it executes is known to treat its arguments as
# data, or when it is a bounded form of a server command. Everything else keeps the anywhere
# match: an unrecognised wrapper (`direnv exec . uvicorn`) must still be refused, so the closed
# set is the excuse list, never a launcher list.
_SEGMENT_SPLIT_RE = re.compile(r"[;&|\n]+")
# Windows launchers name the same command: `npm.cmd run dev`, `uvicorn.exe app:app`.
_WIN_LAUNCHER_SUFFIX_RE = re.compile(r"\b(\w+)\.(?:exe|cmd|bat|ps1)\b", re.IGNORECASE)
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_]\w*=")
# Substitutions run their contents whatever the outer command is (`echo $(npm run dev)`).
_SUBSTITUTION_MARKERS = ("$(", "`", "<(", ">(")
_TRANSPARENT_PREFIXES = frozenset({"sudo", "time", "then", "do", "else", "!"})
_DATA_COMMANDS = frozenset((
    "echo printf cat ls ll dir type grep egrep fgrep rg ag ack head wc stat du df file which whereis "
    "whence rm rmdir mkdir cp mv ln touch chmod chown cd pushd test [ sed awk sort uniq cut tr jq yq "
    "tree basename dirname realpath readlink diff cmp tar zip unzip fd locate pgrep ps lsof "
    "curl wget git export unset"
).split())
_DATA_UNLESS_FLAGGED = {
    "find": ("-exec", "-execdir", "-ok", "-okdir"),
    "tail": ("-f", "-F", "--follow"),
}
_PACKAGE_TOOLS = frozenset("npm pnpm yarn bun pip pip3 uv poetry pdm pipx brew apt apt-get apk dnf conda cargo gem".split())
_PACKAGE_DATA_SUBCOMMANDS = frozenset((
    "install i add update up upgrade remove rm uninstall ls list view info show why outdated audit ci "
    "pkg search freeze download sync lock"
).split())
_VITE_LAUNCHERS = frozenset({"sudo", "npx", "bunx", "pnpx", "npm", "pnpm", "yarn", "bun", "exec", "dlx", "x"})


def _is_bounded_server_form(tokens: list) -> bool:
    """A server command that exits by itself: a one-shot build, a config check, a detached start."""
    words = [t.lower() for t in tokens]
    for i, word in enumerate(words[:-1]):
        if re.split(r"[/\\]", word)[-1] == "vite" and words[i + 1] in ("build", "optimize"):
            # Only when vite itself is the command (`nodemon --exec vite build` is not), and a
            # watched build never exits.
            return (all(w in _VITE_LAUNCHERS or w.startswith("-") or _ENV_ASSIGNMENT_RE.match(w) for w in words[:i])
                    and not any(w in ("--watch", "-w") for w in words))
    if "--check-config" in words:
        return any(re.split(r"[/\\]", w)[-1] == "gunicorn" for w in words)
    # Detached `compose up` returns once the containers have started. What follows a bare `--`
    # belongs to another program, so a `-d` there does not count.
    if "--" in words:
        words = words[:words.index("--")]
    return "compose" in words and "up" in words and any(w in ("-d", "--detach") for w in words)


def _command_word(tokens: list) -> tuple[str, list]:
    """``(word, args)`` for the command a segment executes, past env assignments and the few
    prefixes that pass a command through unchanged. A wrong skip can only land on a word that
    is not a data command, which keeps the anywhere match."""
    i = 0
    while i < len(tokens):
        word = re.split(r"[/\\]", tokens[i].lstrip("({").lower())[-1]
        if word == "sudo":
            i += 1
            while i < len(tokens) and tokens[i].startswith("-"):
                i += 2 if tokens[i] in ("-u", "-g") else 1
        elif word in _TRANSPARENT_PREFIXES or _ENV_ASSIGNMENT_RE.match(tokens[i]):
            i += 1
        else:
            return word, tokens[i + 1:]
    return "", []


def _only_names_a_server(segment: str) -> bool:
    """True when *segment* cannot start a server whatever names it mentions."""
    if any(marker in segment for marker in _SUBSTITUTION_MARKERS):
        return False
    tokens = segment.split()
    if _is_bounded_server_form(tokens):
        return True
    word, args = _command_word(tokens)
    if word in _DATA_UNLESS_FLAGGED:  # `find -exec` runs a command; `tail -f` never exits
        return not any(arg in _DATA_UNLESS_FLAGGED[word] for arg in args)
    if word == "command":
        return args[:1] in (["-v"], ["-V"])
    if (word in ("uv", "python", "python3") and args[:1] == ["pip"]) or args[:2] == ["-m", "pip"]:
        args = args[args.index("pip") + 1:]
    elif word not in _PACKAGE_TOOLS:
        return word in _DATA_COMMANDS
    return next((arg for arg in args if not arg.startswith("-")), "") in _PACKAGE_DATA_SUBCOMMANDS


def _names_a_server(text: str) -> bool:
    return any(pattern.search(text) for pattern in _LONG_LIVED_FOREGROUND_PATTERNS)


def _starts_long_lived_server(unquoted: str) -> bool:
    unquoted = _WIN_LAUNCHER_SUFFIX_RE.sub(r"\1", unquoted)
    # Whole-string check first: nearly every command names no server at all.
    return _names_a_server(unquoted) and any(
        _names_a_server(segment) and not _only_names_a_server(segment)
        for segment in _SEGMENT_SPLIT_RE.split(unquoted))


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
        _starts_long_lived_server,
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
    from tools.environments.sandbox_scope import hosted_scope
    from tools.process_registry import _is_supervised_gateway_process
    from tools.terminal_tool import _resolve_command_cwd, get_session_cwd

    # A hosted command runs in the agent's remote sandbox, which cannot reach this gateway; the
    # scan below would read the scripts it names on this host.
    if hosted_scope() is not None or not _is_supervised_gateway_process():
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


def locked_skill_block(*, command: str, cwd: str, workdir: Optional[str], session_key: str) -> Optional[str]:
    """Refuse commands that write into a skill the user locked, change the lock record, or run
    ``hermes curator unlock`` (see tools/skill_lock). Local backend only: a remote or container
    backend cannot reach this profile's skills dir. Returns the JSON error string when blocked."""
    from tools.skill_lock import command_refusal
    from tools.terminal_tool import _resolve_command_cwd

    guard_cwd = _resolve_command_cwd(workdir=workdir, default_cwd=cwd, session_key=session_key)
    reason = command_refusal(command, guard_cwd)
    if reason is None:
        return None
    logger.warning("Blocked write to a locked skill (command: %s)", _safe_command_preview(command))
    return _blocked_json(f"Blocked: {reason}", "blocked")
