"""Pre-gate floors for :mod:`tools.approval`: decisions that never reach a prompt.

Unconditional blocks (hardline, ``sudo -S`` password piping, the user's own
``approvals.deny`` globs) and the permanent command allowlist match. All of
them run BEFORE yolo / ``approvals.mode: off`` / cron approve-mode; the
allowlist runs after. Session state stays in ``tools.approval`` and is read
through it at call time.
"""

import fnmatch
import logging
import re
import shlex
import time
import uuid
from tools import approval_context as _ctx
from tools.approval_ansi_c import expand_ansi_c_quotes
from utils import unlink_files_older_than
from tools.approval_detection import (
    _FALLBACK_MATCH_SUFFIX, _MALFORMED_EXEC_DESCRIPTION, _PARSER_LIMIT_DESCRIPTION, _deny_command_variants)

logger = logging.getLogger("tools.approval")


def _match_user_deny_rule(command: str) -> str | None:
    """Return the matching ``approvals.deny`` glob, or None. User-defined fnmatch
    globs that block unconditionally — like the hardline floor, a match fires
    BEFORE the yolo / mode=off bypass ("never let the agent run this, even under
    yolo"). Case-insensitive, run over the same normalized/deobfuscated variants
    the dangerous-pattern detector uses so quoting tricks (``r\\m``,
    ``git st""atus``) can't sidestep a rule."""
    try:
        deny_patterns = _ctx._get_approval_config().get("deny") or []
    except Exception:
        return None
    globs = [p.strip() for p in deny_patterns if isinstance(p, str) and p.strip()]
    if not globs:
        return None
    for command_variant in _deny_command_variants(command):
        candidate = command_variant.lower().strip()
        for pattern in globs:
            if fnmatch.fnmatchcase(candidate, pattern.lower()):
                return pattern
    return None


def _user_deny_block_result(pattern: str) -> dict:
    """Build the standard block result for an ``approvals.deny`` match."""
    return {"approved": False, "user_deny": True, "message": (
        f"BLOCKED: this command matches the user-defined deny rule "
        f"'{pattern}' (approvals.deny in config.yaml). It cannot be "
        "executed via the agent — not even with --yolo, /yolo, or "
        "approvals.mode=off. Do NOT retry or rephrase this command; the user has explicitly forbidden it.")}


def _save_blocked_payload(command: str) -> str | None:
    """Persist a parser-limit-blocked command for the user to review. That block fires on payload
    SIZE/nesting, not the operation, so the payload is usually a legitimate script the model
    inlined. The file is NOT a way around the block: `bash <file>` runs a script the guard never
    reads, so the hint points the model at smaller commands or at the user instead. Nothing runs
    here. Returns the path, or None on any failure. Never in a hosted scope: the command is the
    tenant's, and a host path would mean nothing inside its sandbox."""
    from tools.hosted_tool_policy import host_feature_enabled
    if not host_feature_enabled("tool_host_artifacts"):
        return None
    try:
        from hermes_constants import get_hermes_home
        script_dir = get_hermes_home() / "cache" / "blocked-scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        # Opportunistic cleanup: blocked payloads older than 7 days.
        unlink_files_older_than(script_dir, "blocked-*.sh", 7 * 86400)
        path = script_dir / f"blocked-{int(time.time())}-{uuid.uuid4().hex[:8]}.sh"
        path.write_text(
            "#!/usr/bin/env bash\n"
            "# Auto-saved by Hermes: this command exceeded the inline command\n"
            "# parser limit and was blocked from direct execution. The approval\n"
            "# guard has not read it: review all of it before running it yourself.\n"
            + command + ("" if command.endswith("\n") else "\n"),
            # UTF-8 with replacement so a non-UTF-8 command cannot fail the save on a non-UTF-8 locale.
            encoding="utf-8", errors="replace",
        )
        return str(path)
    except Exception:
        logger.debug("failed to save blocked payload", exc_info=True)
        return None


_RECOVERY_PREFIX = (
    " RECOVERY: this block fires on oversized or too deeply nested inline "
    "command payloads (heredocs, giant one-liners), not on the operation itself. "
)
# Running the payload from a file would execute a script the approval guard never read.
_RECOVERY_STEPS = (
    "Split the work into smaller commands the guard can read, or ask the user to run it "
    "themselves. Do not run the payload from a file to get around this check, and do not retry inline."
)


def _hardline_block_result(description: str, command: str = "") -> dict:
    """Build the standard block result for a hardline match."""
    message = (
        f"BLOCKED (hardline): {description}. "
        "This command is on the unconditional blocklist and cannot "
        "be executed via the agent — not even with --yolo, /yolo, "
        "approvals.mode=off, or cron approve mode. If you genuinely "
        "need to run it, run it yourself in a terminal outside the agent."
    )
    # The parser-limit block is almost always a giant inline payload, not a forbidden operation, and is typically
    # followed by blind rephrase retries: say what to do instead, and save the payload for the user.
    if description in (_PARSER_LIMIT_DESCRIPTION, _MALFORMED_EXEC_DESCRIPTION) or description.endswith(_FALLBACK_MATCH_SUFFIX):
        saved = _save_blocked_payload(command) if command else None
        message += _RECOVERY_PREFIX + (f"Your command was saved to {saved} for the user to review. " if saved else "")
        message += _RECOVERY_STEPS
    return {"approved": False, "hardline": True, "message": message}


def _sudo_stdin_block_result(description: str) -> dict:
    """Build the standard block result for sudo stdin guard."""
    return {"approved": False, "message": (
        f"BLOCKED: {description}. "
        "Do not pipe passwords to 'sudo -S' — this is a brute-force "
        "attack vector. Set SUDO_PASSWORD in your .env file if the "
        "agent needs passwordless sudo, or run the sudo command manually in your own terminal.")}


# Shell control characters that make a command compound when they appear OUTSIDE quotes. Inside quotes they are
# literal to the outer shell — but they become executable again if an option like `-c`/`-e`/`--eval` (or a git `-c
# alias.x=!...`) hands the quoted argument to another interpreter, so quoted control chars only disqualify a command
# when such an option is present.
# Port of can1357/oh-my-pi#7553.
_SHELL_CONTROL_CHARS = frozenset("\n\r;&|<>`$()")

_REINTERPRETED_ARGUMENT_RE = re.compile(r"(?:^|[ \t])(?:-[^-\s]*[ce]|--(?:command|eval))(?:[= \t]|$)")


# Bash builtins that evaluate an operand as an array subscript / arithmetic expression, re-expanding `$(...)` and
# backticks the outer shell kept literal inside single quotes (`test -v 'a[$(cmd)]'`, `[[ x -eq 'a[$(cmd)]' ]]`,
# `printf -v`, `unset`, `read`, `declare`, `local`, `let`). Value: an option that must also be present, or None.
_SUBSCRIPT_EVALUATING_BUILTINS = {
    "test": "-v", "[": "-v", "[[": None, "printf": "-v",
    "unset": None, "read": None, "declare": None, "typeset": None, "local": None, "let": None,
}
_ASSIGNMENT_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_SUBSCRIPT_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\[.*\]\+?=")  # a[...]=v also evaluates the subscript
_PREFIX_WORDS = frozenset(("builtin", "command", "time", "!"))


def _evaluates_quoted_subscript(command: str) -> bool:
    # Dequote like bash does, so `'test' '-v'`, `t"es"t`, `\test` cannot dodge the table.
    try:
        words = shlex.split(command.replace("\\\n", ""), posix=True)  # bash joins line continuations first
    except ValueError:
        return True
    while words:
        word = words[0]
        if _SUBSCRIPT_ASSIGNMENT_RE.match(word):
            return True
        if word in _PREFIX_WORDS:
            words = words[1:]
            while word in ("builtin", "command", "time") and words and words[0].startswith("-"):
                words = words[1:]  # builtin --, command -p, time -p
        elif _ASSIGNMENT_WORD_RE.match(word):
            words = words[1:]
        else:
            break
    if not words or words[0] not in _SUBSCRIPT_EVALUATING_BUILTINS:
        return False
    required = _SUBSCRIPT_EVALUATING_BUILTINS[words[0]]
    # printf -vNAME; a glob (-?, -[v]) expands to -v when such a file exists in the cwd
    return required is None or any(w.startswith(required) or fnmatch.fnmatchcase(required, w) for w in words[1:])


def _has_allowlist_shell_operator(command: str) -> bool:
    """Return True when a command is too compound for the allowlist shortcut.
    Quote-aware: metacharacters inside quotes or behind a backslash are literal
    arguments (``cargo bench -- '^a(b|c)$'``), not shell syntax. Still
    disqualifying: ``$`` or backtick inside DOUBLE quotes (expansion stays
    active), and any quoted/escaped control character when the command also
    carries a ``-c``/``-e``/``--command``/``--eval``-style option that would
    hand the quoted text to another interpreter, or is a builtin that
    evaluates an operand as an array subscript (``test -v``, ``[[``, ...),
    or contains an unquoted ``${``/``$[`` whose subscript/offset re-expands it."""
    command = expand_ansi_c_quotes(command or "")  # $'-v' / $"-v" -> '-v' / "-v", so quote tracking matches bash
    quote = None  # None | "'" | '"'
    has_reinterpretable = False
    param_expansion = False  # unquoted ${...} / $[...] re-expand quoted text in subscripts/offsets
    test_v_mode = _evaluates_quoted_subscript(command)
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "\\" and quote != "'":
            nxt = command[i + 1] if i + 1 < n else ""
            if nxt in _SHELL_CONTROL_CHARS:
                has_reinterpretable = True
            i += 2
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            elif quote == '"' and ch in ("`", "$"):
                return True  # expansion is active inside double quotes
            elif ch in _SHELL_CONTROL_CHARS:
                has_reinterpretable = True
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "$":
            # Unquoted $ is only compound when it opens a substitution ("$HOME"
            # stays simple, matching the historical `\$\(` behavior).
            if i + 1 < n and command[i + 1] == "(":
                return True
            if i + 1 < n and command[i + 1] in "{[":
                param_expansion = True
        elif ch == "{":
            param_expansion = True  # brace expansion: test {-v,} 'a[$(cmd)]'
        elif ch in _SHELL_CONTROL_CHARS and ch not in "()":
            return True
        i += 1
    # An unterminated quote means we can't reason about the command shape.
    if quote is not None:
        return True
    if (test_v_mode or param_expansion) and has_reinterpretable:
        return True
    return has_reinterpretable and bool(_REINTERPRETED_ARGUMENT_RE.search(command))


def _matching_permanent_allowlist_entry(command: str) -> str | None:
    """The command_allowlist entry matching this exact command text or as a glob, or None.
    Permanent approvals historically store dangerous-pattern keys such as
    ``recursive delete``; manual entries are command text, possibly with
    shell-style wildcards like ``podman *``."""
    from tools import approval as _a
    from tools.approval_host_policy import permanent_grants_allowed
    if not permanent_grants_allowed():
        return None
    command = (command or "").strip()
    if not command or _has_allowlist_shell_operator(command):
        return None
    with _a._lock:
        patterns = tuple(_a._permanent_set())
    for pattern in patterns:
        pattern = pattern.strip() if isinstance(pattern, str) else ""
        if pattern and (command == pattern or (any(ch in pattern for ch in "*?[")
                                               and fnmatch.fnmatchcase(command, pattern))):
            return pattern
    return None


def _command_matches_permanent_allowlist(command: str) -> bool:
    """True when command_allowlist holds this exact command text or a matching glob."""
    return _matching_permanent_allowlist_entry(command) is not None
