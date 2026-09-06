"""Pre-gate floors for :mod:`tools.approval`: decisions that never reach a prompt.

Unconditional blocks (hardline, ``sudo -S`` password piping, the user's own
``approvals.deny`` globs) and the permanent command allowlist match. All of
them run BEFORE yolo / ``approvals.mode: off`` / cron approve-mode; the
allowlist runs after. Session state stays in ``tools.approval`` and is read
through it at call time.
"""

import contextlib
import fnmatch
import logging
import re
import time
import uuid
from tools import approval_context as _ctx
from tools.approval_detection import (
    _MALFORMED_EXEC_DESCRIPTION, _PARSER_LIMIT_DESCRIPTION, _command_detection_variants)

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
    for command_variant in _command_detection_variants(command):
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
    """Persist a parser-limit-blocked command as a runnable script. That block
    fires on payload SIZE/shape, not the operation — usually a legitimate script
    the model inlined. Saving it makes recovery one turn (`bash <file>`) instead
    of two, and is strictly safer than the hint-only path: the file goes through
    the normal execution pipeline (including the referenced-script content guard)
    and nothing runs here. Returns the path, or None on any failure (hint falls
    back to write_file)."""
    try:
        from hermes_constants import get_hermes_home
        script_dir = get_hermes_home() / "cache" / "blocked-scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        # Opportunistic cleanup: blocked payloads older than 7 days.
        cutoff = time.time() - 7 * 86400
        for old in script_dir.glob("blocked-*.sh"):
            with contextlib.suppress(OSError):
                if old.stat().st_mtime < cutoff:
                    old.unlink()
        path = script_dir / f"blocked-{int(time.time())}-{uuid.uuid4().hex[:8]}.sh"
        path.write_text(
            "#!/bin/bash\n"
            "# Auto-saved by Hermes: this command exceeded the inline command\n"
            "# parser limit and was blocked from direct execution. Review it,\n"
            f"# then run it via: bash {path}\n" + command + ("" if command.endswith("\n") else "\n"),
            # Force UTF-8 + lossy decode so non-UTF-8 child output can't crash the gateway thread on
            # locale-mismatched Windows (#53137).
            # Force UTF-8 + lossy decode so non-UTF-8 child output can't crash the gateway thread on
            # locale-mismatched Windows (#53137).
            # Force UTF-8 + lossy decode so non-UTF-8 child output can't crash the gateway thread on
            # locale-mismatched Windows (#53137).
            encoding="utf-8", errors="replace",
        )
        return str(path)
    except Exception:
        logger.debug("failed to save blocked payload", exc_info=True)
        return None


_RECOVERY_PREFIX = (
    " RECOVERY: this block fires on oversized/unparseable inline "
    "command payloads (heredocs, giant one-liners), not on the operation itself. "
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
    # followed by blind rephrase retries — point at the saved script (or the write_file recipe).
    if description in (_PARSER_LIMIT_DESCRIPTION, _MALFORMED_EXEC_DESCRIPTION):
        saved = _save_blocked_payload(command) if command else None
        if saved:
            message += _RECOVERY_PREFIX + (
                f"Your command was saved to {saved} — review it, then run: terminal(command=\"bash {saved}\"). "
                "Do not retry inline."
            )
        else:
            message += _RECOVERY_PREFIX + (
                "Write the script to a file with write_file, "
                "then run it: terminal(command=\"bash /path/script.sh\") or "
                "\"python3 /path/script.py\". Do not retry inline."
            )
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


def _has_allowlist_shell_operator(command: str) -> bool:
    """Return True when a command is too compound for the allowlist shortcut.
    Quote-aware: metacharacters inside quotes or behind a backslash are literal
    arguments (``cargo bench -- '^a(b|c)$'``), not shell syntax. Still
    disqualifying: ``$`` or backtick inside DOUBLE quotes (expansion stays
    active), and any quoted/escaped control character when the command also
    carries a ``-c``/``-e``/``--command``/``--eval``-style option that would
    hand the quoted text to another interpreter."""
    command = command or ""
    quote = None  # None | "'" | '"'
    has_reinterpretable = False
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
        elif ch in _SHELL_CONTROL_CHARS and ch not in "()":
            return True
        i += 1
    # An unterminated quote means we can't reason about the command shape.
    if quote is not None:
        return True
    return has_reinterpretable and bool(_REINTERPRETED_ARGUMENT_RE.search(command))


def _command_matches_permanent_allowlist(command: str) -> bool:
    """True when command_allowlist holds this exact command text or a matching
    glob. Permanent approvals historically store dangerous-pattern keys such as
    ``recursive delete``; manual entries are command text, possibly with
    shell-style wildcards like ``podman *``."""
    from tools import approval as _a
    command = (command or "").strip()
    if not command or _has_allowlist_shell_operator(command):
        return False
    with _a._lock:
        patterns = tuple(_a._permanent_approved)
    for pattern in patterns:
        pattern = pattern.strip() if isinstance(pattern, str) else ""
        if pattern and (command == pattern or (any(ch in pattern for ch in "*?[")
                                               and fnmatch.fnmatchcase(command, pattern))):
            return True
    return False
