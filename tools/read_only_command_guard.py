"""Allowlist-first read-only command guard for Rob's operator toolset.

Ref: docs/... "Rob read-only operator P0/P1" implementation.

This answers a different question than ``tools.approval``: approval.py asks
"is this command dangerous?" (block known-bad, everything else defaults to
allowed/confirmable). This guard asks "is every part of this command
explicitly known-safe?" (allow only known-good; everything else is denied,
including things that are merely *unknown*, not just things flagged as
dangerous). Both apply to any Rob-facing tool: this guard runs first and is
strict enough to stand alone; ``tools.approval``/Tirith still run after for
any command this guard permits, as defense-in-depth (see
``run_read_only_guard`` docstring below).

Only command families reachable from a REGISTERED ``rob_*`` tool are
allowlisted (``_ALLOWED_SIMPLE`` / ``_SUBCOMMAND_VALIDATORS``). Validator
families no registered tool can emit — curl, find, file, rg, openssl, ip,
tailscale — were REMOVED as dead, security-sensitive surface in the
consolidated security-closure pass rather than kept "for future use"
(YAGNI): the registered tools already route HTTP through urllib and TLS
through Python ssl, and the generic file-traversal families only existed
for the deliberately-unregistered ``container_exec_readonly``.

Reuses ``tools.approval``'s existing shell tokenization/deobfuscation
primitives — the same ones ``tools/self_repo_guard.py`` composes on top of
for its own narrower guard — rather than re-parsing shell syntax from
scratch:

- ``_iter_shell_command_starts`` — quote/substitution-aware top-level
  command-start positions (splits on ``;`` ``&`` ``&&`` ``|`` ``||``
  ``\\n`` and descends into ``$(...)``/backticks/``(...)``/``{...}``).
- ``_read_shell_word`` — reads one shell word without executing expansions.
- ``_deobfuscate_shell_word_for_detection`` — collapses quoting/escaping so
  ``'l''s'`` and ``ls`` compare equal.

No new shell parser is written here. What IS new: an explicit allowlist of
command families (see ``_ALLOWED_SIMPLE``/``_SUBCOMMAND_VALIDATORS``) and a
redirection scanner (``_find_top_level_redirect``), since ``approval.py``'s
own redirect handling is keyed to "is this redirect target dangerous",
never "reject all redirection outright" — a strictly stronger rule this
guard needs that didn't previously exist anywhere in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.approval import (
    _deobfuscate_shell_word_for_detection,
    _iter_shell_command_starts,
    _read_shell_word,
)
from tools.sensitive_path_guard import find_sensitive_path_violation


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    # The exact command word(s) that triggered a denial, for logging/tests.
    offending: str = ""


class ReadOnlyGuardError(Exception):
    """Raised by callers that want an exception instead of a GuardResult."""


# ---------------------------------------------------------------------------
# Redirection / mutation-operator scanner
# ---------------------------------------------------------------------------
# Deliberately independent of approval.py's own redirect logic: that logic
# asks "is this redirect target sensitive", this asks "is there ANY
# redirection at all" (except a bare `>/dev/null` or `2>/dev/null`, which
# is harmless and extremely common in read-only diagnostic one-liners, e.g.
# `command -v foo >/dev/null`). Quote-state tracking mirrors
# `_iter_shell_command_starts`'s own loop so behavior stays consistent.

_HARMLESS_REDIRECT_TARGETS = {"/dev/null", "&1", "&2"}


def _find_top_level_redirect(command: str) -> str | None:
    """Return the offending redirect operator text, or None if none found."""
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        ch = command[i]
        if quote == "'":
            if ch == "'":
                quote = None
            i += 1
            continue
        if quote == '"':
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "<":
            # Here-strings/here-docs (<<, <<<) and plain input redirs (<)
            # are all denied — none are needed for read-only inspection,
            # and heredocs are a well-known way to smuggle arbitrary
            # multi-line payloads into an "innocent" leading command.
            return command[i : i + 3] if command[i : i + 3] == "<<<" else command[i : i + 2]
        if ch == ">":
            j = i + 1
            op = ">>" if j < n and command[j] == ">" else ">"
            target_start = j + (1 if op == ">>" else 0)
            while target_start < n and command[target_start] in " \t":
                target_start += 1
            target_end = target_start
            while target_end < n and command[target_end] not in " \t;\n&|)":
                target_end += 1
            target = command[target_start:target_end]
            if target not in _HARMLESS_REDIRECT_TARGETS:
                return command[i:target_end]
            i = target_end
            continue
        i += 1
    return None


# ---------------------------------------------------------------------------
# Per-command-family validators
# ---------------------------------------------------------------------------
# Each validator receives the deobfuscated argv for ONE top-level segment
# (the command word plus everything up to the next segment boundary,
# whitespace-split with quote-awareness via repeated _read_shell_word calls)
# and returns a GuardResult. Keep these narrow and explicit — a validator
# that "mostly works" for a command family is worse than not allowlisting
# that family at all, since this is an allow-first gate.


def _deny(cmd: str, why: str) -> GuardResult:
    return GuardResult(allowed=False, reason=why, offending=cmd)


def _allow() -> GuardResult:
    return GuardResult(allowed=True)


_GIT_ALLOWED_SUBCOMMANDS = {
    "status",
    "log",
    "diff",
    "show",
    "rev-parse",
    "merge-base",
}


def _git_has_file_output_flag(args: list[str]) -> bool:
    """``--output``/``--output=<file>`` (and the short ``-o<file>`` form) redirect
    a diff/log/show's output straight to an arbitrary file — a write primitive,
    not a read — and are never legitimate for a read-only inspection command.
    Checked independent of subcommand since several git subcommands share the
    same diff/log output-routing machinery."""
    for tok in args:
        if tok == "--output" or tok.startswith("--output="):
            return True
        if tok == "-o" or (tok.startswith("-o") and not tok.startswith("--") and len(tok) > 2):
            return True
    return False


def _validate_git(argv: list[str]) -> GuardResult:
    args = argv[1:]
    if not args:
        return _deny("git", "bare 'git' with no subcommand is not a read-only operation")
    sub = args[0]
    rest = args[1:]
    if _git_has_file_output_flag(rest):
        return _deny(f"git {sub}", "writing output to a file (--output/-o) is a write primitive, not a read")
    if sub == "branch":
        # Only the read-only "what branch am I on" form is allowed — any
        # other `git branch` invocation can create/delete/rename branches.
        if rest == ["--show-current"]:
            return _allow()
        return _deny("git branch", "only 'git branch --show-current' is allowed")
    if sub == "worktree":
        if rest[:1] == ["list"]:
            return _allow()
        return _deny("git worktree", "only 'git worktree list' is allowed")
    if sub == "tag":
        # Only the bare listing form. Any operand (a tag name to create, or
        # -d/-l/etc.) can create or delete a tag, which is a mutation.
        if not rest:
            return _allow()
        return _deny("git tag", "only bare 'git tag' (listing) is allowed — creating/deleting a tag is a mutation")
    if sub == "reflog":
        # Only the read-only "show" form: bare, a numeric limit (`-20`), or
        # an explicit `show`. `expire`/`delete` destroy reflog history.
        if not rest or (rest[0] == "show") or all(tok.lstrip("-").isdigit() for tok in rest):
            return _allow()
        return _deny("git reflog", "only 'git reflog' / 'git reflog -N' / 'git reflog show' are allowed")
    if sub in _GIT_ALLOWED_SUBCOMMANDS:
        return _allow()
    return _deny(f"git {sub}", f"git subcommand '{sub}' is not in the read-only allowlist")


def _validate_docker(argv: list[str]) -> GuardResult:
    """Only the subcommands a registered rob_* tool can emit — no speculative
    entries (compose config / image inspect had no registered caller and were
    removed as dead surface in the consolidated security pass)."""
    args = argv[1:]
    if not args:
        return _deny("docker", "bare 'docker' with no subcommand is not a read-only operation")
    sub = args[0]
    if sub == "compose":
        if args[1:2] == ["ps"]:
            return _allow()
        return _deny("docker compose", "only 'docker compose ps' is allowed")
    if sub == "stats":
        if "--no-stream" in args[1:]:
            return _allow()
        return _deny("docker stats", "docker stats requires --no-stream (a live stream is not a bounded read)")
    if sub in {"ps", "inspect", "logs"}:
        return _allow()
    if sub == "network" and args[1:2] == ["inspect"]:
        return _allow()
    if sub == "volume" and args[1:2] == ["inspect"]:
        return _allow()
    return _deny(f"docker {sub}", f"docker subcommand '{sub}' is not in the read-only allowlist")


def _validate_systemctl(argv: list[str]) -> GuardResult:
    args = argv[1:]
    if not args:
        return _deny("systemctl", "bare 'systemctl' with no verb is not a read-only operation")
    sub = args[0]
    if sub in {"status", "show"}:
        return _allow()
    return _deny(f"systemctl {sub}", f"systemctl verb '{sub}' is not in the read-only allowlist")


_JOURNALCTL_ALLOWED_BOOLEAN_FLAGS = frozenset({"--no-pager"})
_JOURNALCTL_ALLOWED_VALUE_FLAGS = frozenset({
    "-u", "--unit", "--since", "--until", "-p", "--priority",
})


def _validate_journalctl(argv: list[str]) -> GuardResult:
    """Allowlist-first, not a denylist: an earlier version denied a fixed
    set of administrative flags (--rotate, --vacuum-*, --flush, --sync,
    --relinquish-var, --setup-keys) by exact string match. journalctl uses
    GNU getopt_long, which accepts any UNAMBIGUOUS PREFIX of a long option
    (confirmed against the real binary: `journalctl --vacuum-tim=1s` is
    accepted as `--vacuum-time=1s`) — so `--rotat`, `--vacuum-s=1M`,
    `--flus`, `--sy`, `--relinquish-va` all slipped past an exact-match
    denylist while still resolving to the exact denied flag at runtime.
    An allowlist of the handful of flags the tool actually needs has no
    such gap: an abbreviation of an unlisted flag still isn't a match for
    anything in the allowed set, denied or not."""
    args = argv[1:]
    i = 0
    while i < len(args):
        tok = args[i]
        flag, sep, value = tok.partition("=")
        if flag in _JOURNALCTL_ALLOWED_BOOLEAN_FLAGS:
            pass
        elif flag in _JOURNALCTL_ALLOWED_VALUE_FLAGS:
            if not sep and i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1  # space-separated value form, e.g. `-u NAME`
        elif len(tok) > 2 and tok[:2] in ("-u", "-p") and not tok.startswith("--"):
            pass  # attached short form, e.g. `-uNAME`
        else:
            return _deny("journalctl", f"'{flag}' is not in the read-only journalctl allowlist")
        i += 1
    return _allow()


# Simple commands: allowed outright once the executable name matches, with
# no subcommand semantics to police. `dig` is deliberately ABSENT despite
# being a "read" in name: its `-f <file>` flag reads a file of query names
# and sends them to a resolver — a file-read/exfiltration primitive — and
# no registered rob_* tool emits it. `cd` changes the working directory
# only for the rest of the SAME shell invocation (each Rob command runs as
# its own fresh subprocess) — it never persists or mutates anything on
# disk, so `cd <repo> && git status` is exactly as read-only as
# `git status` alone. Needed by git_inspect/docker_compose_ps's templates.
_ALLOWED_SIMPLE = frozenset({
    "ls", "cat", "head", "tail", "grep", "stat", "readlink",
    "du", "df", "pwd",
    "ps", "pgrep", "pstree", "uptime", "uname", "free", "id", "whoami",
    "which", "whereis",
    "ss", "getent", "nslookup",
    "cd",
})


_SUBCOMMAND_VALIDATORS = {
    "git": _validate_git,
    "docker": _validate_docker,
    "systemctl": _validate_systemctl,
    "journalctl": _validate_journalctl,
}

# Executables that are an unconditional escape hatch regardless of args —
# no validator could make these safe for an allowlist-first read-only gate,
# since their entire purpose is running arbitrary user-supplied code.
_UNCONDITIONAL_DENY = frozenset({
    "sudo", "su", "doas",
    "rm", "mv", "cp", "touch", "mkdir", "truncate", "chmod",
    "chown", "chgrp", "ln", "tee", "install", "rsync", "shred",
    "sed", "perl", "awk", "gawk", "mawk",
    "xargs",
    "python", "python3", "node", "nodejs", "ruby", "php", "powershell",
    "pwsh", "lua", "irb", "deno", "bun",
    "sh", "bash", "dash", "zsh", "ksh", "fish", "csh", "tcsh",
    "kill", "pkill", "killall", "pkexec",
    "iptables", "ip6tables", "nft", "ufw", "firewall-cmd",
    "apt", "apt-get", "dpkg", "yum", "dnf", "snap", "pip", "pip3", "npm",
    "yarn", "pnpm", "gem", "cargo", "go",
    "reboot", "shutdown", "halt", "poweroff", "init", "systemd-run",
    "psql",  # SQL access goes through the dedicated db_select tool, never raw psql
    "mysql", "sqlite3",
    "less", "more", "vim", "vi", "nano", "emacs", "man",  # pagers/editors: shell-escape risk
    "eval", "exec", "source", ".",
    "at", "batch", "crontab",
    "docker-compose",  # only the `docker compose` (v2, space form) path is validated
    "nc", "ncat", "netcat", "socat",  # arbitrary network read/write, not a bounded probe
    "wget",  # no registered rob_* tool emits curl/wget; probes go through urllib/socket
    "scp", "sftp", "ftp",
})


def _split_top_level_segment(command: str, start: int, next_start: int | None) -> list[str]:
    """Read whitespace-separated shell words for one segment, quote-aware.

    Stops at the segment boundary (`next_start`, from
    `_iter_shell_command_starts`) or end of string. Does not stop at
    redirection operators — those are scanned separately by
    `_find_top_level_redirect` over the *whole* command so a mid-segment
    redirect is never mistaken for an argument.
    """
    end = next_start if next_start is not None else len(command)
    words: list[str] = []
    pos = start
    while pos < end:
        word_start, word_end, word = _read_shell_word(command, pos)
        if word_end <= pos:
            break
        if word:
            words.append(_deobfuscate_shell_word_for_detection(word))
        pos = word_end
    return words


def run_read_only_guard(command: str) -> GuardResult:
    """Allowlist-first check: every top-level segment's leading executable
    must be explicitly allowed, and no top-level redirection may be present
    anywhere in the command.

    This is the FULL check for a Rob read-only tool — callers should treat
    a GuardResult(allowed=False) as final. A GuardResult(allowed=True)
    should still be passed through ``tools.approval``'s existing
    danger-check (and Tirith, if enabled) before execution, as
    defense-in-depth against anything this allowlist did not anticipate —
    this guard narrows the surface, it does not replace the existing
    safety net.
    """
    if not command or not command.strip():
        return _deny("<empty>", "empty command")

    redirect = _find_top_level_redirect(command)
    if redirect is not None:
        return _deny(redirect, "output/input redirection is never permitted for a read-only operator command")

    starts = list(_iter_shell_command_starts(command))
    if not starts:
        return _deny(command, "no command found")

    for idx, start in enumerate(starts):
        next_start = starts[idx + 1] if idx + 1 < len(starts) else None
        argv = _split_top_level_segment(command, start, next_start)
        if not argv:
            continue
        exe = argv[0]
        if exe in _UNCONDITIONAL_DENY:
            return _deny(exe, f"'{exe}' is never permitted in a read-only operator command")
        if exe in _ALLOWED_SIMPLE:
            pass
        else:
            validator = _SUBCOMMAND_VALIDATORS.get(exe)
            if validator is None:
                return _deny(exe, f"'{exe}' is not in the read-only command allowlist")
            result = validator(argv)
            if not result.allowed:
                return result

        # agent.file_safety.get_read_block_error is NOT enforced against
        # raw shell reads by design (its own docstring: "NOT a security
        # boundary" against terminal_tool) — re-check every path-reading
        # command's arguments here so Rob gets no bypass around it.
        sensitive = find_sensitive_path_violation(argv)
        if sensitive:
            return _deny(exe, sensitive)

    return _allow()
