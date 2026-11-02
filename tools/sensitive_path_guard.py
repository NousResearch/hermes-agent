"""Sensitive-path enforcement for Rob's read-only operator commands.

``agent/file_safety.py``'s own docstring is explicit that
``get_read_block_error`` is "NOT a security boundary" against raw shell
access — it is only consulted by tools that call it directly (e.g. a
dedicated file-read tool), never by ``tools/terminal_tool.py``'s shell
execution path. A Rob command allowed by ``read_only_command_guard`` (e.g.
``cat``, ``grep``, ``head``) would therefore sail straight past that
existing check and reach ``~/.hermes/mcp-tokens/project-os.json`` or a
project's ``.env`` directly — exactly the "no bypass for Rob" the P0 spec
forbids.

This module closes that gap for the specific, narrow surface Rob's
read-only tools expose: it extracts the path-shaped arguments from an
already-guard-approved filesystem-reading command and re-checks each one
against the SAME ``agent.file_safety.get_read_block_error`` used elsewhere
in the codebase — reused, not reimplemented. On top of that reuse it adds
the checks ``get_read_block_error`` deliberately does not cover:

- Shell glob/expansion rejection. The Rob execution path is
  ``shell=True``, so the shell expands ``* ? [ ]`` (and brace/parameter
  expansions) BEFORE the command runs: a guard that reasons only about the
  literal user-supplied text sees wildcard text while the shell resolves
  it to the real sensitive file. Live-proven:
  ``cat <dir>/.s?h/id_ed2551*`` bypassed the literal ``.ssh``/private-key
  checks exactly this way. Any path-shaped argument containing such a
  metacharacter is denied outright rather than emulating shell expansion
  — failing closed is the whole point.
- SSH private-key / generic key-material files (``agent.file_safety``
  does not cover them: ``get_read_block_error`` returns ``None`` for
  ``~/.ssh/id_ed25519``). Matched on both the literal text AND the
  fully-resolved real path, so a symlink to a key under an innocent name
  cannot slip through.
- ``/proc`` policy: ``/proc/*/environ``, ``/proc/*/fd/*`` and
  ``/proc/*/mem`` (including every ``self`` spelling) are structurally
  denied rather than relying on output redaction.
"""

from __future__ import annotations

import os
import re

from agent.file_safety import get_read_block_error

# Only these commands take file paths worth checking — `ps`/`ss`/`uname`
# etc. never do, and `docker inspect <container>` takes a container name,
# not a filesystem path, so it's deliberately excluded here (container
# content is a separate, later concern for container_exec_readonly).
_PATH_READING_COMMANDS = frozenset({"cat", "head", "tail", "stat", "readlink", "grep", "du"})

# grep's FIRST non-flag argument is the search pattern, not a path.
# Rejecting glob metacharacters there would break legitimate registered
# use (`rob_journal_query` pipes user-supplied patterns into grep), and
# treating it as a filesystem path re-blocks patterns that merely LOOK
# like sensitive paths (e.g. `grep .env`). Only grep's subsequent
# non-flag tokens — and the value of `-f`/`--file`, which IS a path —
# are path-shaped.
_PATTERN_ARG_COMMANDS = frozenset({"grep"})

_GLOB_METACHARACTERS = frozenset("*?[]")

# Other shell expansions that rewrite a literal path before execution and
# are therefore equally unsafe to reason about literally: brace expansion
# (`{a,b}`) and parameter expansion (`$VAR` / `${VAR}`). Backticks and
# `$(...)` are structurally guarded elsewhere (approval.py's command-start
# iterator descends into them), so they need no handling here.
_SHELL_EXPANSION_CHARS = _GLOB_METACHARACTERS | frozenset("{}") | frozenset("$")

# Conventional OpenSSH private-key filenames (never their .pub siblings,
# which are not secret) plus generic key-material extensions used by
# TLS/x509 and other key formats.
_SSH_KEY_BASENAMES = frozenset({
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_ed25519_sk", "id_xmss",
})
_KEY_MATERIAL_EXTENSIONS = frozenset({".pem", ".key", ".ppk", ".p12", ".pfx"})
_SSH_SAFE_BASENAMES = frozenset({"known_hosts", "config", "authorized_keys"})

# /proc policy: process environment, open file descriptors and memory
# images are never needed by the registered process-inspection tool
# (`ps` metadata only) and must not depend on secret redaction to stay
# safe. Both the `self` and numeric-pid spellings are matched, and the
# check also runs on the realpath-resolved form (realpath rewrites
# `/proc/self/...` to `/proc/<pid>/...`).
_PROC_DENY_PATTERNS = (
    re.compile(r"^/proc/(?:self|[0-9]+)/environ$"),
    re.compile(r"^/proc/(?:self|[0-9]+)/fd(?:/|$)"),
    re.compile(r"^/proc/(?:self|[0-9]+)/mem$"),
)


def _glob_denial_reason(tok: str) -> str | None:
    """Reject any path argument the shell would expand before execution."""
    for ch in tok:
        if ch in _SHELL_EXPANSION_CHARS:
            return (
                f"path argument '{tok}' contains a shell expansion character "
                f"('{ch}') — globs and expansions are never permitted in read-only paths"
            )
    return None


def _key_material_denial(normalized: str, original: str) -> str | None:
    base = normalized.rsplit("/", 1)[-1]
    if base.endswith(".pub"):
        return None  # public keys are not secret
    if base in _SSH_KEY_BASENAMES:
        return f"'{original}' is an SSH private key file — denied"
    _, ext = os.path.splitext(base)
    if ext.lower() in _KEY_MATERIAL_EXTENSIONS:
        return f"'{original}' has a private-key-material extension ({ext}) — denied"
    if "/.ssh/" in f"/{normalized}" and base not in _SSH_SAFE_BASENAMES:
        return f"'{original}' is inside an .ssh directory and is not a known-safe file — denied"
    return None


def _is_private_key_material(path: str) -> str | None:
    """Check both the literal text and the symlink-resolved real path so a
    symlink to a key under an innocent filename cannot bypass the check."""
    normalized = path.replace("\\", "/")
    reason = _key_material_denial(normalized, path)
    if reason:
        return reason
    try:
        resolved = os.path.realpath(path).replace("\\", "/")
    except (OSError, ValueError):
        resolved = None
    if resolved and resolved != normalized:
        return _key_material_denial(resolved, path)
    return None


def _proc_denial_reason(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    variants = [normalized]
    try:
        variants.append(os.path.realpath(path).replace("\\", "/"))
    except (OSError, ValueError):
        pass
    for variant in variants:
        for pattern in _PROC_DENY_PATTERNS:
            if pattern.match(variant):
                return f"'{path}' is a /proc process memory/environment/fd path — denied"
    return None


def _path_tokens(argv: list[str], exe: str) -> list[str]:
    """Yield the path-shaped tokens of a guard-approved command segment.

    Flags (and their non-path values) are skipped. For pattern-first
    commands (`grep`), the first bare non-flag token and the values of
    `-e`/`--regexp` are patterns — never paths — while the values of
    `-f`/`--file` ARE pattern FILES and are yielded for checking. Once any
    explicit pattern flag (`-e`/`--regexp`/`-f`/`--file`) has appeared,
    every remaining non-flag token is a FILE under grep's own semantics —
    they must all be checked, never skipped as the implicit pattern.
    """
    tokens: list[str] = []
    pattern_seen = False
    explicit_pattern_flag = False
    i = 1
    n = len(argv)
    while i < n:
        tok = argv[i]
        if exe in _PATTERN_ARG_COMMANDS and tok.startswith("-"):
            if tok.startswith("--regexp="):
                explicit_pattern_flag = True
                i += 1
                continue
            if tok.startswith("--file="):
                tokens.append(tok.split("=", 1)[1])
                explicit_pattern_flag = True
                i += 1
                continue
            if tok in ("-e", "--regexp", "-f", "--file") and i + 1 < n:
                if tok in ("-e", "--regexp"):
                    explicit_pattern_flag = True
                    i += 2  # pattern — never a path
                else:
                    tokens.append(argv[i + 1])
                    explicit_pattern_flag = True
                    i += 2
                continue
            if len(tok) > 2 and tok[0] == "-" and tok[1] in "ef" and not tok.startswith("--"):
                # Attached short form: -ePATTERN (pattern, skip) /
                # -fFILE (pattern file, check).
                if tok[1] == "f":
                    tokens.append(tok[2:])
                explicit_pattern_flag = True
                i += 1
                continue
            i += 1
            continue
        if exe in _PATTERN_ARG_COMMANDS and not pattern_seen and not explicit_pattern_flag:
            pattern_seen = True
            i += 1
            continue
        tokens.append(tok)
        i += 1
    return tokens


def find_sensitive_path_violation(argv: list[str]) -> str | None:
    """Return a denial reason if any path-shaped argument in ``argv`` is a
    blocked sensitive path, else None. ``argv`` is the already-deobfuscated
    word list for one command segment, as produced by
    ``read_only_command_guard._split_top_level_segment``."""
    if not argv:
        return None
    exe = argv[0]
    if exe not in _PATH_READING_COMMANDS:
        return None
    for tok in _path_tokens(argv, exe):
        if not tok:
            continue

        glob_denial = _glob_denial_reason(tok)
        if glob_denial:
            return glob_denial

        candidate = os.path.expanduser(tok)

        key_material = _is_private_key_material(candidate)
        if key_material:
            return key_material

        proc_denial = _proc_denial_reason(candidate)
        if proc_denial:
            return proc_denial

        try:
            error = get_read_block_error(candidate)
        except Exception:
            # Fail closed: if the safety check itself errors on a
            # malformed/unusual path, treat it as blocked rather than
            # silently letting the read through.
            return f"path '{tok}' could not be safety-checked — denied closed"
        if error:
            return error
    return None
