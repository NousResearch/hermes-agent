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
read-only tools expose: it extracts the non-flag (path-shaped) arguments
from an already-guard-approved filesystem-reading command and re-checks
each one against the SAME ``agent.file_safety.get_read_block_error`` used
elsewhere in the codebase — reused, not reimplemented. If any argument
resolves to a blocked path, the whole command is denied.

``agent/file_safety.py`` does NOT cover SSH private keys or generic
PEM/key material at all (confirmed directly: ``get_read_block_error``
returns ``None`` for ``~/.ssh/id_ed25519``) — its own category list is
Hermes-internal credential stores and project ``.env`` files only. The
P0 spec explicitly requires SSH-key/private-key-material denial, so that
category is implemented directly here rather than assumed-covered by
reuse.
"""

from __future__ import annotations

import os

from agent.file_safety import get_read_block_error

# Only these commands take file paths worth checking — `ps`/`ss`/`uname`
# etc. never do, and `docker inspect <container>` takes a container name,
# not a filesystem path, so it's deliberately excluded here (container
# content is a separate, later concern for container_exec_readonly).
_PATH_READING_COMMANDS = frozenset({"cat", "head", "tail", "stat", "file", "readlink", "grep", "rg", "find", "du"})

# Conventional OpenSSH private-key filenames (never their .pub siblings,
# which are not secret) plus generic key-material extensions used by
# TLS/x509 and other key formats.
_SSH_KEY_BASENAMES = frozenset({
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_ed25519_sk", "id_xmss",
})
_KEY_MATERIAL_EXTENSIONS = frozenset({".pem", ".key", ".ppk", ".p12", ".pfx"})


def _is_private_key_material(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    base = normalized.rsplit("/", 1)[-1]
    if base.endswith(".pub"):
        return None  # public keys are not secret
    if base in _SSH_KEY_BASENAMES:
        return f"'{path}' is an SSH private key file — denied"
    _, ext = os.path.splitext(base)
    if ext.lower() in _KEY_MATERIAL_EXTENSIONS:
        return f"'{path}' has a private-key-material extension ({ext}) — denied"
    if "/.ssh/" in f"/{normalized}" and base not in ("known_hosts", "config", "authorized_keys"):
        return f"'{path}' is inside an .ssh directory and is not a known-safe file — denied"
    return None


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
    for tok in argv[1:]:
        if not tok or tok.startswith("-"):
            continue
        candidate = os.path.expanduser(tok)

        key_material = _is_private_key_material(candidate)
        if key_material:
            return key_material

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
