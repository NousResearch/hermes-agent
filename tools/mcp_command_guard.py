"""Command allowlist for MCP stdio subprocess spawning.

Handoff from the 69t.9 security audit (bead tasks-69t.4, finding C2): the
OX Apr-2026 MCP SDK disclosure showed the MCP Python SDK does not vet the
``command`` a server config hands to ``StdioServerParameters`` before
exec'ing it — that is a won't-fix at the SDK level, so the mitigation has
to live in the app that constructs ``StdioServerParameters``. This module
is that gate: call :func:`validate_stdio_command` on the fully-resolved
command string immediately before every ``StdioServerParameters(...)``
construction, and let the ``DisallowedMcpCommandError`` it raises propagate
(never silently swallow a rejected command).

Only a fixed set of interpreters/launchers may be spawned as MCP stdio
servers: ``npx``, ``uvx``, ``python``, ``python3``, ``node``, ``docker``,
``deno``. This mirrors the actual shape of every legitimate MCP server
config Hermes has ever shipped or documented (all of them launch through
one of these). Anything else — a bare shell, an arbitrary script, a
config-supplied absolute path to something unexpected — is rejected with a
logged security error.

The check is name-based (basename of the given path, after ``~`` expansion),
not a realpath/binary-provenance check: real-world ``npx`` installs are
routinely symlinks or shims whose resolved target is NOT itself named
``npx`` (e.g. Homebrew's ``npx`` -> `.../npm/bin/npx-cli.js`, a JS file
launched via a node shebang) — resolving through ``os.path.realpath``
before the allowlist check was tried during development and rejected
real, legitimate npx installs, so it was dropped in favor of this simpler
name check. Verifying that a resolved binary's PROVENANCE (not just its
name) matches an expected, trusted install location is real additional
hardening but needs a per-platform trusted-install-path registry; it is
listed as a documented follow-up rather than implemented here (see
``docs/security/mcp-stdio-sandbox.md``, tasks-69t.4).

Opt-in, default off: call sites must check :func:`is_enabled` before
calling :func:`validate_stdio_command`. Some operators run MCP servers
launched via a custom compiled binary or wrapper script that isn't one of
the allowed interpreters — enabling this by default would refuse to start
those servers on upgrade. See ``security.mcp_stdio_command_allowlist_enabled``
in config.yaml.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes"}


def is_enabled() -> bool:
    """Whether the MCP stdio command allowlist is turned on.

    Opt-in (default False). Checked separately from
    :func:`validate_stdio_command` so call sites can skip the check
    entirely rather than relying on the function to no-op, keeping the
    "raises, never silently swallowed" contract intact when it does run.
    """
    default = False
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("security", {}) or {}
    except Exception:
        cfg = {}
    return _env_bool(
        "HERMES_MCP_STDIO_COMMAND_ALLOWLIST_ENABLED",
        cfg.get("mcp_stdio_command_allowlist_enabled", default),
    )

# Interpreters/launchers legitimate MCP stdio servers use. Bare names are
# resolved against PATH by the caller (see tools.mcp_tool._resolve_stdio_command)
# before this check runs, so by the time we see the command it is either
# already absolute or about to be exec'd via PATH lookup unchanged.
ALLOWED_STDIO_COMMANDS = frozenset({
    "npx", "uvx", "python", "python3", "node", "docker", "deno",
})

# Windows executables carry a .exe/.cmd/.bat suffix even for allowlisted
# interpreters (npx.cmd, python.exe, docker.exe, ...).
_WINDOWS_EXEC_SUFFIXES = (".exe", ".cmd", ".bat", ".ps1")


class DisallowedMcpCommandError(ValueError):
    """Raised when an MCP stdio command fails the allowlist check."""


def _basename_without_suffix(path: str) -> str:
    # Split on both separators regardless of host OS: a Windows-style path
    # (backslash-separated) can reach this check on a POSIX host too — e.g.
    # a config authored on Windows and evaluated in a cross-platform test —
    # and posixpath.basename() would not treat '\\' as a separator there.
    basename = re.split(r"[\\/]+", path)[-1] if path else path
    lowered = basename.lower()
    for suffix in _WINDOWS_EXEC_SUFFIXES:
        if lowered.endswith(suffix):
            return basename[: -len(suffix)]
    return basename


def validate_stdio_command(
    command: str,
    *,
    server_name: str = "",
    extra_allowed: "frozenset[str] | None" = None,
) -> None:
    """Validate *command* against the MCP stdio command allowlist.

    Accepts absolute paths (e.g. ``/usr/local/bin/npx``) or bare names
    (e.g. ``npx``) — the check is against the given path's basename, with
    any Windows executable suffix stripped, case-insensitively. This is a
    name check, not a realpath/provenance check — see the module docstring
    for why.

    ``extra_allowed`` lets a specific call site widen the allowlist for a
    fixed, non-configurable launcher that isn't an interpreter (e.g.
    ``cua-driver``, a compiled native binary cua_backend.py spawns directly
    rather than through python/node). It is deliberately NOT a general
    escape hatch — only pass a literal frozenset of exact binary names the
    call site owns and controls, never anything derived from user input.

    Raises :class:`DisallowedMcpCommandError` (never returns a bool) so
    callers fail loudly instead of silently proceeding with a bad command.
    A security error is always logged before raising.
    """
    allowed = ALLOWED_STDIO_COMMANDS | (extra_allowed or frozenset())

    if not command or not isinstance(command, str) or not command.strip():
        logger.error(
            "SECURITY: rejected MCP stdio command for server '%s': empty or "
            "invalid command (tasks-69t.4 C2).",
            server_name,
        )
        raise DisallowedMcpCommandError(
            f"MCP server '{server_name}': empty or invalid command"
        )

    expanded = os.path.expanduser(command.strip())
    basename = _basename_without_suffix(expanded).lower()

    if basename not in allowed:
        logger.error(
            "SECURITY: rejected MCP stdio command for server '%s': %r "
            "(basename %r not in allowlist %s). "
            "Handoff from 69t.9 audit (tasks-69t.4 C2) — only "
            "npx/uvx/python/python3/node/docker/deno may be spawned as MCP "
            "stdio servers.",
            server_name, command, basename,
            sorted(allowed),
        )
        raise DisallowedMcpCommandError(
            f"MCP server '{server_name}': command {command!r} is not in the "
            f"MCP stdio command allowlist {sorted(allowed)}. "
            "Rejected before spawning per security policy (tasks-69t.4 C2)."
        )
