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
not a full binary-identity check: real-world ``npx`` installs are routinely
symlinks or shims whose resolved target is NOT itself named ``npx`` (e.g.
Homebrew's ``npx`` -> `.../npm/bin/npx-cli.js`, a JS file launched via a
node shebang) — requiring the resolved TARGET file to itself be literally
named ``npx`` was tried during development and rejected real, legitimate
npx installs, so it was dropped in favor of this simpler name check.

That name check alone is not sufficient, though (teknium1 review on
PR #62808): ``tools.mcp_tool._resolve_stdio_command`` resolves a bare
command like ``npx`` through the MCP *server's own configured* ``env``
PATH before this guard ever sees it, so a malicious/misconfigured server
entry whose ``env.PATH`` puts an attacker binary named ``npx`` first would
have the resolved absolute path (e.g. ``/attacker/npx``) reach this guard,
pass the basename check, and spawn. To close that, any command given in
PATH FORM — absolute (``/attacker/npx``) OR relative-with-a-separator
(``./npx``, ``subdir/npx``) — names a specific file and is additionally
required to have trusted PROVENANCE (see :func:`_provenance_ok`) before
it's accepted; a relative path form is made absolute against the CWD
first so it can't skip the check. A bare command name (no path separator)
is exempt: it hasn't been resolved to a specific file yet, and if it
can't be resolved at spawn time it fails with ENOENT regardless of this
guard.

Provenance is only enforced for the base interpreter set
(``ALLOWED_STDIO_COMMANDS``); a per-call-site ``extra_allowed`` widening
(e.g. cua-driver, see below) is a fixed, call-site-owned literal the
operator's own environment resolves — not a remote MCP server config — so
it is outside the threat model this closes and is exempt.

Opt-in, default off: call sites must check :func:`is_enabled` before
calling :func:`validate_stdio_command`. Some operators run MCP servers
launched via a custom compiled binary or wrapper script that isn't one of
the allowed interpreters — enabling this by default would refuse to start
those servers on upgrade. See ``security.mcp_stdio_command_allowlist_enabled``
in config.yaml. This is a behavioral flag, not a secret, so per AGENTS.md
it lives only in config.yaml — no ``HERMES_*`` env var override.
"""

from __future__ import annotations

import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    """Whether the MCP stdio command allowlist is turned on.

    Opt-in (default False), config.yaml-only
    (``security.mcp_stdio_command_allowlist_enabled``) — no env var
    override; this is non-secret behavioral config, and AGENTS.md reserves
    ``HERMES_*`` env vars for secrets. Checked separately from
    :func:`validate_stdio_command` so call sites can skip the check
    entirely rather than relying on the function to no-op, keeping the
    "raises, never silently swallowed" contract intact when it does run.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("security", {}) or {}
    except Exception:
        cfg = {}
    return bool(cfg.get("mcp_stdio_command_allowlist_enabled", False))

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


def _trusted_install_dirs() -> tuple[str, ...]:
    """Fixed directories Hermes itself resolves interpreters from.

    Mirrors the hardcoded fallback candidates in
    ``tools.mcp_tool._resolve_stdio_command`` (``HERMES_HOME/node/bin``,
    ``~/.local/bin``, ``/usr/local/bin``): those are locations Hermes
    itself places or finds an interpreter at without going through a
    caller-supplied PATH, so a resolved command living there is trusted
    independent of any MCP server's own ``env.PATH``.

    Residual assumption: these dirs are user-writable, so provenance
    assumes the local user account is not itself the attacker — the
    threat model this closes is a REMOTE MCP-server config supplying a
    hostile ``env.PATH``, not a local-privilege escalation. Tightening the
    set (e.g. dropping the user-writable dirs) risks breaking legitimate
    ``npx``/``uvx`` installs and is a maintainer call, left as-is here.
    """
    hermes_home = os.path.expanduser(
        os.getenv("HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes"))
    )
    return (
        os.path.join(hermes_home, "node", "bin"),
        os.path.join(os.path.expanduser("~"), ".local", "bin"),
        os.path.join(os.sep, "usr", "local", "bin"),
    )


def _provenance_ok(expanded_path: str, basename: str) -> bool:
    """Whether *expanded_path* is trusted to actually BE *basename*, not
    merely named after it.

    A resolved absolute path earns trust two ways:

    1. It resolves (after following symlinks) into one of the fixed
       directories Hermes' own resolver places/finds interpreters in —
       see :func:`_trusted_install_dirs`. These never depend on any
       server-supplied PATH.
    2. It resolves (after following symlinks) to the SAME file the
       AMBIENT PATH — ``os.environ.get("PATH")``, this process's own
       inherited PATH, never an MCP server config's ``env.PATH``
       override — would find for *basename*. If the operator's own,
       non-attacker-controlled PATH would launch this exact binary
       anyway, a server config pointing at it directly grants no new
       capability. This is what lets Homebrew/asdf/nvm/pyenv installs on
       nonstandard PATH entries keep working without a bespoke
       per-platform trusted-path registry.
    """
    real_target = os.path.realpath(expanded_path)

    for trusted_dir in _trusted_install_dirs():
        trusted_real = os.path.realpath(trusted_dir)
        try:
            if os.path.commonpath([real_target, trusted_real]) == trusted_real:
                return True
        except ValueError:
            # No common prefix (e.g. different drives on Windows) — not a
            # match, keep checking other trusted dirs.
            continue

    ambient_hit = shutil.which(basename, path=os.environ.get("PATH"))
    if ambient_hit and os.path.realpath(ambient_hit) == real_target:
        return True

    return False


def validate_stdio_command(
    command: str,
    *,
    server_name: str = "",
    extra_allowed: "frozenset[str] | None" = None,
) -> None:
    """Validate *command* against the MCP stdio command allowlist.

    Accepts absolute paths (e.g. ``/usr/local/bin/npx``) or bare names
    (e.g. ``npx``) — the check is against the given path's basename, with
    any Windows executable suffix stripped, case-insensitively. A resolved
    ABSOLUTE path in the base interpreter set additionally must have
    trusted provenance (see :func:`_provenance_ok`) — see the module
    docstring for why a name-only check is not sufficient on its own.

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

    # Basename matched, but a command given in PATH FORM names a specific
    # file that can be a basename match while pointing at an
    # attacker-controlled binary (e.g. a malicious MCP server's env.PATH
    # resolving "npx" to "/attacker/npx" before this guard ever sees it —
    # teknium1 review on PR #62808). This covers BOTH absolute paths
    # (/attacker/npx) and relative-with-a-separator paths (./npx,
    # subdir/npx); a relative form is made absolute against the CWD first
    # so it can't skip provenance. A bare name (no separator) is exempt —
    # it names no specific file yet. extra_allowed entries are
    # call-site-owned literals, not resolved through any MCP server config,
    # so they're exempt too — see module docstring.
    #
    # "Path form" is judged with the CURRENT platform's separators
    # (os.sep/os.altsep) — the same test tools.mcp_tool._resolve_stdio_command
    # uses to tell a bare name from a path. A Windows-style path string
    # reaching this on a POSIX host (a config authored on Windows, e.g. in a
    # cross-platform test) names no file resolvable here, so it stays on the
    # name-only path; on the Windows host where it IS a real path, os.sep is
    # '\\' and it gets provenance-checked.
    separators = os.sep + (os.altsep or "")
    is_path_form = any(sep in expanded for sep in separators)
    if basename in ALLOWED_STDIO_COMMANDS and is_path_form:
        candidate = expanded if os.path.isabs(expanded) else os.path.abspath(expanded)
        if not _provenance_ok(candidate, basename):
            real_target = os.path.realpath(candidate)
            logger.error(
                "SECURITY: rejected MCP stdio command for server '%s': %r "
                "matched allowlisted basename %r but resolves to %r, which is "
                "not a trusted install location (tasks-69t.4 C2 follow-up — "
                "basename-only checks are bypassable via an attacker-controlled "
                "PATH; see teknium1 review on PR #62808).",
                server_name, command, basename, real_target,
            )
            raise DisallowedMcpCommandError(
                f"MCP server '{server_name}': command {command!r} matched "
                f"allowlisted basename {basename!r} but resolves to "
                f"{real_target!r}, which is not a trusted install location. "
                "Rejected before spawning per security policy (tasks-69t.4 C2)."
            )
