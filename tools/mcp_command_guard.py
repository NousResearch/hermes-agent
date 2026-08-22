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

Provenance for a per-call-site ``extra_allowed`` widening (e.g. cua-driver,
see below) is checked too, but against a DIFFERENT trust source: the base
interpreter set trusts Hermes' own install dirs or the ambient PATH (see
:func:`_provenance_ok`), because those names are common and any of many
installs is legitimate. An ``extra_allowed`` name is call-site-specific and
usually has exactly one legitimate identity, so its caller must pass
``extra_trusted_paths`` — the exact resolved, already-trusted absolute
path(s) that name is allowed to be — and a PATH-FORM ``extra_allowed``
command is rejected unless it realpath-matches one of them. (Earlier this
module exempted ``extra_allowed`` from provenance entirely on the theory
that it is "a fixed, call-site-owned literal the operator's own environment
resolves" — but cua_backend.py's usage passes a value *read back from a
subprocess's manifest output*, not a literal, so a compromised or
misbehaving manifest could name any absolute path and inherit the
exemption; egilewski review on PR #62808.) A bare ``extra_allowed`` name
(no path separator) stays exempt, same as a bare interpreter name.

:func:`validate_stdio_command` returns the exact command string it
validated — for a PATH-FORM command this is the CANONICAL ABSOLUTE form it
ran provenance checks against, not necessarily the string passed in.
Callers must use the RETURNED value in ``StdioServerParameters``, not the
original argument: a relative command is validated by absolutizing it
against the process's CURRENT working directory, and if the caller instead
spawned the original (still-relative) string later — after other work
(e.g. an ``await``, during which the process cwd can move: ``cli.py``'s
mid-chat ``/resume``/``/sessions <id>`` handling calls ``os.chdir()`` on
the SAME event loop an MCP stdio server's reconnect loop runs on; unlike
that path, ``cron/scheduler.py`` deliberately never mutates the process
cwd for exactly this reason, #69396) — the relative string could resolve
to a different file at spawn time than the one that was actually checked
(egilewski review on PR #62808). Returning the resolved value removes the
gap instead of relying on the cwd staying put.

Only the ``command`` (the interpreter/launcher binary) is validated here —
not ``args``. A server config can still pass e.g. ``docker`` with
``args=["run", "--privileged", ...]``; the allowlist bounds WHICH BINARY
launches, not what an allowed launcher is told to do, so it does not by
itself bound blast radius for launchers like ``docker`` that are
general-purpose by design (Enough1122 review on PR #62808).

Provenance for the base interpreter set trusts a fixed list of install
dirs (see :func:`_trusted_install_dirs`) plus whatever the ambient PATH
resolves. A project-local, pinned toolchain launcher (e.g.
``node_modules/.bin/npx``, a common pattern for reproducible builds) is
NOT one of those and will be rejected when referenced by its own absolute
path, even though it is a legitimate npx install (Enough1122 review on PR
#62808) — this is a known compatibility tradeoff of a fixed trusted-dir
allowlist, not a bug; point such a server config at the project's actual
node/npx via the ambient PATH (or a trusted install dir) instead. The same
tradeoff applies to a BARE interpreter (see the ``resolved_path`` param
below): a server config naming e.g. ``python`` with ``env.PATH`` pointing
at a project virtualenv's ``bin/`` — the common way to target a specific
venv interpreter — is rejected once this resolves and provenance-checks
it, for the same reason a venv is not itself a trusted install dir or on
the ambient PATH; same workaround.

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
        # Fail open (allowlist stays disabled) rather than crashing MCP
        # startup over a config read failure -- but a security control that
        # silently disables itself on a syntax error or permissions issue
        # (Enough1122 review on PR #62808) must not do so quietly. Warn so
        # an operator who believes the allowlist is on can notice it isn't.
        logger.warning(
            "MCP stdio command allowlist: could not read config.yaml to "
            "check security.mcp_stdio_command_allowlist_enabled; treating "
            "it as disabled. Fix the config error above if you expect the "
            "allowlist to be active.",
            exc_info=True,
        )
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
    extra_trusted_paths: "frozenset[str] | None" = None,
    resolved_path: "str | None" = None,
) -> str:
    """Validate *command* against the MCP stdio command allowlist.

    Accepts absolute paths (e.g. ``/usr/local/bin/npx``) or bare names
    (e.g. ``npx``) — the check is against the given path's basename, with
    any Windows executable suffix stripped, case-insensitively. A resolved
    PATH-FORM command (absolute, or relative with a separator) in the base
    interpreter set additionally must have trusted provenance (see
    :func:`_provenance_ok`) — see the module docstring for why a name-only
    check is not sufficient on its own.

    ``extra_allowed`` lets a specific call site widen the allowlist for a
    fixed, non-configurable launcher that isn't an interpreter (e.g.
    ``cua-driver``, a compiled native binary cua_backend.py spawns directly
    rather than through python/node). It is deliberately NOT a general
    escape hatch — only pass a literal frozenset of exact binary names the
    call site owns and controls, never anything derived from user input.
    A PATH-FORM ``extra_allowed`` command additionally requires
    ``extra_trusted_paths`` — see that parameter and the module docstring
    (egilewski review on PR #62808: an unconditional exemption here let an
    arbitrary same-named executable through).

    ``extra_trusted_paths``, when given, is a frozenset of exact absolute
    paths (not directories) that a PATH-FORM ``extra_allowed`` command may
    realpath-match to be accepted — e.g. the call site's own
    already-resolved, already-trusted binary. Ignored for the base
    interpreter set, which uses :func:`_provenance_ok` instead. Has no
    effect on bare (separator-free) command names, which are exempt from
    provenance for both the base set and ``extra_allowed``.

    ``resolved_path``, when given, is the PATH string the subprocess env
    will actually carry (e.g. the server config's own ``env.PATH``). A
    bare command normally skips provenance entirely, on the theory that
    it names no specific file yet (see below) — but when this argument is
    given, that theory no longer holds: this function resolves the bare
    name itself, via ``shutil.which`` against this exact PATH, and treats
    the hit exactly like a PATH-FORM command below (full provenance
    check). An unresolved name is rejected outright, not returned bare,
    because it has not actually been checked (egilewski + round-6 panel
    reviews on PR #62808: reachable whenever
    ``tools.mcp_tool._resolve_stdio_command`` can't resolve the bare name
    itself first, e.g. an allowlisted interpreter other than npx/npm/node
    with no fallback candidates, or any allowlisted name given a broken or
    attacker-influenced subprocess PATH). Two narrower attempts at this —
    rejecting a relative PATH entry, then also an empty one — both missed
    an absolute-but-untrusted entry that had no file in it yet at
    validation time; resolving here instead of pattern-matching PATH
    entries closes the whole class rather than one more shape of it.
    Omitting this argument leaves a bare command exempt, unchanged from
    before this parameter existed.

    Returns the validated command string, fully resolved: for a PATH-FORM
    command this is ``os.path.realpath`` of the checked path — the SAME
    file provenance was actually verified against, symlinks followed —
    not merely the absolutized original. Returning the symlink path
    instead would leave a window where the symlink is repointed after
    the check but before spawn; spawning the realpath closes it outright.
    (Known tradeoff: a wrapper that keys behavior off ``argv[0]``/``$0``
    matching the invoking symlink name, rather than its own real
    location, would see a different value. No launcher in the base
    interpreter set or ``extra_allowed`` is known to do this.) A bare
    name is returned unchanged when ``resolved_path`` is omitted; when
    given, a bare name that resolves is returned as the same realpath a
    PATH-FORM command would be (see ``resolved_path`` above).

    Raises :class:`DisallowedMcpCommandError` so callers fail loudly
    instead of silently proceeding with a bad command. A security error is
    always logged before raising.
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
    # it names no specific file yet.
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

    if is_path_form:
        candidate = expanded if os.path.isabs(expanded) else os.path.abspath(expanded)
    elif resolved_path is None:
        # Bare name, no PATH to check it against: legacy call shape,
        # unchanged behavior -- returned exempt from provenance below.
        return expanded
    else:
        # Bare name WITH a subprocess PATH to check: resolve it ourselves,
        # right here, against the exact PATH string the subprocess will
        # carry, and require a hit. Two narrower checks tried instead
        # (reject a relative PATH entry; reject an empty one too) both
        # missed a case: an ABSOLUTE-but-untrusted PATH entry that simply
        # has no file in it yet at validation time still passed, and an
        # attacker able to write into it before spawn got the same
        # exec-time bypass this whole mechanism exists to close (round-6
        # panel, PR #62808). An unresolved bare name has not actually been
        # checked at all, so it is rejected outright instead -- what
        # resolves here falls through to the SAME provenance check below
        # as a path-form command, so a resolved-but-untrusted hit is still
        # rejected, not merely a missing one.
        which_hit = shutil.which(basename, path=resolved_path)
        if not which_hit:
            logger.error(
                "SECURITY: rejected MCP stdio command for server '%s': "
                "%r is a bare command that does not resolve against the "
                "subprocess PATH %r -- an unresolved bare command is not "
                "checkable and is rejected before spawning rather than "
                "left to the OS exec call (round-6 panel, PR #62808).",
                server_name, command, resolved_path,
            )
            raise DisallowedMcpCommandError(
                f"MCP server '{server_name}': command {command!r} is bare "
                f"and does not resolve against PATH {resolved_path!r} -- "
                "rejected before spawning per security policy "
                "(tasks-69t.4 C2)."
            )
        candidate = os.path.abspath(which_hit)

    real_target = os.path.realpath(candidate)

    if basename in ALLOWED_STDIO_COMMANDS:
        provenance_ok = _provenance_ok(candidate, basename)
    else:
        # extra_allowed, in PATH FORM: unlike the base interpreter set,
        # this name is call-site-specific and usually has exactly one
        # legitimate identity, so it is trusted only if it realpath-matches
        # a path the CALLER has already resolved and vouches for — never
        # exempted outright (egilewski review on PR #62808: an
        # unconditional exemption here let an arbitrary same-named
        # executable through, e.g. a compromised cua-driver manifest
        # naming an arbitrary absolute path).
        trusted_targets = {
            os.path.realpath(p) for p in (extra_trusted_paths or frozenset())
        }
        provenance_ok = real_target in trusted_targets

    if not provenance_ok:
        logger.error(
            "SECURITY: rejected MCP stdio command for server '%s': %r "
            "matched allowlisted basename %r but resolves to %r, which is "
            "not a trusted install location (tasks-69t.4 C2 follow-up — "
            "basename-only checks are bypassable via an attacker-controlled "
            "PATH; see teknium1/egilewski reviews on PR #62808).",
            server_name, command, basename, real_target,
        )
        raise DisallowedMcpCommandError(
            f"MCP server '{server_name}': command {command!r} matched "
            f"allowlisted basename {basename!r} but resolves to "
            f"{real_target!r}, which is not a trusted install location. "
            "Rejected before spawning per security policy (tasks-69t.4 C2)."
        )

    return real_target
