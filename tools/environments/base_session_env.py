"""Bash session-snapshot scripting for ``tools.environments.base``.

Pure string builders: the CWD marker, the ``export -p`` dump that strips
per-session vars, the ``init_session`` bootstrap, and the per-command wrapper.
No module state lives here; ``BaseEnvironment`` supplies quoting hooks.
"""

import re
import shlex
from typing import Iterable

# Bridged per-session vars (gateway.session_context._VAR_MAP) are injected fresh onto every
# command's process env and must NEVER persist in the shared bash snapshot: one long-lived
# backend serves many sessions, so a snapshot carrying the FIRST session's HERMES_SESSION_ID
# would make every LATER session source a foreign identity. Every bridged name starts with
# one of these prefixes (or is HERMES_UI_SESSION_ID); unit tests use this regex as the
# Python-side contract for the exclusion set.
# Per-session variables that the gateway bridges freshly onto every command's process environment (via
# tools/environments/local._inject_session_context_env, reading gateway.session_context._VAR_MAP). They must
# NEVER be persisted into the shared bash session snapshot: a single long-lived backend serves many
# concurrent sessions (the messaging gateway, TUI, desktop/web dashboard all collapse the terminal to one
# "default" environment), so ``export -p`` dumping the FIRST session's HERMES_SESSION_ID into the snapshot
# makes every LATER session ``source`` that stale value and see a FOREIGN session's identity — overriding
# the correct per-command Popen env (issue: cross-session HERMES_SESSION_ID leak via the shared snapshot).
# Stripping them from the snapshot is safe because they are re-injected on every command; a snapshot should
# only carry the user's own shell state (PATH, functions, exports they set), not Hermes' per-turn session
# identity. Used by unit tests as the Python-side contract for the exclusion set; the dump path unsets by
# name/prefix instead of grepping declare lines (see below / issue #71296).
_SNAPSHOT_EXCLUDED_ENV_REGEX = (
    "^declare -x (HERMES_SESSION_|HERMES_UI_SESSION_ID|HERMES_CRON_AUTO_DELIVER_|"
    "HERMES_CRON_SESSION|HERMES_DELEGATED_CHILD_CONTEXT|"
    "HERMES_RPC_|HERMES_KERNEL_DIR)")
_SHELL_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# --- secret-shaped env names (Viktor credential-exposure findings 1-3, CWE-312) -----------
# Single source of truth: BOTH the in-loop `unset` (case globs, matched against a
# case-normalised name) and the post-dump fail-closed net (POSIX ERE) are generated from
# this table, so widening one can never leave the other behind. Shape kinds:
#   "exact"    — the whole name equals the token (bare `PASSWORD`)
#   "suffix"   — the name ends with the token (`GITHUB_TOKEN`)
#   "contains" — the token appears anywhere in the name (`GCP_CREDENTIALS`)
# The leading underscore on most suffix tokens is deliberate: `*_KEY` must NOT match
# `MY_MONKEY` (positive control — a scrub that eats ordinary vars gets switched off).
_SECRET_NAME_SHAPES: tuple[tuple[str, str], ...] = (
    ("suffix", "_TOKEN"), ("suffix", "_KEY"), ("suffix", "_SECRET"), ("suffix", "_PASSWORD"),
    ("suffix", "_PASSWD"), ("suffix", "_PASS"), ("suffix", "_AUTH"), ("suffix", "_BEARER"),
    ("suffix", "_KEY_ID"), ("suffix", "_DSN"),
    # No-separator compound names (`STRIPE_APIKEY`, `GITHUB_AUTHTOKEN`, `SSH_PRIVATEKEY`).
    ("suffix", "APIKEY"), ("suffix", "AUTHTOKEN"), ("suffix", "PRIVATEKEY"),
    ("contains", "_CREDENTIAL"),
    ("exact", "PASSWORD"), ("exact", "PASSWD"), ("exact", "PASS"),
    ("exact", "TOKEN"), ("exact", "KEY"), ("exact", "SECRET"),
)


def _secret_name_glob_pattern() -> str:
    """``case``-glob alternation for an already UPPER-CASED name (bash 3.2 and 5 alike)."""
    return "|".join(
        token if kind == "exact" else f"*{token}" if kind == "suffix" else f"*{token}*"
        for kind, token in _SECRET_NAME_SHAPES)


_SECRET_NAME_ERE = "(" + "|".join(
    token if kind == "exact"
    else f"[A-Za-z0-9_]*{token}" if kind == "suffix"
    else f"[A-Za-z0-9_]*{token}[A-Za-z0-9_]*"
    for kind, token in _SECRET_NAME_SHAPES) + ")"

# A line `export -p` can emit for a secret-shaped exported variable. Covers the readonly
# form (`declare -rx`, which no shell-level unset can remove) and builds whose `export -p`
# prints `export NAME=`. Matched case-insensitively, because a name lower-case in the
# environment is still the same secret. The `=` anchor keeps the match on the declaration
# line only, so continuation lines of a multi-line value are never mistaken for one.
_SNAPSHOT_SECRET_DECL_ERE = rf"^(declare -[A-Za-z]*x[A-Za-z]*|export) {_SECRET_NAME_ERE}="

# mktemp template suffix + the shell variable holding the allocated temp path.
_SNAP_TMP_SUFFIX = ".tmp.XXXXXXXXXX"
_SNAP_TMP = '"$__hermes_snap_tmp"'


def _secret_scrub_snippet() -> str:
    """Unset secret-shaped exports before ``export -p`` writes the snapshot.

    Case normalisation goes through ``tr`` deliberately: ``${var^^}`` is a **bash 4.0**
    operator and this module supports the macOS bash 3.2 baseline ("``${!PREFIX*}`` is
    bash 3.2+", "macOS bash 3.2 lacks ``$BASHPID``"). On 3.2 it raises *bad
    substitution* on every iteration, ``unset`` never runs, and the caller's ``|| true``
    turns that into a silently published full secret dump (Viktor Finding 1, CWE-755).
    The name rules come from ``_SECRET_NAME_SHAPES`` so the net below can never drift
    narrower than this loop.

    The value-shape rule drops connection strings carrying credentials in the URL
    (``postgres://user:pw@host/db``) whatever the variable is called, which no name
    pattern can catch (Finding 3). ``${!name-}`` indirect expansion is bash 3.2-safe and
    survives a caller's ``set -u``.
    """
    return (
        "for __hermes_sv in $(compgen -A export 2>/dev/null); do "
        '__hermes_sv_up=$(printf %s "$__hermes_sv" | tr "[:lower:]" "[:upper:]"); '
        f'case "$__hermes_sv_up" in {_secret_name_glob_pattern()}) unset "$__hermes_sv" ;; esac; '
        'case "${!__hermes_sv-}" in *://*:*@*) unset "$__hermes_sv" ;; esac; '
        "done; "
    )


def _snapshot_publish_guard(tmp_path: str) -> str:
    """Fail-closed net over the assembled snapshot: run after the dump, before the ``mv``.

    The in-loop scrub is best-effort by construction. A ``readonly`` exported secret
    cannot be removed at all (``unset`` *and* ``declare +r`` are refused on it), and a
    broken ``tr``/``compgen`` degrades the loop to a no-op — neither failure is
    observable from inside the loop. So the last line of defence reads the finished
    snapshot and, on any secret-shaped declaration, replaces it with a no-op stub
    instead of publishing it. Losing the persisted shell state is the deliberate trade:
    a snapshot is re-derived on the next command, an at-rest credential is not.

    A *broken* ``grep`` (rc > 1) counts as a hit too — every unknown resolves to
    "withhold", never to "publish". Only diagnostic output carries names; values are
    never echoed. The reason is also written into the stub as a comment, because the
    per-command wrapper silences the group's stderr.

    ``grep`` sits in an ``if`` condition on purpose: "no secrets found" exits 1, and a
    bare ``grep -q`` in a profile that ran ``set -e`` would kill the login shell before
    the ``mv`` — turning a clean snapshot into no snapshot at all.
    """
    q_ere = shlex.quote(_SNAPSHOT_SECRET_DECL_ERE)
    leak_names = (
        f"grep -Ei {q_ere} {tmp_path} 2>/dev/null | "
        "sed -E 's/^(declare -[A-Za-z]*x[A-Za-z]*|export) ([A-Za-z0-9_]+)=.*/\\2/' | "
        "paste -sd' ' -")
    return (
        f"if grep -Eiq {q_ere} {tmp_path} 2>/dev/null; then __hermes_sv_rc=0; "
        "else __hermes_sv_rc=$?; fi; "
        'if [ "$__hermes_sv_rc" -ne 1 ]; then '
        f"__hermes_sv_names=$({leak_names}); "
        '__hermes_sv_msg="hermes: snapshot withheld (secret-shaped declarations survived '
        'the scrub: ${__hermes_sv_names:-unknown})"; '
        'echo "$__hermes_sv_msg" >&2; '
        f"""printf '%s\\n' ':' "# $__hermes_sv_msg" 'set +e' 'set +u' > {tmp_path}; """
        "fi")


def _cwd_marker(session_id: str) -> str:
    return f"__HERMES_CWD_{session_id}__"


def _pipe_marker(session_id: str) -> str:
    return f"__HERMES_PIPE_{session_id}__"


def _pipe_marker_printf(marker: str) -> str:
    """Emit the pipeline's per-segment exit statuses (``PIPESTATUS``) on their own line.

    ``cmd | tail`` exits with tail's status, so the shell's code cannot prove ``cmd``
    passed. The array holds each segment's own status and is captured before anything
    else runs.
    """
    return f"printf '\\n{marker}%s{marker}\\n' \"${{__hermes_ps[*]-}}\""


def _eval_capturing_pipestatus(escaped: str) -> str:
    """``eval`` the command and keep its last pipeline's ``PIPESTATUS`` as side data.

    The array must be read inside the same ``eval`` (afterwards it holds only ``eval``'s
    status), while ``$?`` stays the exit code: the two differ for ``!``, ``if``/``while``,
    ``pipefail`` and parse errors. The blank line stops the capture joining a command
    continued by a trailing backslash. The restore sits on the left of ``&&`` so its
    non-zero status fires neither a user's ``ERR`` trap a second time nor ``set -e``.
    """
    return f"eval '{escaped}\n\n__hermes_rc=$? __hermes_ps=(\"${{PIPESTATUS[@]}}\"); (exit $__hermes_rc) && :'"


def _can_capture_pipestatus(command: str) -> bool:
    """An unterminated heredoc would read the capture line as its body (into the user's
    file), and a trailing ``\\`` would join onto it: such commands run unchanged."""
    return "<<" not in command and not command.endswith("\\")


def _cwd_marker_printf(marker: str) -> str:
    """Emit the CWD marker on its own line (leading ``\\n`` guards against a
    command whose output lacks a trailing newline; ``_split_cwd_marker`` strips it)."""
    return f"printf '\\n{marker}%s{marker}\\n' \"$(pwd -P)\""


def _export_dump_excluding_session_vars(tmp_path: str, excluded_names: Iterable[str] = ()) -> str:
    """Shell snippet dumping ``export -p`` to *tmp_path* minus the per-session bridged vars (see
    ``_SNAPSHOT_EXCLUDED_ENV_REGEX``) and *excluded_names*. The vars are ``unset`` in a subshell
    BEFORE ``export -p``: a line-based ``grep -vE`` is unsafe because bash 3.2 prints a value
    containing a newline as a multi-line ``declare -x`` block, so smuggled continuation lines would
    survive into the snapshot and execute on the next ``source``. ``|| true`` keeps the success
    contract. The dump is a brace group with the redirection on the group: *tmp_path* is usually a
    shell-variable expansion, and a redirect on a pipeline segment would expand it inside that
    segment's subshell, inconsistently with the parent that expands the follow-up ``mv``.

    ``curl … | bash #`` smuggled into a Matrix room/display name via ``HERMES_SESSION_CHAT_NAME``) land in
    the snapshot and execute on the next ``source`` (issue #71296). Unsetting first means ``export -p``
    never emits those vars — including any continuation lines.

    Secret-shaped vars (see ``_SECRET_NAME_SHAPES``: ``*_TOKEN``, ``*_KEY``, ``*_SECRET``,
    ``*_PASSWORD``, ``*_PASS(WD)``, ``*_AUTH``, ``*_BEARER``, ``*APIKEY``, ``*_CREDENTIAL*``,
    ``*_KEY_ID``, ``*_DSN``, bare ``PASSWORD``/``KEY``/``SECRET`` …, case-insensitive, plus
    any value shaped like a credential-bearing URL) are unset the same way (Viktor
    credential-exposure Finding 3, CWE-312): the terminal session snapshot lives at-rest on
    disk (``~/.hermes/cache/scratch/hermes-snap-*.sh``, world-unreadable but
    agent-readable) and a value left in it would be re-exported into every later command on
    that backend. "Report names, never values" — the snapshot omits the value entirely
    rather than redacting it in place, since a shared bash snapshot sourced by a LATER
    command must never re-assert a stale secret an outer harness no longer wants live.

    This loop cannot be the whole control: readonly exports survive it and a broken
    ``tr``/``compgen`` degrades it to a no-op, so every caller must run
    ``_snapshot_publish_guard`` on the assembled file before publishing it.
    """
    # ${!PREFIX*} is bash 3.2+ name-prefix expansion; empty matches are ignored
    # under 2>/dev/null. Caller names are quoted so malformed config can never
    # become shell syntax (valid names stay unquoted by shlex.quote()).
    safe_names = {name for name in excluded_names if isinstance(name, str) and name}
    extra_unset = "".join(f" {shlex.quote(name)}" for name in sorted(safe_names))
    secret_scrub = _secret_scrub_snippet()
    return (
        "{ ( unset ${!HERMES_SESSION_*} ${!HERMES_CRON_AUTO_DELIVER_*} "
        # AI_AGENT / HERMES_AGENT are per-command attribution markers re-exported
        # by every wrapper with ${VAR:-default} semantics; persisting them would
        # let the FIRST command's value override a later outer-harness value.
        "AI_AGENT HERMES_AGENT "
        # Scope markers stamped onto a delegate_task child's / cron run's subprocess
        # env; a snapshot taken inside that window would re-assert them on every
        # later ``source`` and fence the PARENT session's kanban CLI (#90782).
        "HERMES_DELEGATED_CHILD_CONTEXT HERMES_CRON_SESSION "
        # Remote code-execution channel vars (RPC token, kernel/rpc dirs): a
        # leaked token in the snapshot would re-export into every later command
        # on the backend and outlive the private dir it protects.
        "${!HERMES_RPC_*} HERMES_KERNEL_DIR "
        f"HERMES_UI_SESSION_ID{extra_unset} 2>/dev/null; "
        f"{secret_scrub}"
        "export -p; ) || true; } "
        f"> {tmp_path}")


def _snapshot_bootstrap_script(
    *, quoted_cwd: str, quoted_snap: str, snap_tmp_template: str, excluded_names: Iterable[str], cwd_marker: str,
) -> str:
    """Login-shell bootstrap that captures env/functions/aliases into the snapshot. Atomic publish:
    assemble in a ``mktemp`` file, then ``mv`` over the final path so a concurrent ``source`` never
    reads a half-written snapshot (``$$`` is the parent PID in ``&``-launched subshells and macOS
    bash 3.2 lacks ``$BASHPID``, so only ``mktemp`` is portable). Functions are filtered by NAME via
    ``declare -F`` (a line-based ``declare -f | grep -v`` strips the header and leaves an orphaned
    body that breaks every sourced command); the non-empty guard matters because bare ``declare -f``
    dumps ALL functions. The assembled file passes through ``_snapshot_publish_guard`` immediately
    before the ``mv`` publishes it, so a snapshot still holding a secret-shaped declaration (a
    readonly export, a scrub that failed) is replaced by a no-op stub instead of being published.
    The trailing ``cd`` restores the configured cwd after profile scripts (e.g.
    ``cd ~``) so ``pwd -P`` reports terminal.cwd, not the profile's directory."""
    return (
        "umask 077\n"
        f"__hermes_snap_tmp=$(mktemp {snap_tmp_template}) || exit 1\n"
        f"{_export_dump_excluding_session_vars(_SNAP_TMP, excluded_names)}\n"
        "__hermes_fns=$(declare -F | awk '{print $3}' | grep -vE '^_[^_]') || true\n"
        f"[ -n \"$__hermes_fns\" ] && declare -f $__hermes_fns >> {_SNAP_TMP} 2>/dev/null || true\n"
        f"alias -p >> {_SNAP_TMP}\n"
        f"echo 'shopt -s expand_aliases' >> {_SNAP_TMP}\n"
        f"echo 'set +e' >> {_SNAP_TMP}\n"
        f"echo 'set +u' >> {_SNAP_TMP}\n"
        # Fail-closed: a snapshot holding a secret-shaped declaration (readonly export,
        # broken scrub) is replaced by a no-op stub rather than published.
        f"{_snapshot_publish_guard(_SNAP_TMP)}\n"
        # Publish only if assembly succeeded; otherwise drop the partial temp.
        f"mv -f {_SNAP_TMP} {quoted_snap} || rm -f {_SNAP_TMP}\n"
        f"builtin cd -- {quoted_cwd} 2>/dev/null || true\n"
        f"{_cwd_marker_printf(cwd_marker)}\n")


def _passthrough_save_restore(names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Shell lines that save profile-scoped passthrough vars before the snapshot is sourced
    and restore (or unset) them afterwards — a shared snapshot may hold the previous
    profile's value. Values stay in environment memory and never enter the command string."""
    save: list[str] = []
    restore: list[str] = []
    for name in names:
        marker = f"_HERMES_RUNTIME_PASSTHROUGH_{name}"
        present, value = f"{marker}_PRESENT", f"{marker}_VALUE"
        save += [f"{present}=${{{name}+x}}", f"{value}=${{{name}-}}"]
        restore += [
            f'if [ "${present}" = x ]; then export {name}="${value}"; else unset {name}; fi',
            f"unset {present} {value}"]
    return save, restore


def _wrap_command_script(
    command: str, *, quoted_cwd: str, quoted_snap: str, snap_tmp_template: str,
    passthrough_names: Iterable[str], snapshot_ready: bool, cwd_marker: str,
    pipe_marker: str = "", persist_env: bool = True, seed_snaps: Iterable[str] = ()) -> str:
    """Per-command bash script: source snapshot, cd, run, re-dump env, emit CWD marker.
    ``persist_env=False`` skips the re-dump, so every command starts from the login-time
    snapshot and exports from one command never reach the next (the CWD marker still runs).
    ``source`` stdout goes to /dev/null because macOS bash 3.2 / some Homebrew builds echo
    ``declare -x`` lines when sourcing. AI_AGENT/HERMES_AGENT advertise the harness to remote
    backends (whose env is not inherited); ``${VAR:-default}`` never clobbers an outer harness.
    GIT_PAGER/PAGER=cat stop pager-happy tools hanging a PTY-backed command. The env re-dump
    uses the same mktemp+mv atomic publish as the bootstrap and chains ``mv`` on the dump
    succeeding so a failed dump never replaces a good snapshot, and on
    ``_snapshot_publish_guard`` passing so a re-dump that still holds a secret-shaped
    declaration publishes a no-op stub instead (its stderr is silenced with the rest of the
    group, so the guard's reason lands in the stub as a comment). ``umask 077`` is applied after
    the user's command so snapshot files (which may carry secrets) are private without
    changing the command's umask. *seed_snaps* (quoted paths) are read, first existing wins,
    while *quoted_snap* does not exist yet: a delegated child's own snapshot starts as a copy of
    its parent's shell state and its re-dump never writes the parent's file.
    """
    escaped = command.replace("'", "'\\''")
    capture = bool(pipe_marker) and _can_capture_pipestatus(command)
    save, restore = _passthrough_save_restore(passthrough_names)
    parts = list(save)
    seeds = list(seed_snaps)
    if snapshot_ready and seeds:
        parts.append(
            f"for __hermes_s in {quoted_snap} {' '.join(seeds)}; do "
            'if [ -f "$__hermes_s" ]; then source "$__hermes_s" >/dev/null 2>&1 || true; break; fi; done')
    elif snapshot_ready:
        parts.append(f"source {quoted_snap} >/dev/null 2>&1 || true")
    parts += restore
    parts += [
        'export AI_AGENT="${AI_AGENT:-hermes-agent}" HERMES_AGENT="${HERMES_AGENT:-true}"',
        'export GIT_PAGER="${GIT_PAGER:-cat}" PAGER="${PAGER:-cat}"',
        # ``--`` keeps hyphen-prefixed directory names from being parsed as options.
        f"builtin cd -- {quoted_cwd} || exit 126",
        _eval_capturing_pipestatus(escaped) if capture else f"eval '{escaped}'",
        "__hermes_ec=$?",
        "umask 077"]
    if snapshot_ready and persist_env:
        parts.append(
            f"__hermes_snap_tmp=$(mktemp {snap_tmp_template}) && "
            f"{{ {_export_dump_excluding_session_vars(_SNAP_TMP, passthrough_names)} "
            f"&& {_snapshot_publish_guard(_SNAP_TMP)} "
            f"&& mv -f {_SNAP_TMP} {quoted_snap}; }} "
            f"2>/dev/null || rm -f {_SNAP_TMP} 2>/dev/null || true")
    if capture:
        parts.append(_pipe_marker_printf(pipe_marker))
    parts += [_cwd_marker_printf(cwd_marker), "exit $__hermes_ec"]
    return "\n".join(parts)


def _split_cwd_marker(output: str, marker: str) -> tuple[str | None, str] | None:
    """Locate the last ``marker<path>marker`` pair in *output*. Returns
    ``(cwd_path_or_None, output_without_marker_line)``, or ``None`` when no complete pair
    exists. The stripped span runs from the ``\\n`` the wrapper injected before the marker
    through the end of the marker line."""
    last = output.rfind(marker)
    if last == -1:
        return None
    search_start = max(0, last - 4096)  # CWD path won't be >4KB
    first = output.rfind(marker, search_start, last)
    if first == -1 or first == last:
        return None
    cwd_path = output[first + len(marker) : last].strip() or None
    line_start = output.rfind("\n", 0, first)
    if line_start == -1:
        line_start = first
    line_end = output.find("\n", last + len(marker))
    line_end = line_end + 1 if line_end != -1 else len(output)
    return cwd_path, output[:line_start] + output[line_end:]
