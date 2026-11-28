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
# The names/prefixes that carry session identity across the bash handoff: each command re-injects this
# session's values, so they are excluded from the shared snapshot (see below) and, in
# ``agent.session_client_env``, refused from a client even when the profile lists them (core owns them).
# Entries ending in ``_`` are prefixes. The regex below is the Python-side contract tests check; the
# dump path unsets by name/prefix (see below / issue #71296).
_SNAPSHOT_BRIDGE_ENV = ('HERMES_SESSION_', 'HERMES_UI_SESSION_ID', 'HERMES_CRON_AUTO_DELIVER_',
                        'HERMES_CRON_SESSION', 'HERMES_DELEGATED_CHILD_CONTEXT',
                        'HERMES_RPC_', 'HERMES_KERNEL_DIR')
_SNAPSHOT_EXCLUDED_ENV_REGEX = "^declare -x (" + "|".join(_SNAPSHOT_BRIDGE_ENV) + ")"
_SHELL_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# mktemp template suffix + the shell variable holding the allocated temp path.
_SNAP_TMP_SUFFIX = ".tmp.XXXXXXXXXX"
_SNAP_TMP = '"$__hermes_snap_tmp"'

# A backend with an in-memory secret channel (``LocalEnvironment``) gets ``_secret_fd_token`` in
# the scripts it runs and swaps in the channel's fd number at spawn; the dump writes each secret
# export there (``NAME=value\0`` records, then one empty record) instead of to the file on disk.
# Names whose value is a credential by convention. A false positive costs nothing: the export
# still reaches the next command through the channel, it is only kept out of the file on disk.
_SECRET_SHAPED_NAME_GLOBS = (
    "*_TOKEN", "*_TOKENS", "TOKEN", "*AUTHTOKEN", "*_KEY", "KEY", "*APIKEY", "*PRIVATEKEY", "*_KEY_ID",
    "*_ACCESS_KEY*", "*_CLIENT_ID", "*_SECRET", "*_SECRET_*", "SECRET", "*PASSWORD", "*_PASSWD", "PASSWD",
    "*_PASS", "PASS", "*_PWD", "*_AUTH", "*_BEARER", "*_DSN", "*_CREDENTIAL*", "*_CREDS*", "*COOKIE*",
    "*_WEBHOOK*")
# Names whose value may embed ``user:pass@`` (``DATABASE_URL=postgres://u:pw@db/app``): only these
# have their VALUE checked, so the dump never reads every variable's value.
_URL_SHAPED_NAME_GLOBS = ("*URL*", "*URI*", "*_DSN*", "*PROXY*", "*DATABASE*", "*CONN*", "*_HOST*")
# A cheap superset of both lists for ``compgen -X``: bash tests every export against every pattern
# there, and the exact lists above as per-letter classes cost ~5 ms of CPU per command. The loop
# then applies the exact lists to the few names this lets through.
_CANDIDATE_NAME_GLOBS = (
    "*TOKEN*", "*KEY*", "*SECRET*", "*PASS*", "*PWD*", "*AUTH*", "*BEARER*", "*DSN*", "*CRED*",
    "*COOKIE*", "*WEBHOOK*", "*_CLIENT_ID", "*UR[LI]*", "*PROXY*", "*DATABASE*", "*CONN*", "*_HOST*")


def _secret_fd_token(session_id: str) -> str:
    return f"__HERMES_SFD_{session_id}__"


def _secret_gen_token(session_id: str) -> str:
    """Placeholder the backend swaps for a fresh dump generation id at each spawn."""
    return f"__HERMES_SGEN_{session_id}__"


def _secret_src_token(session_id: str) -> str:
    """Placeholder the backend swaps for the fd of the snapshot it opened for this command."""
    return f"__HERMES_SSRC_{session_id}__"


def _secret_restore_token(session_id: str) -> str:
    """Placeholder the backend swaps for the fd of the pipe carrying that snapshot's secrets."""
    return f"__HERMES_SRST_{session_id}__"


def _cwd_marker(session_id: str) -> str:
    return f"__HERMES_CWD_{session_id}__"


def _cwd_fallback_marker(session_id: str) -> str:
    return f"__HERMES_CWDFB_{session_id}__"


def _cd_with_fallback(quoted_cwd: str, fallback_marker: str) -> str:
    """The wrapper's ``cd``. A cwd removed inside the sandbox made it ``exit 126`` on every later
    command (#62169); instead walk up to the nearest existing ancestor, then ``$HOME``, then ``/``
    (the rule ``LocalEnvironment._usable_cwd`` applies host-side), and print where it landed so
    ``execute`` reports it as ``cwd_fallback``. ``${d%/*}`` walks without needing ``dirname``.
    Without *fallback_marker* (an explicit ``workdir``) a missing directory is refused."""
    if not fallback_marker:
        return f"builtin cd -- {quoted_cwd} || exit 126"
    return (
        f"builtin cd -- {quoted_cwd} 2>/dev/null || {{ __hermes_d={quoted_cwd}\n"
        'while [ -n "$__hermes_d" ] && ! builtin cd -- "$__hermes_d" 2>/dev/null; do\n'
        '  case $__hermes_d in /*/*|[!/]*/*) __hermes_d=${__hermes_d%/*};; /?*) __hermes_d=/;; *) __hermes_d=;; esac\n'
        'done\n'
        '[ -n "$__hermes_d" ] || builtin cd -- "$HOME" 2>/dev/null || builtin cd / || exit 126\n'
        f"printf '\\n{fallback_marker}%s{fallback_marker}\\n' \"$(pwd -P)\"; }}")


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


def _any_case(glob: str) -> str:
    """*glob* with every letter as a ``[Xx]`` class (bracket expressions kept as written, then
    both cases of their letters), so ``compgen -X`` matches a name in any case: bash 3.2's
    ``compgen -X`` ignores ``nocasematch``, which ``case`` honours on every version."""
    return re.sub(r"\[([^]]*)\]|([A-Za-z])", lambda m: (
        f"[{''.join(dict.fromkeys(c for ch in m[1] for c in (ch.upper(), ch.lower())))}]" if m[1] is not None
        else f"[{m[2].upper()}{m[2].lower()}]"), glob)


def _secret_handoff_snippet(secret_fd: str, secret_names: Iterable[str], generation: str) -> str:
    """Dump-subshell lines that send each secret export to fd *secret_fd* and drop it, so the
    ``export -p`` after them never puts the value in the file on disk. Each record is
    ``NAME=<declare -p NAME>\\0`` (bash's own re-readable declaration, so arrays and the
    readonly/integer/case attributes survive), one empty record closes the set, before the file is
    published. The file's first line is ``__hermes_sgen=<generation>``: the backend opens the file
    the next command will source, reads that line and hands the shell the open file plus that
    dump's declarations only, so the two always come from the same dump.
    A secret is a name the profile's ``.env``/secret sources define or the session ever held
    (*secret_names*), a credential-shaped name (``_SECRET_SHAPED_NAME_GLOBS``, any case) or a
    URL-shaped name whose value holds ``user:pass@``. One ``compgen -X`` picks the candidates inside
    bash and every write is a builtin redirect, so the shell forks nothing per variable: the per-name
    ``tr`` subshell of #148406 made every command ~25x slower. ``export -n`` also clears a readonly
    export, which ``unset`` refuses. A failed write still drops the value and withholds the closing
    record, so the backend keeps nothing from that dump (fail closed: lost, never on disk). IFS /
    ``set -e`` are pinned: the user's command may have changed either, or shadowed a command name."""
    names = sorted({n for n in secret_names if isinstance(n, str) and _SHELL_ENV_NAME_RE.fullmatch(n)})
    secret = "|".join((*_SECRET_SHAPED_NAME_GLOBS, *names))
    url = "|".join(_URL_SHAPED_NAME_GLOBS)
    candidates = "|".join((*map(_any_case, _CANDIDATE_NAME_GLOBS), *names))
    return (
        "builtin set +e; builtin trap '' PIPE; builtin shopt -s extglob; "
        "__hermes_ifs=${IFS-}; IFS=$'\\n'; __hermes_bad=; "
        f"builtin printf '__hermes_sgen=%s\\n' {generation}; "
        f"__hermes_c=$(builtin compgen -A export -X '!@({candidates})'); builtin shopt -s nocasematch; "
        "for __hermes_v in $__hermes_c; do "
        f'case "$__hermes_v" in {secret}) ;; {url}) '
        'case "${!__hermes_v-}" in *://*:*@*) ;; *) continue ;; esac ;; *) continue ;; esac; '
        "{ builtin printf '%s=' \"$__hermes_v\" && builtin declare -p \"$__hermes_v\" && builtin printf '\\0'; } "
        f">&{secret_fd} 2>/dev/null || __hermes_bad=1; "
        'builtin export -n "$__hermes_v"; builtin unset "$__hermes_v" 2>/dev/null; done; '
        f"[ -z \"$__hermes_bad\" ] && builtin printf '\\0' >&{secret_fd} 2>/dev/null; "
        "IFS=$__hermes_ifs; builtin unset __hermes_v __hermes_c __hermes_ifs __hermes_bad; ")


def _secret_restore_lines(restore_fd: str) -> str:
    """Run right after the snapshot is sourced: read the secret declarations the backend wrote to
    fd *restore_fd* (a pipe, never process env: an escaped value doubles in ``declare -p`` and one
    oversize env entry would make every later spawn fail) and ``eval`` them, then drop the marker.
    ``read -d ''`` + ``eval``, not ``source /dev/fd/N``: bash 3.2 sources nothing from a pipe.
    ``read`` returns 1 at the EOF it always reaches (the set holds no NUL), so ``|| :`` keeps an
    inherited ``errexit`` from ending the command before it runs."""
    return (
        f"IFS= builtin read -r -d '' __hermes_d <&{restore_fd} || :; builtin eval \"${{__hermes_d-}}\"; "
        "builtin unset __hermes_d __hermes_sgen")


def _export_dump_excluding_session_vars(
        tmp_path: str, excluded_names: Iterable[str] = (), *, secret_fd: str = "",
        secret_names: Iterable[str] = (), secret_gen: str = "") -> str:
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
    never emits those vars — including any continuation lines. With *secret_fd* the secret exports
    leave through that fd instead (``_secret_handoff_snippet``, #62336).
    """
    # ${!PREFIX*} is bash 3.2+ name-prefix expansion; empty matches are ignored
    # under 2>/dev/null. Caller names are quoted so malformed config can never
    # become shell syntax (valid names stay unquoted by shlex.quote()).
    safe_names = {name for name in excluded_names if isinstance(name, str) and name}
    extra_unset = "".join(f" {shlex.quote(name)}" for name in sorted(safe_names))
    handoff = _secret_handoff_snippet(secret_fd, secret_names, secret_gen) if secret_fd else ""
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
        # Shell-option mirrors are read-only, so bash 3.2's ``declare -rx SHELLOPTS=…`` line fails
        # on ``source`` and an inherited errexit then ends every later command; options are not
        # environment the next command needs, so they stay out of the snapshot.
        "builtin export -n SHELLOPTS BASHOPTS 2>/dev/null; "
        f"{handoff}export -p; ) || true; }} "
        # ``>|``: the target is mktemp's existing file, which ``set -C`` would refuse with ``>``.
        f">| {tmp_path}")


def _snapshot_bootstrap_script(
    *, quoted_cwd: str, quoted_snap: str, snap_tmp_template: str, excluded_names: Iterable[str], cwd_marker: str,
    secret_fd: str = "", secret_names: Iterable[str] = (), secret_gen: str = "", secret_src: str = "",
    secret_restore: str = "",
) -> str:
    """Login-shell bootstrap that captures env/functions/aliases into the snapshot. Atomic publish:
    assemble in a ``mktemp`` file, then ``mv`` over the final path so a concurrent ``source`` never
    reads a half-written snapshot (``$$`` is the parent PID in ``&``-launched subshells and macOS
    bash 3.2 lacks ``$BASHPID``, so only ``mktemp`` is portable). Functions are filtered by NAME via
    ``declare -F`` (a line-based ``declare -f | grep -v`` strips the header and leaves an orphaned
    body that breaks every sourced command); the non-empty guard matters because bare ``declare -f``
    dumps ALL functions. The trailing ``cd`` restores the configured cwd after profile scripts (e.g.
    ``cd ~``) so ``pwd -P`` reports terminal.cwd, not the profile's directory."""
    return (
        "umask 077\n"
        f"__hermes_snap_tmp=$(mktemp {snap_tmp_template}) || exit 1\n"
        f"{_export_dump_excluding_session_vars(_SNAP_TMP, excluded_names, secret_fd=secret_fd, secret_names=secret_names, secret_gen=secret_gen)}\n"
        "__hermes_fns=$(declare -F | awk '{print $3}' | grep -vE '^_[^_]') || true\n"
        f"[ -n \"$__hermes_fns\" ] && declare -f $__hermes_fns >> {_SNAP_TMP} 2>/dev/null || true\n"
        f"alias -p >> {_SNAP_TMP}\n"
        f"echo 'shopt -s expand_aliases' >> {_SNAP_TMP}\n"
        f"echo 'set +e' >> {_SNAP_TMP}\n"
        f"echo 'set +u' >> {_SNAP_TMP}\n"
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
    pipe_marker: str = "", persist_env: bool = True, seed_snaps: Iterable[str] = (),
    cwd_fallback_marker: str = "", secret_fd: str = "", secret_names: Iterable[str] = (),
    secret_gen: str = "", secret_src: str = "", secret_restore: str = "") -> str:
    """Per-command bash script: source snapshot, cd, run, re-dump env, emit CWD marker.
    ``persist_env=False`` skips the re-dump, so every command starts from the login-time
    snapshot and exports from one command never reach the next (the CWD marker still runs).
    ``source`` stdout goes to /dev/null because macOS bash 3.2 / some Homebrew builds echo
    ``declare -x`` lines when sourcing. AI_AGENT/HERMES_AGENT advertise the harness to remote
    backends (whose env is not inherited); ``${VAR:-default}`` never clobbers an outer harness.
    GIT_PAGER/PAGER=cat stop pager-happy tools hanging a PTY-backed command. The env re-dump
    uses the same mktemp+mv atomic publish as the bootstrap and chains ``mv`` on the dump
    succeeding so a failed dump never replaces a good snapshot. ``umask 077`` is applied after
    the user's command so snapshot files (which may carry secrets) are private without
    changing the command's umask. *seed_snaps* (quoted paths) are read, first existing wins,
    while *quoted_snap* does not exist yet: a delegated child's own snapshot starts as a copy of
    its parent's shell state and its re-dump never writes the parent's file. *secret_fd* is
    closed for the user's command, so nothing it starts (``server &``) holds the channel open.
    With *secret_src* the backend already opened the snapshot to source (its own, else the first
    ancestor's) and the script reads that open file, never a newer one published meanwhile.
    """
    escaped = command.replace("'", "'\\''")
    capture = bool(pipe_marker) and _can_capture_pipestatus(command)
    save, restore = _passthrough_save_restore(passthrough_names)
    parts = list(save)
    seeds = list(seed_snaps)
    if snapshot_ready and secret_src:
        parts += [f"source /dev/fd/{secret_src} >/dev/null 2>&1 || true", _secret_restore_lines(secret_restore)]
    elif snapshot_ready and seeds:
        parts.append(
            f"for __hermes_s in {quoted_snap} {' '.join(seeds)}; do "
            'if [ -f "$__hermes_s" ]; then source "$__hermes_s" >/dev/null 2>&1 || true; break; fi; done')
    elif snapshot_ready:
        parts.append(f"source {quoted_snap} >/dev/null 2>&1 || true")
    parts += restore
    run = _eval_capturing_pipestatus(escaped) if capture else f"eval '{escaped}'"
    parts += [
        'export AI_AGENT="${AI_AGENT:-hermes-agent}" HERMES_AGENT="${HERMES_AGENT:-true}"',
        'export GIT_PAGER="${GIT_PAGER:-cat}" PAGER="${PAGER:-cat}"',
        # ``--`` keeps hyphen-prefixed directory names from being parsed as options.
        _cd_with_fallback(quoted_cwd, cwd_fallback_marker),
        f"{run} {secret_fd}>&-{f' {secret_src}>&- {secret_restore}<&-' if snapshot_ready and secret_src else ''}"
        if secret_fd else run,
        "__hermes_ec=$?",
        "umask 077"]
    if snapshot_ready and persist_env:
        dump = _export_dump_excluding_session_vars(
            _SNAP_TMP, passthrough_names, secret_fd=secret_fd, secret_names=secret_names, secret_gen=secret_gen)
        parts.append(
            f"__hermes_snap_tmp=$(mktemp {snap_tmp_template}) && "
            f"{{ {dump} "
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
    cwd_path = output[first + len(marker) : last] or None  # verbatim: a directory name may end in a space
    line_start = output.rfind("\n", 0, first)
    if line_start == -1:
        line_start = first
    line_end = output.find("\n", last + len(marker))
    line_end = line_end + 1 if line_end != -1 else len(output)
    return cwd_path, output[:line_start] + output[line_end:]
