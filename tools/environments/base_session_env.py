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
    "HERMES_CRON_SESSION|HERMES_BROWSER_CONTROL_)")
_SHELL_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# mktemp template suffix + the shell variable holding the allocated temp path.
_SNAP_TMP_SUFFIX = ".tmp.XXXXXXXXXX"
_SNAP_TMP = '"$__hermes_snap_tmp"'


def _cwd_marker(session_id: str) -> str:
    return f"__HERMES_CWD_{session_id}__"


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
    """
    # ${!PREFIX*} is bash 3.2+ name-prefix expansion; empty matches are ignored
    # under 2>/dev/null. Caller names are quoted so malformed config can never
    # become shell syntax (valid names stay unquoted by shlex.quote()).
    safe_names = {name for name in excluded_names if isinstance(name, str) and name}
    extra_unset = "".join(f" {shlex.quote(name)}" for name in sorted(safe_names))
    return (
        "{ ( unset ${!HERMES_SESSION_*} ${!HERMES_CRON_AUTO_DELIVER_*} "
        "${!HERMES_BROWSER_CONTROL_*} "
        # AI_AGENT / HERMES_AGENT are per-command attribution markers re-exported
        # by every wrapper with ${VAR:-default} semantics; persisting them would
        # let the FIRST command's value override a later outer-harness value.
        "AI_AGENT HERMES_AGENT "
        f"HERMES_UI_SESSION_ID{extra_unset} 2>/dev/null; "
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
    dumps ALL functions. The trailing ``cd`` restores the configured cwd after profile scripts (e.g.
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
    passthrough_names: Iterable[str], snapshot_ready: bool, cwd_marker: str) -> str:
    """Per-command bash script: source snapshot, cd, run, re-dump env, emit CWD marker.
    ``source`` stdout goes to /dev/null because macOS bash 3.2 / some Homebrew builds echo
    ``declare -x`` lines when sourcing. AI_AGENT/HERMES_AGENT advertise the harness to remote
    backends (whose env is not inherited); ``${VAR:-default}`` never clobbers an outer harness.
    GIT_PAGER/PAGER=cat stop pager-happy tools hanging a PTY-backed command. The env re-dump
    uses the same mktemp+mv atomic publish as the bootstrap and chains ``mv`` on the dump
    succeeding so a failed dump never replaces a good snapshot. ``umask 077`` is applied after
    the user's command so snapshot files (which may carry secrets) are private without
    changing the command's umask.
    """
    escaped = command.replace("'", "'\\''")
    save, restore = _passthrough_save_restore(passthrough_names)
    parts = list(save)
    if snapshot_ready:
        parts.append(f"source {quoted_snap} >/dev/null 2>&1 || true")
    parts += restore
    parts += [
        'export AI_AGENT="${AI_AGENT:-hermes-agent}" HERMES_AGENT="${HERMES_AGENT:-true}"',
        'export GIT_PAGER="${GIT_PAGER:-cat}" PAGER="${PAGER:-cat}"',
        # ``--`` keeps hyphen-prefixed directory names from being parsed as options.
        f"builtin cd -- {quoted_cwd} || exit 126",
        f"eval '{escaped}'",
        "__hermes_ec=$?",
        "umask 077"]
    if snapshot_ready:
        parts.append(
            f"__hermes_snap_tmp=$(mktemp {snap_tmp_template}) && "
            f"{{ {_export_dump_excluding_session_vars(_SNAP_TMP, passthrough_names)} "
            f"&& mv -f {_SNAP_TMP} {quoted_snap}; }} "
            f"2>/dev/null || rm -f {_SNAP_TMP} 2>/dev/null || true")
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
