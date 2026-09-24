"""Foreground result post-processing for the terminal tool (cwd dual-write,
sudo handling, transform hook, truncation, ANSI strip, redaction, exit-code
notes/hints, spill redaction, verification evidence) + exit-code tables. Lazy
``tools.terminal_tool`` lookups keep the origin's monkeypatch points authoritative.
"""

import json
import logging
import os
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("tools.terminal_tool")


@contextmanager
def _quiet(label: str):
    """Best-effort block: swallow any exception, logging it at DEBUG."""
    try:
        yield
    except Exception:
        logger.debug("%s failed", label, exc_info=True)


# Signal-death notes keyed by signum, used for both ``-signum`` (subprocess)
# and ``128+signum`` (shell) encodings. Curated, not exhaustive, so a
# legitimate application exit code is never mislabeled; 130/SIGINT is owned
# by the executor's interrupt-marker path and excluded.
_SIGNAL_EXIT_NOTES: dict[int, str] = {
    3:  "SIGQUIT (quit from keyboard)",
    4:  "SIGILL (illegal instruction — corrupt binary or wrong architecture)",
    6:  "SIGABRT (abort — assertion failure, fatal runtime error, or glibc abort)",
    7:  "SIGBUS (bus error — misaligned or unmapped memory access)",
    8:  "SIGFPE (fatal arithmetic error, e.g. integer division by zero)",
    9:  "SIGKILL — often the kernel OOM killer on memory exhaustion, or an explicit kill -9",
    11: "SIGSEGV (segmentation fault — the program crashed)",
    13: "SIGPIPE (wrote to a closed pipe — e.g. output piped to a reader that exited)",
    15: "SIGTERM (terminated — kill/timeout or shutdown requested it to stop)",
    24: "SIGXCPU (CPU time limit exceeded)",
    25: "SIGXFSZ (file size limit exceeded)",
}


def _interpret_signal_exit(exit_code: int) -> str | None:
    """Note for a signal-termination exit code, or None. Negative codes are
    definite (subprocess semantics); 128+signum is the shell convention and a
    program *can* exit 139 itself, so those notes hedge with "usually"."""
    if exit_code < 0:
        signum = -exit_code
        if signum == 2:  # SIGINT — executor's interrupt-marker path owns it
            return None
        if note := _SIGNAL_EXIT_NOTES.get(signum):
            return f"Command terminated by signal {signum}: {note}"
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"signal {signum}"
        return f"Command terminated by {name} (signal {signum})"
    if exit_code > 128 and (note := _SIGNAL_EXIT_NOTES.get(exit_code - 128)):
        return (f"Exit code {exit_code} usually means the command was "
                f"terminated by signal {exit_code - 128}: {note}")
    return None


# Informational non-zero exit codes per base command.
_EXIT_CODE_SEMANTICS: dict[str, dict[int, str]] = {
    **dict.fromkeys(("grep", "egrep", "fgrep", "rg", "ag", "ack"), {1: "No matches found (not an error)"}),
    **dict.fromkeys(("diff", "colordiff"), {1: "Files differ (expected, not an error)"}),
    **dict.fromkeys(("test", "["), {1: "Condition evaluated to false (expected, not an error)"}),
    "find": {1: "Some directories were inaccessible (partial results may still be valid)"},
    "curl": {6: "Could not resolve host", 7: "Failed to connect to host",
             22: "HTTP response code indicated error (e.g. 404, 500)", 28: "Operation timed out"},
    "git": {1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"},
}


def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """Note for a non-zero exit code that is informational rather than an
    error (grep=1 "no matches", diff=1 "files differ", signal deaths), so the
    model doesn't burn turns investigating it. None when 0 or a real error."""
    if exit_code == 0:
        return None
    if (signal_note := _interpret_signal_exit(exit_code)) is not None:
        # Signal terminations (ported from Kilo-Org/kilocode#12698, adapted to Python semantics). Two shapes
        # reach the model: * negative codes — subprocess.Popen reports a signal-killed process as
        # ``-signum`` (definite signal death), and * 128+signum — the conventional shell encoding when bash
        # reports a signal-killed child (heuristic: a program *can* ``exit 139``, so these notes say
        # "usually"). Without a note the model sees a bare ``exit_code=-9`` or ``137`` and burns turns
        # re-running or mis-diagnosing (137 = OOM kill is the big one). 130/SIGINT is deliberately absent:
        # the executor has bespoke interrupt-marker handling for rc=130.
        return signal_note
    # The last command of a pipeline/chain determines the exit code; base
    # command = its first word that isn't a VAR=val assignment, basename'd.
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()
    base_cmd = next((w.split("/")[-1] for w in last_segment.split()
                     if "=" not in w or w.startswith("-")), "")
    return _EXIT_CODE_SEMANTICS.get(base_cmd, {}).get(exit_code)


def _sudo_annotations(command: str, output: str, env_type: str) -> tuple[str, bool, bool]:
    """Sudo failure handling -> (output, auth_failed, cache_cleared)."""
    import tools.terminal_tool as tt
    from tools.terminal_tool_sudo import (
        _handle_sudo_failure, _in_delegated_child_context, _invalidate_cached_sudo_on_auth_failure,
        _sudo_wrong_password_failure,
    )
    from utils import env_var_enabled
    output = _handle_sudo_failure(output, env_type)
    auth_failed = _sudo_wrong_password_failure(output)
    cache_cleared = _invalidate_cached_sudo_on_auth_failure(command, output)
    can_reprompt = cache_cleared and (
        tt._get_sudo_password_callback() is not None or env_var_enabled("HERMES_INTERACTIVE")
    ) and not _in_delegated_child_context()
    if can_reprompt:
        output += ("\n\n⚠️ Sudo authentication failed — cached password "
                   "cleared. You will be prompted again on the next sudo command.")
    return output, auth_failed, cache_cleared


def _apply_output_transform_hook(command, output, returncode, task_id, env_type) -> str:
    """Plugin output-transform seam (fail-open; first string result wins).
    Replacements are still subject to the output limit applied afterwards."""
    with _quiet("transform_terminal_output hook"):
        from hermes_cli.lifecycle import invoke_hook
        results = invoke_hook("transform_terminal_output", command=command, output=output,
                              returncode=returncode, task_id=task_id or "", env_type=env_type)
        output = next((r for r in results if isinstance(r, str)), output)
    return output


def _truncate_head_tail(output: str) -> str:
    """Truncate keeping head (errors often appear early) and tail (most recent)."""
    from tools.tool_output_limits import get_max_bytes
    max_chars = get_max_bytes()
    if len(output) <= max_chars:
        return output
    head_chars = int(max_chars * 0.4)
    tail_chars = max_chars - head_chars
    notice = (f"\n\n... [OUTPUT TRUNCATED - {len(output) - head_chars - tail_chars} "
              f"chars omitted out of {len(output)} total] ...\n\n")
    return output[:head_chars] + notice + output[-tail_chars:]


def _failure_hint(command: str, returncode: int, output: str, exit_note) -> Optional[str]:
    """Recovery hints for well-known failure shapes (tools/terminal_hints.py);
    on rc=0, warn when a pipeline tail / `|| echo` may mask an upstream
    failure and the output carries strong failure indicators (advisory only)."""
    with _quiet("failure hint"):
        from tools import terminal_hints
        if returncode == 0:
            return terminal_hints.annotate_masked_success(command, output)
        if not exit_note:
            return terminal_hints.annotate_failure(command, returncode, output)
    return None


def _redact_spill_file(path, total_chars, command) -> list[tuple[str, Any]]:
    """Spill handle so the model can read the omitted middle instead of
    re-running. The collector wrote it raw; redact it with the same pass so no
    secret persists unmasked on disk. On failure drop the handle (and file)."""
    if not path:
        return []
    try:
        from agent.redact import redact_terminal_output
        from tools.ansi_strip import strip_ansi
        from tools.spill_safety import write_text_exclusive
        raw_spill = Path(path).read_text(encoding="utf-8", errors="replace")
        # lstat-checked unlink + exclusive create: the redacted copy can't
        # be diverted through a symlink planted since the collector's write.
        write_text_exclusive(Path(path), redact_terminal_output(strip_ansi(raw_spill), command),
                             private=True, overwrite=True, errors="replace")
    except Exception:
        logger.debug("spill redaction failed; dropping spill handle", exc_info=True)
        with _quiet("spill unlink"):
            Path(path).unlink()
        return []
    note = ("Output exceeded the capture window (head+tail shown). "
            f"Full output ({total_chars:,} chars) saved to {path} — search it with "
            "search_files or page it with read_file instead of re-running the command.")
    return [("output_total_chars", total_chars), ("full_output_path", path), ("truncation_note", note)]


def _verification_evidence(command, cwd, session_id, returncode, output) -> Optional[dict]:
    with _quiet("verification evidence recording"):
        from agent.verification_evidence import record_terminal_result
        evidence = record_terminal_result(command=command, cwd=cwd, session_id=session_id,
                                          exit_code=returncode, output=output)
        if evidence:
            return {k: evidence.get(k) for k in ("status", "kind", "scope", "canonical_command")}
    return None


def finalize_foreground_result(
    *, command: str, result: dict, env: Any, env_type: str, effective_task_id: str,
    task_id: Optional[str], session_id: Optional[str], session_key: str,
    workdir: Optional[str], command_cwd: Optional[str], approval_note: Optional[str],
) -> str:
    """Turn a raw ``env.execute`` result into the tool's JSON result string."""
    from tools.terminal_tool import record_session_cwd

    # Record the cwd this command finished in as THIS session's durable cwd —
    # only when the command actually reported it (an interrupted/killed command
    # emits no marker, and env.cwd then holds another session's directory), and
    # never for a transient per-command ``workdir`` (would hijack the session cwd
    # for every later command). Prefer the result's own cwd; env.cwd is shared
    # mutable compat state kept as fallback for third-party providers.
    observed_cwd = None
    if (result or {}).get("cwd_observed"):
        observed_cwd = (result or {}).get("cwd") or getattr(env, "cwd", None)
    if not workdir and observed_cwd:
        record_session_cwd(session_key, observed_cwd)

    output = result.get("output", "")
    returncode = result.get("returncode", 0)
    output, sudo_auth_failed, sudo_cache_cleared = _sudo_annotations(command, output, env_type)
    output = _apply_output_transform_hook(command, output, returncode, effective_task_id, env_type)
    output = _truncate_head_tail(output)
    # Strip ANSI so the model never copies escapes into file writes, then
    # redact secrets; redact_terminal_output is command-aware (env-dump
    # commands get the KEY=value pass, source/config dumps skip it).
    from agent.redact import redact_terminal_output
    from tools.ansi_strip import strip_ansi
    output = strip_ansi(output)
    # For source/config dumps (MAX_TOKENS=100, "apiKey": "x" fixtures, postgresql:// f-string templates) the
    # ENV/JSON/template passes are skipped to avoid false positives (code_file=True). But for env-dump
    # commands (env/printenv/set/export/declare) the output IS a KEY=value credential dump, so
    # redact_terminal_output runs the ENV pass (code_file=False) to mask opaque tokens with no vendor
    # prefix. Real prefixes, auth headers, JWTs, private keys are masked in both modes. See issue #43025.
    output = redact_terminal_output(output.strip(), command) if output else ""

    exit_note = _interpret_exit_code(command, returncode)
    failure_hint = _failure_hint(command, returncode, output, exit_note)
    # cwd echo when the command changed directory (gated on the observation
    # flag so an interrupted command can't echo another session's cwd).
    changed_cwd = None
    with _quiet("cwd comparison"):
        if observed_cwd and command_cwd and os.path.realpath(str(observed_cwd)) != os.path.realpath(str(command_cwd)):
            changed_cwd = str(observed_cwd)
    # rc=130 is an interrupt only with the executor's marker — a command can
    # legitimately `exit 130` itself. An interrupted approved run keeps the
    # audit note but must never imply success.
    if approval_note and returncode == 130 and "[Command interrupted]" in output:
        approval_note = approval_note.rstrip(".") + ", then interrupted."

    result_dict = {"output": output, "exit_code": returncode, "error": None}
    # Optional fields in observable JSON key order; None means "omit". Spill
    # metadata is present only when output overflowed the capture window.
    optional_fields: list[tuple[str, Any]] = [
        ("cwd", changed_cwd),
        *_redact_spill_file(result.get("full_output_path"), result.get("output_total_chars"), command),
        ("verification_evidence", _verification_evidence(
            command, command_cwd, session_id or task_id or effective_task_id or "default",
            returncode, output)),
        ("approval", approval_note or None),
        ("exit_code_meaning", exit_note or None),
        ("hint", failure_hint or None),
        ("sudo_auth_failed", True if sudo_auth_failed else None),
        ("sudo_cache_cleared", True if sudo_cache_cleared else None),
    ]
    for key, value in optional_fields:
        if value is not None:
            result_dict[key] = value
    return json.dumps(result_dict, ensure_ascii=False)
