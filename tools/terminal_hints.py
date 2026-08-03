"""Output-pattern failure hints for the terminal tool.

When a command exits non-zero, the raw stderr often confuses models into
wasted diagnostic turns (e.g. retrying `python` when only `python3` exists,
or re-sending a gh field list that the installed gh doesn't support).

This module extends the exit-code semantics table in ``terminal_tool`` with
an *output-pattern* tier: a bounded scan of the command output that maps
well-known failure shapes to one short, actionable recovery hint.

Design rules (keep these when adding patterns):

* Only fires on non-zero exit codes — never annotate success.
* At most ONE hint per result, first match wins; patterns are ordered by
  observed frequency in production trajectories (state.db mining, Aug 2026).
* Scans only the first ``_SCAN_CHARS`` of output — hints must key on error
  headers, not deep context.
* Hints state the *next action*, not a diagnosis essay. One or two sentences.
* Pure function, no I/O, no config reads — trivially unit-testable.

Frequencies quoted below come from a 250k-terminal-result window of the
production session DB (Aug 2026): together these classes cover ~14k failed
calls whose retry chains averaged 1.4 extra tool turns each.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

# Bounded scan window: error headers appear early; deep output is noise.
_SCAN_CHARS = 4000


def _hint_gh_unknown_json_field(command: str, output: str) -> Optional[str]:
    # ~9,175x: gh CLI version drift — model asks for fields the installed
    # gh doesn't know. gh already prints the valid field list.
    m = re.search(r'Unknown JSON field: "?(\w+)', output)
    if not m:
        return None
    return (
        f"The installed gh does not support the JSON field '{m.group(1)}'. "
        "The valid field list is printed in the output above — retry using "
        "only fields from that list."
    )


def _hint_command_not_found(command: str, output: str) -> Optional[str]:
    # ~1,010x generic; 837x of them are bare `python` on python3-only distros.
    m = re.search(r"(?:bash: line \d+: |bash: |sh: \d*:? ?)?([\w.+-]+): command not found", output)
    if not m:
        return None
    missing = m.group(1)
    if missing == "python":
        return (
            "This system has no bare `python` — use `python3`, or the "
            "project venv's interpreter (e.g. .venv/bin/python)."
        )
    if missing == "pip":
        return (
            "This system has no bare `pip` — use `pip3`, `python3 -m pip`, "
            "or the project venv's pip (e.g. .venv/bin/pip)."
        )
    return (
        f"`{missing}` is not installed or not on PATH. Verify with "
        f"`which {missing}`; install it or use an absolute path instead of "
        "retrying the same command."
    )


def _hint_module_not_found(command: str, output: str) -> Optional[str]:
    # ~739x: almost always a venv-activation slip, not a missing dependency.
    m = re.search(r"(?:ModuleNotFoundError|ImportError): No module named '?([\w.]+)", output)
    if not m:
        return None
    return (
        f"Python cannot import '{m.group(1)}'. Most often the wrong "
        "interpreter is running: activate the project venv (e.g. `source "
        ".venv/bin/activate`) or invoke its python directly. Only pip "
        "install if the package is genuinely absent from that venv."
    )


def _hint_merge_conflict(command: str, output: str) -> Optional[str]:
    # ~1,172x: models sometimes re-run the failing merge/rebase verbatim.
    if not re.search(r"^CONFLICT |Automatic merge failed|needs merge", output, re.M):
        return None
    return (
        "Git merge conflict. Do not retry this command. Resolve the "
        "conflicted files listed above (edit, then `git add`), then continue "
        "(`git rebase --continue` / commit the merge) — or abort with "
        "`--abort`."
    )


def _hint_already_exists(command: str, output: str) -> Optional[str]:
    # ~633x: branch/dir/file already exists → retrying unchanged always fails.
    m = re.search(r"(?:fatal|error):.*?'([^']+)' already exists", output)
    if not m:
        return None
    return (
        f"'{m.group(1)}' already exists — retrying unchanged will keep "
        "failing. Reuse it, choose another name, or delete it first if it is "
        "genuinely stale."
    )


def _hint_gh_rate_limit(command: str, output: str) -> Optional[str]:
    # ~133x: immediate retries burn turns; the limit is time-based.
    if "API rate limit" not in output and "was submitted too quickly" not in output:
        return None
    return (
        "GitHub API rate limit hit — immediate retries will keep failing. "
        "Continue with other work and retry this operation later."
    )


def _hint_permission_denied(command: str, output: str) -> Optional[str]:
    if "Permission denied" not in output and "EACCES" not in output:
        return None
    return (
        "Permission denied. Check ownership/mode of the target path "
        "(`ls -la`); prefer a user-writable location. Only escalate to sudo "
        "if the task genuinely requires it."
    )


# Ordered by production frequency — first match wins.
_OUTPUT_HINTS: list[Callable[[str, str], Optional[str]]] = [
    _hint_gh_unknown_json_field,
    _hint_merge_conflict,
    _hint_command_not_found,
    _hint_module_not_found,
    _hint_already_exists,
    _hint_gh_rate_limit,
    _hint_permission_denied,
]

# Exit-code-only hints for codes the semantics table in terminal_tool does
# not cover per-command. Checked after output patterns.
_EXIT_CODE_HINTS: dict[int, str] = {
    126: "Exit 126: the file was found but is not executable — `chmod +x` it or invoke it via its interpreter (e.g. `bash script.sh`).",
    137: "Exit 137: the process was SIGKILLed — usually out-of-memory or an external kill. Reduce memory use or check `dmesg | tail` before retrying.",
    124: "Exit 124: the command hit its timeout. Raise timeout= (foreground max 600s) or run it with background=true and notify_on_complete=true.",
}


def annotate_failure(command: str, exit_code: int, output: str) -> Optional[str]:
    """Return one short recovery hint for a failed command, or None.

    Args:
        command: The command string that ran.
        exit_code: Its exit code (non-zero for failures).
        output: Combined stdout/stderr as returned to the model.

    Only the first ``_SCAN_CHARS`` characters of output are examined and at
    most one hint is returned. Returns None for exit_code == 0.
    """
    if exit_code == 0:
        return None
    window = (output or "")[:_SCAN_CHARS]
    if window:
        for fn in _OUTPUT_HINTS:
            try:
                hint = fn(command or "", window)
            except Exception:
                continue
            if hint:
                return hint
    return _EXIT_CODE_HINTS.get(exit_code)
