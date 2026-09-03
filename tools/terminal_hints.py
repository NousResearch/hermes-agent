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


# ---------------------------------------------------------------------------
# Masked-success detection (exit 0 that probably isn't a success)
# ---------------------------------------------------------------------------
#
# `cargo build 2>&1 | tail -20` exits with tail's 0 even when the build
# failed: bash (without pipefail) reports the LAST pipeline command's status.
# Likewise `cargo build || echo "BUILD FAILED"` exits with echo's 0. The
# model treats exit_code 0 as a strong success signal, so it can conclude a
# build passed while the visible output says it failed. OpenCode's answer is
# prompt-side only ("do NOT pipe through head/tail"); this adds a cheap
# result-side backstop for when the model pipes anyway.
#
# Deliberately conservative — BOTH must hold:
#   1. the command's shape can mask an upstream status (a top-level pipe into
#      a passthrough/truncation consumer, or a `|| <cheap fallback>`), and
#   2. the output contains a strong, tool-specific failure shape (rustc /
#      pytest / gcc / npm / tracebacks), not a generic "error" substring.
# Search/read-only pipelines (`grep ... | head`) are excluded: their output
# legitimately CONTAINS error text without anything having failed.
#
# The note is advisory metadata only — exit_code itself is never modified.

# Consumers whose exit status says nothing about the upstream command.
_PASSTHROUGH_CONSUMERS = r"(?:tail|head|cat|tee|less|more|wc|sort|uniq)"

# Top-level `... | tail -20` (not `||`); consumer must be the LAST segment.
_MASKING_PIPE_RE = re.compile(
    r"(?<!\|)\|(?!\|)\s*" + _PASSTHROUGH_CONSUMERS + r"\b[^|]*$"
)

# `cmd || echo ...` / `cmd || true` — fallback swallows the failure status.
_MASKING_OR_RE = re.compile(r"\|\|\s*(?:echo\b|printf\b|true\b|:\s|:$)")

# Read/search/content-producing heads whose piped output legitimately
# contains failure text (search results, log reads, echoed strings) —
# nothing upstream can meaningfully "fail" in the masked sense.
_READONLY_HEADS = frozenset({
    "grep", "rg", "ag", "find", "ls", "cat", "head", "tail", "jq", "awk",
    "sed", "strings", "zcat", "journalctl", "dmesg", "echo", "printf",
})

# Strong failure shapes. Keyed to specific tools so that error-mentioning
# *content* (diffs, logs being read, commit messages) rarely matches.
_FAILURE_SHAPES = re.compile(
    r"(?:"
    r"error\[E\d+\]"                          # rustc
    r"|error: could not compile"              # cargo
    r"|error: aborting due to"                # rustc summary
    r"|Traceback \(most recent call last\)"   # python
    r"|(?m:^(?:=+ )?\d+ failed)"              # pytest summary
    r"|(?m:^FAILED (?:\S+::|\S+\.py))"        # pytest per-test lines
    r"|compilation terminated\."              # gcc/clang
    r"|npm ERR!"                              # npm
    r"|BUILD FAILED|Build FAILED"             # gradle/msbuild/echoed fallbacks
    r"|FAILED: "                              # ninja
    r"|(?m:^make(?:\[\d+\])?: \*\*\*)"        # make
    r")"
)


def _first_token(command: str) -> str:
    for tok in (command or "").strip().split():
        # Skip env-var assignments and common wrappers.
        if "=" in tok and not tok.startswith(("=", "./", "/")):
            continue
        return tok.rsplit("/", 1)[-1]
    return ""


def annotate_masked_success(command: str, output: str) -> Optional[str]:
    """Return a warning note when an exit-0 result likely masks a failure.

    Fires only for exit_code == 0 results (caller gates on that) whose
    command shape can swallow an upstream failure status AND whose output
    carries a strong tool-specific failure shape. Returns None otherwise.
    Never modifies the exit code — advisory only.
    """
    cmd = command or ""
    window = (output or "")[:_SCAN_CHARS]
    if not cmd or not window:
        return None
    if _first_token(cmd) in _READONLY_HEADS:
        return None
    if not _FAILURE_SHAPES.search(window):
        return None
    if _MASKING_PIPE_RE.search(cmd):
        return (
            "exit_code 0 here is the status of the last pipeline command "
            "(tail/head/cat/...), NOT of the command before the pipe — and "
            "the output contains failure indicators. Treat this run as "
            "FAILED until proven otherwise: re-run the command WITHOUT the "
            "pipe (output is auto-truncated and the full text is saved to a "
            "file, so piping through tail/head is never needed) to get the "
            "real exit code."
        )
    if _MASKING_OR_RE.search(cmd):
        return (
            "exit_code 0 here is the status of the `||` fallback (echo/true), "
            "NOT of the command before it — and the output contains failure "
            "indicators. Treat this run as FAILED until proven otherwise: "
            "re-run the command bare to get its real exit code."
        )
    return None


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
