"""Output-pattern failure hints for the terminal tool.

Extends the exit-code semantics table in ``terminal_tool`` with a bounded scan of failed-command
output mapped to ONE short, actionable recovery hint. Rules: only fires on non-zero exit; first
match wins, patterns ordered by observed production frequency; scans only the first
``_SCAN_CHARS`` so hints key on error headers, not deep context; hints state the *next action*
in 1-2 sentences; pure function, no I/O or config reads.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

_SCAN_CHARS = 4000


def _regex_hint(pattern: str, message: str | Callable[[str], str], flags: int = 0) -> Callable[[str, str], Optional[str]]:
    """Hint firing when ``pattern`` matches; ``{0}`` = first group, or ``message(group1)``."""
    rx = re.compile(pattern, flags)

    def hint(command: str, output: str) -> Optional[str]:
        m = rx.search(output)
        if m:
            return message(m.group(1)) if callable(message) else message.format(*m.groups())
        return None

    return hint


# Most `command not found` hits are bare `python` on python3-only distros.
_MISSING_COMMAND_HINTS = {
    "python": "This system has no bare `python` — use `python3`, or the project venv's "
              "interpreter (e.g. .venv/bin/python).",
    "pip": "This system has no bare `pip` — use `pip3`, `python3 -m pip`, or the project venv's "
           "pip (e.g. .venv/bin/pip).",
}


def _missing_command_hint(missing: str) -> str:
    return _MISSING_COMMAND_HINTS.get(missing) or (
        f"`{missing}` is not installed or not on PATH. Verify with `which {missing}`; install it "
        "or use an absolute path instead of retrying the same command.")


# A generated script/command that embeds a natural-language payload in its own
# source dies when a character lands where the source expects code: typographic
# punctuation that cannot stand in code position — a smart quote or guillemet
# used as a delimiter, or a middot / em dash / prime / acute standing where an
# operator or literal belongs — or a payload quote closing the literal that
# carries it (also breaking a single-quoted shell arg).
# Patterns are public so code_execution_tool shares ONE source of truth; they
# match only the collision errors themselves, so typographic punctuation
# inside a correctly delimited literal (which runs fine) never fires.
# The `invalid character` pattern is intentionally broad, NOT quote-only: every
# typographic punctuation mark a payload may use (guillemets, low-9 quotes,
# fullwidth quote, prime, acute, middot, em dash) produces the same
# `invalid character '<ch>' (U+XXXX)` message, so narrowing it to “”‘’ would
# false-negative across the bug class.
# Invisible characters (NBSP, ZWSP) produce a DIFFERENT message —
# `invalid non-printable character U+XXXX` — a different cause with a
# different remedy, which these patterns must NOT match.
PAYLOAD_QUOTING_PATTERNS: tuple[str, ...] = (
    r"SyntaxError: invalid character",
    # payload apostrophe closed the literal; the optional group covers the
    # triple-quoted variant (multi-line payload inside a """...""" literal).
    r"SyntaxError: unterminated (?:triple-quoted )?string literal",
    # Python 3.9 and earlier word those two failures differently ("EOL/EOF while
    # scanning ..."; bpo-40176 changed it in 3.10) — captured verbatim on 3.9.6,
    # which is still the macOS system python3. Both messages are specific to an
    # unterminated literal, so they carry the same remedy. A nested payload
    # apostrophe on 3.9.6 surfaces as the generic `SyntaxError: invalid syntax`,
    # which this rule deliberately leaves to the unrelated-syntax-error boundary.
    r"SyntaxError: EOL while scanning string literal",
    r"SyntaxError: EOF while scanning triple-quoted string literal",
    r"unexpected EOF while looking for matching",         # bash/sh wrapper, unmatched quote
    r"(?:^|\s)zsh:\d*:? unmatched",                       # zsh's wording for the same failure
)

PAYLOAD_QUOTING_HINT: str = (
    "The generated source contains a character that cannot appear in code position — "
    "typographic punctuation, or a payload quote closing the literal that carries it — "
    "so the same source will fail again. Leave the payload text alone: write it to a file "
    "with write_file and pass the file (`gh ... --body-file <file>`, "
    "`gh api -F body=@<file>`, `open(path).read()`); a character that is merely stray in "
    "the generated code may be swapped for ASCII."
)


# Ordered by production frequency — first match wins.
_OUTPUT_HINTS: list[Callable[[str, str], Optional[str]]] = [
    # gh version drift; gh already prints the valid field list.
    _regex_hint(r'Unknown JSON field: "?(\w+)',
                "The installed gh does not support the JSON field '{0}'. The valid field list is "
                "printed in the output above — retry using only fields from that list."),
    _regex_hint(r"^CONFLICT |Automatic merge failed|needs merge",
                "Git merge conflict. Do not retry this command. Resolve the conflicted files "
                "listed above (edit, then `git add`), then continue (`git rebase --continue` / "
                "commit the merge) — or abort with `--abort`.", re.M),
    _regex_hint(r"(?:bash: line \d+: |bash: |sh: \d*:? ?)?([\w.+-]+): command not found",
                _missing_command_hint),
    # Almost always a venv-activation slip, not a missing dependency.
    _regex_hint(r"(?:ModuleNotFoundError|ImportError): No module named '?([\w.]+)",
                "Python cannot import '{0}'. Most often the wrong interpreter is running: "
                "activate the project venv (e.g. `source .venv/bin/activate`) or invoke its python "
                "directly. Only pip install if the package is genuinely absent from that venv."),
    _regex_hint(r"(?:fatal|error):.*?'([^']+)' already exists",
                "'{0}' already exists — retrying unchanged will keep failing. Reuse it, choose "
                "another name, or delete it first if it is genuinely stale."),
    _regex_hint(r"API rate limit|was submitted too quickly",
                "GitHub API rate limit hit — immediate retries will keep failing. Continue with "
                "other work and retry this operation later."),
    _regex_hint(r"Permission denied|EACCES",
                "Permission denied. Check ownership/mode of the target path (`ls -la`); prefer a "
                "user-writable location. Only escalate to sudo if the task genuinely requires it."),
    # Payload-quoting collision in generated code — one entry built from the
    # shared public patterns so code_execution_tool keeps ONE source of truth.
    _regex_hint("(?:" + "|".join(PAYLOAD_QUOTING_PATTERNS) + ")", PAYLOAD_QUOTING_HINT),
]

# Exit-code-only hints for codes the terminal_tool semantics table does not
# cover per-command. Checked after output patterns.
_EXIT_CODE_HINTS: dict[int, str] = {
    126: "Exit 126: the file was found but is not executable — `chmod +x` it or invoke it via its interpreter (e.g. `bash script.sh`).",
    137: "Exit 137: the process was SIGKILLed — usually out-of-memory or an external kill. Reduce memory use or check `dmesg | tail` before retrying.",
    124: "Exit 124: the command hit its timeout. Raise timeout= (foreground max 600s) or run it with background=true and notify_on_complete=true.",
}


# Masked-success detection: `cargo build 2>&1 | tail -20` exits with tail's 0
# (no pipefail) and `cargo build || echo FAILED` with echo's 0, so the model can
# conclude a build passed while the output says it failed. Conservative: BOTH a
# masking command shape AND a strong tool-specific failure shape must hold, and
# read-only heads (`grep ... | head`) are excluded because their output
# legitimately contains error text. Advisory only — exit_code is never changed.

# Consumers whose exit status says nothing about the upstream command.
_PASSTHROUGH_CONSUMERS = r"(?:tail|head|cat|tee|less|more|wc|sort|uniq)"

# Command shapes that swallow an upstream status -> warning, checked in order.
_MASKING_SHAPES: list[tuple[re.Pattern[str], str]] = [
    # Top-level `... | tail -20` (not `||`); consumer must be the LAST segment.
    (re.compile(r"(?<!\|)\|(?!\|)\s*" + _PASSTHROUGH_CONSUMERS + r"\b[^|]*$"),
     "exit_code 0 here is the status of the last pipeline command (tail/head/cat/...), NOT of "
     "the command before the pipe — and the output contains failure indicators. Treat this run "
     "as FAILED until proven otherwise: re-run the command WITHOUT the pipe (output is "
     "auto-truncated and the full text is saved to a file, so piping through tail/head is never "
     "needed) to get the real exit code."),
    # `cmd || echo ...` / `cmd || true` — fallback swallows the failure status.
    (re.compile(r"\|\|\s*(?:echo\b|printf\b|true\b|:\s|:$)"),
     "exit_code 0 here is the status of the `||` fallback (echo/true), NOT of the command before "
     "it — and the output contains failure indicators. Treat this run as FAILED until proven "
     "otherwise: re-run the command bare to get its real exit code."),
]

_READONLY_HEADS = frozenset({
    "grep", "rg", "ag", "find", "ls", "cat", "head", "tail", "jq", "awk",
    "sed", "strings", "zcat", "journalctl", "dmesg", "echo", "printf"})

# Strong failure shapes keyed to specific tools so that error-mentioning
# *content* (diffs, logs, commit messages) rarely matches.
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
    r")")


def _first_token(command: str) -> str:
    """Basename of the command head, skipping leading env-var assignments."""
    for tok in (command or "").strip().split():
        if "=" not in tok or tok.startswith(("=", "./", "/")):
            return tok.rsplit("/", 1)[-1]
    return ""


def annotate_masked_success(command: str, output: str) -> Optional[str]:
    """Warning note when an exit-0 result likely masks a failure (caller gates on exit 0)."""
    cmd = command or ""
    window = (output or "")[:_SCAN_CHARS]
    if (not cmd or not window or _first_token(cmd) in _READONLY_HEADS
            or not _FAILURE_SHAPES.search(window)):
        return None
    return next((note for rx, note in _MASKING_SHAPES if rx.search(cmd)), None)


def annotate_failure(command: str, exit_code: int, output: str) -> Optional[str]:
    """Return one short recovery hint for a failed command, or None (exit 0)."""
    if exit_code == 0:
        return None
    window = (output or "")[:_SCAN_CHARS]
    for fn in _OUTPUT_HINTS if window else ():
        try:
            if hint := fn(command or "", window):
                return hint
        except Exception:
            continue
    return _EXIT_CODE_HINTS.get(exit_code)
